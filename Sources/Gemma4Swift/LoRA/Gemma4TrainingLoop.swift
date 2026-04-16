// Training loop custom avec response masking — ne depend pas de LoRABatchIterator
// Inspire de mlx-lm Python (trainer.py) qui supporte --mask-prompt

import Foundation
import MLX
import MLXNN
import MLXLMCommon
import MLXLLM
import MLXOptimizers
import Tokenizers

// MARK: - Batch Iterator avec prompt/response split

/// Iterateur de batch qui separe les tokens du prompt et de la reponse.
/// Permet le response masking : la loss n'est calculee que sur les tokens de la reponse.
public struct MaskedBatchIterator: Sequence, IteratorProtocol {

    /// Un sample d'entrainement avec la frontiere prompt/reponse
    public struct Sample {
        let promptTokens: [Int]    // tokens du prompt (user)
        let responseTokens: [Int]  // tokens de la reponse (assistant)
    }

    let samples: [Sample]
    let batchSize: Int
    let train: Bool

    var indices: [Int]
    var index = 0

    init(samples: [Sample], batchSize: Int, train: Bool) {
        self.samples = samples
        self.batchSize = batchSize
        self.train = train
        self.indices = Array(0 ..< samples.count)
        if train { indices.shuffle() }
    }

    /// Retourne (inputs, targets, lengths) ou lengths = [prompt_end, total_length] par sample
    public mutating func next() -> (MLXArray, MLXArray, MLXArray)? {
        if index >= indices.count {
            if !train { return nil }
            indices.shuffle()
            index = 0
        }

        let endIndex = Swift.min(index + batchSize, indices.count)
        let batch = (index ..< endIndex).map { samples[indices[$0]] }

        // Concatener prompt + response pour chaque sample
        let fullSequences = batch.map { $0.promptTokens + $0.responseTokens }
        let lengths = fullSequences.map { $0.count }
        let promptLengths = batch.map { $0.promptTokens.count }
        let maxLength = lengths.max() ?? 0

        if maxLength > 2048 {
            print("[WARNING] Sequences > 2048 tokens. Consider shorter data.")
        }

        // Pad et construire le batch
        let batchArray = MLXArray.zeros([lengths.count, maxLength], type: Int32.self)
        for (j, (seq, l)) in zip(fullSequences, lengths).enumerated() {
            batchArray[j, 0 ..< l] = MLXArray(seq.map { Int32($0) })
        }

        // lengths = [[prompt_end, total_length], ...] — format mlx-lm Python
        let lengthPairs = zip(promptLengths, lengths).map { [Int32($0), Int32($1)] }
        let lengthArray = MLXArray(lengthPairs.flatMap { $0 }).reshaped(lengthPairs.count, 2)

        index = endIndex

        return (batchArray[0..., .stride(to: -1)], batchArray[0..., 1...], lengthArray)
    }
}

// MARK: - Loss avec response masking

/// Loss cross-entropy avec masking du prompt (seuls les tokens de la reponse contribuent)
func maskedLoss(model: Module, inputs: MLXArray, targets: MLXArray, lengths: MLXArray) -> (MLXArray, MLXArray) {
    let llm = model as! any LanguageModel
    let logits = llm(inputs, cache: nil as [KVCache]?).asType(.float32)

    // lengths[:, 0] = prompt_end, lengths[:, 1] = total_length
    let promptEnd = lengths[0..., 0 ..< 1]   // [batch, 1]
    let totalLen = lengths[0..., 1 ..< 2]     // [batch, 1]

    // steps = 1, 2, 3, ... (positions des targets, decalees de 1)
    let seqLen = targets.dim(1)
    let steps = MLXArray(Array(Int32(1) ... Int32(seqLen))).reshaped(1, seqLen)  // [1, seq_len]

    // Masque : seuls les tokens apres prompt_end et avant total_length
    let afterPrompt = steps .>= promptEnd
    let beforeEnd = steps .<= totalLen
    let mask = afterPrompt .&& beforeEnd

    let ntoks = mask.sum()
    let ce = (crossEntropy(logits: logits, targets: targets) * mask).sum() / ntoks

    return (ce, ntoks)
}

// MARK: - Training loop avec response masking

/// Training loop custom qui supporte le response masking.
/// Remplace LoRATrain.train() pour un meilleur controle du gradient.
public func trainWithResponseMasking(
    model: Module,
    trainSamples: [MaskedBatchIterator.Sample],
    validSamples: [MaskedBatchIterator.Sample],
    optimizer: any Optimizer,
    iterations: Int,
    batchSize: Int = 1,
    stepsPerReport: Int = 10,
    stepsPerEval: Int = 100,
    saveEvery: Int = 100,
    weightsURL: URL? = nil,
    gradClipMaxNorm: Float = 0,
    isFullFineTune: Bool = false,
    progress: (LoRATrain.Progress) -> LoRATrain.ProgressDisposition
) throws {
    // Activer le mode training (active le dropout LoRA si present)
    // Ref: Python mlx-lm fait model.train() avant le training
    model.train()

    let lossValueGrad = valueAndGrad(model: model) { model, arrays in
        let (ce, ntoks) = maskedLoss(model: model, inputs: arrays[0], targets: arrays[1], lengths: arrays[2])
        return [ce, ntoks]
    }

    var losses = [Float]()
    var tokenCount = 0
    var start = Date.timeIntervalSinceReferenceDate

    for (iteration, (inputs, targets, lengths)) in MaskedBatchIterator(
        samples: trainSamples, batchSize: batchSize, train: true
    ).enumerated() {
        // Forward + backward
        let (resultArray, grad) = lossValueGrad(model, [inputs, targets, lengths])
        let lvalue = resultArray[0]
        let tokens = resultArray[1]

        // Gradient clipping
        var clippedGrad = grad
        if gradClipMaxNorm > 0 {
            let (clipped, _) = clipGradNorm(gradients: clippedGrad, maxNorm: gradClipMaxNorm)
            clippedGrad = clipped
        }

        // Update
        optimizer.update(model: model, gradients: clippedGrad)
        eval(model, optimizer, lvalue)

        losses.append(lvalue.item(Float.self))
        tokenCount += tokens.item(Int.self)

        // Report
        if (iteration + 1) % stepsPerReport == 0 {
            let trainingLoss = MLXArray(losses).mean(stream: .cpu).item(Float.self)
            let now = Date.timeIntervalSinceReferenceDate
            let iterPerSec = Double(stepsPerReport) / (now - start)
            let tokPerSec = Double(tokenCount) / (now - start)

            let trainProgress = LoRATrain.Progress.train(iteration: iteration, trainingLoss: trainingLoss,
                              iterationsPerSecond: iterPerSec, tokensPerSecond: tokPerSec)
            if progress(trainProgress) == .stop {
                break
            }
            losses.removeAll()
            tokenCount = 0
            start = Date.timeIntervalSinceReferenceDate
        }

        // Validation
        if iteration == 0 || (iteration + 1) % stepsPerEval == 0 {
            let valStart = Date.timeIntervalSinceReferenceDate
            let valLoss = evaluateWithMasking(model: model, samples: validSamples, batchSize: batchSize)
            let now = Date.timeIntervalSinceReferenceDate

            let valProgress = LoRATrain.Progress.validation(iteration: iteration, validationLoss: valLoss,
                                   validationTime: now - valStart)
            if progress(valProgress) == .stop {
                break
            }
            start = Date.timeIntervalSinceReferenceDate
        }

        // Save
        if let url = weightsURL, (iteration + 1) % saveEvery == 0 {
            if isFullFineTune {
                let allParams = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
                try save(arrays: allParams, url: url)
            } else {
                try LoRATrain.saveLoRAWeights(model: model, url: url)
            }
            let saveProgress = LoRATrain.Progress.save(iteration: iteration, url: url)
            if progress(saveProgress) == .stop { break }
            start = Date.timeIntervalSinceReferenceDate
        }

        if iteration + 1 >= iterations { break }
    }

    // Sauvegarde finale
    if let url = weightsURL {
        if isFullFineTune {
            let allParams = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
            try save(arrays: allParams, url: url)
        } else {
            try LoRATrain.saveLoRAWeights(model: model, url: url)
        }
    }
}

/// Evaluation avec response masking
func evaluateWithMasking(model: Module, samples: [MaskedBatchIterator.Sample], batchSize: Int) -> Float {
    var allLosses = [Float]()
    var tokenCount = 0

    for (_, (inputs, targets, lengths)) in MaskedBatchIterator(
        samples: samples, batchSize: batchSize, train: false
    ).enumerated() {
        let (losses, tokens) = maskedLoss(model: model as! Module, inputs: inputs, targets: targets, lengths: lengths)
        allLosses.append((losses * tokens).item(Float.self))
        tokenCount += tokens.item(Int.self)
    }

    return tokenCount > 0
        ? (sum(MLXArray(allLosses), stream: .cpu) / tokenCount).item(Float.self)
        : 0
}
