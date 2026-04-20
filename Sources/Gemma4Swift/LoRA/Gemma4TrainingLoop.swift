// Training loop LoRA pour Gemma 4 — colle exactement a la reference mlx-lm Python
// Ref: mlx-lm/tuner/trainer.py (default_loss, iterate_batches, train)

import Foundation
import MLX
import MLXNN
import MLXLMCommon
import MLXLLM
import MLXOptimizers

// MARK: - Loss function (ref: mlx-lm default_loss)

/// Loss cross-entropy avec response masking.
/// Reproduit exactement mlx-lm Python default_loss().
///
/// - batch: [batch_size, seq_len] — la sequence COMPLETE (pas split)
/// - lengths: [batch_size, 2] — [[prompt_offset, total_length]] par sample
func trainingLoss(model: Module, batch: MLXArray, lengths: MLXArray) -> (MLXArray, MLXArray) {
    // Split inputs/targets INSIDE la loss (comme Python)
    let inputs = batch[0..., .stride(to: -1)]
    let targets = batch[0..., 1...]

    // Forward pass — le Python fait model(inputs) sans cache
    let model = model as! any LLMModel
    let logits = model(inputs, cache: nil).asType(.float32)

    // Masque: positions >= offset ET <= total_length
    let seqLen = targets.dim(1)
    let steps = MLXArray(Array(Int32(1) ... Int32(seqLen))).reshaped(1, seqLen)
    let offsets = lengths[0..., 0 ..< 1]    // prompt offset
    let totals = lengths[0..., 1 ..< 2]     // total length
    let mask = (steps .>= offsets) .&& (steps .<= totals)

    // Cross-entropy masquee
    let ntoks = mask.sum()
    let ce = (crossEntropy(logits: logits, targets: targets) * mask).sum() / ntoks

    return (ce, ntoks)
}

// MARK: - Batch iterator (ref: mlx-lm iterate_batches)

/// Prepare un batch a partir de tokens pre-tokenises.
/// Reproduit le comportement de iterate_batches de mlx-lm Python.
public struct TrainingBatchIterator: Sequence, IteratorProtocol {

    public struct TokenizedSample {
        let tokens: [Int]
        let promptOffset: Int  // 0 si pas de masking, sinon position fin du prompt
    }

    let samples: [TokenizedSample]
    let batchSize: Int
    let train: Bool

    // Batches pre-calcules (tries par longueur comme Python)
    var batchIndices: [[Int]]
    var permutation: [Int]
    var permIndex: Int = 0

    init(samples: [TokenizedSample], batchSize: Int, train: Bool) {
        self.samples = samples
        self.batchSize = batchSize
        self.train = train

        // Trier par longueur (comme Python)
        let sortedIdx = (0 ..< samples.count).sorted { samples[$0].tokens.count < samples[$1].tokens.count }

        // Creer les batches
        var batches: [[Int]] = []
        var i = 0
        while i + batchSize <= sortedIdx.count {
            batches.append(Array(sortedIdx[i ..< i + batchSize]))
            i += batchSize
        }
        // Dernier batch partiel
        if i < sortedIdx.count {
            batches.append(Array(sortedIdx[i...]))
        }

        self.batchIndices = batches
        self.permutation = Array(0 ..< batches.count)
        if train { self.permutation.shuffle() }
    }

    /// Retourne (batch, lengths) — comme Python iterate_batches
    public mutating func next() -> (MLXArray, MLXArray)? {
        if permIndex >= permutation.count {
            if !train { return nil }
            permutation.shuffle()
            permIndex = 0
        }

        let batchIdx = batchIndices[permutation[permIndex]]
        permIndex += 1

        let batchSamples = batchIdx.map { samples[$0] }
        let lengths = batchSamples.map { $0.tokens.count }
        let offsets = batchSamples.map { $0.promptOffset }
        let maxLength = lengths.max() ?? 0

        // Padding: utiliser la longueur exacte (le Python ne pad que pour les multi-batch)
        let paddedLength = maxLength

        // Construire le batch
        let batchArray = MLXArray.zeros([batchSamples.count, paddedLength], type: Int32.self)
        for (j, sample) in batchSamples.enumerated() {
            let truncLen = Swift.min(sample.tokens.count, paddedLength)
            batchArray[j, 0 ..< truncLen] = MLXArray(sample.tokens[0 ..< truncLen].map { Int32($0) })
        }

        // lengths = [[offset, length], ...] (comme Python)
        let lengthPairs = zip(offsets, lengths).map { [Int32($0), Int32($1)] }
        let lengthArray = MLXArray(lengthPairs.flatMap { $0 }).reshaped(lengthPairs.count, 2)

        return (batchArray, lengthArray)
    }
}

// MARK: - Tokenisation des samples

/// Tokenise les samples chat via le tokenizer, avec detection du prompt offset.
/// Reproduit ChatDataset.process() de mlx-lm Python.
func tokenizeTrainingSamples(
    texts: [String],
    tokenizer: any Tokenizer,
    maskPrompt: Bool
) -> [TrainingBatchIterator.TokenizedSample] {
    return texts.compactMap { text -> TrainingBatchIterator.TokenizedSample? in
        let tokens = tokenizer.encode(text: text)
        guard tokens.count > 1 else { return nil }

        if maskPrompt {
            // Trouver le dernier <|turn>model\n comme frontiere prompt/reponse
            // Token 105 = <|turn>, Token 4368 = model, Token 107 = \n
            var offset = 0
            for i in 0 ..< tokens.count - 1 {
                if tokens[i] == 105 && tokens[i + 1] == 4368 {
                    offset = i + 3  // <|turn> + model + \n
                }
            }
            return TrainingBatchIterator.TokenizedSample(tokens: tokens, promptOffset: offset)
        } else {
            return TrainingBatchIterator.TokenizedSample(tokens: tokens, promptOffset: 0)
        }
    }
}

// MARK: - Training loop (ref: mlx-lm train())

/// Training loop qui reproduit exactement le comportement de mlx-lm Python.
/// Pas de dependance sur LoRATrain upstream.
public func trainLoRA(
    model: Module,
    trainSamples: [TrainingBatchIterator.TokenizedSample],
    validSamples: [TrainingBatchIterator.TokenizedSample],
    optimizer: any Optimizer,
    iterations: Int,
    batchSize: Int = 1,
    stepsPerReport: Int = 10,
    stepsPerEval: Int = 100,
    saveEvery: Int = 100,
    weightsURL: URL? = nil,
    isFullFineTune: Bool = false,
    progress: (LoRATrain.Progress) -> LoRATrain.ProgressDisposition
) throws {
    // Activer le mode training (ref: Python model.train())
    model.train()

    // Loss + grad
    let lossValueGrad = valueAndGrad(model: model) { model, arrays in
        let (ce, ntoks) = trainingLoss(model: model, batch: arrays[0], lengths: arrays[1])
        return [ce, ntoks]
    }

    var losses = [Float]()
    var tokenCount = 0
    var start = Date.timeIntervalSinceReferenceDate

    for (iteration, (batch, lengths)) in TrainingBatchIterator(
        samples: trainSamples, batchSize: batchSize, train: true
    ).enumerated() {
        // Forward + backward (ref: Python step())
        let (resultArray, grad) = lossValueGrad(model, [batch, lengths])
        let lvalue = resultArray[0]
        let tokens = resultArray[1]

        // Update (ref: Python optimizer.update(model, grad))
        optimizer.update(model: model, gradients: grad)

        // eval APRES l'update pour synchroniser
        eval(model, optimizer, lvalue)

        losses.append(lvalue.item(Float.self))
        tokenCount += tokens.item(Int.self)

        // Report
        if (iteration + 1) % stepsPerReport == 0 {
            let trainingLoss = MLXArray(losses).mean(stream: .cpu).item(Float.self)
            let now = Date.timeIntervalSinceReferenceDate
            let iterPerSec = Double(stepsPerReport) / (now - start)
            let tokPerSec = Double(tokenCount) / (now - start)

            let p = LoRATrain.Progress.train(iteration: iteration, trainingLoss: trainingLoss,
                              iterationsPerSecond: iterPerSec, tokensPerSecond: tokPerSec)
            if progress(p) == .stop { break }
            losses.removeAll()
            tokenCount = 0
            start = Date.timeIntervalSinceReferenceDate
        }

        // Validation
        if iteration == 0 || (iteration + 1) % stepsPerEval == 0 {
            let valStart = Date.timeIntervalSinceReferenceDate
            model.train(false)  // Mode eval pour la validation
            let valLoss = evaluateTraining(model: model, samples: validSamples, batchSize: batchSize)
            model.train()  // Retour en mode training
            let now = Date.timeIntervalSinceReferenceDate

            let p = LoRATrain.Progress.validation(iteration: iteration, validationLoss: valLoss,
                                   validationTime: now - valStart)
            if progress(p) == .stop { break }
            start = Date.timeIntervalSinceReferenceDate
        }

        // Save
        if let url = weightsURL, (iteration + 1) % saveEvery == 0 {
            if isFullFineTune {
                let allParams = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
                try save(arrays: allParams, url: url)
            } else {
                let trainableParams = Dictionary(uniqueKeysWithValues: model.trainableParameters().flattened())
                try save(arrays: trainableParams, url: url)
            }
            let p = LoRATrain.Progress.save(iteration: iteration, url: url)
            if progress(p) == .stop { break }
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
            let trainableParams = Dictionary(uniqueKeysWithValues: model.trainableParameters().flattened())
            try save(arrays: trainableParams, url: url)
        }
    }
}

/// Evaluation (ref: mlx-lm evaluate())
func evaluateTraining(model: Module, samples: [TrainingBatchIterator.TokenizedSample], batchSize: Int) -> Float {
    var allLosses = [Float]()
    var tokenCount = 0

    for (_, (batch, lengths)) in TrainingBatchIterator(
        samples: samples, batchSize: batchSize, train: false
    ).enumerated() {
        let (losses, tokens) = trainingLoss(model: model as! Module, batch: batch, lengths: lengths)
        allLosses.append((losses * tokens).item(Float.self))
        tokenCount += tokens.item(Int.self)
    }

    return tokenCount > 0
        ? (sum(MLXArray(allLosses), stream: .cpu) / tokenCount).item(Float.self)
        : 0
}

// MARK: - Types publics re-exportes (pour ne pas dependre de LoRATrain)

/// Re-export MaskedBatchIterator.Sample pour compatibilite
public typealias MaskedBatchSample = TrainingBatchIterator.TokenizedSample

// MARK: - Multimodal Training

/// Sample multimodal tokenise avec features media pre-calculees
public struct MultimodalTokenizedSample {
    public let tokens: [Int]
    public let promptOffset: Int
    public let pixelValues: MLXArray?      // [1, 3, H, W] pour image
    public let audioFeatures: MLXArray?    // [1, T, 128] mel spectrogram
    public let audioMask: MLXArray?        // [1, T] mask de padding

    public init(tokens: [Int], promptOffset: Int,
                pixelValues: MLXArray? = nil,
                audioFeatures: MLXArray? = nil,
                audioMask: MLXArray? = nil) {
        self.tokens = tokens
        self.promptOffset = promptOffset
        self.pixelValues = pixelValues
        self.audioFeatures = audioFeatures
        self.audioMask = audioMask
    }
}

/// Iterateur batch size 1 pour samples multimodaux (media de taille variable)
public struct MultimodalBatchIterator: Sequence, IteratorProtocol {

    let samples: [MultimodalTokenizedSample]
    let train: Bool
    var permutation: [Int]
    var permIndex: Int = 0

    public init(samples: [MultimodalTokenizedSample], train: Bool) {
        self.samples = samples
        self.train = train
        self.permutation = Array(0 ..< samples.count)
        if train { self.permutation.shuffle() }
    }

    public mutating func next() -> (MLXArray, MLXArray, MLXArray?, MLXArray?, MLXArray?)? {
        if permIndex >= permutation.count {
            if !train { return nil }
            permutation.shuffle()
            permIndex = 0
        }

        let sample = samples[permutation[permIndex]]
        permIndex += 1

        let batch = MLXArray(sample.tokens.map { Int32($0) }).reshaped(1, sample.tokens.count)
        let lengths = MLXArray([Int32(sample.promptOffset), Int32(sample.tokens.count)]).reshaped(1, 2)

        return (batch, lengths, sample.pixelValues, sample.audioFeatures, sample.audioMask)
    }
}

/// Training loop multimodal — set les pending* media avant chaque forward pass
public func trainMultimodalLoRA(
    model: Module,
    trainSamples: [MultimodalTokenizedSample],
    validSamples: [MultimodalTokenizedSample],
    optimizer: any Optimizer,
    iterations: Int,
    stepsPerReport: Int = 10,
    stepsPerEval: Int = 100,
    saveEvery: Int = 100,
    weightsURL: URL? = nil,
    isFullFineTune: Bool = false,
    progress: (LoRATrain.Progress) -> LoRATrain.ProgressDisposition
) throws {
    model.train()

    // Le modele multimodal pour setter les pending properties
    let mmModel = model as! Gemma4MultimodalLLMModel

    // Loss + grad — reutilise trainingLoss() existant
    let lossValueGrad = valueAndGrad(model: model) { model, arrays in
        let (ce, ntoks) = trainingLoss(model: model, batch: arrays[0], lengths: arrays[1])
        return [ce, ntoks]
    }

    var losses = [Float]()
    var tokenCount = 0
    var start = Date.timeIntervalSinceReferenceDate

    for (iteration, (batch, lengths, pixelValues, audioFeatures, audioMask))
        in MultimodalBatchIterator(samples: trainSamples, train: true).enumerated() {

        // Setter les media AVANT le forward pass — consommes par callAsFunction
        mmModel.pendingPixelValues = pixelValues
        mmModel.pendingAudioFeatures = audioFeatures
        mmModel.pendingAudioMask = audioMask

        let (resultArray, grad) = lossValueGrad(model, [batch, lengths])
        let lvalue = resultArray[0]
        let tokens = resultArray[1]

        optimizer.update(model: model, gradients: grad)
        eval(model, optimizer, lvalue)

        losses.append(lvalue.item(Float.self))
        tokenCount += tokens.item(Int.self)

        // Report
        if (iteration + 1) % stepsPerReport == 0 {
            let trainingLoss = MLXArray(losses).mean(stream: .cpu).item(Float.self)
            let now = Date.timeIntervalSinceReferenceDate
            let iterPerSec = Double(stepsPerReport) / (now - start)
            let tokPerSec = Double(tokenCount) / (now - start)

            let p = LoRATrain.Progress.train(iteration: iteration, trainingLoss: trainingLoss,
                              iterationsPerSecond: iterPerSec, tokensPerSecond: tokPerSec)
            if progress(p) == .stop { break }
            losses.removeAll()
            tokenCount = 0
            start = Date.timeIntervalSinceReferenceDate
        }

        // Validation
        if iteration == 0 || (iteration + 1) % stepsPerEval == 0 {
            let valStart = Date.timeIntervalSinceReferenceDate
            model.train(false)
            let valLoss = evaluateMultimodalTraining(model: model, samples: validSamples)
            model.train()
            let now = Date.timeIntervalSinceReferenceDate

            let p = LoRATrain.Progress.validation(iteration: iteration, validationLoss: valLoss,
                                   validationTime: now - valStart)
            if progress(p) == .stop { break }
            start = Date.timeIntervalSinceReferenceDate
        }

        // Save
        if let url = weightsURL, (iteration + 1) % saveEvery == 0 {
            if isFullFineTune {
                let allParams = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
                try save(arrays: allParams, url: url)
            } else {
                let trainableParams = Dictionary(uniqueKeysWithValues: model.trainableParameters().flattened())
                try save(arrays: trainableParams, url: url)
            }
            let p = LoRATrain.Progress.save(iteration: iteration, url: url)
            if progress(p) == .stop { break }
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
            let trainableParams = Dictionary(uniqueKeysWithValues: model.trainableParameters().flattened())
            try save(arrays: trainableParams, url: url)
        }
    }
}

/// Evaluation multimodal
func evaluateMultimodalTraining(model: Module, samples: [MultimodalTokenizedSample]) -> Float {
    let mmModel = model as! Gemma4MultimodalLLMModel
    var allLosses = [Float]()
    var tokenCount = 0

    for (_, (batch, lengths, pixelValues, audioFeatures, audioMask))
        in MultimodalBatchIterator(samples: samples, train: false).enumerated() {

        mmModel.pendingPixelValues = pixelValues
        mmModel.pendingAudioFeatures = audioFeatures
        mmModel.pendingAudioMask = audioMask

        let (losses, tokens) = trainingLoss(model: model as! Module, batch: batch, lengths: lengths)
        allLosses.append((losses * tokens).item(Float.self))
        tokenCount += tokens.item(Int.self)
    }

    return tokenCount > 0
        ? (sum(MLXArray(allLosses), stream: .cpu) / tokenCount).item(Float.self)
        : 0
}
