// Training loop pour le drafter MTP (Gemma 4 Assistant) via auto-distillation.
//
// Idee: target frozen, drafter trainable. Pour chaque batch:
//   1. Forward target (no grad) sur la sequence -> hidden states + sharedKV
//   2. Construit (bonus_token, prev_hidden) pairs pour chaque position
//   3. Forward drafter (avec grad) en parallele avec mask causal
//   4. Loss = cross-entropy entre drafter logits et ground-truth next token
//
// Le drafter apprend a predire EXACTEMENT comme le target — c'est ce qui
// maximise l'acceptance rate au moment de l'inference MTP.

import Foundation
import MLX
import MLXFast
import MLXNN
import MLXOptimizers

public enum Gemma4DrafterTraining {

    // MARK: - Loss function

    /// Computes drafter training loss for a single batch.
    /// - Parameters:
    ///   - drafter: drafter trainable
    ///   - target: target frozen, deja bind() au drafter
    ///   - batchTokens: `[B, L]` tokens entiers (pas de padding interne — tous samples meme longueur)
    ///   - lastFullCacheIdx: index dans target's concrete layers de la derniere full attention
    ///   - lastSlidingCacheIdx: idem pour sliding
    /// - Returns: `(loss, ntoks)` ou ntoks = nombre de positions qui contribuent a la loss
    public static func drafterLoss(
        drafter: Gemma4AssistantDraftModel,
        target: Gemma4LanguageModel,
        batchTokens: MLXArray,
        lastFullCacheIdx: Int,
        lastSlidingCacheIdx: Int
    ) -> (loss: MLXArray, ntoks: MLXArray) {
        let L = batchTokens.dim(1)
        precondition(L >= 3, "batch sequence length doit etre >= 3 (need positions p, p+1, p+2)")

        // 1. Target forward (no grad) — pre-norm hidden + sharedKV + LOGITS
        let targetOut = target.forwardWithIntermediates(inputs: batchTokens)
        // stopGradient sur tout ce qui vient du target — frozen, pas de backprop
        let hiddens = stopGradient(targetOut.preNormHiddenStates)  // [B, L, backbone]

        // Self-distillation targets: utiliser argmax(target_logits) plutot que ground truth.
        // C'est ce qui maximise l'acceptance rate au moment de l'inference MTP — le drafter
        // doit MATCH le target, pas la verite terrain.
        let targetArgmax = stopGradient(argMax(targetOut.logits, axis: -1))  // [B, L]

        guard let fullKV = targetOut.intermediates[lastFullCacheIdx],
              let slidingKV = targetOut.intermediates[lastSlidingCacheIdx] else {
            fatalError("Cannot extract shared K/V from target intermediates")
        }
        let sharedKV: SharedKVStates = [
            "full_attention": (
                keys: stopGradient(fullKV.keys),
                values: stopGradient(fullKV.values)
            ),
            "sliding_attention": (
                keys: stopGradient(slidingKV.keys),
                values: stopGradient(slidingKV.values)
            ),
        ]

        // 2. Construire les (bonus_token, prev_hidden) pour positions 1..L-1
        // bonus[p] = batchTokens[p] (token a la position p, qu'on traite comme bonus)
        // prev_hidden[p] = hiddens[p-1] (etat avant de voir token p)
        let bonusTokens = batchTokens[0..., 1...]                          // [B, L-1]
        let prevHiddens = hiddens[0..., .stride(to: -1)]                   // [B, L-1, backbone]

        let scale = pow(Float(target.config.hiddenSize), 0.5)
        var bonusEmbeds = target.model.embedTokens(bonusTokens)
        bonusEmbeds = bonusEmbeds * MLXArray(scale, dtype: bonusEmbeds.dtype)
        bonusEmbeds = stopGradient(bonusEmbeds)

        let drafterInput = concatenated([bonusEmbeds, prevHiddens], axis: -1)
        // drafterInput: [B, L-1, 2*backbone]

        // 3. Drafter forward (avec grad) en parallele, mask causal
        let drafterOut = drafter.trainForward(
            inputsEmbeds: drafterInput,
            sharedKVStates: sharedKV,
            startPosition: 1,
            mask: .causal
        )
        // drafterOut.logits: [B, L-1, vocab]

        // 4. Loss: drafter at position p predicts what target predicts at position p+1
        // (= argmax of target_logits[p+1]). Pour input index i in 0..L-2 (= position p+1 = i+1),
        // drafter predit le token a position p+2 = i+2. La cible distillation est
        // argmax(target_logits[i+1]) qui represente "ce que target predit apres avoir vu
        // tokens 0..i+1" = predict position i+2.
        let validLogits = drafterOut.logits[0..., .stride(to: -1), 0...].asType(.float32)
        // [B, L-2, vocab]

        // Self-distillation targets: argmax(target_logits) at positions 1..L-2
        // = predicts "what comes next at position 2..L-1"
        let targets = targetArgmax[0..., 1 ..< (L - 1)]  // [B, L-2]

        let logProbs = MLXNN.logSoftmax(validLogits, axis: -1)
        // gather log_probs at target positions
        let targetExpanded = expandedDimensions(targets, axis: -1)  // [B, L-2, 1]
        let pickedLogProbs = takeAlong(logProbs, targetExpanded.asType(.int32), axis: -1)
            .squeezed(axis: -1)
        // [B, L-2]

        let loss = -pickedLogProbs.mean()
        let ntoks = MLXArray(Float(targets.size))
        return (loss, ntoks)
    }

    // MARK: - Training loop

    public struct TrainConfig {
        public var iterations: Int = 100
        public var batchSize: Int = 1
        public var seqLen: Int = 256
        public var stepsPerReport: Int = 10
        public var stepsPerValid: Int = 0  // 0 = pas d'eval validation
        public var validBatches: Int = 8   // nb de batches a evaluer sur la valid
        public var saveEvery: Int = 100
        public var weightsURL: URL? = nil

        public init() {}
    }

    /// Entrainement par auto-distillation contre le target.
    ///
    /// - Parameters:
    ///   - drafter: drafter Module avec poids initiaux deja charges (typiquement les poids
    ///     pretrained du Google Assistant model, on fine-tune par dessus)
    ///   - target: target frozen — ses parametres ne doivent PAS etre modifies
    ///   - tokenizedSamples: liste de sequences de tokens (chaque sample = un long texte tokenise).
    ///     Sera decoupe en chunks de `seqLen` pour les batches.
    ///   - lastFullCacheIdx, lastSlidingCacheIdx: indices des dernieres couches concretes par type
    ///     dans le target (utilises pour extraire la sharedKV)
    ///   - optimizer: typiquement Adam(lr=1e-4)
    ///   - config: hyperparametres
    public static func trainDrafter(
        drafter: Gemma4AssistantDraftModel,
        target: Gemma4LanguageModel,
        tokenizedSamples: [[Int]],
        validSamples: [[Int]] = [],
        lastFullCacheIdx: Int,
        lastSlidingCacheIdx: Int,
        optimizer: any Optimizer,
        config: TrainConfig,
        progress: (Int, Float) -> Void = { _, _ in }
    ) throws {
        target.train(false)   // target en eval mode (frozen)
        target.freeze()
        drafter.train()       // drafter en train mode

        // Decouper les samples en chunks de seqLen (descendants un par un, pas de batch interne)
        let seqLen = config.seqLen
        func chunkify(_ samples: [[Int]]) -> [[Int]] {
            var out: [[Int]] = []
            for sample in samples {
                var idx = 0
                while idx + seqLen <= sample.count {
                    out.append(Array(sample[idx ..< idx + seqLen]))
                    idx += seqLen
                }
            }
            return out
        }
        let chunks = chunkify(tokenizedSamples)
        let validChunks = chunkify(validSamples)
        guard !chunks.isEmpty else {
            throw NSError(domain: "DrafterTraining", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "No chunks of length \(seqLen) found in samples"
            ])
        }
        print("[drafter-train] \(chunks.count) train chunks, \(validChunks.count) valid chunks (longueur \(seqLen))")

        // valueAndGrad sur le DRAFTER seulement
        // batch = [batchTokens] (single MLXArray in array)
        let lossValueGrad = valueAndGrad(model: drafter) { (drafter: Gemma4AssistantDraftModel, arrays: [MLXArray]) -> [MLXArray] in
            let (loss, ntoks) = drafterLoss(
                drafter: drafter,
                target: target,
                batchTokens: arrays[0],
                lastFullCacheIdx: lastFullCacheIdx,
                lastSlidingCacheIdx: lastSlidingCacheIdx
            )
            return [loss, ntoks]
        }

        var losses: [Float] = []
        var iterStart = Date.timeIntervalSinceReferenceDate
        var bestValidLoss: Float = .infinity

        let batchSize = max(config.batchSize, 1)
        for iter in 0 ..< config.iterations {
            // Sample batchSize chunks au hasard (tous de longueur fixe seqLen → pas de padding)
            var flatTokens: [Int32] = []
            flatTokens.reserveCapacity(batchSize * seqLen)
            for _ in 0 ..< batchSize {
                let chunk = chunks.randomElement()!
                flatTokens.append(contentsOf: chunk.map { Int32($0) })
            }
            let batchTokens = MLXArray(flatTokens).reshaped(batchSize, seqLen)

            // Forward + backward
            let (results, grads) = lossValueGrad(drafter, [batchTokens])
            let lossValue = results[0]
            let _ = results[1]  // ntoks (unused for now)

            optimizer.update(model: drafter, gradients: grads)
            eval(drafter, optimizer, lossValue)

            let lf = lossValue.item(Float.self)
            losses.append(lf)

            // Report
            if (iter + 1) % config.stepsPerReport == 0 {
                let avgLoss = losses.suffix(config.stepsPerReport).reduce(0, +) / Float(config.stepsPerReport)
                let now = Date.timeIntervalSinceReferenceDate
                let iterPerSec = Double(config.stepsPerReport) / (now - iterStart)
                print(String(format: "[drafter-train] iter %d/%d  train_loss=%.4f  %.1f it/s",
                             iter + 1, config.iterations, avgLoss, iterPerSec))
                progress(iter + 1, avgLoss)
                iterStart = now
            }

            // Validation eval (no grad)
            if config.stepsPerValid > 0,
               !validChunks.isEmpty,
               (iter + 1) % config.stepsPerValid == 0 {
                drafter.train(false)
                var valLosses: [Float] = []
                let nValBatches = min(config.validBatches, validChunks.count / batchSize)
                for vb in 0 ..< max(nValBatches, 1) {
                    var flat: [Int32] = []
                    flat.reserveCapacity(batchSize * seqLen)
                    for j in 0 ..< batchSize {
                        let idx = (vb * batchSize + j) % validChunks.count
                        flat.append(contentsOf: validChunks[idx].map { Int32($0) })
                    }
                    let valBatch = MLXArray(flat).reshaped(batchSize, seqLen)
                    let (vloss, _) = drafterLoss(
                        drafter: drafter, target: target,
                        batchTokens: valBatch,
                        lastFullCacheIdx: lastFullCacheIdx,
                        lastSlidingCacheIdx: lastSlidingCacheIdx
                    )
                    eval(vloss)
                    valLosses.append(vloss.item(Float.self))
                }
                drafter.train()
                let avgValLoss = valLosses.reduce(0, +) / Float(max(valLosses.count, 1))
                let isBest = avgValLoss < bestValidLoss
                let marker = isBest ? "  [BEST]" : ""
                print(String(format: "[drafter-train] iter %d/%d  valid_loss=%.4f  (n=%d batches)%@",
                             iter + 1, config.iterations, avgValLoss, valLosses.count, marker))
                if isBest {
                    bestValidLoss = avgValLoss
                    if let url = config.weightsURL {
                        let bestURL = url.deletingPathExtension()
                            .appendingPathExtension("best.safetensors")
                        let params = Dictionary(uniqueKeysWithValues: drafter.parameters().flattened())
                        try save(arrays: params, url: bestURL)
                    }
                }
                iterStart = Date.timeIntervalSinceReferenceDate
            }

            // Save checkpoint
            if let url = config.weightsURL, (iter + 1) % config.saveEvery == 0 {
                let params = Dictionary(uniqueKeysWithValues: drafter.parameters().flattened())
                try save(arrays: params, url: url)
                print("[drafter-train] checkpoint saved to \(url.path)")
            }
        }

        // Save final
        if let url = config.weightsURL {
            let params = Dictionary(uniqueKeysWithValues: drafter.parameters().flattened())
            try save(arrays: params, url: url)
            print("[drafter-train] final weights saved to \(url.path)")
            if bestValidLoss.isFinite {
                let bestURL = url.deletingPathExtension().appendingPathExtension("best.safetensors")
                print(String(format: "[drafter-train] best valid_loss=%.4f saved to %@",
                             bestValidLoss, bestURL.path))
            }
        }
    }
}
