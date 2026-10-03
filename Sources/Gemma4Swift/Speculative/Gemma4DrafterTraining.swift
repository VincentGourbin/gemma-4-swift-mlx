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
        let target = targetOutputs(
            target: target, batchTokens: batchTokens,
            lastFullCacheIdx: lastFullCacheIdx, lastSlidingCacheIdx: lastSlidingCacheIdx)
        return drafterLoss(drafter: drafter, batchTokens: batchTokens, target: target)
    }

    /// Ce que le drafter consomme de la cible gelee pour un lot. Calculer ces sorties hors de
    /// `valueAndGrad` a ete mesure (K-30 d, 2026-09-29) : aucun gain (MLX ne propage deja rien
    /// a travers la cible sous stopGradient) ; la boucle garde donc le chemin simple.
    public struct TargetOutputs {
        public let preNormHiddens: MLXArray   // [B, L, backbone]
        public let targetArgmax: MLXArray     // [B, L]
        public let bonusEmbeds: MLXArray      // [B, L-1, backbone]
        public let fullKV: (keys: MLXArray, values: MLXArray)
        public let slidingKV: (keys: MLXArray, values: MLXArray)

        var arrays: [MLXArray] {
            [preNormHiddens, targetArgmax, bonusEmbeds, fullKV.keys, fullKV.values, slidingKV.keys, slidingKV.values]
        }

        init(_ a: [MLXArray]) {
            preNormHiddens = a[0]; targetArgmax = a[1]; bonusEmbeds = a[2]
            fullKV = (a[3], a[4]); slidingKV = (a[5], a[6])
        }

        init(preNormHiddens: MLXArray, targetArgmax: MLXArray, bonusEmbeds: MLXArray,
             fullKV: (keys: MLXArray, values: MLXArray), slidingKV: (keys: MLXArray, values: MLXArray)) {
            self.preNormHiddens = preNormHiddens; self.targetArgmax = targetArgmax; self.bonusEmbeds = bonusEmbeds
            self.fullKV = fullKV; self.slidingKV = slidingKV
        }
    }

    /// Forward de la cible (gelee) : hidden pre-norm, argmax des logits (cibles de
    /// distillation), K/V partages, embeddings des jetons bonus. Tout en stopGradient.
    public static func targetOutputs(
        target: Gemma4LanguageModel,
        batchTokens: MLXArray,
        lastFullCacheIdx: Int,
        lastSlidingCacheIdx: Int
    ) -> TargetOutputs {
        // 1. Target forward (no grad) — pre-norm hidden + sharedKV + LOGITS
        let targetOut = target.forwardWithIntermediates(inputs: batchTokens)
        let hiddens = stopGradient(targetOut.preNormHiddenStates)
        // Self-distillation : argmax(target_logits) plutot que la verite terrain — le
        // drafter doit coller a la cible, c'est ce qui maximise l'acceptation MTP.
        let targetArgmax = stopGradient(argMax(targetOut.logits, axis: -1))
        guard let fullKV = targetOut.intermediates[lastFullCacheIdx],
              let slidingKV = targetOut.intermediates[lastSlidingCacheIdx] else {
            fatalError("Cannot extract shared K/V from target intermediates")  // indices valides par trainDrafter
        }
        // bonus[p] = batchTokens[p], embeddings a l'echelle du modele
        let bonusTokens = batchTokens[0..., 1...]
        let scale = pow(Float(target.config.hiddenSize), 0.5)
        var bonusEmbeds = target.model.embedTokens(bonusTokens)
        bonusEmbeds = stopGradient(bonusEmbeds * MLXArray(scale, dtype: bonusEmbeds.dtype))
        return TargetOutputs(
            preNormHiddens: hiddens, targetArgmax: targetArgmax, bonusEmbeds: bonusEmbeds,
            fullKV: (stopGradient(fullKV.keys), stopGradient(fullKV.values)),
            slidingKV: (stopGradient(slidingKV.keys), stopGradient(slidingKV.values)))
    }

    /// Perte du drafter a partir des sorties de la cible deja calculees.
    public static func drafterLoss(
        drafter: Gemma4AssistantDraftModel,
        batchTokens: MLXArray,
        target: TargetOutputs
    ) -> (loss: MLXArray, ntoks: MLXArray) {
        let L = batchTokens.dim(1)
        precondition(L >= 3, "batch sequence length doit etre >= 3 (need positions p, p+1, p+2)")
        let sharedKV: SharedKVStates = [
            "full_attention": (keys: target.fullKV.keys, values: target.fullKV.values),
            "sliding_attention": (keys: target.slidingKV.keys, values: target.slidingKV.values),
        ]
        let targetArgmax = target.targetArgmax
        // prev_hidden[p] = hiddens[p-1] (etat avant de voir le jeton p)
        let prevHiddens = target.preNormHiddens[0..., .stride(to: -1)]
        let drafterInput = concatenated([target.bonusEmbeds, prevHiddens], axis: -1)
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
        /// Graine du tirage des troncons (A-08 : `randomElement()` non seede).
        public var seed: UInt64 = 0

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
    /// - Warning: exclusif dans le process (`Gemma4ComputeGate`) : echoue avec
    ///   `inferenceInProgress` si une inference du paquet tourne, et toute inference
    ///   lancee pendant l'entrainement echoue avec `trainingInProgress`. Un gradient
    ///   concurrent d'un forward fige le process (deadlock mlx-swift, voir CLAUDE.md).
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
        // K-9 : entrainement exclusif — un gradient et un forward concurrents figent le
        // process (Gemma4ComputeGate). Refuse si une inference du paquet tourne.
        // A-12 : entrees validees ici (erreurs levees) plutot que par precondition/fatalError
        // dans drafterLoss, qui tourne dans valueAndGrad.
        guard config.seqLen >= 3 else {
            throw DrafterTrainingError.invalidInput("seqLen doit etre >= 3 (positions p, p+1, p+2), recu \(config.seqLen)")
        }
        let concreteLayers = target.model.layers.count
        guard (0 ..< concreteLayers).contains(lastFullCacheIdx), (0 ..< concreteLayers).contains(lastSlidingCacheIdx) else {
            throw DrafterTrainingError.invalidInput(
                "indices de cache hors du modele cible (\(lastFullCacheIdx), \(lastSlidingCacheIdx) sur \(concreteLayers) couches)")
        }
        try Gemma4ComputeGate.shared.beginTraining()
        defer { Gemma4ComputeGate.shared.endTraining() }
        let beacon = RuntimeBeacon.begin(task: "train", model: "mtp-drafter")
        defer { beacon?.end() }
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
            throw DrafterTrainingError.invalidInput("aucun troncon de \(seqLen) jetons dans les exemples")
        }
        let tooShort = tokenizedSamples.filter { $0.count < seqLen }.count
        print("[drafter-train] \(chunks.count) train chunks, \(validChunks.count) valid chunks (longueur \(seqLen))"
            + (tooShort > 0 ? " ; \(tooShort) exemple(s) plus court(s) que \(seqLen) ignore(s)" : ""))
        var rng = SeededGenerator(seed: config.seed)

        // valueAndGrad sur le DRAFTER seulement
        // batch = [batchTokens] (single MLXArray in array)
        let lossValueGrad = valueAndGrad(model: drafter) { (drafter: Gemma4AssistantDraftModel, arrays: [MLXArray]) -> [MLXArray] in
            let (loss, ntoks) = arrays.count > 1
                ? drafterLoss(drafter: drafter, batchTokens: arrays[0], target: TargetOutputs(Array(arrays[1...])))
                : drafterLoss(
                    drafter: drafter, target: target, batchTokens: arrays[0],
                    lastFullCacheIdx: lastFullCacheIdx, lastSlidingCacheIdx: lastSlidingCacheIdx)
            return [loss, ntoks]
        }

        var losses: [Float] = []
        var iterStart = Date.timeIntervalSinceReferenceDate
        var bestValidLoss: Float = .infinity

        let batchSize = max(config.batchSize, 1)
        for iter in 0 ..< config.iterations {
            beacon?.update(phase: "train", step: iter + 1, totalSteps: config.iterations)
            // Sample batchSize chunks au hasard (tous de longueur fixe seqLen → pas de padding)
            var flatTokens: [Int32] = []
            flatTokens.reserveCapacity(batchSize * seqLen)
            for _ in 0 ..< batchSize {
                let chunk = chunks[Int.random(in: 0 ..< chunks.count, using: &rng)]
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

public enum DrafterTrainingError: LocalizedError, Equatable {
    case invalidInput(String)
    case ambiguousWeights(String)

    public var errorDescription: String? {
        switch self {
        case .invalidInput(let message): return message
        case .ambiguousWeights(let message): return message
        }
    }
}

/// Fichiers de poids d'un drafter (A-11) : un fichier est pris tel quel ; un dossier est lu
/// dans l'ordre trie, et refuse s'il contient a la fois `drafter.safetensors` et
/// `drafter.best.safetensors` (memes cles, l'ordre du systeme de fichiers decidait).
public enum Gemma4DrafterWeights {
    public static func files(at url: URL) throws -> [URL] {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory) else {
            throw DrafterTrainingError.invalidInput("poids du drafter introuvables : \(url.path)")
        }
        guard isDirectory.boolValue else { return [url] }
        let files = try FileManager.default.contentsOfDirectory(at: url, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        let names = Set(files.map(\.lastPathComponent))
        if names.contains("drafter.safetensors") && names.contains("drafter.best.safetensors") {
            throw DrafterTrainingError.ambiguousWeights(
                "\(url.lastPathComponent) contient drafter.safetensors et drafter.best.safetensors : "
                    + "passer le fichier voulu a --drafter-path")
        }
        return files
    }
}

