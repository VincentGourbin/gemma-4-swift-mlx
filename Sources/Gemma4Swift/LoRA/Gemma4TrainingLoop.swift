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
func trainingLoss(
    model: Module, batch: MLXArray, lengths: MLXArray, headFrom: Int = 0
) -> (MLXArray, MLXArray) {
    // Split inputs/targets INSIDE la loss (comme Python)
    let inputs = batch[0..., .stride(to: -1)]
    var targets = batch[0..., 1...]

    // Forward pass — le Python fait model(inputs) sans cache.
    // K-30 (a) : avec le prompt masque, la tete (262 k logits en fp32) ne tourne que sur
    // les positions a partir de `headFrom` — celles d'avant ne comptent pas dans la perte.
    let logits: MLXArray
    if headFrom > 0, let llm = model as? Gemma4LLMModel {
        logits = llm.logits(inputs, from: headFrom).asType(.float32)
    } else if headFrom > 0, let mm = model as? Gemma4MultimodalLLMModel {
        logits = mm.logits(inputs, from: headFrom).asType(.float32)
    } else {
        logits = (model as! any LLMModel)(inputs, cache: nil).asType(.float32)
    }
    let start = logits.dim(1) == targets.dim(1) ? 0 : headFrom
    targets = targets[0..., start...]

    // Masque: positions >= offset ET <= total_length
    let seqLen = targets.dim(1)
    let steps = MLXArray(Array(Int32(start + 1) ... Int32(start + seqLen))).reshaped(1, seqLen)
    let offsets = lengths[0..., 0 ..< 1]    // prompt offset
    let totals = lengths[0..., 1 ..< 2]     // total length
    let mask = (steps .>= offsets) .&& (steps .<= totals)

    // Cross-entropy masquee
    let ntoks = mask.sum()
    let ce = (crossEntropy(logits: logits, targets: targets) * mask).sum() / ntoks

    return (ce, ntoks)
}

// MARK: - Mesures (K-28)

/// Mesures d'entrainement, une par rapport ou validation (instrument de K-28) : debit
/// **traite** (B x L, ce que coute le pas) et debit **entraine** (positions de la perte,
/// l'ancien « tok/s »), memoire MLX.
public struct Gemma4TrainingMetrics: Sendable, Codable {
    public enum Kind: String, Sendable, Codable { case train, validation }
    public let kind: Kind
    public let iteration: Int
    public let loss: Float
    public let iterationsPerSecond: Double
    public let processedTokensPerSecond: Double
    public let trainedTokensPerSecond: Double
    public let seconds: Double
    public let activeMB: Int
    public let peakMB: Int

    enum CodingKeys: String, CodingKey {
        case kind, iteration, loss, seconds
        case iterationsPerSecond = "it_s"
        case processedTokensPerSecond = "processed_tok_s"
        case trainedTokensPerSecond = "trained_tok_s"
        case activeMB = "active_mlx_mb"
        case peakMB = "peak_mlx_mb"
    }
}

// MARK: - Politique memoire (K-30 b)

/// Politique memoire d'un entrainement (A-20 : aucune jusqu'ici ; E2B bf16 director :
/// 76 Go d'empreinte pour 36 Go de pic MLX, le cache grossissant sans borne).
public struct Gemma4TrainingMemoryPolicy: Sendable, Equatable {
    /// `Memory.cacheLimit` en Mo (nil = inchange).
    public var cacheLimitMB: Int?
    /// `Memory.clearCache()` apres chaque validation et sauvegarde (pics transitoires).
    public var clearCacheAfterValidation: Bool

    public init(cacheLimitMB: Int? = nil, clearCacheAfterValidation: Bool = true) {
        self.cacheLimitMB = cacheLimitMB
        self.clearCacheAfterValidation = clearCacheAfterValidation
    }

    func applyAtStart() {
        if let cacheLimitMB { Memory.cacheLimit = cacheLimitMB << 20 }
    }

    func afterValidationOrSave() {
        if clearCacheAfterValidation { Memory.clearCache() }
    }
}

// MARK: - Generateur seede (A-08)

/// SplitMix64 : melange des exemples reproductible (le generateur systeme rendait deux
/// runs identiques incomparables en « loss a N pas »).
public struct SeededGenerator: RandomNumberGenerator, Sendable {
    private var state: UInt64
    public init(seed: UInt64) { state = seed &+ 0x9E37_79B9_7F4A_7C15 }
    public mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
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
    var rng: SeededGenerator

    init(samples: [TokenizedSample], batchSize: Int, train: Bool, seed: UInt64 = 0) {
        self.rng = SeededGenerator(seed: seed)
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
        if train { self.permutation.shuffle(using: &rng) }
    }

    /// Retourne (batch, lengths) — comme Python iterate_batches
    public mutating func next() -> (MLXArray, MLXArray)? {
        if permIndex >= permutation.count {
            if !train { return nil }
            permutation.shuffle(using: &rng)
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
/// - Warning: exclusif dans le process (`Gemma4ComputeGate`) : echoue avec
///   `inferenceInProgress` si une inference du paquet tourne, et toute inference
///   lancee pendant l'entrainement echoue avec `trainingInProgress`. Un gradient
///   concurrent d'un forward fige le process (deadlock mlx-swift, voir CLAUDE.md).
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
    seed: UInt64 = 0,
    gradClipMaxNorm: Float = 0,
    startIteration: Int = 0,
    checkpointDirectory: URL? = nil,
    validationBatches: Int? = nil,
    metrics: ((Gemma4TrainingMetrics) -> Void)? = nil,
    responseOnlyHead: Bool = false,
    memoryPolicy: Gemma4TrainingMemoryPolicy? = nil,
    progress: (LoRATrain.Progress) -> LoRATrain.ProgressDisposition
) throws {
    // K-9 : entrainement exclusif — un gradient et un forward concurrents figent le
    // process (Gemma4ComputeGate). Refuse si une inference du paquet tourne.
    // Reprise d'un run deja termine : rien a faire (l'iterateur d'entrainement ne finit jamais).
    guard startIteration < iterations else { return }
    try Gemma4ComputeGate.shared.beginTraining()
    defer { Gemma4ComputeGate.shared.endTraining() }
    // Beacon opt-in (SiliconScope) : pas d'entrainement et validations.
    let beacon = RuntimeBeacon.begin(task: "train")
    defer { beacon?.end() }
    // Activer le mode training (ref: Python model.train())
    model.train()

    // Loss + grad
    // K-30 (a) : debut de la tete, fixe avant chaque pas (plus petit offset du lot - 1).
    var headFrom = 0
    let lossValueGrad = valueAndGrad(model: model) { model, arrays in
        let (ce, ntoks) = trainingLoss(model: model, batch: arrays[0], lengths: arrays[1], headFrom: headFrom)
        return [ce, ntoks]
    }
    memoryPolicy?.applyAtStart()

    var losses = [Float]()
    var tokenCount = 0
    var processedCount = 0
    var start = Date.timeIntervalSinceReferenceDate

    /// Validation : perte, metriques, progression ; `.stop` si l'appelant arrete.
    func runValidation(_ iteration: Int) -> LoRATrain.ProgressDisposition {
        beacon?.update(phase: "validation", step: iteration + 1, totalSteps: iterations)
        let valStart = Date.timeIntervalSinceReferenceDate
        model.train(false)  // Mode eval pour la validation
        let valLoss = evaluateTraining(
            model: model, samples: validSamples, batchSize: batchSize, maxBatches: validationBatches)
        model.train()  // Retour en mode training
        let now = Date.timeIntervalSinceReferenceDate
        metrics?(Gemma4TrainingMetrics(
            kind: .validation, iteration: iteration + 1, loss: valLoss, iterationsPerSecond: 0,
            processedTokensPerSecond: 0, trainedTokensPerSecond: 0, seconds: now - valStart,
            activeMB: Memory.activeMemory >> 20, peakMB: Memory.peakMemory >> 20))
        let disposition = progress(LoRATrain.Progress.validation(
            iteration: iteration, validationLoss: valLoss, validationTime: now - valStart))
        memoryPolicy?.afterValidationOrSave()
        start = Date.timeIntervalSinceReferenceDate
        return disposition
    }

    var lastIteration = startIteration
    for (iteration, (batch, lengths)) in TrainingBatchIterator(
        samples: trainSamples, batchSize: batchSize, train: true, seed: seed
    ).enumerated() {
        // Reprise (K-25) : rejouer le melange jusqu'au pas sauvegarde, sans calcul.
        if iteration < startIteration { continue }
        // Arret propre (A-02) : annulation -> sortie de boucle et sauvegarde finale.
        if Task.isCancelled { break }
        // Validation initiale AVANT le premier pas, comme mlx-lm (« Iter 1: Val loss ») : la
        // perte du modele de depart. Mesuree apres le pas (avant 2026-10-01), elle dependait
        // deja du lr (31B 4 bits : 2,08 a 1e-5, 1,55 a 1e-4).
        if iteration == 0 && startIteration == 0, runValidation(iteration) == .stop { break }
        // Forward + backward (ref: Python step())
        beacon?.update(phase: "train", step: iteration + 1, totalSteps: iterations)
        headFrom = responseOnlyHead ? max(0, lengths[0..., 0].min().item(Int.self) - 1) : 0
        let (resultArray, grad) = lossValueGrad(model, [batch, lengths])
        let lvalue = resultArray[0]
        let tokens = resultArray[1]

        // Ecretage global (A-01 : l'option etait acceptee et affichee, jamais appliquee).
        let clipped = gradClipMaxNorm > 0 ? clipGradNorm(gradients: grad, maxNorm: gradClipMaxNorm).0 : grad

        // Update (ref: Python optimizer.update(model, grad))
        optimizer.update(model: model, gradients: clipped)

        // eval APRES l'update pour synchroniser
        eval(model, optimizer, lvalue)
        lastIteration = iteration + 1

        losses.append(lvalue.item(Float.self))
        tokenCount += tokens.item(Int.self)
        processedCount += batch.size

        // Report
        if (iteration + 1) % stepsPerReport == 0 {
            let trainingLoss = MLXArray(losses).mean(stream: .cpu).item(Float.self)
            let now = Date.timeIntervalSinceReferenceDate
            let iterPerSec = Double(stepsPerReport) / (now - start)
            let tokPerSec = Double(tokenCount) / (now - start)

            metrics?(Gemma4TrainingMetrics(
                kind: .train, iteration: iteration + 1, loss: trainingLoss, iterationsPerSecond: iterPerSec,
                processedTokensPerSecond: Double(processedCount) / (now - start), trainedTokensPerSecond: tokPerSec,
                seconds: now - start, activeMB: Memory.activeMemory >> 20, peakMB: Memory.peakMemory >> 20))
            let p = LoRATrain.Progress.train(iteration: iteration, trainingLoss: trainingLoss,
                              iterationsPerSecond: iterPerSec, tokensPerSecond: tokPerSec)
            if progress(p) == .stop { break }
            losses.removeAll()
            tokenCount = 0
            processedCount = 0
            start = Date.timeIntervalSinceReferenceDate
        }

        // Validation
        if (iteration + 1) % stepsPerEval == 0, runValidation(iteration) == .stop { break }

        // Save
        if let url = weightsURL, (iteration + 1) % saveEvery == 0 {
            try saveTrainingWeights(
                model: model, isFullFineTune: isFullFineTune, weightsURL: url, optimizer: optimizer,
                iteration: iteration + 1, seed: seed, checkpointDirectory: checkpointDirectory)
            let p = LoRATrain.Progress.save(iteration: iteration, url: url)
            if progress(p) == .stop { break }
            start = Date.timeIntervalSinceReferenceDate
        }

        if iteration + 1 >= iterations { break }
    }

    // Sauvegarde finale (aussi apres une annulation)
    if let url = weightsURL {
        try saveTrainingWeights(
            model: model, isFullFineTune: isFullFineTune, weightsURL: url, optimizer: optimizer,
            iteration: lastIteration, seed: seed, checkpointDirectory: checkpointDirectory)
    }
}

/// Sauvegarde intermediaire ou finale : checkpoint complet (K-25) si un dossier est
/// donne, sinon ancien comportement (un fichier ecrase).
func saveTrainingWeights(
    model: Module, isFullFineTune: Bool, weightsURL: URL, optimizer: any Optimizer,
    iteration: Int, seed: UInt64, checkpointDirectory: URL?
) throws {
    let weights = Dictionary(
        uniqueKeysWithValues: (isFullFineTune ? model.parameters() : model.trainableParameters()).flattened())
    if let checkpointDirectory {
        try Gemma4TrainingCheckpoint.write(
            weights: weights, optimizer: optimizer,
            state: Gemma4TrainingState(iteration: iteration, seed: seed),
            directory: checkpointDirectory, weightsName: weightsURL.lastPathComponent)
    } else {
        try save(arrays: weights, url: weightsURL)
    }
}

/// Evaluation (ref: mlx-lm evaluate())
func evaluateTraining(
    model: Module, samples: [TrainingBatchIterator.TokenizedSample], batchSize: Int, maxBatches: Int? = nil
) -> Float {
    var allLosses = [Float]()
    var tokenCount = 0

    // A-21 : `maxBatches` borne la validation (tout le jeu a chaque evaluation sinon) ;
    // l'ordre des lots est fixe (trie par longueur), donc la mesure reste comparable.
    for (index, (batch, lengths)) in TrainingBatchIterator(
        samples: samples, batchSize: batchSize, train: false
    ).enumerated() {
        if let maxBatches, index >= maxBatches { break }
        let (losses, tokens) = trainingLoss(model: model, batch: batch, lengths: lengths)
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
    /// Embeddings media deja projetes, fournis par l'appelant : les tours ne sont pas rappelees.
    public let imageEmbeddings: MLXArray?
    public let audioEmbeddings: MLXArray?

    public init(tokens: [Int], promptOffset: Int,
                pixelValues: MLXArray? = nil,
                audioFeatures: MLXArray? = nil,
                audioMask: MLXArray? = nil,
                imageEmbeddings: MLXArray? = nil,
                audioEmbeddings: MLXArray? = nil) {
        self.tokens = tokens
        self.promptOffset = promptOffset
        self.pixelValues = pixelValues
        self.audioFeatures = audioFeatures
        self.audioMask = audioMask
        self.imageEmbeddings = imageEmbeddings
        self.audioEmbeddings = audioEmbeddings
    }
}

/// Embeddings image / audio d'un echantillon par les tours gelees (hors gradient).
func mediaEmbeddings(_ mm: Gemma4MultimodalLLMModel, _ sample: MultimodalTokenizedSample) -> (MLXArray?, MLXArray?) {
    var image = sample.imageEmbeddings
    if image == nil, let pv = sample.pixelValues {
        let features = (0 ..< pv.dim(0)).map { i in mm.embedVision(mm.visionTower(pv[i ..< (i + 1)])) }
        image = stopGradient(concatenated(features, axis: 1))
    }
    var audio = sample.audioEmbeddings
    if audio == nil, let af = sample.audioFeatures, let tower = mm.audioTower, let embedder = mm.embedAudio {
        let mask = sample.audioMask ?? MLXArray.zeros([af.dim(0), af.dim(1)], type: Bool.self)
        let (encodings, _) = tower(af, audioMelMask: mask)
        audio = stopGradient(embedder(encodings))
    }
    return (image, audio)
}


/// Iterateur batch size 1 pour samples multimodaux (media de taille variable)
public struct MultimodalBatchIterator: Sequence, IteratorProtocol {

    let samples: [MultimodalTokenizedSample]
    let train: Bool
    var permutation: [Int]
    var permIndex: Int = 0
    var rng: SeededGenerator

    public init(samples: [MultimodalTokenizedSample], train: Bool, seed: UInt64 = 0) {
        self.samples = samples
        self.train = train
        self.rng = SeededGenerator(seed: seed)
        self.permutation = Array(0 ..< samples.count)
        if train { self.permutation.shuffle(using: &rng) }
    }

    /// Echantillon suivant (ordre melange et seede identique a `next()`).
    mutating func nextSample() -> MultimodalTokenizedSample? {
        if permIndex >= permutation.count {
            if !train { return nil }
            permutation.shuffle(using: &rng)
            permIndex = 0
        }
        let sample = samples[permutation[permIndex]]
        permIndex += 1
        return sample
    }

    public mutating func next() -> (MLXArray, MLXArray, MLXArray?, MLXArray?, MLXArray?)? {
        if permIndex >= permutation.count {
            if !train { return nil }
            permutation.shuffle(using: &rng)
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
/// - Warning: exclusif dans le process (`Gemma4ComputeGate`) : echoue avec
///   `inferenceInProgress` si une inference du paquet tourne, et toute inference
///   lancee pendant l'entrainement echoue avec `trainingInProgress`. Un gradient
///   concurrent d'un forward fige le process (deadlock mlx-swift, voir CLAUDE.md).
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
    seed: UInt64 = 0,
    gradClipMaxNorm: Float = 0,
    startIteration: Int = 0,
    checkpointDirectory: URL? = nil,
    validationBatches: Int? = nil,
    metrics: ((Gemma4TrainingMetrics) -> Void)? = nil,
    responseOnlyHead: Bool = false,
    memoryPolicy: Gemma4TrainingMemoryPolicy? = nil,
    progress: (LoRATrain.Progress) -> LoRATrain.ProgressDisposition
) throws {
    // K-9 : entrainement exclusif — un gradient et un forward concurrents figent le
    // process (Gemma4ComputeGate). Refuse si une inference du paquet tourne.
    // Reprise d'un run deja termine : rien a faire (l'iterateur d'entrainement ne finit jamais).
    guard startIteration < iterations else { return }
    try Gemma4ComputeGate.shared.beginTraining()
    defer { Gemma4ComputeGate.shared.endTraining() }
    // Beacon opt-in (SiliconScope) : pas d'entrainement et validations.
    let beacon = RuntimeBeacon.begin(task: "train")
    defer { beacon?.end() }
    model.train()

    // Le modele multimodal pour setter les pending properties
    let mmModel = model as! Gemma4MultimodalLLMModel

    // Loss + grad — reutilise trainingLoss() existant
    // K-30 (a) : debut de la tete, fixe avant chaque pas (plus petit offset du lot - 1).
    var headFrom = 0
    let lossValueGrad = valueAndGrad(model: model) { model, arrays in
        let (ce, ntoks) = trainingLoss(model: model, batch: arrays[0], lengths: arrays[1], headFrom: headFrom)
        return [ce, ntoks]
    }
    memoryPolicy?.applyAtStart()

    var losses = [Float]()
    var tokenCount = 0
    var processedCount = 0
    var start = Date.timeIntervalSinceReferenceDate

    /// Validation : perte, metriques, progression ; `.stop` si l'appelant arrete.
    func runValidation(_ iteration: Int) -> LoRATrain.ProgressDisposition {
        beacon?.update(phase: "validation", step: iteration + 1, totalSteps: iterations)
        let valStart = Date.timeIntervalSinceReferenceDate
        model.train(false)  // Mode eval pour la validation
        let valLoss = evaluateMultimodalTraining(model: model, samples: validSamples, maxBatches: validationBatches)
        model.train()  // Retour en mode training
        let now = Date.timeIntervalSinceReferenceDate
        metrics?(Gemma4TrainingMetrics(
            kind: .validation, iteration: iteration + 1, loss: valLoss, iterationsPerSecond: 0,
            processedTokensPerSecond: 0, trainedTokensPerSecond: 0, seconds: now - valStart,
            activeMB: Memory.activeMemory >> 20, peakMB: Memory.peakMemory >> 20))
        let disposition = progress(LoRATrain.Progress.validation(
            iteration: iteration, validationLoss: valLoss, validationTime: now - valStart))
        memoryPolicy?.afterValidationOrSave()
        start = Date.timeIntervalSinceReferenceDate
        return disposition
    }

    var lastIteration = startIteration
    var iterator = MultimodalBatchIterator(samples: trainSamples, train: true, seed: seed)
    var iteration = -1
    while let sample = iterator.nextSample() {
        iteration += 1
        if iteration < startIteration { continue }
        if Task.isCancelled { break }
        // Validation initiale AVANT le premier pas, comme mlx-lm (« Iter 1: Val loss ») : la
        // perte du modele de depart. Mesuree apres le pas (avant 2026-10-01), elle dependait
        // deja du lr (31B 4 bits : 2,08 a 1e-5, 1,55 a 1e-4).
        if iteration == 0 && startIteration == 0, runValidation(iteration) == .stop { break }
        let batch = MLXArray(sample.tokens.map { Int32($0) }).reshaped(1, sample.tokens.count)
        let lengths = MLXArray([Int32(sample.promptOffset), Int32(sample.tokens.count)]).reshaped(1, 2)

        // Embeddings media EN DEHORS de valueAndGrad (le graphe de gradient ne trace pas les
        // tours, qui produisaient des NaN), ou fournis par l'appelant dans l'echantillon.
        // Les precalculer pour tout le jeu (K-30 e) a ete mesure sans gain sur TB3.
        let (imageEmbeddings, audioEmbeddings) = mediaEmbeddings(mmModel, sample)
        mmModel.pendingImageEmbeddings = imageEmbeddings
        mmModel.pendingAudioEmbeddings = audioEmbeddings
        mmModel.pendingPixelValues = nil
        mmModel.pendingAudioFeatures = nil
        mmModel.pendingAudioMask = nil

        beacon?.update(phase: "train", step: iteration + 1, totalSteps: iterations)
        headFrom = responseOnlyHead ? max(0, lengths[0..., 0].min().item(Int.self) - 1) : 0
        let (resultArray, grad) = lossValueGrad(model, [batch, lengths])
        let lvalue = resultArray[0]
        let tokens = resultArray[1]

        let clipped = gradClipMaxNorm > 0 ? clipGradNorm(gradients: grad, maxNorm: gradClipMaxNorm).0 : grad
        optimizer.update(model: model, gradients: clipped)
        eval(model, optimizer, lvalue)
        lastIteration = iteration + 1

        losses.append(lvalue.item(Float.self))
        tokenCount += tokens.item(Int.self)
        processedCount += batch.size

        // Report
        if (iteration + 1) % stepsPerReport == 0 {
            let trainingLoss = MLXArray(losses).mean(stream: .cpu).item(Float.self)
            let now = Date.timeIntervalSinceReferenceDate
            let iterPerSec = Double(stepsPerReport) / (now - start)
            let tokPerSec = Double(tokenCount) / (now - start)

            metrics?(Gemma4TrainingMetrics(
                kind: .train, iteration: iteration + 1, loss: trainingLoss, iterationsPerSecond: iterPerSec,
                processedTokensPerSecond: Double(processedCount) / (now - start), trainedTokensPerSecond: tokPerSec,
                seconds: now - start, activeMB: Memory.activeMemory >> 20, peakMB: Memory.peakMemory >> 20))
            let p = LoRATrain.Progress.train(iteration: iteration, trainingLoss: trainingLoss,
                              iterationsPerSecond: iterPerSec, tokensPerSecond: tokPerSec)
            if progress(p) == .stop { break }
            losses.removeAll()
            tokenCount = 0
            processedCount = 0
            start = Date.timeIntervalSinceReferenceDate
        }

        // Validation
        if (iteration + 1) % stepsPerEval == 0, runValidation(iteration) == .stop { break }

        // Save
        if let url = weightsURL, (iteration + 1) % saveEvery == 0 {
            try saveTrainingWeights(
                model: model, isFullFineTune: isFullFineTune, weightsURL: url, optimizer: optimizer,
                iteration: iteration + 1, seed: seed, checkpointDirectory: checkpointDirectory)
            let p = LoRATrain.Progress.save(iteration: iteration, url: url)
            if progress(p) == .stop { break }
            start = Date.timeIntervalSinceReferenceDate
        }

        if iteration + 1 >= iterations { break }
    }

    // Sauvegarde finale (aussi apres une annulation)
    if let url = weightsURL {
        try saveTrainingWeights(
            model: model, isFullFineTune: isFullFineTune, weightsURL: url, optimizer: optimizer,
            iteration: lastIteration, seed: seed, checkpointDirectory: checkpointDirectory)
    }
}

/// Evaluation multimodal — pre-calcule les embeddings comme le training
func evaluateMultimodalTraining(model: Module, samples: [MultimodalTokenizedSample], maxBatches: Int? = nil) -> Float {
    let mmModel = model as! Gemma4MultimodalLLMModel
    var allLosses = [Float]()
    var tokenCount = 0

    var iterator = MultimodalBatchIterator(samples: samples, train: false)
    var index = -1
    while let sample = iterator.nextSample() {
        index += 1
        if let maxBatches, index >= maxBatches { break }
        let batch = MLXArray(sample.tokens.map { Int32($0) }).reshaped(1, sample.tokens.count)
        let lengths = MLXArray([Int32(sample.promptOffset), Int32(sample.tokens.count)]).reshaped(1, 2)
        let (imageEmbeddings, audioEmbeddings) = mediaEmbeddings(mmModel, sample)
        mmModel.pendingImageEmbeddings = imageEmbeddings
        mmModel.pendingAudioEmbeddings = audioEmbeddings
        mmModel.pendingPixelValues = nil
        mmModel.pendingAudioFeatures = nil
        mmModel.pendingAudioMask = nil

        let (losses, tokens) = trainingLoss(model: model, batch: batch, lengths: lengths)
        allLosses.append((losses * tokens).item(Float.self))
        tokenCount += tokens.item(Int.self)
    }

    return tokenCount > 0
        ? (sum(MLXArray(allLosses), stream: .cpu) / tokenCount).item(Float.self)
        : 0
}
