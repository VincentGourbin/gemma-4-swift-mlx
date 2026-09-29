// Orchestrateur de training LoRA pour Gemma 4

import Foundation
import MLX
import MLXNN
import MLXLMCommon
import MLXLLM
import MLXOptimizers
import Tokenizers
import MLXProfiler

/// Orchestrateur de fine-tuning LoRA pour les modeles Gemma 4.
/// Wrapper autour de `LoRATrain` de mlx-swift-lm avec gestion
/// du chat template, config par modele, et profiling.
public enum Gemma4LoRATrain {

    /// Type de fine-tuning
    public enum FineTuneType: String, Sendable {
        case lora
        case dora
        case full  // Full SFT — tous les poids sont entraines
    }

    /// Configuration d'entrainement
    public struct TrainingConfig: Sendable {
        /// Type de fine-tuning (lora, dora, full)
        public var fineTuneType: FineTuneType
        /// Rang LoRA (ignore en mode full)
        public var loraRank: Int
        /// Facteur d'echelle LoRA (ignore en mode full)
        public var loraScale: Float
        /// Nombre de couches a adapter (nil = default par famille, ignore en mode full)
        public var numLayers: Int?
        /// Famille de modele (pour les defaults)
        public var modelFamily: Gemma4LoRADefaults.ModelFamily
        /// Learning rate
        public var learningRate: Float
        /// Taille du batch
        public var batchSize: Int
        /// Nombre d'iterations
        public var iterations: Int
        /// Steps entre les rapports de loss
        public var stepsPerReport: Int
        /// Steps entre les evaluations de validation
        public var stepsPerEval: Int
        /// Sauvegarder tous les N steps
        public var saveEvery: Int
        /// Repertoire de sortie
        public var outputDirectory: URL
        /// Masquer le prompt (ne calculer la loss que sur la reponse)
        public var maskPrompt: Bool
        /// Gradient clipping max norm (0 = desactive, papier arXiv:2512.15943 recommande 0.3)
        public var gradClipMaxNorm: Float
        /// Activer le profiling du training
        public var enableProfiling: Bool
        /// Graine : initialisation LoRA, dropout **et** melange des exemples (A-08).
        public var seed: UInt64
        /// Longueur maximale d'un exemple en jetons ; nil = aucune troncature (defaut).
        /// Pas de 2048 par defaut comme mlx-lm : sur le dataset director de Fluxforge, 59 %
        /// des exemples depassent 2048 jetons et la troncature coupe la fin de la reponse
        /// (E7 29/30 -> 27/30, mesure du 2026-09-29). Les exemples longs sont signales.
        public var maxSeqLength: Int?
        /// Reprendre au dernier checkpoint de `outputDirectory` (poids, etat de l'optimiseur, pas).
        public var resume: Bool
        /// Lots de validation au plus (nil = tout le jeu ; A-21).
        public var validationBatches: Int?
        /// Fichier JSONL ou ecrire une ligne par rapport et validation (K-28).
        public var metricsURL: URL?
        /// Tete et perte sur les seules positions de reponse avec `maskPrompt` (K-30 a ;
        /// director, 200 pas : +13 % de debit, -19 % de pic MLX, perte identique).
        public var responseOnlyHead: Bool
        /// Limite de cache MLX et vidage apres validation (K-30 b ; director : empreinte
        /// 76 -> 16 Go, perte identique) ; nil = aucune politique.
        public var memoryPolicy: Gemma4TrainingMemoryPolicy?

        public init(
            fineTuneType: FineTuneType = .lora,
            loraRank: Int = 8,
            loraScale: Float = 20.0,
            numLayers: Int? = nil,
            modelFamily: Gemma4LoRADefaults.ModelFamily = .e2b,
            learningRate: Float = 1e-5,
            batchSize: Int = 1,
            iterations: Int = 200,
            stepsPerReport: Int = 10,
            stepsPerEval: Int = 50,
            saveEvery: Int = 50,
            outputDirectory: URL = URL(fileURLWithPath: "./adapters"),
            maskPrompt: Bool = false,
            gradClipMaxNorm: Float = 0,
            enableProfiling: Bool = false,
            seed: UInt64 = 0,
            maxSeqLength: Int? = nil,
            resume: Bool = false,
            validationBatches: Int? = nil,
            metricsURL: URL? = nil,
            responseOnlyHead: Bool = true,
            memoryPolicy: Gemma4TrainingMemoryPolicy? = Gemma4TrainingMemoryPolicy(cacheLimitMB: 2048)
        ) {
            self.fineTuneType = fineTuneType
            self.loraRank = loraRank
            self.loraScale = loraScale
            self.numLayers = numLayers
            self.modelFamily = modelFamily
            self.learningRate = learningRate
            self.batchSize = batchSize
            self.iterations = iterations
            self.stepsPerReport = stepsPerReport
            self.stepsPerEval = stepsPerEval
            self.saveEvery = saveEvery
            self.outputDirectory = outputDirectory
            self.maskPrompt = maskPrompt
            self.gradClipMaxNorm = gradClipMaxNorm
            self.enableProfiling = enableProfiling
            self.seed = seed
            self.maxSeqLength = maxSeqLength
            self.resume = resume
            self.validationBatches = validationBatches
            self.metricsURL = metricsURL
            self.responseOnlyHead = responseOnlyHead
            self.memoryPolicy = memoryPolicy
        }
    }

    public enum TrainingSetupError: LocalizedError, Equatable {
        case fullFineTuneOnQuantizedModel

        public var errorDescription: String? {
            switch self {
            case .fullFineTuneOnQuantizedModel:
                return "--fine-tune-type full sur un pack quantifie n'entraine presque rien (les QuantizedLinear "
                    + "sont geles) : utiliser lora/dora, ou un modele bf16 pour full"
            }
        }
    }

    /// Tronque les exemples a `maxLength` jetons (A-06 : un exemple de 20 k jetons partait tel
    /// quel). Rend aussi le nombre d'exemples tronques, a signaler.
    public static func truncate(_ samples: [[Int]], maxLength: Int?) -> (samples: [[Int]], truncated: Int) {
        guard let maxLength, maxLength > 1 else { return (samples, 0) }
        var truncated = 0
        let result = samples.map { tokens -> [Int] in
            guard tokens.count > maxLength else { return tokens }
            truncated += 1
            return Array(tokens.prefix(maxLength))
        }
        return (result, truncated)
    }

    /// Exemple d'entrainement : frontiere prompt / reponse au dernier `<|turn>model\n`
    /// si `maskPrompt`, sinon 0. `nil` si moins de 2 jetons.
    public static func trainingSample(_ tokens: [Int], maskPrompt: Bool) -> TrainingBatchIterator.TokenizedSample? {
        guard tokens.count > 1 else { return nil }
        var offset = 0
        if maskPrompt {
            for i in 0 ..< tokens.count - 1 where tokens[i] == 105 && tokens[i + 1] == 4368 {
                offset = i + 3
            }
        }
        return TrainingBatchIterator.TokenizedSample(tokens: tokens, promptOffset: offset)
    }

    /// Perte moyenne par jeton sur des ids directs, **meme masquage et meme boucle que la
    /// validation de l'entrainement** (A-03 : `evaluate` amont re-encodait du texte et ne
    /// masquait pas le prompt, donc n'etait pas comparable a la val loss).
    public static func evaluateMasked(
        container: ModelContainer, samples: [[Int]], maskPrompt: Bool, batchSize: Int = 1
    ) async throws -> Float {
        try Gemma4ComputeGate.shared.beginInference()
        defer { Gemma4ComputeGate.shared.endInference() }
        return await container.perform { context in
            let prepared = samples.compactMap { trainingSample($0, maskPrompt: maskPrompt) }
            context.model.train(false)
            return evaluateTraining(model: context.model, samples: prepared, batchSize: batchSize)
        }
    }

    /// `full` n'a de sens que si les poids ne sont pas quantifies (A-07).
    static func checkFullFineTune(_ model: Module) throws {
        if model.leafModules().flattened().contains(where: { $0.1 is Quantized }) {
            throw TrainingSetupError.fullFineTuneOnQuantizedModel
        }
    }

    /// Lance le fine-tuning LoRA sur un modele Gemma 4
    ///
    /// - Parameters:
    ///   - container: ModelContainer avec le modele charge
    ///   - trainData: tokens pre-tokenises (chaque element = une sequence de token IDs)
    ///   - validData: tokens de validation pre-tokenises
    ///   - config: configuration d'entrainement
    ///   - progress: callback de progression (retourne .stop pour arreter)
    public static func train(
        container: ModelContainer,
        trainData: [[Int]],
        validData: [[Int]],
        config: TrainingConfig,
        progress: @escaping @Sendable (LoRATrain.Progress) -> LoRATrain.ProgressDisposition
    ) async throws {
        // Creer le repertoire de sortie
        try FileManager.default.createDirectory(
            at: config.outputDirectory,
            withIntermediateDirectories: true
        )

        let isFullFineTune = config.fineTuneType == .full
        let weightsFilename = isFullFineTune ? "model.safetensors" : "adapters.safetensors"
        let weightsURL = config.outputDirectory.appending(component: weightsFilename)

        // Configuration LoRA (ignore en mode full)
        let loraConfig = Gemma4LoRADefaults.configuration(
            for: config.modelFamily,
            rank: config.loraRank,
            scale: config.loraScale,
            numLayers: config.numLayers,
            useDora: config.fineTuneType == .dora
        )
        // Config ecrite des le demarrage (K-25) : un run interrompu laisse un adaptateur
        // chargeable (LoRAContainer.from(directory:) exige ce fichier).
        if config.fineTuneType != .full {
            try Gemma4TrainingCheckpoint.atomicWrite(
                try JSONEncoder().encode(loraConfig),
                to: config.outputDirectory.appending(component: "adapter_config.json"))
        }

        // Profiling
        let profiler = MLXProfiler.shared
        if config.enableProfiling {
            profiler.enable()
            profiler.startTrainingSession(config: [
                "fine_tune_type": config.fineTuneType.rawValue,
                "model_family": config.modelFamily.rawValue,
                "learning_rate": "\(config.learningRate)",
                "batch_size": "\(config.batchSize)",
                "iterations": "\(config.iterations)",
                "grad_clip": "\(config.gradClipMaxNorm)",
                "train_samples": "\(trainData.count)",
                "valid_samples": "\(validData.count)",
            ])
        }

        // Entrainement dans le contexte du container
        let longest = (trainData + validData).map(\.count).max() ?? 0
        let long = (trainData + validData).filter { $0.count > 2048 }.count
        if config.maxSeqLength == nil && long > 0 {
            print("Note : \(long) exemple(s) de plus de 2048 jetons (max \(longest)), non tronques "
                + "(--max-seq-length pour borner la memoire)")
        }
        let (capturedTrainData, truncatedTrain) = truncate(trainData, maxLength: config.maxSeqLength)
        let (capturedValidData, truncatedValid) = truncate(validData, maxLength: config.maxSeqLength)
        if truncatedTrain + truncatedValid > 0 {
            print("Troncature a \(config.maxSeqLength ?? 0) jetons : \(truncatedTrain) exemple(s) d'entrainement, "
                + "\(truncatedValid) de validation")
        }

        try await container.perform { (context: ModelContext) in
            let model = context.model

            // Graine avant l'initialisation LoRA (ref: Python seed=0)
            MLXRandom.seed(config.seed)

            if isFullFineTune {
                try checkFullFineTune(model)
                // Full SFT — tous les poids sont trainables (pas de freeze, pas de LoRA)
                // Ref: arXiv:2512.15943 — small models concentrate capacity on the task
                print("Mode: Full Fine-Tuning (tous les poids)")
            } else {
                // LoRA/DoRA — freeze base + adapter layers
                let _ = try LoRAContainer.from(
                    model: model,
                    configuration: loraConfig
                )
            }

            // Afficher le nombre de parametres trainables
            let trainableParams = model.trainableParameters()
                .flattened()
                .reduce(0) { $0 + $1.1.size }
            let totalParams = model.parameters()
                .flattened()
                .reduce(0) { $0 + $1.1.size }
            let pct = Double(trainableParams) / Double(totalParams) * 100
            print("Parametres trainables: \(trainableParams) / \(totalParams) (\(String(format: "%.2f", pct))%)")

            // Optimizer — AdamW avec weight decay pour full SFT (ref papier: 0.01)
            // Meme calcul qu'Adam / AdamW de MLXOptimizers, etat sauvegardable (K-25).
            let optimizer = Gemma4ResumableAdam(
                learningRate: config.learningRate, weightDecay: isFullFineTune ? 0.01 : 0)

            // Reprise : poids et etat de l'optimiseur du dernier checkpoint, pas suivant.
            var startIteration = 0
            if config.resume, let state = Gemma4TrainingCheckpoint.readState(in: config.outputDirectory) {
                try Gemma4TrainingCheckpoint.restore(
                    into: model, optimizer: optimizer, directory: config.outputDirectory,
                    weightsName: weightsURL.lastPathComponent)
                startIteration = state.iteration
                print("Reprise au pas \(state.iteration) (graine \(state.seed))")
            }

            // Callback avec profiling
            let wrappedProgress: (LoRATrain.Progress) -> LoRATrain.ProgressDisposition = { p in
                if config.enableProfiling {
                    switch p {
                    case .train(let iteration, let loss, _, let tokPerSec):
                        let mem = SystemMetrics.mlxMemory()
                        profiler.recordTrainingStep(TrainingStepMetrics(
                            iteration: iteration,
                            loss: loss,
                            tokensPerSecond: tokPerSec,
                            learningRate: config.learningRate,
                            mlxActiveBytes: mem.activeBytes,
                            mlxPeakBytes: mem.peakBytes,
                            gpuUtilization: SystemMetrics.gpuUtilization(),
                            durationUs: 0
                        ))
                    case .validation(let iteration, let valLoss, let valTime):
                        profiler.recordValidation(
                            iteration: iteration,
                            loss: valLoss,
                            duration: valTime
                        )
                    case .save:
                        break
                    }
                }
                return progress(p)
            }

            // Creer les samples de training a partir des tokens pre-tokenises
            let trainSamples = capturedTrainData.compactMap { tokens -> TrainingBatchIterator.TokenizedSample? in
                guard tokens.count > 1 else { return nil }
                if config.maskPrompt {
                    // Trouver le prompt offset (dernier <|turn>model\n)
                    var offset = 0
                    for i in 0 ..< tokens.count - 1 {
                        if tokens[i] == 105 && tokens[i + 1] == 4368 {
                            offset = i + 3
                        }
                    }
                    return TrainingBatchIterator.TokenizedSample(tokens: tokens, promptOffset: offset)
                } else {
                    return TrainingBatchIterator.TokenizedSample(tokens: tokens, promptOffset: 0)
                }
            }
            let validSamples = capturedValidData.compactMap { tokens -> TrainingBatchIterator.TokenizedSample? in
                guard tokens.count > 1 else { return nil }
                if config.maskPrompt {
                    var offset = 0
                    for i in 0 ..< tokens.count - 1 {
                        if tokens[i] == 105 && tokens[i + 1] == 4368 {
                            offset = i + 3
                        }
                    }
                    return TrainingBatchIterator.TokenizedSample(tokens: tokens, promptOffset: offset)
                } else {
                    return TrainingBatchIterator.TokenizedSample(tokens: tokens, promptOffset: 0)
                }
            }

            let avgResp = config.maskPrompt
                ? trainSamples.map { $0.tokens.count - $0.promptOffset }.reduce(0, +) / max(1, trainSamples.count)
                : trainSamples.map { $0.tokens.count }.reduce(0, +) / max(1, trainSamples.count)
            print("Train: \(trainSamples.count) samples (avg \(config.maskPrompt ? "response" : "total"): \(avgResp) tokens)")

            // Training loop custom (ref: mlx-lm train())
            try trainLoRA(
                model: model,
                trainSamples: trainSamples,
                validSamples: validSamples,
                optimizer: optimizer,
                iterations: config.iterations,
                batchSize: config.batchSize,
                stepsPerReport: config.stepsPerReport,
                stepsPerEval: config.stepsPerEval,
                saveEvery: config.saveEvery,
                weightsURL: weightsURL,
                isFullFineTune: isFullFineTune,
                seed: config.seed,
                gradClipMaxNorm: config.gradClipMaxNorm,
                startIteration: startIteration,
                checkpointDirectory: config.outputDirectory,
                validationBatches: config.validationBatches,
                metrics: config.metricsURL.map { url in { Gemma4TrainingMetricsWriter.append($0, to: url) } },
                responseOnlyHead: config.responseOnlyHead && config.maskPrompt,
                memoryPolicy: config.memoryPolicy,
                progress: wrappedProgress
            )

            // (la sauvegarde finale est dans trainLoRA)
        }

        // Sauvegarder la config
        if isFullFineTune {
            // En mode full, copier les fichiers de config du modele source
            // (le modele complet est autonome)
        } else {
            // En mode LoRA/DoRA, sauvegarder la config de l'adapter
            let configData = try JSONEncoder().encode(loraConfig)
            let configURL = config.outputDirectory.appending(component: "adapter_config.json")
            try configData.write(to: configURL)
        }

        // Exporter le profiling
        if config.enableProfiling, let session = profiler.activeSession {
            let summary = profiler.getTrainingSummary()
            print("\n--- Training Summary ---")
            print("Iterations: \(summary.totalIterations)")
            print("Loss finale: \(String(format: "%.4f", summary.finalLoss))")
            print("Meilleure loss: \(String(format: "%.4f", summary.bestLoss)) (iter \(summary.bestIteration))")
            print("Tokens/sec moyen: \(String(format: "%.1f", summary.avgTokensPerSecond))")
            print("Memoire pic: \(String(format: "%.0f", summary.peakMemoryMB)) Mo")
            print("Duree totale: \(String(format: "%.1f", summary.totalTrainingTime))s")

            let traceData = ChromeTraceExporter.export(session: session)
            let traceURL = config.outputDirectory.appending(component: "training_trace.json")
            try traceData.write(to: traceURL)
            print("Trace Chrome exportee: \(traceURL.path())")
        }

        print("Adapter sauvegarde dans \(config.outputDirectory.path())")
    }

    // MARK: - Training multimodal

    /// Lance le fine-tuning LoRA multimodal (audio/image) sur un modele Gemma 4
    ///
    /// - Parameters:
    ///   - container: ModelContainer avec le modele multimodal charge
    ///   - trainData: samples multimodaux pre-tokenises avec features media
    ///   - validData: samples de validation
    ///   - config: configuration d'entrainement
    ///   - progress: callback de progression
    public static func trainMultimodal(
        container: ModelContainer,
        trainData: [MultimodalTokenizedSample],
        validData: [MultimodalTokenizedSample],
        config: TrainingConfig,
        progress: @escaping @Sendable (LoRATrain.Progress) -> LoRATrain.ProgressDisposition
    ) async throws {
        try FileManager.default.createDirectory(
            at: config.outputDirectory,
            withIntermediateDirectories: true
        )

        let isFullFineTune = config.fineTuneType == .full
        let weightsFilename = isFullFineTune ? "model.safetensors" : "adapters.safetensors"
        let weightsURL = config.outputDirectory.appending(component: weightsFilename)

        let loraConfig = Gemma4LoRADefaults.configuration(
            for: config.modelFamily,
            rank: config.loraRank,
            scale: config.loraScale,
            numLayers: config.numLayers,
            useDora: config.fineTuneType == .dora
        )
        // Config ecrite des le demarrage (K-25) : un run interrompu laisse un adaptateur
        // chargeable (LoRAContainer.from(directory:) exige ce fichier).
        if config.fineTuneType != .full {
            try Gemma4TrainingCheckpoint.atomicWrite(
                try JSONEncoder().encode(loraConfig),
                to: config.outputDirectory.appending(component: "adapter_config.json"))
        }

        // Profiling
        let profiler = MLXProfiler.shared
        if config.enableProfiling {
            profiler.enable()
            profiler.startTrainingSession(config: [
                "fine_tune_type": config.fineTuneType.rawValue,
                "mode": "multimodal",
                "model_family": config.modelFamily.rawValue,
                "learning_rate": "\(config.learningRate)",
                "iterations": "\(config.iterations)",
                "train_samples": "\(trainData.count)",
                "valid_samples": "\(validData.count)",
            ])
        }

        nonisolated(unsafe) let capturedTrainData = trainData
        nonisolated(unsafe) let capturedValidData = validData

        try await container.perform { (context: ModelContext) in
            let model = context.model

            // Convertir le modele en float32 pour eviter les NaN en bf16
            // sur les sequences longues (>300 tokens avec images)
            model.apply { array in
                array.dtype.isFloatingPoint ? array.asType(.float32) : array
            }
            print("Modele converti en float32 pour stabilite numerique")

            MLXRandom.seed(config.seed)

            if isFullFineTune {
                try checkFullFineTune(model)
                print("Mode: Full Fine-Tuning multimodal (tous les poids)")
            } else {
                let _ = try LoRAContainer.from(
                    model: model,
                    configuration: loraConfig
                )
            }

            let trainableParams = model.trainableParameters()
                .flattened()
                .reduce(0) { $0 + $1.1.size }
            let totalParams = model.parameters()
                .flattened()
                .reduce(0) { $0 + $1.1.size }
            let pct = Double(trainableParams) / Double(totalParams) * 100
            print("Parametres trainables: \(trainableParams) / \(totalParams) (\(String(format: "%.2f", pct))%)")

            // Meme calcul qu'Adam / AdamW de MLXOptimizers, etat sauvegardable (K-25).
            let optimizer = Gemma4ResumableAdam(
                learningRate: config.learningRate, weightDecay: isFullFineTune ? 0.01 : 0)

            // Reprise : poids et etat de l'optimiseur du dernier checkpoint, pas suivant.
            var startIteration = 0
            if config.resume, let state = Gemma4TrainingCheckpoint.readState(in: config.outputDirectory) {
                try Gemma4TrainingCheckpoint.restore(
                    into: model, optimizer: optimizer, directory: config.outputDirectory,
                    weightsName: weightsURL.lastPathComponent)
                startIteration = state.iteration
                print("Reprise au pas \(state.iteration) (graine \(state.seed))")
            }

            // Callback avec profiling
            let wrappedProgress: (LoRATrain.Progress) -> LoRATrain.ProgressDisposition = { p in
                if config.enableProfiling {
                    switch p {
                    case .train(let iteration, let loss, _, let tokPerSec):
                        let mem = SystemMetrics.mlxMemory()
                        profiler.recordTrainingStep(TrainingStepMetrics(
                            iteration: iteration,
                            loss: loss,
                            tokensPerSecond: tokPerSec,
                            learningRate: config.learningRate,
                            mlxActiveBytes: mem.activeBytes,
                            mlxPeakBytes: mem.peakBytes,
                            gpuUtilization: SystemMetrics.gpuUtilization(),
                            durationUs: 0
                        ))
                    case .validation(let iteration, let valLoss, let valTime):
                        profiler.recordValidation(
                            iteration: iteration,
                            loss: valLoss,
                            duration: valTime
                        )
                    case .save:
                        break
                    }
                }
                return progress(p)
            }

            let audioCount = capturedTrainData.filter { $0.audioFeatures != nil }.count
            let imageCount = capturedTrainData.filter { $0.pixelValues != nil }.count
            print("Train multimodal: \(capturedTrainData.count) samples (\(audioCount) audio, \(imageCount) image)")

            try trainMultimodalLoRA(
                model: model,
                trainSamples: capturedTrainData,
                validSamples: capturedValidData,
                optimizer: optimizer,
                iterations: config.iterations,
                stepsPerReport: config.stepsPerReport,
                stepsPerEval: config.stepsPerEval,
                saveEvery: config.saveEvery,
                weightsURL: weightsURL,
                isFullFineTune: isFullFineTune,
                seed: config.seed,
                gradClipMaxNorm: config.gradClipMaxNorm,
                startIteration: startIteration,
                checkpointDirectory: config.outputDirectory,
                validationBatches: config.validationBatches,
                metrics: config.metricsURL.map { url in { Gemma4TrainingMetricsWriter.append($0, to: url) } },
                responseOnlyHead: config.responseOnlyHead && config.maskPrompt,
                memoryPolicy: config.memoryPolicy,
                progress: wrappedProgress
            )
        }

        // Sauvegarder la config
        if !isFullFineTune {
            let configData = try JSONEncoder().encode(loraConfig)
            let configURL = config.outputDirectory.appending(component: "adapter_config.json")
            try configData.write(to: configURL)
        }

        // Exporter le profiling
        if config.enableProfiling, let session = profiler.activeSession {
            let summary = profiler.getTrainingSummary()
            print("\n--- Training Summary (Multimodal) ---")
            print("Iterations: \(summary.totalIterations)")
            print("Loss finale: \(String(format: "%.4f", summary.finalLoss))")
            print("Meilleure loss: \(String(format: "%.4f", summary.bestLoss)) (iter \(summary.bestIteration))")
            print("Tokens/sec moyen: \(String(format: "%.1f", summary.avgTokensPerSecond))")
            print("Memoire pic: \(String(format: "%.0f", summary.peakMemoryMB)) Mo")

            let traceData = ChromeTraceExporter.export(session: session)
            let traceURL = config.outputDirectory.appending(component: "training_trace.json")
            try traceData.write(to: traceURL)
        }

        print("Adapter multimodal sauvegarde dans \(config.outputDirectory.path())")
    }

    /// Evalue un modele avec adapter sur un dataset de test
    public static func evaluate(
        container: ModelContainer,
        testData: [String],
        batchSize: Int = 1
    ) async throws -> Float {
        await container.perform { context in
            let model = context.model
            let tokenizer = context.tokenizer
            return LoRATrain.evaluate(
                model: model,
                dataset: testData,
                tokenizer: tokenizer,
                batchSize: batchSize,
                batchCount: 0
            )
        }
    }
}

/// Ecrit les mesures d'entrainement en JSONL (une ligne par evenement), avec l'empreinte
/// physique du processus (ce que voit le systeme, au-dela de la memoire MLX).
public enum Gemma4TrainingMetricsWriter {
    public static func append(_ metrics: Gemma4TrainingMetrics, to url: URL) {
        guard var object = (try? JSONSerialization.jsonObject(with: JSONEncoder().encode(metrics))) as? [String: Any]
        else { return }
        object["phys_footprint_mb"] = physFootprintMB()
        object["date"] = ISO8601DateFormatter().string(from: Date())
        guard var line = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]) else { return }
        line.append(0x0A)
        if let handle = try? FileHandle(forWritingTo: url) {
            defer { try? handle.close() }
            _ = try? handle.seekToEnd()
            try? handle.write(contentsOf: line)
        } else {
            try? line.write(to: url)
        }
    }

    static func physFootprintMB() -> Int {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return result == KERN_SUCCESS ? Int(info.phys_footprint >> 20) : 0
    }
}

