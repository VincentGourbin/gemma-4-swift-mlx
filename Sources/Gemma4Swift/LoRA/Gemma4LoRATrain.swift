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
            enableProfiling: Bool = false
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
        }
    }

    /// Lance le fine-tuning LoRA sur un modele Gemma 4
    ///
    /// - Parameters:
    ///   - container: ModelContainer avec le modele charge
    ///   - trainData: donnees d'entrainement (textes formattes)
    ///   - validData: donnees de validation (textes formattes)
    ///   - config: configuration d'entrainement
    ///   - progress: callback de progression (retourne .stop pour arreter)
    public static func train(
        container: ModelContainer,
        trainData: [String],
        validData: [String],
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
        nonisolated(unsafe) let capturedTrainData = trainData
        nonisolated(unsafe) let capturedValidData = validData

        try await container.perform { (context: ModelContext) in
            let model = context.model
            let tokenizer = context.tokenizer

            if isFullFineTune {
                // Full SFT — tous les poids sont trainables (pas de freeze, pas de LoRA)
                // Ref: arXiv:2512.15943 — small models concentrate capacity on the task
                print("Mode: Full Fine-Tuning (tous les poids)")
            } else {
                // LoRA/DoRA — freeze base + adapter layers
                guard let languageModel = model as? LanguageModel else {
                    throw Gemma4LoRAError.incompatibleModel
                }
                let _ = try LoRAContainer.from(
                    model: languageModel,
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
            let optimizer: any Optimizer
            if isFullFineTune {
                optimizer = AdamW(learningRate: config.learningRate, weightDecay: 0.01)
            } else {
                optimizer = Adam(learningRate: config.learningRate)
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

            if config.maskPrompt {
                // Training avec response masking — loss uniquement sur les tokens de la reponse
                print("Mode: response masking active")
                let trainURL = config.outputDirectory.deletingLastPathComponent()
                let dataDir = capturedTrainData.isEmpty ? trainURL : URL(fileURLWithPath: ".")

                // Preparer les samples avec split prompt/reponse via le tokenizer
                // On reconstruit les samples depuis les donnees texte en cherchant la frontiere
                var trainSamples: [MaskedBatchIterator.Sample] = []
                var validSamples: [MaskedBatchIterator.Sample] = []

                for text in capturedTrainData {
                    // Pour les donnees chat (via chatFormatter), le texte contient tout
                    // On cherche le dernier "<|turn>model\n" comme frontiere
                    let tokens = tokenizer.encode(text: text)
                    // Heuristique: chercher le token "model" (4368) precede de "<|turn>" (105)
                    var promptEnd = 0
                    for i in 0 ..< tokens.count - 1 {
                        if tokens[i] == 105 && tokens[i + 1] == 4368 {
                            // <|turn>model — la reponse commence apres "model\n"
                            promptEnd = i + 3 // <|turn> + model + \n
                        }
                    }
                    if promptEnd > 0 && promptEnd < tokens.count {
                        trainSamples.append(MaskedBatchIterator.Sample(
                            promptTokens: Array(tokens[0 ..< promptEnd]),
                            responseTokens: Array(tokens[promptEnd...])
                        ))
                    } else {
                        // Fallback: tout est reponse
                        trainSamples.append(MaskedBatchIterator.Sample(
                            promptTokens: [], responseTokens: tokens
                        ))
                    }
                }

                for text in capturedValidData {
                    let tokens = tokenizer.encode(text: text)
                    var promptEnd = 0
                    for i in 0 ..< tokens.count - 1 {
                        if tokens[i] == 105 && tokens[i + 1] == 4368 {
                            promptEnd = i + 3
                        }
                    }
                    if promptEnd > 0 && promptEnd < tokens.count {
                        validSamples.append(MaskedBatchIterator.Sample(
                            promptTokens: Array(tokens[0 ..< promptEnd]),
                            responseTokens: Array(tokens[promptEnd...])
                        ))
                    } else {
                        validSamples.append(MaskedBatchIterator.Sample(
                            promptTokens: [], responseTokens: tokens
                        ))
                    }
                }

                print("Train: \(trainSamples.count) samples (avg response: \(trainSamples.map { $0.responseTokens.count }.reduce(0, +) / max(1, trainSamples.count)) tokens)")

                try trainWithResponseMasking(
                    model: model as! Module,
                    trainSamples: trainSamples,
                    validSamples: validSamples,
                    optimizer: optimizer,
                    iterations: config.iterations,
                    batchSize: config.batchSize,
                    stepsPerReport: config.stepsPerReport,
                    stepsPerEval: config.stepsPerEval,
                    saveEvery: config.saveEvery,
                    weightsURL: weightsURL,
                    gradClipMaxNorm: config.gradClipMaxNorm,
                    isFullFineTune: isFullFineTune,
                    progress: wrappedProgress
                )
            } else {
                // Training standard (loss sur tous les tokens)
                let params = LoRATrain.Parameters(
                    batchSize: config.batchSize,
                    iterations: config.iterations,
                    stepsPerReport: config.stepsPerReport,
                    stepsPerEval: config.stepsPerEval,
                    validationBatches: 10,
                    saveEvery: config.saveEvery,
                    adapterURL: weightsURL
                )

                try LoRATrain.train(
                    model: model as! Module,
                    train: capturedTrainData,
                    validate: capturedValidData,
                    optimizer: optimizer,
                    tokenizer: tokenizer,
                    parameters: params,
                    progress: wrappedProgress
                )

                try LoRATrain.saveLoRAWeights(model: model as! Module, url: weightsURL)
            }

            // Sauvegarde finale selon le mode
            if isFullFineTune {
                // Full SFT: sauvegarder TOUS les poids du modele
                print("Sauvegarde de tous les poids du modele...")
                let allParams = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
                try save(arrays: allParams, url: weightsURL)
            }
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

    /// Evalue un modele avec adapter sur un dataset de test
    public static func evaluate(
        container: ModelContainer,
        testData: [String],
        batchSize: Int = 1
    ) async throws -> Float {
        try await container.perform { context in
            let model = context.model
            let tokenizer = context.tokenizer
            return LoRATrain.evaluate(
                model: model as! Module,
                dataset: testData,
                tokenizer: tokenizer,
                batchSize: batchSize,
                batchCount: 0
            )
        }
    }
}
