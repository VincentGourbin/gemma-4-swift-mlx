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

    /// Configuration d'entrainement
    public struct TrainingConfig: Sendable {
        /// Rang LoRA (complexite de l'adapter)
        public var loraRank: Int
        /// Facteur d'echelle LoRA
        public var loraScale: Float
        /// Nombre de couches a adapter (nil = default par famille)
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
        /// Repertoire de sortie pour les adapters
        public var outputDirectory: URL
        /// Utiliser DoRA au lieu de LoRA
        public var useDora: Bool
        /// Activer le profiling du training
        public var enableProfiling: Bool

        public init(
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
            useDora: Bool = false,
            enableProfiling: Bool = false
        ) {
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
            self.useDora = useDora
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

        let adapterURL = config.outputDirectory.appending(component: "adapters.safetensors")

        // Configuration LoRA
        let loraConfig = Gemma4LoRADefaults.configuration(
            for: config.modelFamily,
            rank: config.loraRank,
            scale: config.loraScale,
            numLayers: config.numLayers,
            useDora: config.useDora
        )

        // Profiling
        let profiler = MLXProfiler.shared
        if config.enableProfiling {
            profiler.enable()
            profiler.startTrainingSession(config: [
                "model_family": config.modelFamily.rawValue,
                "lora_rank": "\(config.loraRank)",
                "lora_scale": "\(config.loraScale)",
                "num_layers": "\(loraConfig.numLayers)",
                "learning_rate": "\(config.learningRate)",
                "batch_size": "\(config.batchSize)",
                "iterations": "\(config.iterations)",
                "train_samples": "\(trainData.count)",
                "valid_samples": "\(validData.count)",
            ])
        }

        // Entrainement dans le contexte du container
        nonisolated(unsafe) let capturedTrainData = trainData
        nonisolated(unsafe) let capturedValidData = validData

        try await container.perform { context in
            let model = context.model
            let tokenizer = context.tokenizer

            // Appliquer LoRA au modele
            guard let languageModel = model as? LanguageModel else {
                throw Gemma4LoRAError.incompatibleModel
            }

            let _ = try LoRAContainer.from(
                model: languageModel,
                configuration: loraConfig
            )

            // Afficher le nombre de parametres trainables
            let trainableParams = model.trainableParameters()
                .flattened()
                .reduce(0) { $0 + $1.1.size }
            let totalParams = model.parameters()
                .flattened()
                .reduce(0) { $0 + $1.1.size }
            let pct = Double(trainableParams) / Double(totalParams) * 100
            print("Parametres trainables: \(trainableParams) / \(totalParams) (\(String(format: "%.2f", pct))%)")

            // Optimizer
            let optimizer = Adam(learningRate: config.learningRate)

            // Parametres de training
            let params = LoRATrain.Parameters(
                batchSize: config.batchSize,
                iterations: config.iterations,
                stepsPerReport: config.stepsPerReport,
                stepsPerEval: config.stepsPerEval,
                validationBatches: 10,
                saveEvery: config.saveEvery,
                adapterURL: adapterURL
            )

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

            // Lancer le training
            try LoRATrain.train(
                model: model as! Module,
                train: capturedTrainData,
                validate: capturedValidData,
                optimizer: optimizer,
                tokenizer: tokenizer,
                parameters: params,
                progress: wrappedProgress
            )

            // Sauvegarde finale inconditionnelle (le training ne sauvegarde qu'aux multiples de saveEvery)
            try LoRATrain.saveLoRAWeights(model: model as! Module, url: adapterURL)
        }

        // Sauvegarder la config de l'adapter
        let configData = try JSONEncoder().encode(loraConfig)
        let configURL = config.outputDirectory.appending(component: "adapter_config.json")
        try configData.write(to: configURL)

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
