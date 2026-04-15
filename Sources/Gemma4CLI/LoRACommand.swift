// Commandes CLI pour le fine-tuning LoRA/QLoRA

import ArgumentParser
import Foundation
import Gemma4Swift
import MLX
import MLXLMCommon
import MLXLLM
import MLXProfiler

struct LoRA: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "lora",
        abstract: "Fine-tuning LoRA/QLoRA pour Gemma 4",
        subcommands: [Train.self, Eval.self, Fuse.self, LoRAGenerate.self]
    )
}

// MARK: - Train

extension LoRA {
    struct Train: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Entraine un adapter LoRA sur un dataset"
        )

        @Option(name: .long, help: "Chemin local vers le modele de base")
        var modelPath: String

        @Option(name: .long, help: "Repertoire contenant train.jsonl et valid.jsonl")
        var data: String

        @Option(name: .long, help: "Repertoire de sortie pour l'adapter")
        var output: String = "./adapters"

        @Option(name: .long, help: "Rang LoRA")
        var rank: Int = 8

        @Option(name: .long, help: "Facteur d'echelle LoRA")
        var scale: Float = 20.0

        @Option(name: .long, help: "Nombre de couches a adapter (auto si omis)")
        var numLayers: Int?

        @Option(name: .long, help: "Learning rate")
        var learningRate: Float = 1e-5

        @Option(name: .long, help: "Taille du batch")
        var batchSize: Int = 1

        @Option(name: .long, help: "Nombre d'iterations")
        var iterations: Int = 200

        @Option(name: .long, help: "Steps entre les rapports de loss")
        var stepsPerReport: Int = 10

        @Option(name: .long, help: "Steps entre les evaluations")
        var stepsPerEval: Int = 50

        @Flag(name: .long, help: "Utiliser DoRA au lieu de LoRA (meilleur pour la generation structuree)")
        var dora: Bool = false

        @Flag(name: .long, help: "Activer le profiling (exporte Chrome Trace)")
        var profile: Bool = false

        func run() async throws {
            // 1. Enregistrer et charger le modele
            print("Chargement du modele: \(modelPath)")
            let container = try await loadLocalModel(path: modelPath)
            print("Modele charge. GPU: \(MLX.GPU.activeMemory / (1024 * 1024)) Mo")

            // 2. Detecter la famille de modele
            let family = Gemma4LoRADefaults.ModelFamily.from(modelId: modelPath)
            print("Famille detectee: \(family.rawValue) (\(family.totalLayers) couches)")

            // 3. Charger les donnees avec le tokenizer pour coherence chat template
            let dataURL = URL(fileURLWithPath: data)
            print("Chargement des donnees depuis \(data)...")
            let (trainData, validData) = try await container.perform { context -> ([String], [String]) in
                let tok = context.tokenizer
                let formatter: ([[String: String]]) throws -> String = { messages in
                    let ids = try tok.applyChatTemplate(messages: messages)
                    return tok.decode(tokenIds: ids)
                }
                let train = try loadGemma4TrainingData(directory: dataURL, name: "train", chatFormatter: formatter)
                let valid = try loadGemma4TrainingData(directory: dataURL, name: "valid", chatFormatter: formatter)
                return (train, valid)
            }
            print("Train: \(trainData.count) samples, Valid: \(validData.count) samples")

            // 4. Configurer le profiling
            if profile {
                let profiler = MLXProfiler.shared
                profiler.enable()
                profiler.activeSession = ProfilingSession(config: .detailed)
            }

            // 5. Configurer et lancer le training
            let config = Gemma4LoRATrain.TrainingConfig(
                loraRank: rank,
                loraScale: scale,
                numLayers: numLayers,
                modelFamily: family,
                learningRate: learningRate,
                batchSize: batchSize,
                iterations: iterations,
                stepsPerReport: stepsPerReport,
                stepsPerEval: stepsPerEval,
                saveEvery: 50,
                outputDirectory: URL(fileURLWithPath: output),
                useDora: dora,
                enableProfiling: profile
            )

            print("\n--- Debut du training ---")
            print("Mode: \(dora ? "DoRA" : "LoRA"), Rank: \(rank), Scale: \(scale), LR: \(learningRate)")
            print("Batch: \(batchSize), Iterations: \(iterations)")
            print("Couches: \(numLayers ?? family.defaultNumLayers)")
            print("Sortie: \(output)")
            print("---\n")

            try await Gemma4LoRATrain.train(
                container: container,
                trainData: trainData,
                validData: validData,
                config: config
            ) { progress in
                print(progress)
                return .more
            }

            print("\nTraining termine.")
            print("GPU pic: \(MLX.GPU.peakMemory / (1024 * 1024)) Mo")
        }
    }
}

// MARK: - Eval

extension LoRA {
    struct Eval: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Evalue la loss d'un modele avec adapter sur un dataset"
        )

        @Option(name: .long, help: "Chemin local vers le modele de base")
        var modelPath: String

        @Option(name: .long, help: "Chemin vers le repertoire de l'adapter")
        var adapterPath: String

        @Option(name: .long, help: "Repertoire contenant test.jsonl")
        var data: String

        @Option(name: .long, help: "Taille du batch")
        var batchSize: Int = 1

        func run() async throws {
            print("Chargement du modele: \(modelPath)")
            let container = try await loadLocalModel(path: modelPath)

            print("Chargement de l'adapter: \(adapterPath)")
            try await Gemma4LoRAInference.loadAdapter(
                into: container,
                from: URL(fileURLWithPath: adapterPath)
            )

            let dataURL = URL(fileURLWithPath: data)
            let testData = try await container.perform { context -> [String] in
                let tok = context.tokenizer
                let formatter: ([[String: String]]) throws -> String = { messages in
                    let ids = try tok.applyChatTemplate(messages: messages)
                    return tok.decode(tokenIds: ids)
                }
                return try loadGemma4TrainingData(directory: dataURL, name: "test", chatFormatter: formatter)
            }
            print("Test: \(testData.count) samples")

            print("Evaluation...")
            let loss = try await Gemma4LoRATrain.evaluate(
                container: container,
                testData: testData,
                batchSize: batchSize
            )

            print("Test loss: \(String(format: "%.4f", loss))")
            print("Test perplexite: \(String(format: "%.4f", exp(loss)))")
        }
    }
}

// MARK: - Fuse

extension LoRA {
    struct Fuse: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Fusionne un adapter LoRA dans le modele de base"
        )

        @Option(name: .long, help: "Chemin local vers le modele de base")
        var modelPath: String

        @Option(name: .long, help: "Chemin vers le repertoire de l'adapter")
        var adapterPath: String

        @Option(name: .long, help: "Repertoire de sortie pour le modele fuse")
        var output: String

        func run() async throws {
            print("Chargement du modele: \(modelPath)")
            let container = try await loadLocalModel(path: modelPath)

            print("Fusion de l'adapter: \(adapterPath)")
            try await Gemma4LoRAInference.fuseAdapter(
                into: container,
                from: URL(fileURLWithPath: adapterPath)
            )

            // Sauvegarder les poids fuses
            let outputURL = URL(fileURLWithPath: output)
            try FileManager.default.createDirectory(at: outputURL, withIntermediateDirectories: true)

            print("Sauvegarde du modele fuse dans \(output)...")
            try await container.perform { context in
                let weights = context.model.parameters()
                let flatWeights = Dictionary(uniqueKeysWithValues: weights.flattened())
                try save(arrays: flatWeights, url: outputURL.appending(component: "model.safetensors"))
            }

            // Copier les fichiers de config du modele original
            let sourceURL = URL(fileURLWithPath: modelPath)
            let configFiles = ["config.json", "tokenizer.json", "tokenizer_config.json",
                             "special_tokens_map.json", "generation_config.json"]
            for file in configFiles {
                let src = sourceURL.appending(component: file)
                let dst = outputURL.appending(component: file)
                if FileManager.default.fileExists(atPath: src.path()) {
                    try? FileManager.default.copyItem(at: src, to: dst)
                }
            }

            print("Modele fuse sauvegarde dans \(output)")
        }
    }
}

// MARK: - Generate (avec adapter)

extension LoRA {
    struct LoRAGenerate: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "generate",
            abstract: "Genere une reponse avec un modele + adapter LoRA"
        )

        @Option(name: .long, help: "Chemin local vers le modele de base")
        var modelPath: String

        @Option(name: .long, help: "Chemin vers le repertoire de l'adapter")
        var adapterPath: String

        @Option(name: .long, help: "Prompt systeme")
        var system: String = "Tu es un assistant utile."

        @Option(name: .long, help: "Temperature")
        var temperature: Float = 0.3

        @Option(name: .long, help: "Max tokens")
        var maxTokens: Int = 512

        @Flag(name: .long, help: "Mode raw: envoie le prompt sans chat template (pour classifieurs)")
        var raw: Bool = false

        @Argument(help: "Le prompt utilisateur")
        var prompt: String

        func run() async throws {
            print("Chargement du modele: \(modelPath)")
            let container = try await loadLocalModel(path: modelPath)

            print("Chargement de l'adapter: \(adapterPath)")
            try await Gemma4LoRAInference.loadAdapter(
                into: container,
                from: URL(fileURLWithPath: adapterPath)
            )
            print("Adapter charge.")

            let capturedPrompt = prompt
            let capturedSystem = system
            let capturedTemp = temperature
            let capturedMaxTokens = maxTokens
            let capturedRaw = raw
            print("\nGenerating...\n")
            let startTime = Date()

            let (text, tokenCount) = try await container.perform { context in
                let tokenizer = context.tokenizer
                let model = context.model

                // Tokeniser le prompt
                let tokenIds: [Int]
                if capturedRaw {
                    // Mode raw: encode le texte directement, sans chat template
                    tokenIds = tokenizer.encode(text: capturedPrompt)
                } else {
                    // Mode normal: applique le chat template
                    var messages: [[String: String]] = []
                    if !capturedSystem.isEmpty {
                        messages.append(["role": "system", "content": capturedSystem])
                    }
                    messages.append(["role": "user", "content": capturedPrompt])
                    tokenIds = try tokenizer.applyChatTemplate(messages: messages)
                }
                let inputIds = MLXArray(tokenIds.map { Int32($0) })

                // Prefill
                let cache = model.newCache(parameters: nil)
                let prefillOutput = model(inputIds.reshaped(1, -1), cache: cache)
                var nextToken = argMax(prefillOutput[0..., prefillOutput.dim(1) - 1, 0...], axis: -1).item(Int32.self)

                var generated: [Int] = []
                for _ in 0 ..< capturedMaxTokens {
                    generated.append(Int(nextToken))
                    if nextToken == 1 || nextToken == 106 || nextToken == 50 { break }

                    let nextInput = MLXArray([nextToken]).reshaped(1, 1)
                    let output = model(nextInput, cache: cache)
                    if capturedTemp <= 0.01 {
                        nextToken = argMax(output[0..., 0, 0...], axis: -1).item(Int32.self)
                    } else {
                        let logits = output[0..., 0, 0...] / capturedTemp
                        let probs = softmax(logits, axis: -1)
                        nextToken = MLXRandom.categorical(log(probs)).item(Int32.self)
                    }
                }

                let text = tokenizer.decode(tokenIds: generated)
                return (text, generated.count)
            }

            print(text)
            let elapsed = Date().timeIntervalSince(startTime)
            print("\n--- Stats ---")
            print("Tokens: \(tokenCount), Temps: \(String(format: "%.2f", elapsed))s")
            print("Vitesse: \(String(format: "%.1f", Double(tokenCount) / max(0.01, elapsed))) t/s")
            print("GPU pic: \(MLX.GPU.peakMemory / (1024 * 1024)) Mo")
        }
    }
}
