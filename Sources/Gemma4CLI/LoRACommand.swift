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
        subcommands: [Train.self, Eval.self, Fuse.self, LoRAGenerate.self, BenchMultimodal.self]
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

        @Option(name: .long, help: "Type: lora (defaut), dora, ou full (tous les poids)")
        var fineTuneType: String = "lora"

        @Flag(name: .long, help: "Response masking: loss uniquement sur la reponse, pas le prompt")
        var maskPrompt: Bool = false

        @Option(name: .long, help: "Gradient clipping max norm (0=desactive, papier recommande 0.3 pour full)")
        var gradClip: Float = 0

        @Flag(name: .long, help: "Activer le profiling (exporte Chrome Trace)")
        var profile: Bool = false

        @Flag(name: .long, help: "Mode multimodal: charge le modele complet (vision+audio) et traite les champs image/audio du JSONL")
        var multimodal: Bool = false

        func run() async throws {
            if multimodal {
                try await runMultimodal()
                return
            }

            // 1. Enregistrer et charger le modele
            print("Chargement du modele: \(modelPath)")
            let container = try await loadLocalModel(path: modelPath)
            print("Modele charge. GPU: \(MLX.GPU.activeMemory / (1024 * 1024)) Mo")

            // 2. Detecter la famille de modele
            let family = Gemma4LoRADefaults.ModelFamily.from(modelId: modelPath)
            print("Famille detectee: \(family.rawValue) (\(family.totalLayers) couches)")

            // 3. Charger et pre-tokeniser les donnees
            // IMPORTANT: on tokenise DIRECTEMENT via applyChatTemplate sans roundtrip
            // decode→encode qui corrompt les tokens speciaux dans swift-transformers
            let dataURL = URL(fileURLWithPath: data)
            print("Chargement des donnees depuis \(data)...")
            let (trainTokens, validTokens) = try await container.perform {
                (context: ModelContext) -> ([[Int]], [[Int]]) in
                let tok = context.tokenizer

                func tokenizeFile(name: String) throws -> [[Int]] {
                    let url = dataURL.appending(component: "\(name).jsonl")
                    let lines = try String(contentsOf: url, encoding: .utf8)
                        .components(separatedBy: .newlines)
                        .filter { $0.first == "{" }

                    struct ChatMsg: Codable {
                        let messages: [ChatMessage]?
                        let text: String?
                    }

                    return try lines.compactMap { line -> [Int]? in
                        guard let data = line.data(using: .utf8) else { return nil }
                        let sample = try JSONDecoder().decode(ChatMsg.self, from: data)

                        if let msgs = sample.messages, !msgs.isEmpty {
                            // Chat format: tokeniser DIRECTEMENT via applyChatTemplate
                            let msgDicts = msgs.map { ["role": $0.role, "content": $0.content] }
                            var ids = try tok.applyChatTemplate(messages: msgDicts)
                            // Retirer les 3 derniers tokens (add_generation_prompt: <|turn>model\n)
                            if ids.count >= 3 {
                                let last3 = Array(ids.suffix(3))
                                if last3 == [105, 4368, 107] { // <|turn> model \n
                                    ids = Array(ids.dropLast(3))
                                }
                            }
                            // Fix swift-jinja: retire le \n parasite entre <bos> et <|turn>
                            // Python produit [2, 105, ...] mais Swift produit [2, 107, 105, ...]
                            if ids.count >= 3 && ids[0] == 2 && ids[1] == 107 && ids[2] == 105 {
                                ids.remove(at: 1)
                            }
                            return ids
                        } else if let text = sample.text {
                            return tok.encode(text: text)
                        }
                        return nil
                    }
                }

                let train = try tokenizeFile(name: "train")
                let valid = try tokenizeFile(name: "valid")
                return (train, valid)
            }

            // Convertir en format text pour compatibilite (le training loop retokenise)
            // NON: on passe directement les tokens au training loop!
            let trainData = trainTokens
            let validData = validTokens
            print("Train: \(trainData.count) samples, Valid: \(validData.count) samples")

            // 4. Configurer le profiling
            if profile {
                let profiler = MLXProfiler.shared
                profiler.enable()
                profiler.activeSession = ProfilingSession(config: .detailed)
            }

            // 5. Configurer et lancer le training
            let ftType = Gemma4LoRATrain.FineTuneType(rawValue: fineTuneType) ?? .lora
            let config = Gemma4LoRATrain.TrainingConfig(
                fineTuneType: ftType,
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
                maskPrompt: ftType == .full ? true : maskPrompt,  // Full SFT utilise toujours le masking
                gradClipMaxNorm: ftType == .full && gradClip == 0 ? 0.3 : gradClip,  // Default 0.3 pour full
                enableProfiling: profile
            )

            print("\n--- Debut du training ---")
            let masking = config.maskPrompt ? " + response masking" : ""
            let clipInfo = config.gradClipMaxNorm > 0 ? ", grad_clip: \(config.gradClipMaxNorm)" : ""
            print("Mode: \(ftType.rawValue)\(masking), Rank: \(rank), Scale: \(scale), LR: \(learningRate)\(clipInfo)")
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

        // MARK: - Multimodal training

        func runMultimodal() async throws {
            print("Chargement du modele multimodal: \(modelPath)")
            let container = try await loadLocalMultimodalModel(path: modelPath)
            print("Modele multimodal charge. GPU: \(MLX.GPU.activeMemory / (1024 * 1024)) Mo")

            let family = Gemma4LoRADefaults.ModelFamily.from(modelId: modelPath)
            print("Famille detectee: \(family.rawValue) (\(family.totalLayers) couches)")

            // Charger les donnees multimodales
            let dataURL = URL(fileURLWithPath: data)
            print("Chargement des donnees multimodales depuis \(data)...")

            // Phase 1: Tokeniser les textes (besoin du tokenizer)
            let (trainTexts, validTexts) = try await container.perform {
                (context: ModelContext) -> ([MultimodalTrainingSample], [MultimodalTrainingSample]) in
                let tok = context.tokenizer
                let formatter: ([[String: String]]) throws -> String = { messages in
                    var ids = try tok.applyChatTemplate(messages: messages)
                    // Retirer add_generation_prompt suffix
                    if ids.count >= 3 {
                        let last3 = Array(ids.suffix(3))
                        if last3 == [105, 4368, 107] {
                            ids = Array(ids.dropLast(3))
                        }
                    }
                    // Fix swift-jinja parasitic \n
                    if ids.count >= 3 && ids[0] == 2 && ids[1] == 107 && ids[2] == 105 {
                        ids.remove(at: 1)
                    }
                    return tok.decode(tokenIds: ids)
                }

                let train = try loadGemma4MultimodalJSONL(
                    url: dataURL.appending(component: "train.jsonl"),
                    dataDirectory: dataURL,
                    chatFormatter: formatter
                )
                let valid = try loadGemma4MultimodalJSONL(
                    url: dataURL.appending(component: "valid.jsonl"),
                    dataDirectory: dataURL,
                    chatFormatter: formatter
                )
                return (train, valid)
            }

            print("Train: \(trainTexts.count) samples, Valid: \(validTexts.count) samples")

            // Phase 2: Pre-tokeniser et pre-traiter les media (hors container)
            print("Pre-traitement des media...")
            let trainSamples = try await preprocessMultimodalSamples(trainTexts, container: container)
            let validSamples = try await preprocessMultimodalSamples(validTexts, container: container)

            // Phase 3: Lancer le training
            let ftType = Gemma4LoRATrain.FineTuneType(rawValue: fineTuneType) ?? .lora
            let config = Gemma4LoRATrain.TrainingConfig(
                fineTuneType: ftType,
                loraRank: rank,
                loraScale: scale,
                numLayers: numLayers,
                modelFamily: family,
                learningRate: learningRate,
                batchSize: 1,  // Multimodal = batch 1 obligatoire
                iterations: iterations,
                stepsPerReport: stepsPerReport,
                stepsPerEval: stepsPerEval,
                saveEvery: 50,
                outputDirectory: URL(fileURLWithPath: output),
                maskPrompt: ftType == .full ? true : maskPrompt,
                gradClipMaxNorm: ftType == .full && gradClip == 0 ? 0.3 : gradClip,
                enableProfiling: profile
            )

            print("\n--- Debut du training multimodal ---")
            let masking = config.maskPrompt ? " + response masking" : ""
            print("Mode: \(ftType.rawValue)\(masking), Rank: \(rank), Scale: \(scale), LR: \(learningRate)")
            print("Batch: 1 (multimodal), Iterations: \(iterations)")
            print("Couches: \(numLayers ?? family.defaultNumLayers)")
            print("Sortie: \(output)")
            print("---\n")

            try await Gemma4LoRATrain.trainMultimodal(
                container: container,
                trainData: trainSamples,
                validData: validSamples,
                config: config
            ) { progress in
                print(progress)
                return .more
            }

            print("\nTraining multimodal termine.")
            print("GPU pic: \(MLX.GPU.peakMemory / (1024 * 1024)) Mo")
        }

        /// Pre-traite les samples multimodaux: tokenise le texte, expanse les placeholders,
        /// et charge les features audio/image
        func preprocessMultimodalSamples(
            _ samples: [MultimodalTrainingSample],
            container: ModelContainer
        ) async throws -> [MultimodalTokenizedSample] {
            var results: [MultimodalTokenizedSample] = []

            for (i, sample) in samples.enumerated() {
                if (i + 1) % 100 == 0 || i == 0 {
                    print("  Preprocessing \(i + 1)/\(samples.count)...")
                }

                // Traiter l'audio si present
                var audioFeatures: Gemma4AudioProcessor.AudioFeatures? = nil
                if let audioPath = sample.audioPath {
                    audioFeatures = try await Gemma4AudioProcessor.processAudio(
                        url: URL(fileURLWithPath: audioPath)
                    )
                }

                // Traiter l'image si presente
                var pixelValues: MLXArray? = nil
                if let imagePath = sample.imagePath {
                    pixelValues = try Gemma4ImageProcessor.processImage(
                        url: URL(fileURLWithPath: imagePath)
                    )
                }

                // Tokeniser le texte (sans placeholders — applyChatTemplate les escape)
                let sampleText = sample.text
                var tokens: [Int] = try await container.perform { (context: ModelContext) -> [Int] in
                    var ids = context.tokenizer.encode(text: sampleText)
                    // Fix swift-jinja: retirer \n parasite entre <bos> et <|turn>
                    if ids.count >= 3 && ids[0] == 2 && ids[1] == 107 && ids[2] == 105 {
                        ids.remove(at: 1)
                    }
                    return ids
                }

                // Trouver le point d'injection: juste apres <start_of_turn>user\n
                // Token IDs: 105=<start_of_turn>, 2364=user, 107=\n
                // On cherche la PREMIERE occurrence (le user prompt)
                var insertionIdx: Int? = nil
                for j in 0 ..< tokens.count - 2 {
                    if tokens[j] == 105 && tokens[j + 1] == 2364 && tokens[j + 2] == 107 {
                        insertionIdx = j + 3  // juste apres user\n
                        break
                    }
                }

                // Injecter les tokens image (boi + image_token*280 + eoi)
                if pixelValues != nil, let idx = insertionIdx {
                    let imgId = Int(Gemma4Processor.imageTokenId)
                    let boiId = Int(Gemma4Processor.boiTokenId)
                    let eoiId = Int(Gemma4Processor.eoiTokenId)
                    var mediaTokens = [boiId]
                    mediaTokens.append(contentsOf: Array(repeating: imgId, count: 280))
                    mediaTokens.append(eoiId)
                    tokens.insert(contentsOf: mediaTokens, at: idx)
                    insertionIdx = idx + mediaTokens.count  // avancer le point d'insertion
                }

                // Injecter les tokens audio (boa + audio_token*N + eoa)
                if let af = audioFeatures, let idx = insertionIdx {
                    let audId = Int(Gemma4Processor.audioTokenId)
                    let boaId = Int(Gemma4Processor.boaTokenId)
                    let eoaId = Int(Gemma4Processor.eoaTokenId)
                    var mediaTokens = [boaId]
                    mediaTokens.append(contentsOf: Array(repeating: audId, count: af.numTokens))
                    mediaTokens.append(eoaId)
                    tokens.insert(contentsOf: mediaTokens, at: idx)
                }

                // Calculer le prompt offset (apres expansion)
                var promptOffset = 0
                if maskPrompt {
                    for i in 0 ..< tokens.count - 1 {
                        if tokens[i] == 105 && tokens[i + 1] == 4368 {
                            promptOffset = i + 3  // <|turn> + model + \n
                        }
                    }
                }

                results.append(MultimodalTokenizedSample(
                    tokens: tokens,
                    promptOffset: promptOffset,
                    pixelValues: pixelValues,
                    audioFeatures: audioFeatures?.features,
                    audioMask: audioFeatures?.mask
                ))
            }

            return results
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
                let genPromptSuffix = "<|turn>model\n"
                let formatter: ([[String: String]]) throws -> String = { messages in
                    let ids = try tok.applyChatTemplate(messages: messages)
                    var text = tok.decode(tokenIds: ids)
                    if text.hasSuffix(genPromptSuffix) {
                        text = String(text.dropLast(genPromptSuffix.count))
                    }
                    return text
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
                    var ids = try tokenizer.applyChatTemplate(messages: messages)
                    // Fix swift-jinja: retire le \n parasite entre <bos> et <|turn>
                    if ids.count >= 3 && ids[0] == 2 && ids[1] == 107 && ids[2] == 105 {
                        ids.remove(at: 1)
                    }
                    tokenIds = ids
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

// MARK: - Bench Multimodal

extension LoRA {
    struct BenchMultimodal: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "bench-multimodal",
            abstract: "Benchmark fonctionnel: inference multimodale sur un dataset de validation"
        )

        @Option(name: .long, help: "Chemin local vers le modele de base")
        var modelPath: String

        @Option(name: .long, help: "Chemin vers le repertoire de l'adapter")
        var adapterPath: String

        @Option(name: .long, help: "Repertoire contenant valid.jsonl avec champs audio/image")
        var data: String

        @Option(name: .long, help: "Fichier de sortie JSONL pour les resultats")
        var output: String = "/tmp/birdcall-bench-results.jsonl"

        @Option(name: .long, help: "Temperature (0 = greedy)")
        var temperature: Float = 0.0

        @Option(name: .long, help: "Max tokens a generer par sample")
        var maxTokens: Int = 32

        func run() async throws {
            print("Chargement du modele multimodal: \(modelPath)")
            let container = try await loadLocalMultimodalModel(path: modelPath)

            print("Chargement de l'adapter: \(adapterPath)")
            try await Gemma4LoRAInference.loadAdapter(
                into: container,
                from: URL(fileURLWithPath: adapterPath)
            )
            print("Adapter charge.")

            // Charger les samples de validation
            let dataURL = URL(fileURLWithPath: data)
            let validURL = dataURL.appending(component: "valid.jsonl")
            let lines = try String(contentsOf: validURL, encoding: .utf8)
                .components(separatedBy: .newlines)
                .filter { $0.first == "{" }

            struct Sample: Codable {
                let messages: [ChatMessage]
                let audio: String?
                let image: String?
            }

            let decoder = JSONDecoder()
            let samples = try lines.compactMap { line -> Sample? in
                guard let data = line.data(using: .utf8) else { return nil }
                return try decoder.decode(Sample.self, from: data)
            }

            print("Samples de validation: \(samples.count)")

            var correct = 0
            var total = 0
            var resultsFile = try String()

            for (i, sample) in samples.enumerated() {
                let expected = sample.messages.last { $0.role == "assistant" || $0.role == "model" }?.content ?? ""
                let userPrompt = sample.messages.first { $0.role == "user" }?.content ?? ""

                // Preparer l'audio ou l'image
                var audioFeatures: Gemma4AudioProcessor.AudioFeatures? = nil
                if let audioPath = sample.audio {
                    let fullPath = dataURL.appending(component: audioPath)
                    audioFeatures = try await Gemma4AudioProcessor.processAudio(url: fullPath)
                }

                var pixelValues: MLXArray? = nil
                if let imagePath = sample.image {
                    let fullPath = dataURL.appending(component: imagePath)
                    pixelValues = try Gemma4ImageProcessor.processImage(url: fullPath)
                }

                // Capturer les valeurs pour le closure Sendable
                let capturedTemp = temperature
                let capturedMaxTokens = maxTokens
                let numAudioTokens = audioFeatures?.numTokens ?? 0
                let hasAudio = audioFeatures != nil
                let hasImage = pixelValues != nil
                nonisolated(unsafe) let capturedAudioFeatures = audioFeatures?.features
                nonisolated(unsafe) let capturedAudioMask = audioFeatures?.mask
                nonisolated(unsafe) let capturedPixelValues = pixelValues

                let predicted: String = try await container.perform { context in
                    let tokenizer = context.tokenizer
                    let model = context.model

                    // Construire le prompt multimodal
                    let prompt = Gemma4Processor.buildMultimodalPrompt(
                        userPrompt: userPrompt,
                        hasImage: hasImage,
                        hasAudio: hasAudio,
                        numAudioTokens: numAudioTokens
                    )

                    var tokenIds = tokenizer.encode(text: prompt)
                    // Fix swift-jinja
                    if tokenIds.count >= 3 && tokenIds[0] == 2 && tokenIds[1] == 107 && tokenIds[2] == 105 {
                        tokenIds.remove(at: 1)
                    }

                    // Setter les pending media
                    if let mmModel = model as? Gemma4MultimodalLLMModel {
                        mmModel.pendingPixelValues = capturedPixelValues
                        mmModel.pendingAudioFeatures = capturedAudioFeatures
                        mmModel.pendingAudioMask = capturedAudioMask
                    }

                    let inputIds = MLXArray(tokenIds.map { Int32($0) })
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

                    return tokenizer.decode(tokenIds: generated)
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                }

                let isCorrect = predicted.lowercased() == expected.lowercased()
                if isCorrect { correct += 1 }
                total += 1

                let symbol = isCorrect ? "✓" : "✗"
                print("  [\(i+1)/\(samples.count)] \(symbol) expected: \(expected) | predicted: \(predicted)")

                let resultLine = """
                {"expected": "\(expected)", "predicted": "\(predicted)", "species": "\(expected)", "correct": \(isCorrect)}
                """
                resultsFile += resultLine + "\n"
            }

            // Sauvegarder les resultats
            let outputURL = URL(fileURLWithPath: output)
            try resultsFile.write(to: outputURL, atomically: true, encoding: .utf8)

            print("\n=== Resultats ===")
            print("Accuracy: \(correct)/\(total) (\(String(format: "%.1f", Double(correct) / Double(total) * 100))%)")
            print("Resultats sauvegardes dans \(output)")
            print("GPU pic: \(MLX.GPU.peakMemory / (1024 * 1024)) Mo")
        }
    }
}
