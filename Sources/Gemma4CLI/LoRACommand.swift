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
        subcommands: [Train.self, Profiles.self, Eval.self, Fuse.self, LoRAGenerate.self, BenchMultimodal.self]
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

        @Option(name: .long, help: "Lots de validation au plus (25 par defaut comme mlx-lm ; 0 = tout le jeu)")
        var valBatches: Int = 25

        @Option(name: .long, help: "Fichier JSONL des mesures (une ligne par rapport et validation)")
        var metricsOut: String?

        @Flag(name: .long, help: "Calculer la tete sur toutes les positions (par defaut, avec --mask-prompt : reponse seule, perte identique, +13 % de debit)")
        var fullHead: Bool = false

        @Option(name: .long, help: "Limite du cache MLX en Mo pendant l'entrainement, vidage apres validation (0 = aucune ; defaut 2048 : empreinte 76 -> 16 Go sur director)")
        var trainCacheLimitMb: Int = 2048

        @Flag(name: .long, help: "Multimodal : tout le modele en fp32 (ancien chemin ; defaut : base bf16, LoRA fp32)")
        var fp32Model: Bool = false

        @Flag(name: .long, help: "Gradient checkpointing par couche : pic reduit, pas plus lent (K-32)")
        var gradCheckpoint: Bool = false

        @Flag(name: .long, help: "Reprendre au dernier checkpoint du dossier de sortie (poids, optimiseur, pas)")
        var resume: Bool = false

        @Option(name: .long, help: "Graine : init LoRA, dropout et melange des exemples (reproductible)")
        var seed: UInt64 = 0

        @Option(name: .long, help: "Longueur maximale d'un exemple en jetons, troncature comptee (0 = aucune, defaut : la fin des reponses longues serait perdue)")
        var maxSeqLength: Int = 0

        @Flag(name: .long, help: "Activer le profiling (exporte Chrome Trace)")
        var profile: Bool = false

        @Option(name: .long, help: "Profil d'entrainement (ex. lora-16bit-fast ; voir `lora profiles`) : fixe rang, echelle, couches, lr, batch, checkpointing, cache MLX et lots de validation")
        var reference: String?

        @Flag(name: .long, help: "Avec --reference : accepter un profil candidat pas encore mesure (campagne K-33)")
        var allowUnmeasured: Bool = false

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
            print("Modele charge. GPU: \(MLX.Memory.activeMemory / (1024 * 1024)) Mo")

            // 2. Detecter la famille de modele
            guard let family = Gemma4LoRADefaults.ModelFamily.from(directory: URL(fileURLWithPath: modelPath)) else {
                throw ValidationError("famille non reconnue dans \(modelPath)/config.json (E2B, E4B, 12B, 26B-A4B ou 31B)")
            }
            print("Famille detectee: \(family.rawValue) (\(family.totalLayers) couches)")

            // 3. Charger et pre-tokeniser les donnees
            // IMPORTANT: on tokenise DIRECTEMENT via applyChatTemplate sans roundtrip
            // decode→encode qui corrompt les tokens speciaux dans swift-transformers
            let dataURL = URL(fileURLWithPath: data)
            print("Chargement des donnees depuis \(data)...")
            let (trainTokens, validTokens) = try await container.perform {
                (context: ModelContext) -> ([[Int]], [[Int]]) in
                let tok = context.tokenizer

                let train = try tokenizeTrainingFile(dataURL.appending(component: "train.jsonl"), tokenizer: tok)
                let valid = try tokenizeTrainingFile(dataURL.appending(component: "valid.jsonl"), tokenizer: tok)
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
            guard let ftType = Gemma4LoRATrain.FineTuneType(rawValue: fineTuneType) else {
                throw ValidationError("--fine-tune-type inconnu : \(fineTuneType) (lora, dora ou full)")
            }
            var config = Gemma4LoRATrain.TrainingConfig(
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
                enableProfiling: profile,
                seed: seed,
                maxSeqLength: maxSeqLength > 0 ? maxSeqLength : nil,
                resume: resume,
                validationBatches: valBatches > 0 ? valBatches : nil,
                metricsURL: metricsOut.map { URL(fileURLWithPath: $0) },
                responseOnlyHead: !fullHead,
                memoryPolicy: trainCacheLimitMb > 0 ? Gemma4TrainingMemoryPolicy(cacheLimitMB: trainCacheLimitMb) : nil,
                gradientCheckpointing: gradCheckpoint
            )
            try applyReference(to: &config, family: family)

            print("\n--- Debut du training ---")
            let masking = config.maskPrompt ? " + response masking" : ""
            let clipInfo = config.gradClipMaxNorm > 0 ? ", grad_clip: \(config.gradClipMaxNorm)" : ""
            print("Mode: \(config.fineTuneType.rawValue)\(masking), Rank: \(config.loraRank), Scale: \(config.loraScale), LR: \(config.learningRate)\(clipInfo)")
            print("Batch: \(config.batchSize), Iterations: \(iterations)")
            print("Couches: \(config.numLayers ?? family.defaultNumLayers)")
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
            print("GPU pic: \(MLX.Memory.peakMemory / (1024 * 1024)) Mo")
        }

        /// `--reference` : pose le profil d'entrainement ; ses reglages l'emportent sur les
        /// options correspondantes, et c'est affiche.
        func applyReference(
            to config: inout Gemma4LoRATrain.TrainingConfig, family: Gemma4LoRADefaults.ModelFamily
        ) throws {
            guard let reference else { return }
            guard let profile = Gemma4TrainingProfile.candidates.first(where: {
                $0.qualifiedID == reference || ($0.id == reference && $0.loraFamily == family)
            }) else {
                let ids = Gemma4TrainingProfile.candidates.filter { $0.loraFamily == family }.map(\.id)
                throw ValidationError("profil d'entrainement inconnu pour \(family.rawValue) : \(reference) (connus : \(ids.joined(separator: ", ")))")
            }
            guard profile.loraFamily == family else {
                throw ValidationError("\(profile.qualifiedID) vise \(profile.loraFamily.rawValue), le modele est \(family.rawValue)")
            }
            if profile.measurement == nil && !allowUnmeasured {
                throw ValidationError("\(profile.qualifiedID) n'est pas encore mesure : --allow-unmeasured pour l'utiliser quand meme")
            }
            if config.fineTuneType != .lora {
                throw ValidationError("--reference est un profil LoRA : incompatible avec --fine-tune-type \(config.fineTuneType.rawValue)")
            }
            let bits = Self.baseBits(of: URL(fileURLWithPath: modelPath))
            if bits != Int(profile.bits.rawValue) {
                print("Attention : \(profile.qualifiedID) est mesure sur une base \(profile.bits.rawValue) bits, ce modele est en \(bits) bits.")
            }
            profile.apply(to: &config)
            print("Profil \(profile.qualifiedID)\(profile.measurement == nil ? " (NON MESURE)" : "") : \(profile.summary)")
            print("  rang, echelle, couches, lr, batch, checkpointing, cache MLX et lots de validation fixes par le profil.")
        }

        /// Largeur des poids de base lue dans `config.json` (16 si non quantifie).
        static func baseBits(of directory: URL) -> Int {
            guard let data = try? Data(contentsOf: directory.appendingPathComponent("config.json")),
                  let config = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let quantization = (config["quantization"] ?? config["quantization_config"]) as? [String: Any],
                  let bits = quantization["bits"] as? Int
            else { return 16 }
            return bits
        }

        // MARK: - Multimodal training

        func runMultimodal() async throws {
            print("Chargement du modele multimodal: \(modelPath)")
            let container = try await loadLocalMultimodalModel(path: modelPath)
            print("Modele multimodal charge. GPU: \(MLX.Memory.activeMemory / (1024 * 1024)) Mo")

            guard let family = Gemma4LoRADefaults.ModelFamily.from(directory: URL(fileURLWithPath: modelPath)) else {
                throw ValidationError("famille non reconnue dans \(modelPath)/config.json (E2B, E4B, 12B, 26B-A4B ou 31B)")
            }
            print("Famille detectee: \(family.rawValue) (\(family.totalLayers) couches)")

            // Charger les donnees multimodales
            let dataURL = URL(fileURLWithPath: data)
            print("Chargement des donnees multimodales depuis \(data)...")

            // Phase 1: Tokeniser les textes (besoin du tokenizer)
            let (trainTexts, validTexts) = try await container.perform {
                (context: ModelContext) -> ([MultimodalTrainingSample], [MultimodalTrainingSample]) in
                let tok = context.tokenizer
                let formatter: ([[String: String]]) throws -> String = { messages in
                    let ids = try tok.applyChatTemplate(messages: messages)
                    return tok.decode(tokenIds: Gemma4Processor.strippingTemplateArtifacts(
                        Gemma4Processor.droppingGenerationPrompt(ids)))
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
            guard let ftType = Gemma4LoRATrain.FineTuneType(rawValue: fineTuneType) else {
                throw ValidationError("--fine-tune-type inconnu : \(fineTuneType) (lora, dora ou full)")
            }
            var config = Gemma4LoRATrain.TrainingConfig(
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
                enableProfiling: profile,
                seed: seed,
                resume: resume,
                validationBatches: valBatches > 0 ? valBatches : nil,
                metricsURL: metricsOut.map { URL(fileURLWithPath: $0) },
                responseOnlyHead: !fullHead,
                memoryPolicy: trainCacheLimitMb > 0 ? Gemma4TrainingMemoryPolicy(cacheLimitMB: trainCacheLimitMb) : nil,
                multimodalFloat32: fp32Model,
                gradientCheckpointing: gradCheckpoint
            )
            try applyReference(to: &config, family: family)
            config.batchSize = 1

            print("\n--- Debut du training multimodal ---")
            let masking = config.maskPrompt ? " + response masking" : ""
            print("Mode: \(config.fineTuneType.rawValue)\(masking), Rank: \(config.loraRank), Scale: \(config.loraScale), LR: \(config.learningRate)")
            print("Batch: 1 (multimodal), Iterations: \(iterations)")
            print("Couches: \(config.numLayers ?? family.defaultNumLayers)")
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
            print("GPU pic: \(MLX.Memory.peakMemory / (1024 * 1024)) Mo")
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

                // A-03 / A-04 : ids directs au format de l'inference (marqueurs dans le tour
                // user, joints par \n), sans aller-retour decode -> encode.
                guard let messages = sample.messages else {
                    throw ValidationError("exemple multimodal sans messages")
                }
                let hasImage = pixelValues != nil
                let audioTokens = audioFeatures?.numTokens
                let tokens: [Int] = try await container.perform { (context: ModelContext) -> [Int] in
                    try Gemma4Processor.multimodalTrainingIds(
                        messages: messages, hasImage: hasImage, audioTokens: audioTokens,
                        tokenizer: context.tokenizer)
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
    /// `lora profiles` : matrice des profils d'entrainement, mesures ou non.
    struct Profiles: ParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Liste les profils d'entrainement (K-33) et leurs mesures")

        func run() throws {
            for profile in Gemma4TrainingProfile.candidates {
                let checkpointing = profile.gradientCheckpointing ? "oui" : "non"
                let cache = profile.memoryPolicy.cacheLimitMB.map { "\($0) Mo" } ?? "-"
                print("\(profile.qualifiedID)  base \(profile.model.rawValue)")
                print("  r\(profile.rank) s\(Int(profile.scale)), \(profile.numLayers) couches, lr \(profile.learningRate), batch \(profile.batchSize), checkpointing \(checkpointing), cache \(cache)")
                if let m = profile.measurement {
                    let e7 = m.e7Valid.map { ", E7 \($0)/30" } ?? ""
                    print(String(format: "  mesure %@ (%d pas) : pic MLX %.1f Go, empreinte %.1f Go, %.0f tok/s, val %.3f", m.date, m.steps, m.peakMLXGB, m.footprintGB, m.trainedTokensPerSecond, m.validationLoss) + e7)
                } else {
                    print("  NON MESURE (non publie)")
                }
            }
        }
    }

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

        @Flag(name: .long, help: "Perte sur la reponse seulement (comme train --mask-prompt) : comparable a la val loss")
        var maskPrompt: Bool = false

        func run() async throws {
            print("Chargement du modele: \(modelPath)")
            let container = try await loadLocalModel(path: modelPath)

            print("Chargement de l'adapter: \(adapterPath)")
            try await Gemma4LoRAInference.loadAdapter(
                into: container,
                from: URL(fileURLWithPath: adapterPath)
            )

            let dataURL = URL(fileURLWithPath: data)
            let testTokens = try await container.perform { context -> [[Int]] in
                try tokenizeTrainingFile(dataURL.appending(component: "test.jsonl"), tokenizer: context.tokenizer)
            }
            print("Test: \(testTokens.count) samples")

            print("Evaluation...")
            let loss = try await Gemma4LoRATrain.evaluateMasked(
                container: container, samples: testTokens, maskPrompt: maskPrompt, batchSize: batchSize)

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
            print("Fusion de \(adapterPath) dans \(modelPath)…")
            let files = try await Gemma4LoRAInference.fuseAndSave(
                baseDirectory: URL(fileURLWithPath: modelPath),
                adapterDirectory: URL(fileURLWithPath: adapterPath),
                output: URL(fileURLWithPath: output))
            print("Modele fuse sauvegarde dans \(output) (\(files.count) fichier(s) de poids)")
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
                    tokenIds = Gemma4Processor.strippingTemplateArtifacts(ids)
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
            print("GPU pic: \(MLX.Memory.peakMemory / (1024 * 1024)) Mo")
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
                    tokenIds = Gemma4Processor.strippingTemplateArtifacts(tokenIds)

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

                let resultEntry: [String: Any] = [
                    "expected": expected,
                    "predicted": predicted,
                    "correct": isCorrect,
                ]
                if let jsonData = try? JSONSerialization.data(withJSONObject: resultEntry),
                   let jsonStr = String(data: jsonData, encoding: .utf8) {
                    resultsFile += jsonStr + "\n"
                }
            }

            // Sauvegarder les resultats
            let outputURL = URL(fileURLWithPath: output)
            try resultsFile.write(to: outputURL, atomically: true, encoding: .utf8)

            print("\n=== Resultats ===")
            print("Accuracy: \(correct)/\(total) (\(String(format: "%.1f", Double(correct) / Double(total) * 100))%)")
            print("Resultats sauvegardes dans \(output)")
            print("GPU pic: \(MLX.Memory.peakMemory / (1024 * 1024)) Mo")
        }
    }
}

/// Ids directs d'un JSONL d'entrainement (`messages` ou `text`) : gabarit sans suffixe de
/// generation ni artefacts, rejets comptes et signales (A-06). Partage par train et eval.
func tokenizeTrainingFile(_ url: URL, tokenizer tok: any MLXLMCommon.Tokenizer) throws -> [[Int]] {
    let name = url.deletingPathExtension().lastPathComponent
    let lines = try String(contentsOf: url, encoding: .utf8)
        .components(separatedBy: .newlines)

    struct ChatMsg: Codable {
        let messages: [ChatMessage]?
        let text: String?
    }

    // A-06 : chaque rejet est compte et signale (numero de ligne + raison).
    var rejected: [String] = []
    defer {
        if !rejected.isEmpty {
            print("\(name).jsonl : \(rejected.count) ligne(s) rejetee(s)")
            for reason in rejected.prefix(10) { print("  - \(reason)") }
            if rejected.count > 10 { print("  … +\(rejected.count - 10)") }
        }
    }
    return try lines.enumerated().compactMap { index, raw -> [Int]? in
        let line = raw.trimmingCharacters(in: .whitespaces)
        guard !line.isEmpty else { return nil }
        let sample: ChatMsg
        do {
            sample = try JSONDecoder().decode(ChatMsg.self, from: Data(line.utf8))
        } catch {
            rejected.append("ligne \(index + 1) : JSON invalide ou champs inattendus")
            return nil
        }

        if let msgs = sample.messages, !msgs.isEmpty {
            // Chat format: tokeniser DIRECTEMENT via applyChatTemplate
            let msgDicts = msgs.map { ["role": $0.role, "content": $0.content] }
            let ids = try tok.applyChatTemplate(messages: msgDicts)
            // Invite de generation retiree quelle que soit sa longueur (K-33 : 12B/26B/31B
            // ajoutent un canal de pensee vide apres `<|turn>model\n`).
            return Gemma4Processor.strippingTemplateArtifacts(Gemma4Processor.droppingGenerationPrompt(ids))
        } else if let text = sample.text {
            return tok.encode(text: text)
        }
        rejected.append("ligne \(index + 1) : ni `messages` ni `text`")
        return nil
    }
}
