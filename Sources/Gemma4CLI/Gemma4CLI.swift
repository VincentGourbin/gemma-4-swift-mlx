// Mini CLI pour tester l'inference Gemma 4 via MLX Swift

import ArgumentParser
import Foundation
import Gemma4Swift
import MLX
import MLXHuggingFace
import MLXLMCommon
import MLXLLM
import Tokenizers
import HuggingFace

// MARK: - Helpers

/// Cree un HubClient configure pour utiliser le cache personnalise et le token HF
func makeHubClient(token: String? = nil) -> HubClient {
    let resolvedToken = token ?? ProcessInfo.processInfo.environment["HF_TOKEN"]
    let cacheDir = Gemma4ModelCache.modelsDirectory
    // Creer le repertoire cache si necessaire
    try? FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)
    let cache = HubCache(cacheDirectory: cacheDir)
    if let token = resolvedToken {
        return HubClient(
            host: HubClient.defaultHost,
            bearerToken: token,
            cache: cache
        )
    }
    return HubClient(cache: cache)
}

/// Affiche un avertissement si le modele risque de depasser la RAM
func warnIfLowRAM(modelId: String) {
    guard let model = Gemma4Pipeline.Model(rawValue: modelId) else { return }
    let ram = Gemma4ModelCache.systemRAMGB
    if model.recommendedRAMGB > ram {
        print("⚠ Attention: \(model.displayName) recommande \(model.recommendedRAMGB) Go de RAM (systeme: \(ram) Go)")
        print("  Le chargement risque d'echouer ou d'etre tres lent.")
    }
}

@main
struct Gemma4CLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "gemma4-cli",
        abstract: "Inference Gemma 4 via MLX Swift",
        subcommands: [Generate.self, Chat.self, Describe.self, Models.self, Download.self, Profile.self],
        defaultSubcommand: Generate.self
    )
}

// MARK: - Models (liste et info)

struct Models: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Liste les modeles Gemma 4 disponibles"
    )

    @Flag(name: .long, help: "Afficher uniquement les modeles recommandes pour cette machine")
    var recommended: Bool = false

    func run() {
        let ram = Gemma4ModelCache.systemRAMGB
        print("RAM systeme: \(ram) Go\n")

        let models: [Gemma4Pipeline.Model]
        if recommended {
            models = Gemma4Pipeline.Model.recommended(forRAMGB: ram)
            print("Modeles recommandes pour \(ram) Go de RAM:\n")
        } else {
            models = Gemma4Pipeline.Model.allCases.sorted { $0.estimatedSizeGB < $1.estimatedSizeGB }
            print("Tous les modeles disponibles:\n")
        }

        if models.isEmpty {
            print("  Aucun modele compatible avec \(ram) Go de RAM.")
            return
        }

        for model in models {
            let downloaded = Gemma4ModelCache.isDownloaded(model)
            let status = downloaded ? " [telecharge]" : ""
            let itBadge = model.isInstructionTuned ? "IT" : "base"
            let moeBadge = model.isMoE ? " MoE" : ""
            let format = model.quantization

            var modalities: [String] = []
            if model.supportsImage { modalities.append("image") }
            if model.supportsAudio { modalities.append("audio") }
            if model.supportsVideo { modalities.append("video") }
            let modalitiesStr = modalities.joined(separator: "+")

            print("  \(model.rawValue)\(status)")
            print("    \(model.displayName) | \(model.parameterCount) params (\(model.effectiveParameters) effectifs) | ~\(Int(model.estimatedSizeGB)) Go | \(format) | \(itBadge)\(moeBadge) | \(modalitiesStr) | RAM min: \(model.recommendedRAMGB) Go")

            if downloaded, let size = Gemma4ModelCache.diskSize(for: model) {
                let sizeGB = String(format: "%.1f", Double(size) / 1_073_741_824)
                print("    Taille sur disque: \(sizeGB) Go")
            }
            print()
        }

        print("Utilisation: gemma4-cli generate --model <ID> \"votre prompt\"")
        print("Token HF: export HF_TOKEN=<votre_token> ou --hf-token <token>")
    }
}

// MARK: - Download

struct Download: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Telecharge un ou plusieurs modeles Gemma 4"
    )

    @Argument(help: "IDs des modeles a telecharger (ex: e2b-4bit e4b-8bit 31b-4bit), ou 'all' pour tout telecharger")
    var modelIds: [String] = []

    @Flag(name: .long, help: "Telecharger tous les modeles disponibles")
    var all: Bool = false

    @Flag(name: .long, help: "Telecharger uniquement les modeles recommandes pour cette machine")
    var recommended: Bool = false

    @Option(name: .long, help: "Token HuggingFace (pour modeles prives)")
    var hfToken: String?

    @Flag(name: .long, help: "Forcer le re-telechargement meme si deja present")
    var force: Bool = false

    /// Mappe les raccourcis vers les IDs complets
    static let shortcuts: [String: String] = [
        // E2B
        "e2b-4bit": "mlx-community/gemma-4-e2b-it-4bit",
        "e2b-6bit": "mlx-community/gemma-4-e2b-it-6bit",
        "e2b-8bit": "mlx-community/gemma-4-e2b-it-8bit",
        "e2b-bf16": "mlx-community/gemma-4-e2b-it-bf16",
        // E4B
        "e4b-4bit": "mlx-community/gemma-4-e4b-it-4bit",
        "e4b-6bit": "mlx-community/gemma-4-e4b-it-6bit",
        "e4b-8bit": "mlx-community/gemma-4-e4b-it-8bit",
        "e4b-bf16": "mlx-community/gemma-4-e4b-it-bf16",
        // 31B
        "31b-4bit": "mlx-community/gemma-4-31b-it-4bit",
        "31b-6bit": "mlx-community/gemma-4-31b-it-6bit",
        "31b-8bit": "mlx-community/gemma-4-31b-it-8bit",
        "31b-bf16": "mlx-community/gemma-4-31b-it-bf16",
        // 26B-A4B
        "a4b-4bit": "mlx-community/gemma-4-26b-a4b-it-4bit",
        "a4b-6bit": "mlx-community/gemma-4-26b-a4b-it-6bit",
        "a4b-8bit": "mlx-community/gemma-4-26b-a4b-it-8bit",
        "a4b-bf16": "mlx-community/gemma-4-26b-a4b-it-bf16",
    ]

    func run() async throws {
        let modelsToDownload: [Gemma4Pipeline.Model]

        if all {
            modelsToDownload = Gemma4Pipeline.Model.allCases.sorted { $0.estimatedSizeGB < $1.estimatedSizeGB }
        } else if recommended {
            let ram = Gemma4ModelCache.systemRAMGB
            modelsToDownload = Gemma4Pipeline.Model.recommended(forRAMGB: ram)
            print("Modeles recommandes pour \(ram) Go de RAM:")
        } else if !modelIds.isEmpty {
            var resolved: [Gemma4Pipeline.Model] = []
            for id in modelIds {
                let fullId = Self.shortcuts[id.lowercased()] ?? id
                if let model = Gemma4Pipeline.Model(rawValue: fullId) {
                    resolved.append(model)
                } else {
                    print("Modele inconnu: \(id)")
                    print("  Raccourcis: \(Self.shortcuts.keys.sorted().joined(separator: ", "))")
                    throw ExitCode.failure
                }
            }
            modelsToDownload = resolved
        } else {
            print("Specifiez des modeles, --all, ou --recommended")
            print("Raccourcis: \(Self.shortcuts.keys.sorted().joined(separator: ", "))")
            print("Exemple: gemma4-cli download e2b-4bit e4b-4bit")
            throw ExitCode.failure
        }

        // Estimation taille totale
        let totalGB = modelsToDownload.reduce(Float(0)) { $0 + $1.estimatedSizeGB }
        let alreadyDownloaded = modelsToDownload.filter { Gemma4ModelCache.isDownloaded($0) }
        let toDownload = force ? modelsToDownload : modelsToDownload.filter { !Gemma4ModelCache.isDownloaded($0) }

        print("\n\(modelsToDownload.count) modeles selectionnes (~\(Int(totalGB)) Go total)")
        if !alreadyDownloaded.isEmpty && !force {
            print("\(alreadyDownloaded.count) deja telecharges (utiliser --force pour re-telecharger)")
        }
        print("\(toDownload.count) a telecharger\n")

        if toDownload.isEmpty {
            print("Rien a telecharger.")
            return
        }

        // Enregistrer les types pour le chargement
        await Gemma4Registration.register(multimodal: true)

        // Telecharger sequentiellement
        for (i, model) in toDownload.enumerated() {
            print("[\(i + 1)/\(toDownload.count)] \(model.displayName) (~\(Int(model.estimatedSizeGB)) Go)")
            print("  ID: \(model.rawValue)")

            let startTime = Date()
            var lastPct = -1

            do {
                let hub = makeHubClient(token: hfToken)
                // Decomposer l'ID en namespace/name
                let parts = model.rawValue.split(separator: "/")
                let repoId = Repo.ID(namespace: String(parts[0]), name: String(parts[1]))
                let destDir = Gemma4ModelCache.modelsDirectory
                    .appendingPathComponent(String(parts[0]))
                    .appendingPathComponent(String(parts[1]))
                try? FileManager.default.createDirectory(at: destDir, withIntermediateDirectories: true)

                let _ = try await hub.downloadSnapshot(
                    of: repoId,
                    to: destDir,
                    matching: ["*.safetensors", "*.json", "*.jinja", "*.txt"]
                ) { progress in
                    let pct = Int(progress.fractionCompleted * 100)
                    if pct != lastPct {
                        lastPct = pct
                        print("\r  Progression: \(pct)%", terminator: "")
                        fflush(stdout)
                    }
                }

                let elapsed = Date().timeIntervalSince(startTime)
                print("\r  Termine en \(String(format: "%.0f", elapsed))s")

                // Nettoyer le cache HF interne (models--org--name/) pour ne garder que le format propre
                let hfCacheDir = Gemma4ModelCache.modelsDirectory
                    .appendingPathComponent("models--\(model.rawValue.replacingOccurrences(of: "/", with: "--"))")
                if FileManager.default.fileExists(atPath: hfCacheDir.path) {
                    try? FileManager.default.removeItem(at: hfCacheDir)
                }

                // Verifier
                if Gemma4ModelCache.isDownloaded(model) {
                    if let size = Gemma4ModelCache.diskSize(for: model) {
                        let sizeGB = String(format: "%.1f", Double(size) / 1_073_741_824)
                        print("  Taille: \(sizeGB) Go")
                    }
                }
            } catch {
                print("\r  ERREUR: \(error.localizedDescription)")
                print("  Verifiez votre token HF: export HF_TOKEN=<token> ou --hf-token <token>")
            }
            print()
        }

        print("Telechargement termine.")
    }
}

// MARK: - Generate (single prompt)

struct Generate: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Genere une reponse a un prompt unique"
    )

    @Option(name: .long, help: "ID HuggingFace du modele")
    var model: String = "mlx-community/gemma-4-e2b-it-4bit"

    @Option(name: .long, help: "Chemin local vers le modele (bypass download)")
    var modelPath: String?

    @Option(name: .long, help: "Token HuggingFace (pour modeles Google)")
    var hfToken: String?

    @Option(name: .long, help: "Prompt systeme")
    var system: String = "Tu es un assistant utile. Reponds de maniere concise."

    @Option(name: .long, help: "Temperature (0.0 = deterministe)")
    var temperature: Float = 0.1

    @Option(name: .long, help: "Nombre maximum de tokens")
    var maxTokens: Int = 512

    @Argument(help: "Le prompt utilisateur")
    var prompt: String

    func run() async throws {
        // 1. Enregistrer Gemma 4
        print("Enregistrement du type gemma4_text...")
        await Gemma4Registration.register()

        // 2. Warning RAM
        warnIfLowRAM(modelId: model)

        // 3. Charger le modele
        let modelSource = modelPath ?? model
        print("Chargement du modele: \(modelSource)")
        let startLoad = Date()

        let container: ModelContainer
        if let path = modelPath {
            // Chargement depuis un repertoire local
            let url = URL(fileURLWithPath: path)
            container = try await loadModelContainer(
                from: url,
                using: #huggingFaceTokenizerLoader()
            )
        } else {
            // Telechargement + chargement depuis HuggingFace
            container = try await loadModelContainer(
                from: #hubDownloader(makeHubClient(token: hfToken)),
                using: #huggingFaceTokenizerLoader(),
                id: model
            ) { progress in
                let pct = Int(progress.fractionCompleted * 100)
                print("\rTelechargement/chargement... \(pct)%", terminator: "")
                fflush(stdout)
            }
        }

        let loadTime = Date().timeIntervalSince(startLoad)
        print("\nModele charge en \(String(format: "%.1f", loadTime))s")

        // 4. Stats GPU
        print("GPU: \(MLX.GPU.activeMemory / (1024 * 1024)) Mo actifs, \(MLX.GPU.peakMemory / (1024 * 1024)) Mo pic")

        // 5. Generer
        print("\n--- Generation ---")
        print("Systeme: \(system)")
        print("Prompt: \(prompt)")
        print("Temperature: \(temperature), Max tokens: \(maxTokens)")
        print("---")

        let params = GenerateParameters(
            maxTokens: maxTokens,
            temperature: temperature,
            topP: 0.95
        )

        let session = ChatSession(container, instructions: system, generateParameters: params)

        let startGen = Date()
        var tokenCount = 0

        // Streaming token par token
        let stream = session.streamResponse(to: prompt)
        for try await token in stream {
            print(token, terminator: "")
            fflush(stdout)
            tokenCount += 1
        }

        let genTime = Date().timeIntervalSince(startGen)
        let tokPerSec = genTime > 0 ? Double(tokenCount) / genTime : 0

        print("\n\n--- Stats ---")
        print("Tokens generes: \(tokenCount)")
        print("Temps: \(String(format: "%.2f", genTime))s")
        print("Vitesse: \(String(format: "%.1f", tokPerSec)) tokens/s")
        print("GPU pic: \(MLX.GPU.peakMemory / (1024 * 1024)) Mo")
    }
}

// MARK: - Chat (multi-turn)

struct Chat: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Mode chat interactif multi-tour"
    )

    @Option(name: .long, help: "ID HuggingFace du modele")
    var model: String = "mlx-community/gemma-4-e2b-it-4bit"

    @Option(name: .long, help: "Chemin local vers le modele")
    var modelPath: String?

    @Option(name: .long, help: "Token HuggingFace (pour modeles Google)")
    var hfToken: String?

    @Option(name: .long, help: "Prompt systeme")
    var system: String = "Tu es un assistant utile. Reponds de maniere concise."

    @Option(name: .long, help: "Temperature")
    var temperature: Float = 0.3

    @Option(name: .long, help: "Max tokens par reponse")
    var maxTokens: Int = 1024

    func run() async throws {
        await Gemma4Registration.register()
        warnIfLowRAM(modelId: model)

        print("Chargement de \(modelPath ?? model)...")
        let container: ModelContainer
        if let path = modelPath {
            let url = URL(fileURLWithPath: path)
            container = try await loadModelContainer(
                from: url,
                using: #huggingFaceTokenizerLoader()
            )
        } else {
            container = try await loadModelContainer(
                from: #hubDownloader(makeHubClient(token: hfToken)),
                using: #huggingFaceTokenizerLoader(),
                id: model
            ) { progress in
                print("\rTelechargement... \(Int(progress.fractionCompleted * 100))%", terminator: "")
                fflush(stdout)
            }
        }
        print("\nModele pret. GPU: \(MLX.GPU.activeMemory / (1024 * 1024)) Mo")

        let params = GenerateParameters(
            maxTokens: maxTokens,
            temperature: temperature,
            topP: 0.95
        )
        let session = ChatSession(container, instructions: system, generateParameters: params)

        print("\nChat Gemma 4 (tapez 'quit' pour quitter)\n")

        while true {
            print("Vous> ", terminator: "")
            fflush(stdout)
            guard let input = readLine(), !input.isEmpty else { continue }
            if input.lowercased() == "quit" || input.lowercased() == "exit" { break }

            print("Gemma> ", terminator: "")
            let stream = session.streamResponse(to: input)
            for try await token in stream {
                print(token, terminator: "")
                fflush(stdout)
            }
            print("\n")
        }

        print("Au revoir!")
    }
}

// MARK: - Describe (multimodal: image, audio, video)

struct Describe: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Decris une image, un audio ou une video (multimodal)"
    )

    @Option(name: .long, help: "ID HuggingFace du modele")
    var model: String = "mlx-community/gemma-4-e2b-it-4bit"

    @Option(name: .long, help: "Chemin local vers le modele (bypass download)")
    var modelPath: String?

    @Option(name: .long, help: "Token HuggingFace (pour modeles Google)")
    var hfToken: String?

    @Option(name: .long, help: "Chemin vers une image")
    var image: String?

    @Option(name: .long, help: "Chemin vers un fichier audio")
    var audio: String?

    @Option(name: .long, help: "Chemin vers une video")
    var video: String?

    @Option(name: .long, help: "Prompt/question sur le media")
    var prompt: String = "Decris ce que tu vois/entends en detail."

    @Option(name: .long, help: "Max tokens")
    var maxTokens: Int = 500

    @Option(name: .long, help: "Temperature")
    var temperature: Float = 0.3

    func run() async throws {
        guard image != nil || audio != nil || video != nil else {
            print("Erreur: specifiez --image, --audio ou --video")
            throw ExitCode.failure
        }

        // 1. Enregistrer en mode multimodal
        print("Enregistrement Gemma 4 (multimodal)...")
        await Gemma4Registration.register(multimodal: true)
        warnIfLowRAM(modelId: model)

        // 2. Charger le modele
        let modelSource = modelPath ?? model
        print("Chargement du modele: \(modelSource)")
        let container: ModelContainer
        if let path = modelPath {
            let url = URL(fileURLWithPath: path)
            container = try await loadModelContainer(from: url, using: #huggingFaceTokenizerLoader())
        } else {
            container = try await loadModelContainer(
                from: #hubDownloader(makeHubClient(token: hfToken)),
                using: #huggingFaceTokenizerLoader(),
                id: model
            ) { progress in
                print("\rChargement... \(Int(progress.fractionCompleted * 100))%", terminator: "")
                fflush(stdout)
            }
        }
        print("Modele charge. GPU: \(MLX.GPU.activeMemory / (1024 * 1024)) Mo")

        // 3. Preparer les inputs multimodaux
        var hasImage = false
        var hasAudio = false
        var numImageTokens = 280
        var numAudioTokens = 0
        var numVideoFrames = 0
        var pixelValues: MLXArray?
        var audioFeatures: Gemma4AudioProcessor.AudioFeatures?

        // Image
        if let imagePath = image {
            print("Traitement de l'image: \(imagePath)")
            let imageURL = URL(fileURLWithPath: imagePath)
            pixelValues = try Gemma4ImageProcessor.processImage(url: imageURL)
            hasImage = true
            print("  Image preprocessee: \(pixelValues!.shape)")
        }

        // Audio
        if let audioPath = audio {
            print("Traitement de l'audio: \(audioPath)")
            let audioURL = URL(fileURLWithPath: audioPath)
            audioFeatures = try await Gemma4AudioProcessor.processAudio(url: audioURL)
            hasAudio = true
            numAudioTokens = audioFeatures!.numTokens
            print("  Audio preprocesse: duree \(String(format: "%.1f", audioFeatures!.durationSeconds))s, \(numAudioTokens) tokens")
        }

        // Video
        if let videoPath = video {
            print("Traitement de la video: \(videoPath)")
            let videoURL = URL(fileURLWithPath: videoPath)
            let frames = try await Gemma4VideoProcessor.processVideo(url: videoURL, maxFrames: 8)
            pixelValues = frames.pixelValues
            numVideoFrames = frames.frameCount
            hasImage = true // video utilise le vision encoder
            print("  Video preprocessee: \(frames.frameCount) frames")
        }

        // 4. Construire le contenu multimodal avec les placeholders
        var contentParts: [String] = []
        if hasImage && video == nil {
            // Un seul <|image|> — le chat template et le tokenizer le gèrent
            contentParts.append("<|image|>")
        }
        if video != nil {
            for _ in 0 ..< numVideoFrames {
                contentParts.append("<|image|>")
            }
        }
        if hasAudio && numAudioTokens > 0 {
            contentParts.append("<|audio|>")
        }
        contentParts.append(prompt)
        let content = contentParts.joined(separator: "\n")

        // 5. Tokeniser via applyChatTemplate (gère les tokens spéciaux correctement)
        let messages: [[String: String]] = [["role": "user", "content": content]]
        var tokenIds: [Int] = try await container.perform { context in
            try context.tokenizer.applyChatTemplate(messages: messages)
        }

        // 6. Expanser les tokens image : remplacer chaque image_token par numImageTokens copies
        // Le tokenizer produit un seul token 258880 par <|image|>, on le remplace par 280
        let imageTokenId = Int(Gemma4Processor.imageTokenId)
        let boiTokenId = Int(Gemma4Processor.boiTokenId)
        let eoiTokenId = Int(Gemma4Processor.eoiTokenId)
        var expandedTokenIds: [Int] = []
        for tid in tokenIds {
            if tid == imageTokenId {
                // Remplacer le single image token par: boi + image_token * N + eoi
                expandedTokenIds.append(boiTokenId)
                for _ in 0 ..< numImageTokens {
                    expandedTokenIds.append(imageTokenId)
                }
                expandedTokenIds.append(eoiTokenId)
            } else {
                expandedTokenIds.append(tid)
            }
        }
        tokenIds = expandedTokenIds
        let inputIds = MLXArray(tokenIds.map { Int32($0) })

        // Debug: premiers et derniers tokens
        let first10 = Array(tokenIds.prefix(15))
        let last10 = Array(tokenIds.suffix(15))
        print("  Premiers tokens: \(first10)")
        print("  Derniers tokens: \(last10)")
        // Compter les tokens speciaux
        let audioCount = tokenIds.filter { $0 == 258881 }.count
        let boaCount = tokenIds.filter { $0 == 256000 }.count
        let eoaCount = tokenIds.filter { $0 == 258883 }.count
        print("  audio_token(\(258881)): \(audioCount), boa(\(256000)): \(boaCount), eoa(\(258883)): \(eoaCount)")

        // Verifier les token counts
        let counts = Gemma4Processor.validateTokenCounts(
            inputIds: inputIds,
            expectedImageTokens: hasImage ? numImageTokens : 0,
            expectedAudioTokens: numAudioTokens
        )
        print("  Tokens image dans le prompt: \(counts.imageCount)")
        if hasAudio { print("  Tokens audio dans le prompt: \(counts.audioCount)") }
        print("  Total input tokens: \(inputIds.shape[0])")

        // 6. Injecter les donnees multimodales dans le modele
        nonisolated(unsafe) let finalPixelValues = pixelValues
        nonisolated(unsafe) let finalAudioFeatures = audioFeatures
        await container.perform { context in
            if let model = context.model as? Gemma4MultimodalLLMModel {
                model.pendingPixelValues = finalPixelValues
                if let af = finalAudioFeatures {
                    model.pendingAudioFeatures = af.features
                    model.pendingAudioMask = af.mask
                }
            }
        }

        print("\n--- Generation multimodale ---")
        if hasImage { print("  Mode: vision") }
        if hasAudio { print("  Mode: audio") }
        print("  Prompt: \(prompt)")
        print("---")

        // 7. Generer la reponse via generate() du container
        let startTime = Date()
        var tokenCount = 0
        nonisolated(unsafe) let capturedInputIds = inputIds

        // Utiliser container.generate directement avec les input_ids prepares
        let result = try await container.perform { context in
            let lmInput = LMInput(text: .init(tokens: expandedDimensions(capturedInputIds, axis: 0)))
            var generatedTokens: [Int] = []

            let params = GenerateParameters(
                maxTokens: self.maxTokens,
                temperature: self.temperature,
                topP: 0.95
            )

            // Preparer le cache
            let cache = context.model.newCache(parameters: params)

            // Prefill: traiter tous les input tokens d'un coup
            let prefillOutput = context.model(capturedInputIds.reshaped(1, -1), cache: cache)
            var nextToken = argMax(prefillOutput[0..., prefillOutput.dim(1) - 1, 0...], axis: -1).item(Int32.self)

            for _ in 0 ..< self.maxTokens {
                generatedTokens.append(Int(nextToken))

                // Decoder et afficher le token
                let text = context.tokenizer.decode(tokenIds: [Int(nextToken)])
                print(text, terminator: "")
                fflush(stdout)

                // EOS tokens officiels Gemma 4 (generation_config.json)
                // 1 = <eos>, 106 = <start_of_turn>, 50 = (token de fin)
                if nextToken == 1 || nextToken == 106 || nextToken == 50 { break }

                // Generer le token suivant
                let nextInput = MLXArray([nextToken]).reshaped(1, 1)
                let output = context.model(nextInput, cache: cache)
                if self.temperature <= 0.01 {
                    nextToken = argMax(output[0..., 0, 0...], axis: -1).item(Int32.self)
                } else {
                    let logits = output[0..., 0, 0...] / self.temperature
                    let probs = softmax(logits, axis: -1)
                    nextToken = MLXRandom.categorical(log(probs)).item(Int32.self)
                }
            }

            return generatedTokens
        }

        tokenCount = result.count
        let elapsed = Date().timeIntervalSince(startTime)
        print("\n\n--- Stats ---")
        print("Tokens: \(tokenCount), Temps: \(String(format: "%.2f", elapsed))s, Vitesse: \(String(format: "%.1f", Double(tokenCount) / max(0.01, elapsed))) t/s")
        print("GPU pic: \(MLX.GPU.peakMemory / (1024 * 1024)) Mo")
    }
}
