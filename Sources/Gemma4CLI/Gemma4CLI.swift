// Mini CLI pour tester l'inference Gemma 4 via MLX Swift

import ArgumentParser
import Foundation
import Gemma4Swift
import MLX
import MLXLMCommon
import MLXLLM
import Tokenizers

// MARK: - Helpers

/// Resout le token HuggingFace depuis l'option CLI ou l'environnement
func resolveHFToken(_ token: String?) -> String? {
    token ?? ProcessInfo.processInfo.environment["HF_TOKEN"]
}

/// Charge un modele depuis un chemin local
func loadLocalModel(path: String) async throws -> ModelContainer {
    let url = URL(fileURLWithPath: path)
    await Gemma4Registration.register()
    return try await loadModelContainer(from: url, using: LocalTokenizerLoader())
}

/// Charge un modele multimodal depuis un chemin local
func loadLocalMultimodalModel(path: String) async throws -> ModelContainer {
    let url = URL(fileURLWithPath: path)
    await Gemma4Registration.register(multimodal: true)
    return try await loadModelContainer(from: url, using: LocalTokenizerLoader())
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

        let token = resolveHFToken(hfToken)

        // Telecharger sequentiellement
        for (i, model) in toDownload.enumerated() {
            print("[\(i + 1)/\(toDownload.count)] \(model.displayName) (~\(Int(model.estimatedSizeGB)) Go)")
            print("  ID: \(model.rawValue)")

            let startTime = Date()
            let parts = model.rawValue.split(separator: "/")
            let destDir = Gemma4ModelCache.modelsDirectory
                .appendingPathComponent(String(parts[0]))
                .appendingPathComponent(String(parts[1]))

            do {
                try await LocalModelDownloader.download(
                    modelId: model.rawValue,
                    to: destDir,
                    token: token
                ) { pct in
                    print("\r  Progression: \(Int(pct * 100))%", terminator: "")
                    fflush(stdout)
                }

                let elapsed = Date().timeIntervalSince(startTime)
                print("\r  Termine en \(String(format: "%.0f", elapsed))s")

                if let size = Gemma4ModelCache.diskSize(for: model) {
                    let sizeGB = String(format: "%.1f", Double(size) / 1_073_741_824)
                    print("  Taille: \(sizeGB) Go")
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

        guard let path = modelPath else {
            print("Erreur: --model-path requis. Utilisez 'gemma4-cli download' pour telecharger un modele.")
            throw ExitCode.failure
        }
        let container = try await loadLocalModel(path: path)

        let loadTime = Date().timeIntervalSince(startLoad)
        print("Modele charge en \(String(format: "%.1f", loadTime))s")

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

        guard let path = modelPath else {
            print("Erreur: --model-path requis. Utilisez 'gemma4-cli download' pour telecharger un modele.")
            throw ExitCode.failure
        }
        print("Chargement de \(path)...")
        let container = try await loadLocalModel(path: path)
        print("Modele pret.")

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

    @Option(name: .long, help: "Chemin vers une ou plusieurs images (repetable)")
    var image: [String] = []

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
        guard !image.isEmpty || audio != nil || video != nil else {
            print("Erreur: specifiez --image, --audio ou --video")
            throw ExitCode.failure
        }

        // 1. Enregistrer en mode multimodal
        print("Enregistrement Gemma 4 (multimodal)...")
        await Gemma4Registration.register(multimodal: true)
        warnIfLowRAM(modelId: model)

        // 2. Charger le modele
        guard let path = modelPath else {
            print("Erreur: --model-path requis. Utilisez 'gemma4-cli download' pour telecharger un modele.")
            throw ExitCode.failure
        }
        print("Chargement du modele: \(path)")
        let container = try await loadLocalMultimodalModel(path: path)
        print("Modele charge.")

        // 3. Preparer les inputs multimodaux
        let numImages = image.count
        var hasImage = false
        var hasAudio = false
        var hasVideo = false
        let numImageTokens = 280
        var numAudioTokens = 0
        var numVideoFrames = 0
        var videoSoftTokensPerFrame = Gemma4VideoProcessor.defaultSoftTokensPerFrame
        var videoTimestamps: [Double] = []
        var pixelValues: MLXArray?
        var videoFrameValues: MLXArray?
        var audioFeatures: Gemma4AudioProcessor.AudioFeatures?

        // Images (une ou plusieurs)
        var allPixelValues: [MLXArray] = []
        if !image.isEmpty {
            for imagePath in image {
                print("Traitement de l'image: \(imagePath)")
                let imageURL = URL(fileURLWithPath: imagePath)
                let pixels = try Gemma4ImageProcessor.processImage(url: imageURL)
                print("  Image preprocessee: \(pixels.shape)")
                allPixelValues.append(pixels)
            }

            if allPixelValues.count == 1 {
                pixelValues = allPixelValues[0]
            } else {
                // Multi-image: padder a la meme taille pour pouvoir batacher
                let maxH = allPixelValues.map { $0.dim(2) }.max()!
                let maxW = allPixelValues.map { $0.dim(3) }.max()!
                var padded: [MLXArray] = []
                for pv in allPixelValues {
                    let h = pv.dim(2), w = pv.dim(3)
                    if h == maxH && w == maxW {
                        padded.append(pv)
                    } else {
                        let result = MLXArray.zeros([1, 3, maxH, maxW], dtype: pv.dtype)
                        result[0..., 0..., 0 ..< h, 0 ..< w] = pv
                        padded.append(result)
                    }
                }
                pixelValues = concatenated(padded, axis: 0)
            }
            hasImage = true
            print("  Total: \(numImages) image(s), batch: \(pixelValues!.shape)")
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

        // Video (pipeline aligné sur la référence Python)
        if let videoPath = video {
            print("Traitement de la video: \(videoPath)")
            let videoURL = URL(fileURLWithPath: videoPath)
            let frames = try await Gemma4VideoProcessor.processVideo(url: videoURL)
            videoFrameValues = frames.pixelValues
            numVideoFrames = frames.frameCount
            videoSoftTokensPerFrame = frames.softTokensPerFrame
            videoTimestamps = frames.timestamps
            hasVideo = true
            print("  Video preprocessee: \(frames.frameCount) frames, \(frames.softTokensPerFrame) tokens/frame, \(String(format: "%.1f", frames.sourceFPS)) fps source")
            print("  Timestamps: \(frames.timestamps.map { Gemma4VideoProcessor.formatTimestamp($0) }.joined(separator: " "))")
        }

        // 4. Construire le contenu multimodal avec les placeholders
        var contentParts: [String] = []
        if hasImage {
            for _ in 0 ..< numImages {
                contentParts.append("<|image|>")
            }
        }
        if hasVideo {
            // Video: timestamp + <|video|> par frame (ref Python)
            for i in 0 ..< numVideoFrames {
                let ts = Gemma4VideoProcessor.formatTimestamp(videoTimestamps[i])
                contentParts.append("\(ts)\n<|video|>")
            }
        }
        if hasAudio && numAudioTokens > 0 {
            contentParts.append("<|audio|>")
        }
        contentParts.append(prompt)
        let content = contentParts.joined(separator: "\n")

        // 5. Tokeniser via applyChatTemplate
        let messages: [[String: String]] = [["role": "user", "content": content]]
        var tokenIds: [Int] = try await container.perform { context in
            try context.tokenizer.applyChatTemplate(messages: messages)
        }

        // 6. Expanser les tokens: remplacer chaque token special par N copies
        let imageTokenId = Int(Gemma4Processor.imageTokenId)
        let videoTokenId = Int(Gemma4Processor.videoTokenId)
        let audioTokenId = Int(Gemma4Processor.audioTokenId)
        let boiTokenId = Int(Gemma4Processor.boiTokenId)
        let eoiTokenId = Int(Gemma4Processor.eoiTokenId)
        let boaTokenId = Int(Gemma4Processor.boaTokenId)
        let eoaTokenId = Int(Gemma4Processor.eoaTokenId)
        var expandedTokenIds: [Int] = []
        for tid in tokenIds {
            if tid == imageTokenId {
                // Image: boi + image_token * 280 + eoi
                expandedTokenIds.append(boiTokenId)
                for _ in 0 ..< numImageTokens {
                    expandedTokenIds.append(imageTokenId)
                }
                expandedTokenIds.append(eoiTokenId)
            } else if tid == videoTokenId {
                // Video: boi + video_token * 70 + eoi
                expandedTokenIds.append(boiTokenId)
                for _ in 0 ..< videoSoftTokensPerFrame {
                    expandedTokenIds.append(videoTokenId)
                }
                expandedTokenIds.append(eoiTokenId)
            } else if tid == audioTokenId {
                // Audio: boa + audio_token * N + eoa
                expandedTokenIds.append(boaTokenId)
                for _ in 0 ..< numAudioTokens {
                    expandedTokenIds.append(audioTokenId)
                }
                expandedTokenIds.append(eoaTokenId)
            } else {
                expandedTokenIds.append(tid)
            }
        }
        tokenIds = expandedTokenIds
        let inputIds = MLXArray(tokenIds.map { Int32($0) })

        // Debug tokens
        let first15 = Array(tokenIds.prefix(15))
        let last15 = Array(tokenIds.suffix(15))
        print("  Premiers tokens: \(first15)")
        print("  Derniers tokens: \(last15)")
        let imgCount = tokenIds.filter { $0 == imageTokenId }.count
        let vidCount = tokenIds.filter { $0 == videoTokenId }.count
        let audCount = tokenIds.filter { $0 == 258881 }.count
        print("  image_token: \(imgCount), video_token: \(vidCount), audio_token: \(audCount)")
        print("  Total input tokens: \(inputIds.shape[0])")

        // 7. Injecter les donnees multimodales dans le modele
        nonisolated(unsafe) let finalPixelValues = pixelValues
        nonisolated(unsafe) let finalVideoFrames = videoFrameValues
        nonisolated(unsafe) let finalVideoSoftTokens = videoSoftTokensPerFrame
        nonisolated(unsafe) let finalAudioFeatures = audioFeatures
        await container.perform { context in
            if let model = context.model as? Gemma4MultimodalLLMModel {
                model.pendingPixelValues = finalPixelValues
                if finalVideoFrames != nil {
                    model.pendingVideoFrames = finalVideoFrames
                    model.pendingVideoSoftTokensPerFrame = finalVideoSoftTokens
                }
                if let af = finalAudioFeatures {
                    model.pendingAudioFeatures = af.features
                    model.pendingAudioMask = af.mask
                }
            }
        }

        print("\n--- Generation multimodale ---")
        if hasImage { print("  Mode: vision (\(numImages) image(s), \(numImageTokens) tokens/image)") }
        if hasVideo { print("  Mode: video (\(numVideoFrames) frames, \(videoSoftTokensPerFrame) tokens/frame)") }
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
