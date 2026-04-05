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

@main
struct Gemma4CLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "gemma4-cli",
        abstract: "Test d'inference Gemma 4 via MLX Swift",
        subcommands: [Generate.self, Chat.self, Describe.self],
        defaultSubcommand: Generate.self
    )
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

        // 2. Charger le modele
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
                from: #hubDownloader(),
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

        // 3. Stats GPU
        print("GPU: \(MLX.GPU.activeMemory / (1024 * 1024)) Mo actifs, \(MLX.GPU.peakMemory / (1024 * 1024)) Mo pic")

        // 4. Generer
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

    @Option(name: .long, help: "Prompt systeme")
    var system: String = "Tu es un assistant utile. Reponds de maniere concise."

    @Option(name: .long, help: "Temperature")
    var temperature: Float = 0.3

    @Option(name: .long, help: "Max tokens par reponse")
    var maxTokens: Int = 1024

    func run() async throws {
        await Gemma4Registration.register()

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
                from: #hubDownloader(),
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

        // 2. Charger le modele
        print("Chargement du modele: \(model)")
        let container = try await loadModelContainer(
            from: #hubDownloader(),
            using: #huggingFaceTokenizerLoader(),
            id: model
        ) { progress in
            print("\rChargement... \(Int(progress.fractionCompleted * 100))%", terminator: "")
            fflush(stdout)
        }
        print("\nModele charge. GPU: \(MLX.GPU.activeMemory / (1024 * 1024)) Mo")

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

        // 4. Construire le prompt avec les vrais tokens multimodaux
        let multimodalPrompt = Gemma4Processor.buildMultimodalPrompt(
            userPrompt: prompt,
            hasImage: hasImage && video == nil,
            numImageTokens: numImageTokens,
            hasAudio: hasAudio,
            numAudioTokens: numAudioTokens,
            hasVideo: video != nil,
            numVideoFrames: numVideoFrames
        )

        // 5. Tokeniser avec le vrai tokenizer (qui connait les tokens speciaux)
        let tokenIds = await container.encode(multimodalPrompt)
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

                // Verifier EOS
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
