// Commande CLI pour tester DiffusionGemma (block-AR diffusion).
//
// Usage :
//   gemma4-cli diffusion --model-path /path/to/diffusion_gemma_dir "Mon prompt"
//
// Pour la Phase 4, on ne supporte que le chargement depuis un repertoire local
// avec config.json + safetensors + tokenizer.json. Pas de quantization
// ni de cache encoder incremental encore.

import ArgumentParser
import Foundation
import Gemma4Swift
import MLX
import MLXNN
import Tokenizers

struct DiffusionCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "diffusion",
        abstract: "Generation block-AR via DiffusionGemma (experimental Phase 4)"
    )

    @Option(name: .long, help: "ID HuggingFace du modele DiffusionGemma (ex: google/diffusiongemma-26B-A4B-it)")
    var model: String = "google/diffusiongemma-26B-A4B-it"

    @Option(name: .long, help: "Chemin local du modele (bypass cache et model ID)")
    var modelPath: String?

    @Option(name: .long, help: "Token HuggingFace pour le telechargement")
    var hfToken: String?

    @Flag(name: .long, help: "Telecharger le modele s'il n'est pas en cache")
    var downloadIfNeeded: Bool = false

    @Option(name: .long, help: "Prompt systeme. Vide = pas de prompt systeme.")
    var system: String = ""

    @Option(name: .long, help: "Nombre maximum de canvases a generer")
    var maxBlocks: Int = 4

    @Option(name: .long, help: "Seed PRNG pour le sampler (varier pour observer la diversite)")
    var seed: UInt64 = 0

    @Option(name: .long, help: "Override t_min du schedule de temperature (defaut: generation_config.json, typiquement 0.4)")
    var tMin: Float?

    @Option(name: .long, help: "Override t_max du schedule de temperature (defaut: generation_config.json, typiquement 0.8)")
    var tMax: Float?

    @Option(name: .long, help: "Override max_denoising_steps (defaut: 48). Plus = plus de variation mais plus lent.")
    var maxSteps: Int?

    @Flag(name: .long, help: "Charger les poids vision (Phase 5+, ignore pour l'instant).")
    var includeVision: Bool = false

    @Flag(name: .long, help: "Skip generation: charge seulement le modele et affiche les shapes.")
    var smoke: Bool = false

    @Flag(name: .long, help: "Mode dry-run keys : instancie le modele sans poids, compare avec le safetensors index si present. N'instancie PAS de tensor lourd.")
    var validateKeys: Bool = false

    @Option(name: .long, help: "Chemin vers une image à passer au modèle (active le vision_tower). Requires --include-vision pour charger les poids vision.")
    var image: String?

    @Option(name: .customLong("quantize-bits"), help: "Quantification a la volee (4, 6, 8). Applique apres le load, AVANT la generation. 48 Go -> ~14 Go en 4-bit, 3-4x speedup forwards.")
    var quantizeBits: Int?

    @Option(name: .customLong("quantize-group-size"), help: "Group size de quantization (defaut 64 pour affine, 32 pour mxfp).")
    var quantizeGroupSize: Int = 64

    @Option(name: .customLong("quantize-mode"), help: "Mode quantization : affine | mxfp4 | mxfp8.")
    var quantizeMode: String = "affine"

    @Flag(name: .customLong("quantize-text-only"), help: "Preserve vision_tower / embed_vision en bf16, quantize seulement encoder.language_model + decoder + self_conditioning.")
    var quantizeTextOnly: Bool = false

    @Option(name: .customLong("mixed-precision"), help: "Mixed precision (Q-DiT/ViDiT-Q) : preset 'default' | 'conservative' | 'aggressive'. Mutuellement exclusif avec --quantize-bits.")
    var mixedPrecision: String?

    @Option(name: .customLong("cache-limit-gb"), help: "Force MLX cache GPU limit a N GB. Test pour forcer la liberation des anciens bf16 apres quantize. 0 = no limit (defaut MLX).")
    var cacheLimitGB: Int?

    @Flag(name: .long, help: "Streaming step-by-step : affiche le canvas decode (argmax) apres CHAQUE step de denoising. Equivalent du `streamer.put_draft` Python = voir le texte se debruiter en live.")
    var streamSteps: Bool = false

    @Option(name: .long, help: "Avec --stream-steps : effacer l'ecran entre steps (ANSI clear). Defaut: print en-place sans effacer.")
    var streamMode: String = "inplace"

    @Argument(help: "Prompt utilisateur")
    var prompt: String = "Hello"

    /// Resout le repertoire local du modele (chemin direct ou cache HF).
    private func resolveDirectory() async throws -> URL {
        if let modelPath = modelPath {
            return URL(fileURLWithPath: modelPath)
        }
        let modelId = Self.shortcuts[model.lowercased()] ?? model
        let token = hfToken ?? ProcessInfo.processInfo.environment["HF_TOKEN"]

        // Telecharger si besoin
        if downloadIfNeeded && !Gemma4ModelCache.isDownloaded(modelId: modelId) {
            print("Telechargement de \(modelId)...")
            _ = try await Gemma4ModelDownloader.download(modelId: modelId, token: token) { p in
                print("\r  \(p.formattedProgress) (\(p.formattedSpeed))", terminator: "")
                fflush(stdout)
            }
            print()
        }

        // Lookup cache
        var dir = Gemma4ModelCache.modelsDirectory
        for part in modelId.split(separator: "/") {
            dir = dir.appendingPathComponent(String(part))
        }
        guard FileManager.default.fileExists(atPath: dir.path) else {
            throw ValidationError("Modele '\(modelId)' non telecharge. Utilisez --download-if-needed ou `gemma4-cli download a4b-diff-bf16`.")
        }
        return dir
    }

    static let shortcuts: [String: String] = [
        "a4b-diff-bf16": "google/diffusiongemma-26B-A4B-it",
        "diff-bf16": "google/diffusiongemma-26B-A4B-it",
    ]

    func run() async throws {
        let directory = try await resolveDirectory()

        print("--- DiffusionGemma loader (Phase 4 experimental) ---")
        print("Repertoire : \(directory.path)")

        // Mode validation des cles : instancie le modele SANS charger les poids
        // et compare avec le safetensors index s'il est present.
        if validateKeys {
            try runValidateKeys(directory: directory)
            return
        }

        // 1) Loader modele
        let loadStart = Date()
        let (model, config) = try DiffusionGemmaLoader.load(
            from: directory, includeVision: includeVision
        )
        let loadTime = Date().timeIntervalSince(loadStart)
        print("Modele charge en \(String(format: "%.1f", loadTime))s")
        print("Config: hidden=\(config.textConfig.base.hiddenSize), layers=\(config.textConfig.base.numHiddenLayers), canvas=\(config.textConfig.canvasLength), vocab=\(config.textConfig.base.vocabSize)")
        print("GPU: \(MLX.GPU.activeMemory / (1024 * 1024)) Mo actifs, \(MLX.GPU.peakMemory / (1024 * 1024)) Mo pic")

        // 1a-bis) Override du cache MLX si demande
        if let cacheGB = cacheLimitGB {
            let bytes = cacheGB * 1024 * 1024 * 1024
            MLX.Memory.cacheLimit = bytes
            print("MLX cache limit force a \(cacheGB) GB (\(bytes) bytes)")
        }

        // 1b) Quantification a la volee
        if let preset = mixedPrecision {
            // Mixed precision (Q-DiT pattern)
            let mpConfig: DiffusionOnTheFlyQuantization.MixedPrecisionConfig
            switch preset.lowercased() {
            case "default": mpConfig = .default
            case "conservative": mpConfig = .conservative
            case "aggressive": mpConfig = .aggressive
            default:
                print("Erreur : --mixed-precision doit etre 'default' | 'conservative' | 'aggressive'")
                throw ExitCode.failure
            }
            let quantStart = Date()
            let stats = DiffusionOnTheFlyQuantization.applyMixedPrecision(to: model, config: mpConfig)
            let quantTime = Date().timeIntervalSince(quantStart)
            print("Mixed precision (\(preset)) : \(stats.quantizedHigh) modules en \(mpConfig.highPrecisionBits)-bit, \(stats.quantizedLow) modules en \(mpConfig.lowPrecisionBits)-bit (skipped \(stats.skipped.count)) en \(String(format: "%.1f", quantTime))s")
            print("Layers high-precision (\(mpConfig.highPrecisionBits)-bit) : \(mpConfig.highPrecisionLayers.sorted())")
            print("GPU apres mixed-precision : \(MLX.GPU.activeMemory / (1024 * 1024)) Mo actifs")
        } else if let bits = quantizeBits {
            guard let mode = DiffusionOnTheFlyQuantization.Mode(rawValue: quantizeMode) else {
                print("Erreur: --quantize-mode doit etre affine|mxfp4|mxfp8")
                throw ExitCode.failure
            }
            let excluded = quantizeTextOnly ? DiffusionOnTheFlyQuantization.multimodalEncoderPrefixes : []
            let quantStart = Date()
            let count = DiffusionOnTheFlyQuantization.apply(
                to: model, bits: bits, groupSize: quantizeGroupSize, mode: mode,
                excludedPathPrefixes: excluded
            )
            let quantTime = Date().timeIntervalSince(quantStart)
            let scope = quantizeTextOnly ? "text-only" : "all"
            print("Quantification a la volee (\(scope)): \(count) modules (\(bits)-bit, group=\(quantizeGroupSize), \(mode.rawValue)) en \(String(format: "%.1f", quantTime))s")
            print("GPU apres quantization : \(MLX.GPU.activeMemory / (1024 * 1024)) Mo actifs")
        }

        if smoke {
            print("Smoke test : modele charge avec succes. Pas de generation.")
            return
        }

        // 2) Tokenizer
        let tokenizer = try await AutoTokenizer.from(modelFolder: directory)

        // 2b) Image optionnelle
        let pixelValues: MLXArray?
        let numImageSoftTokens = config.visionSoftTokensPerImage  // 280
        if let imagePath = image {
            guard includeVision else {
                print("Erreur : --image necessite --include-vision pour charger le vision_tower")
                throw ExitCode.failure
            }
            let imageURL = URL(fileURLWithPath: imagePath)
            let pixels = try Gemma4ImageProcessor.processImage(url: imageURL)
            print("Image preprocessee : \(pixels.shape)")
            pixelValues = pixels
        } else {
            pixelValues = nil
        }

        // 3) Tokeniser le prompt (avec <|image|> au debut si image fournie)
        let content: String
        if pixelValues != nil {
            content = "<|image|>\n\(prompt)"
        } else {
            content = prompt
        }
        let messages: [[String: String]]
        if system.isEmpty {
            messages = [["role": "user", "content": content]]
        } else {
            messages = [
                ["role": "system", "content": system],
                ["role": "user", "content": content],
            ]
        }
        var tokenIds = try tokenizer.applyChatTemplate(messages: messages)

        // 3b) Expanser <|image|> en boi + image_token x N + eoi
        if pixelValues != nil {
            let imageTokenId = config.imageTokenId
            let boiTokenId = config.boiTokenId
            let eoiTokenId = config.eoiTokenId
            var expanded: [Int] = []
            for tid in tokenIds {
                if tid == imageTokenId {
                    expanded.append(boiTokenId)
                    for _ in 0 ..< numImageSoftTokens {
                        expanded.append(imageTokenId)
                    }
                    expanded.append(eoiTokenId)
                } else {
                    expanded.append(tid)
                }
            }
            tokenIds = expanded
        }

        let promptIds = MLXArray(tokenIds.map { Int32($0) }).reshaped(1, -1)
        print("Prompt tokens : \(tokenIds.count)")
        print("Premiers tokens : \(Array(tokenIds.prefix(10)))")

        // 4) Generation : permettre overrides de temperature / steps
        var genConfig = try loadGenerationConfig(directory: directory)
        if tMin != nil || tMax != nil || maxSteps != nil {
            genConfig = DiffusionGenerationConfig(
                tMin: tMin ?? genConfig.tMin,
                tMax: tMax ?? genConfig.tMax,
                maxDenoisingSteps: maxSteps ?? genConfig.maxDenoisingSteps,
                entropyBound: genConfig.entropyBound,
                stabilityThreshold: genConfig.stabilityThreshold,
                confidenceThreshold: genConfig.confidenceThreshold,
                eosTokenIds: genConfig.eosTokenIds,
                padTokenId: genConfig.padTokenId
            )
        }
        let pipeline = DiffusionGemmaPipeline(model: model, genConfig: genConfig)

        print("\n--- Generation ---")
        print("max_blocks=\(maxBlocks), canvas_length=\(config.textConfig.canvasLength), max_denoising_steps=\(genConfig.maxDenoisingSteps)")
        print("temperature schedule : \(genConfig.tMin) -> \(genConfig.tMax), seed=\(seed)")
        print("---")

        // Captures pour les closures (Sendable Tokenizer copy)
        let captureTokenizer = tokenizer
        let captureStreamMode = streamMode

        let genStart = Date()
        let result: DiffusionGenerationResult
        if streamSteps {
            // Stream chaque step : decode l'argmax_canvas et l'affiche en place
            result = await pipeline.generate(
                promptIds: promptIds,
                pixelValues: pixelValues,
                maxBlocks: maxBlocks,
                seed: seed,
                onCanvas: { canvasIdx, canvas in
                    canvas.eval()
                    let tokens = canvas.asArray(Int32.self).map { Int($0) }
                    let text = captureTokenizer.decode(tokens: tokens)
                    print("\n[canvas \(canvasIdx) COMMITTED] \(text)\n")
                },
                onStep: { canvasIdx, step, argmaxCanvas in
                    argmaxCanvas.eval()
                    let tokens = argmaxCanvas.asArray(Int32.self).map { Int($0) }
                    let text = captureTokenizer.decode(tokens: tokens)
                    if captureStreamMode == "clear" {
                        // Efface l'ecran + curseur en haut a gauche
                        print("\u{001B}[2J\u{001B}[H", terminator: "")
                    }
                    print("[c\(canvasIdx) step \(String(format: "%2d", step))] \(text)")
                    if captureStreamMode == "inplace" {
                        // Separateur leger
                        print(String(repeating: "-", count: 80))
                    }
                    fflush(stdout)
                }
            )
        } else {
            result = await pipeline.generate(
                promptIds: promptIds,
                pixelValues: pixelValues,
                maxBlocks: maxBlocks,
                seed: seed,
                onCanvas: { canvasIdx, canvas in
                    canvas.eval()
                    let tokens = canvas.asArray(Int32.self).map { Int($0) }
                    let text = captureTokenizer.decode(tokens: tokens)
                    print("[canvas \(canvasIdx)] \(text)")
                }
            )
        }
        let genTime = Date().timeIntervalSince(genStart)

        // 5) Stats
        result.generatedIds.eval()
        let allTokens = result.generatedIds.asArray(Int32.self).map { Int($0) }
        let fullText = tokenizer.decode(tokens: allTokens)
        print("\n--- Output complete ---")
        print(fullText)
        print("\n--- Stats ---")
        print("Tokens generes : \(allTokens.count)")
        print("Canvases : \(result.canvases)")
        print("Decoder forwards : \(result.totalDecoderSteps)")
        print("Temps : \(String(format: "%.2f", genTime))s")
        if result.totalDecoderSteps > 0 {
            let tokPerSec = Double(allTokens.count) / max(0.01, genTime)
            print("Vitesse : \(String(format: "%.1f", tokPerSec)) tok/s (\(String(format: "%.2f", genTime / Double(result.totalDecoderSteps)))s/step)")
        }
        print("GPU pic : \(MLX.GPU.peakMemory / (1024 * 1024)) Mo")
    }

    private func loadGenerationConfig(directory: URL) throws -> DiffusionGenerationConfig {
        let url = directory.appendingPathComponent("generation_config.json")
        if FileManager.default.fileExists(atPath: url.path) {
            let data = try Data(contentsOf: url)
            return try JSONDecoder().decode(DiffusionGenerationConfig.self, from: data)
        }
        return DiffusionGenerationConfig()
    }

    // MARK: - Validation des cles (sans charger les poids)

    private func runValidateKeys(directory: URL) throws {
        // 1) Lire config et instancier le modele (les poids restent a zero)
        let config = try DiffusionGemmaLoader.loadConfig(from: directory)
        let modelInstance = DiffusionGemmaForBlockDiffusion(config)

        // 2) Lister les cles attendues via mapParameters (sans materialiser les valeurs)
        let expectedKeys = Set(modelInstance.parameters().flattened().map { $0.0 })
        print("Cles attendues par le modele Swift : \(expectedKeys.count)")

        // 3) Lire safetensors index si present
        let indexURL = directory.appendingPathComponent("model.safetensors.index.json")
        guard FileManager.default.fileExists(atPath: indexURL.path) else {
            print("[INFO] Pas de model.safetensors.index.json -> impossible de comparer.")
            print("Premiers 20 cles attendues :")
            for k in expectedKeys.sorted().prefix(20) {
                print("  \(k)")
            }
            return
        }
        let indexData = try Data(contentsOf: indexURL)
        guard let parsed = try JSONSerialization.jsonObject(with: indexData) as? [String: Any],
              let weightMap = parsed["weight_map"] as? [String: String]
        else {
            print("[ERROR] Impossible de parser model.safetensors.index.json")
            return
        }
        let rawKeys = Array(weightMap.keys)
        print("Cles brutes dans le checkpoint : \(rawKeys.count)")

        // 4) Sanitize les cles (dummy values shape [2,2] pour pouvoir swappedAxes)
        let dummy = MLXArray.zeros([2, 2])
        let dummyArrays: [String: MLXArray] = Dictionary(
            uniqueKeysWithValues: rawKeys.map { ($0, dummy) }
        )
        let sanitized = DiffusionWeightSanitizer.sanitize(dummyArrays, includeVision: includeVision)
        let sanitizedKeys = Set(sanitized.keys)
        print("Cles produites par le sanitizer : \(sanitizedKeys.count)")

        // 5) Comparer
        let missingInSanitized = expectedKeys.subtracting(sanitizedKeys)
        let extraInSanitized = sanitizedKeys.subtracting(expectedKeys)

        print("\n=== DIFF ===")
        if missingInSanitized.isEmpty && extraInSanitized.isEmpty {
            print("✓ Toutes les cles correspondent !")
        } else {
            print("Manquantes (attendues par Swift mais pas produites par le sanitizer) : \(missingInSanitized.count)")
            for k in missingInSanitized.sorted().prefix(30) {
                print("  - \(k)")
            }
            if missingInSanitized.count > 30 { print("  ... +\(missingInSanitized.count - 30)") }
            print()
            print("En trop (produites par le sanitizer mais pas attendues par Swift) : \(extraInSanitized.count)")
            for k in extraInSanitized.sorted().prefix(30) {
                print("  + \(k)")
            }
            if extraInSanitized.count > 30 { print("  ... +\(extraInSanitized.count - 30)") }
        }
    }
}
