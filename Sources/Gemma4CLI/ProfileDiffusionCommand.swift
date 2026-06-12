// Profiling de DiffusionGemma — phases + denoising steps + Chrome Trace
//
// Mesure 7 phases :
//   1. Model Loading       (lecture safetensors + sanitize + update)
//   2. Tokenization         (chat template + expand image tokens)
//   3. Vision encoding      (vision_tower + embed_vision, si image)
//   4. Encoder forward      (DiffusionGemmaEncoderTextModel)
//   5. Denoising loop       (recordStep par step)
//   6. Sampling             (categorical + accept + renoise)
//   7. Decode               (tokenizer.decode du canvas final)
//
// Pour 1024 tokens (4 canvas), on s'attend a voir :
// - Vision (si image) : ~1s
// - Encoder forward : se REPETE a chaque canvas (= O(N) re-encodings)
//   -> sans doute la principale source de lenteur ; sera optimisee
//      par le KV cache encoder incremental (Phase 6).
// - Decoder denoising : ~0.7-1s par step, 9-48 steps par canvas
//
// Sortie : rapport texte + Chrome Trace JSON (chargeable dans chrome://tracing)

import ArgumentParser
import Foundation
import Gemma4Swift
import MLX
import MLXProfiler
import Tokenizers

struct ProfileDiffusion: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "profile-diffusion",
        abstract: "Profile DiffusionGemma : phases + denoising steps + Chrome Trace"
    )

    @Option(name: .long, help: "ID HuggingFace du modele")
    var model: String = "google/diffusiongemma-26B-A4B-it"

    @Option(name: .long, help: "Chemin local du modele (bypass cache)")
    var modelPath: String?

    @Option(name: .long, help: "Prompt utilisateur")
    var prompt: String = "Why is the sky blue? Answer in 4 paragraphs with examples and analogies."

    @Option(name: .long, help: "Chemin vers une image (active vision_tower)")
    var image: String?

    @Flag(name: .long, help: "Charger les poids vision (necessaire pour --image, ajoute ~600 MB)")
    var includeVision: Bool = false

    @Option(name: .long, help: "Nombre maximum de canvases (chacun = 256 tokens)")
    var maxBlocks: Int = 4

    @Option(name: .long, help: "Seed PRNG")
    var seed: UInt64 = 0

    @Option(name: .long, help: "Override t_min")
    var tMin: Float?

    @Option(name: .long, help: "Override t_max")
    var tMax: Float?

    @Option(name: .long, help: "Override max_denoising_steps")
    var maxSteps: Int?

    @Flag(name: .long, help: "Tracker la memoire par step (overhead)")
    var perStepMemory: Bool = false

    @Flag(name: .long, help: "Desactiver l'export Chrome Trace")
    var noChromeTrace: Bool = false

    @Option(name: .long, help: "Repertoire de sortie pour la trace (defaut: cwd)")
    var output: String?

    func run() async throws {
        // 1) Config profiling
        let config = ProfilingConfig(
            trackMemory: true,
            trackPerStepMemory: perStepMemory,
            exportChromeTrace: !noChromeTrace,
            printSummary: true
        )
        let session = ProfilingSession(config: config)
        session.metadata["model"] = "diffusiongemma-26B-A4B-it"
        session.metadata["maxBlocks"] = "\(maxBlocks)"
        session.metadata["seed"] = "\(seed)"

        // 2) Resoudre le chemin local
        let directory: URL
        if let modelPath = modelPath {
            directory = URL(fileURLWithPath: modelPath)
        } else {
            var dir = Gemma4ModelCache.modelsDirectory
            for part in model.split(separator: "/") {
                dir = dir.appendingPathComponent(String(part))
            }
            directory = dir
        }
        guard FileManager.default.fileExists(atPath: directory.path) else {
            print("Erreur : modele non trouve a \(directory.path)")
            throw ExitCode.failure
        }

        // 3) Phase 1 — Chargement modele
        print("Profiling DiffusionGemma : \(model)")
        session.beginPhase("1. Model Loading", category: .modelLoad)
        let needVision = includeVision || image != nil
        let (diffModel, diffConfig) = try DiffusionGemmaLoader.load(
            from: directory, includeVision: needVision
        )
        session.endPhase("1. Model Loading", category: .modelLoad)
        session.metadata["weightsBytes"] = "\(MLX.GPU.activeMemory / (1024 * 1024)) MB GPU"

        // 4) Tokenizer
        let tokenizer = try await AutoTokenizer.from(modelFolder: directory)

        // 5) Phase 2 — Tokenization + image preprocessing
        session.beginPhase("2. Tokenization", category: .tokenization)
        let pixelValues: MLXArray?
        if let imagePath = image {
            guard needVision else {
                print("Erreur : --image necessite --include-vision")
                throw ExitCode.failure
            }
            session.beginPhase("2a. Image preprocessing", category: .textEncoding)
            pixelValues = try Gemma4ImageProcessor.processImage(url: URL(fileURLWithPath: imagePath))
            session.endPhase("2a. Image preprocessing", category: .textEncoding)
        } else {
            pixelValues = nil
        }

        let content = pixelValues != nil ? "<|image|>\n\(prompt)" : prompt
        let messages: [[String: String]] = [["role": "user", "content": content]]
        var tokenIds = try tokenizer.applyChatTemplate(messages: messages)

        if pixelValues != nil {
            let imageTokenId = diffConfig.imageTokenId
            let boi = diffConfig.boiTokenId
            let eoi = diffConfig.eoiTokenId
            let nSoft = diffConfig.visionSoftTokensPerImage
            var expanded: [Int] = []
            for tid in tokenIds {
                if tid == imageTokenId {
                    expanded.append(boi)
                    for _ in 0 ..< nSoft { expanded.append(imageTokenId) }
                    expanded.append(eoi)
                } else {
                    expanded.append(tid)
                }
            }
            tokenIds = expanded
        }
        let promptIds = MLXArray(tokenIds.map { Int32($0) }).reshaped(1, -1)
        session.metadata["promptTokenCount"] = "\(tokenIds.count)"
        session.endPhase("2. Tokenization", category: .tokenization)

        print("Prompt tokens : \(tokenIds.count)")

        // 6) Genconfig avec overrides
        var genConfig = (try? loadGenerationConfig(directory: directory)) ?? DiffusionGenerationConfig()
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
        session.metadata["tempSchedule"] = "\(genConfig.tMin) -> \(genConfig.tMax)"
        session.metadata["maxDenoisingSteps"] = "\(genConfig.maxDenoisingSteps)"

        // 7) Generation avec profiling
        // On replique manuellement la boucle du pipeline pour pouvoir poser les
        // begin/endPhase et recordStep aux bons endroits.
        session.beginPhase("3. Generation", category: .denoisingLoop)

        let sampler = EntropyBoundSampler(
            entropyBound: genConfig.entropyBound,
            vocabSize: diffConfig.textConfig.base.vocabSize,
            canvasLength: diffConfig.textConfig.canvasLength
        )
        let temperatureSchedule = LinearTemperatureSchedule(
            tMin: genConfig.tMin,
            tMax: genConfig.tMax,
            maxDenoisingSteps: genConfig.maxDenoisingSteps
        )
        let stopping = StableConfidentStopping(
            stabilityThreshold: genConfig.stabilityThreshold,
            confidenceThreshold: genConfig.confidenceThreshold
        )

        var key = MLXRandom.key(seed)
        var fullIds = promptIds
        var totalForwards = 0
        var canvasesUsed = 0
        let totalStepsExpected = maxBlocks * genConfig.maxDenoisingSteps
        var stepGlobalIdx = 0

        for canvasIdx in 0 ..< maxBlocks {
            // Phase 4 — Encoder forward
            session.beginPhase("4. Encoder forward (canvas \(canvasIdx))", category: .textEncoding)
            let encOut = diffModel.encodePrompt(promptIds: fullIds, pixelValues: pixelValues)
            _ = encOut.lastHiddenState
            // Force eval pour mesurer le temps reel
            for entry in encOut.kvCache.entries.prefix(1) {
                if let e = entry { e.keys.eval() }
            }
            session.endPhase("4. Encoder forward (canvas \(canvasIdx))", category: .textEncoding)

            // Phase 5 — Init canvas
            let split = MLXRandom.split(key: key, into: 2)
            var canvas = sampler.initializeCanvas(batchSize: 1, key: split[0])
            var argmaxCanvas = canvas
            var prevLogits: MLXArray? = nil
            var rngKey = split[1]
            key = split[1]
            stopping.reset()

            // Phase 6 — Denoising loop
            session.beginPhase("5. Denoising (canvas \(canvasIdx))", category: .denoisingLoop)
            for step in (1 ... genConfig.maxDenoisingSteps).reversed() {
                let stepStart = CFAbsoluteTimeGetCurrent()

                let logits = diffModel.denoiseStep(
                    canvasIds: canvas,
                    encoderCache: encOut.kvCache,
                    selfConditioningLogits: prevLogits,
                    decoderAttentionMask: nil
                )

                let scaled = temperatureSchedule.apply(logits, curStep: step)
                argmaxCanvas = MLX.argMax(scaled, axis: -1).asType(.int32)

                let splitS = MLXRandom.split(key: rngKey, into: 2)
                let denoiserCanvas = MLXRandom.categorical(scaled, axis: -1, key: splitS[0]).asType(.int32)
                rngKey = splitS[1]

                canvas = sampler.accept(
                    currentCanvas: canvas,
                    denoiserCanvas: denoiserCanvas,
                    logits: scaled
                )

                totalForwards += 1
                stepGlobalIdx += 1

                // Force eval pour mesure precise
                argmaxCanvas.eval()
                let stepDurationUs = UInt64((CFAbsoluteTimeGetCurrent() - stepStart) * 1_000_000)
                session.recordStep(
                    index: stepGlobalIdx,
                    total: totalStepsExpected,
                    durationUs: stepDurationUs,
                    category: .denoisingStep
                )

                let shouldStop = stopping.shouldStop(argmaxCanvas: argmaxCanvas, logits: scaled)
                if shouldStop.all().item(Bool.self) { break }

                let splitR = MLXRandom.split(key: rngKey, into: 2)
                canvas = sampler.renoise(acceptedCanvas: canvas, batchSize: 1, key: splitR[0])
                rngKey = splitR[1]
                prevLogits = scaled
            }
            session.endPhase("5. Denoising (canvas \(canvasIdx))", category: .denoisingLoop)

            // Commit canvas
            fullIds = concatenated([fullIds, argmaxCanvas], axis: -1)
            canvasesUsed += 1

            // EOS check rapide
            argmaxCanvas.eval()
            let arr = argmaxCanvas.asArray(Int32.self)
            if arr.contains(Int32(1)) || arr.contains(Int32(106)) { break }
        }
        session.endPhase("3. Generation", category: .denoisingLoop)

        session.metadata["totalForwards"] = "\(totalForwards)"
        session.metadata["canvasesUsed"] = "\(canvasesUsed)"
        session.metadata["generatedTokens"] = "\(fullIds.dim(1) - promptIds.dim(1))"

        // 8) Decode
        session.beginPhase("6. Decode", category: .vaeDecode)
        let promptLen = promptIds.dim(1)
        let generatedIds = fullIds[0..., promptLen ..< fullIds.dim(1)]
        generatedIds.eval()
        let tokens = generatedIds.asArray(Int32.self).map { Int($0) }
        let response = tokenizer.decode(tokens: tokens, skipSpecialTokens: true)
        session.endPhase("6. Decode", category: .vaeDecode)

        print("\nReponse (\(tokens.count) tokens, \(canvasesUsed) canvas, \(totalForwards) forwards):")
        print(response)
        print()

        // 9) Rapport + trace
        print(session.generateReport())

        if config.exportChromeTrace {
            let traceData = ChromeTraceExporter.export(session: session)
            let outputDir = output.map { URL(fileURLWithPath: $0) }
                ?? URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            let fileName = "diffgemma_profile_\(Int(Date().timeIntervalSince1970)).json"
            let traceURL = outputDir.appendingPathComponent(fileName)
            try traceData.write(to: traceURL)
            print("Chrome Trace exporte : \(traceURL.path)")
            print("Charger dans chrome://tracing ou ui.perfetto.dev")
        }

        print("GPU pic : \(MLX.GPU.peakMemory / (1024 * 1024)) Mo")
    }

    private func loadGenerationConfig(directory: URL) throws -> DiffusionGenerationConfig {
        let url = directory.appendingPathComponent("generation_config.json")
        guard FileManager.default.fileExists(atPath: url.path) else {
            return DiffusionGenerationConfig()
        }
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(DiffusionGenerationConfig.self, from: data)
    }
}
