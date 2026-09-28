// gemma4-cli bench-diffusion — instrument de mesure de DiffusionGemma (K-D8, audit diffusion §C.6)

import ArgumentParser
import Foundation
import Gemma4Swift
import MLX
import Tokenizers

/// Mesure le chemin de la bibliotheque : `DiffusionGemmaRegistration.load(from:profile:)`
/// (quantification a la volee comprise) puis `DiffusionGemmaPipeline.generate`. Une ligne
/// JSON par point, meme format que `bench` + champs diffusion. Les temps de pas sont lus
/// dans `onStep` (le pas est deja synchronise par le critere d'arret).
///
/// Charges (audit diffusion §C.4) : d1 texte (4 canvases), d2 une image (1 canvas),
/// d3 contexte long ≈ 2,5 k jetons, image + texte de page (2 canvases, exerce la fenetre
/// glissante).
struct BenchDiffusion: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "bench-diffusion",
        abstract: "Mesure DiffusionGemma : pas de debruitage, forwards, memoire (une ligne JSON par point)"
    )

    enum Workload: String, ExpressibleByArgument, CaseIterable { case d1, d2, d3 }

    @Option(name: .long, help: "Dossier du checkpoint bf16 officiel (google/diffusiongemma-26B-A4B-it)")
    var modelPath: String

    @Option(name: .long, help: "Profil de reference (a4bdiff/16bit-fast, a4bdiff/4bit-lean, …)")
    var reference: String = "a4bdiff/16bit-fast"

    @Option(name: .long, help: "Variante de quantification a la place de celle du profil : 4bit-sensitive8 (couches 4 bits, embeddings/tete et self_conditioning 8 bits), 4bit-mixed (couches 0-3 et 26-29 en 8 bits + sensibles en 8 bits, defaut des profils 4 bits), 4bit-uniform (tout en 4 bits, pour comparer)")
    var quantVariant: String?

    @Option(name: .long, help: "Charge : d1 (texte), d2 (image), d3 (contexte long)")
    var workload: Workload = .d1

    @Option(name: .long, help: "Image pour d2/d3")
    var image: String?

    @Option(name: .long, help: "Canvases max (defaut : d1 4, d2 1, d3 2)")
    var maxBlocks: Int?

    @Option(name: .long, help: "Secondes de repos avant chaque point")
    var cooldown: Int = 0

    @Option(name: .long, help: "Nombre de passes")
    var repeats: Int = 1

    @Option(name: .long, help: "Fichier JSONL ou ajouter les lignes")
    var out: String?

    @Option(name: .long, help: "Etiquette libre (variante A/B)")
    var label: String = ""

    @Flag(name: .long, help: "Ne pas faire la passe d'echauffement (1 canvas, non chronometree)")
    var noWarmup = false

    func run() async throws {
        guard var profile = DiffusionReferenceProfile.named(reference) else {
            throw ValidationError("profil inconnu : \(reference) (voir `gemma4-cli references --family a4bdiff`)")
        }
        switch quantVariant {
        case nil: break
        case "4bit-sensitive8":
            profile = profile.withQuantization(.mixed(.init(highPrecisionLayers: [], quantizeSensitiveAtHighPrecision: true)))
        case "4bit-mixed":
            profile = profile.withQuantization(.mixed(.default))
        case "4bit-uniform":
            profile = profile.withQuantization(.uniform(bits: 4, groupSize: 64))
        case let other?:
            throw ValidationError("variante inconnue : \(other)")
        }
        if workload != .d1 && image == nil {
            throw ValidationError("--image requis pour \(workload.rawValue)")
        }
        let url = URL(fileURLWithPath: modelPath)

        Memory.peakMemory = 0
        let loadStart = Date()
        let container = try await DiffusionGemmaRegistration.load(from: url, profile: profile)
        let loadSeconds = Date().timeIntervalSince(loadStart)

        var context = BenchContext.collect(modelURL: url, loadSeconds: loadSeconds)
        context.fields["profile"] = profile.qualifiedID
        if let quantVariant { context.fields["quant_variant"] = quantVariant }
        context.fields["profile_weights_match"] =
            url.lastPathComponent == DiffusionReferenceProfile.checkpointID.split(separator: "/").last.map(String.init)
        context.fields["workload"] = workload.rawValue
        context.fields["load_peak_mlx_mb"] = Memory.peakMemory / 1_048_576

        let pixels: MLXArray? = try image.map { path in
            let pixels = try Gemma4ImageProcessor.processImage(url: URL(fileURLWithPath: path))
            eval(pixels)
            return pixels
        }
        let usePixels = workload == .d1 ? nil : pixels
        let ids = try promptIds(container: container, withImage: usePixels != nil)
        let blocks = maxBlocks ?? (workload == .d1 ? 4 : workload == .d2 ? 1 : 2)

        if !noWarmup {
            _ = try await measure(container: container, ids: ids, pixels: usePixels, blocks: 1)
        }
        for pass in 1 ... max(1, repeats) {
            if cooldown > 0 { try await Task.sleep(for: .seconds(cooldown)) }
            var line = try await measure(container: container, ids: ids, pixels: usePixels, blocks: blocks)
            line.merge(context.fields) { current, _ in current }
            line["pass"] = pass
            line["label"] = label
            line["cooldown_s"] = cooldown
            try Bench.writeLine(line, to: out)
        }
    }

    /// Horodatage des pas, rempli depuis `onStep` (appele dans l'acteur du pipeline).
    private final class StepClock: @unchecked Sendable {
        var stamps: [Date] = []
    }

    private func measure(
        container: DiffusionGemmaContainer, ids: [Int], pixels: MLXArray?, blocks: Int
    ) async throws -> [String: Any] {
        let pipeline = container.makePipeline()
        let clock = StepClock()
        nonisolated(unsafe) let pixelsCapture = pixels
        let promptIds = MLXArray(ids.map { Int32($0) }).reshaped(1, -1)

        Memory.clearCache()
        Memory.peakMemory = 0
        let start = Date()
        let result = await pipeline.generate(
            promptIds: promptIds, pixelValues: pixelsCapture, maxBlocks: blocks, seed: 0,
            onStep: { _, _, _ in clock.stamps.append(Date()) })
        let total = Date().timeIntervalSince(start)
        let snapshot = Memory.snapshot()

        let intervals = zip(clock.stamps.dropFirst(), clock.stamps).map { $0.timeIntervalSince($1) * 1000 }.sorted()
        let canvasLength = container.config.textConfig.canvasLength
        let tokens = result.canvases * canvasLength
        let footprint = BenchContext.physFootprint()
        let stop: String
        switch result.stopReason {
        case .completed: stop = result.canvases < blocks ? "eos" : "max_blocks"
        case .cancelled: stop = "cancelled"
        case .trainingInProgress: stop = "training_in_progress"
        case .invalidInput(let message): stop = "invalid_input: \(message)"
        }
        // Debut de la reponse : detecte une generation qui ignore l'image (vision
        // dechargee) ou qui deraille, sans relire les tokens a la main.
        let generated = result.generatedIds.reshaped(-1).asArray(Int32.self).map(Int.init)
        let text = container.tokenizer.decode(tokens: generated, skipSpecialTokens: true)
        return [
            "output_head": String(text.trimmingCharacters(in: .whitespacesAndNewlines).prefix(160)),
            "mode": "diffusion",
            "prompt_tokens": ids.count,
            "canvases": result.canvases,
            "forwards": result.totalDecoderSteps,
            "forwards_per_canvas": Bench.round(Double(result.totalDecoderSteps) / Double(max(1, result.canvases))),
            "generated_tokens": tokens,
            "total_ms": Bench.round(total * 1000),
            "first_step_ms": Bench.round(((clock.stamps.first ?? start).timeIntervalSince(start)) * 1000),
            "step_ms_median": Bench.round(Bench.percentile(intervals, 0.5)),
            "step_ms_p90": Bench.round(Bench.percentile(intervals, 0.9)),
            "tokens_per_s": Bench.round(Double(tokens) / max(total, 1e-9)),
            "stop": stop,
            "peak_mlx_mb": snapshot.peakMemory / 1_048_576,
            "active_mlx_mb": snapshot.activeMemory / 1_048_576,
            "cache_mlx_mb": snapshot.cacheMemory / 1_048_576,
            "phys_footprint_mb": footprint.current,
            "phys_footprint_peak_mb": footprint.peak,
        ]
    }

    /// Gabarit de chat + expansion `<|image|>` -> boi + image_token x N + eoi (comme `diffusion`).
    private func promptIds(container: DiffusionGemmaContainer, withImage: Bool) throws -> [Int] {
        let text: String
        switch workload {
        case .d1:
            text = "Why is the sky blue? Answer in 4 paragraphs with examples and analogies."
        case .d2:
            text = "CLICK: the main search field. Answer with the click coordinates."
        case .d3:
            text = "Here is the text of the page shown in the screenshot:\n"
                + String(repeating: Bench.defaultFiller + " ", count: 3).prefix(2_000)
                + "\n\nSummarize the page and say where to click to open the first result."
        }
        let content = withImage ? "<|image|>\n\(text)" : text
        var tokens = try container.tokenizer.applyChatTemplate(messages: [["role": "user", "content": content]])
        guard withImage else { return tokens }
        let config = container.config
        var expanded: [Int] = []
        for id in tokens {
            if id == config.imageTokenId {
                expanded.append(config.boiTokenId)
                expanded.append(contentsOf: Array(repeating: config.imageTokenId, count: config.visionSoftTokensPerImage))
                expanded.append(config.eoiTokenId)
            } else {
                expanded.append(id)
            }
        }
        tokens = expanded
        return tokens
    }
}
