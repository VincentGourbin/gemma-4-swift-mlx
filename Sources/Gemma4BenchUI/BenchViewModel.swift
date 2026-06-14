// ViewModel central : gère le chargement séquentiel des modèles
// et l'exécution des 2 générations (AR + Diffusion).

import AppKit
import Foundation
import Gemma4Swift
import MLX
import SwiftUI
import Tokenizers

@MainActor
final class BenchViewModel: ObservableObject {

    // MARK: - État chargement

    enum LoadState: Equatable {
        case idle
        case loadingAR(String)
        case loadingDiffusion(String)
        case unloading(String)
        case arReady
        case diffusionReady
        case error(String)

        var label: String {
            switch self {
            case .idle: return "Aucun modele charge"
            case .loadingAR(let msg): return "Chargement AR : \(msg)"
            case .loadingDiffusion(let msg): return "Chargement Diffusion : \(msg)"
            case .unloading(let msg): return "Dechargement : \(msg)"
            case .arReady: return "AR pret (Diffusion non charge)"
            case .diffusionReady: return "Diffusion pret (AR non charge)"
            case .error(let e): return "Erreur : \(e)"
            }
        }

        var hasARLoaded: Bool { self == .arReady }
        var hasDiffusionLoaded: Bool { self == .diffusionReady }
        var isBusy: Bool {
            switch self {
            case .loadingAR, .loadingDiffusion, .unloading: return true
            default: return false
            }
        }
    }

    // MARK: - État panneaux

    enum PanelPhase: Equatable {
        case idle
        case loading(detail: String)
        case generating(progress: Double, detail: String)  // 0...1
        case done
        case error(String)

        var labelShort: String {
            switch self {
            case .idle: return "Inactif"
            case .loading: return "Chargement"
            case .generating: return "Generation"
            case .done: return "Termine"
            case .error: return "Erreur"
            }
        }
    }

    struct PanelState {
        var text: String = ""
        var tokensGenerated: Int = 0
        var elapsed: TimeInterval = 0
        var tokPerSec: Double { elapsed > 0 ? Double(tokensGenerated) / elapsed : 0 }
        var status: String = "En attente"
        var isRunning: Bool = false
        var currentStep: Int? = nil
        var totalSteps: Int = 0
        var phase: PanelPhase = .idle

        // Spécifique au pipeline Diffusion : séparation entre canvas committés
        // (textes figés) et canvas actif (en cours de denoising). Permet
        // d'éviter que canvas N+1 écrase visuellement le canvas N déjà commit.
        var committedCanvases: [String] = []
        var activeCanvasText: String = ""
        var activeCanvasIdx: Int = 0
        var maxCanvases: Int = 1
    }

    @Published var loadState: LoadState = .idle
    @Published var arPanel = PanelState()
    @Published var diffusionPanel = PanelState()

    /// True du moment où on clique un bouton jusqu'à la fin de la
    /// generation correspondante (incluant load + unload + run).
    /// Sert à bloquer l'AUTRE bouton de bout en bout.
    @Published var isPipelineActive: Bool = false

    @Published var prompt: String = "Why is the sky blue? Answer in 3 short paragraphs."
    @Published var maxTokens: Int = 256

    // MARK: - Parametres de generation (avec presets)

    enum Preset: String, CaseIterable, Identifiable {
        case deterministic = "Deterministe"
        case balanced = "Equilibre"
        case creative = "Creatif"
        case chaotic = "Chaotique"
        case custom = "Personnalise"

        var id: String { rawValue }

        var icon: String {
            switch self {
            case .deterministic: return "snowflake"
            case .balanced: return "scale.3d"
            case .creative: return "sparkles"
            case .chaotic: return "tornado"
            case .custom: return "slider.horizontal.3"
            }
        }

        var color: String {
            switch self {
            case .deterministic: return "cyan"
            case .balanced: return "green"
            case .creative: return "orange"
            case .chaotic: return "pink"
            case .custom: return "white"
            }
        }

        /// (temperatureAR, tMinDiff, tMaxDiff, seed, maxStepsDiff?)
        var params: (arTemp: Float, tMin: Float, tMax: Float, seed: UInt64, maxSteps: Int?) {
            switch self {
            case .deterministic:
                return (0.1, 0.4, 0.8, 0, nil)
            case .balanced:
                return (0.5, 0.5, 1.0, 42, nil)
            case .creative:
                return (0.8, 0.8, 1.5, 42, nil)
            case .chaotic:
                return (1.2, 1.0, 2.0, 99, 64)
            case .custom:
                return (0.5, 0.5, 1.0, 0, nil)  // pas appliquee
            }
        }

        var description: String {
            switch self {
            case .deterministic: return "Defaut du checkpoint, tres reproductible"
            case .balanced: return "Compromis qualite/diversite"
            case .creative: return "Plus de variation, encore coherent"
            case .chaotic: return "Diversite max, peut deraper"
            case .custom: return "Sliders manuels"
            }
        }
    }

    @Published var preset: Preset = .deterministic {
        didSet { if preset != .custom { applyPreset() } }
    }

    @Published var arTemperature: Float = 0.1
    @Published var diffTMin: Float = 0.4
    @Published var diffTMax: Float = 0.8
    @Published var diffSeed: UInt64 = 0
    @Published var diffMaxSteps: Int = 48

    private func applyPreset() {
        let p = preset.params
        arTemperature = p.arTemp
        diffTMin = p.tMin
        diffTMax = p.tMax
        diffSeed = p.seed
        diffMaxSteps = p.maxSteps ?? 48
    }

    // MARK: - Quantization mixed precision (Diffusion only)

    enum QuantPreset: String, CaseIterable, Identifiable {
        case none = "bf16"
        case `default` = "Default"
        case conservative = "Conservative"
        case aggressive = "Aggressive"

        var id: String { rawValue }

        var description: String {
            switch self {
            case .none: return "Pas de quantization, bf16 natif (~50 Go)"
            case .default: return "4 first + 4 last layers en 8-bit, reste en 4-bit (Q-DiT)"
            case .conservative: return "6+6 layers en 8-bit, reste en 4-bit (qualite max)"
            case .aggressive: return "2+2 layers en 8-bit, reste en 4-bit (RAM min ~15 Go)"
            }
        }

        var icon: String {
            switch self {
            case .none: return "circle"
            case .default: return "circle.lefthalf.filled"
            case .conservative: return "circle.fill"
            case .aggressive: return "bolt.fill"
            }
        }

        var color: String {
            switch self {
            case .none: return "gray"
            case .default: return "blue"
            case .conservative: return "cyan"
            case .aggressive: return "orange"
            }
        }

        var mixedConfig: DiffusionOnTheFlyQuantization.MixedPrecisionConfig? {
            switch self {
            case .none: return nil
            case .default: return .default
            case .conservative: return .conservative
            case .aggressive: return .aggressive
            }
        }
    }

    @Published var quantPreset: QuantPreset = .none {
        didSet {
            // Changement de preset : on doit recharger le modele pour reappliquer.
            // On flag l'etat pour signaler a l'utilisateur (status bar).
            if quantPreset != oldValue, loadState.hasDiffusionLoaded {
                diffusionPanel.status = "Quant change : prochain Lancer rechargera Diffusion"
                // Force unload : la prochaine generation rechargera.
                Task { await unloadAll() }
            }
        }
    }

    // MARK: - Image upload (Diffusion only)

    @Published var imageURL: URL? = nil

    func selectImage() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [.image]
        panel.message = "Selectionnez une image (JPEG, PNG, HEIC...)"
        if panel.runModal() == .OK, let url = panel.url {
            imageURL = url
        }
    }

    func clearImage() { imageURL = nil }

    // MARK: - Modèles

    private var arPipeline: Gemma4Pipeline?
    private var diffusionModel: DiffusionGemmaForBlockDiffusion?
    private var diffusionConfig: DiffusionGemmaConfig?
    private var diffusionGenConfig: DiffusionGenerationConfig?
    private var diffusionTokenizer: Tokenizer?

    // MARK: - Paths

    let arModelID = "mlx-community/gemma-4-26b-a4b-it-bf16"
    let diffusionModelID = "google/diffusiongemma-26B-A4B-it"

    var arPath: URL {
        var p = Gemma4ModelCache.modelsDirectory
        for part in arModelID.split(separator: "/") {
            p = p.appendingPathComponent(String(part))
        }
        return p
    }

    var diffusionPath: URL {
        var p = Gemma4ModelCache.modelsDirectory
        for part in diffusionModelID.split(separator: "/") {
            p = p.appendingPathComponent(String(part))
        }
        return p
    }

    // MARK: - Déchargement

    /// Décharge tout, libère la RAM / GPU cache.
    private func unloadAll() async {
        if arPipeline != nil {
            loadState = .unloading("AR…")
            arPipeline?.unload()
            arPipeline = nil
        }
        if diffusionModel != nil {
            loadState = .unloading("Diffusion…")
            diffusionModel = nil
            diffusionConfig = nil
            diffusionGenConfig = nil
            diffusionTokenizer = nil
            MLX.GPU.clearCache()
        }
        // Petit délai pour laisser ARC + GPU free les ressources
        try? await Task.sleep(nanoseconds: 300_000_000)
        loadState = .idle
    }

    // MARK: - Chargement individuel (un modèle à la fois)

    func loadAR() async {
        await unloadAll()
        loadState = .loadingAR("\(arModelID)…")
        arPanel.phase = .loading(detail: "Chargement \(arModelID) (~48 Go bf16 + vision_tower)…")
        arPanel.status = "Chargement…"
        do {
            // Multimodal: true — le 26B-A4B bf16 a un vision_config (~600 MB additionnels)
            // qui permet d'accepter une image en entree via chatStreamMultimodal.
            await Gemma4Registration.register(multimodal: true)
            let pipeline = Gemma4Pipeline()
            try await pipeline.load(from: arPath, multimodal: true)
            arPipeline = pipeline
            loadState = .arReady
            arPanel.phase = .idle
            arPanel.status = "Modele charge, pret (vision OK)"
        } catch {
            loadState = .error("AR : \(error.localizedDescription)")
            arPanel.phase = .error(error.localizedDescription)
            arPanel.status = "Erreur chargement"
        }
    }

    func loadDiffusion() async {
        await unloadAll()
        loadState = .loadingDiffusion("\(diffusionModelID)…")
        diffusionPanel.phase = .loading(detail: "Chargement \(diffusionModelID) (~48 Go bf16)…")
        diffusionPanel.status = "Chargement…"
        do {
            // includeVision: true car le checkpoint a vision_config != nil,
            // donc DiffusionGemmaEncoderModel instancie quand meme vision_tower
            // qui attend ses poids (~600 MB additionnels).
            let (model, config) = try DiffusionGemmaLoader.load(
                from: diffusionPath, includeVision: true
            )
            diffusionModel = model
            diffusionConfig = config

            // Appliquer la quantization mixed precision si demande (Q-DiT)
            if let mpConfig = quantPreset.mixedConfig {
                diffusionPanel.phase = .loading(detail: "Quantization \(quantPreset.rawValue)…")
                let stats = DiffusionOnTheFlyQuantization.applyMixedPrecision(to: model, config: mpConfig)
                print("Quantization \(quantPreset.rawValue): \(stats.quantizedHigh) en 8-bit, \(stats.quantizedLow) en 4-bit, \(stats.skipped.count) skipped")
            }

            let genConfigURL = diffusionPath.appendingPathComponent("generation_config.json")
            if FileManager.default.fileExists(atPath: genConfigURL.path),
               let data = try? Data(contentsOf: genConfigURL),
               let parsed = try? JSONDecoder().decode(DiffusionGenerationConfig.self, from: data)
            {
                diffusionGenConfig = parsed
            } else {
                diffusionGenConfig = DiffusionGenerationConfig()
            }

            diffusionTokenizer = try await AutoTokenizer.from(modelFolder: diffusionPath)
            loadState = .diffusionReady
            diffusionPanel.phase = .idle
            diffusionPanel.status = "Modele charge, pret"
        } catch {
            loadState = .error("Diffusion : \(error.localizedDescription)")
            diffusionPanel.phase = .error(error.localizedDescription)
            diffusionPanel.status = "Erreur chargement"
        }
    }

    // MARK: - Génération AR

    func runAR() async {
        guard let pipeline = arPipeline else { return }
        // Garde le label "chargement OK" un instant puis bascule generation
        arPanel.text = ""
        arPanel.tokensGenerated = 0
        arPanel.elapsed = 0
        arPanel.currentStep = nil
        arPanel.isRunning = true
        arPanel.phase = .generating(progress: 0, detail: "Token 0 / \(maxTokens)")
        arPanel.status = imageURL == nil
            ? "Generation AR (token-par-token)…"
            : "Generation AR multimodale (vision active)…"

        let start = Date()
        let targetTokens = maxTokens
        do {
            // Choisir le path : si une image est selectionnee, on bypass ChatSession
            // (qui ne supporte pas les pixelValues) via chatStreamMultimodal.
            let stream: AsyncThrowingStream<String, Error>
            if let url = imageURL {
                let pixels = try Gemma4ImageProcessor.processImage(url: url)
                stream = try pipeline.chatStreamMultimodal(
                    prompt: prompt,
                    pixelValues: pixels,
                    temperature: arTemperature,
                    maxTokens: maxTokens
                )
            } else {
                stream = try pipeline.chatStream(
                    prompt: prompt,
                    systemPrompt: nil,
                    temperature: arTemperature,
                    maxTokens: maxTokens
                )
            }
            for try await piece in stream {
                arPanel.text.append(piece)
                arPanel.tokensGenerated += 1
                arPanel.elapsed = Date().timeIntervalSince(start)
                let p = min(1.0, Double(arPanel.tokensGenerated) / Double(targetTokens))
                arPanel.phase = .generating(
                    progress: p,
                    detail: "Token \(arPanel.tokensGenerated) / \(targetTokens)"
                )
            }
            arPanel.status = "Fini"
            arPanel.phase = .done
        } catch {
            arPanel.status = "Erreur : \(error.localizedDescription)"
            arPanel.phase = .error(error.localizedDescription)
        }
        arPanel.isRunning = false
        arPanel.elapsed = Date().timeIntervalSince(start)
    }

    // MARK: - Génération Diffusion

    /// Event émis pendant le denoising. Texte déjà décodé pour éviter
    /// de traverser l'actor boundary avec un MLXArray non-Sendable.
    private struct DiffusionEvent: Sendable {
        enum Kind: Sendable { case step, canvas, done(totalSteps: Int, canvases: Int, totalTokens: Int) }
        let kind: Kind
        let canvasIdx: Int
        let step: Int
        let text: String
        let elapsed: TimeInterval
    }

    func runDiffusion() async {
        guard let model = diffusionModel,
              let config = diffusionConfig,
              let baseGenConfig = diffusionGenConfig,
              let tokenizer = diffusionTokenizer
        else { return }

        // Override avec les params UI
        let genConfig = DiffusionGenerationConfig(
            tMin: diffTMin,
            tMax: diffTMax,
            maxDenoisingSteps: diffMaxSteps,
            entropyBound: baseGenConfig.entropyBound,
            stabilityThreshold: baseGenConfig.stabilityThreshold,
            confidenceThreshold: baseGenConfig.confidenceThreshold,
            eosTokenIds: baseGenConfig.eosTokenIds,
            padTokenId: baseGenConfig.padTokenId
        )
        let captureSeed = diffSeed

        diffusionPanel.text = ""
        diffusionPanel.tokensGenerated = 0
        diffusionPanel.elapsed = 0
        diffusionPanel.currentStep = nil
        diffusionPanel.isRunning = true
        diffusionPanel.totalSteps = genConfig.maxDenoisingSteps
        diffusionPanel.phase = .generating(progress: 0, detail: "Denoising step \(genConfig.maxDenoisingSteps) / \(genConfig.maxDenoisingSteps)")
        diffusionPanel.status = "Generation Diffusion (denoising)…"
        diffusionPanel.committedCanvases = []
        diffusionPanel.activeCanvasText = ""
        diffusionPanel.activeCanvasIdx = 0
        diffusionPanel.maxCanvases = max(1, Int(ceil(Double(maxTokens) / Double(config.textConfig.canvasLength))))

        let canvasLength = config.textConfig.canvasLength
        let maxBlocks = max(1, Int(ceil(Double(maxTokens) / Double(canvasLength))))
        let promptCopy = prompt
        let totalSteps = diffusionPanel.totalSteps

        // Charger l'image si selectionnee (capture pixelValues + ids pour expansion)
        let pixelValues: MLXArray?
        let imageTokenId = config.imageTokenId
        let boiTokenId = config.boiTokenId
        let eoiTokenId = config.eoiTokenId
        let numImageSoftTokens = config.visionSoftTokensPerImage
        if let url = imageURL {
            do {
                pixelValues = try Gemma4ImageProcessor.processImage(url: url)
                diffusionPanel.status = "Image chargee (\(pixelValues!.shape))"
            } catch {
                diffusionPanel.phase = .error("Image : \(error.localizedDescription)")
                diffusionPanel.status = "Erreur image"
                diffusionPanel.isRunning = false
                return
            }
        } else {
            pixelValues = nil
        }
        let hasImage = pixelValues != nil

        // Stream d'events Sendable (text déjà décodé côté pipeline)
        let stream = AsyncStream<DiffusionEvent>(bufferingPolicy: .unbounded) { continuation in
            nonisolated(unsafe) let unsafeModel = model
            nonisolated(unsafe) let unsafeTokenizer = tokenizer
            nonisolated(unsafe) let unsafePixels = pixelValues

            let task = Task.detached {
                let pipeline = DiffusionGemmaPipeline(model: unsafeModel, genConfig: genConfig)

                let content = hasImage ? "<|image|>\n\(promptCopy)" : promptCopy
                let messages: [[String: String]] = [["role": "user", "content": content]]
                guard var tokenIds = try? unsafeTokenizer.applyChatTemplate(messages: messages) else {
                    continuation.finish()
                    return
                }
                // Expanser <|image|> en boi + image_token x N + eoi (cf DiffusionGemmaCommand)
                if hasImage {
                    var expanded: [Int] = []
                    for tid in tokenIds {
                        if tid == imageTokenId {
                            expanded.append(boiTokenId)
                            for _ in 0 ..< numImageSoftTokens { expanded.append(imageTokenId) }
                            expanded.append(eoiTokenId)
                        } else {
                            expanded.append(tid)
                        }
                    }
                    tokenIds = expanded
                }
                let promptIds = MLXArray(tokenIds.map { Int32($0) }).reshaped(1, -1)
                let startDate = Date()

                nonisolated(unsafe) let nonisolatedTokenizer = unsafeTokenizer

                let result = await pipeline.generate(
                    promptIds: promptIds,
                    pixelValues: unsafePixels,
                    maxBlocks: maxBlocks,
                    seed: captureSeed,
                    onCanvas: { @Sendable canvasIdx, canvas in
                        canvas.eval()
                        let tokens = canvas.asArray(Int32.self).map { Int($0) }
                        let text = nonisolatedTokenizer.decode(tokens: tokens, skipSpecialTokens: true)
                        continuation.yield(DiffusionEvent(
                            kind: .canvas,
                            canvasIdx: canvasIdx,
                            step: 0,
                            text: text,
                            elapsed: Date().timeIntervalSince(startDate)
                        ))
                    },
                    onStep: { @Sendable canvasIdx, step, argmax in
                        argmax.eval()
                        let tokens = argmax.asArray(Int32.self).map { Int($0) }
                        let text = nonisolatedTokenizer.decode(tokens: tokens, skipSpecialTokens: true)
                        continuation.yield(DiffusionEvent(
                            kind: .step,
                            canvasIdx: canvasIdx,
                            step: step,
                            text: text,
                            elapsed: Date().timeIntervalSince(startDate)
                        ))
                    }
                )
                continuation.yield(DiffusionEvent(
                    kind: .done(
                        totalSteps: result.totalDecoderSteps,
                        canvases: result.canvases,
                        totalTokens: result.generatedIds.dim(1)
                    ),
                    canvasIdx: result.canvases - 1,
                    step: 0,
                    text: "",
                    elapsed: Date().timeIntervalSince(startDate)
                ))
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }

        for await event in stream {
            switch event.kind {
            case .step:
                // Stocker dans activeCanvasText, pas text — évite l'écrasement
                // visuel des canvases déjà commit (bug avec maxTokens > 256).
                diffusionPanel.activeCanvasText = event.text
                diffusionPanel.activeCanvasIdx = event.canvasIdx
                diffusionPanel.currentStep = event.step
                diffusionPanel.elapsed = event.elapsed
                // Steps décroissent de totalSteps à 1 → progress = (totalSteps - step + 1) / totalSteps
                let stepsDone = totalSteps - event.step + 1
                let p = min(1.0, Double(stepsDone) / Double(totalSteps))
                diffusionPanel.phase = .generating(
                    progress: p,
                    detail: "Canvas \(event.canvasIdx + 1) / \(diffusionPanel.maxCanvases), denoising step \(event.step) / \(totalSteps)"
                )
                diffusionPanel.status = "Canvas \(event.canvasIdx), step \(event.step) / \(totalSteps)"
            case .canvas:
                // Commit : on fige le texte courant dans committedCanvases
                // et on remet à zéro l'activeCanvasText pour le prochain canvas.
                diffusionPanel.committedCanvases.append(event.text)
                diffusionPanel.activeCanvasText = ""
                diffusionPanel.status = "Canvas \(event.canvasIdx) commit"
            case .done(let steps, let canvases, let totalTokens):
                diffusionPanel.elapsed = event.elapsed
                diffusionPanel.tokensGenerated = totalTokens
                diffusionPanel.status = "Fini (\(steps) forwards, \(canvases) canvas)"
                diffusionPanel.isRunning = false
                diffusionPanel.currentStep = nil
                diffusionPanel.phase = .done
                // Le `text` complet = concat des committed (sans le dernier
                // doublon) pour les stats / export
                diffusionPanel.text = diffusionPanel.committedCanvases.joined(separator: "\n\n")
            }
        }
    }

    // MARK: - One-shot : charge le bon modèle (décharge l'autre) puis génère.

    /// Bouton "Lancer AR" : décharge la Diffusion si présente, charge l'AR si besoin, génère.
    func runARFull() async {
        guard !isPipelineActive else { return }
        isPipelineActive = true
        defer { isPipelineActive = false }
        if !loadState.hasARLoaded {
            await loadAR()
            guard loadState == .arReady else { return }
        }
        await runAR()
    }

    /// Bouton "Lancer Diffusion" : décharge l'AR si présent, charge la Diffusion si besoin, génère.
    func runDiffusionFull() async {
        guard !isPipelineActive else { return }
        isPipelineActive = true
        defer { isPipelineActive = false }
        if !loadState.hasDiffusionLoaded {
            await loadDiffusion()
            guard loadState == .diffusionReady else { return }
        }
        await runDiffusion()
    }
}
