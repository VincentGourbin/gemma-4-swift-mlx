// ViewModel central : gère le chargement séquentiel des modèles
// et l'exécution des 2 générations (AR + Diffusion).

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
        case ready
        case error(String)

        var label: String {
            switch self {
            case .idle: return "Pas de modele charge"
            case .loadingAR(let msg): return "Chargement AR : \(msg)"
            case .loadingDiffusion(let msg): return "Chargement Diffusion : \(msg)"
            case .ready: return "Modeles prets"
            case .error(let e): return "Erreur : \(e)"
            }
        }
    }

    // MARK: - État panneaux

    struct PanelState {
        var text: String = ""
        var tokensGenerated: Int = 0
        var elapsed: TimeInterval = 0
        var tokPerSec: Double { elapsed > 0 ? Double(tokensGenerated) / elapsed : 0 }
        var status: String = "En attente"
        var isRunning: Bool = false
        var currentStep: Int? = nil
        var totalSteps: Int = 0
    }

    @Published var loadState: LoadState = .idle
    @Published var arPanel = PanelState()
    @Published var diffusionPanel = PanelState()

    @Published var prompt: String = "Why is the sky blue? Answer in 3 short paragraphs."
    @Published var maxTokens: Int = 256
    @Published var temperature: Float = 0.3

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

    // MARK: - Chargement séquentiel

    func loadModels() async {
        // 1) AR
        loadState = .loadingAR("\(arModelID)…")
        do {
            await Gemma4Registration.register(multimodal: false)
            let pipeline = Gemma4Pipeline()
            try await pipeline.load(from: arPath, multimodal: false)
            arPipeline = pipeline
        } catch {
            loadState = .error("AR : \(error.localizedDescription)")
            return
        }

        // 2) Diffusion
        loadState = .loadingDiffusion("\(diffusionModelID)…")
        do {
            let (model, config) = try DiffusionGemmaLoader.load(
                from: diffusionPath, includeVision: false
            )
            diffusionModel = model
            diffusionConfig = config

            // generation_config.json optionnel
            let genConfigURL = diffusionPath.appendingPathComponent("generation_config.json")
            if FileManager.default.fileExists(atPath: genConfigURL.path),
               let data = try? Data(contentsOf: genConfigURL),
               let parsed = try? JSONDecoder().decode(DiffusionGenerationConfig.self, from: data)
            {
                diffusionGenConfig = parsed
            } else {
                diffusionGenConfig = DiffusionGenerationConfig()
            }

            // Tokenizer Diffusion (l'AR a le sien dans le ModelContainer)
            diffusionTokenizer = try await AutoTokenizer.from(modelFolder: diffusionPath)
        } catch {
            loadState = .error("Diffusion : \(error.localizedDescription)")
            return
        }

        loadState = .ready
    }

    // MARK: - Génération AR

    func runAR() async {
        guard let pipeline = arPipeline else { return }
        arPanel = PanelState()
        arPanel.isRunning = true
        arPanel.status = "Generation AR (token-par-token)…"

        let start = Date()
        do {
            let stream = try pipeline.chatStream(
                prompt: prompt,
                systemPrompt: nil,
                temperature: temperature,
                maxTokens: maxTokens
            )
            for try await piece in stream {
                arPanel.text.append(piece)
                arPanel.tokensGenerated += 1
                arPanel.elapsed = Date().timeIntervalSince(start)
            }
            arPanel.status = "Fini"
        } catch {
            arPanel.status = "Erreur : \(error.localizedDescription)"
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
              let genConfig = diffusionGenConfig,
              let tokenizer = diffusionTokenizer
        else { return }

        diffusionPanel = PanelState()
        diffusionPanel.isRunning = true
        diffusionPanel.status = "Generation Diffusion (denoising)…"
        diffusionPanel.totalSteps = genConfig.maxDenoisingSteps

        let canvasLength = config.textConfig.canvasLength
        let maxBlocks = max(1, Int(ceil(Double(maxTokens) / Double(canvasLength))))
        let promptCopy = prompt
        let totalSteps = diffusionPanel.totalSteps

        // Stream d'events Sendable (text déjà décodé côté pipeline)
        let stream = AsyncStream<DiffusionEvent>(bufferingPolicy: .unbounded) { continuation in
            nonisolated(unsafe) let unsafeModel = model
            nonisolated(unsafe) let unsafeTokenizer = tokenizer

            let task = Task.detached {
                let pipeline = DiffusionGemmaPipeline(model: unsafeModel, genConfig: genConfig)

                let messages: [[String: String]] = [["role": "user", "content": promptCopy]]
                guard let tokenIds = try? unsafeTokenizer.applyChatTemplate(messages: messages) else {
                    continuation.finish()
                    return
                }
                let promptIds = MLXArray(tokenIds.map { Int32($0) }).reshaped(1, -1)
                let startDate = Date()

                nonisolated(unsafe) let nonisolatedTokenizer = unsafeTokenizer

                let result = await pipeline.generate(
                    promptIds: promptIds,
                    maxBlocks: maxBlocks,
                    seed: 0,
                    onCanvas: { @Sendable canvasIdx, canvas in
                        canvas.eval()
                        let tokens = canvas.asArray(Int32.self).map { Int($0) }
                        let text = nonisolatedTokenizer.decode(tokens: tokens)
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
                        let text = nonisolatedTokenizer.decode(tokens: tokens)
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
                diffusionPanel.text = event.text
                diffusionPanel.currentStep = event.step
                diffusionPanel.elapsed = event.elapsed
                diffusionPanel.status = "Canvas \(event.canvasIdx), step \(event.step) / \(totalSteps)"
            case .canvas:
                diffusionPanel.status = "Canvas \(event.canvasIdx) commit"
            case .done(let steps, let canvases, let totalTokens):
                diffusionPanel.elapsed = event.elapsed
                diffusionPanel.tokensGenerated = totalTokens
                diffusionPanel.status = "Fini (\(steps) forwards, \(canvases) canvas)"
                diffusionPanel.isRunning = false
                diffusionPanel.currentStep = nil
            }
        }
    }

    // MARK: - Race (les 2 en séquence)

    func runRace() async {
        await runAR()
        // Pause pour laisser l'oeil respirer
        try? await Task.sleep(nanoseconds: 1_500_000_000)
        await runDiffusion()
    }
}
