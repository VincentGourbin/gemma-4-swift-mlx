// Boucle d'agent web : a chaque step,
//   1. screenshot WKWebView -> NSImage -> CGImage -> MLXArray pixels
//   2. E4B (multimodal AR, 4-bit) decide la prochaine action JSON
//   3. si action == click, DiffusionGemma fait le grounding (target -> coords)
//      via le meme prompt format que le bench ScreenSpot (79% global)
//   4. dispatcher execute l'action sur le browser (navigate/click/type/scroll/...)
//   5. observation = nouveau title + court extrait innerText pour l'historique
//
// Garde E4B (4-bit, ~5 Go) + Diffusion (bf16, ~50 Go) en RAM simultanement.
// Sur M3 Max 96 Go : OK ; le bench ne doit pas etre charge en parallele.

import AppKit
import Foundation
import Gemma4Swift
import MLX
import Tokenizers

@MainActor
final class WebAgentLoop: ObservableObject {

    enum LoadState: Equatable {
        case idle
        case loading(String)
        case ready
        case error(String)

        var label: String {
            switch self {
            case .idle: return "Aucun modele charge"
            case .loading(let m): return "Chargement \(m)"
            case .ready: return "E4B + Diffusion prets"
            case .error(let e): return "Erreur : \(e)"
            }
        }

        var isReady: Bool { if case .ready = self { return true }; return false }
        var isLoading: Bool { if case .loading = self { return true }; return false }
    }

    @Published var loadState: LoadState = .idle
    @Published var steps: [AgentStep] = []
    @Published var isRunning: Bool = false
    @Published var currentGoal: String = ""

    // Modeles
    private var e4bPipeline: Gemma4Pipeline?
    private var diffModel: DiffusionGemmaForBlockDiffusion?
    private var diffConfig: DiffusionGemmaConfig?
    private var diffGenConfig: DiffusionGenerationConfig?
    private var diffTokenizer: Tokenizer?

    // Configurable
    let e4bModelID = "mlx-community/gemma-4-e4b-it-4bit"
    let diffModelID = "google/diffusiongemma-26B-A4B-it"
    let maxSteps = 12

    private var e4bPath: URL {
        var p = Gemma4ModelCache.modelsDirectory
        for part in e4bModelID.split(separator: "/") {
            p = p.appendingPathComponent(String(part))
        }
        return p
    }
    private var diffPath: URL {
        var p = Gemma4ModelCache.modelsDirectory
        for part in diffModelID.split(separator: "/") {
            p = p.appendingPathComponent(String(part))
        }
        return p
    }

    // MARK: - Chargement

    func loadModels() async {
        guard !loadState.isReady, !loadState.isLoading else { return }
        loadState = .loading("E4B 4-bit…")
        do {
            // E4B en multimodal:true (necessite vision_tower + embed_vision)
            await Gemma4Registration.register(multimodal: true)
            let pipe = Gemma4Pipeline()
            try await pipe.load(from: e4bPath, multimodal: true)
            e4bPipeline = pipe

            loadState = .loading("DiffusionGemma bf16 (~50 Go)…")
            let (model, config) = try DiffusionGemmaLoader.load(
                from: diffPath, includeVision: true
            )
            diffModel = model
            diffConfig = config

            let genConfigURL = diffPath.appendingPathComponent("generation_config.json")
            if FileManager.default.fileExists(atPath: genConfigURL.path),
               let data = try? Data(contentsOf: genConfigURL),
               let parsed = try? JSONDecoder().decode(DiffusionGenerationConfig.self, from: data)
            {
                diffGenConfig = parsed
            } else {
                diffGenConfig = DiffusionGenerationConfig()
            }
            diffTokenizer = try await AutoTokenizer.from(modelFolder: diffPath)

            loadState = .ready
        } catch {
            loadState = .error(error.localizedDescription)
        }
    }

    func unloadModels() {
        e4bPipeline?.unload()
        e4bPipeline = nil
        diffModel = nil
        diffConfig = nil
        diffGenConfig = nil
        diffTokenizer = nil
        MLX.GPU.clearCache()
        loadState = .idle
    }

    // MARK: - Run

    func run(goal: String, browser: WebBrowserHostController) async {
        guard !isRunning else { return }
        guard loadState.isReady,
              let e4b = e4bPipeline,
              let diff = diffModel,
              let dconf = diffConfig,
              let dgen = diffGenConfig,
              let dtok = diffTokenizer
        else { return }

        isRunning = true
        defer { isRunning = false }
        steps = []
        currentGoal = goal

        // Diag : dump des screenshots envoyes a E4B pour inspection visuelle
        let diagDir = URL(fileURLWithPath: "/tmp/agent-run-\(Int(Date().timeIntervalSince1970))")
        try? FileManager.default.createDirectory(at: diagDir, withIntermediateDirectories: true)
        print("[Agent] Goal: \(goal)")
        print("[Agent] Diag dir: \(diagDir.path)")

        var history: [String] = []
        // Pour detecter une vraie boucle : 3 mêmes action shortLabels sans
        // changement d'URL/title -> on force un done avec ce qu'on a.
        var lastActionLabels: [String] = []
        var lastURL: String = ""

        for n in 1 ... maxSteps {
            print("[Agent] === Step \(n) ===")
            print("[Agent] URL: \(browser.currentURL) | Title: \(browser.pageTitle)")
            var step = AgentStep(n: n)
            steps.append(step)

            // 1. Screenshot
            guard let img = await browser.screenshot() else {
                step.error = "screenshot failed"
                replaceLast(step)
                break
            }
            step.screenshot = img
            replaceLast(step)
            // Dump du screenshot envoye a E4B (taille + path)
            let shotURL = diagDir.appendingPathComponent("step-\(n)-screenshot.png")
            if let tiff = img.tiffRepresentation,
               let rep = NSBitmapImageRep(data: tiff),
               let png = rep.representation(using: .png, properties: [:])
            {
                try? png.write(to: shotURL)
            }
            print("[Agent] Screenshot: \(Int(img.size.width))x\(Int(img.size.height)) -> \(shotURL.lastPathComponent)")

            // CGImage + MLXArray
            guard let cg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
                step.error = "cgImage conversion failed"
                replaceLast(step)
                break
            }
            let pixels: MLXArray
            do {
                pixels = try Gemma4ImageProcessor.processImage(cg)
            } catch {
                step.error = "processImage: \(error.localizedDescription)"
                replaceLast(step)
                break
            }

            // 2. E4B planning — collecte le stream entier
            let pageText = await browser.pageText(maxChars: 2_500)
            let planPrompt = AgentPrompts.e4bPlanning(
                goal: goal,
                currentURL: browser.currentURL,
                pageTitle: browser.pageTitle,
                history: history,
                pageText: pageText
            )
            print("[Agent] pageText \(pageText.count) chars (first 120: \(pageText.replacingOccurrences(of: "\n", with: " ").prefix(120)))")
            var planRaw = ""
            do {
                let stream = try e4b.chatStreamMultimodal(
                    prompt: planPrompt,
                    pixelValues: pixels,
                    temperature: 0.2,
                    maxTokens: 220
                )
                for try await piece in stream {
                    planRaw.append(piece)
                }
            } catch {
                step.error = "E4B planning: \(error.localizedDescription)"
                replaceLast(step)
                break
            }
            step.planRaw = planRaw
            step.thought = planRaw.replacingOccurrences(of: "\n", with: " ")
            replaceLast(step)
            print("[Agent] E4B raw: \(planRaw)")

            // Parse action
            guard let json = AgentPrompts.extractActionJSON(from: planRaw),
                  let action = AgentPrompts.parseAction(from: json)
            else {
                step.error = "Action JSON non parseable : \(planRaw.prefix(120))"
                replaceLast(step)
                print("[Agent] !! JSON unparseable")
                break
            }
            step.action = action
            replaceLast(step)
            print("[Agent] Action: \(action.shortLabel)")

            // Loop guard cote code : 3 memes actions de suite sans changement
            // d'URL -> on force un done synthese de l'historique.
            lastActionLabels.append(action.shortLabel)
            if lastActionLabels.count > 4 { lastActionLabels.removeFirst() }
            let sameURL = (browser.currentURL == lastURL)
            let stuck = lastActionLabels.count >= 3
                && lastActionLabels.suffix(3).allSatisfy { $0 == action.shortLabel }
                && sameURL
            if stuck, !(action.shortLabel.hasPrefix("done")) {
                print("[Agent] !! HARD LOOP — forcing done(summary)")
                let forced = AgentAction.done(summary: """
                Agent stopped after \(n) steps because the same action (\(action.shortLabel)) was \
                repeated 3+ times without page change. Likely cause: the page does not respond to \
                that action (e.g. typing without a focused input, or clicking a non-clickable element).
                Last URL: \(browser.currentURL). Last history line: \(history.last ?? "(none)").
                Last visible text snippet: \(pageText.replacingOccurrences(of: "\n", with: " ").prefix(400))
                """)
                step.action = forced
                replaceLast(step)
                history.append("forced-done: hard loop after 3 identical actions")
                break
            }
            lastURL = browser.currentURL

            // 3. Si click, grounding via Diffusion
            var coords: (x: Double, y: Double)? = nil
            if case .click(let target) = action {
                let imgW = Int(img.size.width)
                let imgH = Int(img.size.height)
                let groundPrompt = AgentPrompts.diffusionGrounding(
                    target: target, imageWidth: imgW, imageHeight: imgH
                )
                let groundStart = Date()
                do {
                    let messages: [[String: String]] = [["role": "user", "content": "<|image|>\n\(groundPrompt)"]]
                    var ids = try dtok.applyChatTemplate(messages: messages)
                    // Expansion <|image|> -> boi + image_token x 280 + eoi
                    let imgTokId = dconf.imageTokenId
                    let boi = dconf.boiTokenId
                    let eoi = dconf.eoiTokenId
                    let nImg = dconf.visionSoftTokensPerImage
                    var expanded: [Int] = []
                    for tid in ids {
                        if tid == imgTokId {
                            expanded.append(boi)
                            for _ in 0 ..< nImg { expanded.append(imgTokId) }
                            expanded.append(eoi)
                        } else {
                            expanded.append(tid)
                        }
                    }
                    ids = expanded

                    nonisolated(unsafe) let unsafeIds = MLXArray(ids.map { Int32($0) }).reshaped(1, -1)
                    nonisolated(unsafe) let unsafePixels: MLXArray? = pixels
                    nonisolated(unsafe) let unsafeModel = diff
                    nonisolated(unsafe) let unsafeTokenizer = dtok
                    let outText: String = await Task.detached {
                        let pipeline = DiffusionGemmaPipeline(model: unsafeModel, genConfig: dgen)
                        let result = await pipeline.generate(
                            promptIds: unsafeIds,
                            pixelValues: unsafePixels,
                            maxBlocks: 1,
                            seed: 0
                        )
                        let outIds = result.generatedIds.asArray(Int32.self).map { Int($0) }
                        return unsafeTokenizer.decode(tokens: outIds, skipSpecialTokens: true)
                    }.value
                    print("[Agent] Diffusion grounding raw: \(outText.replacingOccurrences(of: "\n", with: " ").prefix(200))")
                    if let (x, y) = AgentPrompts.parseClickCoords(from: outText) {
                        coords = (x, y)
                        step.groundingCoords = coords
                        step.groundingElapsed = Date().timeIntervalSince(groundStart)
                        replaceLast(step)
                        print(String(format: "[Agent] Grounding coords: (%.3f, %.3f)", x, y))
                    } else {
                        step.error = "Diffusion grounding non parseable : \(outText.prefix(120))"
                        replaceLast(step)
                        print("[Agent] !! Grounding unparseable")
                        break
                    }
                } catch {
                    step.error = "Diffusion grounding: \(error.localizedDescription)"
                    replaceLast(step)
                    break
                }
            }

            // 4. Dispatch action
            let dispatchResult = await dispatch(action: action, coords: coords, on: browser)

            // 5. Petite pause pour laisser la page charger
            try? await Task.sleep(nanoseconds: 800_000_000)

            // Observation : title + nouvel URL
            let observation = "URL: \(browser.currentURL) | Title: \(browser.pageTitle)"
            step.observation = observation
            replaceLast(step)

            // Historique pour le prochain step (inclut le resultat dispatcher)
            let historyLine: String
            switch action {
            case .click(let t):
                let coordStr = coords.map { String(format: " @ (%.2f, %.2f)", $0.x, $0.y) } ?? ""
                historyLine = "clicked '\(t)'\(coordStr) -> \(dispatchResult.prefix(60)) | \(observation)"
            case .type(let t):
                let effect = dispatchResult == "NO_INPUT_FOUND"
                    ? "⚠ no input was focused — typing had no effect, you must click an input first"
                    : "into \(dispatchResult.replacingOccurrences(of: "TYPED_INPUT:", with: ""))"
                historyLine = "typed '\(t)' \(effect)"
            case .navigate(let u):
                historyLine = "navigated to \(u) -> \(observation)"
            case .pressEnter:
                historyLine = "pressed Enter -> \(observation)"
            case .scroll(let d):
                historyLine = "scrolled \(d.rawValue)"
            case .done(let s):
                historyLine = "done: \(s.prefix(60))"
            }
            history.append(historyLine)

            // 6. Done ?
            if case .done = action {
                break
            }
        }
    }

    @discardableResult
    private func dispatch(action: AgentAction, coords: (x: Double, y: Double)?, on browser: WebBrowserHostController) async -> String {
        switch action {
        case .navigate(let url):
            browser.navigate(to: url)
            for _ in 0 ..< 20 where browser.isLoading {
                try? await Task.sleep(nanoseconds: 200_000_000)
            }
            return "navigated"
        case .click:
            if let c = coords {
                return (await browser.click(normalizedX: c.x, normalizedY: c.y)) ?? "click failed"
            }
            return "no coords"
        case .type(let text):
            let result = await browser.type(text: text) ?? "no result"
            print("[Agent] type() -> \(result)")
            return result
        case .pressEnter:
            await browser.pressEnter()
            return "enter sent"
        case .scroll(let dir):
            await browser.scroll(deltaY: dir == .up ? -600 : 600)
            return "scrolled"
        case .done:
            return "done"
        }
    }

    private func replaceLast(_ step: AgentStep) {
        guard !steps.isEmpty else { return }
        steps[steps.count - 1] = step
    }
}
