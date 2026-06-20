// Pipeline agent step-by-step pour piloter une app iOS dans le Simulator.
// Reprend le pattern du web Agent (AgentStepViewModel) mais avec :
//   - screenshot via simctl
//   - actions iOS : tap, swipe (up/down/left/right), type (texte), done
//   - pas d'URL : on suit la page courante de l'app
//   - grille de coords overlay reutilisee

import AppKit
import Foundation
import Gemma4Swift
import MLX
import Tokenizers

@MainActor
final class IOSAgentStepViewModel: ObservableObject {

    enum StepState: String, Sendable {
        case proposing, awaiting, validated, rejected, errored
    }

    /// Direction du scroll dans le sens de l'EFFET VISIBLE (pas du doigt) :
    ///   - .down : on veut voir le contenu plus bas dans la page (geste doigt vers le haut)
    ///   - .up   : on veut voir le contenu plus haut (geste doigt vers le bas)
    ///   - .right/.left : pareil pour l'horizontal
    /// Ce vocabulaire colle a celui de l'utilisateur ("scroller vers le bas")
    /// et evite la confusion de la convention swipe.
    enum ScrollDirection: String, Sendable, CaseIterable {
        case down, up, right, left
    }

    enum StepAction: Sendable, Equatable {
        case tap(x: Double, y: Double)
        case scroll(ScrollDirection)
        case type(text: String, pressEnter: Bool)
        case tapAndType(x: Double, y: Double, text: String, pressEnter: Bool)
        /// Chaine 2 taps en une seule generation. Utile pour les patterns
        /// "choisir une option puis valider" qui s'enchainent toujours sans
        /// information nouvelle entre les 2.
        case tapThenTap(x1: Double, y1: Double, x2: Double, y2: Double)
        case done(summary: String)

        var kind: String {
            switch self {
            case .tap: return "tap"
            case .scroll(let d): return "scroll_\(d.rawValue)"
            case .type(_, let e): return e ? "type+enter" : "type"
            case .tapAndType(_, _, _, let e): return e ? "tap+type+enter" : "tap+type"
            case .tapThenTap: return "tap+tap"
            case .done: return "done"
            }
        }
    }

    struct StepCapture: Identifiable, Sendable {
        let id = UUID()
        let n: Int
        var deviceLabel: String
        var screenshotBefore: NSImage
        var screenshotGrid: NSImage? = nil
        var screenshotAfter: NSImage? = nil
        var coordsProposed: (x: Double, y: Double)? = nil
        var actionProposed: StepAction? = nil
        var reason: String = ""
        var notes: String = ""
        var rawOutput: String = ""
        var elapsedDiffusion: TimeInterval? = nil
        var diffusionForwards: Int? = nil
        var state: StepState = .proposing
        var error: String? = nil
    }

    @Published var steps: [StepCapture] = []
    @Published var goal: String = "Ouvre l'app Réglages puis va dans la section Général."
    /// Instructions specifiques a l'application cible. Injectees dans le prompt
    /// systeme. Permet d'expliquer le flow particulier d'une app (quiz, sondage,
    /// formulaire multi-pages...) que le modele ne peut pas deviner du seul
    /// screenshot. Garde vide pour rester generique.
    @Published var appContext: String = ""
    @Published var selectedStepIndex: Int? = nil
    @Published var isBusy: Bool = false
    @Published var useCoordGrid: Bool = true
    /// Si actif, on envoie au modele une capture composite (haut + separateur
    /// jaune + bas) representant toute la page. Permet au modele de voir un
    /// bouton qui serait sous le fold (Valider, Suivant...) sans avoir besoin
    /// de scroller. Le tap est ensuite re-mappe selon que les coords proposees
    /// sont dans la zone HAUT (y<sepY) ou la zone BAS (y>sepY).
    ///
    /// EXPERIMENTAL : desactive par defaut. Le modele de diffusion s'embrouille
    /// avec la bande jaune et le mapping coords. On preferera le screenshot
    /// simple du viewport courant et laisser le modele proposer scroll_down
    /// explicitement si necessaire.
    @Published var useFullPageCapture: Bool = false
    /// Position normalisee du separateur dans la derniere capture composite,
    /// utilisee pour re-mapper les taps du modele.
    private var lastSeparatorY: Double = 0.5
    @Published var autoPlay: Bool = false
    @Published var maxAutoSteps: Int = 8
    @Published var stopRequested: Bool = false

    private var attemptSeed: UInt64 = 0

    /// Dossier de logs pour la session courante. Cree au 1er step et reutilise
    /// pour les steps suivants. Permet a Claude Code de lire chaque generation
    /// (prompt envoye + sortie brute du modele + action parsee + screenshot)
    /// pour debugger le comportement.
    @Published private(set) var sessionLogDir: URL? = nil

    func canPropose(host: IOSSimulatorHostController, registry: ModelRegistry) -> Bool {
        registry.isDiffusionLoaded && host.connection.isReadyForActions
            && !isBusy && lastStepResolved && !sessionFinished
    }
    var lastStepResolved: Bool {
        guard let last = steps.last else { return true }
        return last.state == .validated || last.state == .rejected || last.state == .errored
    }
    var sessionFinished: Bool {
        for s in steps where s.state == .validated {
            if case .done = s.actionProposed { return true }
        }
        return false
    }
    var finalSummary: String? {
        for s in steps where s.state == .validated {
            if case .done(let summary) = s.actionProposed { return summary }
        }
        return nil
    }
    var currentStep: StepCapture? {
        if let idx = selectedStepIndex, idx < steps.count { return steps[idx] }
        return steps.last
    }

    func reset() {
        steps = []
        selectedStepIndex = nil
        attemptSeed = 0
        stopRequested = false
        sessionLogDir = nil
    }

    private func ensureLogDir() -> URL {
        if let d = sessionLogDir { return d }
        let ts = Int(Date().timeIntervalSince1970)
        let dir = URL(fileURLWithPath: "/tmp/ios-agent-runs/run-\(ts)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        // Manifest avec goal + appContext
        let manifest = """
        # iOS Agent run — \(Date())
        Goal:
        \(goal)

        App context:
        \(appContext)
        """
        try? manifest.write(to: dir.appendingPathComponent("manifest.md"), atomically: true, encoding: .utf8)
        sessionLogDir = dir
        print("[iOSAgent] session log dir: \(dir.path)")
        return dir
    }

    private func writeLog(_ relativePath: String, content: String) {
        let dir = ensureLogDir()
        try? content.write(to: dir.appendingPathComponent(relativePath), atomically: true, encoding: .utf8)
    }
    private func writePNG(_ relativePath: String, image: NSImage) {
        let dir = ensureLogDir()
        if let tiff = image.tiffRepresentation,
           let rep = NSBitmapImageRep(data: tiff),
           let png = rep.representation(using: .png, properties: [:])
        {
            try? png.write(to: dir.appendingPathComponent(relativePath))
        }
    }
    func requestStop() {
        stopRequested = true
        autoPlay = false
    }

    // MARK: - Propose

    func proposeNextStep(host: IOSSimulatorHostController, registry: ModelRegistry) async {
        guard !stopRequested else { return }
        guard canPropose(host: host, registry: registry) else { return }
        guard let model = registry.diffModel,
              let dconf = registry.diffConfig,
              let dgen = registry.diffGenConfig,
              let dtok = registry.diffTokenizer
        else { return }

        isBusy = true
        defer { isBusy = false }

        let shotBefore: NSImage
        if useFullPageCapture {
            let (composite, sepY) = await host.fullPageCapture()
            guard let composite = composite else { return }
            shotBefore = composite
            lastSeparatorY = sepY
        } else {
            guard let shot = await host.screenshot() else { return }
            shotBefore = shot
            lastSeparatorY = 1.0
        }
        let n = steps.count + 1
        var step = StepCapture(
            n: n,
            deviceLabel: { if case .running(let nm, _) = host.connection { return nm }; return "iOS" }(),
            screenshotBefore: shotBefore,
            state: .proposing
        )
        steps.append(step)
        selectedStepIndex = steps.count - 1

        let imgW = Int(shotBefore.size.width)
        let imgH = Int(shotBefore.size.height)
        let accumulatedNotes = steps.dropLast().compactMap { $0.notes.isEmpty ? nil : $0.notes }
        // Description courte des 3 dernieres actions validees — sert a casser
        // les boucles type "tap-tap-tap au meme endroit" en rendant explicite
        // au modele ce qu'il vient de faire.
        let recentActions: [String] = steps.dropLast().reversed().prefix(4)
            .filter { $0.state == .validated }
            .compactMap { s -> String? in
                guard let a = s.actionProposed else { return nil }
                switch a {
                case .tap(let x, let y):
                    return String(format: "step %d: tap at (%.2f, %.2f)", s.n, x, y)
                case .scroll(let dir):
                    return "step \(s.n): scroll_\(dir.rawValue)"
                case .type(let t, _):
                    return "step \(s.n): typed \"\(t.prefix(30))\""
                case .tapAndType(let x, let y, let t, _):
                    return String(format: "step %d: tap (%.2f, %.2f) + typed \"%@\"", s.n, x, y, String(t.prefix(30)))
                case .tapThenTap(let x1, let y1, let x2, let y2):
                    return String(format: "step %d: tap_then_tap (%.2f, %.2f) -> (%.2f, %.2f)", s.n, x1, y1, x2, y2)
                case .done:
                    return nil
                }
            }
        let prompt = makePrompt(goal: goal, step: n, device: step.deviceLabel,
                                imageW: imgW, imageH: imgH,
                                withGrid: useCoordGrid,
                                accumulatedNotes: accumulatedNotes,
                                appContext: appContext,
                                recentActions: Array(recentActions.reversed()))

        // Image envoyee au modele : grille + overlay du dernier tap pour
        // feedback visuel ("tu as tape ICI au step precedent, ajuste si rate").
        var imageForModel: NSImage = shotBefore
        if useCoordGrid {
            imageForModel = overlayCoordGrid(on: imageForModel)
        }
        // Cherche le dernier tap valide pour annoter
        if let lastTapCoords = lastValidatedTapCoords() {
            imageForModel = overlayLastTapMarker(on: imageForModel, nx: lastTapCoords.x, ny: lastTapCoords.y)
        }
        step.screenshotGrid = imageForModel
        replaceLast(step)

        guard let cg = imageForModel.cgImage(forProposedRect: nil, context: nil, hints: nil),
              let pixels = try? Gemma4ImageProcessor.processImage(cg)
        else {
            step.state = .errored
            step.error = "screenshot -> pixels failed"
            replaceLast(step)
            return
        }

        var lastRawOut = ""
        var lastForwards: Int? = nil
        var totalElapsed: TimeInterval = 0
        let start = Date()
        let seed = attemptSeed
        do {
            let messages: [[String: String]] = [["role": "user", "content": "<|image|>\n\(prompt)"]]
            var ids = try dtok.applyChatTemplate(messages: messages)
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
            nonisolated(unsafe) let unsafeModel = model
            nonisolated(unsafe) let unsafeTokenizer = dtok

            var parsedAction: StepAction? = nil
            var parsedNotes = ""
            var parsedReason = ""
            for attempt in 0 ..< 2 {
                let attemptSeedLocal = seed &+ UInt64(attempt)
                let out: (text: String, forwards: Int) = await Task.detached {
                    let pipeline = DiffusionGemmaPipeline(model: unsafeModel, genConfig: dgen)
                    let r = await pipeline.generate(
                        promptIds: unsafeIds, pixelValues: unsafePixels,
                        maxBlocks: 1, seed: attemptSeedLocal
                    )
                    let outIds = r.generatedIds.asArray(Int32.self).map { Int($0) }
                    return (unsafeTokenizer.decode(tokens: outIds, skipSpecialTokens: true), r.totalDecoderSteps)
                }.value
                lastRawOut = out.text
                lastForwards = out.forwards
                totalElapsed = Date().timeIntervalSince(start)
                print("[iOSAgent] step \(n) attempt \(attempt + 1) raw: \(out.text.replacingOccurrences(of: "\n", with: " | "))")
                let parsed = parseProposal(out.text)
                parsedNotes = parsed.notes
                parsedReason = parsed.reason
                parsedAction = parsed.action
                if parsedAction != nil { break }
            }

            step.elapsedDiffusion = totalElapsed
            step.diffusionForwards = lastForwards
            step.rawOutput = lastRawOut
            step.notes = parsedNotes
            step.reason = parsedReason.isEmpty ? "(pas de raison)" : parsedReason
            step.actionProposed = parsedAction
            switch parsedAction {
            case .tap(let x, let y), .tapAndType(let x, let y, _, _):
                step.coordsProposed = (x, y)
            case .tapThenTap(let x1, let y1, _, _):
                // Crosshair sur le 1er tap. Le 2e tap est implicite et affiche
                // dans le badge mais pas dans le crosshair principal.
                step.coordsProposed = (x1, y1)
            default: break
            }
            step.state = .awaiting
            replaceLast(step)
            // Log de la generation (prompt + raw + parsed + screenshot)
            writeLog("step-\(n)-prompt.md", content: """
            # Step \(n) — prompt envoyé au modèle

            URL/Device: \(step.deviceLabel)
            Image: \(imgW)x\(imgH), grid=\(useCoordGrid)
            Diffusion : \(String(format: "%.2f", totalElapsed))s, \(lastForwards ?? 0) forwards

            ---

            \(prompt)
            """)
            writeLog("step-\(n)-raw.txt", content: lastRawOut)
            let parsedDump: String = {
                guard let a = parsedAction else { return "(no action parsed)" }
                return "kind: \(a.kind)\nnotes: \(parsedNotes)\nreason: \(parsedReason)\nfull action: \(a)"
            }()
            writeLog("step-\(n)-parsed.txt", content: parsedDump)
            writePNG("step-\(n)-screenshot.png", image: shotBefore)
            if let g = step.screenshotGrid { writePNG("step-\(n)-screenshot-grid.png", image: g) }
        } catch {
            step.state = .errored
            step.error = "Diffusion: \(error.localizedDescription)"
            replaceLast(step)
        }
    }

    // MARK: - Dispatch

    func validateAndExecute(host: IOSSimulatorHostController, registry: ModelRegistry, chainNext: Bool = true) async {
        guard var step = steps.last, step.state == .awaiting, let action = step.actionProposed else { return }
        isBusy = true

        switch action {
        case .tap(let x, let y):
            let nowScrolled = await dispatchTapWithCompositeRemap(host: host, x: x, y: y, isScrolledDown: false)
            try? await Task.sleep(nanoseconds: 800_000_000)
            await scrollBackToTopIfNeeded(host: host, isScrolledDown: nowScrolled)
            if useFullPageCapture {
                let (c, _) = await host.fullPageCapture()
                step.screenshotAfter = c
            } else {
                step.screenshotAfter = await host.screenshot()
            }

        case .scroll(let dir):
            // dir = effet visible. Le geste du doigt est l'INVERSE :
            //   scroll_down (voir le bas)   → doigt du bas (0.75) vers le haut (0.25)
            //   scroll_up   (voir le haut)  → doigt du haut (0.25) vers le bas (0.75)
            let (fx, fy, tx, ty): (Double, Double, Double, Double) = {
                switch dir {
                case .down:  return (0.5, 0.75, 0.5, 0.25)
                case .up:    return (0.5, 0.25, 0.5, 0.75)
                case .right: return (0.75, 0.5, 0.25, 0.5)
                case .left:  return (0.25, 0.5, 0.75, 0.5)
                }
            }()
            await host.swipe(fromX: fx, fromY: fy, toX: tx, toY: ty)
            try? await Task.sleep(nanoseconds: 600_000_000)
            step.screenshotAfter = await host.screenshot()

        case .type(let text, let pressEnter):
            await host.type(text: text, pressEnter: pressEnter)
            try? await Task.sleep(nanoseconds: 1_000_000_000)
            step.screenshotAfter = await host.screenshot()

        case .tapThenTap(let x1, let y1, let x2, let y2):
            // Execute 2 taps en sequence. On track l'etat de scroll pour eviter
            // de double-scroll quand les 2 taps sont en zone BAS.
            print(String(format: "[iOSAgent] tap_then_tap : (%.2f,%.2f) -> (%.2f,%.2f), sep=%.2f",
                         x1, y1, x2, y2, lastSeparatorY))
            let scrolledAfter1 = await dispatchTapWithCompositeRemap(host: host, x: x1, y: y1, isScrolledDown: false)
            try? await Task.sleep(nanoseconds: 500_000_000)
            let scrolledAfter2 = await dispatchTapWithCompositeRemap(host: host, x: x2, y: y2, isScrolledDown: scrolledAfter1)
            try? await Task.sleep(nanoseconds: 900_000_000)
            await scrollBackToTopIfNeeded(host: host, isScrolledDown: scrolledAfter2)
            if useFullPageCapture {
                let (c, _) = await host.fullPageCapture()
                step.screenshotAfter = c
            } else {
                step.screenshotAfter = await host.screenshot()
            }

        case .tapAndType(let x, let y, let text, let pressEnter):
            await host.tap(normalizedX: x, normalizedY: y)
            try? await Task.sleep(nanoseconds: 500_000_000)
            await host.type(text: text, pressEnter: pressEnter)
            try? await Task.sleep(nanoseconds: 1_000_000_000)
            step.screenshotAfter = await host.screenshot()

        case .done:
            step.screenshotAfter = step.screenshotBefore
        }
        step.state = .validated
        replaceLast(step)
        attemptSeed = 0
        isBusy = false

        if case .done = action { return }
        if stopRequested { return }
        if chainNext && steps.count < maxAutoSteps {
            await proposeNextStep(host: host, registry: registry)
        }
    }

    func runAutoPlay(host: IOSSimulatorHostController, registry: ModelRegistry) async {
        while autoPlay, !stopRequested, !isBusy, !sessionFinished,
              steps.count < maxAutoSteps,
              let last = steps.last, last.state == .awaiting, last.actionProposed != nil
        {
            await validateAndExecute(host: host, registry: registry, chainNext: true)
        }
    }

    func rejectAndRetry(host: IOSSimulatorHostController, registry: ModelRegistry) async {
        guard var step = steps.last, step.state == .awaiting else { return }
        step.state = .rejected
        replaceLast(step)
        attemptSeed = attemptSeed &+ 1
        await proposeNextStep(host: host, registry: registry)
    }
    func rejectAndSkip() {
        guard var step = steps.last, step.state == .awaiting else { return }
        step.state = .rejected
        replaceLast(step)
        attemptSeed = 0
    }

    // MARK: - Prompt

    private func makePrompt(goal: String, step: Int, device: String, imageW: Int, imageH: Int, withGrid: Bool, accumulatedNotes: [String], appContext: String, recentActions: [String]) -> String {
        let compositeBlock: String = {
            guard useFullPageCapture else { return "" }
            // sepY ≥ 0.95 = page courte qui tient dans le viewport, pas de
            // bande jaune dans l'image, on traite le screenshot comme normal.
            if lastSeparatorY >= 0.95 {
                return """

                == SHORT-PAGE SCREENSHOT (no composite needed) ==
                This iOS screen fits entirely in the viewport — nothing is hidden below the fold.
                The screenshot below is a normal viewport capture, NO composite, NO yellow
                separator band. You can tap any visible element directly with its normalized
                coords. No scroll is needed.

                """
            }
            return """

            == COMPOSITE FULL-PAGE SCREENSHOT ==
            The screenshot below is NOT a normal viewport screenshot — it is a COMPOSITE built
            by the host: it stacks the TOP of the page and the BOTTOM of the page vertically,
            with a YELLOW HORIZONTAL BAND separator between them. The yellow band carries the
            labels "ZONE HAUT" (above) and "ZONE BAS" (below).

            - Anything ABOVE the yellow separator is in ZONE HAUT (visible at the initial scroll).
            - Anything BELOW the yellow separator is in ZONE BAS (visible after scrolling down).
            - Coordinates in your TAP command are normalized on the FULL composite image (0..1).
              The host knows the position of the yellow separator (\(String(format: "y=%.2f", lastSeparatorY)))
              and will automatically scroll to the right zone before tapping.

            Consequence: you can DIRECTLY tap any visible element, even if it's normally below
            the fold. NO scroll_down action is ever needed for a Validate/Submit/Continue button
            — just emit a tap with coordinates inside ZONE BAS.

            """
        }()

        // Detection de boucle : si les 2 dernieres actions sont des taps aux
        // memes coords (±0.03), on injecte un warning explicite.
        let loopWarning: String = {
            let last2 = steps.dropLast().reversed().prefix(2).map { $0 }
            guard last2.count >= 2 else { return "" }
            let actions = last2.compactMap { $0.actionProposed }
            guard actions.count >= 2 else { return "" }
            let a = actions[0], b = actions[1]
            var sameCoords = false
            if case .tap(let x1, let y1) = a, case .tap(let x2, let y2) = b {
                sameCoords = abs(x1 - x2) < 0.03 && abs(y1 - y2) < 0.03
            }
            if sameCoords {
                return """

                ⚠ LOOP DETECTED: your previous tap at almost the same coordinates is being
                proposed again. The screen DID NOT change. This means either:
                  (a) you missed your target by a few pixels — try the same option but
                      shift y by ±0.04 (e.g. y=0.83 → y=0.87 or y=0.79).
                  (b) your option is already selected (look for a colored border around it
                      in the screenshot) — in that case, TAP THE NEXT ACTION button
                      (Valider, Suivant…) instead of re-tapping the option.
                  (c) the RED CIRCLE on the screenshot shows where your previous tap actually
                      landed — use it to estimate your offset and correct.

                """
            }
            return ""
        }()

        let recentBlock: String = {
            guard !recentActions.isEmpty else { return "" }
            let lines = recentActions.map { "  • \($0)" }.joined(separator: "\n")
            return """

            == RECENT ACTIONS ALREADY EXECUTED ==
            \(lines)
            Use this as a MEMORY of what just happened. Hard constraint: NEVER repeat the
            same action twice in a row. If the most recent action was a tap, your NEXT action
            should almost always be a scroll (unless the page clearly changed and you need to
            tap something new). If the most recent action was a scroll, your next action is
            usually a tap (or another scroll if you still haven't reached the target).

            """
        }()

        let appBlock: String = {
            let trimmed = appContext.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { return "" }
            return """

            == APPLICATION-SPECIFIC INSTRUCTIONS ==
            The current app has a specific interaction pattern that the user describes below.
            Follow these instructions STRICTLY — they OVERRIDE the generic rules whenever they conflict.

            \(trimmed)

            == END APP INSTRUCTIONS ==

            """
        }()

        let gridBlock = withGrid ? """

        The image has a NORMALIZED COORDINATE GRID overlaid as a visual ruler.
        Magenta lines mark every 0.1 increment on both axes, labeled. The
        TOP-LEFT corner is "(0,0)" and the BOTTOM-RIGHT is "(1,1)". Read coords
        from the magenta gridlines, do not guess.

        """ : ""

        let notesBlock: String = {
            if accumulatedNotes.isEmpty { return "(none yet)" }
            return accumulatedNotes.enumerated().map { "  \($0.offset + 1). \($0.element)" }.joined(separator: "\n")
        }()

        return """
        You are a step-by-step iOS UI navigation assistant. The user will validate every action.

        Goal: \(goal)

        Device: \(device)
        Step number: \(step)

        Notes you already gathered from previous steps:
        \(notesBlock)
        \(loopWarning)\(recentBlock)\(compositeBlock)\(appBlock)
        You are looking at a screenshot of the current iOS screen (\(imageW)x\(imageH) pixels).\(gridBlock)
        Your job: pick the SINGLE next action that advances toward the goal, AND extract a short
        note of what is visible.

        Available actions:
          - tap          : tap on a UI element (provide normalized coords)
          - tap_then_tap : chain TWO taps in a SINGLE step. Use it when you already know both
                           taps in advance (typical: choose an option then tap a Validate button).
                           Saves a whole 25-second Diffusion generation per chained pair.
          - tap_and_type : tap on a text field then type some text in one go. Add ENTER if it's a search.
          - type         : type text into a focused field (only useful if a previous tap focused an input)
          - scroll_down  : scroll the page DOWN (reveal what is currently below the visible area).
                           Use whenever you need to see something that is "lower" on the page —
                           a button at the bottom, more list items, the end of an article…
          - scroll_up    : scroll the page UP (reveal what is currently above the visible area).
                           Use to go back to the top, or to see the start of a question / form.
          - scroll_left  / scroll_right : horizontal scroll if the app uses paged horizontal layouts.
          - done         : you have completed the goal, return a final synthesis

        IMPORTANT decision rules:
          - To open an app from the home screen, tap on its icon. To go back, tap the back arrow / chevron
            at the top-left of the screen.
          - SCROLLING is expressed by VISIBLE EFFECT, not by finger direction:
              "scroll_down" means "make the page move so that I see what's currently below" — this
              is the natural meaning of "scroller vers le bas" in French. The agent does NOT need
              to think about finger direction, the system handles that.
          - QUIZ / FORM / MULTIPLE-CHOICE PAGES: after tapping an answer, the Validate/Submit
            button is often below the current viewport. If you don't immediately see a
            Suivant/Valider/Submit button, your FIRST action is scroll_down. Do NOT re-tap the
            answer that is already selected.
          - For a text search: ALWAYS use tap_and_type in a single step.
          - NEVER repeat the same action twice in a row if the screenshot didn't change.

        Output EXACTLY this format, no markdown, no preamble:

        NOTES: <one short sentence — what is on the iOS screen useful for the goal>
        REASON: <one short sentence — why you pick this action>
        ACTION: <tap | tap_then_tap | tap_and_type | type | scroll_down | scroll_up | scroll_left | scroll_right | done>
        TAP: (x=0.XX, y=0.XX)                  <-- if ACTION is tap, tap_and_type, OR first tap of tap_then_tap
        TAP: (x=0.XX, y=0.XX)                  <-- second TAP line ONLY if ACTION is tap_then_tap
        TEXT: <text to type>                   <-- if ACTION is type OR tap_and_type
        SUMMARY: <2-4 sentence synthesis>      <-- if ACTION is done

        Rules for TAP coords:
        - x and y are normalized in (0, 1) on the screenshot. NOT pixels.
        - Use exactly TWO decimals.

        VALID example to tap a Settings icon:
        NOTES: I see the iOS home screen with the Settings app icon.
        REASON: I tap the Settings icon to open it and look for "Général".
        ACTION: tap
        TAP: (x=0.50, y=0.30)

        VALID example to scroll down in a long list:
        NOTES: I am in Settings list but the section I need is below the fold.
        REASON: I scroll the page down to reveal more items.
        ACTION: scroll_down

        VALID example for a quiz where the Validate button is below the fold:
        NOTES: I selected my answer (the option is now highlighted in blue) but the Valider button
        is not yet visible — it sits at the bottom of the page.
        REASON: Scroll the page down to bring the Valider button into view.
        ACTION: scroll_down

        VALID example to scroll back up after submission:
        NOTES: The next question's title is at the top of the page, not yet visible from my
        current scroll position.
        REASON: Scroll up to bring the next question header into view.
        ACTION: scroll_up

        VALID example for done:
        NOTES: I'm now in Réglages > Général which is the goal.
        REASON: Goal reached.
        ACTION: done
        SUMMARY: The Réglages > Général screen is open. It contains "Informations", "Mise à jour logicielle", "AirDrop"…
        """
    }

    // MARK: - Parsing
    private func parseProposal(_ raw: String) -> (notes: String, reason: String, action: StepAction?) {
        let cleaned = raw
            .replacingOccurrences(of: "<eos>", with: " ")
            .replacingOccurrences(of: "<turn|>", with: " ")
            .replacingOccurrences(of: "<|turn>", with: " ")
            .replacingOccurrences(of: "**", with: "")

        func extract(_ key: String) -> String? {
            let stops = ["NOTES", "REASON", "ACTION", "TAP", "TEXT", "SUMMARY", "PRESS_ENTER"]
                .filter { $0 != key }.joined(separator: "|")
            let pattern = #"\#(key)\s*:\s*(.*?)(?=\b(?:\#(stops))\b\s*:|$)"#
            guard let r = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive, .dotMatchesLineSeparators]),
                  let m = r.firstMatch(in: cleaned, range: NSRange(cleaned.startIndex..., in: cleaned)),
                  let rng = Range(m.range(at: 1), in: cleaned)
            else { return nil }
            return cleaned[rng].trimmingCharacters(in: .whitespacesAndNewlines)
        }

        let notes = extract("NOTES") ?? ""
        let reason = extract("REASON") ?? ""
        let actionRaw = (extract("ACTION") ?? "").lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.union(.init(charactersIn: "_-")).inverted)
            .first ?? ""

        if actionRaw.contains("done") {
            return (notes, reason, .done(summary: extract("SUMMARY") ?? reason))
        }
        // tap_then_tap : 2 coords successifs (TAP + TAP_2 ou TAP1/TAP2)
        if actionRaw.contains("tap_then_tap") || actionRaw == "double_tap"
            || actionRaw == "tap2" || actionRaw == "answer_and_validate"
        {
            let coords = extractAllCoords(cleaned)
            if coords.count >= 2 {
                return (notes, reason, .tapThenTap(x1: coords[0].x, y1: coords[0].y,
                                                    x2: coords[1].x, y2: coords[1].y))
            }
        }
        // Vocabulaire principal : scroll_X dans le sens de l'effet visible.
        // On accepte aussi swipe_X (compat) en MAPPANT correctement : ce que
        // le modele appelle "swipe_up" en termes de geste doigt = scroll_down
        // en termes d'effet, et reciproquement.
        if actionRaw.contains("scroll_down") { return (notes, reason, .scroll(.down)) }
        if actionRaw.contains("scroll_up")   { return (notes, reason, .scroll(.up)) }
        if actionRaw.contains("scroll_right"){ return (notes, reason, .scroll(.right)) }
        if actionRaw.contains("scroll_left") { return (notes, reason, .scroll(.left)) }
        if actionRaw == "scroll" { return (notes, reason, .scroll(.down)) }
        // Retrocompat / synonymes swipe (sens du geste doigt)
        if actionRaw.contains("swipe_up")    { return (notes, reason, .scroll(.down)) }
        if actionRaw.contains("swipe_down")  { return (notes, reason, .scroll(.up)) }
        if actionRaw.contains("swipe_left")  { return (notes, reason, .scroll(.right)) }
        if actionRaw.contains("swipe_right") || actionRaw == "back" { return (notes, reason, .scroll(.left)) }
        if actionRaw == "swipe" { return (notes, reason, .scroll(.down)) }

        if actionRaw.contains("tap_and_type") || actionRaw.contains("tapandtype") ||
            (actionRaw.contains("tap") && actionRaw.contains("type"))
        {
            let text = extract("TEXT")?
                .trimmingCharacters(in: CharacterSet(charactersIn: "\"'"))
                .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            let pe = extract("PRESS_ENTER")?.lowercased() ?? "yes"
            let pressEnter = !(pe.hasPrefix("no") || pe.hasPrefix("false") || pe == "0")
            if !text.isEmpty, let c = extractCoords(cleaned) {
                return (notes, reason, .tapAndType(x: c.x, y: c.y, text: text, pressEnter: pressEnter))
            }
        }
        if actionRaw.contains("type") {
            let text = extract("TEXT")?
                .trimmingCharacters(in: CharacterSet(charactersIn: "\"'"))
                .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            let pe = extract("PRESS_ENTER")?.lowercased() ?? "yes"
            let pressEnter = !(pe.hasPrefix("no") || pe.hasPrefix("false") || pe == "0")
            if !text.isEmpty { return (notes, reason, .type(text: text, pressEnter: pressEnter)) }
        }
        if let c = extractCoords(cleaned) {
            return (notes, reason, .tap(x: c.x, y: c.y))
        }
        return (notes, reason, nil)
    }

    /// Extrait TOUTES les paires (x, y) trouvees dans la sortie, dans l'ordre.
    /// Sert pour les actions multi-tap comme tap_then_tap.
    private func extractAllCoords(_ s: String) -> [(x: Double, y: Double)] {
        let pattern = #"\(\s*x\s*=\s*([0-9.]+)\s*,\s*y\s*=\s*([0-9.]+)\s*\)"#
        guard let r = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]) else { return [] }
        var out: [(Double, Double)] = []
        let ns = s as NSString
        let matches = r.matches(in: s, range: NSRange(location: 0, length: ns.length))
        for m in matches {
            guard m.numberOfRanges >= 3,
                  let xr = Range(m.range(at: 1), in: s),
                  let yr = Range(m.range(at: 2), in: s),
                  let x = Double(s[xr]), let y = Double(s[yr]),
                  x >= 0, x <= 1, y >= 0, y <= 1
            else { continue }
            out.append((x, y))
        }
        return out
    }

    private func extractCoords(_ s: String) -> (x: Double, y: Double)? {
        let patterns: [String] = [
            #"TAP\s*:?\s*\(?\s*x\s*=\s*([0-9.]+)\s*,\s*y\s*=\s*([0-9.]+)"#,
            #"CLICK\s*:?\s*\(?\s*x\s*=\s*([0-9.]+)\s*,\s*y\s*=\s*([0-9.]+)"#,
            #"x\s*=\s*([0-9.]+)\s*,\s*y\s*=\s*([0-9.]+)"#,
            #"\(\s*([01]?\.[0-9]+)\s*,\s*([01]?\.[0-9]+)\s*\)"#,
        ]
        for p in patterns {
            guard let r = try? NSRegularExpression(pattern: p, options: [.caseInsensitive]),
                  let m = r.firstMatch(in: s, range: NSRange(s.startIndex..., in: s)),
                  let xr = Range(m.range(at: 1), in: s),
                  let yr = Range(m.range(at: 2), in: s),
                  let x = Double(s[xr]), let y = Double(s[yr]),
                  x >= 0, x <= 1, y >= 0, y <= 1
            else { continue }
            return (x, y)
        }
        return nil
    }

    private func replaceLast(_ step: StepCapture) {
        guard !steps.isEmpty else { return }
        steps[steps.count - 1] = step
    }

    /// Coords du dernier tap (ou 1er d'un tap_then_tap / tap_and_type) execute
    /// avec succes au step precedent. Sert a annoter le screenshot du step
    /// courant pour aider le modele a corriger.
    private func lastValidatedTapCoords() -> (x: Double, y: Double)? {
        guard steps.count >= 2 else { return nil }
        let prev = steps[steps.count - 2]
        guard prev.state == .validated, let a = prev.actionProposed else { return nil }
        switch a {
        case .tap(let x, let y), .tapAndType(let x, let y, _, _):
            return (x, y)
        case .tapThenTap(let x, let y, _, _):
            return (x, y)
        default:
            return nil
        }
    }

    /// Tap en remappant si on est en mode composite : zone HAUT = tap direct,
    /// zone BAS = scroll d'abord puis tap. Le parametre `isScrolledDown` evite
    /// de re-scroller inutilement quand on enchaine plusieurs taps en zone BAS.
    /// Retourne le nouvel etat de scroll (true si la page est maintenant
    /// scrollee vers le bas).
    @discardableResult
    private func dispatchTapWithCompositeRemap(host: IOSSimulatorHostController, x: Double, y: Double, isScrolledDown: Bool = false) async -> Bool {
        let sep = lastSeparatorY
        guard useFullPageCapture, sep < 1.0 else {
            // Mode simple : on tape directement sur le viewport courant, sans
            // aucun remap. C'est ce que le modele s'attend a voir.
            print(String(format: "[iOSAgent] direct tap (%.3f, %.3f) sur viewport actuel", x, y))
            await host.tap(normalizedX: x, normalizedY: y)
            return false
        }
        if y < sep - 0.02 {
            // zone HAUT : si on etait scrolled, on remet au top d'abord
            if isScrolledDown {
                await host.swipe(fromX: 0.5, fromY: 0.25, toX: 0.5, toY: 0.75, durationMs: 220)
                try? await Task.sleep(nanoseconds: 350_000_000)
            }
            await host.tap(normalizedX: x, normalizedY: y / sep)
            return false
        } else if y > sep + 0.02 {
            // zone BAS : on ne scroll QUE si on ne l'a pas deja fait
            if !isScrolledDown {
                await host.swipe(fromX: 0.5, fromY: 0.75, toX: 0.5, toY: 0.25, durationMs: 220)
                try? await Task.sleep(nanoseconds: 400_000_000)
            }
            await host.tap(normalizedX: x, normalizedY: (y - sep) / (1.0 - sep))
            return true
        }
        // Coord dans le separateur : ignoré
        return isScrolledDown
    }

    /// Si la page est scrollee vers le bas, remet au top. Appelle apres chaque
    /// dispatch pour que le screenshot composite suivant reparte d'un etat propre.
    private func scrollBackToTopIfNeeded(host: IOSSimulatorHostController, isScrolledDown: Bool) async {
        guard isScrolledDown else { return }
        await host.swipe(fromX: 0.5, fromY: 0.25, toX: 0.5, toY: 0.75, durationMs: 220)
        try? await Task.sleep(nanoseconds: 350_000_000)
    }

    // MARK: - Grid overlay (copy from web)

    private func overlayCoordGrid(on source: NSImage, mainStep: Double = 0.1, subStep: Double = 0.05) -> NSImage {
        guard let cg = source.cgImage(forProposedRect: nil, context: nil, hints: nil) else { return source }
        let W = cg.width, H = cg.height
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(data: nil, width: W, height: H,
                                  bitsPerComponent: 8, bytesPerRow: 0,
                                  space: colorSpace,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { return source }
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: W, height: H))

        let mainLine = CGColor(red: 1.0, green: 0.0, blue: 1.0, alpha: 0.40)
        let subLine  = CGColor(red: 1.0, green: 0.6, blue: 1.0, alpha: 0.20)
        let labelBg = CGColor(red: 0.0, green: 0.0, blue: 0.0, alpha: 0.85)
        let labelMain = NSColor(red: 1.0, green: 0.40, blue: 1.0, alpha: 1.0)
        let labelSub  = NSColor(red: 1.0, green: 0.75, blue: 1.0, alpha: 0.95)

        // Sous-lignes (step 0.05) en violet pale
        ctx.setLineWidth(0.7)
        ctx.setStrokeColor(subLine)
        var v = subStep
        while v < 1.0 {
            let x = CGFloat(v) * CGFloat(W)
            ctx.move(to: CGPoint(x: x, y: 0)); ctx.addLine(to: CGPoint(x: x, y: CGFloat(H))); ctx.strokePath()
            v += subStep
        }
        var h = subStep
        while h < 1.0 {
            let y = CGFloat(H) - CGFloat(h) * CGFloat(H)
            ctx.move(to: CGPoint(x: 0, y: y)); ctx.addLine(to: CGPoint(x: CGFloat(W), y: y)); ctx.strokePath()
            h += subStep
        }
        // Lignes principales (step 0.1) en magenta plus marque
        ctx.setLineWidth(1.0)
        ctx.setStrokeColor(mainLine)
        v = mainStep
        while v < 1.0 {
            let x = CGFloat(v) * CGFloat(W)
            ctx.move(to: CGPoint(x: x, y: 0)); ctx.addLine(to: CGPoint(x: x, y: CGFloat(H))); ctx.strokePath()
            v += mainStep
        }
        h = mainStep
        while h < 1.0 {
            let y = CGFloat(H) - CGFloat(h) * CGFloat(H)
            ctx.move(to: CGPoint(x: 0, y: y)); ctx.addLine(to: CGPoint(x: CGFloat(W), y: y)); ctx.strokePath()
            h += mainStep
        }

        do {
            let nsCtx = NSGraphicsContext(cgContext: ctx, flipped: false)
            NSGraphicsContext.saveGraphicsState()
            NSGraphicsContext.current = nsCtx
            let fontSize: CGFloat = max(11, CGFloat(H) / 70)
            let font = NSFont.systemFont(ofSize: fontSize, weight: .bold)
            let fontSub = NSFont.systemFont(ofSize: fontSize - 2, weight: .regular)
            func drawLabel(_ text: String, atX cx: CGFloat, atY cyTopDown: CGFloat, isMain: Bool) {
                let f: NSFont = isMain ? font : fontSub
                let color: NSColor = isMain ? labelMain : labelSub
                let attrs: [NSAttributedString.Key: Any] = [.font: f, .foregroundColor: color]
                let attrStr = NSAttributedString(string: text, attributes: attrs)
                let size = attrStr.size()
                let pad: CGFloat = 2
                let yUp = CGFloat(H) - cyTopDown - size.height
                let bgRect = CGRect(x: cx - pad, y: yUp - pad, width: size.width + 2 * pad, height: size.height + 2 * pad)
                ctx.setFillColor(labelBg)
                ctx.fill(bgRect)
                attrStr.draw(at: CGPoint(x: cx, y: yUp))
            }
            // Labels TOUS les 0.05 sur l'axe Y (gauche) — c'est ce qui aide
            // le modele a etre precis. Sur X on garde 0.10 (moins critique).
            var hl = mainStep
            while hl < 1.0 {
                drawLabel(String(format: "%.1f", hl), atX: 4, atY: CGFloat(hl) * CGFloat(H) - fontSize / 2, isMain: true)
                hl += mainStep
            }
            hl = subStep
            while hl < 1.0 {
                // Ne pas dessiner si c'est aussi un main step
                let isAlsoMain = abs(hl.truncatingRemainder(dividingBy: mainStep)) < 0.001
                if !isAlsoMain {
                    drawLabel(String(format: "%.2f", hl), atX: 4, atY: CGFloat(hl) * CGFloat(H) - fontSize / 2, isMain: false)
                }
                hl += subStep
            }
            // Labels X
            var vl = mainStep
            while vl < 1.0 {
                drawLabel(String(format: "%.1f", vl), atX: CGFloat(vl) * CGFloat(W) - 8, atY: 4, isMain: true)
                vl += mainStep
            }
            NSGraphicsContext.restoreGraphicsState()
        }

        guard let outCG = ctx.makeImage() else { return source }
        return NSImage(cgImage: outCG, size: source.size)
    }

    /// Dessine un point rouge sur l'image a la position normalisee (nx, ny) avec
    /// une etiquette indiquant que c'est le dernier tap. Sert de feedback visuel
    /// pour le modele : "tu as tape ici la derniere fois — verifie si c'etait
    /// la bonne position".
    private func overlayLastTapMarker(on source: NSImage, nx: Double, ny: Double) -> NSImage {
        guard let cg = source.cgImage(forProposedRect: nil, context: nil, hints: nil) else { return source }
        let W = cg.width, H = cg.height
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(data: nil, width: W, height: H,
                                  bitsPerComponent: 8, bytesPerRow: 0,
                                  space: colorSpace,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { return source }
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: W, height: H))
        let cx = CGFloat(nx) * CGFloat(W)
        let cy = CGFloat(H) - CGFloat(ny) * CGFloat(H)
        // Cercle rouge avec contour blanc
        let r: CGFloat = max(18, CGFloat(W) / 35)
        ctx.setLineWidth(4)
        ctx.setStrokeColor(CGColor(red: 1, green: 1, blue: 1, alpha: 0.95))
        ctx.strokeEllipse(in: CGRect(x: cx - r, y: cy - r, width: r * 2, height: r * 2))
        ctx.setStrokeColor(CGColor(red: 1, green: 0.1, blue: 0.1, alpha: 1))
        ctx.setLineWidth(2.5)
        ctx.strokeEllipse(in: CGRect(x: cx - r, y: cy - r, width: r * 2, height: r * 2))
        ctx.setFillColor(CGColor(red: 1, green: 0, blue: 0, alpha: 0.6))
        ctx.fillEllipse(in: CGRect(x: cx - r / 2, y: cy - r / 2, width: r, height: r))
        // Label
        do {
            let nsCtx = NSGraphicsContext(cgContext: ctx, flipped: false)
            NSGraphicsContext.saveGraphicsState()
            NSGraphicsContext.current = nsCtx
            let txt = String(format: "DERNIER TAP (%.2f, %.2f)", nx, ny)
            let f = NSFont.systemFont(ofSize: max(11, CGFloat(H) / 60), weight: .bold)
            let attrs: [NSAttributedString.Key: Any] = [.font: f, .foregroundColor: NSColor.white]
            let a = NSAttributedString(string: txt, attributes: attrs)
            let size = a.size()
            let bx = max(8, min(cx + r + 8, CGFloat(W) - size.width - 12))
            let by = cy - size.height / 2
            ctx.setFillColor(CGColor(red: 0.8, green: 0, blue: 0, alpha: 0.9))
            ctx.fill(CGRect(x: bx - 4, y: by - 3, width: size.width + 8, height: size.height + 6))
            a.draw(at: CGPoint(x: bx, y: by))
            NSGraphicsContext.restoreGraphicsState()
        }
        guard let outCG = ctx.makeImage() else { return source }
        return NSImage(cgImage: outCG, size: source.size)
    }
}
