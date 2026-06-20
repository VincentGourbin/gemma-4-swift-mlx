// Step-by-step web navigation POC : DiffusionGemma seul propose, l'utilisateur
// valide ou rejette a chaque etape.
//
// Workflow :
//   1. User tape goal + URL, clique Lancer
//   2. Navigate -> attend que la page charge
//   3. Screenshot du viewport
//   4. Prompt DiffusionGemma : "Pour [goal], où dois-je cliquer ? Réponds REASON + CLICK(x,y)"
//   5. Affiche screenshot + crosshair sur (x,y) + reason
//   6. User clique Valider -> click effectif -> wait -> screenshot final commit dans la timeline
//      User clique Rejeter -> nouvelle proposition (seed different) ou skip
//   7. Loop : "Step suivant" relance a 3.

import AppKit
import Foundation
import Gemma4Swift
import MLX
import Tokenizers

@MainActor
final class AgentStepViewModel: ObservableObject {

    // MARK: - State

    enum StepState: String, Sendable {
        case proposing      // DiffusionGemma reflechit
        case awaiting       // proposition prete, attend validation
        case validated      // user a valide -> action executee -> screenshot apres
        case rejected       // user a rejete -> on passera a l'etape suivante
        case errored
    }

    enum ScrollDir: String, Sendable { case up, down }

    /// Action proposee par le modele pour le step courant.
    enum StepAction: Sendable, Equatable {
        case click(x: Double, y: Double)
        case scroll(ScrollDir)
        case type(text: String, pressEnter: Bool)
        /// Action composite : click sur (x, y), focus l'input cible, tape `text`,
        /// puis (par defaut) appuie sur Enter. Reduit a 1 step les sequences
        /// search bar : click + saisie. Le modele propose les coords ET le texte
        /// dans la meme reponse, on enchaine tout cote dispatcher.
        case clickAndType(x: Double, y: Double, text: String, pressEnter: Bool)
        case done(summary: String)

        var kind: String {
            switch self {
            case .click: return "click"
            case .scroll(let d): return "scroll_\(d.rawValue)"
            case .type(_, let e): return e ? "type+enter" : "type"
            case .clickAndType(_, _, _, let e): return e ? "click+type+enter" : "click+type"
            case .done: return "done"
            }
        }
    }

    struct StepCapture: Identifiable, Sendable {
        let id = UUID()
        let n: Int
        var url: String
        var pageTitle: String
        var screenshotBefore: NSImage
        var screenshotGrid: NSImage? = nil   // version avec grille envoyee au modele
        var screenshotAfter: NSImage? = nil
        var coordsProposed: (x: Double, y: Double)? = nil   // si action == click
        var actionProposed: StepAction? = nil               // action complete proposee
        var reason: String = ""
        var notes: String = ""    // ce que le modele a retenu de cette page pour le goal
        var rawOutput: String = ""
        var elapsedDiffusion: TimeInterval? = nil
        var diffusionForwards: Int? = nil
        var state: StepState = .proposing
        var error: String? = nil
        var elementHit: String? = nil  // tagName#id.class :: text apres click
    }

    @Published var steps: [StepCapture] = []
    @Published var goal: String = "Sur huggingface.co, trouve les modèles Gemma 4 de Google et clique sur la première carte de modèle Google."
    @Published var startURL: String = "https://huggingface.co/models?search=gemma"
    @Published var selectedStepIndex: Int? = nil   // ID du step actif dans le viewer
    @Published var isBusy: Bool = false

    /// Si vrai, on superpose une grille de coordonnees normalisees (magenta,
    /// tous les 0.1) sur le screenshot envoye au modele. Ameliore drastiquement
    /// le grounding : sur le bench Wikipedia FR, le 1er click sur la search
    /// bar est passe de y=0.42 (centre logo Wikipedia, errone) a y=0.03 (zone
    /// search bar, correct). Le prompt mentionne la grille comme repere.
    @Published var useCoordGrid: Bool = true

    /// Mode auto-play : apres une proposition, on auto-valide et on enchaine
    /// directement sur le step suivant sans intervention. Sert pour les demos
    /// quand on a confiance dans le modele. L'utilisateur peut couper le mode
    /// pour reprendre la main.
    @Published var autoPlay: Bool = false
    @Published var maxAutoSteps: Int = 8

    /// Flag de stop demande par l'utilisateur. Quand `true`, la boucle
    /// auto-play s'arrete au prochain check, et les fonctions proposeNextStep
    /// / validateAndClick n'enchainent plus sur le step suivant. Le step en
    /// cours peut quand meme terminer (le forward Diffusion ~25s n'est pas
    /// interruptible). L'user reprend la main pour relancer manuellement.
    @Published var stopRequested: Bool = false

    /// Dossier de logs de la session courante. Cree au 1er step et reutilise
    /// par les steps suivants. Permet d'analyser offline ce que le modele a
    /// recu / produit. Pareil que IOSAgentStepViewModel mais pour le web.
    @Published private(set) var sessionLogDir: URL? = nil

    private func ensureLogDir() -> URL {
        if let d = sessionLogDir { return d }
        let ts = Int(Date().timeIntervalSince1970)
        let dir = URL(fileURLWithPath: "/tmp/web-agent-runs/run-\(ts)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let manifest = """
        # Web Agent run — \(Date())
        Goal:
        \(goal)

        Start URL:
        \(startURL)
        """
        try? manifest.write(to: dir.appendingPathComponent("manifest.md"), atomically: true, encoding: .utf8)
        sessionLogDir = dir
        print("[WebAgent] session log dir: \(dir.path)")
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

    private var attemptSeed: UInt64 = 0  // change a chaque rejet

    /// Le ViewModel ne tient plus les modeles — il les recoit du registry
    /// partage via les arguments des methodes (proposeNextStep, etc.).
    func canProposeNext(registry: ModelRegistry) -> Bool {
        registry.isDiffusionLoaded && !isBusy && lastStepResolved && !sessionFinished
    }
    var lastStepResolved: Bool {
        guard let last = steps.last else { return true }
        return last.state == .validated || last.state == .rejected || last.state == .errored
    }
    /// True si l'agent a emis un step terminal (action=done validee).
    var sessionFinished: Bool {
        for s in steps where s.state == .validated {
            if case .done = s.actionProposed { return true }
        }
        return false
    }
    /// Synthese finale si l'agent a conclu, sinon nil.
    var finalSummary: String? {
        for s in steps where s.state == .validated {
            if case .done(let summary) = s.actionProposed { return summary }
        }
        return nil
    }
    var awaitingValidation: Bool {
        steps.last?.state == .awaiting
    }
    var currentStep: StepCapture? {
        if let idx = selectedStepIndex, idx < steps.count { return steps[idx] }
        return steps.last
    }

    /// Demande l'arret de la chaine auto-play. Coupe egalement le toggle
    /// autoPlay pour eviter qu'une nouvelle iteration ne demarre.
    func requestStop() {
        stopRequested = true
        autoPlay = false
    }

    func reset() {
        steps = []
        selectedStepIndex = nil
        attemptSeed = 0
        stopRequested = false
        sessionLogDir = nil
    }

    // MARK: - Propose / validate / reject

    /// Lance le step suivant : screenshot + DiffusionGemma -> proposition.
    func proposeNextStep(browser: WebBrowserHostController, registry: ModelRegistry) async {
        guard !stopRequested else { return }
        guard canProposeNext(registry: registry) else { return }
        guard let model = registry.diffModel,
              let dconf = registry.diffConfig,
              let dgen = registry.diffGenConfig,
              let dtok = registry.diffTokenizer
        else { return }

        // Si pas encore de steps, on navigate d'abord au startURL
        if steps.isEmpty {
            browser.navigate(to: startURL)
            for _ in 0 ..< 30 where browser.isLoading {
                try? await Task.sleep(nanoseconds: 200_000_000)
            }
            try? await Task.sleep(nanoseconds: 600_000_000) // marge pour rendu JS
            // Best-effort : ferme un eventuel bandeau cookies pour eviter que
            // l'agent gaspille un step dessus.
            if let dismissed = await browser.tryDismissCookieBanner() {
                print("[AgentStep] cookie banner auto-dismissed: \(dismissed)")
                try? await Task.sleep(nanoseconds: 500_000_000)
            }
        }

        isBusy = true
        defer { isBusy = false }

        guard let shotBefore = await browser.screenshot(width: 1024) else { return }
        let n = steps.count + 1
        var step = StepCapture(
            n: n,
            url: browser.currentURL,
            pageTitle: browser.pageTitle,
            screenshotBefore: shotBefore,
            state: .proposing
        )
        steps.append(step)
        selectedStepIndex = steps.count - 1

        // Prompt + VQA
        let imgW = Int(shotBefore.size.width)
        let imgH = Int(shotBefore.size.height)
        // Accumule toutes les NOTES non vides des steps precedents
        let accumulatedNotes = steps.dropLast().compactMap { $0.notes.isEmpty ? nil : $0.notes }
        // Compte combien de steps consecutifs valides ont la meme URL que la
        // page actuelle -> permet de detecter une boucle (forme, quiz...) et
        // d'injecter un warning explicite dans le prompt.
        let currentURL = browser.currentURL
        var sameURLCount = 0
        for s in steps.dropLast().reversed() where s.state == .validated {
            if s.url == currentURL { sameURLCount += 1 } else { break }
        }
        // Detecte les boutons de navigation visibles (Suivant/Submit/...)
        // pour les injecter dans le prompt. Le modele a leurs coords sans
        // avoir a grounder visuellement.
        let navButtons = await browser.findNavigationButtons()
        let prompt = makePrompt(
            goal: goal, step: n,
            url: currentURL, title: browser.pageTitle,
            imageW: imgW, imageH: imgH,
            withGrid: useCoordGrid,
            accumulatedNotes: accumulatedNotes,
            sameURLStreak: sameURLCount,
            navButtons: navButtons
        )

        // Si la grille est activee, on dessine la grille SUR le screenshot
        // avant de le passer au modele. La version original est gardee dans
        // screenshotBefore (pour annoter le crosshair user-facing).
        let imageForModel: NSImage
        if useCoordGrid {
            let withGrid = overlayCoordGrid(on: shotBefore)
            step.screenshotGrid = withGrid
            replaceLast(step)
            imageForModel = withGrid
        } else {
            imageForModel = shotBefore
        }

        guard let cg = imageForModel.cgImage(forProposedRect: nil, context: nil, hints: nil),
              let pixels = try? Gemma4ImageProcessor.processImage(cg)
        else {
            step.state = .errored
            step.error = "screenshot -> pixels failed"
            replaceLast(step)
            return
        }

        // Boucle interne : si le parser echoue (CLICK manquant ou hors range),
        // on retry automatiquement 1 fois avec un seed different. Au-dela, on
        // laisse l'user voir la raw output et decider.
        var seed = attemptSeed
        var lastRawOut = ""
        var lastForwards: Int? = nil
        var totalElapsed: TimeInterval = 0
        let start = Date()
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
            var parsedNotes: String = ""
            var parsedReason: String = ""

            // Jusqu'a 2 tentatives : on retry avec un seed different si le parser
            // n'arrive pas a extraire une action valide (NOTES/REASON/ACTION).
            for attempt in 0 ..< 2 {
                let attemptSeedLocal = seed &+ UInt64(attempt)
                let out: (text: String, forwards: Int) = await Task.detached {
                    let pipeline = DiffusionGemmaPipeline(model: unsafeModel, genConfig: dgen)
                    let r = await pipeline.generate(
                        promptIds: unsafeIds,
                        pixelValues: unsafePixels,
                        maxBlocks: 1,
                        seed: attemptSeedLocal
                    )
                    let outIds = r.generatedIds.asArray(Int32.self).map { Int($0) }
                    return (unsafeTokenizer.decode(tokens: outIds, skipSpecialTokens: true), r.totalDecoderSteps)
                }.value
                lastRawOut = out.text
                lastForwards = out.forwards
                totalElapsed = Date().timeIntervalSince(start)
                print("[AgentStep] step \(n) attempt \(attempt + 1) raw: \(out.text.replacingOccurrences(of: "\n", with: " | "))")

                let parsed = parseProposal(out.text)
                parsedNotes = parsed.notes
                parsedReason = parsed.reason
                parsedAction = parsed.action
                if parsedAction != nil { break }
                print("[AgentStep] step \(n) attempt \(attempt + 1) — action non parseable, retry seed")
            }

            step.elapsedDiffusion = totalElapsed
            step.diffusionForwards = lastForwards
            step.rawOutput = lastRawOut
            step.notes = parsedNotes
            step.reason = parsedReason.isEmpty ? "(modele n'a pas renvoye de REASON)" : parsedReason
            step.actionProposed = parsedAction
            switch parsedAction {
            case .click(let x, let y):
                step.coordsProposed = (x, y)
            case .clickAndType(let x, let y, _, _):
                step.coordsProposed = (x, y)
            default:
                break
            }
            if let a = parsedAction {
                print("[AgentStep] step \(n) action: \(a.kind)")
            } else {
                print("[AgentStep] step \(n) !! NO ACTION after retries — raw was:\n\(lastRawOut)")
            }
            step.state = .awaiting
            replaceLast(step)
            // Log de la generation (prompt + raw + parsed + screenshot)
            writeLog("step-\(n)-prompt.md", content: """
            # Step \(n) — prompt envoyé au modèle

            URL: \(browser.currentURL)
            Title: \(browser.pageTitle)
            Image: \(imgW)x\(imgH), grid=\(useCoordGrid)
            Diffusion: \(String(format: "%.2f", totalElapsed))s, \(lastForwards ?? 0) forwards

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

    /// User valide la proposition -> execute l'action choisie par le modele
    /// (click / scroll / done) + screenshot apres -> enchaine IMMEDIATEMENT
    /// sur la prochaine proposition si l'action n'est pas terminale.
    func validateAndClick(browser: WebBrowserHostController, registry: ModelRegistry, chainNext: Bool = true) async {
        guard var step = steps.last, step.state == .awaiting, let action = step.actionProposed else { return }
        isBusy = true

        switch action {
        case .click(let x, let y):
            let hit = await browser.click(normalizedX: x, normalizedY: y)
            step.elementHit = hit
            try? await Task.sleep(nanoseconds: 900_000_000)
            for _ in 0 ..< 20 where browser.isLoading {
                try? await Task.sleep(nanoseconds: 200_000_000)
            }
            if let dismissed = await browser.tryDismissCookieBanner() {
                print("[AgentStep] cookie banner auto-dismissed after click: \(dismissed)")
                try? await Task.sleep(nanoseconds: 500_000_000)
            }
            step.screenshotAfter = await browser.screenshot(width: 1024)

        case .scroll(let dir):
            await browser.scroll(deltaY: dir == .up ? -600 : 600)
            try? await Task.sleep(nanoseconds: 400_000_000)
            step.screenshotAfter = await browser.screenshot(width: 1024)

        case .type(let text, let pressEnter):
            let typed = await browser.type(text: text)
            step.elementHit = typed
            if pressEnter {
                try? await Task.sleep(nanoseconds: 200_000_000)
                await browser.pressEnter()
            }
            try? await Task.sleep(nanoseconds: 1_200_000_000)
            for _ in 0 ..< 25 where browser.isLoading {
                try? await Task.sleep(nanoseconds: 200_000_000)
            }
            step.screenshotAfter = await browser.screenshot(width: 1024)

        case .clickAndType(let x, let y, let text, let pressEnter):
            // 1. Click sur les coords -> focus l'input cible
            let hit = await browser.click(normalizedX: x, normalizedY: y)
            try? await Task.sleep(nanoseconds: 300_000_000)
            // 2. Tape le texte (auto-focus le 1er search input visible si le click n'a pas focused)
            let typed = await browser.type(text: text)
            step.elementHit = "click=\(hit ?? "?") | type=\(typed ?? "?")"
            // 3. Enter -> submit du form/search
            if pressEnter {
                try? await Task.sleep(nanoseconds: 200_000_000)
                await browser.pressEnter()
            }
            // 4. Attente du load + auto-dismiss cookies + screenshot
            try? await Task.sleep(nanoseconds: 1_200_000_000)
            for _ in 0 ..< 25 where browser.isLoading {
                try? await Task.sleep(nanoseconds: 200_000_000)
            }
            if let dismissed = await browser.tryDismissCookieBanner() {
                print("[AgentStep] cookie banner auto-dismissed after click_and_type: \(dismissed)")
                try? await Task.sleep(nanoseconds: 500_000_000)
            }
            step.screenshotAfter = await browser.screenshot(width: 1024)

        case .done:
            // Pas d'action navigateur : on conserve l'etat actuel comme final.
            step.screenshotAfter = step.screenshotBefore
        }

        step.state = .validated
        step.url = browser.currentURL
        step.pageTitle = browser.pageTitle
        replaceLast(step)
        attemptSeed = 0
        isBusy = false

        // Enchainement : si done ou stop demande, on s'arrete ici. Sinon on enchaine.
        if case .done = action { return }
        if stopRequested { return }
        if chainNext && steps.count < maxAutoSteps {
            await proposeNextStep(browser: browser, registry: registry)
        }
    }

    /// Boucle auto-play : valide automatiquement les propositions du modele
    /// et continue. S'arrete a `maxAutoSteps`, ou si une proposition est
    /// rejetee (action=nil), ou si l'agent emet `done`, ou si l'user coupe.
    func runAutoPlay(browser: WebBrowserHostController, registry: ModelRegistry) async {
        while autoPlay,
              !stopRequested,
              !isBusy,
              !sessionFinished,
              steps.count < maxAutoSteps,
              let last = steps.last,
              last.state == .awaiting,
              last.actionProposed != nil
        {
            await validateAndClick(browser: browser, registry: registry, chainNext: true)
        }
    }

    /// User rejette -> reset le seed et redemande une proposition sur le meme screenshot.
    func rejectAndRetry(browser: WebBrowserHostController, registry: ModelRegistry) async {
        guard var step = steps.last, step.state == .awaiting else { return }
        step.state = .rejected
        replaceLast(step)
        attemptSeed = attemptSeed &+ 1
        await proposeNextStep(browser: browser, registry: registry)
    }

    /// User rejette et veut passer au step suivant sans cliquer (force scroll/skip).
    func rejectAndSkip() {
        guard var step = steps.last, step.state == .awaiting else { return }
        step.state = .rejected
        replaceLast(step)
        attemptSeed = 0
    }

    // MARK: - Prompt
    private func makePrompt(goal: String, step: Int, url: String, title: String, imageW: Int, imageH: Int, withGrid: Bool, accumulatedNotes: [String], sameURLStreak: Int, navButtons: [WebBrowserHostController.NavButton]) -> String {
        let gridBlock = withGrid ? """

        The image has a NORMALIZED COORDINATE GRID overlaid as a visual ruler.
        Magenta lines mark every 0.1 increment on both axes, with values labeled
        in magenta on the borders. The TOP-LEFT corner is "(0,0)" and the
        BOTTOM-RIGHT corner is "(1,1)". Use this grid as a ruler: read your x
        from the closest magenta vertical line and your y from the closest
        magenta horizontal line. DO NOT guess — read from the grid.

        """ : ""

        let notesBlock: String = {
            if accumulatedNotes.isEmpty { return "(none yet)" }
            return accumulatedNotes.enumerated()
                .map { "  \($0.offset + 1). \($0.element)" }
                .joined(separator: "\n")
        }()

        // Liste explicite des boutons de navigation visibles avec coords pretes
        // a etre reutilisees. Le modele n'a plus a grounder un "Suivant".
        let navBlock: String = {
            guard !navButtons.isEmpty else { return "" }
            let lines = navButtons.prefix(10).map { btn -> String in
                String(format: "  - \"%@\" at (x=%.2f, y=%.2f)", btn.label, btn.x, btn.y)
            }.joined(separator: "\n")
            return """

            Navigation / action buttons currently visible on the page (use these EXACT coords
            with ACTION=click whenever you want to advance, submit, validate, or go back —
            no need to ground these yourself, the coordinates are already accurate):
            \(lines)

            """
        }()

        // Warning si on est coince sur la meme URL depuis plusieurs steps
        let stuckBlock: String = {
            guard sameURLStreak >= 2 else { return "" }
            return """

            ⚠ STUCK ON SAME URL: your last \(sameURLStreak) actions did NOT change the URL.
            This usually means:
              - It's a form/quiz/SPA page that requires a Next/Submit button to advance.
              - You clicked an element that was already in its target state (e.g. a radio
                option already selected, a toggle already on) — clicking it again has no effect.
              - You clicked into empty space.
            ACTION REQUIRED: pick a DIFFERENT action this turn. Strong hints:
              - Look for a button labeled "Suivant", "Continuer", "Next", "Continue", "Submit",
                "Valider", "OK", "→", or a primary-colored CTA at the bottom or right of the
                current form/card — click it to advance.
              - If you don't see one, scroll to find it.
              - If you're sure the answer is already filled in, click the navigation button.
              - DO NOT re-click the same answer / option.

            """
        }()

        return """
        You are a step-by-step web navigation assistant. The user will validate every action.

        Goal: \(goal)

        Current page URL: \(url)
        Current page title: \(title)
        Step number: \(step)

        Notes you already gathered from previous steps:
        \(notesBlock)
        \(navBlock)\(stuckBlock)
        You are looking at a screenshot of the current page (\(imageW)x\(imageH) pixels).\(gridBlock)
        Your job: pick the SINGLE next action that advances toward the goal, AND extract a short
        note of what is currently useful on this page.

        Available actions:
          - click          : click on an element (provide normalized coords on the screenshot)
          - click_and_type : SHORTCUT for search bars — click on coords, focus the input there, then
                             type the text. By default Enter is fired so you land on the results page.
                             USE THIS WHENEVER you need to search something — it's 1 step instead of 2.
          - type           : type text into the currently focused input (only useful if a previous
                             click already focused an input). Prefer click_and_type for new searches.
          - scroll         : scroll the page DOWN or UP to see more content
          - done           : you have enough information, return a final synthesis

        IMPORTANT decision rules:
          - COOKIE BANNERS / GDPR MODALS / CONSENT OVERLAYS: if the page is partially or fully
            covered by a cookie consent banner, dismiss it FIRST as your single next action,
            BEFORE attempting anything else toward the goal. Prefer reject-style buttons to keep
            cookies minimal, in this order of preference:
              "Refuse all" / "Tout refuser" / "Reject all" / "Refuser tout" / "Decline" /
              "Decliner" / "Non merci" / "Continue without accepting" / "Continuer sans accepter"
            If only an Accept option exists, click it. After the dismiss, the banner should
            disappear in the next screenshot — if it does NOT, do NOT retry the same dismiss
            point: try a different button (e.g. the close X icon, often top-right of the modal)
            or scroll to find the consent area. NEVER spend more than 2 steps on a cookie banner.
          - FORMS / QUIZZES / MULTI-PAGE FLOWS: when a page shows a question with answer choices
            (radio, slider, scale) AND a "Suivant" / "Next" / "Continue" / "Submit" navigation
            button, the canonical flow is:
              step N   : click the answer choice
              step N+1 : click the navigation button (Suivant/Next/Submit) to advance
            Always look for that navigation button after selecting. If a choice is already visibly
            selected (filled-in circle, checkmark, highlighted background), do NOT click it again —
            click the navigation button instead.
          - To search for something, ALWAYS use ACTION=click_and_type in a SINGLE step. Do not split
            it into click then type — that wastes a whole 25-second step.
          - If the current page already contains all the info needed to answer the goal, pick "done"
            with a thorough synthesis built from your notes + this page.
          - If there is clearly MORE useful content below the current viewport (e.g. a long article
            you've only seen the top of), pick "scroll" and direction "down".
          - If you can make progress by clicking a visible element (link, button), pick "click".
          - NEVER repeat the SAME action twice in a row on the same page — if your previous action did
            not change the URL, switch strategy (try type, scroll down, or done).

        Output EXACTLY this format, no markdown, no preamble. The first 3 lines are MANDATORY,
        the rest depends on the action:

        NOTES: <one short sentence — what you see on this page that is useful for the goal>
        REASON: <one short sentence — why you pick this action>
        ACTION: <click | click_and_type | type | scroll_down | scroll_up | done>
        CLICK: (x=0.XX, y=0.XX)            <-- if ACTION is click OR click_and_type
        TEXT: <text to type>               <-- if ACTION is type OR click_and_type
        SUMMARY: <2-4 sentence synthesis>  <-- only if ACTION is done

        Rules for CLICK coords:
        - x and y are NORMALIZED coordinates strictly inside (0, 1). NOT pixels and NOT percentages.
        - Use exactly TWO decimals (e.g. 0.42).

        VALID example for searching in one step (PREFERRED for any search):
        NOTES: This is the Hugging Face models listing page with a search bar at the top.
        REASON: I search for the model name in one shot — click the search bar, type, hit Enter.
        ACTION: click_and_type
        CLICK: (x=0.50, y=0.05)
        TEXT: gemma 4

        VALID example for clicking (without typing):
        NOTES: I see the first result card in the list.
        REASON: I click on the first Gemma 4 model card to open it.
        ACTION: click
        CLICK: (x=0.20, y=0.30)

        VALID example for scrolling:
        NOTES: I see the top of the Gemma 4 12B model card with download counts and tags.
        REASON: The relevant details (parameters, license, downloads) are below the fold.
        ACTION: scroll_down

        VALID example for done:
        NOTES: This page has the model size (12B), license (Apache 2.0), and modalities (Any-to-Any).
        REASON: All required info has been gathered across the previous notes.
        ACTION: done
        SUMMARY: The first Google model on Hugging Face is google/gemma-4-12B-it, a 12B-parameter multimodal model licensed Apache 2.0, supporting Any-to-Any I/O including text, images and audio. It was updated 10 days ago with 1.08M downloads.

        INVALID examples (DO NOT output these):
        - "CLICK: (x=3.38, y=0.44)"   ← x > 1 is wrong, max is 1.00
        - "CLICK: (x=50, y=5)"        ← those are percentages or pixels
        - Omitting ACTION              ← ACTION is mandatory on every step
        """
    }

    // MARK: - Parsing
    /// Parse une proposition multi-lignes NOTES/REASON/ACTION/[CLICK|SUMMARY].
    /// Retourne notes, reason et l'action structuree (ou nil si erreur).
    private func parseProposal(_ raw: String) -> (notes: String, reason: String, action: StepAction?) {
        let cleaned = raw
            .replacingOccurrences(of: "<eos>", with: " ")
            .replacingOccurrences(of: "<turn|>", with: " ")
            .replacingOccurrences(of: "<|turn>", with: " ")
            .replacingOccurrences(of: "**", with: "")

        func extract(_ key: String, terminate: [String] = ["NOTES", "REASON", "ACTION", "CLICK", "TEXT", "SUMMARY", "PRESS_ENTER"]) -> String? {
            let stops = terminate.filter { $0 != key }.joined(separator: "|")
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

        // Determine l'action
        if actionRaw.contains("done") || actionRaw.contains("finish") {
            let summary = extract("SUMMARY") ?? reason
            return (notes, reason, .done(summary: summary))
        }
        // Click composite : click_and_type (extract avant le simple "type")
        if actionRaw.contains("click_and_type") || actionRaw.contains("clickandtype")
            || actionRaw == "search" || actionRaw == "click+type"
            || (actionRaw.contains("click") && actionRaw.contains("type"))
        {
            let text = extract("TEXT")?
                .trimmingCharacters(in: CharacterSet(charactersIn: "\"'"))
                .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            let pe = extract("PRESS_ENTER")?.lowercased() ?? "yes"
            let pressEnter = !(pe.hasPrefix("no") || pe.hasPrefix("false") || pe == "0")
            if !text.isEmpty, let coords = extractCoords(cleaned) {
                return (notes, reason, .clickAndType(x: coords.x, y: coords.y, text: text, pressEnter: pressEnter))
            }
        }
        if actionRaw.contains("type") {
            let text = extract("TEXT")?
                .trimmingCharacters(in: CharacterSet(charactersIn: "\"'"))
                .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            // PRESS_ENTER default = true (search-like)
            let pe = extract("PRESS_ENTER")?.lowercased() ?? "yes"
            let pressEnter = !(pe.hasPrefix("no") || pe.hasPrefix("false") || pe == "0")
            if !text.isEmpty {
                return (notes, reason, .type(text: text, pressEnter: pressEnter))
            }
        }
        if actionRaw.contains("scroll_up") || actionRaw == "scrollup" || (actionRaw == "scroll" && cleaned.lowercased().contains("up")) {
            return (notes, reason, .scroll(.up))
        }
        if actionRaw.contains("scroll") {
            return (notes, reason, .scroll(.down))
        }
        // Click par defaut
        if let coords = extractCoords(cleaned) {
            return (notes, reason, .click(x: coords.x, y: coords.y))
        }
        // Echec
        return (notes, reason, nil)
    }

    /// Tente plusieurs patterns pour extraire des coords normalisees (x, y) dans [0, 1].
    private func extractCoords(_ s: String) -> (x: Double, y: Double)? {
        let patterns: [String] = [
            // CLICK: (x=0.42, y=0.18)
            #"CLICK\s*:?\s*\(?\s*x\s*=\s*([0-9.]+)\s*,\s*y\s*=\s*([0-9.]+)"#,
            // x=0.42, y=0.18 (sans CLICK)
            #"x\s*=\s*([0-9.]+)\s*,\s*y\s*=\s*([0-9.]+)"#,
            // x: 0.42, y: 0.18
            #"x\s*:\s*([0-9.]+)\s*,\s*y\s*:\s*([0-9.]+)"#,
            // at (0.42, 0.18) ou click at (0.42, 0.18)
            #"(?:at|on)\s+\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)"#,
            // (0.42, 0.18) seul (deux flottants entre parens, le 1er trouve dans la ligne CLICK ou apres)
            #"\(\s*([01]?\.[0-9]+)\s*,\s*([01]?\.[0-9]+)\s*\)"#,
            // 0.42, 0.18 — deux flottants avec point separes par virgule
            #"([01]?\.[0-9]+)\s*,\s*([01]?\.[0-9]+)"#,
        ]
        for p in patterns {
            guard let r = try? NSRegularExpression(pattern: p, options: [.caseInsensitive]),
                  let m = r.firstMatch(in: s, range: NSRange(s.startIndex..., in: s)),
                  let xr = Range(m.range(at: 1), in: s),
                  let yr = Range(m.range(at: 2), in: s),
                  let x = Double(s[xr]),
                  let y = Double(s[yr]),
                  x >= 0, x <= 1, y >= 0, y <= 1
            else { continue }
            return (x, y)
        }
        // Fallback pourcentage : x=42%, y=18%
        if let r = try? NSRegularExpression(pattern: #"x\s*=?\s*([0-9.]+)\s*%\s*,?\s*y\s*=?\s*([0-9.]+)\s*%"#, options: [.caseInsensitive]),
           let m = r.firstMatch(in: s, range: NSRange(s.startIndex..., in: s)),
           let xr = Range(m.range(at: 1), in: s),
           let yr = Range(m.range(at: 2), in: s),
           let x = Double(s[xr]), let y = Double(s[yr]),
           x >= 0, x <= 100, y >= 0, y <= 100
        {
            return (x / 100, y / 100)
        }
        return nil
    }

    private func replaceLast(_ step: StepCapture) {
        guard !steps.isEmpty else { return }
        steps[steps.count - 1] = step
    }

    // MARK: - Coord grid overlay

    /// Dessine une grille de coordonnees normalisees (magenta semi-transparent,
    /// lignes tous les 0.1 avec labels lisibles) sur l'image fournie. Le modele
    /// l'utilise comme repere visuel pour produire des coords precises.
    /// Retourne une nouvelle NSImage en RGB.
    private func overlayCoordGrid(on source: NSImage, step: Double = 0.1) -> NSImage {
        guard let cg = source.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            return source
        }
        let W = cg.width
        let H = cg.height
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(
            data: nil, width: W, height: H,
            bitsPerComponent: 8, bytesPerRow: 0,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return source }
        // 1. Background = original image
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: W, height: H))

        // CoreGraphics est y-up — on travaille avec yUp = H - yDown pour aligner
        // avec la convention top=0 du grounding modele.
        let lineColor = CGColor(red: 1.0, green: 0.0, blue: 1.0, alpha: 0.30)
        let labelBg = CGColor(red: 0.0, green: 0.0, blue: 0.0, alpha: 0.78)
        let labelFg = NSColor(red: 1.0, green: 0.40, blue: 1.0, alpha: 1.0)

        ctx.setLineWidth(1.0)
        ctx.setStrokeColor(lineColor)

        // Lignes verticales tous les `step`
        var v = step
        while v < 1.0 {
            let x = CGFloat(v) * CGFloat(W)
            ctx.move(to: CGPoint(x: x, y: 0))
            ctx.addLine(to: CGPoint(x: x, y: CGFloat(H)))
            ctx.strokePath()
            v += step
        }
        // Lignes horizontales tous les `step`
        var h = step
        while h < 1.0 {
            let yTop = CGFloat(h) * CGFloat(H)
            // Convertir top-down (h * H) vers bottom-up (H - h*H)
            let y = CGFloat(H) - yTop
            ctx.move(to: CGPoint(x: 0, y: y))
            ctx.addLine(to: CGPoint(x: CGFloat(W), y: y))
            ctx.strokePath()
            h += step
        }

        // Labels : il faut basculer en NSGraphicsContext pour le drawing texte
        do {
            let nsCtx = NSGraphicsContext(cgContext: ctx, flipped: false)
            NSGraphicsContext.saveGraphicsState()
            NSGraphicsContext.current = nsCtx

            let fontSize: CGFloat = max(11, CGFloat(H) / 70)
            let font = NSFont.systemFont(ofSize: fontSize, weight: .semibold)
            let attrs: [NSAttributedString.Key: Any] = [
                .font: font,
                .foregroundColor: labelFg,
            ]

            func drawLabel(_ text: String, atX cx: CGFloat, atY cyTopDown: CGFloat) {
                let attrStr = NSAttributedString(string: text, attributes: attrs)
                let size = attrStr.size()
                let pad: CGFloat = 3
                let yUp = CGFloat(H) - cyTopDown - size.height
                let bgRect = CGRect(x: cx - pad, y: yUp - pad,
                                    width: size.width + 2 * pad, height: size.height + 2 * pad)
                ctx.setFillColor(labelBg)
                ctx.fill(bgRect)
                attrStr.draw(at: CGPoint(x: cx, y: yUp))
            }

            // Labels verticaux (x = 0.1, 0.2, ..., 0.9) en haut
            v = step
            while v < 1.0 {
                let x = CGFloat(v) * CGFloat(W)
                drawLabel(String(format: "%.1f", v), atX: x - 8, atY: 4)
                v += step
            }
            // Labels horizontaux (y = 0.1, ..., 0.9) a gauche
            h = step
            while h < 1.0 {
                let yTop = CGFloat(h) * CGFloat(H)
                drawLabel(String(format: "%.1f", h), atX: 4, atY: yTop - fontSize / 2)
                h += step
            }

            // Coins (0,0) et (1,1)
            drawLabel("(0,0) top-left", atX: 4, atY: 4 + fontSize + 6)
            let brText = "(1,1) bottom-right"
            let brSize = NSAttributedString(string: brText, attributes: attrs).size()
            drawLabel(brText, atX: CGFloat(W) - brSize.width - 4, atY: CGFloat(H) - brSize.height - 4)

            NSGraphicsContext.restoreGraphicsState()
        }

        guard let outCG = ctx.makeImage() else { return source }
        return NSImage(cgImage: outCG, size: source.size)
    }
}
