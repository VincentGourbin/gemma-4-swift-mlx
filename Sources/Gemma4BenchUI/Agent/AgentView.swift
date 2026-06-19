// Onglet Agent — POC step-by-step :
// WKWebView rendu en arriere-plan (rendering reel), UI principale = screenshots
// successifs + crosshair de proposition DiffusionGemma + reason + validation
// utilisateur.

import AppKit
import SwiftUI

struct AgentView: View {
    @StateObject private var browser = WebBrowserHostController()
    @StateObject private var vm = AgentStepViewModel()
    @EnvironmentObject private var registry: ModelRegistry
    @State private var showHiddenBrowser: Bool = false
    @State private var viewerShowGrid: Bool = true   // bascule grid/clean dans le viewer central
    @State private var cacheClearedLabel: String? = nil  // feedback après clear

    var body: some View {
        VStack(spacing: 0) {
            topBar
            Divider().overlay(Color.white.opacity(0.1))
            HSplitView {
                // Timeline gauche
                timelineColumn
                    .frame(minWidth: 220, idealWidth: 260, maxWidth: 320)
                    .background(Color.black.opacity(0.35))

                // Viewer central
                stepViewer
                    .frame(minWidth: 500)
            }
            // Mini preview du WKWebView (debug live, repliable)
            if showHiddenBrowser {
                Divider().overlay(Color.white.opacity(0.1))
                miniBrowser
            }
        }
        // Le WKWebView doit etre dans la hierarchy pour fonctionner — on le
        // garde monté mais a taille zéro quand l'utilisateur ne veut pas le voir.
        .overlay(alignment: .bottomTrailing) {
            if !showHiddenBrowser {
                WebBrowserView(host: browser)
                    .frame(width: 1280, height: 800)
                    .opacity(0.001)
                    .allowsHitTesting(false)
                    .accessibilityHidden(true)
            }
        }
        .background(
            LinearGradient(
                colors: [
                    Color(red: 0.04, green: 0.04, blue: 0.08),
                    Color(red: 0.07, green: 0.04, blue: 0.12),
                ],
                startPoint: .topLeading, endPoint: .bottomTrailing
            )
            .ignoresSafeArea()
        )
    }

    // MARK: - Top bar
    private var topBar: some View {
        VStack(spacing: 8) {
            HStack(spacing: 10) {
                Image(systemName: "globe.americas.fill")
                    .foregroundStyle(.purple)
                    .font(.system(size: 14))
                Text("Web step-by-step")
                    .font(.system(size: 14, weight: .bold, design: .rounded))
                    .foregroundStyle(.white)
                Spacer()
                ModelStatusChip(kind: .diffusion)
                ModelControlButtons(kind: .diffusion)
                Toggle(isOn: $vm.useCoordGrid) {
                    Text("Grille de coords").font(.system(size: 10))
                }
                .toggleStyle(.switch)
                .controlSize(.mini)
                .help("Superpose une grille magenta normalisée 0..1 sur le screenshot envoyé au modèle. Améliore la précision du grounding (search bar Wikipedia : y=0.42 sans grille → y=0.03 avec grille).")
                Toggle(isOn: $showHiddenBrowser) {
                    Text("Voir le browser").font(.system(size: 10))
                }
                .toggleStyle(.switch)
                .controlSize(.mini)

                Button {
                    Task {
                        let count = await browser.clearAllBrowserData()
                        cacheClearedLabel = "Cache vidé (\(count) types)"
                        try? await Task.sleep(nanoseconds: 2_500_000_000)
                        cacheClearedLabel = nil
                    }
                } label: {
                    Label(cacheClearedLabel ?? "Vider cache",
                          systemImage: cacheClearedLabel == nil ? "trash" : "checkmark.circle.fill")
                        .font(.system(size: 10))
                }
                .buttonStyle(GlowButtonStyle(color: cacheClearedLabel == nil ? .orange : .green))
                .disabled(vm.isBusy)
                .help("Efface cookies, cache disque, localStorage, IndexedDB, etc. Utile après une page 'you have been blocked' ou pour repartir d'une session vierge.")
            }

            HStack(alignment: .top, spacing: 10) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Goal").font(.system(size: 9, weight: .semibold)).foregroundStyle(.cyan.opacity(0.85))
                    TextEditor(text: $vm.goal)
                        .font(.system(size: 11, design: .monospaced))
                        .scrollContentBackground(.hidden)
                        .padding(4)
                        .frame(minHeight: 44, maxHeight: 60)
                        .background(
                            RoundedRectangle(cornerRadius: 6)
                                .fill(Color.white.opacity(0.06))
                                .overlay(RoundedRectangle(cornerRadius: 6).strokeBorder(Color.cyan.opacity(0.3), lineWidth: 1))
                        )
                }
                VStack(alignment: .leading, spacing: 4) {
                    Text("URL initiale").font(.system(size: 9, weight: .semibold)).foregroundStyle(.cyan.opacity(0.85))
                    TextField("", text: $vm.startURL)
                        .textFieldStyle(.plain)
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 6).padding(.vertical, 6)
                        .background(
                            RoundedRectangle(cornerRadius: 6)
                                .fill(Color.white.opacity(0.06))
                                .overlay(RoundedRectangle(cornerRadius: 6).strokeBorder(Color.cyan.opacity(0.3), lineWidth: 1))
                        )
                }
                .frame(width: 280)
                VStack(spacing: 6) {
                    if vm.steps.isEmpty {
                        Button {
                            Task {
                                await vm.proposeNextStep(browser: browser, registry: registry)
                                if vm.autoPlay {
                                    await vm.runAutoPlay(browser: browser, registry: registry)
                                }
                            }
                        } label: {
                            Label("Lancer", systemImage: "play.fill")
                                .font(.system(size: 12, weight: .bold))
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(GlowButtonStyle(color: .purple))
                        .disabled(!vm.canProposeNext(registry: registry))
                    } else if vm.isBusy || vm.autoPlay {
                        // Bouton Stop bien visible — coupe la chaine autoplay
                        // immediatement (le step en cours termine, pas de suivant)
                        Button {
                            vm.requestStop()
                        } label: {
                            Label(vm.stopRequested ? "Arrêt en cours…" : "Stop",
                                  systemImage: vm.stopRequested ? "hourglass" : "stop.fill")
                                .font(.system(size: 12, weight: .bold))
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(GlowButtonStyle(color: .red))
                        .disabled(vm.stopRequested)
                        .help("Coupe la chaîne d'autoplay. Le step en cours (forward Diffusion ~25s) ne peut pas être interrompu mais aucun nouveau step ne sera lancé.")
                        if vm.isBusy {
                            HStack(spacing: 4) {
                                ProgressView().controlSize(.small).tint(.purple)
                                Text(vm.stopRequested ? "Termine le step…" : "Travail en cours…")
                                    .font(.system(size: 9)).foregroundStyle(.white.opacity(0.7))
                            }
                            .frame(maxWidth: .infinity)
                        }
                    }
                    Button {
                        vm.reset()
                    } label: {
                        Label("Reset", systemImage: "arrow.counterclockwise")
                            .font(.system(size: 10))
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(GlowButtonStyle(color: .orange))
                    .disabled(vm.steps.isEmpty || vm.isBusy)
                }
                .frame(width: 170)
            }

            // Ligne dediee au mode auto-play
            HStack(spacing: 12) {
                Toggle(isOn: $vm.autoPlay) {
                    HStack(spacing: 4) {
                        Image(systemName: "infinity").font(.system(size: 10)).foregroundStyle(.purple)
                        Text("Auto-play").font(.system(size: 10, weight: .semibold))
                    }
                }
                .toggleStyle(.switch).controlSize(.mini)
                .help("Si actif, l'agent valide automatiquement les propositions du modèle et enchaîne jusqu'à atteindre la limite de steps ou la fin du goal.")

                Text("Max steps").font(.system(size: 9)).foregroundStyle(.white.opacity(0.5))
                Stepper("\(vm.maxAutoSteps)", value: $vm.maxAutoSteps, in: 1 ... 30)
                    .labelsHidden().controlSize(.mini)
                Text("\(vm.maxAutoSteps)")
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.75))
                    .frame(width: 20)

                Spacer()

                Text("\(vm.steps.count) / \(vm.maxAutoSteps) steps")
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.55))
            }
        }
        .padding(12)
    }

    // MARK: - Timeline (gauche)
    private var timelineColumn: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text("Captures").font(.system(size: 11, weight: .semibold)).foregroundStyle(.white)
                Spacer()
                Text("\(vm.steps.count) step\(vm.steps.count > 1 ? "s" : "")")
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.55))
            }
            .padding(.horizontal, 12).padding(.vertical, 10)
            .background(Color.black.opacity(0.3))

            Divider().overlay(Color.white.opacity(0.1))

            if vm.steps.isEmpty {
                VStack(alignment: .leading, spacing: 6) {
                    Image(systemName: "photo.stack").font(.system(size: 24)).foregroundStyle(.white.opacity(0.35))
                    Text("Aucune capture pour le moment.")
                        .font(.system(size: 10)).foregroundStyle(.white.opacity(0.6))
                    Text("Charge le modèle puis clique Lancer step 1.")
                        .font(.system(size: 10)).foregroundStyle(.white.opacity(0.45))
                }
                .padding(14)
            } else {
                ScrollView {
                    LazyVStack(spacing: 8) {
                        ForEach(Array(vm.steps.enumerated()), id: \.element.id) { idx, step in
                            timelineThumbnail(step: step, index: idx)
                        }
                    }
                    .padding(10)
                }
            }
            Spacer()
        }
    }

    private func timelineThumbnail(step: AgentStepViewModel.StepCapture, index: Int) -> some View {
        let isSelected = vm.selectedStepIndex == index
        return Button {
            vm.selectedStepIndex = index
        } label: {
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("Step \(step.n)")
                        .font(.system(size: 10, weight: .bold, design: .rounded))
                        .foregroundStyle(.white)
                    stateBadge(step.state)
                    Spacer()
                    if let c = step.coordsProposed {
                        Text(String(format: "(%.2f, %.2f)", c.x, c.y))
                            .font(.system(size: 8, design: .monospaced))
                            .foregroundStyle(.green.opacity(0.7))
                    }
                }
                // Image with crosshair — on prend la meme source que le viewer
                // (avec grille si dispo et state != validated) pour que sidebar
                // et viewer affichent EXACTEMENT la meme chose -> crosshair pile.
                ZStack {
                    let img: NSImage = {
                        if step.state == .validated, let after = step.screenshotAfter {
                            return after
                        }
                        if let g = step.screenshotGrid { return g }
                        return step.screenshotBefore
                    }()
                    Image(nsImage: img)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                    if let c = step.coordsProposed, step.state != .validated {
                        GeometryReader { geo in
                            crosshair(small: true)
                                .position(x: c.x * geo.size.width, y: c.y * geo.size.height)
                        }
                    }
                }
                .aspectRatio(step.screenshotBefore.size.width / max(1, step.screenshotBefore.size.height), contentMode: .fit)
                .frame(maxWidth: .infinity)
                .clipShape(RoundedRectangle(cornerRadius: 5))
                .overlay(RoundedRectangle(cornerRadius: 5).strokeBorder(isSelected ? .purple : .white.opacity(0.15), lineWidth: isSelected ? 2 : 1))

                if !step.reason.isEmpty {
                    Text(step.reason)
                        .font(.system(size: 9))
                        .foregroundStyle(.white.opacity(0.6))
                        .lineLimit(2)
                }
            }
            .padding(7)
            .background(
                RoundedRectangle(cornerRadius: 7)
                    .fill(isSelected ? Color.purple.opacity(0.18) : Color.white.opacity(0.04))
                    .overlay(RoundedRectangle(cornerRadius: 7).strokeBorder((isSelected ? Color.purple : Color.white).opacity(0.3), lineWidth: 1))
            )
        }
        .buttonStyle(.plain)
    }

    private func stateBadge(_ s: AgentStepViewModel.StepState) -> some View {
        let (icon, color, label): (String, Color, String) = {
            switch s {
            case .proposing:  return ("hourglass", .cyan,   "proposing")
            case .awaiting:   return ("questionmark.diamond.fill", .yellow, "awaiting")
            case .validated:  return ("checkmark.circle.fill", .green,  "validated")
            case .rejected:   return ("xmark.circle.fill", .red,    "rejected")
            case .errored:    return ("exclamationmark.triangle.fill", .orange, "error")
            }
        }()
        return HStack(spacing: 3) {
            Image(systemName: icon).font(.system(size: 8))
            Text(label).font(.system(size: 8, weight: .semibold))
        }
        .foregroundStyle(color)
        .padding(.horizontal, 4).padding(.vertical, 1)
        .background(Capsule().fill(color.opacity(0.18))
            .overlay(Capsule().strokeBorder(color.opacity(0.5), lineWidth: 0.5)))
    }

    // MARK: - Step viewer (centre)
    private var stepViewer: some View {
        VStack(spacing: 0) {
            if let step = vm.currentStep {
                stepViewerContent(step)
            } else {
                emptyViewer
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var emptyViewer: some View {
        VStack(alignment: .center, spacing: 14) {
            Image(systemName: "rectangle.grid.2x2").font(.system(size: 48)).foregroundStyle(.white.opacity(0.25))
            Text("POC step-by-step navigation")
                .font(.system(size: 16, weight: .semibold)).foregroundStyle(.white.opacity(0.85))
            Text("Le modèle voit le screenshot du navigateur, te propose UN click avec une raison, tu valides ou tu rejettes.\nL'objectif est de démontrer que DiffusionGemma seul peut conduire une navigation simple.")
                .font(.system(size: 12))
                .foregroundStyle(.white.opacity(0.55))
                .multilineTextAlignment(.center)
                .padding(.horizontal, 30)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    @ViewBuilder
    private func stepViewerContent(_ step: AgentStepViewModel.StepCapture) -> some View {
        VStack(spacing: 10) {
            // Header
            HStack(alignment: .center, spacing: 8) {
                Text("Step \(step.n)").font(.system(size: 14, weight: .bold, design: .rounded))
                    .foregroundStyle(.white)
                stateBadge(step.state)
                Spacer()
                if let elapsed = step.elapsedDiffusion {
                    Label(String(format: "%.1fs Diff", elapsed), systemImage: "clock")
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(.white.opacity(0.6))
                }
                if let fwd = step.diffusionForwards {
                    Label("\(fwd) forwards", systemImage: "waveform")
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(.white.opacity(0.6))
                }
            }
            .padding(.horizontal, 14).padding(.top, 10)

            // URL / title
            HStack(spacing: 6) {
                Image(systemName: "link").foregroundStyle(.white.opacity(0.5)).font(.system(size: 10))
                Text(step.url).font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.6)).lineLimit(1)
                Spacer()
                if !step.pageTitle.isEmpty {
                    Text(step.pageTitle).font(.system(size: 10))
                        .foregroundStyle(.white.opacity(0.5)).lineLimit(1)
                }
            }
            .padding(.horizontal, 14)

            // Toggle clean/grid pour voir ce que le modele voyait
            if step.screenshotGrid != nil {
                HStack(spacing: 6) {
                    Image(systemName: "grid").font(.system(size: 10)).foregroundStyle(.purple)
                    Toggle(isOn: $viewerShowGrid) {
                        Text("Afficher la grille de coords (ce que le modèle a vu)").font(.system(size: 10))
                    }
                    .toggleStyle(.switch).controlSize(.mini)
                    Spacer()
                }
                .padding(.horizontal, 14)
            }

            // Screenshot grand + crosshair
            GeometryReader { outer in
                let img: NSImage = {
                    if viewerShowGrid, let g = step.screenshotGrid { return g }
                    return step.screenshotBefore
                }()
                let aspect = img.size.width / max(1, img.size.height)
                let fitW = min(outer.size.width, outer.size.height * aspect)
                let fitH = fitW / aspect
                ZStack {
                    Color.black.opacity(0.3)
                    ZStack {
                        Image(nsImage: img)
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                        if let c = step.coordsProposed {
                            GeometryReader { geo in
                                crosshair(small: false)
                                    .position(x: c.x * geo.size.width, y: c.y * geo.size.height)
                            }
                        }
                    }
                    .frame(width: fitW, height: fitH)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                    .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(Color.purple.opacity(0.5), lineWidth: 1))
                    .shadow(color: .purple.opacity(0.25), radius: 8)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .padding(.horizontal, 14)

            // Reason + actions
            VStack(alignment: .leading, spacing: 8) {
                if step.state == .proposing {
                    HStack(spacing: 8) {
                        ProgressView().controlSize(.small).tint(.purple)
                        Text("DiffusionGemma analyse le screenshot…").font(.system(size: 11)).foregroundStyle(.white.opacity(0.75))
                    }
                } else if let err = step.error {
                    Text("⚠ \(err)").font(.system(size: 11, design: .monospaced)).foregroundStyle(.red)
                } else {
                    HStack(alignment: .top, spacing: 8) {
                        Image(systemName: "lightbulb.fill").foregroundStyle(.yellow).font(.system(size: 12))
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Proposition du modèle").font(.system(size: 9, weight: .semibold)).foregroundStyle(.yellow.opacity(0.85))
                            if !step.notes.isEmpty {
                                HStack(alignment: .top, spacing: 4) {
                                    Image(systemName: "note.text").font(.system(size: 9)).foregroundStyle(.cyan)
                                    Text(step.notes)
                                        .font(.system(size: 11))
                                        .foregroundStyle(.cyan.opacity(0.85))
                                        .textSelection(.enabled)
                                }
                            }
                            Text(step.reason.isEmpty ? "(pas de raison renvoyée)" : step.reason)
                                .font(.system(size: 12))
                                .foregroundStyle(.white)
                                .textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                            actionBadge(for: step)
                            if let c = step.coordsProposed {
                                Text(String(format: "→ CLICK normalisé (%.3f, %.3f)", c.x, c.y))
                                    .font(.system(size: 10, design: .monospaced))
                                    .foregroundStyle(.green.opacity(0.8))
                            } else if case .click = step.actionProposed {
                                VStack(alignment: .leading, spacing: 4) {
                                    Text("⚠ coordonnées non parsables — sortie brute du modèle :")
                                        .font(.system(size: 10, design: .monospaced))
                                        .foregroundStyle(.orange)
                                    ScrollView {
                                        Text(step.rawOutput.isEmpty ? "(vide)" : step.rawOutput)
                                            .font(.system(size: 10, design: .monospaced))
                                            .foregroundStyle(.white.opacity(0.65))
                                            .textSelection(.enabled)
                                            .frame(maxWidth: .infinity, alignment: .leading)
                                    }
                                    .frame(maxHeight: 110)
                                    .padding(6)
                                    .background(
                                        RoundedRectangle(cornerRadius: 5).fill(Color.black.opacity(0.4))
                                            .overlay(RoundedRectangle(cornerRadius: 5).strokeBorder(Color.orange.opacity(0.4), lineWidth: 1))
                                    )
                                }
                            }
                            if let hit = step.elementHit {
                                Text("→ élément touché : \(hit)")
                                    .font(.system(size: 10, design: .monospaced))
                                    .foregroundStyle(.cyan.opacity(0.85))
                                    .lineLimit(2)
                            }
                        }
                    }
                }

                // Carte synthese finale quand l'agent a emis 'done'
                if case .done(let summary) = step.actionProposed, step.state == .validated {
                    finalSummaryView(summary)
                }

                if step.state == .awaiting {
                    HStack(spacing: 8) {
                        let (label, icon) = validateButtonLabel(for: step.actionProposed)
                        Button {
                            Task {
                                await vm.validateAndClick(browser: browser, registry: registry, chainNext: true)
                            }
                        } label: {
                            Label(label, systemImage: icon)
                                .font(.system(size: 12, weight: .semibold))
                        }
                        .buttonStyle(GlowButtonStyle(color: .green))
                        .disabled(step.actionProposed == nil || vm.isBusy)
                        .help("Exécute l'action proposée puis enchaîne sur le step suivant (sauf si action=done).")

                        Button {
                            Task { await vm.rejectAndRetry(browser: browser, registry: registry) }
                        } label: {
                            Label("Rejeter — re-proposer", systemImage: "arrow.triangle.2.circlepath")
                                .font(.system(size: 11))
                        }
                        .buttonStyle(GlowButtonStyle(color: .orange))
                        .disabled(vm.isBusy)

                        Button {
                            vm.rejectAndSkip()
                        } label: {
                            Label("Skip", systemImage: "forward.end.fill")
                                .font(.system(size: 11))
                        }
                        .buttonStyle(GlowButtonStyle(color: .red))
                        .disabled(vm.isBusy)
                        Spacer()
                    }
                }
            }
            .padding(12)
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .fill(Color.white.opacity(0.05))
                    .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(Color.white.opacity(0.15), lineWidth: 1))
            )
            .padding(.horizontal, 14).padding(.bottom, 12)
        }
    }

    // MARK: - Mini browser
    private var miniBrowser: some View {
        VStack(spacing: 4) {
            HStack {
                Text("Browser live (debug)").font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(.white.opacity(0.7))
                Spacer()
                Text(browser.currentURL).font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.55)).lineLimit(1)
            }
            .padding(.horizontal, 10).padding(.top, 6)
            WebBrowserView(host: browser)
                .frame(height: 220)
        }
        .background(Color.black.opacity(0.4))
    }

    // MARK: - Action badge / labels
    @ViewBuilder
    private func actionBadge(for step: AgentStepViewModel.StepCapture) -> some View {
        if let action = step.actionProposed {
            let (label, icon, color): (String, String, Color) = {
                switch action {
                case .click: return ("CLICK", "hand.point.up.fill", .green)
                case .scroll(let d): return (d == .up ? "SCROLL UP" : "SCROLL DOWN", "arrow.up.arrow.down", .orange)
                case .type(let text, let press):
                    let extra = press ? " ⏎" : ""
                    return ("TYPE \"\(text.prefix(20))\"\(extra)", "keyboard", .cyan)
                case .clickAndType(_, _, let text, let press):
                    let extra = press ? " ⏎" : ""
                    return ("CLICK + TYPE \"\(text.prefix(20))\"\(extra)", "hand.point.up.left.and.text.fill", .teal)
                case .done: return ("DONE", "checkmark.seal.fill", .purple)
                }
            }()
            HStack(spacing: 5) {
                Image(systemName: icon).font(.system(size: 10))
                Text(label).font(.system(size: 10, weight: .bold, design: .rounded))
            }
            .foregroundStyle(color)
            .padding(.horizontal, 6).padding(.vertical, 2)
            .background(
                Capsule().fill(color.opacity(0.15))
                    .overlay(Capsule().strokeBorder(color.opacity(0.5), lineWidth: 1))
            )
        }
    }

    private func validateButtonLabel(for action: AgentStepViewModel.StepAction?) -> (String, String) {
        guard let action = action else { return ("Valider", "checkmark.circle.fill") }
        switch action {
        case .click: return ("Valider, cliquer et continuer", "hand.point.up.fill")
        case .scroll(let d): return (d == .up ? "Valider et scroller vers le haut" : "Valider et scroller vers le bas", "arrow.up.arrow.down.circle.fill")
        case .type(_, let press): return (press ? "Valider, taper, Enter et continuer" : "Valider, taper et continuer", "keyboard")
        case .clickAndType(_, _, _, let press): return (press ? "Valider : cliquer, taper, Enter" : "Valider : cliquer et taper", "hand.point.up.left.and.text.fill")
        case .done: return ("Valider la fin de session", "checkmark.seal.fill")
        }
    }

    @ViewBuilder
    private func finalSummaryView(_ summary: String) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                Image(systemName: "sparkles").foregroundStyle(.purple).font(.system(size: 14))
                Text("Synthèse finale de l'agent")
                    .font(.system(size: 12, weight: .bold, design: .rounded))
                    .foregroundStyle(.white)
                Spacer()
                Button {
                    let pb = NSPasteboard.general
                    pb.clearContents()
                    pb.setString(summary, forType: .string)
                } label: {
                    Label("Copier", systemImage: "doc.on.doc").font(.system(size: 10))
                }
                .buttonStyle(.borderless)
                .foregroundStyle(.purple)
            }
            ScrollView {
                Text(summary)
                    .font(.system(size: 12))
                    .foregroundStyle(.white)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
            }
            .frame(minHeight: 60, maxHeight: 200)
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(Color.purple.opacity(0.18))
                .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(Color.purple.opacity(0.7), lineWidth: 1.5))
        )
        .shadow(color: .purple.opacity(0.4), radius: 8)
    }

    // MARK: - Crosshair
    @ViewBuilder
    private func crosshair(small: Bool) -> some View {
        let size: CGFloat = small ? 14 : 24
        ZStack {
            Circle().stroke(Color.black.opacity(0.85), lineWidth: small ? 2 : 3).frame(width: size, height: size)
            Circle().stroke(Color.green, lineWidth: small ? 1 : 1.5).frame(width: size, height: size)
            Rectangle().fill(Color.green).frame(width: size - 4, height: 1)
            Rectangle().fill(Color.green).frame(width: 1, height: size - 4)
            Circle().fill(Color.green).frame(width: small ? 2 : 3, height: small ? 2 : 3)
        }
        .shadow(color: .black, radius: 1)
    }
}
