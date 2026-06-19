// Onglet iOS Simulator : 2 modes
//   - Agent  : pipeline DiffusionGemma step-by-step (similaire au web Agent)
//   - Test   : sliders, screenshot manuel, log technique
//
// Le top bar (status, Launch, Permission, mode picker) est commun aux 2 modes.

import AppKit
import SwiftUI

struct IOSSimulatorView: View {
    @StateObject private var host = IOSSimulatorHostController()
    @StateObject private var vm = IOSAgentStepViewModel()
    @EnvironmentObject private var registry: ModelRegistry
    @State private var mode: Mode = .agent
    @State private var tapX: Double = 0.5
    @State private var tapY: Double = 0.5
    @State private var viewerShowGrid: Bool = true

    enum Mode: String, CaseIterable, Identifiable {
        case agent = "Agent"
        case test = "Test technique"
        var id: String { rawValue }
    }

    var body: some View {
        VStack(spacing: 0) {
            topBar
            Divider().overlay(Color.white.opacity(0.1))
            switch mode {
            case .agent: agentPanel
            case .test:  testPanel
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
        .task { await host.refreshState() }
    }

    // MARK: - Top bar commun
    private var topBar: some View {
        VStack(spacing: 8) {
            HStack(spacing: 10) {
                Image(systemName: "iphone.gen3").foregroundStyle(.purple).font(.system(size: 14))
                Text("iOS Simulator agent").font(.system(size: 14, weight: .bold, design: .rounded))
                    .foregroundStyle(.white)
                Spacer()
                HStack(spacing: 4) {
                    Circle().fill(host.connection.color).frame(width: 7, height: 7)
                    Text(host.connection.statusLabel)
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(.white.opacity(0.7))
                        .lineLimit(1)
                    Button {
                        Task { await host.refreshState() }
                    } label: {
                        Image(systemName: "arrow.clockwise").font(.system(size: 10))
                    }
                    .buttonStyle(.borderless)
                    .foregroundStyle(.white.opacity(0.7))
                }
                Button {
                    Task { await host.launchSimulator() }
                } label: {
                    Label("Launch Simulator", systemImage: "play.fill").font(.system(size: 10))
                }
                .buttonStyle(GlowButtonStyle(color: .purple))
                .disabled(host.isBusy)

                // Permission AX
                HStack(spacing: 4) {
                    Image(systemName: host.hasAccessibilityPermission ? "checkmark.shield.fill" : "exclamationmark.shield")
                        .font(.system(size: 10))
                        .foregroundStyle(host.hasAccessibilityPermission ? .green : .orange)
                    Text(host.hasAccessibilityPermission ? "AX OK" : "AX manquant")
                        .font(.system(size: 9, weight: .semibold))
                        .foregroundStyle(host.hasAccessibilityPermission ? .green : .orange)
                    if !host.hasAccessibilityPermission {
                        Button("Demander") { host.requestAccessibilityPermission() }
                            .font(.system(size: 9)).buttonStyle(.bordered).controlSize(.mini)
                    }
                }
            }
            HStack(spacing: 8) {
                ModelStatusChip(kind: .diffusion)
                ModelControlButtons(kind: .diffusion)
                Spacer()
                Picker("", selection: $mode) {
                    ForEach(Mode.allCases) { m in Text(m.rawValue).tag(m) }
                }
                .pickerStyle(.segmented)
                .frame(maxWidth: 220)
            }
        }
        .padding(12)
    }

    // MARK: - Mode Agent
    private var agentPanel: some View {
        VStack(spacing: 0) {
            agentControlBar
            Divider().overlay(Color.white.opacity(0.1))
            HSplitView {
                timelineColumn
                    .frame(minWidth: 220, idealWidth: 260, maxWidth: 320)
                    .background(Color.black.opacity(0.35))
                stepViewer
                    .frame(minWidth: 500)
            }
        }
    }

    private var agentControlBar: some View {
        VStack(spacing: 8) {
            HStack(alignment: .top, spacing: 10) {
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text("Goal").font(.system(size: 9, weight: .semibold)).foregroundStyle(.cyan.opacity(0.85))
                        Spacer()
                        Menu {
                            ForEach(IOSAppPreset.Category.allCases, id: \.self) { cat in
                                let presets = IOSAppPreset.presets(in: cat)
                                if !presets.isEmpty {
                                    Section(cat.rawValue) {
                                        ForEach(presets) { preset in
                                            Button {
                                                vm.goal = preset.goal
                                                vm.appContext = preset.appContext
                                            } label: {
                                                Label(preset.name, systemImage: preset.icon)
                                            }
                                        }
                                    }
                                }
                            }
                            Divider()
                            Button {
                                vm.appContext = ""
                            } label: {
                                Label("Effacer l'app context", systemImage: "trash")
                            }
                        } label: {
                            Label("Preset", systemImage: "list.bullet.rectangle")
                                .font(.system(size: 9))
                        }
                        .menuStyle(.borderlessButton)
                        .controlSize(.mini)
                    }
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

                    Text("App context (instructions spécifiques)")
                        .font(.system(size: 9, weight: .semibold))
                        .foregroundStyle(.orange.opacity(0.85))
                    TextEditor(text: $vm.appContext)
                        .font(.system(size: 10, design: .monospaced))
                        .scrollContentBackground(.hidden)
                        .padding(4)
                        .frame(minHeight: 60, maxHeight: 140)
                        .background(
                            RoundedRectangle(cornerRadius: 6)
                                .fill(Color.white.opacity(0.06))
                                .overlay(RoundedRectangle(cornerRadius: 6).strokeBorder(Color.orange.opacity(0.3), lineWidth: 1))
                        )
                }
                VStack(spacing: 6) {
                    if vm.steps.isEmpty {
                        Button {
                            Task {
                                await vm.proposeNextStep(host: host, registry: registry)
                                if vm.autoPlay {
                                    await vm.runAutoPlay(host: host, registry: registry)
                                }
                            }
                        } label: {
                            Label("Lancer", systemImage: "play.fill")
                                .font(.system(size: 12, weight: .bold))
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(GlowButtonStyle(color: .purple))
                        .disabled(!vm.canPropose(host: host, registry: registry))
                    } else if vm.isBusy || vm.autoPlay {
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
                    }
                    Button { vm.reset() } label: {
                        Label("Reset", systemImage: "arrow.counterclockwise")
                            .font(.system(size: 10)).frame(maxWidth: .infinity)
                    }
                    .buttonStyle(GlowButtonStyle(color: .orange))
                    .disabled(vm.steps.isEmpty || vm.isBusy)
                }
                .frame(width: 170)
            }
            HStack(spacing: 12) {
                Toggle(isOn: $vm.autoPlay) {
                    HStack(spacing: 4) {
                        Image(systemName: "infinity").font(.system(size: 10)).foregroundStyle(.purple)
                        Text("Auto-play").font(.system(size: 10, weight: .semibold))
                    }
                }
                .toggleStyle(.switch).controlSize(.mini)
                Text("Max steps").font(.system(size: 9)).foregroundStyle(.white.opacity(0.5))
                Stepper("\(vm.maxAutoSteps)", value: $vm.maxAutoSteps, in: 1 ... 30)
                    .labelsHidden().controlSize(.mini)
                Text("\(vm.maxAutoSteps)").font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.75)).frame(width: 20)
                Toggle(isOn: $vm.useCoordGrid) {
                    Text("Grille de coords").font(.system(size: 10))
                }
                .toggleStyle(.switch).controlSize(.mini)
                Toggle(isOn: $vm.useFullPageCapture) {
                    Text("Capture composite").font(.system(size: 10))
                }
                .toggleStyle(.switch).controlSize(.mini)
                .help("Le screenshot envoyé au modèle est composite : zone haut + séparateur jaune + zone bas après scroll. Le modèle voit tout d'un coup et tape directement la bonne zone.")
                if let logDir = vm.sessionLogDir {
                    Button {
                        NSWorkspace.shared.open(logDir)
                    } label: {
                        Label("Voir les logs", systemImage: "folder")
                            .font(.system(size: 9))
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.mini)
                    .help("Ouvre le dossier contenant les prompts envoyés et sorties brutes du modèle pour chaque step.")
                }
                Spacer()
                Text("\(vm.steps.count) / \(vm.maxAutoSteps) steps")
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.55))
            }
        }
        .padding(12)
    }

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
                Text("Aucune capture pour le moment. Charge DiffusionGemma, lance le Simulator puis clique Lancer.")
                    .font(.system(size: 10)).foregroundStyle(.white.opacity(0.5))
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

    private func timelineThumbnail(step: IOSAgentStepViewModel.StepCapture, index: Int) -> some View {
        let isSelected = vm.selectedStepIndex == index
        return Button {
            vm.selectedStepIndex = index
        } label: {
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("Step \(step.n)").font(.system(size: 10, weight: .bold, design: .rounded))
                        .foregroundStyle(.white)
                    stateBadge(step.state)
                    Spacer()
                    if let c = step.coordsProposed {
                        Text(String(format: "(%.2f, %.2f)", c.x, c.y))
                            .font(.system(size: 8, design: .monospaced))
                            .foregroundStyle(.green.opacity(0.7))
                    }
                }
                ZStack {
                    let img: NSImage = {
                        if step.state == .validated, let a = step.screenshotAfter { return a }
                        if let g = step.screenshotGrid { return g }
                        return step.screenshotBefore
                    }()
                    Image(nsImage: img).resizable().aspectRatio(contentMode: .fit)
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
                    Text(step.reason).font(.system(size: 9))
                        .foregroundStyle(.white.opacity(0.6)).lineLimit(2)
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

    private func stateBadge(_ s: IOSAgentStepViewModel.StepState) -> some View {
        let (icon, color, label): (String, Color, String) = {
            switch s {
            case .proposing: return ("hourglass", .cyan, "proposing")
            case .awaiting:  return ("questionmark.diamond.fill", .yellow, "awaiting")
            case .validated: return ("checkmark.circle.fill", .green, "validated")
            case .rejected:  return ("xmark.circle.fill", .red, "rejected")
            case .errored:   return ("exclamationmark.triangle.fill", .orange, "error")
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

    private var stepViewer: some View {
        VStack(spacing: 0) {
            if let step = vm.currentStep {
                stepViewerContent(step)
            } else {
                VStack(spacing: 14) {
                    Image(systemName: "iphone.gen3").font(.system(size: 48)).foregroundStyle(.white.opacity(0.25))
                    Text("iOS step-by-step navigation")
                        .font(.system(size: 16, weight: .semibold)).foregroundStyle(.white.opacity(0.85))
                    Text("Connecte le Simulator, charge DiffusionGemma, écris ton goal, clique Lancer.")
                        .font(.system(size: 12)).foregroundStyle(.white.opacity(0.55))
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
    }

    @ViewBuilder
    private func stepViewerContent(_ step: IOSAgentStepViewModel.StepCapture) -> some View {
        VStack(spacing: 10) {
            HStack(spacing: 8) {
                Text("Step \(step.n)").font(.system(size: 14, weight: .bold, design: .rounded))
                    .foregroundStyle(.white)
                stateBadge(step.state)
                Spacer()
                if let e = step.elapsedDiffusion {
                    Label(String(format: "%.1fs Diff", e), systemImage: "clock")
                        .font(.system(size: 10, design: .monospaced)).foregroundStyle(.white.opacity(0.6))
                }
            }
            .padding(.horizontal, 14).padding(.top, 10)

            if step.screenshotGrid != nil {
                HStack {
                    Toggle(isOn: $viewerShowGrid) {
                        Text("Afficher la grille").font(.system(size: 10))
                    }
                    .toggleStyle(.switch).controlSize(.mini)
                    Spacer()
                }
                .padding(.horizontal, 14)
            }

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
                        Image(nsImage: img).resizable().aspectRatio(contentMode: .fit)
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

            VStack(alignment: .leading, spacing: 8) {
                if let s = vm.finalSummary {
                    finalSummaryView(s)
                }
                if step.state == .proposing {
                    HStack(spacing: 8) {
                        ProgressView().controlSize(.small).tint(.purple)
                        Text("DiffusionGemma analyse l'écran…")
                            .font(.system(size: 11)).foregroundStyle(.white.opacity(0.75))
                    }
                } else {
                    HStack(alignment: .top, spacing: 8) {
                        Image(systemName: "lightbulb.fill").foregroundStyle(.yellow).font(.system(size: 12))
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Proposition du modèle")
                                .font(.system(size: 9, weight: .semibold)).foregroundStyle(.yellow.opacity(0.85))
                            if !step.notes.isEmpty {
                                HStack(alignment: .top, spacing: 4) {
                                    Image(systemName: "note.text").font(.system(size: 9)).foregroundStyle(.cyan)
                                    Text(step.notes).font(.system(size: 11))
                                        .foregroundStyle(.cyan.opacity(0.85))
                                }
                            }
                            Text(step.reason).font(.system(size: 12)).foregroundStyle(.white)
                                .textSelection(.enabled)
                            actionBadge(for: step)
                            if let c = step.coordsProposed {
                                Text(String(format: "→ TAP (%.3f, %.3f)", c.x, c.y))
                                    .font(.system(size: 10, design: .monospaced))
                                    .foregroundStyle(.green.opacity(0.8))
                            }
                        }
                    }
                }
                if step.state == .awaiting {
                    HStack(spacing: 8) {
                        let (label, icon) = validateButtonLabel(for: step.actionProposed)
                        Button {
                            Task { await vm.validateAndExecute(host: host, registry: registry, chainNext: true) }
                        } label: {
                            Label(label, systemImage: icon).font(.system(size: 12, weight: .semibold))
                        }
                        .buttonStyle(GlowButtonStyle(color: .green))
                        .disabled(step.actionProposed == nil || vm.isBusy)
                        Button {
                            Task { await vm.rejectAndRetry(host: host, registry: registry) }
                        } label: {
                            Label("Rejeter — re-proposer", systemImage: "arrow.triangle.2.circlepath")
                                .font(.system(size: 11))
                        }
                        .buttonStyle(GlowButtonStyle(color: .orange))
                        .disabled(vm.isBusy)
                        Button {
                            vm.rejectAndSkip()
                        } label: {
                            Label("Skip", systemImage: "forward.end.fill").font(.system(size: 11))
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

    // MARK: - Helpers reusable
    @ViewBuilder
    private func actionBadge(for step: IOSAgentStepViewModel.StepCapture) -> some View {
        if let action = step.actionProposed {
            let (label, icon, color): (String, String, Color) = {
                switch action {
                case .tap: return ("TAP", "hand.point.up.fill", .green)
                case .scroll(let d):
                    let arrow = ["down": "arrow.down", "up": "arrow.up", "left": "arrow.left", "right": "arrow.right"][d.rawValue] ?? "arrow.up.arrow.down"
                    return ("SCROLL \(d.rawValue.uppercased())", arrow, .orange)
                case .type(let t, let e):
                    return ("TYPE \"\(t.prefix(20))\"\(e ? " ⏎" : "")", "keyboard", .cyan)
                case .tapAndType(_, _, let t, let e):
                    return ("TAP + TYPE \"\(t.prefix(20))\"\(e ? " ⏎" : "")", "hand.point.up.left.and.text.fill", .teal)
                case .tapThenTap:
                    return ("TAP + TAP", "hand.tap.fill", .mint)
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

    private func validateButtonLabel(for action: IOSAgentStepViewModel.StepAction?) -> (String, String) {
        guard let action = action else { return ("Valider", "checkmark.circle.fill") }
        switch action {
        case .tap: return ("Valider, taper et continuer", "hand.point.up.fill")
        case .scroll(let d):
            let fr = ["down": "vers le bas", "up": "vers le haut", "left": "vers la gauche", "right": "vers la droite"][d.rawValue] ?? d.rawValue
            return ("Valider et scroller \(fr)", "arrow.up.arrow.down.circle.fill")
        case .type: return ("Valider, taper et continuer", "keyboard")
        case .tapAndType: return ("Valider : taper, écrire et Entrée", "hand.point.up.left.and.text.fill")
        case .tapThenTap: return ("Valider : 2 taps enchaînés", "hand.tap.fill")
        case .done: return ("Valider la fin de session", "checkmark.seal.fill")
        }
    }

    @ViewBuilder
    private func finalSummaryView(_ summary: String) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                Image(systemName: "sparkles").foregroundStyle(.purple)
                Text("Synthèse finale").font(.system(size: 11, weight: .bold, design: .rounded))
                    .foregroundStyle(.white)
                Spacer()
                Button {
                    let pb = NSPasteboard.general
                    pb.clearContents()
                    pb.setString(summary, forType: .string)
                } label: { Label("Copier", systemImage: "doc.on.doc").font(.system(size: 9)) }
                    .buttonStyle(.borderless).foregroundStyle(.purple)
            }
            Text(summary).font(.system(size: 11)).foregroundStyle(.white)
                .textSelection(.enabled)
        }
        .padding(10)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(Color.purple.opacity(0.18))
                .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(Color.purple.opacity(0.7), lineWidth: 1.5))
        )
    }

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
    }

    // MARK: - Mode Test
    private var testPanel: some View {
        HSplitView {
            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    Text("Test technique").font(.system(size: 12, weight: .semibold)).foregroundStyle(.white)

                    Button {
                        Task { _ = await host.screenshot() }
                    } label: {
                        Label("Screenshot", systemImage: "camera.fill").font(.system(size: 11))
                    }
                    .buttonStyle(GlowButtonStyle(color: .blue))
                    .disabled(!host.connection.isReadyForActions)

                    if let frame = host.simulatorWindowFrame() {
                        Text(String(format: "Fenêtre Simulator @ (%.0f, %.0f) %.0f×%.0f",
                                    frame.origin.x, frame.origin.y, frame.width, frame.height))
                            .font(.system(size: 9, design: .monospaced))
                            .foregroundStyle(.white.opacity(0.5))
                    }

                    Divider().overlay(Color.white.opacity(0.1))

                    Text("Tap test").font(.system(size: 10, weight: .semibold)).foregroundStyle(.green.opacity(0.85))
                    HStack {
                        Text("x").font(.system(size: 10)).foregroundStyle(.white.opacity(0.6)).frame(width: 12)
                        Slider(value: $tapX, in: 0 ... 1, step: 0.01)
                        Text(String(format: "%.2f", tapX)).font(.system(size: 10, design: .monospaced))
                            .foregroundStyle(.white).frame(width: 32)
                    }
                    HStack {
                        Text("y").font(.system(size: 10)).foregroundStyle(.white.opacity(0.6)).frame(width: 12)
                        Slider(value: $tapY, in: 0 ... 1, step: 0.01)
                        Text(String(format: "%.2f", tapY)).font(.system(size: 10, design: .monospaced))
                            .foregroundStyle(.white).frame(width: 32)
                    }
                    Button {
                        Task { await host.tap(normalizedX: tapX, normalizedY: tapY) }
                    } label: {
                        Label("Tap", systemImage: "hand.tap.fill")
                            .font(.system(size: 11, weight: .semibold))
                    }
                    .buttonStyle(GlowButtonStyle(color: .green))
                    .disabled(!host.connection.isReadyForActions)

                    Divider().overlay(Color.white.opacity(0.1))

                    Text("Log").font(.system(size: 10, weight: .semibold)).foregroundStyle(.white.opacity(0.55))
                    ScrollView {
                        VStack(alignment: .leading, spacing: 1) {
                            ForEach(Array(host.logLines.enumerated()), id: \.offset) { _, line in
                                Text(line).font(.system(size: 9, design: .monospaced))
                                    .foregroundStyle(.white.opacity(0.7))
                                    .frame(maxWidth: .infinity, alignment: .leading)
                            }
                        }
                        .padding(6)
                    }
                    .frame(maxHeight: 240)
                    .background(
                        RoundedRectangle(cornerRadius: 5).fill(Color.black.opacity(0.4))
                            .overlay(RoundedRectangle(cornerRadius: 5).strokeBorder(Color.white.opacity(0.15), lineWidth: 1))
                    )
                }
                .padding(14)
            }
            .frame(minWidth: 320, maxWidth: 400)
            .background(Color.black.opacity(0.35))

            // Screenshot avec crosshair test
            VStack {
                if let img = host.lastScreenshot {
                    GeometryReader { geo in
                        let aspect = img.size.width / max(1, img.size.height)
                        let fitW = min(geo.size.width, geo.size.height * aspect)
                        let fitH = fitW / aspect
                        ZStack {
                            Color.black.opacity(0.3)
                            ZStack {
                                Image(nsImage: img).resizable().aspectRatio(contentMode: .fit)
                                GeometryReader { inner in
                                    crosshair(small: false)
                                        .position(x: tapX * inner.size.width, y: tapY * inner.size.height)
                                }
                            }
                            .frame(width: fitW, height: fitH)
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                            .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(Color.purple.opacity(0.5), lineWidth: 1))
                        }
                    }
                    .padding(14)
                } else {
                    VStack(spacing: 8) {
                        Image(systemName: "iphone.slash").font(.system(size: 48)).foregroundStyle(.white.opacity(0.25))
                        Text("Aucun screenshot").font(.system(size: 12, weight: .semibold))
                            .foregroundStyle(.white.opacity(0.6))
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
        }
    }
}
