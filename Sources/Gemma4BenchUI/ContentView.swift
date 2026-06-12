// ContentView : 2 panneaux côte à côte (AR + Diffusion) + prompt + stats
// Style sombre, glow autour des panneaux actifs.

import SwiftUI

struct ContentView: View {
    @StateObject private var vm = BenchViewModel()

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider().overlay(Color.white.opacity(0.1))
            controlBar
            Divider().overlay(Color.white.opacity(0.1))
            panels
            Divider().overlay(Color.white.opacity(0.1))
            statusBar
        }
        .background(
            LinearGradient(
                colors: [
                    Color(red: 0.04, green: 0.04, blue: 0.08),
                    Color(red: 0.07, green: 0.04, blue: 0.12),
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            .ignoresSafeArea()
        )
    }

    // MARK: - Sections

    private var header: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text("Gemma 4 — AR vs Diffusion")
                    .font(.system(size: 18, weight: .bold, design: .rounded))
                    .foregroundStyle(.white)
                Text("Block-AR diffusion vs autoregressive streaming, même prompt, mêmes 26B params")
                    .font(.system(size: 11))
                    .foregroundStyle(.white.opacity(0.5))
            }
            Spacer()
            loadButton
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 14)
    }

    private var loadButton: some View {
        HStack(spacing: 12) {
            if case .loadingAR = vm.loadState {
                ProgressView().controlSize(.small).tint(.white)
            } else if case .loadingDiffusion = vm.loadState {
                ProgressView().controlSize(.small).tint(.white)
            }
            Button {
                Task { await vm.loadModels() }
            } label: {
                Label(loadButtonTitle, systemImage: loadButtonIcon)
                    .font(.system(size: 13, weight: .semibold))
            }
            .buttonStyle(GlowButtonStyle(color: .cyan))
            .disabled(isLoading || vm.loadState == .ready)
        }
    }

    private var loadButtonTitle: String {
        switch vm.loadState {
        case .idle: return "Charger les 2 modèles"
        case .loadingAR: return "Chargement AR…"
        case .loadingDiffusion: return "Chargement Diffusion…"
        case .ready: return "Modèles prêts"
        case .error: return "Réessayer"
        }
    }

    private var loadButtonIcon: String {
        switch vm.loadState {
        case .idle: return "arrow.down.circle"
        case .loadingAR, .loadingDiffusion: return "hourglass"
        case .ready: return "checkmark.circle.fill"
        case .error: return "exclamationmark.triangle.fill"
        }
    }

    private var isLoading: Bool {
        if case .loadingAR = vm.loadState { return true }
        if case .loadingDiffusion = vm.loadState { return true }
        return false
    }

    private var controlBar: some View {
        VStack(spacing: 10) {
            HStack(spacing: 12) {
                Text("Prompt").font(.caption).foregroundStyle(.white.opacity(0.6))
                TextField("", text: $vm.prompt)
                    .textFieldStyle(.plain)
                    .font(.system(size: 13, design: .monospaced))
                    .foregroundStyle(.white)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 8)
                    .background(
                        RoundedRectangle(cornerRadius: 8)
                            .fill(Color.white.opacity(0.06))
                            .overlay(
                                RoundedRectangle(cornerRadius: 8)
                                    .strokeBorder(Color.white.opacity(0.15), lineWidth: 1)
                            )
                    )
            }

            HStack(spacing: 16) {
                HStack(spacing: 6) {
                    Text("Max tokens").font(.caption).foregroundStyle(.white.opacity(0.6))
                    Stepper(value: $vm.maxTokens, in: 64 ... 1024, step: 64) {
                        Text("\(vm.maxTokens)").font(.system(size: 12, design: .monospaced)).foregroundStyle(.white)
                    }
                    .labelsHidden()
                }
                HStack(spacing: 6) {
                    Text("Temperature").font(.caption).foregroundStyle(.white.opacity(0.6))
                    Slider(value: $vm.temperature, in: 0 ... 1, step: 0.05).frame(width: 120)
                    Text(String(format: "%.2f", vm.temperature)).font(.system(size: 12, design: .monospaced)).foregroundStyle(.white).frame(width: 36)
                }

                Spacer()

                Button {
                    Task { await vm.runRace() }
                } label: {
                    Label("Lancer le bench", systemImage: "play.fill")
                        .font(.system(size: 13, weight: .semibold))
                }
                .buttonStyle(GlowButtonStyle(color: .pink))
                .disabled(vm.loadState != .ready || vm.arPanel.isRunning || vm.diffusionPanel.isRunning)

                Button {
                    Task { await vm.runAR() }
                } label: {
                    Label("AR seul", systemImage: "arrow.right.circle")
                        .font(.system(size: 12))
                }
                .buttonStyle(GlowButtonStyle(color: .green))
                .disabled(vm.loadState != .ready || vm.arPanel.isRunning)

                Button {
                    Task { await vm.runDiffusion() }
                } label: {
                    Label("Diffusion seul", systemImage: "waveform")
                        .font(.system(size: 12))
                }
                .buttonStyle(GlowButtonStyle(color: .purple))
                .disabled(vm.loadState != .ready || vm.diffusionPanel.isRunning)
            }
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 12)
    }

    private var panels: some View {
        HSplitView {
            PanelView(
                title: "Autoregressive — Gemma 4 26B-A4B bf16",
                subtitle: "Streaming token par token (greedy + topP)",
                accent: .green,
                icon: "arrow.right.circle.fill",
                state: vm.arPanel,
                stepInfo: nil
            )
            PanelView(
                title: "Diffusion — DiffusionGemma 26B-A4B bf16",
                subtitle: "Denoising bloc-AR (canvas 256, max 48 steps)",
                accent: .purple,
                icon: "waveform",
                state: vm.diffusionPanel,
                stepInfo: vm.diffusionPanel.currentStep.map { "\($0) / \(vm.diffusionPanel.totalSteps)" }
            )
        }
        .frame(minHeight: 400)
    }

    private var statusBar: some View {
        HStack(spacing: 18) {
            Image(systemName: statusIcon)
                .foregroundStyle(statusColor)
            Text(vm.loadState.label)
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(.white.opacity(0.7))
            Spacer()
            Text("AR : \(String(format: "%.1f", vm.arPanel.tokPerSec)) tok/s   •   Diffusion : \(String(format: "%.1f", vm.diffusionPanel.tokPerSec)) tok/s")
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(.white.opacity(0.7))
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 10)
        .background(Color.black.opacity(0.4))
    }

    private var statusIcon: String {
        switch vm.loadState {
        case .idle: return "circle"
        case .loadingAR, .loadingDiffusion: return "arrow.triangle.2.circlepath"
        case .ready: return "checkmark.circle.fill"
        case .error: return "exclamationmark.triangle.fill"
        }
    }

    private var statusColor: Color {
        switch vm.loadState {
        case .idle: return .gray
        case .loadingAR, .loadingDiffusion: return .cyan
        case .ready: return .green
        case .error: return .red
        }
    }
}

// MARK: - PanelView

struct PanelView: View {
    let title: String
    let subtitle: String
    let accent: Color
    let icon: String
    let state: BenchViewModel.PanelState
    let stepInfo: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header du panneau
            HStack(spacing: 10) {
                Image(systemName: icon)
                    .foregroundStyle(accent)
                    .font(.system(size: 16))
                VStack(alignment: .leading, spacing: 1) {
                    Text(title)
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundStyle(.white)
                    Text(subtitle)
                        .font(.system(size: 10))
                        .foregroundStyle(.white.opacity(0.5))
                }
                Spacer()
                if state.isRunning {
                    ProgressView().controlSize(.small).tint(accent)
                }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .background(Color.black.opacity(0.3))

            // Texte (scrollable)
            ScrollView {
                Text(state.text.isEmpty ? "—" : state.text)
                    .font(.system(size: 13, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.92))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(16)
                    .id(state.text.hashValue)
                    .animation(.easeInOut(duration: 0.25), value: state.text)
            }

            // Footer stats
            HStack(spacing: 14) {
                stat("\(state.tokensGenerated) tok", "text.bubble")
                stat(String(format: "%.2fs", state.elapsed), "clock")
                stat(String(format: "%.1f tok/s", state.tokPerSec), "speedometer")
                if let stepInfo = stepInfo {
                    stat("step \(stepInfo)", "waveform.path")
                }
                Spacer()
                Text(state.status)
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.5))
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .background(Color.black.opacity(0.3))
        }
        .background(
            RoundedRectangle(cornerRadius: 0)
                .fill(Color.white.opacity(0.02))
        )
        .overlay(
            Rectangle()
                .strokeBorder(state.isRunning ? accent.opacity(0.7) : Color.white.opacity(0.1), lineWidth: state.isRunning ? 2 : 1)
        )
        .shadow(color: state.isRunning ? accent.opacity(0.5) : .clear, radius: state.isRunning ? 18 : 0)
    }

    private func stat(_ value: String, _ icon: String) -> some View {
        HStack(spacing: 4) {
            Image(systemName: icon).font(.system(size: 9)).foregroundStyle(accent.opacity(0.8))
            Text(value)
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(.white.opacity(0.85))
        }
    }
}

// MARK: - Style

struct GlowButtonStyle: ButtonStyle {
    let color: Color

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .foregroundStyle(.white)
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .fill(color.opacity(configuration.isPressed ? 0.5 : 0.25))
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .strokeBorder(color.opacity(0.6), lineWidth: 1)
                    )
            )
            .shadow(color: color.opacity(0.35), radius: 6)
            .scaleEffect(configuration.isPressed ? 0.97 : 1.0)
    }
}
