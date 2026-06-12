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
                Text("26B params bf16, un modèle à la fois (l'autre est déchargé automatiquement)")
                    .font(.system(size: 11))
                    .foregroundStyle(.white.opacity(0.5))
            }
            Spacer()
            if vm.loadState.isBusy {
                ProgressView().controlSize(.small).tint(.white)
            }
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 14)
    }

    private var controlBar: some View {
        VStack(spacing: 12) {
            // Ligne 1 : Prompt
            HStack(spacing: 12) {
                Text("Prompt").font(.caption).foregroundStyle(.white.opacity(0.6)).frame(width: 60, alignment: .leading)
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

            // Ligne 2 : Presets (picker chips)
            HStack(spacing: 12) {
                Text("Preset").font(.caption).foregroundStyle(.white.opacity(0.6)).frame(width: 60, alignment: .leading)
                HStack(spacing: 6) {
                    ForEach(BenchViewModel.Preset.allCases) { preset in
                        presetChip(preset)
                    }
                }
                Spacer()
            }

            // Ligne 3 : Parametres detailles
            HStack(spacing: 14) {
                Text("Params").font(.caption).foregroundStyle(.white.opacity(0.6)).frame(width: 60, alignment: .leading)

                // Max tokens
                paramBox(label: "Tokens", color: .white) {
                    Stepper(value: $vm.maxTokens, in: 64 ... 1024, step: 64) {
                        Text("\(vm.maxTokens)").font(.system(size: 11, design: .monospaced)).foregroundStyle(.white)
                    }
                    .labelsHidden().controlSize(.mini)
                }

                // Seed Diffusion
                paramBox(label: "Seed", color: .purple) {
                    HStack(spacing: 3) {
                        Text("\(vm.diffSeed)").font(.system(size: 11, design: .monospaced)).foregroundStyle(.white).frame(minWidth: 30, alignment: .leading)
                        Stepper("", value: $vm.diffSeed, in: 0 ... 999_999, step: 1).labelsHidden().controlSize(.mini)
                    }
                }
                .onChange(of: vm.diffSeed) { _, _ in vm.preset = .custom }

                // Temperature AR
                paramBox(label: "T (AR)", color: .green) {
                    HStack(spacing: 4) {
                        Slider(value: $vm.arTemperature, in: 0 ... 2, step: 0.05).frame(width: 80)
                        Text(String(format: "%.2f", vm.arTemperature)).font(.system(size: 11, design: .monospaced)).foregroundStyle(.white).frame(width: 30)
                    }
                }
                .onChange(of: vm.arTemperature) { _, _ in vm.preset = .custom }

                // Temperature schedule Diffusion (tMin -> tMax)
                paramBox(label: "T (Diff)", color: .purple) {
                    HStack(spacing: 4) {
                        Slider(value: $vm.diffTMin, in: 0.1 ... 2, step: 0.05).frame(width: 60)
                        Text(String(format: "%.1f", vm.diffTMin)).font(.system(size: 10, design: .monospaced)).foregroundStyle(.white).frame(width: 22)
                        Text("→").font(.system(size: 10)).foregroundStyle(.white.opacity(0.5))
                        Slider(value: $vm.diffTMax, in: 0.1 ... 2, step: 0.05).frame(width: 60)
                        Text(String(format: "%.1f", vm.diffTMax)).font(.system(size: 10, design: .monospaced)).foregroundStyle(.white).frame(width: 22)
                    }
                }
                .onChange(of: vm.diffTMin) { _, _ in vm.preset = .custom }
                .onChange(of: vm.diffTMax) { _, _ in vm.preset = .custom }

                // Max steps Diffusion
                paramBox(label: "Steps", color: .purple) {
                    Stepper(value: $vm.diffMaxSteps, in: 8 ... 128, step: 4) {
                        Text("\(vm.diffMaxSteps)").font(.system(size: 11, design: .monospaced)).foregroundStyle(.white)
                    }
                    .labelsHidden().controlSize(.mini)
                }
                .onChange(of: vm.diffMaxSteps) { _, _ in vm.preset = .custom }

                Spacer()

                Button {
                    Task { await vm.runARFull() }
                } label: {
                    Label("Lancer AR", systemImage: "arrow.right.circle.fill")
                        .font(.system(size: 13, weight: .semibold))
                }
                .buttonStyle(GlowButtonStyle(color: .green))
                .disabled(vm.isPipelineActive)

                Button {
                    Task { await vm.runDiffusionFull() }
                } label: {
                    Label("Lancer Diffusion", systemImage: "waveform")
                        .font(.system(size: 13, weight: .semibold))
                }
                .buttonStyle(GlowButtonStyle(color: .purple))
                .disabled(vm.isPipelineActive)
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
        case .loadingAR, .loadingDiffusion, .unloading: return "arrow.triangle.2.circlepath"
        case .arReady, .diffusionReady: return "checkmark.circle.fill"
        case .error: return "exclamationmark.triangle.fill"
        }
    }

    private var statusColor: Color {
        switch vm.loadState {
        case .idle: return .gray
        case .loadingAR, .loadingDiffusion, .unloading: return .cyan
        case .arReady: return .green
        case .diffusionReady: return .purple
        case .error: return .red
        }
    }

    // MARK: - Preset chip

    private func presetChip(_ preset: BenchViewModel.Preset) -> some View {
        let isSelected = vm.preset == preset
        let color = presetColor(preset)
        return Button {
            vm.preset = preset
        } label: {
            HStack(spacing: 4) {
                Image(systemName: preset.icon).font(.system(size: 10))
                Text(preset.rawValue).font(.system(size: 11, weight: isSelected ? .semibold : .regular))
            }
            .foregroundStyle(isSelected ? Color.white : color.opacity(0.9))
            .padding(.horizontal, 10)
            .padding(.vertical, 5)
            .background(
                Capsule()
                    .fill(isSelected ? color.opacity(0.5) : color.opacity(0.12))
                    .overlay(
                        Capsule().strokeBorder(color.opacity(isSelected ? 0.9 : 0.4), lineWidth: 1)
                    )
            )
            .shadow(color: isSelected ? color.opacity(0.6) : .clear, radius: 4)
        }
        .buttonStyle(.plain)
        .help(preset.description)
        .disabled(vm.isPipelineActive)
    }

    private func presetColor(_ preset: BenchViewModel.Preset) -> Color {
        switch preset {
        case .deterministic: return .cyan
        case .balanced: return .green
        case .creative: return .orange
        case .chaotic: return .pink
        case .custom: return .white
        }
    }

    // MARK: - Param box

    private func paramBox<Content: View>(label: String, color: Color, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label)
                .font(.system(size: 9, weight: .semibold))
                .foregroundStyle(color.opacity(0.8))
            content()
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(
            RoundedRectangle(cornerRadius: 6)
                .fill(color.opacity(0.08))
                .overlay(
                    RoundedRectangle(cornerRadius: 6).strokeBorder(color.opacity(0.2), lineWidth: 1)
                )
        )
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
                Text(state.phase.labelShort)
                    .font(.system(size: 10, weight: .semibold, design: .rounded))
                    .foregroundStyle(accent)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(
                        Capsule()
                            .fill(accent.opacity(0.18))
                            .overlay(Capsule().strokeBorder(accent.opacity(0.5), lineWidth: 1))
                    )
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .background(Color.black.opacity(0.3))

            // Progress bar : indéterminée si loading, déterminée si generating
            progressBar
                .frame(height: 24)
                .padding(.horizontal, 14)
                .padding(.vertical, 8)
                .background(Color.black.opacity(0.15))

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

    @ViewBuilder
    private var progressBar: some View {
        switch state.phase {
        case .idle:
            HStack(spacing: 6) {
                Image(systemName: "circle").font(.system(size: 10)).foregroundStyle(.white.opacity(0.3))
                Text("Pret a generer").font(.system(size: 10, design: .monospaced)).foregroundStyle(.white.opacity(0.5))
                Spacer()
            }
        case .loading(let detail):
            HStack(spacing: 8) {
                ProgressView().controlSize(.small).tint(accent)
                Text(detail).font(.system(size: 10, design: .monospaced)).foregroundStyle(.white.opacity(0.8))
                Spacer()
            }
        case .generating(let progress, let detail):
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 6) {
                    Text(detail).font(.system(size: 10, design: .monospaced)).foregroundStyle(.white.opacity(0.8))
                    Spacer()
                    Text("\(Int(progress * 100))%")
                        .font(.system(size: 10, weight: .semibold, design: .monospaced))
                        .foregroundStyle(accent)
                }
                GeometryReader { geo in
                    ZStack(alignment: .leading) {
                        RoundedRectangle(cornerRadius: 3)
                            .fill(Color.white.opacity(0.08))
                        RoundedRectangle(cornerRadius: 3)
                            .fill(LinearGradient(
                                colors: [accent.opacity(0.6), accent],
                                startPoint: .leading,
                                endPoint: .trailing
                            ))
                            .frame(width: max(2, geo.size.width * progress))
                            .shadow(color: accent.opacity(0.6), radius: 4)
                            .animation(.easeOut(duration: 0.2), value: progress)
                    }
                }
                .frame(height: 6)
            }
        case .done:
            HStack(spacing: 6) {
                Image(systemName: "checkmark.circle.fill").font(.system(size: 11)).foregroundStyle(accent)
                Text("Termine").font(.system(size: 10, design: .monospaced)).foregroundStyle(.white.opacity(0.7))
                Spacer()
            }
        case .error(let msg):
            HStack(spacing: 6) {
                Image(systemName: "exclamationmark.triangle.fill").font(.system(size: 11)).foregroundStyle(.red)
                Text(msg).font(.system(size: 10, design: .monospaced)).foregroundStyle(.red.opacity(0.8)).lineLimit(1)
                Spacer()
            }
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
