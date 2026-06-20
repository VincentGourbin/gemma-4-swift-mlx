// View "Akinator VQA" : image cachee, chat tour-a-tour avec DiffusionGemma.

import AppKit
import SwiftUI

struct VQAGameView: View {
    @StateObject private var vm = VQAGameViewModel()
    @EnvironmentObject private var registry: ModelRegistry
    @State private var input: String = ""
    @State private var mode: VQAGameViewModel.QuestionType = .question
    @State private var imagePreview: NSImage? = nil
    @State private var sessionId = UUID()

    var body: some View {
        HSplitView {
            // Gauche : image masquee + controles
            leftPanel
                .frame(minWidth: 360, maxWidth: 480)
                .background(Color.black.opacity(0.3))

            // Droite : chat tour-a-tour
            rightPanel
                .frame(minWidth: 460)
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

    // MARK: - Left panel
    private var leftPanel: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("DiffusionGuess")
                    .font(.system(size: 16, weight: .bold, design: .rounded))
                    .foregroundStyle(.white)
                Spacer()
                Text("VQA Akinator")
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(.purple)
                    .padding(.horizontal, 6).padding(.vertical, 2)
                    .background(Capsule().fill(Color.purple.opacity(0.18))
                        .overlay(Capsule().strokeBorder(Color.purple.opacity(0.5), lineWidth: 1)))
            }

            Text("Le modèle voit l'image. Tu poses des questions oui/non ou des devinettes, il répond via VQA réel.")
                .font(.system(size: 10))
                .foregroundStyle(.white.opacity(0.55))
                .fixedSize(horizontal: false, vertical: true)

            ModelStatusChip(kind: .diffusion)
            ModelControlButtons(kind: .diffusion)

            Divider().overlay(Color.white.opacity(0.1))

            // Image
            HStack {
                Text("Image")
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(.cyan.opacity(0.85))
                Spacer()
                if vm.imageURL != nil {
                    Toggle(isOn: $vm.imageMasked) {
                        Text(vm.imageMasked ? "Masquée" : "Visible")
                            .font(.system(size: 10))
                    }
                    .toggleStyle(.switch)
                    .controlSize(.mini)
                }
            }

            ZStack {
                if let url = vm.imageURL, let img = imagePreview {
                    if vm.imageMasked {
                        // Placeholder couvrant
                        ZStack {
                            RoundedRectangle(cornerRadius: 10)
                                .fill(LinearGradient(colors: [.purple.opacity(0.25), .blue.opacity(0.25)],
                                                     startPoint: .topLeading, endPoint: .bottomTrailing))
                            VStack(spacing: 8) {
                                Image(systemName: "eye.slash.fill")
                                    .font(.system(size: 28))
                                    .foregroundStyle(.white.opacity(0.7))
                                Text("Image cachée — seul le modèle la voit")
                                    .font(.system(size: 11, weight: .semibold))
                                    .foregroundStyle(.white.opacity(0.8))
                                Text(url.lastPathComponent)
                                    .font(.system(size: 9, design: .monospaced))
                                    .foregroundStyle(.white.opacity(0.45))
                            }
                        }
                        .frame(maxWidth: .infinity, minHeight: 200, maxHeight: 280)
                    } else {
                        Image(nsImage: img)
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(maxHeight: 280)
                            .cornerRadius(10)
                            .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(Color.purple.opacity(0.5), lineWidth: 1))
                    }
                } else {
                    RoundedRectangle(cornerRadius: 10)
                        .fill(Color.white.opacity(0.04))
                        .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(Color.white.opacity(0.15), style: StrokeStyle(lineWidth: 1, dash: [4])))
                        .overlay(
                            VStack(spacing: 6) {
                                Image(systemName: "photo.badge.plus")
                                    .font(.system(size: 24))
                                    .foregroundStyle(.white.opacity(0.4))
                                Text("Choisis une image")
                                    .font(.system(size: 11))
                                    .foregroundStyle(.white.opacity(0.5))
                            }
                        )
                        .frame(maxWidth: .infinity, minHeight: 200, maxHeight: 280)
                }
            }

            HStack(spacing: 6) {
                Button {
                    vm.selectImage()
                    if let url = vm.imageURL {
                        imagePreview = NSImage(contentsOf: url)
                        sessionId = UUID()
                    }
                } label: {
                    Label(vm.imageURL == nil ? "Choisir une image" : "Changer d'image", systemImage: "photo")
                        .font(.system(size: 11, weight: .semibold))
                }
                .buttonStyle(GlowButtonStyle(color: .purple))

                if vm.imageURL != nil {
                    Button {
                        Task { await vm.revealAnswer(registry: registry) }
                    } label: {
                        Label("Je donne ma langue au chat", systemImage: "flag.fill")
                            .font(.system(size: 10))
                    }
                    .buttonStyle(GlowButtonStyle(color: .orange))
                    .disabled(!registry.isDiffusionLoaded || vm.solved || vm.isThinking)
                }
            }

            // Stats
            HStack(spacing: 14) {
                stat(label: "Tour", value: "\(vm.questionsAsked) / \(vm.maxQuestions)", color: .cyan)
                if vm.solved {
                    HStack(spacing: 4) {
                        Image(systemName: "trophy.fill").foregroundStyle(.yellow)
                        Text("Partie terminée").font(.system(size: 11, weight: .semibold))
                            .foregroundStyle(.yellow)
                    }
                }
                Spacer()
            }

            Spacer()
        }
        .padding(16)
    }

    // MARK: - Right panel (chat)
    private var rightPanel: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Image(systemName: "bubble.left.and.bubble.right.fill")
                    .foregroundStyle(.purple)
                Text("Tour à tour")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(.white)
                Spacer()
                if vm.isThinking {
                    HStack(spacing: 4) {
                        ProgressView().controlSize(.small).tint(.purple)
                        Text("Le modèle réfléchit…")
                            .font(.system(size: 10))
                            .foregroundStyle(.white.opacity(0.65))
                    }
                }
            }
            .padding(12)
            .background(Color.black.opacity(0.3))

            Divider().overlay(Color.white.opacity(0.1))

            // Chat
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 10) {
                        if vm.turns.isEmpty {
                            emptyChatHint
                        }
                        ForEach(vm.turns) { turn in
                            TurnBubble(turn: turn)
                                .id(turn.id)
                        }
                    }
                    .padding(14)
                }
                .onChange(of: vm.turns.count) { _, _ in
                    if let last = vm.turns.last?.id {
                        withAnimation(.easeOut(duration: 0.2)) {
                            proxy.scrollTo(last, anchor: .bottom)
                        }
                    }
                }
            }

            // Input
            Divider().overlay(Color.white.opacity(0.1))
            inputBar
                .padding(12)
                .background(Color.black.opacity(0.3))
        }
    }

    private var emptyChatHint: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Tour 1 — commence par une question large")
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(.white.opacity(0.7))
            Text("Exemples :")
                .font(.system(size: 10))
                .foregroundStyle(.white.opacity(0.5))
            ForEach([
                "Y a-t-il un être vivant ?",
                "Est-ce qu'on voit une personne ?",
                "C'est une photo en intérieur ou en extérieur ?",
                "Quelle est la couleur dominante ?",
            ], id: \.self) { ex in
                Button {
                    input = ex
                    mode = ex.lowercased().hasPrefix("c'est") ? .question : .question
                } label: {
                    Text("• \(ex)")
                        .font(.system(size: 11))
                        .foregroundStyle(.cyan.opacity(0.8))
                }
                .buttonStyle(.plain)
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: 8).fill(Color.white.opacity(0.04)))
    }

    private var inputBar: some View {
        VStack(spacing: 8) {
            HStack(spacing: 4) {
                Picker("", selection: $mode) {
                    Text("Question").tag(VQAGameViewModel.QuestionType.question)
                    Text("Devinette").tag(VQAGameViewModel.QuestionType.guess)
                }
                .pickerStyle(.segmented)
                .frame(maxWidth: 220)
                Spacer()
                if vm.questionsAsked >= vm.maxQuestions && !vm.solved {
                    Text("Plafond atteint — abandonne ou recommence").font(.system(size: 10)).foregroundStyle(.orange)
                }
            }

            // Hint : si tu poses une devinette en mode Question, on suggere
            // de basculer pour pouvoir gagner.
            if mode == .question, looksLikeGuess(input) {
                HStack(spacing: 6) {
                    Image(systemName: "lightbulb.fill")
                        .font(.system(size: 10)).foregroundStyle(.yellow)
                    Text("Ça ressemble à une devinette — bascule en mode \"Devinette\" pour pouvoir gagner.")
                        .font(.system(size: 10)).foregroundStyle(.yellow.opacity(0.85))
                    Spacer()
                    Button("Basculer") { mode = .guess }
                        .font(.system(size: 10, weight: .semibold))
                        .buttonStyle(.borderless)
                        .foregroundStyle(.yellow)
                }
                .padding(.horizontal, 8).padding(.vertical, 4)
                .background(
                    RoundedRectangle(cornerRadius: 6).fill(Color.yellow.opacity(0.1))
                        .overlay(RoundedRectangle(cornerRadius: 6).strokeBorder(Color.yellow.opacity(0.4), lineWidth: 1))
                )
            }

            HStack(spacing: 8) {
                TextField(mode == .question ? "Pose une question oui/non ou ouverte…" : "Ta devinette (ex. \"un chat\")",
                          text: $input)
                    .textFieldStyle(.plain)
                    .font(.system(size: 12, design: .monospaced))
                    .foregroundStyle(.white)
                    .padding(.horizontal, 10).padding(.vertical, 8)
                    .background(
                        RoundedRectangle(cornerRadius: 8)
                            .fill(Color.white.opacity(0.06))
                            .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(Color.purple.opacity(0.4), lineWidth: 1))
                    )
                    .onSubmit { send() }
                Button { send() } label: {
                    Label(mode == .guess ? "Deviner" : "Demander",
                          systemImage: mode == .guess ? "questionmark.diamond.fill" : "paperplane.fill")
                        .font(.system(size: 12, weight: .semibold))
                }
                .buttonStyle(GlowButtonStyle(color: mode == .guess ? .orange : .purple))
                .disabled(!vm.canPlay(registry: registry) || vm.isThinking || input.trimmingCharacters(in: .whitespaces).isEmpty)
            }

            if !registry.isDiffusionLoaded {
                Text("Charge DiffusionGemma à gauche avant de jouer.").font(.system(size: 10)).foregroundStyle(.orange)
            } else if vm.imageURL == nil {
                Text("Choisis une image à gauche.").font(.system(size: 10)).foregroundStyle(.orange)
            }
        }
    }

    private func send() {
        let text = input
        input = ""
        Task { await vm.ask(text: text, type: mode, registry: registry) }
    }

    /// Detection grossiere : la question commence par "est-ce", "c'est", "is it", etc.
    private func looksLikeGuess(_ text: String) -> Bool {
        let q = text.lowercased()
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .trimmingCharacters(in: .punctuationCharacters)
        guard !q.isEmpty else { return false }
        let prefixes = [
            "est-ce ", "est ce ", "c'est ", "c est ",
            "is it ", "is this ", "is that ", "it is ", "it's ", "this is ", "that is ",
        ]
        return prefixes.contains { q.hasPrefix($0) }
    }

    // MARK: - Helpers
    private func stat(label: String, value: String, color: Color) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(label).font(.system(size: 9)).foregroundStyle(color.opacity(0.8))
            Text(value).font(.system(size: 12, weight: .semibold, design: .monospaced)).foregroundStyle(.white)
        }
        .padding(.horizontal, 8).padding(.vertical, 4)
        .background(RoundedRectangle(cornerRadius: 6).fill(color.opacity(0.1))
            .overlay(RoundedRectangle(cornerRadius: 6).strokeBorder(color.opacity(0.3), lineWidth: 1)))
    }
}

// MARK: - Bulle de chat
struct TurnBubble: View {
    let turn: VQAGameViewModel.Turn

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            if turn.role == .user {
                Spacer(minLength: 40)
                bubble(align: .trailing)
                avatar(letter: "U", color: .cyan)
            } else {
                avatar(letter: "D", color: .purple)
                bubble(align: .leading)
                Spacer(minLength: 40)
            }
        }
    }

    @ViewBuilder
    private func bubble(align: TextAlignment) -> some View {
        VStack(alignment: align == .trailing ? .trailing : .leading, spacing: 4) {
            if turn.role == .user, let t = turn.questionType {
                Text(t == .guess ? "Devinette" : "Question")
                    .font(.system(size: 8, weight: .semibold))
                    .foregroundStyle(.cyan.opacity(0.7))
                    .padding(.horizontal, 5).padding(.vertical, 1)
                    .background(Capsule().fill(Color.cyan.opacity(0.15))
                        .overlay(Capsule().strokeBorder(Color.cyan.opacity(0.4), lineWidth: 0.5)))
            }
            Text(turn.text)
                .font(.system(size: 12))
                .foregroundStyle(.white)
                .padding(10)
                .background(
                    RoundedRectangle(cornerRadius: 10)
                        .fill(turn.role == .user
                              ? Color.cyan.opacity(0.18)
                              : Color.purple.opacity(0.18))
                        .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(
                            (turn.role == .user ? Color.cyan : Color.purple).opacity(0.45), lineWidth: 1))
                )
                .textSelection(.enabled)
                .frame(maxWidth: 480, alignment: align == .trailing ? .trailing : .leading)
            if turn.role == .model, let e = turn.elapsed {
                HStack(spacing: 6) {
                    Image(systemName: "clock").font(.system(size: 8))
                    Text(String(format: "%.1fs", e)).font(.system(size: 8, design: .monospaced))
                    if let s = turn.stepsUsed {
                        Text("• \(s) forwards").font(.system(size: 8, design: .monospaced))
                    }
                }
                .foregroundStyle(.white.opacity(0.45))
            }
        }
    }

    private func avatar(letter: String, color: Color) -> some View {
        Text(letter)
            .font(.system(size: 11, weight: .bold, design: .rounded))
            .foregroundStyle(.white)
            .frame(width: 24, height: 24)
            .background(
                Circle().fill(color.opacity(0.4))
                    .overlay(Circle().strokeBorder(color, lineWidth: 1))
            )
    }
}
