// Akinator-VQA : tour-a-tour entre l'utilisateur et DiffusionGemma.
//
// Mecanique :
//   1. L'utilisateur upload une image (qu'il peut masquer pour s'auto-imposer
//      la regle "je ne triche pas en regardant").
//   2. A chaque tour l'utilisateur pose une question oui/non ou une devinette.
//   3. DiffusionGemma fait un VRAI VQA sur l'image et repond.
//   4. Compteur de questions. Si la reponse contient "Yes" ou "Correct" en
//      reponse a une devinette explicite, on peut conclure win.
//
// Met en valeur le VQA fort de DiffusionGemma (79% ScreenSpot, 80% OCR).

import AppKit
import Foundation
import Gemma4Swift
import MLX
import Tokenizers

@MainActor
final class VQAGameViewModel: ObservableObject {

    // MARK: - Etat du jeu
    enum TurnRole: Sendable { case user, model }

    struct Turn: Identifiable, Sendable {
        let id = UUID()
        let role: TurnRole
        let text: String
        let elapsed: TimeInterval?
        let stepsUsed: Int?
        let questionType: QuestionType?
    }

    /// Type de message utilisateur : question ouverte ou devinette.
    /// Permet d'ajuster le prompt pour que le modele reponde "correct"/"incorrect"
    /// vs une vraie reponse VQA detaillee.
    enum QuestionType: String, Sendable {
        case question // VQA libre
        case guess    // devinette : modele dit Yes / No / Close
    }

    @Published var turns: [Turn] = []
    @Published var isThinking: Bool = false
    @Published var imageURL: URL? = nil
    @Published var imageMasked: Bool = true
    @Published var solved: Bool = false
    @Published var maxQuestions: Int = 20

    var questionsAsked: Int { turns.filter { $0.role == .user }.count }
    func canPlay(registry: ModelRegistry) -> Bool {
        registry.isDiffusionLoaded && imageURL != nil && !solved && questionsAsked < maxQuestions
    }

    private var cachedPixels: MLXArray?

    // MARK: - Image
    func selectImage() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [.image]
        panel.message = "Choisis une image — le modele va la voir et tu vas devoir la deviner."
        if panel.runModal() == .OK, let url = panel.url {
            imageURL = url
            imageMasked = true
            solved = false
            turns = []
            cachedPixels = nil // sera recalcule au premier tour
        }
    }

    func reset() {
        turns = []
        solved = false
    }

    // MARK: - Tour de jeu
    func ask(text: String, type: QuestionType, registry: ModelRegistry) async {
        guard canPlay(registry: registry),
              let url = imageURL,
              let model = registry.diffModel,
              let dconf = registry.diffConfig,
              let dgen = registry.diffGenConfig,
              let dtok = registry.diffTokenizer
        else { return }

        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        // Append user turn immediately
        turns.append(Turn(role: .user, text: trimmed, elapsed: nil, stepsUsed: nil, questionType: type))

        isThinking = true
        defer { isThinking = false }

        // Pre-process image once
        if cachedPixels == nil {
            do {
                cachedPixels = try Gemma4ImageProcessor.processImage(url: url)
            } catch {
                turns.append(Turn(role: .model, text: "⚠ Erreur de preprocessing : \(error.localizedDescription)", elapsed: nil, stepsUsed: nil, questionType: nil))
                return
            }
        }
        guard let pixels = cachedPixels else { return }

        let prompt = makePrompt(question: trimmed, type: type)
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
            let result: (text: String, steps: Int) = await Task.detached {
                let pipeline = DiffusionGemmaPipeline(model: unsafeModel, genConfig: dgen)
                let r = await pipeline.generate(
                    promptIds: unsafeIds,
                    pixelValues: unsafePixels,
                    maxBlocks: 1,
                    seed: 0
                )
                let outIds = r.generatedIds.asArray(Int32.self).map { Int($0) }
                return (unsafeTokenizer.decode(tokens: outIds, skipSpecialTokens: true), r.totalDecoderSteps)
            }.value

            let elapsed = Date().timeIntervalSince(start)
            let cleaned = cleanModelOutput(result.text)
            turns.append(Turn(role: .model, text: cleaned, elapsed: elapsed, stepsUsed: result.steps, questionType: nil))

            // Detection de victoire
            //  - mode guess : reponse contient un marqueur explicite
            //  - mode question : si la question ressemble a une devinette implicite
            //    ("est-ce la Joconde ?", "is it a cat ?", "c'est un X ?") et que la
            //    reponse commence par une confirmation positive
            if detectWin(question: trimmed, answer: cleaned, type: type) {
                solved = true
                imageMasked = false // auto-reveal a la victoire
            }
        } catch {
            turns.append(Turn(role: .model, text: "⚠ Erreur : \(error.localizedDescription)", elapsed: nil, stepsUsed: nil, questionType: nil))
        }
    }

    /// Demande au modele de reveler la solution (fin de partie).
    func revealAnswer(registry: ModelRegistry) async {
        guard registry.isDiffusionLoaded, imageURL != nil else { return }
        await ask(text: "I give up — describe the image in one short sentence.", type: .question, registry: registry)
        solved = true
    }

    // MARK: - Prompt
    private func makePrompt(question: String, type: QuestionType) -> String {
        switch type {
        case .question:
            return """
            You are playing a 20-questions style game. The user cannot see this image and is trying to guess what is in it.
            They will ask short questions about the image. Answer based ONLY on what is actually visible.

            Rules:
            - If the question is yes/no, answer first with "Yes." or "No." (or "Partially.") then one short justifying sentence (10 words max).
            - If the question asks for an attribute (count, color, position, type), give the precise visual answer in one short sentence.
            - Never reveal the full content of the image unless explicitly asked to give up.

            User question: \(question)

            Your answer:
            """
        case .guess:
            return """
            You are playing a 20-questions style game. The user is guessing the content of this image.
            Their guess: "\(question)"

            Look at the image and answer in ONE short line. IMPORTANT: you must always repeat the user's guess literally in your answer (spelling and casing preserved), so the user can verify you confirmed the right thing.

            - If the guess matches what is in the image (the main subject or scene), reply EXACTLY:
              Yes — you got it, it's \(question)!
            - If it is close but not quite right, reply:
              Close, but \(question) is not exactly it — <one-sentence hint pointing in the right direction>
            - If it is wrong, reply:
              No, \(question) is not what is in the image — <one-sentence hint about what category the actual subject is in>

            Your answer:
            """
        }
    }

    // MARK: - Detection de victoire
    /// Strict : la victoire n'est detectee QUE en mode Devinette, et seulement
    /// si la reponse positive du modele cite EXPLICITEMENT au moins un terme
    /// significatif (mot >= 4 caracteres) que l'utilisateur a tape (normalise
    /// pour ignorer casse + accents + ponctuation).
    private func detectWin(question: String, answer: String, type: QuestionType) -> Bool {
        guard type == .guess else { return false }

        let a = answer.lowercased()
        let positive = a.hasPrefix("yes")
            || a.hasPrefix("oui")
            || a.contains("you got it")
            || a.contains("correct")
            || a.contains("bravo")
            || a.contains("c'est bien")
            || a.contains("c'est exactement")
            || a.contains("you guessed it")
        guard positive else { return false }

        // Le terme du user doit apparaitre dans la reponse, normalise.
        let qNorm = normalizeForCitation(question)
        let aNorm = normalizeForCitation(answer)
        let stopWords: Set<String> = [
            "est", "ce", "que", "qu", "une", "un", "le", "la", "les", "des", "du", "de",
            "is", "it", "this", "that", "the", "a", "an", "are", "you", "your", "of",
        ]
        let significantWords = qNorm
            .components(separatedBy: " ")
            .filter { $0.count >= 4 && !stopWords.contains($0) }
        guard !significantWords.isEmpty else { return false }
        return significantWords.contains { aNorm.contains($0) }
    }

    /// Normalise pour citer : minuscules + sans accents + sans ponctuation.
    /// "Joconde !" -> "joconde", "L'Été" -> "ete".
    private func normalizeForCitation(_ s: String) -> String {
        let stripped = s.applyingTransform(.stripDiacritics, reverse: false) ?? s
        return stripped
            .lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { !$0.isEmpty }
            .joined(separator: " ")
    }

    private func cleanModelOutput(_ raw: String) -> String {
        raw
            .replacingOccurrences(of: "<eos>", with: "")
            .replacingOccurrences(of: "<turn|>", with: "")
            .replacingOccurrences(of: "<|turn>", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
