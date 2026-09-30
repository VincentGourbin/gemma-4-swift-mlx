import Testing
import Foundation
import MLX
@testable import Gemma4Swift

/// Porte K-19 (B10) : sur 4 tours d'une boucle a long prompt systeme, la reutilisation du
/// prefixe sert > 80 % du prompt des tours 2+ depuis le cache et divise le TTFT par >= 2.
/// La parite exacte des caches est prouvee en fp32 (ConversationSnapshotParityTests). Vrai modele :
///   GEMMA4_INTEGRATION_MODEL_PATH=… Scripts/run-tests.sh -only-testing:Gemma4SwiftTests/ChatEngineConversationTests
private let modelPath = ProcessInfo.processInfo.environment["GEMMA4_INTEGRATION_MODEL_PATH"]

@Suite("Moteur : reutilisation de conversation", .serialized, .enabled(if: modelPath != nil))
struct ChatEngineConversationTests {

    struct Turn { let text: String; let usage: Gemma4ChatUsage }

    private func conversation(_ engine: Gemma4ChatEngine) async throws -> [Turn] {
        let questions = [
            "Give me three facts about the Moon, one line each.",
            "Now three facts about Mars, same format.",
            "Which of the two is farther from the Sun?",
            "Summarize everything in one sentence.",
        ]
        // Boucle d'agent realiste : long prompt systeme (~3 000 jetons, comme des
        // definitions d'outils), que chaque tour re-envoie.
        let manual = (1 ... 60).map { i in
            "Rule \(i): when the user asks about topic \(i), answer precisely, cite the relevant section, "
                + "keep units consistent, and never invent numbers you cannot justify from the context."
        }.joined(separator: "\n")
        var messages: [Gemma4ChatMessage] = [.init(role: .system, content: "You are concise.\n" + manual)]
        var turns: [Turn] = []
        for question in questions {
            messages.append(.init(role: .user, content: question))
            var text = ""
            var usage: Gemma4ChatUsage?
            for try await event in await engine.respond(
                to: messages, options: .init(maxTokens: 96, temperature: 0)) {
                if case .text(let t) = event { text += t }
                if case .done(let u) = event { usage = u }
            }
            turns.append(Turn(text: text, usage: try #require(usage)))
            messages.append(.init(role: .assistant, content: text))
        }
        return turns
    }

    @Test("4 tours : > 80 % du prompt servi par le cache, TTFT -50 %")
    func testReuse() async throws {
        let url = URL(fileURLWithPath: modelPath!)
        let engine = try await Gemma4ChatEngine.load(from: url)
        await engine.setReusesConversation(false)
        _ = try await conversation(engine)  // echauffement
        let full = try await conversation(engine)
        await engine.setReusesConversation(true)
        let reused = try await conversation(engine)

        for (i, (a, b)) in zip(full, reused).enumerated() {
            // Parite exacte : ConversationSnapshotParityTests (fp32). Ici, en bf16, le re-prefill
            // complet (tranches de 512) et le suffixe seul arrondissent differemment : une
            // bascule d'argmax est possible sur un long contexte (constatee au tour 3 apres 50
            // caracteres, meme sens). On la rapporte sans echouer.
            if a.text != b.text {
                let common = zip(a.text, b.text).prefix { $0 == $1 }.count
                print("DIAG K-19 tour \(i + 1) : bascule apres \(common) car.")
            }
            if i > 0 {
                let ratio = Double(b.usage.cachedPromptTokens) / Double(b.usage.promptTokens)
                #expect(ratio > 0.8, "tour \(i + 1) : \(b.usage.cachedPromptTokens)/\(b.usage.promptTokens) servis du cache")
            }
        }
        let ttft = { (turns: [Turn]) in turns.dropFirst().compactMap(\.usage.timeToFirstToken).map { Int($0 * 1000) } }
        for (f, r) in zip(ttft(full), ttft(reused)) {
            #expect(Double(r) <= 0.5 * Double(f), "TTFT \(r) ms contre \(f) ms en re-prefill complet")
        }
        print("DIAG K-19 prompt/cache : \(reused.map { "\($0.usage.cachedPromptTokens)/\($0.usage.promptTokens)" }) ; "
            + "TTFT tours 2-4 (ms) complet \(ttft(full)) reutilise \(ttft(reused))")
    }
}
