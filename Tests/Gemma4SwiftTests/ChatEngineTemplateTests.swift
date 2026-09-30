import Testing
import Foundation
@testable import Gemma4Swift

/// Porte de K-36 : `Gemma4ChatEngine.promptIds` rend exactement les ids de HF
/// (`apply_chat_template` de transformers, Scripts/quality/chat-engine-hf-fixtures.py) pour
/// texte, pensee, image, outils et tour `tool`. Besoin du tokenizer et du gabarit d'un pack :
///   GEMMA4_INTEGRATION_MODEL_PATH=/Volumes/Lexar/models/mlx-community/gemma-4-e2b-it-4bit \
///     Scripts/run-tests.sh -only-testing:Gemma4SwiftTests/ChatEngineTemplateTests
private let modelPath = ProcessInfo.processInfo.environment["GEMMA4_INTEGRATION_MODEL_PATH"]

@Suite("Moteur de conversation : rendu du gabarit", .enabled(if: modelPath != nil))
struct ChatEngineTemplateTests {

    struct Fixture: Decodable {
        let name: String
        let thinking: Bool
        let images: Int?
        let ids: [Int]
    }

    private func fixtures() throws -> (raw: [[String: Any]], decoded: [Fixture]) {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appendingPathComponent("Fixtures/chat-engine-hf.json")
        let data = try Data(contentsOf: url)
        let raw = try #require(try JSONSerialization.jsonObject(with: data) as? [[String: Any]])
        return (raw, try JSONDecoder().decode([Fixture].self, from: data))
    }

    private func messages(_ raw: [[String: Any]], images: Int) -> [Gemma4ChatMessage] {
        raw.map { m in
            let role = Gemma4ChatMessage.Role(rawValue: m["role"] as! String)!
            var content = m["content"] as? String ?? ""
            var imageData: [Data] = []
            if content.hasPrefix(Gemma4Processor.imageToken + "\n") {
                content.removeFirst(Gemma4Processor.imageToken.count + 1)
                imageData = Array(repeating: Data(), count: images)
            }
            let calls = (m["tool_calls"] as? [[String: Any]] ?? []).map { c -> Gemma4ToolCall in
                let function = c["function"] as! [String: Any]
                let arguments = try! JSONSerialization.data(withJSONObject: function["arguments"]!, options: [.sortedKeys])
                return Gemma4ToolCall(id: c["id"] as! String, name: function["name"] as! String,
                                      argumentsJSON: String(decoding: arguments, as: UTF8.self))
            }
            return Gemma4ChatMessage(role: role, content: content, images: imageData, toolCalls: calls,
                                     toolName: m["name"] as? String, toolCallID: m["tool_call_id"] as? String)
        }
    }

    @Test("ids identiques a HF pour les 5 cas")
    func testMatchesHF() async throws {
        let tokenizer = try await Gemma4TokenizerLoader().load(from: URL(fileURLWithPath: modelPath!))
        let (raw, decoded) = try fixtures()
        for (rawCase, fixture) in zip(raw, decoded) {
            let tools = (rawCase["tools"] as? [[String: Any]] ?? []).map { $0 as! [String: any Sendable] }
            let ids = try Gemma4ChatEngine.promptIds(
                messages: messages(rawCase["messages"] as! [[String: Any]], images: fixture.images ?? 0),
                tools: tools, enableThinking: fixture.thinking, tokenizer: tokenizer)
            if let first = zip(ids, fixture.ids).enumerated().first(where: { $0.element.0 != $0.element.1 })?.offset {
                let window = max(0, first - 3) ..< min(ids.count, first + 12)
                Issue.record("""
                    cas \(fixture.name) : premier ecart a \(first)
                    swift : \(tokenizer.decode(tokenIds: Array(ids[window])))
                    hf    : \(tokenizer.decode(tokenIds: Array(fixture.ids[window.clamped(to: 0 ..< fixture.ids.count)])))
                    """)
            }
            #expect(ids.count == fixture.ids.count, "cas \(fixture.name) : \(ids.count) ids contre \(fixture.ids.count) (HF)")
        }
    }

    @Test("marqueur tape dans le texte : refuse")
    func testMarkerInTextRejected() async throws {
        let tokenizer = try await Gemma4TokenizerLoader().load(from: URL(fileURLWithPath: modelPath!))
        #expect(throws: Gemma4ChatEngineError.markerInText("system")) {
            _ = try Gemma4ChatEngine.promptIds(
                messages: [.init(role: .system, content: "use <|image|> tokens"), .init(role: .user, content: "hi")],
                tools: [], enableThinking: false, tokenizer: tokenizer)
        }
    }
}
