import Foundation
import Gemma4Swift
import Hummingbird
import HummingbirdTesting
import Testing
@testable import Gemma4Server

/// Faux moteur qui rejoue une suite d'evenements (pensee, texte, appel d'outil) et garde
/// ce qu'il a recu.
actor ScriptedBackend: Gemma4ChatBackend {
    let script: [Gemma4ChatEvent]
    private(set) var lastMessages: [Gemma4ChatMessage] = []
    private(set) var lastTools: [[String: any Sendable]] = []
    private(set) var lastOptions: Gemma4ChatOptions?

    init(_ script: [Gemma4ChatEvent]) { self.script = script }

    var maxTokensCap: Int? { nil }
    var acceptsImages: Bool { true }

    func start(
        messages: [Gemma4ChatMessage], tools: [[String: any Sendable]], options: Gemma4ChatOptions
    ) -> Gemma4ChatRun {
        lastMessages = messages
        lastTools = tools
        lastOptions = options
        let (events, continuation) = AsyncThrowingStream<Gemma4ChatEvent, Error>.makeStream()
        let script = script
        let task = Task {
            for event in script { continuation.yield(event) }
            continuation.finish()
        }
        return Gemma4ChatRun(events: events, task: task)
    }
}

@Suite("Serveur : API Messages d'Anthropic (K-42)")
struct AnthropicMessagesTests {

    static let text: [Gemma4ChatEvent] = [
        .text("Bon"), .text("jour"),
        .done(Gemma4ChatUsage(promptTokens: 12, cachedPromptTokens: 5, completionTokens: 2, finishReason: .stop)),
    ]
    static let toolUse: [Gemma4ChatEvent] = [
        .reasoning("il faut la meteo"),
        .toolCall(Gemma4ToolCall(id: "call_1", name: "get_weather", argumentsJSON: #"{"city":"Paris"}"#)),
        .done(Gemma4ChatUsage(promptTokens: 30, completionTokens: 9, finishReason: .toolCalls)),
    ]

    private func app(_ backend: some Gemma4ChatBackend, _ config: Gemma4ServerConfiguration = .init()) -> some ApplicationProtocol {
        Application(router: Gemma4Server(backend: backend, configuration: config).makeRouter())
    }

    private func json(_ response: TestResponse) throws -> [String: Any] {
        try #require(try JSONSerialization.jsonObject(with: Data(buffer: response.body)) as? [String: Any])
    }

    @Test("reponse : bloc texte, end_turn, usage hors cache, modele renvoye")
    func testText() async throws {
        let body = #"{"model":"claude-x","max_tokens":64,"messages":[{"role":"user","content":"Salut"}]}"#
        let backend = ScriptedBackend(Self.text)
        try await app(backend).test(.router) { client in
            try await client.execute(uri: "/v1/messages", method: .post, body: ByteBuffer(string: body)) { response in
                #expect(response.status == .ok)
                let object = try json(response)
                #expect(object["type"] as? String == "message")
                #expect(object["model"] as? String == "claude-x")
                #expect(object["stop_reason"] as? String == "end_turn")
                let content = try #require(object["content"] as? [[String: Any]])
                #expect(content.count == 1 && content[0]["type"] as? String == "text" && content[0]["text"] as? String == "Bonjour")
                let usage = try #require(object["usage"] as? [String: Any])
                #expect(usage["input_tokens"] as? Int == 7 && usage["cache_read_input_tokens"] as? Int == 5)
                #expect(usage["output_tokens"] as? Int == 2)
            }
        }
        #expect(await backend.lastOptions?.maxTokens == 64)
        #expect(await backend.lastOptions?.enableThinking == false)
    }

    @Test("conversion : system en blocs, tool_use / tool_result, outils au format function, pensee")
    func testConversion() async throws {
        let body = #"""
        {"max_tokens":32,"thinking":{"type":"enabled","budget_tokens":1024},
         "system":[{"type":"text","text":"Tu es utile."}],
         "tools":[{"name":"get_weather","description":"meteo","input_schema":{"type":"object","properties":{"city":{"type":"string"}}}}],
         "messages":[
          {"role":"user","content":"Meteo a Paris ?"},
          {"role":"system","content":[{"type":"text","text":"Contexte de session."}]},
          {"role":"assistant","content":[{"type":"thinking","thinking":"...","signature":"s"},
                                         {"type":"text","text":"Je regarde."},
                                         {"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"city":"Paris"}}]},
          {"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":[{"type":"text","text":"18 C"}]},
                                    {"type":"text","text":"Et demain ?"}]}
         ]}
        """#
        let backend = ScriptedBackend(Self.text)
        try await app(backend).test(.router) { client in
            // Comme Claude Code : `?beta=true` et pensee `adaptive`.
            let adaptive = body.replacingOccurrences(of: #""type":"enabled","budget_tokens":1024"#, with: #""type":"adaptive""#)
            try await client.execute(uri: "/v1/messages?beta=true", method: .post, body: ByteBuffer(string: adaptive)) {
                #expect($0.status == .ok)
            }
        }
        let messages = await backend.lastMessages
        #expect(messages.map(\.role) == [.system, .user, .system, .assistant, .tool, .user])
        #expect(messages[0].content == "Tu es utile.")
        #expect(messages[2].content == "Contexte de session.")
        #expect(messages[3].content == "Je regarde.")
        #expect(messages[3].toolCalls.first?.name == "get_weather")
        #expect(messages[3].toolCalls.first?.argumentsJSON == #"{"city":"Paris"}"#)
        #expect(messages[4].content == "18 C" && messages[4].toolName == "get_weather" && messages[4].toolCallID == "toolu_1")
        #expect(messages[5].content == "Et demain ?")
        #expect(await backend.lastOptions?.enableThinking == true)
        let tool = try #require(await backend.lastTools.first)
        #expect(tool["type"] as? String == "function")
        #expect((tool["function"] as? [String: any Sendable])?["name"] as? String == "get_weather")
        #expect((tool["function"] as? [String: any Sendable])?["parameters"] != nil)
    }

    @Test("reponse : blocs thinking puis tool_use (input objet), stop_reason tool_use")
    func testToolUse() async throws {
        let body = #"{"max_tokens":64,"messages":[{"role":"user","content":"Meteo ?"}]}"#
        try await app(ScriptedBackend(Self.toolUse)).test(.router) { client in
            try await client.execute(uri: "/v1/messages", method: .post, body: ByteBuffer(string: body)) { response in
                let object = try json(response)
                #expect(object["stop_reason"] as? String == "tool_use")
                let content = try #require(object["content"] as? [[String: Any]])
                #expect(content.map { $0["type"] as? String } == ["thinking", "tool_use"])
                #expect(content[0]["thinking"] as? String == "il faut la meteo")
                #expect(content[1]["id"] as? String == "call_1" && content[1]["name"] as? String == "get_weather")
                #expect((content[1]["input"] as? [String: Any])?["city"] as? String == "Paris")
            }
        }
    }

    @Test("SSE : sequence d'evenements complete, blocs indexes, input_json_delta")
    func testStream() async throws {
        let body = #"{"max_tokens":64,"stream":true,"messages":[{"role":"user","content":"Meteo ?"}]}"#
        let script: [Gemma4ChatEvent] = [.reasoning("hm")] + Self.text.dropLast() + Self.toolUse.dropFirst()
        try await app(ScriptedBackend(script)).test(.router) { client in
            try await client.execute(uri: "/v1/messages", method: .post, body: ByteBuffer(string: body)) { response in
                #expect(response.headers[.contentType]?.hasPrefix("text/event-stream") == true)
                let text = String(buffer: response.body)
                let events = text.split(separator: "\n").filter { $0.hasPrefix("event: ") }.map { String($0.dropFirst(7)) }
                #expect(events == [
                    "message_start",
                    "content_block_start", "content_block_delta", "content_block_stop",          // thinking
                    "content_block_start", "content_block_delta", "content_block_delta", "content_block_stop", // texte
                    "content_block_start", "content_block_delta", "content_block_stop",          // tool_use
                    "message_delta", "message_stop",
                ])
                #expect(text.contains(#""type":"thinking_delta""#))
                #expect(text.contains(#""text":"Bon""#) && text.contains(#""type":"text_delta""#))
                #expect(text.contains(#""partial_json":"{\"city\":\"Paris\"}""#))
                #expect(text.contains(#""stop_reason":"tool_use""#))
                #expect(text.contains(#""index":2"#))
            }
        }
    }

    @Test("budget de pensee : budget_tokens borne par le plafond du serveur ; adaptive prend le plafond")
    func testThinkingBudget() async throws {
        var config = Gemma4ServerConfiguration()
        config.maxThinkingTokens = 512
        let cases: [(String, Int?)] = [
            (#"{"type":"enabled","budget_tokens":1024}"#, 512),
            (#"{"type":"enabled","budget_tokens":128}"#, 128),
            (#"{"type":"adaptive"}"#, 512),
        ]
        for (thinking, expected) in cases {
            let backend = ScriptedBackend(Self.text)
            let body = #"{"max_tokens":64,"thinking":"# + thinking + #","messages":[{"role":"user","content":"x"}]}"#
            try await app(backend, config).test(.router) { client in
                try await client.execute(uri: "/v1/messages", method: .post, body: ByteBuffer(string: body)) {
                    #expect($0.status == .ok)
                }
            }
            #expect(await backend.lastOptions?.maxThinkingTokens == expected, "\(thinking)")
        }
        // Sans plafond serveur ni budget : aucun.
        let backend = ScriptedBackend(Self.text)
        try await app(backend).test(.router) { client in
            try await client.execute(uri: "/v1/messages", method: .post,
                                     body: ByteBuffer(string: #"{"max_tokens":8,"thinking":{"type":"adaptive"},"messages":[{"role":"user","content":"x"}]}"#)) {
                #expect($0.status == .ok)
            }
        }
        #expect(await backend.lastOptions?.maxThinkingTokens == nil)
    }

    @Test("cle d'API : x-api-key accepte, erreurs au format Anthropic")
    func testAuthAndErrors() async throws {
        let config = Gemma4ServerConfiguration(apiKey: "secret")
        let body = #"{"max_tokens":8,"messages":[{"role":"user","content":"x"}]}"#
        try await app(ScriptedBackend(Self.text), config).test(.router) { client in
            try await client.execute(uri: "/v1/messages", method: .post, body: ByteBuffer(string: body)) { response in
                #expect(response.status == .unauthorized)
                let object = try json(response)
                #expect(object["type"] as? String == "error")
                #expect((object["error"] as? [String: Any])?["type"] as? String == "authentication_error")
            }
            try await client.execute(uri: "/v1/messages", method: .post, headers: [.init("x-api-key")!: "secret"],
                                     body: ByteBuffer(string: body)) { #expect($0.status == .ok) }
            try await client.execute(uri: "/v1/messages", method: .post, headers: [.authorization: "Bearer secret"],
                                     body: ByteBuffer(string: body)) { #expect($0.status == .ok) }
        }
    }

    @Test("400 : stop_sequences, image par URL, document, role inconnu")
    func testRejected() async throws {
        let bodies = [
            #"{"max_tokens":8,"stop_sequences":["\n"],"messages":[{"role":"user","content":"x"}]}"#,
            #"{"max_tokens":8,"messages":[{"role":"user","content":[{"type":"image","source":{"type":"url","url":"https://example.com/a.png"}}]}]}"#,
            #"{"max_tokens":8,"messages":[{"role":"user","content":[{"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"AA=="}}]}]}"#,
            #"{"max_tokens":8,"messages":[{"role":"tool","content":"x"}]}"#,
        ]
        try await app(ScriptedBackend(Self.text)).test(.router) { client in
            for body in bodies {
                try await client.execute(uri: "/v1/messages", method: .post, body: ByteBuffer(string: body)) { response in
                    #expect(response.status == .badRequest, "\(body)")
                    #expect(((try? json(response))?["error"] as? [String: Any])?["type"] as? String == "invalid_request_error")
                }
            }
        }
    }
}
