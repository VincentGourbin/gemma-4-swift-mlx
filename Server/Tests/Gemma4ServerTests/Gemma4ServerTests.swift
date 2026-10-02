import Foundation
import Gemma4Swift
import Hummingbird
import HummingbirdTesting
import Testing
@testable import Gemma4Server

/// Faux moteur : deux morceaux de texte espaces de `delay`, respecte l'annulation, et
/// compte les generations simultanees (K-39 : jamais plus d'une).
actor FakeBackend: Gemma4ChatBackend {
    let delay: Duration
    private(set) var active = 0
    private(set) var maxActive = 0
    private(set) var finished = 0
    private(set) var cancelledRuns = 0
    private(set) var lastOptions: Gemma4ChatOptions?
    private(set) var lastMessages: [Gemma4ChatMessage] = []

    init(delay: Duration = .milliseconds(20)) { self.delay = delay }

    var maxTokensCap: Int? { nil }
    var acceptsImages: Bool { true }

    func start(
        messages: [Gemma4ChatMessage], tools: [[String: any Sendable]], options: Gemma4ChatOptions
    ) -> Gemma4ChatRun {
        lastOptions = options
        lastMessages = messages
        let (events, continuation) = AsyncThrowingStream<Gemma4ChatEvent, Error>.makeStream()
        let delay = delay
        let task = Task {
            await self.enter()
            var cancelled = false
            for word in ["Bon", "jour"] {
                do { try await Task.sleep(for: delay) } catch { cancelled = true; break }
                continuation.yield(.text(word))
            }
            if !cancelled {
                continuation.yield(.done(Gemma4ChatUsage(promptTokens: 5, completionTokens: 2, finishReason: .stop)))
            }
            continuation.finish()
            await self.leave(cancelled: cancelled)
        }
        continuation.onTermination = { _ in task.cancel() }
        return Gemma4ChatRun(events: events, task: task)
    }

    private func enter() {
        active += 1
        maxActive = max(maxActive, active)
    }

    private func leave(cancelled: Bool) {
        active -= 1
        finished += 1
        if cancelled { cancelledRuns += 1 }
    }
}

@Suite("Serveur Gemma 4")
struct Gemma4ServerTests {

    static let chatBody = #"{"messages":[{"role":"user","content":"Salut"}],"max_tokens":16}"#

    private func app(_ backend: FakeBackend, _ config: Gemma4ServerConfiguration = .init()) async -> some ApplicationProtocol {
        let server = Gemma4Server(backend: backend, configuration: config)
        return Application(router: server.makeRouter())
    }

    @Test("reponse JSON : texte, usage, finish_reason ; max_tokens transmis")
    func testChatJSON() async throws {
        let backend = FakeBackend()
        try await app(backend).test(.router) { client in
            try await client.execute(uri: "/v1/chat/completions", method: .post, body: ByteBuffer(string: Self.chatBody)) { response in
                #expect(response.status == .ok)
                let json = try #require(try JSONSerialization.jsonObject(with: Data(buffer: response.body)) as? [String: Any])
                let choice = try #require((json["choices"] as? [[String: Any]])?.first)
                #expect((choice["message"] as? [String: Any])?["content"] as? String == "Bonjour")
                #expect(choice["finish_reason"] as? String == "stop")
                #expect((json["usage"] as? [String: Any])?["total_tokens"] as? Int == 7)
            }
        }
        #expect(await backend.lastOptions?.maxTokens == 16)
    }

    @Test("SSE : role, deltas, fin et [DONE]")
    func testStream() async throws {
        let body = #"{"messages":[{"role":"user","content":"Salut"}],"stream":true}"#
        try await app(FakeBackend()).test(.router) { client in
            try await client.execute(uri: "/v1/chat/completions", method: .post, body: ByteBuffer(string: body)) { response in
                #expect(response.headers[.contentType]?.hasPrefix("text/event-stream") == true)
                let text = String(buffer: response.body)
                #expect(text.contains(#""role":"assistant""#))
                #expect(text.contains(#""content":"Bon""#) && text.contains(#""content":"jour""#))
                #expect(text.contains(#""finish_reason":"stop""#))
                #expect(text.hasSuffix("data: [DONE]\n\n"))
            }
        }
    }

    @Test("cle d'API : 401 sans cle ou mauvaise cle, 200 avec la bonne ; /healthz ouvert")
    func testAuth() async throws {
        let config = Gemma4ServerConfiguration(apiKey: "secret")
        try await app(FakeBackend(), config).test(.router) { client in
            try await client.execute(uri: "/v1/models", method: .get) { #expect($0.status == .unauthorized) }
            try await client.execute(uri: "/metrics", method: .get, headers: [.authorization: "Bearer nope"]) {
                #expect($0.status == .unauthorized)
            }
            try await client.execute(uri: "/v1/models", method: .get, headers: [.authorization: "Bearer secret"]) {
                #expect($0.status == .ok)
            }
            try await client.execute(uri: "/healthz", method: .get) { #expect($0.status == .ok) }
        }
    }

    @Test("413 : corps trop gros")
    func testBodyTooLarge() async throws {
        let config = Gemma4ServerConfiguration(maxBodyBytes: 32)
        try await app(FakeBackend(), config).test(.router) { client in
            try await client.execute(uri: "/v1/chat/completions", method: .post, body: ByteBuffer(string: Self.chatBody)) {
                #expect($0.status == .contentTooLarge)
            }
        }
    }

    @Test("400 : image file:// ou http, audio, role inconnu")
    func testRejectedMedia() async throws {
        let bodies = [
            #"{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"file:///etc/passwd"}}]}]}"#,
            #"{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.com/a.png"}}]}]}"#,
            #"{"messages":[{"role":"user","content":[{"type":"input_audio","input_audio":{"data":"AA==","format":"wav"}}]}]}"#,
            #"{"messages":[{"role":"robot","content":"x"}]}"#,
        ]
        try await app(FakeBackend()).test(.router) { client in
            for body in bodies {
                try await client.execute(uri: "/v1/chat/completions", method: .post, body: ByteBuffer(string: body)) { response in
                    #expect(response.status == .badRequest, "\(body)")
                    #expect(String(buffer: response.body).contains(#""error""#))
                }
            }
        }
    }

    @Test("demarrage refuse sur 0.0.0.0 sans cle, accepte avec")
    func testRemoteNeedsKey() throws {
        #expect(throws: Gemma4ServerError.remoteAccessNeedsAPIKey("0.0.0.0")) {
            try Gemma4ServerConfiguration(host: "0.0.0.0").validate()
        }
        try Gemma4ServerConfiguration(host: "0.0.0.0", apiKey: "k").validate()
        try Gemma4ServerConfiguration().validate()
    }

    @Test("K-39 : requetes concurrentes serialisees, jamais deux generations a la fois")
    func testSerialized() async throws {
        let backend = FakeBackend(delay: .milliseconds(40))
        try await app(backend).test(.router) { client in
            try await withThrowingTaskGroup(of: Void.self) { group in
                for _ in 0 ..< 3 {
                    group.addTask {
                        try await client.execute(uri: "/v1/chat/completions", method: .post, body: ByteBuffer(string: Self.chatBody)) {
                            #expect($0.status == .ok)
                        }
                    }
                }
                try await group.waitForAll()
            }
        }
        #expect(await backend.maxActive == 1)
        #expect(await backend.finished == 3)
    }

    @Test("lot (K-41) : avec concurrentGenerations = 2, deux generations a la fois au plus")
    func testConcurrentGenerations() async throws {
        let backend = FakeBackend(delay: .milliseconds(40))
        var config = Gemma4ServerConfiguration()
        config.concurrentGenerations = 2
        try await app(backend, config).test(.router) { client in
            try await withThrowingTaskGroup(of: Void.self) { group in
                for _ in 0 ..< 4 {
                    group.addTask {
                        try await client.execute(uri: "/v1/chat/completions", method: .post, body: ByteBuffer(string: Self.chatBody)) {
                            #expect($0.status == .ok)
                        }
                    }
                }
                try await group.waitForAll()
            }
        }
        #expect(await backend.maxActive == 2)
        #expect(await backend.finished == 4)
    }

    @Test("429 : file pleine")
    func testQueueFull() async throws {
        let backend = FakeBackend(delay: .milliseconds(150))
        let config = Gemma4ServerConfiguration(maxQueueDepth: 0)
        try await app(backend, config).test(.router) { client in
            async let first: Void = client.execute(uri: "/v1/chat/completions", method: .post, body: ByteBuffer(string: Self.chatBody)) {
                #expect($0.status == .ok)
            }
            try await Task.sleep(for: .milliseconds(50))
            try await client.execute(uri: "/v1/chat/completions", method: .post, body: ByteBuffer(string: Self.chatBody)) { response in
                #expect(response.status == .tooManyRequests)
            }
            try await first
        }
    }

    @Test("comparaison de cle a temps constant")
    func testConstantTime() {
        #expect(Gemma4Server.constantTimeEquals("abc", "abc"))
        #expect(!Gemma4Server.constantTimeEquals("abc", "abd"))
        #expect(!Gemma4Server.constantTimeEquals("abc", "abcd"))
        #expect(!Gemma4Server.constantTimeEquals("", "a"))
    }
}
