// Serveur OpenAI-compatible pour Gemma 4 (K-37 a K-39) : adaptateur HTTP autour d'un
// `Gemma4ChatBackend` (le moteur de la bibliotheque, ou un faux en test).
//
// Securite (K-38) : 127.0.0.1 par defaut, hote non-loopback refuse sans cle, cle comparee
// a temps constant, /metrics authentifie et sans contenu, limites (corps, medias,
// pixels, max_tokens, file), images en `data:` seulement, decodees en memoire.
// Serialisation (K-39) : un modele, une generation a la fois ; la file est tenue jusqu'a
// la fin **reelle** du calcul, y compris apres une deconnexion (annulation < 1 pas).

import CoreGraphics
import Foundation
import Gemma4Swift
import Hummingbird
import ImageIO
import NIOCore

public struct Gemma4ServerConfiguration: Sendable {
    public var host: String
    public var port: Int
    public var apiKey: String?
    /// Identifiant publie par `/v1/models` et renvoye dans les reponses.
    public var modelID: String
    public var maxBodyBytes: Int
    public var maxMediaPerRequest: Int
    public var maxImagePixels: Int
    /// Requetes en attente au-dela de celle qui tourne ; au-dela : 429.
    public var maxQueueDepth: Int
    public var defaultMaxTokens: Int
    public var maxTokensCap: Int

    public init(
        host: String = "127.0.0.1", port: Int = 8080, apiKey: String? = nil, modelID: String = "gemma-4",
        maxBodyBytes: Int = 32 << 20, maxMediaPerRequest: Int = 4, maxImagePixels: Int = 20_000_000,
        maxQueueDepth: Int = 16, defaultMaxTokens: Int = 1024, maxTokensCap: Int = 8192
    ) {
        self.host = host
        self.port = port
        self.apiKey = apiKey
        self.modelID = modelID
        self.maxBodyBytes = maxBodyBytes
        self.maxMediaPerRequest = maxMediaPerRequest
        self.maxImagePixels = maxImagePixels
        self.maxQueueDepth = maxQueueDepth
        self.defaultMaxTokens = defaultMaxTokens
        self.maxTokensCap = maxTokensCap
    }

    public static func isLoopback(_ host: String) -> Bool {
        ["127.0.0.1", "::1", "localhost"].contains(host.lowercased())
    }

    /// Refus de demarrer : ecouter hors loopback sans cle ouvrirait le modele au reseau.
    public func validate() throws {
        if !Self.isLoopback(host) && (apiKey ?? "").isEmpty {
            throw Gemma4ServerError.remoteAccessNeedsAPIKey(host)
        }
        guard (1 ... 65535).contains(port) else { throw Gemma4ServerError.invalidRequest("port invalide : \(port)") }
    }
}

public enum Gemma4ServerError: Error, LocalizedError, Equatable {
    case remoteAccessNeedsAPIKey(String)
    case unauthorized
    case invalidRequest(String)
    case mediaTooLarge(String)
    case queueFull

    public var errorDescription: String? {
        switch self {
        case .remoteAccessNeedsAPIKey(let host):
            return "ecoute sur \(host) refusee sans --api-key (seul le loopback est ouvert sans cle)"
        case .unauthorized: return "cle d'API absente ou invalide"
        case .invalidRequest(let message): return message
        case .mediaTooLarge(let message): return message
        case .queueFull: return "file pleine, reessayer plus tard"
        }
    }

    var status: HTTPResponse.Status {
        switch self {
        case .remoteAccessNeedsAPIKey, .invalidRequest: return .badRequest
        case .unauthorized: return .unauthorized
        case .mediaTooLarge: return .contentTooLarge
        case .queueFull: return .tooManyRequests
        }
    }

    var type: String {
        switch self {
        case .unauthorized: return "authentication_error"
        case .queueFull: return "rate_limit_error"
        default: return "invalid_request_error"
        }
    }
}

/// File FIFO a une place : une generation a la fois, `maxDepth` en attente.
actor RequestQueue {
    private var busy = false
    private var waiters: [CheckedContinuation<Void, Never>] = []
    let maxDepth: Int

    init(maxDepth: Int) { self.maxDepth = maxDepth }

    var waiting: Int { waiters.count }
    var running: Bool { busy }

    func acquire() async throws {
        if !busy {
            busy = true
            return
        }
        guard waiters.count < maxDepth else { throw Gemma4ServerError.queueFull }
        await withCheckedContinuation { waiters.append($0) }
    }

    func release() {
        if waiters.isEmpty {
            busy = false
        } else {
            waiters.removeFirst().resume()
        }
    }
}

/// Compteurs publies par `/metrics` (jamais de contenu genere).
struct ServerCounters: Encodable {
    var requests = 0
    var completed = 0
    var cancelled = 0
    var failed = 0
    var rejected = 0
    var promptTokens = 0
    var completionTokens = 0
}

public actor Gemma4Server {
    public let configuration: Gemma4ServerConfiguration
    private let backend: any Gemma4ChatBackend
    private let queue: RequestQueue
    private var counters = ServerCounters()

    public init(backend: any Gemma4ChatBackend, configuration: Gemma4ServerConfiguration) {
        self.backend = backend
        self.configuration = configuration
        self.queue = RequestQueue(maxDepth: configuration.maxQueueDepth)
    }

    /// Demarre et bloque jusqu'a l'arret du service.
    public func run() async throws {
        try configuration.validate()
        let application = Application(
            router: makeRouter(),
            configuration: .init(
                address: .hostname(configuration.host, port: configuration.port), serverName: "gemma4-server"))
        try await application.runService()
    }

    public nonisolated func makeRouter() -> Router<BasicRequestContext> {
        let router = Router()
        router.get("healthz") { _, _ in
            Self.json(["status": "ok"])
        }
        router.get("v1/models") { [self] request, _ in
            await self.handling { try await self.models(request) }
        }
        router.get("metrics") { [self] request, _ in
            await self.handling { try await self.metrics(request) }
        }
        router.post("v1/chat/completions") { [self] request, _ in
            await self.handling { try await self.chatCompletions(request) }
        }
        return router
    }

    // MARK: - Routes

    private func models(_ request: Request) throws -> Response {
        try authorize(request)
        return Self.json(ModelList(data: [.init(id: configuration.modelID)]))
    }

    private func metrics(_ request: Request) async throws -> Response {
        try authorize(request)
        struct Metrics: Encodable {
            let counters: ServerCounters
            let queueWaiting: Int
            let running: Bool
            enum CodingKeys: String, CodingKey { case counters; case queueWaiting = "queue_waiting"; case running }
        }
        return Self.json(Metrics(counters: counters, queueWaiting: await queue.waiting, running: await queue.running))
    }

    private func chatCompletions(_ request: Request) async throws -> Response {
        try authorize(request)
        counters.requests += 1
        var request = request
        let body: ByteBuffer
        do {
            body = try await request.collectBody(upTo: configuration.maxBodyBytes)
        } catch {
            throw Gemma4ServerError.mediaTooLarge("corps de requete au-dela de \(configuration.maxBodyBytes) octets")
        }
        let input: ChatCompletionRequest
        do {
            input = try JSONDecoder().decode(ChatCompletionRequest.self, from: Data(buffer: body))
        } catch {
            throw Gemma4ServerError.invalidRequest("requete chat invalide : \(error.localizedDescription)")
        }
        let messages = try await convert(input.messages)
        let tools = (input.tools ?? []).compactMap { JSONAny.sendable($0.value) as? [String: any Sendable] }
        var options = Gemma4ChatOptions(
            maxTokens: min(input.maxCompletionTokens ?? input.maxTokens ?? configuration.defaultMaxTokens,
                           configuration.maxTokensCap, await backend.maxTokensCap ?? .max),
            enableThinking: input.chatTemplateKwargs?.enableThinking ?? input.enableThinking ?? false)
        if let t = input.temperature { options.temperature = t }
        if let p = input.topP { options.topP = p }
        if let k = input.topK { options.topK = k }
        guard options.maxTokens > 0 else { throw Gemma4ServerError.invalidRequest("max_tokens doit etre > 0") }

        do { try await queue.acquire() } catch {
            counters.rejected += 1
            throw error
        }
        let run = await backend.start(messages: messages, tools: tools, options: options)
        let release = ReleaseOnce { [queue] in
            await run.waitUntilFinished()
            await queue.release()
        }
        let id = "chatcmpl-" + UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(24)
        let model = input.model ?? configuration.modelID
        let created = Int(Date().timeIntervalSince1970)

        if input.stream == true {
            return streamResponse(run: run, release: release, id: String(id), model: model, created: created)
        }
        defer { Task { await release.fire() } }
        var content = ""
        var reasoning = ""
        var calls: [ToolCall] = []
        var usage: Gemma4ChatUsage?
        do {
            for try await event in run.events {
                switch event {
                case .text(let text): content += text
                case .reasoning(let text): reasoning += text
                case .toolCall(let call):
                    calls.append(ToolCall(index: nil, id: call.id, function: .init(name: call.name, arguments: call.argumentsJSON)))
                case .done(let done): usage = done
                }
            }
        } catch {
            counters.failed += 1
            throw error
        }
        record(usage)
        let response = ChatCompletionResponse(
            id: String(id), created: created, model: model,
            choices: [.init(
                message: .init(content: content.isEmpty && !calls.isEmpty ? nil : content,
                               reasoningContent: reasoning.isEmpty ? nil : reasoning,
                               toolCalls: calls.isEmpty ? nil : calls),
                finishReason: usage?.finishReason.rawValue ?? "stop")],
            usage: Usage(promptTokens: usage?.promptTokens ?? 0, completionTokens: usage?.completionTokens ?? 0,
                         totalTokens: (usage?.promptTokens ?? 0) + (usage?.completionTokens ?? 0)))
        return Self.json(response)
    }

    private func streamResponse(
        run: Gemma4ChatRun, release: ReleaseOnce, id: String, model: String, created: Int
    ) -> Response {
        let body = ResponseBody { [self] writer in
            func send(_ delta: ChatCompletionChunk.Delta, finish: String? = nil, usage: Usage? = nil) async throws {
                let chunk = ChatCompletionChunk(
                    id: id, created: created, model: model,
                    choices: [.init(delta: delta, finishReason: finish)], usage: usage)
                var line = ByteBuffer(string: "data: ")
                line.writeBytes(try JSONEncoder().encode(chunk))
                line.writeString("\n\n")
                try await writer.write(line)
            }
            var toolIndex = 0
            var outcome: Outcome = .failed
            do {
                try await send(.init(role: "assistant"))
                for try await event in run.events {
                    switch event {
                    case .text(let text): try await send(.init(content: text))
                    case .reasoning(let text): try await send(.init(reasoningContent: text))
                    case .toolCall(let call):
                        try await send(.init(toolCalls: [ToolCall(
                            index: toolIndex, id: call.id,
                            function: .init(name: call.name, arguments: call.argumentsJSON))]))
                        toolIndex += 1
                    case .done(let usage):
                        await self.record(usage)
                        try await send(
                            .init(), finish: usage.finishReason.rawValue,
                            usage: Usage(promptTokens: usage.promptTokens, completionTokens: usage.completionTokens,
                                         totalTokens: usage.promptTokens + usage.completionTokens))
                    }
                }
                try await writer.write(ByteBuffer(string: "data: [DONE]\n\n"))
                try await writer.finish(nil)
                outcome = .completed
            } catch {
                // Client parti (ecriture impossible) ou erreur du moteur : on arrete le
                // calcul, et la file n'est rendue qu'une fois le calcul vraiment fini.
                run.cancel()
                outcome = error is CancellationError || Self.isWriteFailure(error) ? .cancelled : .failed
            }
            await self.count(outcome)
            await release.fire()
        }
        var headers = HTTPFields()
        headers[.contentType] = "text/event-stream; charset=utf-8"
        headers[.cacheControl] = "no-cache"
        return Response(status: .ok, headers: headers, body: body)
    }

    enum Outcome { case completed, cancelled, failed }

    private func count(_ outcome: Outcome) {
        switch outcome {
        case .completed: break
        case .cancelled: counters.cancelled += 1
        case .failed: counters.failed += 1
        }
    }

    private func record(_ usage: Gemma4ChatUsage?) {
        guard let usage else { return }
        counters.completed += 1
        counters.promptTokens += usage.promptTokens
        counters.completionTokens += usage.completionTokens
    }

    private static func isWriteFailure(_ error: any Error) -> Bool {
        error is ChannelError || error is IOError
    }

    // MARK: - Conversion et controle des medias

    private func convert(_ input: [RequestMessage]) async throws -> [Gemma4ChatMessage] {
        guard !input.isEmpty else { throw Gemma4ServerError.invalidRequest("messages vide") }
        var mediaCount = 0
        var result: [Gemma4ChatMessage] = []
        for message in input {
            guard let role = Gemma4ChatMessage.Role(rawValue: message.role == "developer" ? "system" : message.role) else {
                throw Gemma4ServerError.invalidRequest("role inconnu : \(message.role)")
            }
            var text = ""
            var images: [Data] = []
            switch message.content {
            case .none: break
            case .text(let string): text = string
            case .parts(let parts):
                for part in parts {
                    switch part.type {
                    case "text": text += part.text ?? ""
                    case "image_url":
                        mediaCount += 1
                        guard mediaCount <= configuration.maxMediaPerRequest else {
                            throw Gemma4ServerError.mediaTooLarge("plus de \(configuration.maxMediaPerRequest) medias")
                        }
                        images.append(try decodeImage(part.imageURL?.url ?? ""))
                    case "input_audio":
                        throw Gemma4ServerError.invalidRequest("input_audio n'est pas encore pris en charge par ce serveur")
                    default:
                        throw Gemma4ServerError.invalidRequest("partie de contenu inconnue : \(part.type)")
                    }
                }
            }
            if !images.isEmpty, !(await backend.acceptsImages) {
                throw Gemma4ServerError.invalidRequest("le modele est charge sans vision")
            }
            let calls = (message.toolCalls ?? []).map {
                Gemma4ToolCall(id: $0.id ?? "call_" + UUID().uuidString.prefix(8), name: $0.function.name,
                               argumentsJSON: $0.function.arguments)
            }
            result.append(Gemma4ChatMessage(
                role: role, content: text, images: images, toolCalls: calls,
                toolName: message.name, toolCallID: message.toolCallID))
        }
        return result
    }

    /// `data:image/…;base64,…` seulement : ni `file://` (lecture de fichiers locaux) ni
    /// `http(s)://` (le serveur ne va rien chercher). Taille controlee apres decodage des
    /// seuls en-tetes de l'image.
    private func decodeImage(_ url: String) throws -> Data {
        guard url.hasPrefix("data:"), let comma = url.firstIndex(of: ","),
              url[..<comma].hasSuffix(";base64")
        else {
            throw Gemma4ServerError.invalidRequest("image_url doit etre une URL data: en base64 (ni file:// ni http)")
        }
        guard let data = Data(base64Encoded: String(url[url.index(after: comma)...]), options: .ignoreUnknownCharacters),
              let source = CGImageSourceCreateWithData(data as CFData, nil),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let width = properties[kCGImagePropertyPixelWidth] as? Int,
              let height = properties[kCGImagePropertyPixelHeight] as? Int
        else { throw Gemma4ServerError.invalidRequest("image illisible") }
        guard width * height <= configuration.maxImagePixels else {
            throw Gemma4ServerError.mediaTooLarge(
                "image de \(width)x\(height) au-dela de \(configuration.maxImagePixels) pixels")
        }
        return data
    }

    // MARK: - Authentification et reponses

    private func authorize(_ request: Request) throws {
        guard let key = configuration.apiKey, !key.isEmpty else { return }
        let header = request.headers[.authorization] ?? ""
        let presented = header.hasPrefix("Bearer ") ? String(header.dropFirst(7)) : ""
        guard Self.constantTimeEquals(presented, key) else { throw Gemma4ServerError.unauthorized }
    }

    static func constantTimeEquals(_ a: String, _ b: String) -> Bool {
        let x = Array(a.utf8), y = Array(b.utf8)
        var diff = UInt8(x.count == y.count ? 0 : 1)
        for i in 0 ..< max(x.count, y.count) {
            diff |= (i < x.count ? x[i] : 0) ^ (i < y.count ? y[i] : 0)
        }
        return diff == 0
    }

    private func handling(_ handler: () async throws -> Response) async -> Response {
        do {
            return try await handler()
        } catch let error as Gemma4ServerError {
            return Self.error(error.status, message: error.localizedDescription, type: error.type)
        } catch let error as Gemma4ChatEngineError {
            return Self.error(.badRequest, message: error.localizedDescription, type: "invalid_request_error")
        } catch {
            return Self.error(.internalServerError, message: "erreur interne", type: "server_error")
        }
    }

    static func error(_ status: HTTPResponse.Status, message: String, type: String) -> Response {
        var response = json(ErrorBody(error: .init(message: message, type: type, code: nil)))
        response.status = status
        return response
    }

    static func json(_ value: some Encodable) -> Response {
        let data = (try? JSONEncoder().encode(value)) ?? Data("{}".utf8)
        var headers = HTTPFields()
        headers[.contentType] = "application/json"
        return Response(status: .ok, headers: headers, body: .init(byteBuffer: ByteBuffer(bytes: data)))
    }
}

/// Action executee une seule fois, quel que soit le chemin (fin normale, erreur, client parti).
final class ReleaseOnce: @unchecked Sendable {
    private let lock = NSLock()
    private var fired = false
    private let action: @Sendable () async -> Void

    init(_ action: @escaping @Sendable () async -> Void) { self.action = action }

    func fire() async {
        let first = lock.withLock { () -> Bool in
            defer { fired = true }
            return !fired
        }
        if first { await action() }
    }
}
