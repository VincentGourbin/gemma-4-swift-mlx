// API Messages d'Anthropic (K-42) : `POST /v1/messages`, pour brancher Claude Code
// (ANTHROPIC_BASE_URL) ou un client Anthropic sur un Gemma 4 local.
//
// Pris en charge : system (chaine ou blocs), texte, images en base64, outils (tool_use /
// tool_result), pensee (`thinking: {type: "enabled"}` -> canal de pensee), SSE complet
// (message_start … message_stop). Refuse clairement ce que le moteur ne sait pas faire :
// stop_sequences, images par URL, documents, images dans un tool_result.

import Foundation
import Gemma4Swift
import Hummingbird

// MARK: - Requete

struct AnthropicRequest: Decodable {
    let model: String?
    let maxTokens: Int?
    let system: AnthropicText?
    let messages: [AnthropicMessage]
    let tools: [AnthropicTool]?
    let toolChoice: ToolChoice?
    let stream: Bool?
    let temperature: Float?
    let topP: Float?
    let topK: Int?
    let stopSequences: [String]?
    let thinking: Thinking?

    struct ToolChoice: Decodable { let type: String }
    struct Thinking: Decodable { let type: String }

    enum CodingKeys: String, CodingKey {
        case model, system, messages, tools, stream, temperature, thinking
        case maxTokens = "max_tokens"
        case toolChoice = "tool_choice"
        case topP = "top_p"
        case topK = "top_k"
        case stopSequences = "stop_sequences"
    }
}

/// `system` et contenu de `tool_result` : une chaine ou des blocs.
enum AnthropicText: Decodable {
    case text(String)
    case blocks([AnthropicBlock])

    init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if let s = try? c.decode(String.self) { self = .text(s) } else { self = .blocks(try c.decode([AnthropicBlock].self)) }
    }
}

struct AnthropicMessage: Decodable {
    let role: String
    let content: AnthropicContent
}

enum AnthropicContent: Decodable {
    case text(String)
    case blocks([AnthropicContentBlock])

    init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if let s = try? c.decode(String.self) { self = .text(s) } else { self = .blocks(try c.decode([AnthropicContentBlock].self)) }
    }
}

/// Bloc simple (texte, image) : ce que contient un `system` ou un `tool_result`.
struct AnthropicBlock: Decodable {
    let type: String
    let text: String?
    let source: ImageSource?

    struct ImageSource: Decodable {
        let type: String
        let mediaType: String?
        let data: String?
        enum CodingKeys: String, CodingKey { case type, data; case mediaType = "media_type" }
    }
}

/// Bloc d'un message : texte, image, tool_use, tool_result, thinking.
struct AnthropicContentBlock: Decodable {
    let type: String
    let text: String?
    let source: AnthropicBlock.ImageSource?
    let id: String?
    let name: String?
    let input: JSONAny?
    let toolUseID: String?
    let content: AnthropicText?
    let isError: Bool?

    enum CodingKeys: String, CodingKey {
        case type, text, source, id, name, input, content
        case toolUseID = "tool_use_id"
        case isError = "is_error"
    }
}

struct AnthropicTool: Decodable {
    let name: String
    let description: String?
    let inputSchema: JSONAny?
    enum CodingKeys: String, CodingKey { case name, description; case inputSchema = "input_schema" }

    /// Au format « function » attendu par le gabarit Gemma 4 (le meme que l'API OpenAI).
    var openAIFunction: [String: any Sendable] {
        var function: [String: any Sendable] = ["name": name]
        if let description { function["description"] = description }
        function["parameters"] = inputSchema.map { JSONAny.sendable($0.value) }
            ?? (["type": "object", "properties": [String: any Sendable]()] as [String: any Sendable])
        return ["type": "function", "function": function]
    }
}

// MARK: - Route

extension Gemma4Server {

    func anthropicMessages(_ request: Request) async throws -> Response {
        try authorize(request)
        noteRequest()
        let input: AnthropicRequest
        do {
            input = try JSONDecoder().decode(AnthropicRequest.self, from: try await readBody(request))
        } catch let error as Gemma4ServerError {
            throw error
        } catch {
            throw Gemma4ServerError.invalidRequest("requete messages invalide : \(error.localizedDescription)")
        }
        if let stops = input.stopSequences, !stops.isEmpty {
            throw Gemma4ServerError.invalidRequest("stop_sequences n'est pas pris en charge par ce serveur")
        }
        let messages = try await anthropicConvert(input)
        let tools = input.toolChoice?.type == "none" ? [] : (input.tools ?? []).map(\.openAIFunction)
        var options = Gemma4ChatOptions(
            maxTokens: await cappedMaxTokens(input.maxTokens),
            // Claude Code envoie `{"type": "adaptive"}` ; `disabled` ou absent : sans pensee.
            enableThinking: ["enabled", "adaptive"].contains(input.thinking?.type ?? ""))
        if let t = input.temperature { options.temperature = t }
        if let p = input.topP { options.topP = p }
        if let k = input.topK { options.topK = k }
        let (run, release) = try await begin(messages: messages, tools: tools, options: options)
        let id = "msg_" + UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(24)
        let model = input.model ?? configuration.modelID

        if input.stream == true {
            return anthropicStream(run: run, release: release, id: String(id), model: model)
        }
        defer { Task { await release.fire() } }
        var reasoning = ""
        var text = ""
        var calls: [Gemma4ToolCall] = []
        var usage: Gemma4ChatUsage?
        do {
            for try await event in run.events {
                switch event {
                case .text(let t): text += t
                case .reasoning(let t): reasoning += t
                case .toolCall(let call): calls.append(call)
                case .done(let done): usage = done
                }
            }
        } catch {
            noteFailure()
            throw error
        }
        record(usage)
        var content: [[String: Any]] = []
        if !reasoning.isEmpty { content.append(["type": "thinking", "thinking": reasoning, "signature": ""]) }
        if !text.isEmpty || (content.isEmpty && calls.isEmpty) { content.append(["type": "text", "text": text]) }
        for call in calls {
            content.append(["type": "tool_use", "id": call.id, "name": call.name, "input": Self.jsonInput(call.argumentsJSON)])
        }
        return Self.jsonObject([
            "id": String(id), "type": "message", "role": "assistant", "model": model, "content": content,
            "stop_reason": Self.stopReason(usage?.finishReason), "stop_sequence": NSNull(),
            "usage": Self.anthropicUsage(usage),
        ])
    }

    // MARK: SSE

    private func anthropicStream(run: Gemma4ChatRun, release: ReleaseOnce, id: String, model: String) -> Response {
        let body = ResponseBody { [self] writer in
            func send(_ event: String, _ payload: [String: Any]) async throws {
                var all = payload
                all["type"] = event
                var buffer = ByteBuffer(string: "event: \(event)\ndata: ")
                buffer.writeBytes(try JSONSerialization.data(withJSONObject: all, options: [.sortedKeys]))
                buffer.writeString("\n\n")
                try await writer.write(buffer)
            }
            /// Bloc ouvert : type ("thinking" / "text") et index.
            var open: (kind: String, index: Int)?
            var nextIndex = 0
            func close() async throws {
                guard let current = open else { return }
                try await send("content_block_stop", ["index": current.index])
                open = nil
            }
            func ensure(_ kind: String) async throws -> Int {
                if let current = open, current.kind == kind { return current.index }
                try await close()
                let block: [String: Any] = kind == "thinking"
                    ? ["type": "thinking", "thinking": "", "signature": ""] : ["type": "text", "text": ""]
                try await send("content_block_start", ["index": nextIndex, "content_block": block])
                open = (kind, nextIndex)
                nextIndex += 1
                return nextIndex - 1
            }
            var outcome: Outcome = .failed
            do {
                try await send("message_start", ["message": [
                    "id": id, "type": "message", "role": "assistant", "model": model, "content": [Any](),
                    "stop_reason": NSNull(), "stop_sequence": NSNull(),
                    "usage": ["input_tokens": 0, "output_tokens": 0],
                ] as [String: Any]])
                for try await event in run.events {
                    switch event {
                    case .reasoning(let t):
                        let index = try await ensure("thinking")
                        try await send("content_block_delta", ["index": index, "delta": ["type": "thinking_delta", "thinking": t]])
                    case .text(let t):
                        let index = try await ensure("text")
                        try await send("content_block_delta", ["index": index, "delta": ["type": "text_delta", "text": t]])
                    case .toolCall(let call):
                        try await close()
                        let index = nextIndex
                        nextIndex += 1
                        try await send("content_block_start", ["index": index, "content_block": [
                            "type": "tool_use", "id": call.id, "name": call.name, "input": [String: Any](),
                        ] as [String: Any]])
                        try await send("content_block_delta", ["index": index, "delta": [
                            "type": "input_json_delta", "partial_json": call.argumentsJSON,
                        ]])
                        try await send("content_block_stop", ["index": index])
                    case .done(let usage):
                        try await close()
                        await self.record(usage)
                        try await send("message_delta", [
                            "delta": ["stop_reason": Self.stopReason(usage.finishReason), "stop_sequence": NSNull()] as [String: Any],
                            "usage": Self.anthropicUsage(usage),
                        ])
                        try await send("message_stop", [:])
                    }
                }
                try await writer.finish(nil)
                outcome = .completed
            } catch {
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

    // MARK: Conversion

    /// Messages Anthropic -> messages du moteur. Un message user qui porte des `tool_result`
    /// devient un message `tool` par resultat (dans l'ordre), puis le texte restant.
    func anthropicConvert(_ input: AnthropicRequest) async throws -> [Gemma4ChatMessage] {
        guard !input.messages.isEmpty else { throw Gemma4ServerError.invalidRequest("messages vide") }
        var result: [Gemma4ChatMessage] = []
        if let system = input.system {
            let text = try Self.plainText(system, context: "system")
            if !text.isEmpty { result.append(Gemma4ChatMessage(role: .system, content: text)) }
        }
        var toolNames: [String: String] = [:]
        var mediaCount = 0
        for message in input.messages {
            guard ["user", "assistant", "system"].contains(message.role) else {
                throw Gemma4ServerError.invalidRequest("role inconnu : \(message.role) (user, assistant ou system)")
            }
            let blocks: [AnthropicContentBlock]
            switch message.content {
            case .text(let s):
                let role: Gemma4ChatMessage.Role = message.role == "user" ? .user : message.role == "system" ? .system : .assistant
                result.append(Gemma4ChatMessage(role: role, content: s))
                continue
            case .blocks(let b): blocks = b
            }
            // Claude Code (2.1.x) place un message `system` apres le tour user (contexte de session).
            // Garde a sa place : le gabarit rend un tour system a toute position, et le fusionner en
            // tete casserait le cache de prefixe quand son contenu change.
            if message.role == "system" {
                let text = try blocks.map { block -> String in
                    guard block.type == "text" else {
                        throw Gemma4ServerError.invalidRequest("message system : seuls les blocs texte sont acceptes (\(block.type))")
                    }
                    return block.text ?? ""
                }.joined(separator: "\n")
                if !text.isEmpty { result.append(Gemma4ChatMessage(role: .system, content: text)) }
                continue
            }
            var text = ""
            var images: [Data] = []
            var calls: [Gemma4ToolCall] = []
            for block in blocks {
                switch block.type {
                case "text":
                    text += block.text ?? ""
                case "image":
                    guard message.role == "user" else {
                        throw Gemma4ServerError.invalidRequest("image dans un message assistant")
                    }
                    mediaCount += 1
                    guard mediaCount <= configuration.maxMediaPerRequest else {
                        throw Gemma4ServerError.mediaTooLarge("plus de \(configuration.maxMediaPerRequest) medias")
                    }
                    guard block.source?.type == "base64", let data = block.source?.data else {
                        throw Gemma4ServerError.invalidRequest("image : seule la source base64 est acceptee (pas d'URL)")
                    }
                    images.append(try checkedImage(base64: data))
                case "tool_use":
                    guard message.role == "assistant", let id = block.id, let name = block.name else {
                        throw Gemma4ServerError.invalidRequest("tool_use invalide (assistant, id et name requis)")
                    }
                    toolNames[id] = name
                    calls.append(Gemma4ToolCall(id: id, name: name, argumentsJSON: block.input?.jsonString ?? "{}"))
                case "tool_result":
                    guard message.role == "user", let id = block.toolUseID else {
                        throw Gemma4ServerError.invalidRequest("tool_result invalide (user, tool_use_id requis)")
                    }
                    var output = try block.content.map { try Self.plainText($0, context: "tool_result") } ?? ""
                    if block.isError == true { output = "Erreur : " + output }
                    result.append(Gemma4ChatMessage(
                        role: .tool, content: output, toolName: toolNames[id], toolCallID: id))
                case "thinking", "redacted_thinking":
                    // Le gabarit retire la pensee des tours passes : rien a transmettre.
                    continue
                default:
                    throw Gemma4ServerError.invalidRequest("bloc de contenu non pris en charge : \(block.type)")
                }
            }
            if !images.isEmpty, !(await imagesAccepted) {
                throw Gemma4ServerError.invalidRequest("le modele est charge sans vision")
            }
            if message.role == "assistant" {
                result.append(Gemma4ChatMessage(role: .assistant, content: text, toolCalls: calls))
            } else if !text.isEmpty || !images.isEmpty {
                result.append(Gemma4ChatMessage(role: .user, content: text, images: images))
            }
        }
        return result
    }

    /// Texte seul (system, tool_result) ; une image a cet endroit est refusee.
    static func plainText(_ value: AnthropicText, context: String) throws -> String {
        switch value {
        case .text(let s): return s
        case .blocks(let blocks):
            return try blocks.map { block -> String in
                guard block.type == "text" else {
                    throw Gemma4ServerError.invalidRequest("\(context) : seuls les blocs texte sont acceptes (\(block.type))")
                }
                return block.text ?? ""
            }.joined(separator: "\n")
        }
    }

    static func stopReason(_ reason: Gemma4ChatUsage.FinishReason?) -> String {
        switch reason {
        case .length: return "max_tokens"
        case .toolCalls: return "tool_use"
        case .stop, .cancelled, .none: return "end_turn"
        }
    }

    /// `input_tokens` hors jetons repris du cache de conversation, comme l'API Anthropic.
    static func anthropicUsage(_ usage: Gemma4ChatUsage?) -> [String: Any] {
        let prompt = usage?.promptTokens ?? 0
        let cached = usage?.cachedPromptTokens ?? 0
        return [
            "input_tokens": max(0, prompt - cached),
            "output_tokens": usage?.completionTokens ?? 0,
            "cache_read_input_tokens": cached,
            "cache_creation_input_tokens": 0,
        ]
    }

    /// Arguments d'un appel d'outil en objet JSON (`{}` si illisibles).
    static func jsonInput(_ json: String) -> Any {
        (try? JSONSerialization.jsonObject(with: Data(json.utf8))) ?? [String: Any]()
    }
}
