// Format OpenAI Chat Completions : ce que le serveur lit et ecrit.

import Foundation

/// Contenu d'un message : une chaine, ou des parties (`text`, `image_url`, `input_audio`).
enum MessageContent: Decodable {
    case text(String)
    case parts([ContentPart])

    struct ContentPart: Decodable {
        let type: String
        let text: String?
        let imageURL: ImageURL?
        let inputAudio: InputAudio?

        struct ImageURL: Decodable { let url: String }
        struct InputAudio: Decodable { let data: String; let format: String? }

        enum CodingKeys: String, CodingKey {
            case type, text
            case imageURL = "image_url"
            case inputAudio = "input_audio"
        }
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .text("")
        } else if let string = try? container.decode(String.self) {
            self = .text(string)
        } else {
            self = .parts(try container.decode([ContentPart].self))
        }
    }
}

struct RequestToolCall: Decodable {
    let id: String?
    let function: Function
    struct Function: Decodable {
        let name: String
        /// OpenAI : chaine JSON ; tolere un objet.
        let arguments: String

        enum CodingKeys: String, CodingKey { case name, arguments }

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            name = try c.decode(String.self, forKey: .name)
            if let string = try? c.decode(String.self, forKey: .arguments) {
                arguments = string
            } else if let object = try? c.decode(JSONAny.self, forKey: .arguments) {
                arguments = object.jsonString
            } else {
                arguments = "{}"
            }
        }
    }
}

struct RequestMessage: Decodable {
    let role: String
    let content: MessageContent?
    let toolCalls: [RequestToolCall]?
    let toolCallID: String?
    let name: String?

    enum CodingKeys: String, CodingKey {
        case role, content, name
        case toolCalls = "tool_calls"
        case toolCallID = "tool_call_id"
    }
}

struct ChatCompletionRequest: Decodable {
    let model: String?
    let messages: [RequestMessage]
    let tools: [JSONAny]?
    let stream: Bool?
    let maxTokens: Int?
    let maxCompletionTokens: Int?
    let temperature: Float?
    let topP: Float?
    let topK: Int?
    let enableThinking: Bool?
    let chatTemplateKwargs: TemplateKwargs?

    struct TemplateKwargs: Decodable {
        let enableThinking: Bool?
        enum CodingKeys: String, CodingKey { case enableThinking = "enable_thinking" }
    }

    enum CodingKeys: String, CodingKey {
        case model, messages, tools, stream, temperature
        case maxTokens = "max_tokens"
        case maxCompletionTokens = "max_completion_tokens"
        case topP = "top_p"
        case topK = "top_k"
        case enableThinking = "enable_thinking"
        case chatTemplateKwargs = "chat_template_kwargs"
    }
}

/// Valeur JSON quelconque (outils declares par le client), reconvertie en dictionnaire
/// `Sendable` pour le gabarit.
struct JSONAny: Decodable, @unchecked Sendable {
    let value: Any

    init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() { value = NSNull() }
        else if let b = try? c.decode(Bool.self) { value = b }
        else if let i = try? c.decode(Int.self) { value = i }
        else if let d = try? c.decode(Double.self) { value = d }
        else if let s = try? c.decode(String.self) { value = s }
        else if let a = try? c.decode([JSONAny].self) { value = a.map(\.value) }
        else { value = try c.decode([String: JSONAny].self).mapValues(\.value) }
    }

    var jsonString: String {
        guard JSONSerialization.isValidJSONObject(value),
              let data = try? JSONSerialization.data(withJSONObject: value, options: [.sortedKeys])
        else { return "{}" }
        return String(decoding: data, as: UTF8.self)
    }

    /// Conversion recursive vers des types `Sendable` (le gabarit attend
    /// `[String: any Sendable]`).
    static func sendable(_ value: Any) -> any Sendable {
        switch value {
        case let dict as [String: Any]: return dict.mapValues { sendable($0) } as [String: any Sendable]
        case let array as [Any]: return array.map { sendable($0) } as [any Sendable]
        case let s as String: return s
        case let b as Bool: return b
        case let i as Int: return i
        case let d as Double: return d
        default: return ""
        }
    }
}

// MARK: - Reponses

struct ChatCompletionResponse: Encodable {
    let id: String
    let object = "chat.completion"
    let created: Int
    let model: String
    let choices: [Choice]
    let usage: Usage

    struct Choice: Encodable {
        let index = 0
        let message: Message
        let finishReason: String
        enum CodingKeys: String, CodingKey { case index, message; case finishReason = "finish_reason" }
    }

    struct Message: Encodable {
        let role = "assistant"
        let content: String?
        let reasoningContent: String?
        let toolCalls: [ToolCall]?
        enum CodingKeys: String, CodingKey {
            case role, content
            case reasoningContent = "reasoning_content"
            case toolCalls = "tool_calls"
        }
    }
}

struct ToolCall: Encodable {
    let index: Int?
    let id: String
    let type = "function"
    let function: Function
    struct Function: Encodable { let name: String; let arguments: String }
}

struct Usage: Encodable {
    let promptTokens: Int
    let completionTokens: Int
    let totalTokens: Int
    enum CodingKeys: String, CodingKey {
        case promptTokens = "prompt_tokens"
        case completionTokens = "completion_tokens"
        case totalTokens = "total_tokens"
    }
}

struct ChatCompletionChunk: Encodable {
    let id: String
    let object = "chat.completion.chunk"
    let created: Int
    let model: String
    let choices: [Choice]
    let usage: Usage?

    struct Choice: Encodable {
        let index = 0
        let delta: Delta
        let finishReason: String?
        enum CodingKeys: String, CodingKey { case index, delta; case finishReason = "finish_reason" }
    }

    struct Delta: Encodable {
        var role: String?
        var content: String?
        var reasoningContent: String?
        var toolCalls: [ToolCall]?
        enum CodingKeys: String, CodingKey {
            case role, content
            case reasoningContent = "reasoning_content"
            case toolCalls = "tool_calls"
        }
    }
}

struct ErrorBody: Encodable {
    let error: Detail
    struct Detail: Encodable {
        let message: String
        let type: String
        let code: String?
    }
}

struct ModelList: Encodable {
    let object = "list"
    let data: [Model]
    struct Model: Encodable {
        let id: String
        let object = "model"
        let ownedBy = "local"
        enum CodingKeys: String, CodingKey { case id, object; case ownedBy = "owned_by" }
    }
}
