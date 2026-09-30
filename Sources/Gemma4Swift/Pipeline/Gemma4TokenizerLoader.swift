// TokenizerLoader local pour Gemma 4
// Charge un tokenizer depuis un repertoire local (tokenizer.json)

import Foundation
import MLXLMCommon
import Tokenizers

/// Charge un tokenizer Gemma 4 depuis un repertoire local contenant tokenizer.json.
/// Utilise swift-transformers AutoTokenizer en interne.
public struct Gemma4TokenizerLoader: TokenizerLoader {
    public init() {}

    public func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        let upstream = try await AutoTokenizer.from(modelFolder: directory)
        return Gemma4TokenizerBridge(upstream, chatTemplate: Self.normalizedChatTemplate(in: directory))
    }

    /// Gabarit du dossier (`chat_template.jinja`, sinon `tokenizer_config.json`), avec le
    /// controle d'espaces de Jinja applique au texte : blancs retires avant `{{-`, `{%-`,
    /// `{#-` et apres `-}}`, `-%}`, `-#}`. C'est la regle de la specification, que
    /// swift-jinja n'applique pas partout — `properties:{ {{- … -}} }` des outils Gemma 4
    /// y gardait l'espace (`{ properties:{ city:{` au lieu de `{properties:{city:{` chez HF,
    /// `ChatEngineTemplateTests`). Sans effet sur un gabarit qui n'a pas de bloc `raw`.
    static func normalizedChatTemplate(in directory: URL) -> String? {
        var template = try? String(contentsOf: directory.appendingPathComponent("chat_template.jinja"), encoding: .utf8)
        if template == nil,
           let data = try? Data(contentsOf: directory.appendingPathComponent("tokenizer_config.json")),
           let config = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            template = config["chat_template"] as? String
        }
        guard let template, !template.contains("raw %}"), !template.contains("raw -%}") else { return nil }
        return normalizeWhitespaceControl(template)
    }

    static func normalizeWhitespaceControl(_ template: String) -> String {
        // Une accolade litterale qui toucherait la balise (`{ {{-` -> `{{{-`) serait lue
        // comme un delimiteur : elle devient une expression au meme rendu.
        let rules: [(String, String)] = [
            (#"\{\s+(\{[{%#]-)"#, "{{ '{' }}$1"),
            (#"\s+(\{[{%#]-)"#, "$1"),
            (#"(-[}%#]\})\s+\}"#, "$1{{ '}' }}"),
            (#"(-[}%#]\})\s+"#, "$1"),
        ]
        var out = template
        for (pattern, replacement) in rules {
            out = out.replacingOccurrences(of: pattern, with: replacement, options: .regularExpression)
        }
        return out
    }
}

/// Bridge entre Tokenizers.Tokenizer (swift-transformers) et MLXLMCommon.Tokenizer
struct Gemma4TokenizerBridge: MLXLMCommon.Tokenizer {
    private let upstream: any Tokenizers.Tokenizer
    /// Gabarit normalise (`Gemma4TokenizerLoader.normalizedChatTemplate`), sinon celui de l'amont.
    private let chatTemplate: String?

    init(_ upstream: any Tokenizers.Tokenizer, chatTemplate: String? = nil) {
        self.upstream = upstream
        self.chatTemplate = chatTemplate
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        upstream.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        upstream.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }

    func convertTokenToId(_ token: String) -> Int? {
        upstream.convertTokenToId(token)
    }

    func convertIdToToken(_ id: Int) -> String? {
        upstream.convertIdToToken(id)
    }

    var bosToken: String? { upstream.bosToken }
    var eosToken: String? { upstream.eosToken }
    var unknownToken: String? { upstream.unknownToken }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        guard let chatTemplate else {
            return try upstream.applyChatTemplate(
                messages: messages, tools: tools, additionalContext: additionalContext)
        }
        return try upstream.applyChatTemplate(
            messages: messages, chatTemplate: .literal(chatTemplate), addGenerationPrompt: true,
            truncation: false, maxLength: nil, tools: tools, additionalContext: additionalContext)
    }
}
