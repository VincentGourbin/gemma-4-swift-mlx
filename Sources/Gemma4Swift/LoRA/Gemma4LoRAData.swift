// Data loader pour le fine-tuning LoRA — supporte les formats text et chat JSONL

import Foundation
import Tokenizers

// MARK: - Types de donnees

/// Message dans un format chat (role + content)
public struct ChatMessage: Codable, Sendable {
    public let role: String
    public let content: String

    public init(role: String, content: String) {
        self.role = role
        self.content = content
    }
}

/// Sample au format chat: {"messages": [...]}
struct ChatSample: Codable {
    let messages: [ChatMessage]
}

/// Sample au format text: {"text": "..."}
struct TextSample: Codable {
    let text: String
}

// MARK: - Chat Template Gemma 4

/// Applique le chat template Gemma 4 a une liste de messages.
///
/// Format:
/// ```
/// <start_of_turn>user
/// {message}<end_of_turn>
/// <start_of_turn>model
/// {message}<end_of_turn>
/// ```
public func applyGemma4ChatTemplate(messages: [ChatMessage]) -> String {
    var parts: [String] = []

    for message in messages {
        let role: String
        switch message.role {
        case "assistant", "model":
            role = "model"
        case "system":
            role = "system"
        default:
            role = "user"
        }

        parts.append("<start_of_turn>\(role)\n\(message.content)<end_of_turn>")
    }

    return parts.joined(separator: "\n")
}

// MARK: - Chargement des donnees

public enum Gemma4LoRADataError: LocalizedError {
    case fileNotFound(URL, String)
    case emptyDataset(String)

    public var errorDescription: String? {
        switch self {
        case .fileNotFound(let directory, let name):
            return "Fichier '\(name)' introuvable dans '\(directory.path())'"
        case .emptyDataset(let name):
            return "Le dataset '\(name)' est vide"
        }
    }
}

/// Charge un dataset d'entrainement depuis un repertoire.
/// Supporte les formats:
/// - `{"text": "..."}` — texte brut
/// - `{"messages": [...]}` — format chat (converti via le template Gemma 4)
///
/// - Parameters:
///   - directory: repertoire contenant les fichiers de donnees
///   - name: nom de base du fichier (train, valid, test)
/// - Returns: tableau de textes formattes, prets pour le tokenizer
public func loadGemma4TrainingData(directory: URL, name: String) throws -> [String] {
    let extensions = ["jsonl", "txt"]

    for ext in extensions {
        let url = directory.appending(component: "\(name).\(ext)")
        if FileManager.default.fileExists(atPath: url.path()) {
            let data = try loadGemma4TrainingFile(url: url)
            if data.isEmpty {
                throw Gemma4LoRADataError.emptyDataset(name)
            }
            return data
        }
    }

    throw Gemma4LoRADataError.fileNotFound(directory, name)
}

/// Charge un fichier de donnees et retourne les textes formattes
func loadGemma4TrainingFile(url: URL) throws -> [String] {
    switch url.pathExtension {
    case "jsonl":
        return try loadGemma4JSONL(url: url)
    case "txt":
        return try String(contentsOf: url, encoding: .utf8)
            .components(separatedBy: .newlines)
            .filter { !$0.isEmpty }
    default:
        fatalError("Type de fichier non supporte: \(url.pathExtension)")
    }
}

/// Charge un fichier JSONL avec detection automatique du format (chat vs text)
func loadGemma4JSONL(url: URL) throws -> [String] {
    let lines = try String(contentsOf: url, encoding: .utf8)
        .components(separatedBy: .newlines)
        .filter { $0.first == "{" }

    let decoder = JSONDecoder()

    return lines.compactMap { line -> String? in
        guard let data = line.data(using: .utf8) else { return nil }

        // Essayer le format chat d'abord
        if let chatSample = try? decoder.decode(ChatSample.self, from: data),
           !chatSample.messages.isEmpty {
            return applyGemma4ChatTemplate(messages: chatSample.messages)
        }

        // Fallback vers le format text
        if let textSample = try? decoder.decode(TextSample.self, from: data) {
            return textSample.text
        }

        return nil
    }
}
