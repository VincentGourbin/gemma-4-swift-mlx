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

/// Sample multimodal: {"messages": [...], "audio": "path.wav", "image": "path.jpg"}
struct MultimodalChatSample: Codable {
    let messages: [ChatMessage]
    let audio: String?
    let image: String?
}

/// Sample multimodal pre-formatte (texte sans placeholders + chemins media)
/// Les tokens media sont injectes au niveau token ID apres tokenisation.
public struct MultimodalTrainingSample: Sendable {
    public let text: String
    public let audioPath: String?
    public let imagePath: String?
    public let hasAudio: Bool
    public let hasImage: Bool

    public init(text: String, audioPath: String? = nil, imagePath: String? = nil,
                hasAudio: Bool = false, hasImage: Bool = false) {
        self.text = text
        self.audioPath = audioPath
        self.imagePath = imagePath
        self.hasAudio = hasAudio
        self.hasImage = hasImage
    }
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
/// - `{"messages": [...]}` — format chat (converti via le chat template du tokenizer)
///
/// - Parameters:
///   - directory: repertoire contenant les fichiers de donnees
///   - name: nom de base du fichier (train, valid, test)
///   - tokenizer: le tokenizer du modele (requis pour le format chat, garantit la
///     coherence entre training et inference via applyChatTemplate)
/// - Returns: tableau de textes formattes, prets pour le tokenizer
public func loadGemma4TrainingData(directory: URL, name: String, chatFormatter: (([[String: String]]) throws -> String)? = nil) throws -> [String] {
    let extensions = ["jsonl", "txt"]

    for ext in extensions {
        let url = directory.appending(component: "\(name).\(ext)")
        if FileManager.default.fileExists(atPath: url.path()) {
            let data = try loadGemma4TrainingFile(url: url, chatFormatter: chatFormatter)
            if data.isEmpty {
                throw Gemma4LoRADataError.emptyDataset(name)
            }
            return data
        }
    }

    throw Gemma4LoRADataError.fileNotFound(directory, name)
}

/// Charge un fichier de donnees et retourne les textes formattes
func loadGemma4TrainingFile(url: URL, chatFormatter: (([[String: String]]) throws -> String)? = nil) throws -> [String] {
    switch url.pathExtension {
    case "jsonl":
        return try loadGemma4JSONL(url: url, chatFormatter: chatFormatter)
    case "txt":
        return try String(contentsOf: url, encoding: .utf8)
            .components(separatedBy: .newlines)
            .filter { !$0.isEmpty }
    default:
        fatalError("Type de fichier non supporte: \(url.pathExtension)")
    }
}

// MARK: - Chargement multimodal

/// Charge un dataset multimodal depuis un fichier JSONL.
/// Chaque ligne peut contenir des champs optionnels "audio" et "image" avec des chemins relatifs.
/// Les placeholders media sont inseres dans le contenu user avant le texte.
public func loadGemma4MultimodalJSONL(
    url: URL,
    dataDirectory: URL,
    chatFormatter: (([[String: String]]) throws -> String)? = nil
) throws -> [MultimodalTrainingSample] {
    let lines = try String(contentsOf: url, encoding: .utf8)
        .components(separatedBy: .newlines)
        .filter { $0.first == "{" }

    let decoder = JSONDecoder()

    return try lines.compactMap { line -> MultimodalTrainingSample? in
        guard let data = line.data(using: .utf8) else { return nil }
        let sample = try decoder.decode(MultimodalChatSample.self, from: data)
        guard !sample.messages.isEmpty else { return nil }

        // NE PAS inserer les placeholders media dans le texte ici.
        // applyChatTemplate escape les tokens speciaux (<|audio|> etc.)
        // L'injection des tokens media se fait APRES tokenisation, au niveau token ID,
        // dans le preprocessing CLI.

        // Formatter le texte (messages originaux sans placeholders)
        let text: String
        if let chatFormatter {
            let msgDicts = sample.messages.map { ["role": $0.role, "content": $0.content] }
            text = try chatFormatter(msgDicts)
        } else {
            text = applyGemma4ChatTemplate(messages: sample.messages)
        }

        // Resoudre les chemins media
        let audioPath = sample.audio.map { dataDirectory.appending(component: $0).path() }
        let imagePath = sample.image.map { dataDirectory.appending(component: $0).path() }

        return MultimodalTrainingSample(text: text, audioPath: audioPath, imagePath: imagePath,
                                        hasAudio: sample.audio != nil, hasImage: sample.image != nil)
    }
}

/// Charge un fichier JSONL avec detection automatique du format (chat vs text)
///
/// Pour le format chat, si un tokenizer est fourni, utilise `applyChatTemplate` pour
/// garantir la coherence avec l'inference. Sinon, utilise le template Gemma 4 interne.
func loadGemma4JSONL(url: URL, chatFormatter: (([[String: String]]) throws -> String)? = nil) throws -> [String] {
    let lines = try String(contentsOf: url, encoding: .utf8)
        .components(separatedBy: .newlines)
        .filter { $0.first == "{" }

    let decoder = JSONDecoder()

    return try lines.compactMap { line -> String? in
        guard let data = line.data(using: .utf8) else { return nil }

        // Essayer le format chat d'abord
        if let chatSample = try? decoder.decode(ChatSample.self, from: data),
           !chatSample.messages.isEmpty {
            // Si on a un tokenizer, faire un roundtrip applyChatTemplate → decode
            // pour que le texte d'entrainement soit tokenise exactement comme a l'inference
            if let chatFormatter {
                let messages = chatSample.messages.map { ["role": $0.role, "content": $0.content] }
                return try chatFormatter(messages)
            }
            return applyGemma4ChatTemplate(messages: chatSample.messages)
        }

        // Fallback vers le format text
        if let textSample = try? decoder.decode(TextSample.self, from: data) {
            return textSample.text
        }

        return nil
    }
}
