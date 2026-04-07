// Telechargement de modeles depuis HuggingFace sans dépendance swift-huggingface
// Utilise URLSession directement pour telecharger les fichiers safetensors + config

import Foundation
import MLXLMCommon

/// Telecharge un modele HuggingFace dans le cache local
/// Remplace HubClient.downloadSnapshot() sans dépendance swift-huggingface
enum LocalModelDownloader {

    /// Fichiers a telecharger pour un modele LLM
    static let requiredFiles = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]

    static let optionalFiles = [
        "generation_config.json",
        "processor_config.json",
        "chat_template.jinja",
    ]

    /// Telecharge un modele dans le repertoire de destination
    /// - Parameters:
    ///   - modelId: ID HuggingFace (ex: "mlx-community/gemma-4-e2b-it-4bit")
    ///   - destination: repertoire local de destination
    ///   - token: token HuggingFace optionnel
    ///   - progress: callback de progression (0.0 → 1.0)
    static func download(
        modelId: String,
        to destination: URL,
        token: String? = nil,
        progress: @escaping (Double) -> Void = { _ in }
    ) async throws {
        // Creer le repertoire
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)

        // 1. Lister les fichiers du repo via l'API HuggingFace
        let files = try await listRepoFiles(modelId: modelId, token: token)

        // Filtrer les fichiers pertinents
        let targetFiles = files.filter { name in
            name.hasSuffix(".safetensors") ||
            name.hasSuffix(".json") ||
            name.hasSuffix(".jinja") ||
            name.hasSuffix(".txt") ||
            name == "tokenizer.model"
        }

        guard !targetFiles.isEmpty else {
            throw DownloadError.noFilesFound(modelId)
        }

        // 2. Calculer la taille totale (pour la progression)
        let totalFiles = targetFiles.count
        var completedFiles = 0

        // 3. Telecharger chaque fichier
        for fileName in targetFiles {
            let fileURL = destination.appendingPathComponent(fileName)

            // Skip si deja present
            if FileManager.default.fileExists(atPath: fileURL.path) {
                completedFiles += 1
                progress(Double(completedFiles) / Double(totalFiles))
                continue
            }

            let downloadURL = URL(string: "https://huggingface.co/\(modelId)/resolve/main/\(fileName)")!
            try await downloadFile(from: downloadURL, to: fileURL, token: token)

            completedFiles += 1
            progress(Double(completedFiles) / Double(totalFiles))
        }
    }

    /// Liste les fichiers d'un repo HuggingFace via l'API
    private static func listRepoFiles(modelId: String, token: String?) async throws -> [String] {
        let apiURL = URL(string: "https://huggingface.co/api/models/\(modelId)")!
        var request = URLRequest(url: apiURL)
        if let token = token {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
            throw DownloadError.apiFailed(modelId)
        }

        // Parser la reponse JSON pour extraire les noms de fichiers
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let siblings = json["siblings"] as? [[String: Any]] else {
            throw DownloadError.parseError(modelId)
        }

        return siblings.compactMap { $0["rfilename"] as? String }
    }

    /// Telecharge un fichier unique avec support du token
    private static func downloadFile(from url: URL, to destination: URL, token: String?) async throws {
        var request = URLRequest(url: url)
        if let token = token {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        // Suivre les redirections (HF renvoie des 302)
        request.setValue("identity", forHTTPHeaderField: "Accept-Encoding")

        let (tempURL, response) = try await URLSession.shared.download(for: request)
        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            let status = (response as? HTTPURLResponse)?.statusCode ?? 0
            throw DownloadError.httpError(url.lastPathComponent, status)
        }

        // Deplacer le fichier temporaire vers la destination
        let dir = destination.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        if FileManager.default.fileExists(atPath: destination.path) {
            try FileManager.default.removeItem(at: destination)
        }
        try FileManager.default.moveItem(at: tempURL, to: destination)
    }

    enum DownloadError: LocalizedError {
        case noFilesFound(String)
        case apiFailed(String)
        case parseError(String)
        case httpError(String, Int)

        var errorDescription: String? {
            switch self {
            case .noFilesFound(let id): return "Aucun fichier trouve pour \(id)"
            case .apiFailed(let id): return "API HuggingFace inaccessible pour \(id)"
            case .parseError(let id): return "Impossible de parser la reponse pour \(id)"
            case .httpError(let file, let code): return "Erreur HTTP \(code) pour \(file)"
            }
        }
    }
}
