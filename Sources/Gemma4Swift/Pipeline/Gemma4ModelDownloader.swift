// Telechargement de modeles depuis HuggingFace — API publique du framework
// Utilise URLSession directement (pas de dependance swift-huggingface)

import Foundation

/// Telecharge un modele HuggingFace dans le cache local
public enum Gemma4ModelDownloader {

    /// Fichiers a telecharger pour un modele Gemma 4
    static let targetExtensions = [".safetensors", ".json", ".jinja", ".txt"]
    static let targetExactFiles = ["tokenizer.model"]

    /// Progression du telechargement
    public struct Progress: Sendable {
        /// Nombre de fichiers telecharges
        public let completedFiles: Int
        /// Nombre total de fichiers
        public let totalFiles: Int
        /// Ratio 0.0 → 1.0
        public var fraction: Double { Double(completedFiles) / Double(max(1, totalFiles)) }
        /// Nom du fichier en cours
        public let currentFile: String
    }

    /// Telecharge un modele dans le cache local
    /// - Parameters:
    ///   - model: le modele a telecharger (enum Gemma4Pipeline.Model)
    ///   - token: token HuggingFace optionnel (pour modeles prives)
    ///   - force: re-telecharger meme si deja present
    ///   - progress: callback de progression
    /// - Returns: URL du repertoire local du modele telecharge
    @discardableResult
    public static func download(
        _ model: Gemma4Pipeline.Model,
        token: String? = nil,
        force: Bool = false,
        progress: (@Sendable (Progress) -> Void)? = nil
    ) async throws -> URL {
        try await download(modelId: model.rawValue, token: token, force: force, progress: progress)
    }

    /// Telecharge un modele par son ID HuggingFace
    /// - Returns: URL du repertoire local du modele telecharge
    @discardableResult
    public static func download(
        modelId: String,
        token: String? = nil,
        force: Bool = false,
        progress: (@Sendable (Progress) -> Void)? = nil
    ) async throws -> URL {
        let destination = Gemma4ModelCache.modelsDirectory
        let parts = modelId.split(separator: "/")
        var modelDir = destination
        for part in parts { modelDir = modelDir.appendingPathComponent(String(part)) }

        // Skip si deja telecharge
        if !force && Gemma4ModelCache.isDownloaded(modelId: modelId) {
            progress?(Progress(completedFiles: 1, totalFiles: 1, currentFile: "done"))
            return modelDir
        }

        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)

        // Lister les fichiers du repo
        let files = try await listRepoFiles(modelId: modelId, token: token)
        let targetFiles = files.filter { name in
            targetExtensions.contains(where: { name.hasSuffix($0) }) ||
            targetExactFiles.contains(name)
        }

        guard !targetFiles.isEmpty else {
            throw Gemma4DownloadError.noFilesFound(modelId)
        }

        let totalFiles = targetFiles.count
        var completedFiles = 0

        for fileName in targetFiles {
            let fileURL = modelDir.appendingPathComponent(fileName)

            if !force && FileManager.default.fileExists(atPath: fileURL.path) {
                completedFiles += 1
                progress?(Progress(completedFiles: completedFiles, totalFiles: totalFiles, currentFile: fileName))
                continue
            }

            let downloadURL = URL(string: "https://huggingface.co/\(modelId)/resolve/main/\(fileName)")!
            try await downloadFile(from: downloadURL, to: fileURL, token: token)

            completedFiles += 1
            progress?(Progress(completedFiles: completedFiles, totalFiles: totalFiles, currentFile: fileName))
        }

        return modelDir
    }

    // MARK: - Private

    private static func listRepoFiles(modelId: String, token: String?) async throws -> [String] {
        let apiURL = URL(string: "https://huggingface.co/api/models/\(modelId)")!
        var request = URLRequest(url: apiURL)
        if let token { request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
            throw Gemma4DownloadError.apiFailed(modelId)
        }

        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let siblings = json["siblings"] as? [[String: Any]] else {
            throw Gemma4DownloadError.parseError(modelId)
        }

        return siblings.compactMap { $0["rfilename"] as? String }
    }

    private static func downloadFile(from url: URL, to destination: URL, token: String?) async throws {
        var request = URLRequest(url: url)
        if let token { request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }
        request.setValue("identity", forHTTPHeaderField: "Accept-Encoding")

        let (tempURL, response) = try await URLSession.shared.download(for: request)
        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            let status = (response as? HTTPURLResponse)?.statusCode ?? 0
            throw Gemma4DownloadError.httpError(url.lastPathComponent, status)
        }

        let dir = destination.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        if FileManager.default.fileExists(atPath: destination.path) {
            try FileManager.default.removeItem(at: destination)
        }
        try FileManager.default.moveItem(at: tempURL, to: destination)
    }
}

public enum Gemma4DownloadError: LocalizedError {
    case noFilesFound(String)
    case apiFailed(String)
    case parseError(String)
    case httpError(String, Int)

    public var errorDescription: String? {
        switch self {
        case .noFilesFound(let id): return "Aucun fichier trouve pour \(id)"
        case .apiFailed(let id): return "API HuggingFace inaccessible pour \(id)"
        case .parseError(let id): return "Impossible de parser la reponse pour \(id)"
        case .httpError(let file, let code): return "Erreur HTTP \(code) pour \(file)"
        }
    }
}
