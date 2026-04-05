// Gestion du cache local des modeles Gemma 4

import Foundation

/// Gestion du cache local des modeles Gemma 4
/// Par defaut: ~/Library/Caches/models/
public enum Gemma4ModelCache {

    /// Chemin personnalise pour le stockage des modeles (overridable)
    nonisolated(unsafe) public static var customModelsDirectory: URL?

    /// Repertoire de stockage des modeles
    public static var modelsDirectory: URL {
        if let custom = customModelsDirectory {
            return custom
        }
        let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        return cacheDir.appendingPathComponent("models", isDirectory: true)
    }

    /// RAM systeme en Go
    public static var systemRAMGB: Int {
        Int(ProcessInfo.processInfo.physicalMemory / (1024 * 1024 * 1024))
    }

    /// Verifie si un modele est telecharge localement (dans le cache personnalise ou le cache HF)
    public static func isDownloaded(_ model: Gemma4Pipeline.Model) -> Bool {
        possiblePaths(for: model.rawValue).contains { path in
            hasModelFiles(at: path)
        }
    }

    /// Verifie si un modele (par ID HuggingFace) est telecharge localement
    public static func isDownloaded(modelId: String) -> Bool {
        possiblePaths(for: modelId).contains { path in
            hasModelFiles(at: path)
        }
    }

    /// Chemin local du modele s'il existe, nil sinon
    public static func localPath(for model: Gemma4Pipeline.Model) -> URL? {
        possiblePaths(for: model.rawValue).first { hasModelFiles(at: $0) }
    }

    /// Taille sur disque d'un modele telecharge (en octets), nil si non telecharge
    public static func diskSize(for model: Gemma4Pipeline.Model) -> Int64? {
        guard let path = localPath(for: model) else { return nil }
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(at: path, includingPropertiesForKeys: [.fileSizeKey]) else {
            return nil
        }
        var total: Int64 = 0
        for case let fileURL as URL in enumerator {
            if let values = try? fileURL.resourceValues(forKeys: [.fileSizeKey]),
               let size = values.fileSize {
                total += Int64(size)
            }
        }
        return total > 0 ? total : nil
    }

    // MARK: - Private

    private static func hasModelFiles(at path: URL) -> Bool {
        let fm = FileManager.default
        let configExists = fm.fileExists(atPath: path.appendingPathComponent("config.json").path)
        guard configExists else { return false }
        let contents = (try? fm.contentsOfDirectory(atPath: path.path)) ?? []
        return contents.contains { $0.hasSuffix(".safetensors") }
    }

    private static func possiblePaths(for modelId: String) -> [URL] {
        let parts = modelId.split(separator: "/")
        var paths: [URL] = []

        // Cache personnalise: ~/Library/Caches/models/{org}/{model}
        var customPath = modelsDirectory
        for part in parts { customPath = customPath.appendingPathComponent(String(part)) }
        paths.append(customPath)

        // Cache HuggingFace par defaut: ~/.cache/huggingface/hub/models--{org}--{model}/snapshots/*
        let homeDir = FileManager.default.homeDirectoryForCurrentUser
        let modelFolder = "models--\(modelId.replacingOccurrences(of: "/", with: "--"))"
        let hfSnapshotsDir = homeDir
            .appendingPathComponent(".cache/huggingface/hub")
            .appendingPathComponent(modelFolder)
            .appendingPathComponent("snapshots")

        // Prendre le dernier snapshot (le plus recent)
        if let snapshots = try? FileManager.default.contentsOfDirectory(
            at: hfSnapshotsDir,
            includingPropertiesForKeys: nil
        ) {
            if let latest = snapshots.last {
                paths.append(latest)
            }
        }

        return paths
    }
}
