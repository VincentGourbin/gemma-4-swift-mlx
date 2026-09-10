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
    ///
    /// Parcourt via les APIs `atPath:` (et non `URL`-based) car un fichier de poids
    /// peut avoir ete remplace par un symlink absolu vers un disque externe — le
    /// symlink lui-meme doit etre resolu avant de lire sa taille, sinon on ne
    /// rapporte que la taille du lien (quelques octets) plutot que celle de la
    /// cible.
    public static func diskSize(for model: Gemma4Pipeline.Model) -> Int64? {
        guard let path = localPath(for: model) else { return nil }
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(atPath: path.path) else {
            return nil
        }
        var total: Int64 = 0
        for case let relativePath as String in enumerator {
            let entryPath = path.appendingPathComponent(relativePath).path
            guard var attrs = try? fm.attributesOfItem(atPath: entryPath) else { continue }
            if attrs[.type] as? FileAttributeType == .typeSymbolicLink {
                let resolved = URL(fileURLWithPath: entryPath).resolvingSymlinksInPath().path
                guard let resolvedAttrs = try? fm.attributesOfItem(atPath: resolved) else {
                    // Cible manquante (disque externe non monte) : pas une erreur.
                    continue
                }
                attrs = resolvedAttrs
            }
            if let size = attrs[.size] as? Int64 {
                total += size
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
        // homeDirectoryForCurrentUser n'est pas dispo sur iOS — utiliser NSHomeDirectory()
        let homeDir = URL(fileURLWithPath: NSHomeDirectory())
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
