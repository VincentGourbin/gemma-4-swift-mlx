import Foundation

/// Downloads a Gemma 4 model from HuggingFace into the local cache.
///
/// For UI-facing code, prefer `Gemma4DownloadManager.shared` which provides an
/// `@Observable` task with byte-level progress, cancel, and retry support.
/// This enum keeps the original convenience API for non-UI call sites.
public enum Gemma4ModelDownloader {

    /// Backwards-compatible alias for `Gemma4DownloadProgress`.
    public typealias Progress = Gemma4DownloadProgress

    // MARK: - Public API

    /// Downloads a model into the local cache.
    /// - Parameters:
    ///   - model: The model to download.
    ///   - token: Optional HuggingFace bearer token.
    ///   - force: Re-download even if already cached.
    ///   - progress: Called on an arbitrary thread whenever progress changes.
    /// - Returns: Local directory URL of the downloaded model.
    @discardableResult
    public static func download(
        _ model: Gemma4Pipeline.Model,
        token: String? = nil,
        force: Bool = false,
        progress: (@Sendable (Gemma4DownloadProgress) -> Void)? = nil
    ) async throws -> URL {
        try await download(modelId: model.rawValue, token: token, force: force, progress: progress)
    }

    /// Downloads a model by HuggingFace ID.
    @discardableResult
    public static func download(
        modelId: String,
        token: String? = nil,
        force: Bool = false,
        progress: (@Sendable (Gemma4DownloadProgress) -> Void)? = nil
    ) async throws -> URL {
        var modelDir = Gemma4ModelCache.modelsDirectory
        for part in modelId.split(separator: "/") {
            modelDir = modelDir.appendingPathComponent(String(part))
        }

        if !force && Gemma4ModelCache.isDownloaded(modelId: modelId) {
            progress?(.cached(fileCount: 1))
            return modelDir
        }

        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)

        let specs = try await fetchHFFileSpecs(modelId: modelId, token: token)
        guard !specs.isEmpty else { throw Gemma4DownloadError.noFilesFound(modelId) }

        let coordinator = DownloadCoordinator()
        await coordinator.configure(files: specs)
        if let progress {
            await coordinator.setProgressHandler(progress)
        }

        let delegate = DownloadSessionDelegate(coordinator: coordinator)
        let session = URLSession(
            configuration: .default,
            delegate: delegate,
            delegateQueue: nil
        )
        defer { session.finishTasksAndInvalidate() }

        try await runFileLoop(
            modelId: modelId,
            specs: specs,
            coordinator: coordinator,
            session: session,
            modelDir: modelDir,
            token: token,
            force: force
        )

        return modelDir
    }
}

// MARK: - Errors

public enum Gemma4DownloadError: LocalizedError {
    case noFilesFound(String)
    case apiFailed(String)
    case parseError(String)
    case httpError(String, Int)
    case networkError(String, Error)
    case cancelled(String)

    public var errorDescription: String? {
        switch self {
        case .noFilesFound(let id):        return "No files found for \(id)"
        case .apiFailed(let id):           return "HuggingFace API unreachable for \(id)"
        case .parseError(let id):          return "Could not parse API response for \(id)"
        case .httpError(let file, let c):  return "HTTP \(c) for \(file)"
        case .networkError(let id, let e): return "Network error for \(id): \(e.localizedDescription)"
        case .cancelled:                   return "Download cancelled"
        }
    }
}
