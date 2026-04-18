import Foundation
import Observation

/// Central download manager. UI code observes this to track model download state.
///
/// Usage:
/// ```swift
/// let task = Gemma4DownloadManager.shared.download(.e2b4bit)
/// // observe task.status / task.progress
/// await Gemma4DownloadManager.shared.cancel(modelId: task.modelId)
/// Gemma4DownloadManager.shared.retry(modelId: task.modelId)
/// ```
@Observable
@MainActor
public final class Gemma4DownloadManager {

    public static let shared = Gemma4DownloadManager()

    /// All known tasks keyed by model ID (active and terminal).
    public private(set) var tasks: [String: Gemma4DownloadTask] = [:]

    private init() {}

    // MARK: - Status

    /// Live status, combining active task state with the on-disk cache.
    public func status(for model: Gemma4Pipeline.Model) -> ModelStatus {
        status(forModelId: model.rawValue)
    }

    public func status(forModelId modelId: String) -> ModelStatus {
        if let task = tasks[modelId] { return task.status }
        return Gemma4ModelCache.isDownloaded(modelId: modelId) ? .downloaded : .notDownloaded
    }

    // MARK: - Download

    /// Starts downloading a model. Returns the existing task if one is already active.
    @discardableResult
    public func download(
        _ model: Gemma4Pipeline.Model,
        token: String? = nil
    ) -> Gemma4DownloadTask {
        download(modelId: model.rawValue, token: token)
    }

    /// Starts downloading a model by HuggingFace ID. Returns the existing task if active.
    /// Returns a completed task immediately when the model is already on disk.
    @discardableResult
    public func download(modelId: String, token: String? = nil) -> Gemma4DownloadTask {
        // In-flight: return the existing task without starting a second download.
        if let existing = tasks[modelId], existing.status.isDownloading {
            return existing
        }
        let coordinator = DownloadCoordinator()
        let task = Gemma4DownloadTask(modelId: modelId, coordinator: coordinator)
        tasks[modelId] = task
        // Already cached: skip the network entirely.
        if Gemma4ModelCache.isDownloaded(modelId: modelId) {
            task.markCompleted()
            return task
        }
        Task { await self.runDownload(task: task, coordinator: coordinator, token: token) }
        return task
    }

    // MARK: - Cancel

    public func cancel(modelId: String) async {
        await tasks[modelId]?.cancel()
    }

    // MARK: - Retry

    /// Discards the existing task and starts a fresh download.
    @discardableResult
    public func retry(_ model: Gemma4Pipeline.Model, token: String? = nil) -> Gemma4DownloadTask {
        retry(modelId: model.rawValue, token: token)
    }

    @discardableResult
    public func retry(modelId: String, token: String? = nil) -> Gemma4DownloadTask {
        tasks.removeValue(forKey: modelId)
        return download(modelId: modelId, token: token)
    }

    // MARK: - Delete

    public func delete(_ model: Gemma4Pipeline.Model) throws {
        try delete(modelId: model.rawValue)
    }

    public func delete(modelId: String) throws {
        var dir = Gemma4ModelCache.modelsDirectory
        for part in modelId.split(separator: "/") {
            dir = dir.appendingPathComponent(String(part))
        }
        if FileManager.default.fileExists(atPath: dir.path) {
            try FileManager.default.removeItem(at: dir)
        }
        tasks.removeValue(forKey: modelId)
    }

    // MARK: - Package-internal test support

    /// Removes a task entry without touching the file system. Used by tests for cleanup.
    func clearTask(modelId: String) {
        tasks.removeValue(forKey: modelId)
    }

    // MARK: - Private: download loop

    private func runDownload(
        task: Gemma4DownloadTask,
        coordinator: DownloadCoordinator,
        token: String?
    ) async {
        let modelId = task.modelId

        do {
            var modelDir = Gemma4ModelCache.modelsDirectory
            for part in modelId.split(separator: "/") {
                modelDir = modelDir.appendingPathComponent(String(part))
            }
            try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)

            let specs = try await fetchHFFileSpecs(modelId: modelId, token: token)
            guard !specs.isEmpty else { throw Gemma4DownloadError.noFilesFound(modelId) }

            await coordinator.configure(files: specs)
            await coordinator.setProgressHandler { [weak task] progress in
                Task { @MainActor in task?.updateProgress(progress) }
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
                token: token
            )

            if !(await coordinator.isCancelled) {
                task.markCompleted()
            }

        } catch let e as Gemma4DownloadError {
            task.markFailed(e)
        } catch {
            task.markFailed(.networkError(modelId, error))
        }
    }
}
