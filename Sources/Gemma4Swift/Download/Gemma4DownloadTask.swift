import Foundation
import Observation

/// An observable handle representing the download of a single model.
///
/// Created and vended by `Gemma4DownloadManager`. UI code observes `status` and
/// `progress` directly. To retry after failure, call `Gemma4DownloadManager.shared.retry(modelId:)`.
///
/// `@MainActor`: all mutations run on the main actor, which satisfies Swift 6 strict
/// concurrency and makes the type safe to observe directly from SwiftUI.
@Observable
@MainActor
public final class Gemma4DownloadTask {

    // MARK: - Public state

    public let modelId: String
    public private(set) var progress: Gemma4DownloadProgress
    public private(set) var status: ModelStatus

    // MARK: - Internal

    private let coordinator: DownloadCoordinator

    // MARK: - Init

    init(modelId: String, coordinator: DownloadCoordinator) {
        self.modelId = modelId
        self.coordinator = coordinator
        let initial = Gemma4DownloadProgress(
            completedBytes: 0,
            totalBytes: 0,
            completedFiles: 0,
            totalFiles: 0,
            currentFile: "",
            bytesPerSecond: 0,
            estimatedSecondsRemaining: nil
        )
        self.progress = initial
        self.status = .downloading(initial)
    }

    // MARK: - Control

    /// Cancels all in-flight URLSession tasks for this model.
    /// Status transitions to `.failed` with a cancellation error.
    /// No-op if the download has already completed or failed.
    public func cancel() async {
        guard status.isDownloading else { return }
        await coordinator.cancelAll()
        status = .failed(Gemma4DownloadError.cancelled(modelId))
    }

    // MARK: - Internal updates (called by DownloadManager)

    func updateProgress(_ p: Gemma4DownloadProgress) {
        self.progress = p
        self.status = .downloading(p)
    }

    func markCompleted() {
        self.status = .downloaded
    }

    func markFailed(_ error: Gemma4DownloadError) {
        self.status = .failed(error)
    }
}
