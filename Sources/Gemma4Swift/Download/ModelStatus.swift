import Foundation

/// Observable lifecycle status for a model.
///
/// Covers the full arc from nothing-on-disk through download, load, and inference.
/// UI code observes this via `Gemma4DownloadManager.shared.status(for:)`.
public enum ModelStatus: Sendable {
    /// No files present on disk and no active download.
    case notDownloaded
    /// Download is in progress.
    case downloading(Gemma4DownloadProgress)
    /// All files are on disk; the model has not been loaded into memory.
    case downloaded
    /// Model weights are being read from disk into GPU/unified memory.
    /// Set by `Gemma4Pipeline` during `load()` — not by the download system.
    case loading
    /// Model is loaded and ready for inference.
    /// Set by `Gemma4Pipeline` once `load()` completes — not by the download system.
    case ready
    /// Download or load failed. The associated error describes the cause.
    case failed(Gemma4DownloadError)
}

// MARK: - Convenience

public extension ModelStatus {

    /// True when no further state transitions are expected without user action.
    var isTerminal: Bool {
        switch self {
        case .downloaded, .ready, .failed: return true
        default: return false
        }
    }

    /// True while bytes are actively being transferred.
    var isDownloading: Bool {
        if case .downloading = self { return true }
        return false
    }

    /// True once weights are resident in memory.
    var isLoaded: Bool {
        if case .ready = self { return true }
        return false
    }

    /// Progress associated with the current state, if any.
    var progress: Gemma4DownloadProgress? {
        if case .downloading(let p) = self { return p }
        return nil
    }

    /// Short human-readable label suitable for UI display.
    var label: String {
        switch self {
        case .notDownloaded:        return "Not downloaded"
        case .downloading(let p):   return "Downloading \(Int(p.bytesFraction * 100))%"
        case .downloaded:           return "Downloaded"
        case .loading:              return "Loading"
        case .ready:                return "Ready"
        case .failed:               return "Failed"
        }
    }
}
