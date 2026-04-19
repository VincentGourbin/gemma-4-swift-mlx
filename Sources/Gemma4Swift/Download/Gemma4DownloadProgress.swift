import Foundation

/// Byte-level download progress for a model download.
public struct Gemma4DownloadProgress: Sendable {

    // MARK: - Raw state

    public let completedBytes: Int64
    /// Zero when the server did not advertise a Content-Length.
    public let totalBytes: Int64
    public let completedFiles: Int
    public let totalFiles: Int
    /// File currently being downloaded.
    public let currentFile: String
    /// Instantaneous transfer rate in bytes/s (3-second rolling window).
    public let bytesPerSecond: Double
    public let estimatedSecondsRemaining: Double?

    // MARK: - Derived fractions

    /// Byte-level fraction 0…1. Falls back to file fraction when totalBytes is unknown.
    public var bytesFraction: Double {
        guard totalBytes > 0 else { return filesFraction }
        return min(1, Double(completedBytes) / Double(totalBytes))
    }

    /// Backwards-compatible alias for `bytesFraction`.
    public var fraction: Double { bytesFraction }

    /// File-count fraction 0…1.
    public var filesFraction: Double {
        guard totalFiles > 0 else { return 0 }
        return min(1, Double(completedFiles) / Double(totalFiles))
    }

    // MARK: - Formatted strings

    // ByteCountFormatter is thread-safe (internal NSLock); nonisolated(unsafe) opts
    // out of the Sendable check while keeping a single reusable instance.
    nonisolated(unsafe) private static let byteFormatter: ByteCountFormatter = {
        let f = ByteCountFormatter()
        f.allowedUnits = [.useAll]
        f.countStyle = .file
        return f
    }()

    /// Human-readable transfer rate, e.g. "4.2 MB/s".
    public var formattedSpeed: String {
        guard bytesPerSecond > 0 else { return "-" }
        let formatted = Self.byteFormatter.string(fromByteCount: Int64(bytesPerSecond))
        return "\(formatted)/s"
    }

    /// Human-readable ETA, e.g. "~2 min remaining". Nil when unknown.
    public var formattedETA: String? {
        guard let seconds = estimatedSecondsRemaining, seconds > 0 else { return nil }
        if seconds < 60 {
            return "~\(Int(seconds))s remaining"
        }
        let minutes = Int((seconds / 60).rounded(.up))
        return "~\(minutes) min remaining"
    }

    /// Human-readable progress, e.g. "142 MB of 3.6 GB".
    public var formattedProgress: String {
        let done = Self.byteFormatter.string(fromByteCount: completedBytes)
        guard totalBytes > 0 else { return done }
        let total = Self.byteFormatter.string(fromByteCount: totalBytes)
        return "\(done) of \(total)"
    }

    // MARK: - Init

    public init(
        completedBytes: Int64,
        totalBytes: Int64,
        completedFiles: Int,
        totalFiles: Int,
        currentFile: String,
        bytesPerSecond: Double,
        estimatedSecondsRemaining: Double?
    ) {
        self.completedBytes = completedBytes
        self.totalBytes = totalBytes
        self.completedFiles = completedFiles
        self.totalFiles = totalFiles
        self.currentFile = currentFile
        self.bytesPerSecond = bytesPerSecond
        self.estimatedSecondsRemaining = estimatedSecondsRemaining
    }

    // MARK: - Convenience

    /// Sentinel used to report "already cached, nothing to download".
    static func cached(fileCount: Int) -> Gemma4DownloadProgress {
        Gemma4DownloadProgress(
            completedBytes: 0,
            totalBytes: 0,
            completedFiles: fileCount,
            totalFiles: fileCount,
            currentFile: "",
            bytesPerSecond: 0,
            estimatedSecondsRemaining: nil
        )
    }
}
