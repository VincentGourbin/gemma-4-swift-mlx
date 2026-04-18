import Foundation

// MARK: - Speed ring buffer

/// Fixed-size ring buffer of (timestamp, delta-bytes) samples for rolling transfer rate.
struct SpeedWindow {
    private struct Sample {
        let time: Double
        let bytes: Int64
    }

    private let windowSeconds: Double
    private var samples: [Sample] = []

    init(window: Double = 3) {
        self.windowSeconds = window
    }

    mutating func record(bytes: Int64, at time: Double) {
        samples.append(Sample(time: time, bytes: bytes))
        let cutoff = time - windowSeconds
        samples.removeAll { $0.time < cutoff }
    }

    /// Bytes per second over the rolling window. Zero when insufficient data.
    func bytesPerSecond(at now: Double) -> Double {
        let cutoff = now - windowSeconds
        let relevant = samples.filter { $0.time >= cutoff }
        guard relevant.count >= 2,
              let oldest = relevant.first,
              let newest = relevant.last else { return 0 }
        let elapsed = newest.time - oldest.time
        guard elapsed > 0 else { return 0 }
        let total = relevant.dropFirst().reduce(Int64(0)) { $0 + $1.bytes }
        return Double(total) / elapsed
    }
}

// MARK: - Coordinator

/// Actor that owns all mutable download state for one model download.
///
/// `DownloadSessionDelegate` forwards `URLSessionDownloadDelegate` callbacks here.
/// `Gemma4DownloadManager` drives the sequential file loop by calling `startFileDownload`,
/// which suspends until the delegate resumes the stored continuation.
actor DownloadCoordinator {

    // MARK: - Types

    struct FileSpec {
        let name: String
        /// Zero when the server did not advertise a size.
        let expectedBytes: Int64
    }

    // MARK: - File manifest

    private(set) var files: [FileSpec] = []

    // MARK: - Byte accounting

    private var completedFileBytes: Int64 = 0
    private var currentFileWritten: Int64 = 0
    private var currentFileTotalBytes: Int64 = 0   // from Content-Length; 0 = unknown
    private var completedFileCount: Int = 0
    private var currentFileName: String = ""
    private var speedWindow = SpeedWindow()

    // MARK: - Cancellation

    private(set) var isCancelled = false
    private var activeTasks: [URLSessionDownloadTask] = []

    // MARK: - Per-file continuations (one in-flight at a time, sequential download)

    private var pendingContinuation: CheckedContinuation<URL, Error>?
    private var pendingTaskId: Int?

    // MARK: - Progress callback

    var onProgress: (@Sendable (Gemma4DownloadProgress) -> Void)?

    // MARK: - Setup

    func configure(files: [FileSpec]) {
        self.files = files
        completedFileBytes = 0
        currentFileWritten = 0
        currentFileTotalBytes = 0
        completedFileCount = 0
        currentFileName = files.first?.name ?? ""
        speedWindow = SpeedWindow()
        isCancelled = false
        activeTasks = []
    }

    func setProgressHandler(_ handler: @escaping @Sendable (Gemma4DownloadProgress) -> Void) {
        onProgress = handler
    }

    /// Account for a file that was already present on disk (no network fetch needed).
    func skipFile(index: Int) {
        completedFileBytes += files[index].expectedBytes
        completedFileCount += 1
        emitProgress()
    }

    // MARK: - Download control

    /// Registers the URLSessionDownloadTask and suspends until it completes.
    /// Returns the temporary file URL that the delegate moved off the URLSession temp location.
    func startFileDownload(task: URLSessionDownloadTask, index: Int) async throws -> URL {
        currentFileName = files[index].name
        currentFileTotalBytes = 0   // reset; populated from Content-Length on first didWriteBytes
        activeTasks.append(task)
        return try await withCheckedThrowingContinuation { continuation in
            pendingContinuation = continuation
            pendingTaskId = task.taskIdentifier
            task.resume()
        }
    }

    /// Cancels any in-flight task and poisons the pending continuation.
    func cancelAll() {
        isCancelled = true
        activeTasks.forEach { $0.cancel() }
        activeTasks = []
        if let cont = pendingContinuation {
            cont.resume(throwing: Gemma4DownloadError.cancelled("download cancelled"))
            pendingContinuation = nil
            pendingTaskId = nil
        }
    }

    // MARK: - Delegate callbacks

    func didWriteBytes(taskId: Int, written: Int64, totalWritten: Int64, expectedTotal: Int64) {
        guard taskId == pendingTaskId else { return }
        currentFileWritten = totalWritten
        // Capture Content-Length on first callback (URLSession passes -1 when unknown).
        if expectedTotal > 0 && currentFileTotalBytes == 0 {
            currentFileTotalBytes = expectedTotal
        }
        let now = CFAbsoluteTimeGetCurrent()
        speedWindow.record(bytes: written, at: now)
        emitProgress()
    }

    func didFinishFile(taskId: Int, location: URL) {
        guard taskId == pendingTaskId else { return }
        activeTasks.removeAll { $0.taskIdentifier == taskId }
        // Use Content-Length-derived size so completedFileBytes stays accurate.
        let fileSize = currentFileTotalBytes > 0 ? currentFileTotalBytes : currentFileWritten
        completedFileBytes += fileSize
        currentFileWritten = 0
        currentFileTotalBytes = 0
        completedFileCount += 1
        let cont = pendingContinuation
        pendingContinuation = nil
        pendingTaskId = nil
        cont?.resume(returning: location)
    }

    func didFailFile(taskId: Int, error: Error) {
        guard taskId == pendingTaskId else { return }
        activeTasks.removeAll { $0.taskIdentifier == taskId }
        let cont = pendingContinuation
        pendingContinuation = nil
        pendingTaskId = nil
        cont?.resume(throwing: error)
    }

    // MARK: - Private

    private func emitProgress() {
        let now = CFAbsoluteTimeGetCurrent()
        let speed = speedWindow.bytesPerSecond(at: now)
        let completedBytes = completedFileBytes + currentFileWritten
        // Build best-effort total from Content-Length (current file) +
        // accumulated completed bytes + API sizes for not-yet-started files.
        // HF API rarely returns sizes, so currentFileTotalBytes is the only
        // reliable source; for remaining files we use whatever the API gave us.
        let remainingFilesBytes: Int64 = files
            .dropFirst(completedFileCount + (currentFileWritten > 0 ? 1 : 0))
            .reduce(Int64(0)) { $0 + $1.expectedBytes }
        let totalBytes: Int64
        if currentFileTotalBytes > 0 {
            totalBytes = completedFileBytes + currentFileTotalBytes + remainingFilesBytes
        } else {
            // Fall back to API-supplied sizes (may remain 0 for all files).
            totalBytes = files.reduce(Int64(0)) { $0 + $1.expectedBytes }
        }
        let remaining: Double? = speed > 0 && totalBytes > completedBytes
            ? Double(totalBytes - completedBytes) / speed
            : nil

        let progress = Gemma4DownloadProgress(
            completedBytes: completedBytes,
            totalBytes: totalBytes,
            completedFiles: completedFileCount,
            totalFiles: files.count,
            currentFile: currentFileName,
            bytesPerSecond: speed,
            estimatedSecondsRemaining: remaining
        )
        onProgress?(progress)
    }
}

// MARK: - URLSession delegate bridge

/// Bridges `URLSessionDownloadDelegate` callbacks to the `DownloadCoordinator` actor.
///
/// Each delegate method is `nonisolated` (required by the protocol). We dispatch to
/// the actor via an unstructured `Task` — the standard Swift 6 pattern for this boundary.
final class DownloadSessionDelegate: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {

    private let coordinator: DownloadCoordinator

    init(coordinator: DownloadCoordinator) {
        self.coordinator = coordinator
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        let id = downloadTask.taskIdentifier
        let expected = totalBytesExpectedToWrite  // -1 when unknown; >0 = Content-Length
        Task {
            await coordinator.didWriteBytes(
                taskId: id,
                written: bytesWritten,
                totalWritten: totalBytesWritten,
                expectedTotal: expected
            )
        }
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        // Reject non-2xx responses before treating the body as a valid file.
        if let http = downloadTask.response as? HTTPURLResponse,
           !(200...299).contains(http.statusCode) {
            let id = downloadTask.taskIdentifier
            let filename = downloadTask.originalRequest?.url?.lastPathComponent ?? ""
            Task {
                await coordinator.didFailFile(
                    taskId: id,
                    error: Gemma4DownloadError.httpError(filename, http.statusCode)
                )
            }
            return
        }

        // The temp file is deleted when this method returns — move it first.
        let preserved: URL
        do {
            let tmp = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString)
            try FileManager.default.moveItem(at: location, to: tmp)
            preserved = tmp
        } catch {
            let id = downloadTask.taskIdentifier
            Task { await coordinator.didFailFile(taskId: id, error: error) }
            return
        }
        let id = downloadTask.taskIdentifier
        Task { await coordinator.didFinishFile(taskId: id, location: preserved) }
    }

    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        didCompleteWithError error: Error?
    ) {
        guard let error else { return }
        // This fires for all task failures (network drop, timeout, 404, etc.).
        // If didFinishDownloadingTo already called didFailFile (e.g. temp-file move
        // threw), the coordinator's `guard taskId == pendingTaskId` will discard this
        // second call because pendingTaskId was already cleared — no double-resume.
        let id = task.taskIdentifier
        Task { await coordinator.didFailFile(taskId: id, error: error) }
    }
}

// MARK: - Shared download helpers

private let hfTargetExtensions = [".safetensors", ".json", ".jinja", ".txt"]
private let hfTargetExactFiles = ["tokenizer.model"]

/// Fetches the file manifest from the HuggingFace API and returns specs for files we need.
func fetchHFFileSpecs(modelId: String, token: String?) async throws -> [DownloadCoordinator.FileSpec] {
    let url = URL(string: "https://huggingface.co/api/models/\(modelId)")!
    var request = URLRequest(url: url)
    if let token { request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }

    let (data, response) = try await URLSession.shared.data(for: request)
    guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
        throw Gemma4DownloadError.apiFailed(modelId)
    }
    guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
          let siblings = json["siblings"] as? [[String: Any]] else {
        throw Gemma4DownloadError.parseError(modelId)
    }

    return siblings.compactMap { sibling -> DownloadCoordinator.FileSpec? in
        guard let name = sibling["rfilename"] as? String else { return nil }
        let included = hfTargetExtensions.contains(where: { name.hasSuffix($0) })
                    || hfTargetExactFiles.contains(name)
        guard included else { return nil }
        let size = (sibling["size"] as? Int).map { Int64($0) } ?? 0
        return DownloadCoordinator.FileSpec(name: name, expectedBytes: size)
    }
}

/// Drives the sequential per-file download loop for a model.
///
/// Shared by `Gemma4ModelDownloader` (one-shot) and `Gemma4DownloadManager` (observable).
/// Already-present files are skipped unless `force` is true.
func runFileLoop(
    modelId: String,
    specs: [DownloadCoordinator.FileSpec],
    coordinator: DownloadCoordinator,
    session: URLSession,
    modelDir: URL,
    token: String?,
    force: Bool = false
) async throws {
    for (index, spec) in specs.enumerated() {
        guard !(await coordinator.isCancelled) else { break }

        let destination = modelDir.appendingPathComponent(spec.name)
        if !force && FileManager.default.fileExists(atPath: destination.path) {
            await coordinator.skipFile(index: index)
            continue
        }

        let url = URL(string: "https://huggingface.co/\(modelId)/resolve/main/\(spec.name)")!
        var request = URLRequest(url: url)
        if let token { request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }
        request.setValue("identity", forHTTPHeaderField: "Accept-Encoding")

        let downloadTask = session.downloadTask(with: request)
        let tempURL = try await coordinator.startFileDownload(task: downloadTask, index: index)

        let dir = destination.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        if FileManager.default.fileExists(atPath: destination.path) {
            try FileManager.default.removeItem(at: destination)
        }
        try FileManager.default.moveItem(at: tempURL, to: destination)
    }
}
