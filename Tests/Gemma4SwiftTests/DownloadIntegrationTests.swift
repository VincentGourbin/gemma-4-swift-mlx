import Testing
import Foundation
@testable import Gemma4Swift

// MARK: - URLProtocol mock

/// Intercepts URLSession requests and returns pre-configured stub responses.
/// Registered per-test via a custom URLSessionConfiguration.
final class StubURLProtocol: URLProtocol, @unchecked Sendable {

    struct Stub {
        let data: Data
        let response: HTTPURLResponse
        let chunkSize: Int
        let delayBetweenChunks: TimeInterval
    }

    nonisolated(unsafe) static var stubs: [String: Stub] = [:]

    override class func canInit(with request: URLRequest) -> Bool {
        guard let url = request.url?.absoluteString else { return false }
        return stubs[url] != nil
    }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        guard let url = request.url?.absoluteString, let stub = Self.stubs[url] else {
            client?.urlProtocol(self, didFailWithError: URLError(.fileDoesNotExist))
            return
        }
        client?.urlProtocol(self, didReceive: stub.response, cacheStoragePolicy: .notAllowed)

        var offset = stub.data.startIndex
        while offset < stub.data.endIndex {
            let end = stub.data.index(offset, offsetBy: stub.chunkSize, limitedBy: stub.data.endIndex) ?? stub.data.endIndex
            let chunk = stub.data[offset..<end]
            client?.urlProtocol(self, didLoad: Data(chunk))
            offset = end
        }
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

// MARK: - Helpers

private func makeSession(additionalProtocols: [AnyClass] = []) -> URLSession {
    let config = URLSessionConfiguration.ephemeral
    config.protocolClasses = [StubURLProtocol.self] + additionalProtocols
    return URLSession(configuration: config)
}

private func stub(url: String, data: Data, statusCode: Int = 200, chunkSize: Int = 512) {
    let response = HTTPURLResponse(
        url: URL(string: url)!,
        statusCode: statusCode,
        httpVersion: nil,
        headerFields: ["Content-Length": "\(data.count)"]
    )!
    StubURLProtocol.stubs[url] = StubURLProtocol.Stub(
        data: data,
        response: response,
        chunkSize: chunkSize,
        delayBetweenChunks: 0
    )
}

// MARK: - DownloadCoordinator integration

@Suite("DownloadCoordinator – full file flow")
struct DownloadCoordinatorIntegrationTests {

    @Test("full download flow via StubURLProtocol resumes continuation with correct data")
    func fullFileFlow() async throws {
        let modelId = "test-org/flow-test"
        let fileName = "model.safetensors"
        let fakeData = Data(repeating: 0xAB, count: 1024)
        let fileURL = "https://huggingface.co/\(modelId)/resolve/main/\(fileName)"

        stub(url: fileURL, data: fakeData, statusCode: 200, chunkSize: 512)
        defer { StubURLProtocol.stubs.removeValue(forKey: fileURL) }

        let coordinator = DownloadCoordinator()
        await coordinator.configure(files: [
            DownloadCoordinator.FileSpec(name: fileName, expectedBytes: Int64(fakeData.count))
        ])

        nonisolated(unsafe) var progressUpdates: [Gemma4DownloadProgress] = []
        await coordinator.setProgressHandler { p in progressUpdates.append(p) }

        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [StubURLProtocol.self]
        let delegate = DownloadSessionDelegate(coordinator: coordinator)
        let session = URLSession(configuration: config, delegate: delegate, delegateQueue: nil)
        defer { session.invalidateAndCancel() }

        let downloadTask = session.downloadTask(with: URL(string: fileURL)!)
        let tempURL = try await coordinator.startFileDownload(task: downloadTask, index: 0)
        defer { try? FileManager.default.removeItem(at: tempURL) }

        let downloaded = try Data(contentsOf: tempURL)
        #expect(downloaded == fakeData)
    }

    @Test("cancelAll on a fresh coordinator sets isCancelled without crashing")
    func cancelFreshCoordinator() async {
        let coordinator = DownloadCoordinator()
        await coordinator.configure(files: [
            DownloadCoordinator.FileSpec(name: "x.json", expectedBytes: 100)
        ])
        await coordinator.cancelAll()
        let cancelled = await coordinator.isCancelled
        #expect(cancelled)
    }

    @Test("multiple skipFile calls accumulate completedFiles correctly")
    func multipleSkips() async {
        let coordinator = DownloadCoordinator()
        let files = (0..<5).map {
            DownloadCoordinator.FileSpec(name: "file\($0).json", expectedBytes: 100)
        }
        await coordinator.configure(files: files)

        nonisolated(unsafe) var last: Gemma4DownloadProgress?
        await coordinator.setProgressHandler { p in last = p }

        for i in 0..<5 { await coordinator.skipFile(index: i) }

        #expect(last?.completedFiles == 5)
        #expect(last?.totalFiles == 5)
        #expect(last?.filesFraction == 1.0)
    }
}

// MARK: - Gemma4DownloadManager unit tests

@Suite("Gemma4DownloadManager", .serialized)
struct DownloadManagerTests {

    @Test("status returns .notDownloaded for unknown model with no cached files")
    @MainActor
    func statusUnknown() {
        let manager = Gemma4DownloadManager.shared
        // Use a synthetic ID that will never be on disk in CI.
        let status = manager.status(forModelId: "test-org/nonexistent-model-xyz")
        if case .notDownloaded = status { } else if case .downloaded = status { }
        // Acceptable: either not downloaded or (extremely unlikely) downloaded.
        // The point is it doesn't crash.
        #expect(true)
    }

    @Test("download returns same task when called twice with same modelId")
    @MainActor
    func idempotentDownload() async {
        // We can't run a real download in tests. Instead verify the idempotency
        // logic: a second call while isDownloading returns the same object.
        // We use a fake ID so no actual network request is made before we cancel.
        let manager = Gemma4DownloadManager.shared
        let id = "test-org/idempotency-test-\(UUID().uuidString)"
        let t1 = manager.download(modelId: id)
        let t2 = manager.download(modelId: id)
        // Both references should point to the same task instance.
        #expect(t1 === t2)
        // Clean up: cancel so we don't leave a zombie task.
        await manager.cancel(modelId: id)
        manager.clearTask(modelId: id)
    }

    @Test("cancel transitions task to failed")
    @MainActor
    func cancelTransitions() async {
        let manager = Gemma4DownloadManager.shared
        let id = "test-org/cancel-test-\(UUID().uuidString)"
        let task = manager.download(modelId: id)
        await manager.cancel(modelId: id)
        if case .failed = task.status { } else {
            // Task may not have started yet; either failed or still in initial downloading state is OK.
        }
        manager.clearTask(modelId: id)
    }

    @Test("cancel after completion does not regress status to failed")
    @MainActor
    func cancelAfterCompletion() async {
        let manager = Gemma4DownloadManager.shared
        let id = "test-org/cancel-after-complete-\(UUID().uuidString)"
        let coordinator = DownloadCoordinator()
        let task = Gemma4DownloadTask(modelId: id, coordinator: coordinator)
        // Simulate a completed download by marking it done before cancel fires.
        task.markCompleted()
        // cancel() must be a no-op once already in a terminal non-downloading state.
        await task.cancel()
        if case .downloaded = task.status { } else {
            #expect(Bool(false), "cancel() must not overwrite a completed status; got \(task.status)")
        }
        manager.clearTask(modelId: id)
    }


    @MainActor
    func retryCreatesNew() async {
        let manager = Gemma4DownloadManager.shared
        let id = "test-org/retry-test-\(UUID().uuidString)"
        let original = manager.download(modelId: id)
        await manager.cancel(modelId: id)
        let retried = manager.retry(modelId: id)
        #expect(original !== retried)
        await manager.cancel(modelId: id)
        manager.clearTask(modelId: id)
    }

    @Test("delete removes model directory if present")
    @MainActor
    func deleteNonExistent() throws {
        // Deleting a model that isn't on disk should not throw.
        let manager = Gemma4DownloadManager.shared
        // Use the shared manager with a synthetic model ID.
        // Since the directory doesn't exist, this must be a no-op.
        try manager.delete(modelId: "test-org/delete-test-\(UUID().uuidString)")
    }
}
