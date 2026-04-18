import Testing
import Foundation
@testable import Gemma4Swift

private let dummyProgress = Gemma4DownloadProgress(
    completedBytes: 100,
    totalBytes: 1000,
    completedFiles: 0,
    totalFiles: 5,
    currentFile: "model.safetensors",
    bytesPerSecond: 1_000_000,
    estimatedSecondsRemaining: 10
)

@Suite("ModelStatus")
struct ModelStatusTests {

    // MARK: - isTerminal

    @Test("notDownloaded is not terminal")
    func notDownloadedNotTerminal() {
        #expect(!ModelStatus.notDownloaded.isTerminal)
    }

    @Test("downloading is not terminal")
    func downloadingNotTerminal() {
        #expect(!ModelStatus.downloading(dummyProgress).isTerminal)
    }

    @Test("downloaded is terminal")
    func downloadedIsTerminal() {
        #expect(ModelStatus.downloaded.isTerminal)
    }

    @Test("loading is not terminal")
    func loadingNotTerminal() {
        #expect(!ModelStatus.loading.isTerminal)
    }

    @Test("ready is terminal")
    func readyIsTerminal() {
        #expect(ModelStatus.ready.isTerminal)
    }

    @Test("failed is terminal")
    func failedIsTerminal() {
        #expect(ModelStatus.failed(.cancelled("x")).isTerminal)
    }

    // MARK: - isDownloading

    @Test("isDownloading only true for .downloading")
    func isDownloadingFlag() {
        #expect(ModelStatus.downloading(dummyProgress).isDownloading)
        #expect(!ModelStatus.downloaded.isDownloading)
        #expect(!ModelStatus.notDownloaded.isDownloading)
        #expect(!ModelStatus.ready.isDownloading)
    }

    // MARK: - isLoaded

    @Test("isLoaded only true for .ready")
    func isLoadedFlag() {
        #expect(ModelStatus.ready.isLoaded)
        #expect(!ModelStatus.downloaded.isLoaded)
        #expect(!ModelStatus.loading.isLoaded)
        #expect(!ModelStatus.downloading(dummyProgress).isLoaded)
    }

    // MARK: - progress associated value

    @Test("progress is non-nil only for .downloading")
    func progressAssociated() {
        #expect(ModelStatus.downloading(dummyProgress).progress != nil)
        #expect(ModelStatus.downloaded.progress == nil)
        #expect(ModelStatus.notDownloaded.progress == nil)
        #expect(ModelStatus.failed(.cancelled("x")).progress == nil)
    }

    // MARK: - label

    @Test("label for notDownloaded")
    func labelNotDownloaded() {
        #expect(ModelStatus.notDownloaded.label == "Not downloaded")
    }

    @Test("label for downloading shows percentage")
    func labelDownloading() {
        let label = ModelStatus.downloading(dummyProgress).label
        // bytesFraction = 100/1000 = 10%
        #expect(label.contains("10%"))
    }

    @Test("label for downloaded")
    func labelDownloaded() {
        #expect(ModelStatus.downloaded.label == "Downloaded")
    }

    @Test("label for loading")
    func labelLoading() {
        #expect(ModelStatus.loading.label == "Loading")
    }

    @Test("label for ready")
    func labelReady() {
        #expect(ModelStatus.ready.label == "Ready")
    }

    @Test("label for failed")
    func labelFailed() {
        #expect(ModelStatus.failed(.cancelled("x")).label == "Failed")
    }
}
