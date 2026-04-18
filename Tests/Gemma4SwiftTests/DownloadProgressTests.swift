import Testing
import Foundation
@testable import Gemma4Swift

private func makeProgress(
    completedBytes: Int64 = 0,
    totalBytes: Int64 = 0,
    completedFiles: Int = 0,
    totalFiles: Int = 1,
    currentFile: String = "model.safetensors",
    bytesPerSecond: Double = 0,
    eta: Double? = nil
) -> Gemma4DownloadProgress {
    Gemma4DownloadProgress(
        completedBytes: completedBytes,
        totalBytes: totalBytes,
        completedFiles: completedFiles,
        totalFiles: totalFiles,
        currentFile: currentFile,
        bytesPerSecond: bytesPerSecond,
        estimatedSecondsRemaining: eta
    )
}

@Suite("Gemma4DownloadProgress")
struct DownloadProgressTests {

    // MARK: - bytesFraction

    @Test("bytesFraction is 0 when totalBytes is 0")
    func bytesFractionUnknownTotal() {
        let p = makeProgress(completedBytes: 500, totalBytes: 0, completedFiles: 0, totalFiles: 2)
        // Falls back to filesFraction = 0/2 = 0
        #expect(p.bytesFraction == 0)
    }

    @Test("bytesFraction is correct mid-download")
    func bytesFractionMid() {
        let p = makeProgress(completedBytes: 500, totalBytes: 1000)
        #expect(p.bytesFraction == 0.5)
    }

    @Test("bytesFraction is clamped to 1 on overshoot")
    func bytesFractionClamp() {
        let p = makeProgress(completedBytes: 1200, totalBytes: 1000)
        #expect(p.bytesFraction == 1.0)
    }

    @Test("bytesFraction matches fraction alias")
    func fractionAlias() {
        let p = makeProgress(completedBytes: 300, totalBytes: 600)
        #expect(p.bytesFraction == p.fraction)
    }

    // MARK: - filesFraction

    @Test("filesFraction is 0 when totalFiles is 0")
    func filesFractionZeroTotal() {
        let p = makeProgress(completedFiles: 0, totalFiles: 0)
        #expect(p.filesFraction == 0)
    }

    @Test("filesFraction is correct")
    func filesFractionMid() {
        let p = makeProgress(completedFiles: 2, totalFiles: 4)
        #expect(p.filesFraction == 0.5)
    }

    @Test("filesFraction is clamped to 1")
    func filesFractionClamp() {
        let p = makeProgress(completedFiles: 5, totalFiles: 4)
        #expect(p.filesFraction == 1.0)
    }

    // MARK: - formattedSpeed

    @Test("formattedSpeed returns dash when speed is 0")
    func speedZero() {
        let p = makeProgress(bytesPerSecond: 0)
        #expect(p.formattedSpeed == "-")
    }

    @Test("formattedSpeed contains /s suffix")
    func speedSuffix() {
        let p = makeProgress(bytesPerSecond: 4_200_000)
        #expect(p.formattedSpeed.hasSuffix("/s"))
    }

    // MARK: - formattedETA

    @Test("formattedETA is nil when eta is nil")
    func etaNil() {
        let p = makeProgress(eta: nil)
        #expect(p.formattedETA == nil)
    }

    @Test("formattedETA is nil when eta is 0")
    func etaZero() {
        let p = makeProgress(eta: 0)
        #expect(p.formattedETA == nil)
    }

    @Test("formattedETA shows seconds for values under 60")
    func etaSeconds() {
        let p = makeProgress(eta: 30)
        let label = p.formattedETA
        #expect(label != nil)
        #expect(label!.contains("30s"))
    }

    @Test("formattedETA shows minutes for values >= 60")
    func etaMinutes() {
        let p = makeProgress(eta: 90)
        let label = p.formattedETA
        #expect(label != nil)
        #expect(label!.contains("min"))
    }

    // MARK: - formattedProgress

    @Test("formattedProgress omits total when totalBytes is 0")
    func progressUnknownTotal() {
        let p = makeProgress(completedBytes: 512, totalBytes: 0)
        let text = p.formattedProgress
        #expect(!text.contains(" of "))
    }

    @Test("formattedProgress shows completed of total when both are known")
    func progressKnownTotal() {
        let p = makeProgress(completedBytes: 1024 * 1024, totalBytes: 10 * 1024 * 1024)
        #expect(p.formattedProgress.contains(" of "))
    }
}
