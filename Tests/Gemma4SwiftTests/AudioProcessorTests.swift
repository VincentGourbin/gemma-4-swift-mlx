import XCTest
import Foundation
@testable import Gemma4Swift

final class AudioProcessorTests: XCTestCase {

    // MARK: - Static parameters

    func testMelParameters() {
        XCTAssertEqual(Gemma4AudioProcessor.numMelFilters, 128)
        XCTAssertEqual(Gemma4AudioProcessor.sampleRate, 16000)
        XCTAssertEqual(Gemma4AudioProcessor.frameLength, 320)
        XCTAssertEqual(Gemma4AudioProcessor.hopLength, 160)
        XCTAssertEqual(Gemma4AudioProcessor.msPerToken, 40.0, accuracy: 1e-6)
    }

    func testFFTLength() {
        // fftOverdrive=false → next power of 2 >= frameLength(320) = 512
        XCTAssertEqual(Gemma4AudioProcessor.fftLength, 512)
    }

    func testMaxAudioSamples() {
        XCTAssertEqual(Gemma4AudioProcessor.maxAudioSamples, 480_000)
    }

    func testMaxAudioTokens() {
        XCTAssertEqual(Gemma4AudioProcessor.maxAudioTokens, 750)
    }

    // MARK: - computeAudioNumTokens

    func testComputeAudioNumTokens30s() {
        // 30s @ 16kHz = 480000 samples
        // Semicausal padding adds frameLength/2 = 160 zeros
        // numFrames = (480000 + 160 - 320) / 160 + 1 = 3000
        // Pass 1: (3000 - 1) / 2 + 1 = 1500
        // Pass 2: (1500 - 1) / 2 + 1 = 750
        // min(750, 750) = 750
        XCTAssertEqual(Gemma4AudioProcessor.computeAudioNumTokens(melFrames: 3000), 750)
    }

    func testComputeAudioNumTokens10s() {
        // 10s @ 16kHz = 160000 samples
        // numFrames = (160000 + 160 - 320) / 160 + 1 = 1000
        // Pass 1: (1000 - 1) / 2 + 1 = 500
        // Pass 2: (500 - 1) / 2 + 1 = 250
        XCTAssertEqual(Gemma4AudioProcessor.computeAudioNumTokens(melFrames: 1000), 250)
    }

    func testComputeAudioNumTokens1s() {
        // 1s @ 16kHz = 16000 samples
        // numFrames = (16000 + 160 - 320) / 160 + 1 = 100
        // Pass 1: (100 - 1) / 2 + 1 = 50
        // Pass 2: (50 - 1) / 2 + 1 = 25
        XCTAssertEqual(Gemma4AudioProcessor.computeAudioNumTokens(melFrames: 100), 25)
    }

    func testComputeAudioNumTokensCappedAt750() {
        // Very large mel frame counts must be capped at maxAudioTokens (750)
        XCTAssertEqual(Gemma4AudioProcessor.computeAudioNumTokens(melFrames: 100_000), 750)
    }

    // MARK: - createMelFilterBank

    func testMelFilterBankShape() {
        // fftLength = 512, numBins = fftLength/2 + 1 = 257
        // Returns [numMelFilters][numBins] = [128][257]
        let numMelFilters = Gemma4AudioProcessor.numMelFilters  // 128
        let fftLength = Gemma4AudioProcessor.fftLength          // 512
        let numBins = fftLength / 2 + 1                         // 257
        let sampleRate = Gemma4AudioProcessor.sampleRate

        let filterBank = Gemma4AudioProcessor.createMelFilterBank(
            numFilters: numMelFilters,
            numBins: numBins,
            sampleRate: sampleRate,
            minFreq: 0.0,
            maxFreq: 8000.0
        )

        XCTAssertEqual(filterBank.count, 128)
        XCTAssertTrue(filterBank.allSatisfy { $0.count == 257 })
    }
}
