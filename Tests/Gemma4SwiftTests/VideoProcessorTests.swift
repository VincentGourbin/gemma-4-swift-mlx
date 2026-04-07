import XCTest
import Foundation
@testable import Gemma4Swift

final class VideoProcessorTests: XCTestCase {

    // MARK: - Constants

    func testDefaultConstants() {
        XCTAssertEqual(Gemma4VideoProcessor.defaultSoftTokensPerFrame, 70)
        XCTAssertEqual(Gemma4VideoProcessor.defaultMaxFrames, 32)
        XCTAssertEqual(Gemma4VideoProcessor.maxVideoDurationSeconds, 60.0, accuracy: 1e-9)
    }

    // MARK: - formatTimestamp

    func testFormatTimestamp() {
        XCTAssertEqual(Gemma4VideoProcessor.formatTimestamp(0), "00:00")
        XCTAssertEqual(Gemma4VideoProcessor.formatTimestamp(65), "01:05")
        XCTAssertEqual(Gemma4VideoProcessor.formatTimestamp(3599), "59:59")
    }

    func testFormatTimestampFractional() {
        // Fractional seconds truncate to integer seconds
        XCTAssertEqual(Gemma4VideoProcessor.formatTimestamp(1.5), "00:01")
    }
}
