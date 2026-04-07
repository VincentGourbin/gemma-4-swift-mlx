// Tests for Gemma4Processor — token constants, prompt building, and token validation

import XCTest
import MLX
@testable import Gemma4Swift

final class Gemma4ProcessorTests: XCTestCase {

    // MARK: - Token ID Constants

    func testTokenConstants() {
        XCTAssertEqual(Gemma4Processor.imageTokenId, 258880)
        XCTAssertEqual(Gemma4Processor.audioTokenId, 258881)
        XCTAssertEqual(Gemma4Processor.videoTokenId, 258884)
        XCTAssertEqual(Gemma4Processor.boiTokenId, 255999)
        XCTAssertEqual(Gemma4Processor.eoiTokenId, 258882)
        XCTAssertEqual(Gemma4Processor.boaTokenId, 256000)
        XCTAssertEqual(Gemma4Processor.eoaTokenId, 258883)
        XCTAssertEqual(Gemma4Processor.thinkTokenId, 98)
        XCTAssertEqual(Gemma4Processor.channelStartTokenId, 100)
        XCTAssertEqual(Gemma4Processor.channelEndTokenId, 101)
    }

    // MARK: - EOS Token IDs

    func testEOSTokenIds() {
        let eos = Gemma4Processor.eosTokenIds
        XCTAssertEqual(eos.count, 3)
        XCTAssertTrue(eos.contains(1))
        XCTAssertTrue(eos.contains(106))
        XCTAssertTrue(eos.contains(50))
    }

    // MARK: - buildMultimodalPrompt — text only

    func testBuildMultimodalPromptTextOnly() {
        let userPrompt = "What is the capital of France?"
        let result = Gemma4Processor.buildMultimodalPrompt(userPrompt: userPrompt)

        XCTAssertTrue(result.contains("<start_of_turn>user"), "Missing user turn marker")
        XCTAssertTrue(result.contains(userPrompt), "Missing user prompt text")
        XCTAssertTrue(result.contains("<start_of_turn>model"), "Missing model turn marker")
        // No multimodal tokens
        XCTAssertFalse(result.contains(Gemma4Processor.imageToken))
        XCTAssertFalse(result.contains(Gemma4Processor.audioToken))
        XCTAssertFalse(result.contains(Gemma4Processor.videoToken))
    }

    // MARK: - buildMultimodalPrompt — image

    func testBuildMultimodalPromptWithImage() {
        let result = Gemma4Processor.buildMultimodalPrompt(
            userPrompt: "Describe this image.",
            hasImage: true,
            numImageTokens: 280
        )

        XCTAssertTrue(result.contains(Gemma4Processor.boiToken), "Missing boi token <|image>")
        XCTAssertTrue(result.contains(Gemma4Processor.eoiToken), "Missing eoi token <image|>")

        // Exactly 280 image soft tokens
        let imageTokenCount = result.components(separatedBy: Gemma4Processor.imageToken).count - 1
        XCTAssertEqual(imageTokenCount, 280, "Expected 280 image soft tokens")

        // No video or audio tokens
        XCTAssertFalse(result.contains(Gemma4Processor.videoToken))
        XCTAssertFalse(result.contains(Gemma4Processor.audioToken))
    }

    // MARK: - buildMultimodalPrompt — video

    func testBuildMultimodalPromptWithVideo() {
        let numFrames = 3
        let softTokensPerFrame = 70
        let timestamps: [Double] = [0, 1, 2]

        let result = Gemma4Processor.buildMultimodalPrompt(
            userPrompt: "Describe this video.",
            hasVideo: true,
            numVideoFrames: numFrames,
            softTokensPerFrame: softTokensPerFrame,
            videoTimestamps: timestamps
        )

        // Video tokens, not image tokens
        XCTAssertTrue(result.contains(Gemma4Processor.videoToken), "Missing video soft token <|video|>")
        XCTAssertFalse(result.contains(Gemma4Processor.imageToken), "Should not contain image tokens")

        // 70 video tokens per frame x 3 frames = 210 total
        let videoTokenCount = result.components(separatedBy: Gemma4Processor.videoToken).count - 1
        XCTAssertEqual(videoTokenCount, numFrames * softTokensPerFrame, "Expected \(numFrames * softTokensPerFrame) video soft tokens")

        // MM:SS timestamps for each frame
        XCTAssertTrue(result.contains("00:00"), "Missing timestamp 00:00 for frame 0")
        XCTAssertTrue(result.contains("00:01"), "Missing timestamp 00:01 for frame 1")
        XCTAssertTrue(result.contains("00:02"), "Missing timestamp 00:02 for frame 2")
    }

    // MARK: - buildMultimodalPrompt — audio

    func testBuildMultimodalPromptWithAudio() {
        let numAudioTokens = 100
        let result = Gemma4Processor.buildMultimodalPrompt(
            userPrompt: "Transcribe this audio.",
            hasAudio: true,
            numAudioTokens: numAudioTokens
        )

        XCTAssertTrue(result.contains(Gemma4Processor.boaToken), "Missing boa token <|audio>")
        XCTAssertTrue(result.contains(Gemma4Processor.eoaToken), "Missing eoa token <audio|>")

        let audioTokenCount = result.components(separatedBy: Gemma4Processor.audioToken).count - 1
        XCTAssertEqual(audioTokenCount, numAudioTokens, "Expected \(numAudioTokens) audio soft tokens")

        XCTAssertFalse(result.contains(Gemma4Processor.imageToken))
        XCTAssertFalse(result.contains(Gemma4Processor.videoToken))
    }

    // MARK: - buildMultimodalPrompt — system prompt

    func testBuildMultimodalPromptWithSystemPrompt() {
        let systemPrompt = "You are a helpful assistant."
        let result = Gemma4Processor.buildMultimodalPrompt(
            userPrompt: "Hello.",
            systemPrompt: systemPrompt
        )

        XCTAssertTrue(result.contains(systemPrompt), "System prompt text not found in output")
        XCTAssertTrue(result.contains("<start_of_turn>system"), "Missing system turn marker")
    }

    // MARK: - buildMultimodalPrompt — combined modalities

    func testBuildMultimodalPromptCombined() {
        let userPrompt = "Analyze everything."
        let result = Gemma4Processor.buildMultimodalPrompt(
            userPrompt: userPrompt,
            hasImage: true,
            numImageTokens: 280,
            hasAudio: true,
            numAudioTokens: 50,
            hasVideo: true,
            numVideoFrames: 2,
            softTokensPerFrame: 70,
            videoTimestamps: [0.0, 1.0]
        )

        // All modality markers present
        XCTAssertTrue(result.contains(Gemma4Processor.boiToken))
        XCTAssertTrue(result.contains(Gemma4Processor.eoiToken))
        XCTAssertTrue(result.contains(Gemma4Processor.boaToken))
        XCTAssertTrue(result.contains(Gemma4Processor.eoaToken))
        XCTAssertTrue(result.contains(Gemma4Processor.imageToken))
        XCTAssertTrue(result.contains(Gemma4Processor.videoToken))
        XCTAssertTrue(result.contains(Gemma4Processor.audioToken))

        // Ordering: image first, then video, then audio, then user text
        let imageRange = result.range(of: Gemma4Processor.imageToken)
        let videoRange = result.range(of: Gemma4Processor.videoToken)
        let audioRange = result.range(of: Gemma4Processor.audioToken)
        let userPromptRange = result.range(of: userPrompt)

        XCTAssertNotNil(imageRange)
        XCTAssertNotNil(videoRange)
        XCTAssertNotNil(audioRange)
        XCTAssertNotNil(userPromptRange)

        if let ir = imageRange, let vr = videoRange, let ar = audioRange, let ur = userPromptRange {
            XCTAssertTrue(ir.lowerBound < vr.lowerBound, "Image should appear before video")
            XCTAssertTrue(vr.lowerBound < ar.lowerBound, "Video should appear before audio")
            XCTAssertTrue(ar.lowerBound < ur.lowerBound, "Audio should appear before user text")
        }
    }

    // MARK: - validateTokenCounts

    func testValidateTokenCounts() {
        // Build an array containing 3 image tokens and 2 audio tokens plus some other IDs
        let ids: [Int32] = [
            Gemma4Processor.imageTokenId,
            Gemma4Processor.audioTokenId,
            Gemma4Processor.imageTokenId,
            99,
            Gemma4Processor.audioTokenId,
            Gemma4Processor.imageTokenId,
            1,
        ]
        let inputIds = MLXArray(ids)

        let counts = Gemma4Processor.validateTokenCounts(
            inputIds: inputIds,
            expectedImageTokens: 3,
            expectedAudioTokens: 2
        )

        XCTAssertEqual(counts.imageCount, 3)
        XCTAssertEqual(counts.audioCount, 2)
    }
}
