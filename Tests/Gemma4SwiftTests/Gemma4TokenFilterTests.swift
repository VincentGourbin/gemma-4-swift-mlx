// Tests for Gemma4TokenFilter — thinking mode filtering and channel detection

import XCTest
@testable import Gemma4Swift

final class Gemma4TokenFilterTests: XCTestCase {

    // Convenience token IDs
    private let channelStart = Gemma4Processor.channelStartTokenId   // 100
    private let channelEnd   = Gemma4Processor.channelEndTokenId     // 101
    private let thinkToken   = Gemma4Processor.thinkTokenId          // 98

    // Feed a sequence of (tokenId, text) pairs into a filter and collect non-empty outputs
    private func feed(
        filter: Gemma4TokenFilter,
        tokens: [(Int32, String)]
    ) -> [String] {
        tokens.compactMap { id, text in
            let out = filter.process(tokenId: id, text: text)
            return out.isEmpty ? nil : out
        }
    }

    // MARK: - testDisabledModePassesResponseTokens

    func testDisabledModePassesResponseTokens() {
        let filter = Gemma4TokenFilter(mode: .disabled)
        let tokens: [(Int32, String)] = [(10, "Hello"), (11, " world"), (12, "!")]
        // No channel markers — tokens fall through .none channel and are treated as response
        let output = feed(filter: filter, tokens: tokens)
        XCTAssertEqual(output, ["Hello", " world", "!"])
    }

    // MARK: - testDisabledModeFiltersThinkingBlock

    func testDisabledModeFiltersThinkingBlock() {
        let filter = Gemma4TokenFilter(mode: .disabled)

        // Sequence: <|channel>(start) + "thought" -> thinking channel
        //           "secret thought" tokens (should be suppressed)
        //           <channel|>(end)
        //           <|channel>(start) + "response" -> response channel
        //           "visible answer" tokens (should pass through)
        //           regular text after channel ends
        let tokens: [(Int32, String)] = [
            (channelStart, "<|channel>"),
            (200, "thought"),
            (201, "secret"),
            (202, " thought"),
            (channelEnd, "<channel|>"),
            (channelStart, "<|channel>"),
            (203, "response"),
            (204, "visible"),
            (205, " answer"),
        ]

        let output = feed(filter: filter, tokens: tokens)

        // Thinking tokens must be suppressed
        XCTAssertFalse(output.contains("secret"), "Thinking content should be filtered")
        XCTAssertFalse(output.contains(" thought"), "Thinking content should be filtered")

        // Response tokens must be visible
        XCTAssertTrue(output.contains("visible"), "Response content should pass through")
        XCTAssertTrue(output.contains(" answer"), "Response content should pass through")
    }

    // MARK: - testEnabledModePassesEverything

    func testEnabledModePassesEverything() {
        let filter = Gemma4TokenFilter(mode: .enabled)

        let tokens: [(Int32, String)] = [
            (channelStart, "<|channel>"),
            (200, "thought"),
            (201, "secret"),
            (channelEnd, "<channel|>"),
            (channelStart, "<|channel>"),
            (203, "response"),
            (204, "public"),
        ]

        // In .enabled mode, process() returns the raw text for every token unchanged
        let rawOutputs = tokens.map { id, text in filter.process(tokenId: id, text: text) }

        // Every token text should be returned as-is
        for (index, (_, text)) in tokens.enumerated() {
            XCTAssertEqual(rawOutputs[index], text, "Enabled mode should return token text unchanged at index \(index)")
        }
    }

    // MARK: - testStructuredModeSeparation

    func testStructuredModeSeparation() {
        let filter = Gemma4TokenFilter(mode: .structured)

        let tokens: [(Int32, String)] = [
            (channelStart, "<|channel>"),
            (200, "thought"),
            (201, "thinking content"),
            (channelEnd, "<channel|>"),
            (channelStart, "<|channel>"),
            (203, "response"),
            (204, "final answer"),
        ]

        for (id, text) in tokens {
            _ = filter.process(tokenId: id, text: text)
        }

        let structured = filter.structuredResponse()

        XCTAssertNotNil(structured.thinking, "Structured response should capture thinking content")
        XCTAssertTrue(structured.thinking?.contains("thinking content") ?? false,
                      "Thinking block should contain captured thinking text")
        XCTAssertTrue(structured.response.contains("final answer"),
                      "Response block should contain response text")
        XCTAssertFalse(structured.response.contains("thinking content"),
                       "Response block should not contain thinking text")
    }

    // MARK: - testIsEOS

    func testIsEOS() {
        let filter = Gemma4TokenFilter(mode: .disabled)

        // EOS tokens
        XCTAssertTrue(filter.isEOS(1))
        XCTAssertTrue(filter.isEOS(106))
        XCTAssertTrue(filter.isEOS(50))

        // Non-EOS tokens
        XCTAssertFalse(filter.isEOS(0))
        XCTAssertFalse(filter.isEOS(2))
        XCTAssertFalse(filter.isEOS(98))
        XCTAssertFalse(filter.isEOS(258880))
    }

    // MARK: - testThinkTokenFiltered

    func testThinkTokenFiltered() {
        let filter = Gemma4TokenFilter(mode: .disabled)

        // <|think|> token should always produce empty output and not appear in response
        let out = filter.process(tokenId: thinkToken, text: "<|think|>")
        XCTAssertEqual(out, "", "thinkTokenId should be filtered (empty output)")
        XCTAssertEqual(filter.responseTokenCount, 0, "Think token should not be counted as response token")
    }

    // MARK: - testTokenCounts

    func testTokenCounts() {
        let filter = Gemma4TokenFilter(mode: .disabled)

        let tokens: [(Int32, String)] = [
            (channelStart, "<|channel>"),
            (200, "thought"),       // channel name — consumed during detection
            (201, "think1"),        // thinking token
            (202, "think2"),        // thinking token
            (channelEnd, "<channel|>"),
            (channelStart, "<|channel>"),
            (203, "response"),      // channel name — consumed during detection
            (204, "resp1"),         // response token
            (205, "resp2"),         // response token
            (206, "resp3"),         // response token
        ]

        for (id, text) in tokens {
            _ = filter.process(tokenId: id, text: text)
        }

        XCTAssertEqual(filter.thinkingTokenCount, 2, "Should count 2 thinking tokens")
        XCTAssertEqual(filter.responseTokenCount, 3, "Should count 3 response tokens")
    }

    // MARK: - testNoChannelDefaultsToResponse

    func testNoChannelDefaultsToResponse() {
        let filter = Gemma4TokenFilter(mode: .disabled)

        // Regular tokens with no channel markers — channel starts as .none which routes to response
        let tokens: [(Int32, String)] = [
            (10, "Hello"),
            (11, ", "),
            (12, "world"),
        ]

        for (id, text) in tokens {
            _ = filter.process(tokenId: id, text: text)
        }

        XCTAssertEqual(filter.responseTokenCount, 3, "Tokens without channel markers should be counted as response")
        XCTAssertEqual(filter.thinkingTokenCount, 0, "No thinking tokens expected")
    }

    // MARK: - testChannelDetectionWithPartialText

    func testChannelDetectionWithPartialText() {
        let filter = Gemma4TokenFilter(mode: .disabled)

        // Feed channelStart, then the word "thought" split across two tokens
        let detectStart = filter.process(tokenId: channelStart, text: "<|channel>")
        XCTAssertEqual(detectStart, "", "Channel start token should produce no output")

        // "th" alone does not match "thought" or "response"
        let partialA = filter.process(tokenId: 200, text: "th")
        XCTAssertEqual(partialA, "", "Partial channel name should produce no output while detecting")

        // "ought" completes "thought" — channel switches to thinking
        let partialB = filter.process(tokenId: 201, text: "ought")
        XCTAssertEqual(partialB, "", "Channel name completion token should produce no output")

        // Next token should now be treated as a thinking token (suppressed)
        let thinkingToken = filter.process(tokenId: 202, text: "secret thought")
        XCTAssertEqual(thinkingToken, "", "Token after 'thought' detection should be filtered as thinking")

        XCTAssertEqual(filter.thinkingTokenCount, 1, "One thinking token should have been captured")
        XCTAssertEqual(filter.responseTokenCount, 0, "No response tokens should have been captured")
    }
}
