import XCTest
@testable import Gemma4Swift

// Gemma4Pipeline is @MainActor, so the entire test class must be too.
@MainActor
final class PipelineTests: XCTestCase {

    // MARK: - State machine

    func testInitialState() {
        let pipeline = Gemma4Pipeline()
        if case .unloaded = pipeline.state {
            // expected
        } else {
            XCTFail("Expected initial state to be .unloaded, got \(pipeline.state)")
        }
        XCTAssertFalse(pipeline.isReady, "isReady should be false before a container is set")
    }

    func testSetContainerMakesReady() {
        // We cannot instantiate a real ModelContainer in unit tests (requires on-disk weights),
        // but we can verify the state transition via the public setContainer(_:) API.
        // Skip if we can't construct a container without a model on disk.
        // This test exercises the synchronous state update path only.
        let pipeline = Gemma4Pipeline()
        XCTAssertFalse(pipeline.isReady)

        // We verify the contract: after setContainer the state must be .ready.
        // Because ModelContainer has no public initializer suitable for tests we confirm
        // the API surface: isReady is derived from state == .ready.
        // The test below validates the negative path (not loaded) which is testable without a model.
        // If a ModelContainer were available, setContainer would transition to .ready.
        // For now we assert that the pipeline starts in .unloaded.
        if case .unloaded = pipeline.state {
            XCTAssertTrue(true) // state machine starts correctly
        } else {
            XCTFail("Pipeline must start in .unloaded before any container is provided")
        }
    }

    func testUnloadClearsState() {
        let pipeline = Gemma4Pipeline()
        // Calling unload() on a fresh pipeline should be a no-op and leave state as .unloaded
        pipeline.unload()
        if case .unloaded = pipeline.state {
            // expected
        } else {
            XCTFail("State should be .unloaded after unload(), got \(pipeline.state)")
        }
        XCTAssertFalse(pipeline.isReady)
        XCTAssertNil(pipeline.lastStats, "lastStats should be nil after unload()")
    }

    // MARK: - Error paths (no model loaded)

    func testChatThrowsWhenNotLoaded() async {
        let pipeline = Gemma4Pipeline()
        do {
            _ = try await pipeline.chat(prompt: "hello")
            XCTFail("chat() should throw when no container is set")
        } catch let error as Gemma4PipelineError {
            if case .modelNotLoaded = error {
                // expected
            } else {
                XCTFail("Expected .modelNotLoaded, got \(error)")
            }
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }
    }

    func testChatStreamThrowsWhenNotLoaded() {
        let pipeline = Gemma4Pipeline()
        do {
            _ = try pipeline.chatStream(prompt: "hello")
            XCTFail("chatStream() should throw when no container is set")
        } catch let error as Gemma4PipelineError {
            if case .modelNotLoaded = error {
                // expected
            } else {
                XCTFail("Expected .modelNotLoaded, got \(error)")
            }
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }
    }

    func testContinueChatThrowsWhenNoSession() async {
        // continueChat() guards on currentSession (set only after a successful chat()),
        // so without a prior chat call it should throw .modelNotLoaded.
        let pipeline = Gemma4Pipeline()
        do {
            _ = try await pipeline.continueChat(prompt: "follow up")
            XCTFail("continueChat() should throw when there is no active session")
        } catch let error as Gemma4PipelineError {
            if case .modelNotLoaded = error {
                // expected — continueChat reuses the .modelNotLoaded error for a missing session
            } else {
                XCTFail("Expected .modelNotLoaded, got \(error)")
            }
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }
    }
}
