import Testing
import Foundation
import CoreGraphics
import MLX
@testable import Gemma4Swift

/// Garde-fou S-01 (audit du 2026-09-27) : un consommateur qui lache un stream
/// doit arreter la generation. Avant le correctif, la Task interne continuait
/// jusqu'a `maxTokens` en tenant le `ModelContainer` : l'etat restait
/// `.processing` et l'appel suivant attendait.
///
/// ```
/// GEMMA4_INTEGRATION_MODEL_PATH=~/Library/Caches/models/mlx-community/gemma-4-e2b-it-4bit \
///   Scripts/run-tests.sh -only-testing:Gemma4SwiftTests/StreamCancellationIntegrationTests
/// ```
private let integrationModelPath = ProcessInfo.processInfo
    .environment["GEMMA4_INTEGRATION_MODEL_PATH"]

@Suite("Annulation des streams (integration)", .serialized)
struct StreamCancellationIntegrationTests {

    /// Assez long pour que la generation complete prenne bien plus que le
    /// delai accorde au retour a `.ready`.
    static let maxTokens = 1_500
    static let prompt = "Write a very long, detailed essay about the history of the bicycle."

    @MainActor
    private func loadPipeline() async throws -> Gemma4Pipeline {
        let pipeline = Gemma4Pipeline()
        try await pipeline.load(
            from: URL(fileURLWithPath: integrationModelPath!), multimodal: true)
        return pipeline
    }

    /// Consomme `chunks` morceaux dans une Task, puis annule cette Task —
    /// ce que fait une UI qui ferme sa vue. Si le stream se termine ou echoue
    /// avant, l'erreur remonte au lieu de laisser le test attendre.
    @MainActor
    private func consumeThenCancel(
        _ stream: AsyncThrowingStream<String, Error>, chunks: Int = 5
    ) async throws {
        let received = AsyncStream<Void>.makeStream()
        let consumer = Task {
            defer { received.continuation.finish() }
            var n = 0
            for try await _ in stream {
                n += 1
                if n == chunks { received.continuation.yield() }
            }
        }
        var reached = false
        for await _ in received.stream { reached = true; break }
        consumer.cancel()
        if !reached {
            try await consumer.value
            Issue.record("le stream s'est termine avant \(chunks) morceaux")
        }
    }

    /// Attend le retour a `.ready`, ou echoue apres `seconds`.
    @MainActor
    private func expectReady(_ pipeline: Gemma4Pipeline, within seconds: Double) async {
        let deadline = Date().addingTimeInterval(seconds)
        while Date() < deadline {
            if case .ready = pipeline.state { return }
            try? await Task.sleep(for: .milliseconds(20))
        }
        Issue.record("toujours \(pipeline.state) \(seconds) s apres l'annulation du consommateur")
    }

    private func syntheticImage() throws -> CGImage {
        let ctx = try #require(CGContext(
            data: nil, width: 224, height: 224, bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
        ctx.setFillColor(CGColor(red: 0.9, green: 0.7, blue: 0.1, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: 224, height: 224))
        return try #require(ctx.makeImage())
    }

    @Test("chatStream (ChatSession) s'arrete quand le consommateur annule",
          .enabled(if: integrationModelPath != nil))
    @MainActor
    func testChatSessionPath() async throws {
        let pipeline = try await loadPipeline()
        let stream = try pipeline.chatStream(prompt: Self.prompt, maxTokens: Self.maxTokens)
        try await consumeThenCancel(stream)
        await expectReady(pipeline, within: 3)
    }

    @Test("chatStream (TokenIterator, n-gramme) s'arrete quand le consommateur annule",
          .enabled(if: integrationModelPath != nil))
    @MainActor
    func testBypassPath() async throws {
        let pipeline = try await loadPipeline()
        let stream = try pipeline.chatStream(
            prompt: Self.prompt, maxTokens: Self.maxTokens, noRepeatNGramSize: 5)
        try await consumeThenCancel(stream)
        await expectReady(pipeline, within: 3)
    }

    @Test("chatStreamMultimodal s'arrete et le modele est libre pour l'appel suivant",
          .enabled(if: integrationModelPath != nil))
    @MainActor
    func testMultimodalPathAndNextCall() async throws {
        let pipeline = try await loadPipeline()
        let pixels = try Gemma4ImageProcessor.processImage(try syntheticImage())
        let stream = try pipeline.chatStreamMultimodal(
            prompt: Self.prompt, pixelValues: pixels, maxTokens: Self.maxTokens)
        try await consumeThenCancel(stream)
        await expectReady(pipeline, within: 3)

        // L'appel suivant obtient son premier morceau sans attendre la fin
        // d'une generation fantome.
        let start = Date()
        for try await _ in try pipeline.chatStream(prompt: "Say hi.", maxTokens: 8) { break }
        #expect(Date().timeIntervalSince(start) < 10)
    }
}
