import Testing
import Foundation
import CoreGraphics
import MLX
@testable import Gemma4Swift

/// Chemin d'un modele Gemma 4 local (E2B/E4B) — les tests de cette suite ne
/// tournent que s'il est fourni :
///
/// ```
/// # xcodebuild ne transmet au runner que les variables prefixees TEST_RUNNER_
/// TEST_RUNNER_GEMMA4_INTEGRATION_MODEL_PATH=~/Library/Caches/models/mlx-community/gemma-4-e4b-it-4bit \
///   xcodebuild -scheme Gemma4Swift-Package -destination "platform=macOS" \
///   -derivedDataPath .build/xcode -skipMacroValidation test \
///   -only-testing:Gemma4SwiftTests/NoRepeatNGramIntegrationTests
/// ```
private let integrationModelPath = ProcessInfo.processInfo
    .environment["GEMMA4_INTEGRATION_MODEL_PATH"]

@Suite("No-repeat n-gram (integration)", .serialized)
struct NoRepeatNGramIntegrationTests {

    // MARK: - Helpers

    @MainActor
    private func loadPipeline() async throws -> Gemma4Pipeline {
        let pipeline = Gemma4Pipeline()
        try await pipeline.load(
            from: URL(fileURLWithPath: integrationModelPath!), multimodal: true)
        return pipeline
    }

    private func collect(_ stream: AsyncThrowingStream<String, Error>) async throws -> String {
        var text = ""
        for try await chunk in stream { text += chunk }
        return text
    }

    /// Longueur du plus long n-gramme de mots repete dans `text`.
    private func longestRepeatedWordNGram(_ text: String, upTo maxN: Int = 12) -> Int {
        let words = text.split(whereSeparator: { $0.isWhitespace || $0.isNewline }).map(String.init)
        var longest = 0
        for n in 1 ... maxN where words.count >= n {
            var seen = Set<String>()
            for start in 0 ... (words.count - n) {
                let gram = words[start ..< (start + n)].joined(separator: " ")
                if !seen.insert(gram).inserted {
                    longest = n
                    break
                }
            }
        }
        return longest
    }

    /// Image synthetique (degrade) — evite un asset binaire dans le repo.
    private func syntheticImage(width: Int = 224, height: Int = 224) throws -> CGImage {
        let space = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(
            data: nil, width: width, height: height, bitsPerComponent: 8,
            bytesPerRow: 0, space: space,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else {
            throw Gemma4PipelineError.invalidInput("CGContext indisponible")
        }
        ctx.setFillColor(CGColor(red: 0.1, green: 0.2, blue: 0.6, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
        ctx.setFillColor(CGColor(red: 0.9, green: 0.7, blue: 0.1, alpha: 1))
        ctx.fillEllipse(in: CGRect(
            x: width / 4, y: height / 4, width: width / 2, height: height / 2))
        guard let image = ctx.makeImage() else {
            throw Gemma4PipelineError.invalidInput("makeImage a echoue")
        }
        return image
    }

    // MARK: - Tests

    @Test("Chargement multimodal reel (regression PR #37 : K/V des tours conserves)",
          .enabled(if: integrationModelPath != nil))
    @MainActor
    func testMultimodalLoadSucceeds() async throws {
        let pipeline = try await loadPipeline()
        #expect(pipeline.isReady)
        pipeline.unload()
    }

    @Test("chatStream greedy : n=5 casse la repetition que le baseline produit",
          .enabled(if: integrationModelPath != nil))
    @MainActor
    func testTextStreamBlocksRepetition() async throws {
        let pipeline = try await loadPipeline()
        defer { pipeline.unload() }

        let system = "Tu obeis litteralement, sans commentaire."
        let prompt = "Ecris exactement 12 fois la phrase suivante, une par ligne : "
            + "le chat dort sur le tapis rouge."

        let baseline = try await collect(pipeline.chatStream(
            prompt: prompt, systemPrompt: system,
            temperature: 0.0, maxTokens: 200))
        let blocked = try await collect(pipeline.chatStream(
            prompt: prompt, systemPrompt: system,
            temperature: 0.0, maxTokens: 200, noRepeatNGramSize: 5))

        #expect(!blocked.isEmpty)
        // Le baseline greedy repete la phrase ; avec n=5 aucun 5-gramme ne peut
        // reapparaitre, donc le plus long n-gramme repete est <= 4.
        #expect(longestRepeatedWordNGram(baseline) >= 5)
        #expect(longestRepeatedWordNGram(blocked) <= 4)
    }

    @Test("chatStreamMultimodal greedy : n=5 respecte le contrat, sortie non vide",
          .enabled(if: integrationModelPath != nil))
    @MainActor
    func testMultimodalStreamBlocksRepetition() async throws {
        let pipeline = try await loadPipeline()
        defer { pipeline.unload() }

        let pixels = try Gemma4ImageProcessor.processImage(try syntheticImage())
        let text = try await collect(pipeline.chatStreamMultimodal(
            prompt: "Decris cette image en detail.",
            pixelValues: pixels,
            temperature: 0.0,
            maxTokens: 150,
            noRepeatNGramSize: 5))

        #expect(!text.isEmpty)
        #expect(longestRepeatedWordNGram(text) <= 4)
    }

    @Test("noRepeatNGramSize invalide -> invalidInput",
          .enabled(if: integrationModelPath != nil))
    @MainActor
    func testInvalidNGramSizeThrows() async throws {
        let pipeline = try await loadPipeline()
        defer { pipeline.unload() }

        #expect(throws: Gemma4PipelineError.self) {
            _ = try pipeline.chatStream(prompt: "bonjour", noRepeatNGramSize: 0)
        }
        let pixels = try Gemma4ImageProcessor.processImage(try syntheticImage())
        #expect(throws: Gemma4PipelineError.self) {
            _ = try pipeline.chatStreamMultimodal(
                prompt: "bonjour", pixelValues: pixels, noRepeatNGramSize: 0)
        }
    }
}
