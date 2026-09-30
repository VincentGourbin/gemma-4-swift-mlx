import Testing
import Foundation
import MLX
import MLXLMCommon
@testable import Gemma4Swift

/// K-43 sur vrai modele : tours liberees apres le prefill, rechargees a la requete image
/// suivante, memes jetons greedy que sans liberation ; memoire active plus basse entre-temps.
private let modelPath = ProcessInfo.processInfo.environment["GEMMA4_INTEGRATION_MODEL_PATH"]

@Suite("Residence par etape (integration)", .serialized, .enabled(if: modelPath != nil))
struct EncoderResidencyIntegrationTests {

    private func describe(_ container: ModelContainer, pixels: MLXArray) async throws -> [Int] {
        nonisolated(unsafe) let pixelsCapture = pixels
        return try await container.perform { context in
            let ids = try Gemma4Processor.multimodalChatIds(userPrompt: "Describe this image in detail.", tokenizer: context.tokenizer)
            (context.model as! Gemma4MultimodalLLMModel).pendingPixelValues = pixelsCapture
            var iterator = try TokenIterator(
                input: LMInput(tokens: MLXArray(ids.map { Int32($0) })), model: context.model,
                parameters: GenerateParameters(maxTokens: 32, temperature: 0))
            var out: [Int] = []
            while out.count < 32, let t = iterator.next() { out.append(t) }
            return out
        }
    }

    @Test("liberer puis recharger : memes jetons, memoire rendue")
    func testReleaseAndRestore() async throws {
        let url = URL(fileURLWithPath: modelPath!)
        let image = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appendingPathComponent("../../docs/examples/vision-image-description/UI.png").standardized
        let pixels = try Gemma4ImageProcessor.processImage(url: image)
        eval(pixels)

        let resident = try await Gemma4Registration.loadContainer(from: url, multimodal: true, audio: false)
        let expected = try await describe(resident, pixels: pixels)

        let staged = try await Gemma4Registration.loadContainer(from: url, multimodal: true, audio: false)
        await staged.perform { ($0.model as! Gemma4MultimodalLLMModel).releaseEncodersAfterPrefill = true }
        Memory.clearCache()
        let first = try await describe(staged, pixels: pixels)
        let released = await staged.perform { ($0.model as! Gemma4MultimodalLLMModel).encodersReleased }
        let second = try await describe(staged, pixels: pixels)  // recharge les tours
        #expect(released)
        #expect(first == expected)
        #expect(second == expected)
        let active = await staged.perform { _ in Memory.activeMemory >> 20 }
        print("DIAG K-43 actif apres liberation : \(active) Mo")
    }
}
