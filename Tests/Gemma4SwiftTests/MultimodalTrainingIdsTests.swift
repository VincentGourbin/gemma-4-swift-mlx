import Testing
import Foundation
import MLXLMCommon
@testable import Gemma4Swift

/// A-03 / A-04 : les ids d'entrainement multimodal ont exactement le format de l'inference
/// (le prompt jusqu'au tour du modele = `multimodalChatIds`), et un media sans tour user est
/// une erreur. Tokenizer d'un vrai pack :
///   GEMMA4_INTEGRATION_MODEL_PATH=… Scripts/run-tests.sh -only-testing:Gemma4SwiftTests/MultimodalTrainingIdsTests
private let modelPath = ProcessInfo.processInfo.environment["GEMMA4_INTEGRATION_MODEL_PATH"]

@Suite("Ids d'entrainement multimodal", .enabled(if: modelPath != nil))
struct MultimodalTrainingIdsTests {

    @Test("image : prompt d'entrainement == prompt d'inference, reponse a la suite")
    func testImageMatchesInference() async throws {
        let tokenizer = try await Gemma4TokenizerLoader().load(from: URL(fileURLWithPath: modelPath!))
        let training = try Gemma4Processor.multimodalTrainingIds(
            messages: [["role": "user", "content": "Describe this image."],
                       ["role": "assistant", "content": "A lighthouse at dusk."]],
            hasImage: true, audioTokens: nil, tokenizer: tokenizer)
        let inference = try Gemma4Processor.multimodalChatIds(userPrompt: "Describe this image.", tokenizer: tokenizer)
        #expect(Array(training.prefix(inference.count)) == inference)
        let answer = tokenizer.decode(tokenIds: Array(training.dropFirst(inference.count)))
        #expect(answer.contains("A lighthouse at dusk."))
        #expect(training.filter { $0 == Int(Gemma4Processor.imageTokenId) }.count == 280)
    }

    @Test("audio : boa + N + eoa ; media sans tour user = erreur")
    func testAudioAndMissingUser() async throws {
        let tokenizer = try await Gemma4TokenizerLoader().load(from: URL(fileURLWithPath: modelPath!))
        let ids = try Gemma4Processor.multimodalTrainingIds(
            messages: [["role": "user", "content": "What bird is this?"], ["role": "assistant", "content": "A robin."]],
            hasImage: false, audioTokens: 37, tokenizer: tokenizer)
        #expect(ids.filter { $0 == Int(Gemma4Processor.audioTokenId) }.count == 37)
        #expect(ids.contains(Int(Gemma4Processor.boaTokenId)) && ids.contains(Int(Gemma4Processor.eoaTokenId)))
        #expect(throws: Gemma4PipelineError.self) {
            try Gemma4Processor.multimodalTrainingIds(
                messages: [["role": "assistant", "content": "x"]], hasImage: true, audioTokens: nil, tokenizer: tokenizer)
        }
    }
}
