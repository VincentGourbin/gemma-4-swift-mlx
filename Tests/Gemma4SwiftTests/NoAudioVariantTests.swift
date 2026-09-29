import Testing
import Foundation
import MLX
import MLXLMCommon
@testable import Gemma4Swift

/// K-16 : variante sans tour audio (0,61 Go d'E2B, en bf16 meme dans les packs 4 bits) et
/// garde contre un audio ignore en silence.
@Suite("Variante sans audio")
struct NoAudioVariantTests {

    @Test("configDataWithoutAudio retire audio_config et garde le reste")
    func testStripAudioConfig() throws {
        let data = Data(ConfigurationTests().a4bConfigJSON.replacingOccurrences(
            of: "\"model_type\": \"gemma4\",", with: "\"model_type\": \"gemma4\", \"audio_config\": {\"bogus\": 1},").utf8)
        let stripped = try Gemma4Registration.configDataWithoutAudio(data)
        let object = try #require(try JSONSerialization.jsonObject(with: stripped) as? [String: Any])
        #expect(object["audio_config"] == nil)
        let config = try JSONDecoder().decode(Gemma4Config.self, from: stripped)
        #expect(config.audioConfig == nil)
        #expect(config.textConfig.hiddenSize == 2816)
    }

    @Test("audio envoye a un modele sans tour audio : le prefill echoue au lieu de l'ignorer")
    func testAudioWithoutTowerThrows() throws {
        let config = try JSONDecoder().decode(Gemma4Config.self, from: Data(ConfigurationTests().a4bConfigJSON.utf8))
        let model = Gemma4MultimodalLLMModel(config: config)
        #expect(model.audioTower == nil)
        model.pendingAudioFeatures = MLXArray.zeros([1, 8, 128])
        #expect(throws: Gemma4PipelineError.self) {
            _ = try model.prepare(LMInput(tokens: MLXArray([Int32(2), 3])), cache: [])
        }
        #expect(model.pendingAudioFeatures == nil, "l'audio refuse ne reste pas en attente")
    }
}
