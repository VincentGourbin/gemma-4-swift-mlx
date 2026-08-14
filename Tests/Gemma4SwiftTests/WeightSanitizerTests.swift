import Testing
import Foundation
import MLX
@testable import Gemma4Swift

@Suite("Weight Sanitizer")
struct WeightSanitizerTests {

    @Test("Detection format Google vs MLX")
    func testFormatDetection() {
        let googleWeights: [String: MLXArray] = [
            "model.language_model.layers.0.self_attn.q_proj.weight": MLXArray.zeros([4, 4]),
        ]
        #expect(WeightSanitizer.isGoogleFormat(googleWeights) == true)

        let mlxWeights: [String: MLXArray] = [
            "language_model.model.layers.0.self_attn.q_proj.weight": MLXArray.zeros([4, 4]),
        ]
        #expect(WeightSanitizer.isGoogleFormat(mlxWeights) == false)
    }

    @Test("Strip prefixe model.")
    func testStripModelPrefix() {
        let weights: [String: MLXArray] = [
            "model.language_model.embed_tokens.weight": MLXArray.zeros([4, 4]),
        ]
        let sanitized = WeightSanitizer.sanitize(weights: weights)
        #expect(sanitized["language_model.model.embed_tokens.weight"] != nil)
        #expect(sanitized["model.language_model.embed_tokens.weight"] == nil)
    }

    @Test("Remap language_model paths")
    func testLanguageModelRemap() {
        let weights: [String: MLXArray] = [
            "language_model.layers.0.self_attn.q_proj.weight": MLXArray.zeros([4, 4]),
        ]
        let sanitized = WeightSanitizer.sanitize(weights: weights)
        #expect(sanitized["language_model.model.layers.0.self_attn.q_proj.weight"] != nil)
    }

    @Test("Ne pas re-mapper les poids deja au format MLX")
    func testNoDoubleRemap() {
        let weights: [String: MLXArray] = [
            "language_model.model.layers.0.mlp.gate_proj.weight": MLXArray.zeros([4, 4]),
        ]
        let sanitized = WeightSanitizer.sanitize(weights: weights)
        #expect(sanitized["language_model.model.layers.0.mlp.gate_proj.weight"] != nil)
        // Pas de double "model.model."
        #expect(sanitized["language_model.model.model.layers.0.mlp.gate_proj.weight"] == nil)
    }

    @Test("Skip rotary_emb")
    func testSkipRotaryEmb() {
        let weights: [String: MLXArray] = [
            "language_model.model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.zeros([4]),
            "language_model.model.layers.0.self_attn.q_proj.weight": MLXArray.zeros([4, 4]),
        ]
        let sanitized = WeightSanitizer.sanitize(weights: weights)
        #expect(sanitized.count == 1)
        #expect(sanitized.keys.first?.contains("rotary_emb") != true)
    }

    @Test("Skip vision/audio si non demandes")
    func testSkipModalityWeights() {
        let weights: [String: MLXArray] = [
            "vision_tower.encoder.layers.0.weight": MLXArray.zeros([4, 4]),
            "audio_tower.encoder.layers.0.weight": MLXArray.zeros([4, 4]),
            "language_model.model.norm.weight": MLXArray.zeros([4]),
        ]
        let sanitized = WeightSanitizer.sanitize(weights: weights, hasVision: false, hasAudio: false)
        #expect(sanitized.count == 1)
        #expect(sanitized["language_model.model.norm.weight"] != nil)
    }

    @Test("Conserver vision/audio si demandes")
    func testKeepModalityWeights() {
        let weights: [String: MLXArray] = [
            "vision_tower.encoder.layers.0.weight": MLXArray.zeros([4, 4]),
            "audio_tower.encoder.layers.0.weight": MLXArray.zeros([4, 4]),
        ]
        let sanitized = WeightSanitizer.sanitize(weights: weights, hasVision: true, hasAudio: true)
        #expect(sanitized.count == 2)
    }

    @Test("MoE experts.down_proj remap")
    func testMoEDownProjRemap() {
        let weights: [String: MLXArray] = [
            "language_model.model.layers.0.experts.down_proj": MLXArray.zeros([4, 4]),
        ]
        let sanitized = WeightSanitizer.sanitize(weights: weights)
        #expect(sanitized["language_model.model.layers.0.experts.switch_glu.down_proj.weight"] != nil)
    }

    @Test("MoE experts.gate_up_proj split")
    func testMoEGateUpProjSplit() {
        // gate_up_proj: [num_experts, 2*hidden, input] -> split en gate et up
        let weights: [String: MLXArray] = [
            "language_model.model.layers.0.experts.gate_up_proj": MLXArray.zeros([4, 8, 4]),
        ]
        let sanitized = WeightSanitizer.sanitize(weights: weights)
        #expect(sanitized["language_model.model.layers.0.experts.switch_glu.gate_proj.weight"] != nil)
        #expect(sanitized["language_model.model.layers.0.experts.switch_glu.up_proj.weight"] != nil)
        // L'original ne doit plus etre present
        #expect(sanitized["language_model.model.layers.0.experts.gate_up_proj"] == nil)
    }

    @Test("KV-shared: ne supprime que les K/V du LLM, pas ceux des tours vision/audio")
    func testKvSharedDropIsLLMOnly() {
        let weights: [String: MLXArray] = [
            // LLM couche 20 (>= firstKvSharedLayerIdx 15) : K/V morts, a supprimer
            "language_model.model.layers.20.self_attn.k_proj.weight": MLXArray.zeros([4, 4]),
            "language_model.model.layers.20.self_attn.v_norm.weight": MLXArray.zeros([4]),
            // Vision couche 15 : K/V bien vivants, a conserver
            "vision_tower.encoder.layers.15.self_attn.k_proj.linear.weight": MLXArray.zeros([4, 4]),
            "vision_tower.encoder.layers.15.self_attn.k_norm.weight": MLXArray.zeros([4]),
            "audio_tower.encoder.layers.15.self_attn.v_proj.linear.weight": MLXArray.zeros([4, 4]),
        ]
        let sanitized = WeightSanitizer.sanitize(
            weights: weights, hasVision: true, hasAudio: true,
            firstKvSharedLayerIdx: 15)
        #expect(sanitized["language_model.model.layers.20.self_attn.k_proj.weight"] == nil)
        #expect(sanitized["language_model.model.layers.20.self_attn.v_norm.weight"] == nil)
        #expect(sanitized["vision_tower.encoder.layers.15.self_attn.k_proj.linear.weight"] != nil)
        #expect(sanitized["vision_tower.encoder.layers.15.self_attn.k_norm.weight"] != nil)
        #expect(sanitized["audio_tower.encoder.layers.15.self_attn.v_proj.linear.weight"] != nil)
    }
}
