import Testing
import Foundation
import MLX
import MLXNN
import MLXLMCommon
@testable import Gemma4Swift

/// K-32 : gradient checkpointing par couche = meme perte et memes gradients LoRA, avec
/// couches KV-partagees et entrees par couche (chemins E2B).
@Suite("Gradient checkpointing", .serialized)
struct GradientCheckpointingTests {

    static let config = """
    {
        "model_type": "gemma4_text", "hidden_size": 64, "num_hidden_layers": 4,
        "intermediate_size": 128, "num_attention_heads": 2, "head_dim": 32,
        "global_head_dim": 32, "rms_norm_eps": 1e-6, "vocab_size": 128,
        "num_key_value_heads": 1, "num_kv_shared_layers": 2, "sliding_window": 8,
        "sliding_window_pattern": 2, "max_position_embeddings": 256,
        "hidden_size_per_layer_input": 16, "vocab_size_per_layer_input": 128,
        "attention_bias": false, "use_double_wide_mlp": false, "enable_moe_block": false,
        "tie_word_embeddings": true,
        "rope_parameters": {
            "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
            "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}
        },
        "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention"]
    }
    """

    private func model() throws -> Gemma4LLMModel {
        let config = try JSONDecoder().decode(Gemma4TextConfig.self, from: Data(Self.config.utf8))
        MLXRandom.seed(21)
        let model = Gemma4LLMModel(config: config)
        MLXRandom.seed(22)
        _ = try LoRAContainer.from(model: model, configuration: Gemma4LoRADefaults.configuration(numLayers: 4))
        // lora_b vaut 0 a l'init : le perturber pour que tous les gradients soient non nuls.
        let perturbed = model.trainableParameters().flattened().map { ($0.0, $0.1 + 0.01) }
        model.update(parameters: ModuleParameters.unflattened(perturbed))
        return model
    }

    @Test("meme perte, memes gradients, avec et sans checkpointing")
    func testEquivalence() throws {
        try Device.withDefaultDevice(.cpu) {
            let batch = MLXArray((0 ..< 24).map { Int32(($0 * 7 + 3) % 128) }).reshaped(2, 12)
            let lengths = MLXArray([Int32(4), 12, 6, 12]).reshaped(2, 2)
            func run(_ checkpointing: Bool) throws -> ([MLXArray], [String: MLXArray]) {
                let m = try model()
                m.languageModel.model.gradientCheckpointing = checkpointing
                let (values, grads) = valueAndGrad(model: m) { model, a -> [MLXArray] in
                    let (ce, n) = trainingLoss(model: model, batch: a[0], lengths: a[1])
                    return [ce, n]
                }(m, [batch, lengths])
                return (values, Dictionary(grads.flattened(), uniquingKeysWith: { x, _ in x }))
            }
            let (plain, g1) = try run(false)
            let (checked, g2) = try run(true)
            #expect(abs(plain[0] - checked[0]).item(Float.self) < 1e-5)
            #expect(g1.count == g2.count && !g1.isEmpty)
            for (key, value) in g1 {
                #expect(allClose(value, g2[key]!, atol: 1e-5).item(Bool.self), "\(key)")
            }
        }
    }
}
