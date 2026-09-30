import Testing
import Foundation
import MLX
import MLXNN
import MLXLMCommon
@testable import Gemma4Swift

/// K-33 : un LoRA sur un modele MoE (26B-A4B) mourait au premier pas, `gatherMM` refusant
/// une VJP par rapport aux indices du routeur. Les indices sont maintenant hors gradient.
@Suite("Entrainement MoE", .serialized)
struct MoETrainingTests {

    static let config = """
    {
        "model_type": "gemma4_text", "hidden_size": 64, "num_hidden_layers": 2,
        "intermediate_size": 128, "num_attention_heads": 2, "head_dim": 32,
        "global_head_dim": 32, "rms_norm_eps": 1e-6, "vocab_size": 128,
        "num_key_value_heads": 1, "num_kv_shared_layers": 0, "sliding_window": 8,
        "sliding_window_pattern": 2, "max_position_embeddings": 256,
        "hidden_size_per_layer_input": 0,
        "attention_bias": false, "use_double_wide_mlp": false, "enable_moe_block": true,
        "num_experts": 4, "top_k_experts": 2, "moe_intermediate_size": 32,
        "tie_word_embeddings": true,
        "rope_parameters": {
            "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
            "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}
        },
        "layer_types": ["sliding_attention", "full_attention"]
    }
    """

    @Test("valueAndGrad LoRA a travers les experts : pas d'erreur, gradients finis")
    func testLoRAGradientThroughExperts() throws {
        try Device.withDefaultDevice(.cpu) {
            let config = try JSONDecoder().decode(Gemma4TextConfig.self, from: Data(Self.config.utf8))
            MLXRandom.seed(5)
            let model = Gemma4LLMModel(config: config)
            _ = try LoRAContainer.from(model: model, configuration: Gemma4LoRADefaults.configuration(numLayers: 2))
            let batch = MLXArray((0 ..< 16).map { Int32(($0 * 5 + 1) % 128) }).reshaped(2, 8)
            let lengths = MLXArray([Int32(2), 8, 3, 8]).reshaped(2, 2)
            let (values, grads) = valueAndGrad(model: model) { m, a -> [MLXArray] in
                let (ce, n) = trainingLoss(model: m, batch: a[0], lengths: a[1])
                return [ce, n]
            }(model, [batch, lengths])
            #expect(values[0].item(Float.self).isFinite)
            let flat = grads.flattened()
            #expect(!flat.isEmpty)
            for (key, g) in flat {
                #expect(isNaN(g).any().item(Bool.self) == false, "\(key)")
            }
        }
    }
}
