import Testing
import Foundation
import MLX
import MLXNN
import MLXLMCommon
@testable import Gemma4Swift

/// Garde-fou D-01 / D-03 (audit diffusion du 2026-09-28) : la quantification a la
/// volee filtrait `m is Linear || m is Embedding`. Les experts MoE (`SwitchLinear`,
/// Quantizable sans heriter de Linear) restaient en bf16 — 22,8 G parametres sur 25,8
/// pour 26B-A4B et DiffusionGemma. Le routeur, lui, doit rester en 8 bits comme dans
/// les packs mlx-community.
@Suite("Quantification a la volee des MoE")
struct MoEQuantizationTests {

    static let moeConfig = """
    {
        "model_type": "gemma4_text", "hidden_size": 64, "num_hidden_layers": 2,
        "intermediate_size": 128, "num_attention_heads": 2, "head_dim": 32,
        "global_head_dim": 32, "rms_norm_eps": 1e-6, "vocab_size": 128,
        "num_key_value_heads": 1, "num_kv_shared_layers": 0, "sliding_window": 8,
        "sliding_window_pattern": 2, "max_position_embeddings": 256,
        "attention_bias": false, "use_double_wide_mlp": false,
        "enable_moe_block": true, "num_experts": 4, "top_k_experts": 2, "moe_intermediate_size": 64,
        "tie_word_embeddings": true,
        "rope_parameters": {
            "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
            "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}
        },
        "layer_types": ["sliding_attention", "full_attention"]
    }
    """

    @Test("4 bits : experts MoE quantifies, routeur en 8 bits, forward intact")
    func testExpertsAndRouter() throws {
        let config = try JSONDecoder().decode(Gemma4TextConfig.self, from: Data(Self.moeConfig.utf8))
        MLXRandom.seed(3)
        let model = Gemma4LLMModel(config: config)
        let before = model(MLXArray([Int32(1), 5, 9])[.newAxis], cache: nil)

        _ = Gemma4OnTheFlyQuantization.apply(to: model, bits: 4)

        let modules = model.namedModules()
        let experts = modules.filter { $0.1 is SwitchLinear }
        #expect(!experts.isEmpty, "le modele de test doit avoir des experts")
        for (path, module) in experts {
            #expect(module is QuantizedSwitchLinear, "expert non quantifie : \(path)")
        }
        let routers = modules.filter { $0.0.hasSuffix("router.proj") }
        #expect(!routers.isEmpty)
        for (path, module) in routers {
            let quantized = try #require(module as? QuantizedLinear, "routeur non quantifie : \(path)")
            #expect(quantized.bits == 8, "routeur en \(quantized.bits) bits : \(path)")
        }

        let after = model(MLXArray([Int32(1), 5, 9])[.newAxis], cache: nil)
        eval(before, after)
        #expect(after.shape == before.shape)
        #expect(!any(isNaN(after)).item(Bool.self))
    }
}
