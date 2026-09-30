import Testing
import Foundation
import MLX
import MLXLMCommon
@testable import Gemma4Swift

/// K-19 : un instantane de caches (copy) prolonge d'un suffixe donne les memes logits
/// qu'un forward complet, y compris au-dela de la fenetre glissante (RotatingKVCache).
/// fp32 sur CPU : un ecart ici est un bug, pas du bruit bf16.
@Suite("Instantane de conversation : parite des caches")
struct ConversationSnapshotParityTests {

    static let config = """
    {
        "model_type": "gemma4_text", "hidden_size": 64, "num_hidden_layers": 4,
        "intermediate_size": 128, "num_attention_heads": 2, "head_dim": 32,
        "global_head_dim": 32, "rms_norm_eps": 1e-6, "vocab_size": 128,
        "num_key_value_heads": 1, "num_kv_shared_layers": 2, "sliding_window": 8,
        "sliding_window_pattern": 2, "max_position_embeddings": 256,
        "attention_bias": false, "use_double_wide_mlp": false, "enable_moe_block": false,
        "tie_word_embeddings": true,
        "rope_parameters": {
            "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
            "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}
        },
        "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention"]
    }
    """

    @Test("prefixe 30 + suffixe 5 (fenetre 8) = forward complet de 35", arguments: [1, 5])
    func testSnapshotExtension(suffixLength: Int) throws {
        try Device.withDefaultDevice(.cpu) {
            let config = try JSONDecoder().decode(Gemma4TextConfig.self, from: Data(Self.config.utf8))
            MLXRandom.seed(9)
            let model = Gemma4LLMModel(config: config)
            let ids = (0 ..< (30 + suffixLength)).map { Int32(($0 * 37 + 11) % 128) }

            let fullCache = model.newCache(parameters: nil)
            let full = model(MLXArray(ids)[.newAxis], cache: fullCache)[0..., -1, 0...]

            let cache = model.newCache(parameters: nil)
            _ = model(MLXArray(Array(ids[..<30]))[.newAxis], cache: cache)
            eval(cache.flatMap(\.state))
            let snapshot = cache.map { $0.copy() }
            // Le cache d'origine continue sa vie (generation) : l'instantane ne doit pas bouger.
            _ = model(MLXArray([Int32(3), 4, 5, 6, 7, 8, 9, 10, 11])[.newAxis], cache: cache)
            let extended = model(MLXArray(Array(ids[30...]))[.newAxis], cache: snapshot)[0..., -1, 0...]
            eval(full, extended)
            let err = (abs(full - extended).max() / abs(full).max()).item(Float.self)
            #expect(err < 1e-4, "ecart relatif \(err)")
        }
    }
}
