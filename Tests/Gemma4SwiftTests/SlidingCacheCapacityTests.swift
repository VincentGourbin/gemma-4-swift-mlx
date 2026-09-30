import Testing
import Foundation
import MLX
import MLXLMCommon
@testable import Gemma4Swift

/// Garde-fou P-04 (audit du 2026-09-27) : le decodage speculatif retire les
/// brouillons rejetes avec trimPromptCache, qui ne fait rien si un cache n'est plus
/// « trimmable ». Un RotatingKVCache a la taille de la fenetre cesse de l'etre des
/// qu'il a tourne : le MTP laissait alors des jetons faux dans le KV.
@Suite("Capacite des caches glissants")
struct SlidingCacheCapacityTests {

    /// Fenetre de 8 jetons, 2 couches (glissante + pleine).
    static let configJSON = """
    {
        "model_type": "gemma4_text", "hidden_size": 32, "num_hidden_layers": 2,
        "intermediate_size": 64, "num_attention_heads": 2, "head_dim": 16,
        "global_head_dim": 16, "rms_norm_eps": 1e-6, "vocab_size": 128,
        "num_key_value_heads": 1, "num_kv_shared_layers": 0, "sliding_window": 8,
        "sliding_window_pattern": 2, "max_position_embeddings": 256,
        "attention_bias": false, "use_double_wide_mlp": false, "enable_moe_block": false,
        "tie_word_embeddings": true,
        "rope_parameters": {
            "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
            "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}
        },
        "layer_types": ["sliding_attention", "full_attention"]
    }
    """

    private func run(slidingCapacity: Int?) throws -> [any KVCache] {
        let config = try JSONDecoder().decode(Gemma4TextConfig.self, from: Data(Self.configJSON.utf8))
        MLXRandom.seed(0)
        let model = Gemma4LLMModel(config: config)
        let cache = model.languageModel.makeCache(slidingCapacity: slidingCapacity)
        // 20 jetons : plus du double de la fenetre.
        let logits = model(MLXArray((0 ..< 20).map { Int32($0 + 1) }).reshaped(1, 20), cache: cache)
        eval(logits)
        return cache
    }

    @Test("cache a la taille de la fenetre : plus trimmable apres avoir tourne")
    func testWindowSizedCacheStopsBeingTrimmable() throws {
        let cache = try run(slidingCapacity: nil)
        #expect(!canTrimPromptCache(cache))
    }

    @Test("capacite du run : le retrait des brouillons reste possible")
    func testRunSizedCacheStaysTrimmable() throws {
        let cache = try run(slidingCapacity: 64)
        #expect(canTrimPromptCache(cache))
        #expect(trimPromptCache(cache, numTokens: 3) == 3)
        #expect(cache.allSatisfy { $0.offset == 17 })
    }
}
