import Testing
import Foundation
import MLX
import MLXLMCommon
@testable import Gemma4Swift

/// Garde-fou P-05 (audit du 2026-09-27) : avec `GenerateParameters.kvBits`,
/// mlx-swift-lm remplace le cache de la couche source par un `QuantizedKVCache`.
/// Les couches a KV partage (E2B/E4B) lisaient alors `state[0]` / `state[1]`
/// comme (K, V), c'est-a-dire les poids packes et leurs echelles.
@Suite("KV quantifie et couches partagees")
struct QuantizedSharedKVTests {

    /// 4 couches dont 2 a KV partage (couches 2 et 3 relisent les caches des
    /// couches 0 et 1). head_dim 64 : la quantification du KV se fait par groupes de 64.
    static let configJSON = """
    {
        "model_type": "gemma4_text",
        "hidden_size": 64,
        "num_hidden_layers": 4,
        "intermediate_size": 128,
        "num_attention_heads": 2,
        "head_dim": 64,
        "global_head_dim": 64,
        "rms_norm_eps": 1e-6,
        "vocab_size": 128,
        "num_key_value_heads": 1,
        "num_kv_shared_layers": 2,
        "sliding_window": 16,
        "sliding_window_pattern": 2,
        "max_position_embeddings": 256,
        "attention_bias": false,
        "use_double_wide_mlp": false,
        "enable_moe_block": false,
        "tie_word_embeddings": true,
        "rope_parameters": {
            "full_attention": {
                "partial_rotary_factor": 0.25,
                "rope_theta": 1000000.0,
                "rope_type": "proportional"
            },
            "sliding_attention": {
                "rope_theta": 10000.0,
                "rope_type": "default"
            }
        },
        "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention"]
    }
    """

    @Test("kvBits 8 : le decodage des couches partagees suit le cache non quantifie")
    func testQuantizedCacheMatchesReference() throws {
        let config = try JSONDecoder().decode(Gemma4TextConfig.self, from: Data(Self.configJSON.utf8))
        let model = Gemma4LLMModel(config: config)

        let prompt = MLXArray((0 ..< 12).map { Int32(($0 * 7 + 3) % 128) }).reshaped(1, 12)
        let next = MLXArray([Int32(42)]).reshaped(1, 1)

        let reference = model.newCache(parameters: nil)
        _ = model(prompt, cache: reference)
        let referenceLogits = model(next, cache: reference)

        var quantized = model.newCache(parameters: nil)
        _ = model(prompt, cache: quantized)
        maybeQuantizeKVCache(cache: &quantized, kvBits: 8, quantizedKVStart: 0)
        #expect(quantized.contains { $0 is QuantizedKVCache }, "aucun cache quantifie : le test ne prouve rien")
        let quantizedLogits = model(next, cache: quantized)

        eval(referenceLogits, quantizedLogits)
        let error = abs(referenceLogits - quantizedLogits).max().item(Float.self)
        let magnitude = abs(referenceLogits).max().item(Float.self)
        #expect(error <= 0.05 * magnitude, "ecart \(error) pour une amplitude \(magnitude)")
    }
}
