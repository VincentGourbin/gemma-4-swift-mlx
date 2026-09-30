import Testing
import Foundation
import MLX
import MLXLMCommon
@testable import Gemma4Swift

/// Garde-fou P-02 (audit du 2026-09-27) : le prefill multimodal ne doit pas
/// promouvoir le graphe en fp32. `embedScale` multiplie par une constante du
/// dtype des embeddings ; une constante fp32 promouvait tout le prefill, puis le
/// cache KV (alloue au dtype des K/V ecrits) et donc tout le decodage.
@Suite("Dtype du prefill multimodal")
struct MultimodalDtypeTests {

    /// Config minimale ; ids media dans le vocab (128) pour que la lookup
    /// d'embedding reste dans les bornes.
    static let configJSON = """
    {
        "model_type": "gemma4",
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "intermediate_size": 64,
            "num_attention_heads": 2,
            "head_dim": 16,
            "global_head_dim": 16,
            "rms_norm_eps": 1e-6,
            "vocab_size": 128,
            "num_key_value_heads": 1,
            "num_kv_shared_layers": 0,
            "sliding_window": 8,
            "sliding_window_pattern": 2,
            "max_position_embeddings": 128,
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
            "layer_types": ["sliding_attention", "full_attention"]
        },
        "image_token_id": 100,
        "audio_token_id": 101,
        "video_token_id": 102,
        "vision_soft_tokens_per_image": 4,
        "tie_word_embeddings": true
    }
    """

    @Test("prefill avec image en bf16 : le cache KV reste en bf16")
    func testImagePrefillKeepsBf16() throws {
        let config = try JSONDecoder().decode(Gemma4Config.self, from: Data(Self.configJSON.utf8))
        let model = Gemma4MultimodalLLMModel(config: config)
        model.languageModel.apply { $0.dtype.isFloatingPoint ? $0.asType(.bfloat16) : $0 }

        // 2 jetons texte + 4 jetons image (embeddings pre-calcules, chemin
        // `pendingImageEmbeddings`) + 1 jeton texte.
        let ids: [Int32] = [2, 5, 100, 100, 100, 100, 7]
        let inputs = MLXArray(ids).reshaped(1, ids.count)
        model.pendingImageEmbeddings = MLXArray.zeros([1, 4, 32], dtype: .bfloat16)

        let cache = model.newCache(parameters: nil)
        let logits = model(inputs, cache: cache)
        eval(logits)

        let keys = try #require(cache.first?.state.first)
        #expect(keys.dtype == .bfloat16, "cache KV en \(keys.dtype) apres un prefill image bf16")
        #expect(model.pendingImageEmbeddings == nil)
    }
}
