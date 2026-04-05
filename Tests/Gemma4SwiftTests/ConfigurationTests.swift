import Testing
import Foundation
@testable import Gemma4Swift

@Suite("Configuration Gemma 4")
struct ConfigurationTests {

    let e2bConfigJSON = """
    {
        "model_type": "gemma4",
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 1536,
            "num_hidden_layers": 35,
            "intermediate_size": 6144,
            "num_attention_heads": 8,
            "head_dim": 256,
            "global_head_dim": 512,
            "rms_norm_eps": 1e-6,
            "vocab_size": 262144,
            "num_key_value_heads": 1,
            "num_kv_shared_layers": 20,
            "hidden_size_per_layer_input": 256,
            "vocab_size_per_layer_input": 262144,
            "sliding_window": 512,
            "sliding_window_pattern": 5,
            "max_position_embeddings": 131072,
            "final_logit_softcapping": 30.0,
            "attention_bias": false,
            "attention_k_eq_v": false,
            "use_double_wide_mlp": true,
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
            "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention"]
        },
        "image_token_id": 258880,
        "audio_token_id": 258881,
        "video_token_id": 258884,
        "vision_soft_tokens_per_image": 280,
        "tie_word_embeddings": true
    }
    """

    @Test("Decodage config E2B")
    func testDecodeE2BConfig() throws {
        let data = e2bConfigJSON.data(using: .utf8)!
        let config = try JSONDecoder().decode(Gemma4Config.self, from: data)

        #expect(config.modelType == "gemma4")
        #expect(config.textConfig.hiddenSize == 1536)
        #expect(config.textConfig.numHiddenLayers == 35)
        #expect(config.textConfig.globalHeadDim == 512)
        #expect(config.textConfig.headDim == 256)
        #expect(config.textConfig.useDoubleWideMlp == true)
        #expect(config.textConfig.attentionKEqV == false)
        #expect(config.textConfig.numKvSharedLayers == 20)
        #expect(config.textConfig.firstKvSharedLayerIdx == 15)
        #expect(config.imageTokenId == 258880)
        #expect(config.videoTokenId == 258884)
    }

    @Test("RoPE parameters")
    func testRoPEParameters() throws {
        let data = e2bConfigJSON.data(using: .utf8)!
        let config = try JSONDecoder().decode(Gemma4Config.self, from: data)
        let textConfig = config.textConfig

        #expect(textConfig.ropeTheta(forLayerType: "sliding_attention") == 10000.0)
        #expect(textConfig.ropeTheta(forLayerType: "full_attention") == 1000000.0)
        #expect(textConfig.ropeType(forLayerType: "full_attention") == "proportional")
        #expect(textConfig.ropeType(forLayerType: "sliding_attention") == "default")
        #expect(textConfig.fullAttentionPartialRotaryFactor == 0.25)
    }

    @Test("Layer types resolus")
    func testResolvedLayerTypes() throws {
        let data = e2bConfigJSON.data(using: .utf8)!
        let config = try JSONDecoder().decode(Gemma4Config.self, from: data)
        let layerTypes = config.textConfig.resolvedLayerTypes

        #expect(layerTypes.count == 5) // truncated config only has 5
        #expect(layerTypes[0] == "sliding_attention")
        #expect(layerTypes[4] == "full_attention")
    }
}
