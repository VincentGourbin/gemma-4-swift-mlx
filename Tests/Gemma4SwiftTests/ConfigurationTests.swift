import Testing
import Foundation
@testable import Gemma4Swift

@Suite("Configuration Gemma 4")
struct ConfigurationTests {

    // MARK: - E2B Config

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
        #expect(config.textConfig.hiddenSizePerLayerInput == 256)
        #expect(config.textConfig.enableMoeBlock == false)
        #expect(config.textConfig.moeIntermediateSize == 0)
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

    // MARK: - E4B Config

    let e4bConfigJSON = """
    {
        "model_type": "gemma4",
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 2560,
            "num_hidden_layers": 42,
            "intermediate_size": 10240,
            "num_attention_heads": 8,
            "head_dim": 256,
            "global_head_dim": 512,
            "vocab_size": 262144,
            "num_key_value_heads": 2,
            "num_kv_shared_layers": 18,
            "hidden_size_per_layer_input": 256,
            "vocab_size_per_layer_input": 262144,
            "sliding_window": 512,
            "sliding_window_pattern": 6,
            "max_position_embeddings": 131072,
            "final_logit_softcapping": 30.0,
            "attention_bias": false,
            "attention_k_eq_v": false,
            "use_double_wide_mlp": false,
            "enable_moe_block": false,
            "tie_word_embeddings": true,
            "rope_parameters": {
                "full_attention": { "partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional" },
                "sliding_attention": { "rope_theta": 10000.0, "rope_type": "default" }
            }
        },
        "image_token_id": 258880,
        "audio_token_id": 258881,
        "video_token_id": 258884,
        "vision_soft_tokens_per_image": 280,
        "tie_word_embeddings": true
    }
    """

    @Test("Decodage config E4B")
    func testDecodeE4BConfig() throws {
        let data = e4bConfigJSON.data(using: .utf8)!
        let config = try JSONDecoder().decode(Gemma4Config.self, from: data)

        #expect(config.textConfig.hiddenSize == 2560)
        #expect(config.textConfig.numHiddenLayers == 42)
        #expect(config.textConfig.numKeyValueHeads == 2)
        #expect(config.textConfig.numKvSharedLayers == 18)
        #expect(config.textConfig.firstKvSharedLayerIdx == 24)
        #expect(config.textConfig.hiddenSizePerLayerInput == 256)
        #expect(config.textConfig.useDoubleWideMlp == false)
        #expect(config.textConfig.enableMoeBlock == false)
        #expect(config.textConfig.attentionKEqV == false)
    }

    // MARK: - 26B-A4B Config (MoE)

    let a4bConfigJSON = """
    {
        "model_type": "gemma4",
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 2816,
            "num_hidden_layers": 30,
            "intermediate_size": 2112,
            "num_attention_heads": 16,
            "head_dim": 256,
            "global_head_dim": 512,
            "vocab_size": 262144,
            "num_key_value_heads": 8,
            "num_global_key_value_heads": 2,
            "num_kv_shared_layers": 0,
            "hidden_size_per_layer_input": 0,
            "vocab_size_per_layer_input": 262144,
            "sliding_window": 1024,
            "sliding_window_pattern": 6,
            "max_position_embeddings": 262144,
            "final_logit_softcapping": 30.0,
            "attention_bias": false,
            "attention_k_eq_v": true,
            "use_double_wide_mlp": false,
            "enable_moe_block": true,
            "num_experts": 128,
            "top_k_experts": 8,
            "moe_intermediate_size": 704,
            "tie_word_embeddings": true,
            "rope_parameters": {
                "full_attention": { "partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional" },
                "sliding_attention": { "rope_theta": 10000.0, "rope_type": "default" }
            }
        },
        "audio_config": null,
        "image_token_id": 258880,
        "audio_token_id": 258881,
        "video_token_id": 258884,
        "vision_soft_tokens_per_image": 280,
        "tie_word_embeddings": true
    }
    """

    @Test("Decodage config 26B-A4B (MoE)")
    func testDecodeA4BConfig() throws {
        let data = a4bConfigJSON.data(using: .utf8)!
        let config = try JSONDecoder().decode(Gemma4Config.self, from: data)

        #expect(config.textConfig.hiddenSize == 2816)
        #expect(config.textConfig.numHiddenLayers == 30)
        #expect(config.textConfig.enableMoeBlock == true)
        #expect(config.textConfig.numExperts == 128)
        #expect(config.textConfig.topKExperts == 8)
        #expect(config.textConfig.moeIntermediateSize == 704)
        #expect(config.textConfig.attentionKEqV == true)
        #expect(config.textConfig.numGlobalKeyValueHeads == 2)
        #expect(config.textConfig.numKeyValueHeads == 8)
        #expect(config.textConfig.numKvSharedLayers == 0)
        #expect(config.textConfig.hiddenSizePerLayerInput == 0)
        #expect(config.textConfig.slidingWindow == 1024)
        #expect(config.textConfig.maxPositionEmbeddings == 262144)
        #expect(config.audioConfig == nil)
    }

    // MARK: - 31B Config

    let b31bConfigJSON = """
    {
        "model_type": "gemma4",
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 5376,
            "num_hidden_layers": 60,
            "intermediate_size": 21504,
            "num_attention_heads": 32,
            "head_dim": 256,
            "global_head_dim": 512,
            "vocab_size": 262144,
            "num_key_value_heads": 16,
            "num_global_key_value_heads": 4,
            "num_kv_shared_layers": 0,
            "hidden_size_per_layer_input": 0,
            "vocab_size_per_layer_input": 262144,
            "sliding_window": 1024,
            "sliding_window_pattern": 6,
            "max_position_embeddings": 262144,
            "final_logit_softcapping": 30.0,
            "attention_bias": false,
            "attention_k_eq_v": true,
            "use_double_wide_mlp": false,
            "enable_moe_block": false,
            "tie_word_embeddings": true,
            "rope_parameters": {
                "full_attention": { "partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional" },
                "sliding_attention": { "rope_theta": 10000.0, "rope_type": "default" }
            }
        },
        "audio_config": null,
        "image_token_id": 258880,
        "audio_token_id": 258881,
        "video_token_id": 258884,
        "vision_soft_tokens_per_image": 280,
        "tie_word_embeddings": true
    }
    """

    @Test("Decodage config 31B")
    func testDecode31BConfig() throws {
        let data = b31bConfigJSON.data(using: .utf8)!
        let config = try JSONDecoder().decode(Gemma4Config.self, from: data)

        #expect(config.textConfig.hiddenSize == 5376)
        #expect(config.textConfig.numHiddenLayers == 60)
        #expect(config.textConfig.numAttentionHeads == 32)
        #expect(config.textConfig.numKeyValueHeads == 16)
        #expect(config.textConfig.numGlobalKeyValueHeads == 4)
        #expect(config.textConfig.attentionKEqV == true)
        #expect(config.textConfig.enableMoeBlock == false)
        #expect(config.textConfig.numKvSharedLayers == 0)
        #expect(config.textConfig.hiddenSizePerLayerInput == 0)
        #expect(config.textConfig.slidingWindow == 1024)
        #expect(config.textConfig.maxPositionEmbeddings == 262144)
        #expect(config.textConfig.intermediateSize == 21504)
        #expect(config.audioConfig == nil)
    }

    // Model Registry tests are in ModelRegistryTests.swift
}
