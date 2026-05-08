import Testing
import Foundation
import MLX
@testable import Gemma4Swift

@Suite("Configuration & sanitizer Gemma 4 Assistant (MTP drafter)")
struct Gemma4AssistantConfigTests {

    /// Capture verbatim de google/gemma-4-E2B-it-assistant/config.json (mai 2026)
    let assistantConfigJSON = """
    {
      "architectures": ["Gemma4AssistantForCausalLM"],
      "audio_token_id": 258881,
      "backbone_hidden_size": 1536,
      "boa_token_id": 256000,
      "boi_token_id": 255999,
      "centroid_intermediate_top_k": 32,
      "dtype": "bfloat16",
      "eoa_token_id": 258883,
      "eoi_token_id": 258882,
      "image_token_id": 258880,
      "model_type": "gemma4_assistant",
      "num_centroids": 2048,
      "text_config": {
        "attention_bias": false,
        "attention_dropout": 0.0,
        "attention_k_eq_v": false,
        "bos_token_id": 2,
        "dtype": "bfloat16",
        "enable_moe_block": false,
        "eos_token_id": 1,
        "final_logit_softcapping": null,
        "global_head_dim": 512,
        "head_dim": 256,
        "hidden_activation": "gelu_pytorch_tanh",
        "hidden_size": 256,
        "hidden_size_per_layer_input": 0,
        "intermediate_size": 2048,
        "layer_types": [
          "sliding_attention",
          "sliding_attention",
          "sliding_attention",
          "full_attention"
        ],
        "max_position_embeddings": 131072,
        "model_type": "gemma4_text",
        "num_attention_heads": 4,
        "num_hidden_layers": 4,
        "num_key_value_heads": 1,
        "num_kv_shared_layers": 4,
        "rms_norm_eps": 1e-06,
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
        "sliding_window": 512,
        "tie_word_embeddings": true,
        "use_double_wide_mlp": false,
        "vocab_size": 262144,
        "vocab_size_per_layer_input": 0
      },
      "tie_word_embeddings": true,
      "transformers_version": "5.7.0.dev0",
      "use_ordered_embeddings": true
    }
    """

    @Test("Decodage du config.json Assistant E2B")
    func testDecodeAssistantConfig() throws {
        let data = assistantConfigJSON.data(using: .utf8)!
        let config = try JSONDecoder().decode(Gemma4AssistantConfig.self, from: data)

        // Champs top-level MTP
        #expect(config.modelType == "gemma4_assistant")
        #expect(config.backboneHiddenSize == 1536)
        #expect(config.numCentroids == 2048)
        #expect(config.centroidIntermediateTopK == 32)
        #expect(config.useOrderedEmbeddings == true)
        #expect(config.tieWordEmbeddings == true)

        // Tokens speciaux
        #expect(config.imageTokenId == 258880)
        #expect(config.audioTokenId == 258881)
        #expect(config.boiTokenId == 255999)
        #expect(config.eoiTokenId == 258882)
        #expect(config.boaTokenId == 256000)
        #expect(config.eoaTokenId == 258883)

        // Inner text config — drafter 4-layer / hidden 256 / kv-shared partout
        let t = config.textConfig
        #expect(t.modelType == "gemma4_text")
        #expect(t.numHiddenLayers == 4)
        #expect(t.hiddenSize == 256)
        #expect(t.intermediateSize == 2048)
        #expect(t.numAttentionHeads == 4)
        #expect(t.numKeyValueHeads == 1)
        #expect(t.headDim == 256)
        #expect(t.globalHeadDim == 512)
        #expect(t.numKvSharedLayers == 4)
        #expect(t.firstKvSharedLayerIdx == 0)  // toutes les couches sont kv-shared
        #expect(t.vocabSize == 262144)
        #expect(t.slidingWindow == 512)
        #expect(t.hiddenSizePerLayerInput == 0)
        #expect(t.vocabSizePerLayerInput == 0)
        #expect(t.tieWordEmbeddings == true)
        #expect(t.enableMoeBlock == false)
        #expect(t.useDoubleWideMlp == false)
        #expect(t.layerTypes == ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"])
    }

    @Test("RoPE parameters du drafter")
    func testDrafterRoPE() throws {
        let data = assistantConfigJSON.data(using: .utf8)!
        let config = try JSONDecoder().decode(Gemma4AssistantConfig.self, from: data)

        let t = config.textConfig
        #expect(t.ropeTheta(forLayerType: "full_attention") == 1000000.0)
        #expect(t.ropeTheta(forLayerType: "sliding_attention") == 10000.0)
        #expect(t.ropeType(forLayerType: "full_attention") == "proportional")
        #expect(t.ropeType(forLayerType: "sliding_attention") == "default")
        #expect(t.fullAttentionPartialRotaryFactor == 0.25)
    }

    @Test("Resolved layer types — pattern alignement")
    func testResolvedLayerTypes() throws {
        let data = assistantConfigJSON.data(using: .utf8)!
        let config = try JSONDecoder().decode(Gemma4AssistantConfig.self, from: data)

        let resolved = config.textConfig.resolvedLayerTypes
        #expect(resolved.count == 4)
        #expect(resolved[0] == "sliding_attention")
        #expect(resolved[3] == "full_attention")
    }

    // MARK: - Sanitizer

    @Test("Sanitizer convertit token_ordering en Int32")
    func testTokenOrderingInt32() {
        let weights: [String: MLXArray] = [
            "masked_embedding.token_ordering": MLXArray.zeros([2048, 128], type: Float32.self),
            "masked_embedding.centroids.weight": MLXArray.zeros([2048, 256]),
        ]
        let sanitized = Gemma4AssistantWeightSanitizer.sanitize(
            weights: weights,
            tieWordEmbeddings: true
        )
        #expect(sanitized["masked_embedding.token_ordering"] != nil)
        #expect(sanitized["masked_embedding.token_ordering"]?.dtype == .int32)
        #expect(sanitized["masked_embedding.centroids.weight"]?.dtype != .int32)
    }

    @Test("Sanitizer skip lm_head si tie_word_embeddings=true")
    func testSkipLmHeadWhenTied() {
        let weights: [String: MLXArray] = [
            "lm_head.weight": MLXArray.zeros([262144, 256]),
            "model.embed_tokens.weight": MLXArray.zeros([262144, 256]),
        ]
        let sanitized = Gemma4AssistantWeightSanitizer.sanitize(
            weights: weights,
            tieWordEmbeddings: true
        )
        #expect(sanitized["lm_head.weight"] == nil)
        #expect(sanitized["model.embed_tokens.weight"] != nil)
    }

    @Test("Sanitizer garde lm_head si tie_word_embeddings=false")
    func testKeepLmHeadWhenUntied() {
        let weights: [String: MLXArray] = [
            "lm_head.weight": MLXArray.zeros([262144, 256]),
        ]
        let sanitized = Gemma4AssistantWeightSanitizer.sanitize(
            weights: weights,
            tieWordEmbeddings: false
        )
        #expect(sanitized["lm_head.weight"] != nil)
    }

    @Test("Sanitizer skip rotary_emb")
    func testSkipRotaryEmb() {
        let weights: [String: MLXArray] = [
            "model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.zeros([4]),
            "model.layers.0.self_attn.q_proj.weight": MLXArray.zeros([4, 4]),
        ]
        let sanitized = Gemma4AssistantWeightSanitizer.sanitize(
            weights: weights,
            tieWordEmbeddings: true
        )
        #expect(sanitized["model.layers.0.self_attn.rotary_emb.inv_freq"] == nil)
        #expect(sanitized["model.layers.0.self_attn.q_proj.weight"] != nil)
    }

    @Test("Sanitizer preserve les cles standard inchangees")
    func testStandardKeysUnchanged() {
        let weights: [String: MLXArray] = [
            "model.embed_tokens.weight": MLXArray.zeros([262144, 256]),
            "model.layers.0.self_attn.q_proj.weight": MLXArray.zeros([1024, 256]),
            "model.layers.0.mlp.gate_proj.weight": MLXArray.zeros([2048, 256]),
            "model.norm.weight": MLXArray.zeros([256]),
            "pre_projection.weight": MLXArray.zeros([256, 3072]),
            "pre_projection.bias": MLXArray.zeros([256]),
            "post_projection.weight": MLXArray.zeros([1536, 256]),
            "post_projection.bias": MLXArray.zeros([1536]),
            "masked_embedding.centroids.weight": MLXArray.zeros([2048, 256]),
        ]
        let sanitized = Gemma4AssistantWeightSanitizer.sanitize(
            weights: weights,
            tieWordEmbeddings: true
        )
        #expect(sanitized.count == weights.count)
        for k in weights.keys {
            #expect(sanitized[k] != nil, "cle absente: \(k)")
        }
    }
}
