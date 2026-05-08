import Testing
import Foundation
import MLX
import MLXNN
@testable import Gemma4Swift

@Suite("Drafter MTP Gemma 4 Assistant — sanity forward")
struct Gemma4AssistantDrafterTests {

    // Mini config representative du E2B Assistant: 4 couches kv-shared,
    // hidden=256, vocab=262144, num_centroids=2048, top_k=32. Reduit a un
    // backbone fictif de 64 et vocab 256 pour tester sans allouer 256 Mo.
    let miniConfigJSON = """
    {
      "model_type": "gemma4_assistant",
      "backbone_hidden_size": 64,
      "num_centroids": 8,
      "centroid_intermediate_top_k": 2,
      "use_ordered_embeddings": true,
      "tie_word_embeddings": true,
      "image_token_id": 258880,
      "audio_token_id": 258881,
      "boi_token_id": 255999,
      "eoi_token_id": 258882,
      "boa_token_id": 256000,
      "eoa_token_id": 258883,
      "text_config": {
        "model_type": "gemma4_text",
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "intermediate_size": 64,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 16,
        "global_head_dim": 32,
        "rms_norm_eps": 1e-06,
        "vocab_size": 256,
        "num_kv_shared_layers": 2,
        "hidden_size_per_layer_input": 0,
        "vocab_size_per_layer_input": 0,
        "sliding_window": 128,
        "max_position_embeddings": 1024,
        "tie_word_embeddings": true,
        "enable_moe_block": false,
        "use_double_wide_mlp": false,
        "attention_bias": false,
        "attention_k_eq_v": false,
        "layer_types": ["sliding_attention", "full_attention"],
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
        }
      }
    }
    """

    @Test("Forward du drafter retourne shapes (lastHidden, logits) attendues sans NaN")
    func testDrafterForwardShapes() throws {
        let data = miniConfigJSON.data(using: .utf8)!
        let config = try JSONDecoder().decode(Gemma4AssistantConfig.self, from: data)
        let drafter = Gemma4AssistantDraftModel(config)

        let B = 1
        let L = 1
        let kvLen = 4
        let backbone = config.backboneHiddenSize  // 64
        let textHidden = config.textConfig.hiddenSize  // 32
        let vocab = config.textConfig.vocabSize  // 256
        let numKVHeads = config.textConfig.numKeyValueHeads  // 1
        let slidingHeadDim = config.textConfig.headDim  // 16
        let fullHeadDim = config.textConfig.globalHeadDim  // 32

        // K/V partages: shapes [B, numKVHeads, kvLen, headDim] par layer_type
        let slidingK = MLXRandom.normal([B, numKVHeads, kvLen, slidingHeadDim])
        let slidingV = MLXRandom.normal([B, numKVHeads, kvLen, slidingHeadDim])
        let fullK = MLXRandom.normal([B, numKVHeads, kvLen, fullHeadDim])
        let fullV = MLXRandom.normal([B, numKVHeads, kvLen, fullHeadDim])

        let sharedKV: SharedKVStates = [
            "sliding_attention": (keys: slidingK, values: slidingV),
            "full_attention": (keys: fullK, values: fullV),
        ]

        // Input du drafter: concat(target_embed, target_hidden) [B, L, 2*backbone]
        let inputs = MLXRandom.normal([B, L, 2 * backbone])

        let (lastHidden, logits) = drafter(
            inputsEmbeds: inputs,
            sharedKVStates: sharedKV,
            position: kvLen
        )

        // Shapes
        #expect(lastHidden.shape == [B, L, backbone])
        #expect(logits.shape == [B, L, vocab])

        // Eval pour materialiser et controler les NaN
        eval(lastHidden, logits)
        let logitsHasNaN = isNaN(logits).any().item(Bool.self)
        let hiddenHasNaN = isNaN(lastHidden).any().item(Bool.self)
        #expect(!hiddenHasNaN, "lastHidden contient des NaN")
        #expect(!logitsHasNaN, "logits contient des NaN")
    }

    @Test("Forward avec L>1 (verify-style batch de drafts)")
    func testDrafterForwardMultiToken() throws {
        let data = miniConfigJSON.data(using: .utf8)!
        let config = try JSONDecoder().decode(Gemma4AssistantConfig.self, from: data)
        let drafter = Gemma4AssistantDraftModel(config)

        let B = 1
        let L = 3
        let kvLen = 4
        let numKVHeads = config.textConfig.numKeyValueHeads
        let slidingHeadDim = config.textConfig.headDim
        let fullHeadDim = config.textConfig.globalHeadDim

        let sharedKV: SharedKVStates = [
            "sliding_attention": (
                keys: MLXRandom.normal([B, numKVHeads, kvLen, slidingHeadDim]),
                values: MLXRandom.normal([B, numKVHeads, kvLen, slidingHeadDim])
            ),
            "full_attention": (
                keys: MLXRandom.normal([B, numKVHeads, kvLen, fullHeadDim]),
                values: MLXRandom.normal([B, numKVHeads, kvLen, fullHeadDim])
            ),
        ]

        let inputs = MLXRandom.normal([B, L, 2 * config.backboneHiddenSize])

        let (lastHidden, logits) = drafter(
            inputsEmbeds: inputs,
            sharedKVStates: sharedKV,
            position: kvLen
        )

        #expect(lastHidden.shape == [B, L, config.backboneHiddenSize])
        #expect(logits.shape == [B, L, config.textConfig.vocabSize])

        eval(lastHidden, logits)
        #expect(!isNaN(logits).any().item(Bool.self))
        #expect(!isNaN(lastHidden).any().item(Bool.self))
    }

    @Test("setSharedKV stocke etat correctement")
    func testSetSharedKV() throws {
        let data = miniConfigJSON.data(using: .utf8)!
        let config = try JSONDecoder().decode(Gemma4AssistantConfig.self, from: data)
        let drafter = Gemma4AssistantDraftModel(config)

        let kv = MLXRandom.normal([1, 1, 8, 16])
        let kvDict: SharedKVStates = [
            "sliding_attention": (keys: kv, values: kv),
            "full_attention": (keys: kv, values: kv),
        ]

        drafter.setSharedKV(kvDict, kvOffset: 8)
        #expect(drafter.kvOffset == 8)
        #expect(drafter.position == 8)
        #expect(drafter.sharedKV != nil)

        drafter.setSharedKV(kvDict, kvOffset: 8, position: 5)
        #expect(drafter.position == 5)

        drafter.reset()
        #expect(drafter.sharedKV == nil)
        #expect(drafter.kvOffset == 0)
        #expect(drafter.position == 0)
    }
}
