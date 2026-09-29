import Testing
import Foundation
import MLX
import MLXNN
import MLXOptimizers
@testable import Gemma4Swift

@Suite("Gemma4DrafterTraining — loss + step")
struct DrafterTrainingTests {

    /// Mini drafter config + mini target config compatibles.
    /// Target: 2 layers (1 sliding, 1 full, pas de kv-shared), hidden 32, vocab 256.
    /// Drafter: 2 layers kv-shared-only, hidden 16, backbone 32, vocab 256.
    let targetConfigJSON = """
    {
      "model_type": "gemma4_text",
      "hidden_size": 32, "num_hidden_layers": 2, "intermediate_size": 64,
      "num_attention_heads": 2, "num_key_value_heads": 1,
      "head_dim": 16, "global_head_dim": 32, "rms_norm_eps": 1e-06,
      "vocab_size": 256, "num_kv_shared_layers": 0,
      "hidden_size_per_layer_input": 0, "vocab_size_per_layer_input": 0,
      "sliding_window": 128, "max_position_embeddings": 1024,
      "tie_word_embeddings": true, "enable_moe_block": false,
      "use_double_wide_mlp": false, "attention_bias": false, "attention_k_eq_v": false,
      "final_logit_softcapping": 0,
      "layer_types": ["sliding_attention", "full_attention"]
    }
    """

    let drafterConfigJSON = """
    {
      "model_type": "gemma4_assistant",
      "backbone_hidden_size": 32,
      "num_centroids": 8, "centroid_intermediate_top_k": 2,
      "use_ordered_embeddings": true, "tie_word_embeddings": true,
      "text_config": {
        "model_type": "gemma4_text",
        "hidden_size": 16, "num_hidden_layers": 2, "intermediate_size": 32,
        "num_attention_heads": 2, "num_key_value_heads": 1,
        "head_dim": 16, "global_head_dim": 32, "rms_norm_eps": 1e-06,
        "vocab_size": 256, "num_kv_shared_layers": 2,
        "hidden_size_per_layer_input": 0, "vocab_size_per_layer_input": 0,
        "sliding_window": 128, "max_position_embeddings": 1024,
        "tie_word_embeddings": true, "enable_moe_block": false,
        "use_double_wide_mlp": false, "attention_bias": false, "attention_k_eq_v": false,
        "layer_types": ["sliding_attention", "full_attention"]
      }
    }
    """

    @Test("drafterLoss retourne un scalaire fini non-NaN")
    func testDrafterLossScalar() {
        let targetCfg = try! JSONDecoder().decode(
            Gemma4TextConfig.self, from: targetConfigJSON.data(using: .utf8)!)
        let drafterCfg = try! JSONDecoder().decode(
            Gemma4AssistantConfig.self, from: drafterConfigJSON.data(using: .utf8)!)

        let target = Gemma4LanguageModel(targetCfg)
        let drafter = Gemma4AssistantDraftModel(drafterCfg)
        drafter.bind(target: target)

        let B = 1, L = 8
        let batch = MLXArray((0 ..< B * L).map { Int32($0 % 100) }).reshaped(B, L)

        // last full + last sliding concrete idx
        let layerTypes = targetCfg.resolvedLayerTypes
        let concrete = Array(layerTypes.prefix(targetCfg.firstKvSharedLayerIdx))
        let lastFullIdx = concrete.lastIndex(of: "full_attention") ?? 0
        let lastSlidingIdx = concrete.lastIndex(of: "sliding_attention") ?? 0

        let (loss, ntoks) = Gemma4DrafterTraining.drafterLoss(
            drafter: drafter,
            target: target,
            batchTokens: batch,
            lastFullCacheIdx: lastFullIdx,
            lastSlidingCacheIdx: lastSlidingIdx
        )
        eval(loss, ntoks)
        #expect(loss.shape == [])
        let lossVal = loss.item(Float.self)
        #expect(lossVal.isFinite, "loss = \(lossVal) doit etre fini")
        #expect(lossVal > 0, "loss CE doit etre > 0 (sur init random)")
        #expect(ntoks.item(Float.self) == Float(B * (L - 2)))  // L-2 positions valid
    }

    @Test("trainDrafter sur 2 iter ne crash pas + loss decroit (ou stable)")
    func testTrainDrafterSmoke() throws {
        let targetCfg = try JSONDecoder().decode(
            Gemma4TextConfig.self, from: targetConfigJSON.data(using: .utf8)!)
        let drafterCfg = try JSONDecoder().decode(
            Gemma4AssistantConfig.self, from: drafterConfigJSON.data(using: .utf8)!)

        let target = Gemma4LanguageModel(targetCfg)
        let drafter = Gemma4AssistantDraftModel(drafterCfg)
        drafter.bind(target: target)

        // Mini dataset: 4 samples de 32 tokens chacun
        let samples: [[Int]] = (0 ..< 4).map { _ in (0 ..< 32).map { _ in Int.random(in: 0 ..< 256) } }

        let layerTypes = targetCfg.resolvedLayerTypes
        let concrete = Array(layerTypes.prefix(targetCfg.firstKvSharedLayerIdx))
        let lastFullIdx = concrete.lastIndex(of: "full_attention") ?? 0
        let lastSlidingIdx = concrete.lastIndex(of: "sliding_attention") ?? 0

        let optimizer = Adam(learningRate: 1e-3)

        var config = Gemma4DrafterTraining.TrainConfig()
        config.iterations = 2
        config.seqLen = 16
        config.batchSize = 1
        config.stepsPerReport = 1

        try Gemma4DrafterTraining.trainDrafter(
            drafter: drafter, target: target,
            tokenizedSamples: samples,
            lastFullCacheIdx: lastFullIdx,
            lastSlidingCacheIdx: lastSlidingIdx,
            optimizer: optimizer,
            config: config
        )
        // Si on est ici, pas de crash. Pas de save de poids (weightsURL nil).
    }

    @Test("K-30 d : cible hors du gradient = memes perte et gradients du drafter")
    func testTargetOutsideGradient() throws {
        try Device.withDefaultDevice(.cpu) {
            let targetCfg = try JSONDecoder().decode(Gemma4TextConfig.self, from: Data(targetConfigJSON.utf8))
            let drafterCfg = try JSONDecoder().decode(Gemma4AssistantConfig.self, from: Data(drafterConfigJSON.utf8))
            MLXRandom.seed(4)
            let target = Gemma4LanguageModel(targetCfg)
            let drafter = Gemma4AssistantDraftModel(drafterCfg)
            drafter.bind(target: target)
            let batch = MLXArray((0 ..< 8).map { Int32(($0 * 11) % 100) }).reshaped(1, 8)
            let concrete = Array(targetCfg.resolvedLayerTypes.prefix(targetCfg.firstKvSharedLayerIdx))
            let full = concrete.lastIndex(of: "full_attention") ?? 0
            let sliding = concrete.lastIndex(of: "sliding_attention") ?? 0

            let inside = valueAndGrad(model: drafter) { (d: Gemma4AssistantDraftModel, a: [MLXArray]) -> [MLXArray] in
                let (l, n) = Gemma4DrafterTraining.drafterLoss(
                    drafter: d, target: target, batchTokens: a[0], lastFullCacheIdx: full, lastSlidingCacheIdx: sliding)
                return [l, n]
            }
            let outputs = Gemma4DrafterTraining.targetOutputs(
                target: target, batchTokens: batch, lastFullCacheIdx: full, lastSlidingCacheIdx: sliding)
            let outside = valueAndGrad(model: drafter) { (d: Gemma4AssistantDraftModel, a: [MLXArray]) -> [MLXArray] in
                let (l, n) = Gemma4DrafterTraining.drafterLoss(drafter: d, batchTokens: a[0], target: outputs)
                return [l, n]
            }
            let (l1, g1) = inside(drafter, [batch])
            let (l2, g2) = outside(drafter, [batch])
            #expect(abs(l1[0] - l2[0]).item(Float.self) < 1e-6)
            let reference = Dictionary(g1.flattened(), uniquingKeysWith: { a, _ in a })
            for (key, value) in g2.flattened() {
                #expect(allClose(value, reference[key]!, atol: 1e-5).item(Bool.self), "\(key)")
            }
        }
    }
}

