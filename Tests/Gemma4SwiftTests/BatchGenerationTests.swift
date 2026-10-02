import Testing
import Foundation
import MLX
import MLXLMCommon
@testable import Gemma4Swift

/// K-41 : un lot complete a gauche doit generer, ligne par ligne, exactement ce que chaque
/// prompt genere seul (petit modele type E2B : KV partage, entrees par couche, fenetre
/// glissante de 8 franchie pendant le decodage).
@Suite("Generation par lot", .serialized)
struct BatchGenerationTests {

    static let config = """
    {
        "model_type": "gemma4_text", "hidden_size": 64, "num_hidden_layers": 4,
        "intermediate_size": 128, "num_attention_heads": 2, "head_dim": 32,
        "global_head_dim": 32, "rms_norm_eps": 1e-6, "vocab_size": 128,
        "num_key_value_heads": 1, "num_kv_shared_layers": 2, "sliding_window": 8,
        "sliding_window_pattern": 2, "max_position_embeddings": 256,
        "hidden_size_per_layer_input": 16, "vocab_size_per_layer_input": 128,
        "attention_bias": false, "use_double_wide_mlp": false, "enable_moe_block": false,
        "tie_word_embeddings": true,
        "rope_parameters": {
            "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
            "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}
        },
        "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention"]
    }
    """

    private func model() throws -> Gemma4LLMModel {
        let config = try JSONDecoder().decode(Gemma4TextConfig.self, from: Data(Self.config.utf8))
        MLXRandom.seed(41)
        let model = Gemma4LLMModel(config: config)
        eval(model)
        return model
    }

    /// Generation gloutonne d'un prompt seul, chemin standard (cache glissant rotatif).
    private func single(_ model: Gemma4LLMModel, _ ids: [Int], steps: Int) -> [Int] {
        let cache: [KVCache?] = model.languageModel.makeCache().map { $0 }
        var logits = model.languageModel(inputs: MLXArray(ids.map(Int32.init)).reshaped(1, -1), cache: cache)
        var out: [Int] = []
        for _ in 0 ..< steps {
            let next = argMax(logits[0..., -1, 0...], axis: -1)
            let token = next.item(Int.self)
            out.append(token)
            logits = model.languageModel(inputs: next.reshaped(1, 1), cache: cache)
        }
        return out
    }

    @Test("lot de 3 prompts de longueurs 3, 7, 5 : memes jetons que chaque prompt seul")
    func testBatchMatchesSingle() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try model()
            let prompts = [[2, 17, 33], [2, 5, 9, 61, 70, 12, 44], [2, 90, 3, 27, 8]]
            let steps = 6
            let expected = prompts.map { single(model, $0, steps: steps) }

            var got = Array(repeating: [Int](), count: prompts.count)
            let produced = try Gemma4BatchGeneration.run(
                model: model,
                requests: prompts.map { .init(ids: $0, maxTokens: steps) },
                stopTokens: []
            ) { row, token in
                got[row].append(token)
                return true
            }
            #expect(produced.map(\.tokens) == [steps, steps, steps])
            #expect(got == expected)
            #expect(model.languageModel.model.batchPadding == nil)
        }
    }

    @Test("une ligne arretee par l'appelant ne produit plus, les autres continuent")
    func testRowStop() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try model()
            let prompts = [[2, 17, 33], [2, 5, 9, 61]]
            var got = Array(repeating: [Int](), count: prompts.count)
            let produced = try Gemma4BatchGeneration.run(
                model: model, requests: prompts.map { .init(ids: $0, maxTokens: 5) }, stopTokens: []
            ) { row, token in
                got[row].append(token)
                return !(row == 0 && got[0].count == 2)
            }
            #expect(produced.map(\.tokens) == [2, 5])
            #expect(got[1] == single(model, prompts[1], steps: 5))
        }
    }

    @Test("masques de lot : padding cache, fenetre glissante comme createCausalMask")
    func testMasks() {
        Device.withDefaultDevice(.cpu) {
            let (global, sliding) = Gemma4TextModel.paddedBatchMasks(
                padding: [2, 0], queryLength: 4, offset: 0, windowSize: 2)
            let g = global.asArray(Bool.self)
            // Ligne 0, requete 3 (position absolue 3) : cles 2 et 3 visibles, 0-1 (padding) non.
            #expect(Array(g[12 ..< 16]) == [false, false, true, true])
            // Ligne 0, requete 0 (padding) : ne voit qu'elle-meme.
            #expect(Array(g[0 ..< 4]) == [true, false, false, false])
            let reference = createCausalMask(n: 4, offset: 0, windowSize: 2).asArray(Bool.self)
            let s = sliding.asArray(Bool.self)
            // Ligne 1 sans padding : identique a createCausalMask.
            #expect(Array(s[16 ..< 32]) == reference)
        }
    }
}
