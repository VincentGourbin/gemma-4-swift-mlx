import Testing
import Foundation
import MLX
import MLXNN
import MLXLMCommon
@testable import Gemma4Swift

/// Couverture de `forwardCollectingHiddenStates` — l'API "encodeur texte" qui expose
/// les hidden states de toutes les couches (convention HuggingFace
/// `output_hidden_states=True`), consommee par les ports de modeles de diffusion.
@Suite("forwardCollectingHiddenStates — hidden states par couche")
struct ForwardCollectingHiddenStatesTests {

    /// Mini config sans KV-sharing ni per-layer inputs (profil 12B Unified / 31B).
    let plainConfigJSON = """
    {
      "model_type": "gemma4_text",
      "hidden_size": 32, "num_hidden_layers": 4, "intermediate_size": 64,
      "num_attention_heads": 2, "num_key_value_heads": 1,
      "head_dim": 16, "global_head_dim": 32,
      "rms_norm_eps": 1e-06,
      "vocab_size": 256, "num_kv_shared_layers": 0,
      "hidden_size_per_layer_input": 0, "vocab_size_per_layer_input": 0,
      "sliding_window": 128, "max_position_embeddings": 1024,
      "tie_word_embeddings": true, "enable_moe_block": false,
      "use_double_wide_mlp": false, "attention_bias": false, "attention_k_eq_v": false,
      "final_logit_softcapping": 0,
      "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"]
    }
    """

    /// Mini config AVEC KV-sharing et per-layer inputs (profil E2B/E4B) : c'est le
    /// chemin ou `forwardCollectingHiddenStates` doit conserver la collecte des
    /// intermediates K/V, sans quoi les couches partagees recoivent des K/V nuls.
    let sharedConfigJSON = """
    {
      "model_type": "gemma4_text",
      "hidden_size": 32, "num_hidden_layers": 4, "intermediate_size": 64,
      "num_attention_heads": 2, "num_key_value_heads": 1,
      "head_dim": 16, "global_head_dim": 32,
      "rms_norm_eps": 1e-06,
      "vocab_size": 256, "num_kv_shared_layers": 2,
      "hidden_size_per_layer_input": 8, "vocab_size_per_layer_input": 64,
      "sliding_window": 128, "max_position_embeddings": 1024,
      "tie_word_embeddings": true, "enable_moe_block": false,
      "use_double_wide_mlp": false, "attention_bias": false, "attention_k_eq_v": false,
      "final_logit_softcapping": 0,
      "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention"]
    }
    """

    func decode(_ json: String) -> Gemma4TextConfig {
        try! JSONDecoder().decode(Gemma4TextConfig.self, from: json.data(using: .utf8)!)
    }

    @Test("retourne numHiddenLayers + 1 tenseurs [B, T, hidden]")
    func testCountAndShapes() {
        let cfg = decode(plainConfigJSON)
        let model = Gemma4TextModel(cfg)

        let B = 1, L = 5
        let inputs = MLXArray((0 ..< B * L).map { Int32($0 % 100) }).reshaped(B, L)

        let states = model.forwardCollectingHiddenStates(inputs: inputs)

        #expect(states.count == cfg.numHiddenLayers + 1)
        for (i, s) in states.enumerated() {
            #expect(s.shape == [B, L, cfg.hiddenSize], "state \(i) shape=\(s.shape)")
        }

        eval(states)
        for (i, s) in states.enumerated() {
            #expect(!isNaN(s).any().item(Bool.self), "state \(i) contient des NaN")
        }
    }

    @Test("states[0] == embeddings scalees par sqrt(hidden_size)")
    func testFirstStateIsScaledEmbeddings() {
        let cfg = decode(plainConfigJSON)
        let model = Gemma4TextModel(cfg)

        let B = 1, L = 3
        let inputs = MLXArray((0 ..< B * L).map { Int32($0 % 100) }).reshaped(B, L)

        let states = model.forwardCollectingHiddenStates(inputs: inputs)
        let expected = model.embedTokens(inputs) * MLXArray(model.embedScale, dtype: .float32)

        eval(states[0], expected)
        let diff = abs(states[0] - expected).max().item(Float.self)
        #expect(diff == 0, "states[0] doit etre exactement les embeddings scalees (diff \(diff))")
    }

    @Test("states.last == callAsFunction (derniere entree post-RMSNorm)")
    func testLastStateIsNormedOutput() {
        let cfg = decode(plainConfigJSON)
        let model = Gemma4TextModel(cfg)

        let B = 1, L = 4
        let inputs = MLXArray((0 ..< B * L).map { Int32($0 % 100) }).reshaped(B, L)

        let states = model.forwardCollectingHiddenStates(inputs: inputs)
        let direct = model(inputs: inputs)

        eval(states.last!, direct)
        let diff = abs(states.last! - direct).max().item(Float.self)
        #expect(diff == 0, "states.last doit etre identique a callAsFunction (diff \(diff))")
    }

    @Test("la derniere entree est normee, les precedentes ne le sont pas")
    func testOnlyLastStateIsNormed() {
        let cfg = decode(plainConfigJSON)
        let model = Gemma4TextModel(cfg)

        let B = 1, L = 3
        let inputs = MLXArray((0 ..< B * L).map { Int32($0 % 100) }).reshaped(B, L)

        let states = model.forwardCollectingHiddenStates(inputs: inputs)
        let out = model.forwardCollectingIntermediates(inputs: inputs)

        // states[N-1] = entree de la derniere couche != preNormHidden (sa sortie)
        eval(states[cfg.numHiddenLayers - 1], out.preNormHidden, states.last!, out.hidden)
        let notPreNorm = abs(states[cfg.numHiddenLayers - 1] - out.preNormHidden).max().item(Float.self)
        #expect(notPreNorm > 1e-4,
                "states[N-1] est l'ENTREE de la derniere couche, pas sa sortie pre-norm (diff \(notPreNorm))")

        let isNormed = abs(states.last! - out.hidden).max().item(Float.self)
        #expect(isNormed == 0, "states.last doit etre la sortie post-norm (diff \(isNormed))")
    }

    @Test("KV-sharing + per-layer inputs : parite avec callAsFunction preservee")
    func testKvSharedParity() {
        let cfg = decode(sharedConfigJSON)
        #expect(cfg.firstKvSharedLayerIdx < cfg.numHiddenLayers, "config de test sans KV-sharing")

        let model = Gemma4TextModel(cfg)

        let B = 1, L = 4
        let inputs = MLXArray((0 ..< B * L).map { Int32($0 % 50) }).reshaped(B, L)

        let states = model.forwardCollectingHiddenStates(inputs: inputs)
        let direct = model(inputs: inputs)

        #expect(states.count == cfg.numHiddenLayers + 1)
        eval(states.last!, direct)
        let diff = abs(states.last! - direct).max().item(Float.self)
        #expect(diff == 0,
                "sur le path KV-shared, states.last doit rester identique a callAsFunction (diff \(diff))")
    }

    @Test("LanguageModel expose le meme resultat que le TextModel")
    func testLanguageModelPassthrough() {
        let cfg = decode(plainConfigJSON)
        let langModel = Gemma4LanguageModel(cfg)

        let B = 1, L = 3
        let inputs = MLXArray((0 ..< B * L).map { Int32($0 % 100) }).reshaped(B, L)

        let viaLang = langModel.forwardCollectingHiddenStates(inputs: inputs)
        let viaText = langModel.model.forwardCollectingHiddenStates(inputs: inputs)

        #expect(viaLang.count == cfg.numHiddenLayers + 1)
        #expect(viaLang.count == viaText.count)
        eval(viaLang.last!, viaText.last!)
        #expect(abs(viaLang.last! - viaText.last!).max().item(Float.self) == 0)
    }

    @Test("Gemma4LLMModel (wrapper rendu par le registry en text-only) expose la meme API")
    func testLLMModelPassthrough() {
        let cfg = decode(plainConfigJSON)
        let llm = Gemma4LLMModel(config: cfg)

        let B = 1, L = 3
        let inputs = MLXArray((0 ..< B * L).map { Int32($0 % 100) }).reshaped(B, L)

        let states = llm.forwardCollectingHiddenStates(inputs)
        let direct = llm(inputs, cache: nil)

        #expect(states.count == cfg.numHiddenLayers + 1)
        for s in states {
            #expect(s.shape == [B, L, cfg.hiddenSize])
        }

        // callAsFunction applique le lm_head : on compare a la hidden post-norm,
        // pas aux logits. states.last doit etre l'entree du lm_head.
        #expect(direct.shape == [B, L, cfg.vocabSize])
        eval(states.last!, direct)
        #expect(!isNaN(states.last!).any().item(Bool.self))
    }

    @Test("forwardCollectingIntermediates ne collecte pas les hidden states")
    func testIntermediatesPathStaysLean() {
        let cfg = decode(plainConfigJSON)
        let model = Gemma4TextModel(cfg)

        let inputs = MLXArray((0 ..< 3).map { Int32($0) }).reshaped(1, 3)
        let out = model.forwardCollectingIntermediates(inputs: inputs)

        #expect(out.hiddenStates.isEmpty,
                "le path MTP ne doit pas retenir N+1 tenseurs supplementaires")
    }

    @Test("inputsEmbeds : states[0] est l'embedding fourni tel quel")
    func testInputsEmbedsPath() {
        let cfg = decode(plainConfigJSON)
        let model = Gemma4TextModel(cfg)

        let B = 1, L = 3
        let embeds = MLXArray.zeros([B, L, cfg.hiddenSize], dtype: .float32) + MLXArray(Float(0.02))

        let states = model.forwardCollectingHiddenStates(inputsEmbeds: embeds)

        #expect(states.count == cfg.numHiddenLayers + 1)
        eval(states[0], embeds)
        let diff = abs(states[0] - embeds).max().item(Float.self)
        #expect(diff == 0, "avec inputsEmbeds, aucun scaling supplementaire ne doit etre applique")
    }
}
