import Testing
import Foundation
import MLX
import MLXLMCommon
@testable import Gemma4Swift

/// Garde-fou K-13 / P-01 : le prefill par tranches (prepare) doit donner le meme
/// dernier jeton et le meme pas de decodage suivant que la passe unique d'avant,
/// y compris avec des tranches plus petites que la fenetre glissante et des couches
/// a KV partage.
@Suite("Parite du prefill par tranches")
struct ChunkedPrefillParityTests {

    /// 4 couches (2 a KV partage), fenetre glissante 8, vocabulaire 128 ; ids media dans le vocab.
    static let textConfig = """
    {
        "model_type": "gemma4_text", "hidden_size": 32, "num_hidden_layers": 4,
        "intermediate_size": 64, "num_attention_heads": 2, "head_dim": 16,
        "global_head_dim": 16, "rms_norm_eps": 1e-6, "vocab_size": 128,
        "num_key_value_heads": 1, "num_kv_shared_layers": 2, "sliding_window": 8,
        "sliding_window_pattern": 2, "max_position_embeddings": 256,
        "attention_bias": false, "use_double_wide_mlp": false, "enable_moe_block": false,
        "tie_word_embeddings": true,
        "rope_parameters": {
            "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
            "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}
        },
        "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention"]
    }
    """

    static let prompt: [Int32] = (0 ..< 23).map { Int32(($0 * 11 + 5) % 100) }

    private func relativeError(_ a: MLXArray, _ b: MLXArray) -> Float {
        (sqrt(sum(square(a - b))) / sqrt(sum(square(a)))).item(Float.self)
    }

    /// Reference : passe unique (comportement d'avant), puis un pas de decodage.
    private func reference(
        _ model: any LanguageModel, tokens: MLXArray, cache: [KVCache]
    ) -> (last: MLXArray, next: MLXArray) {
        let all = model(tokens[.newAxis], cache: cache)
        let last = all[0..., -1, 0...]
        let next = model(MLXArray([Int32(7)])[.newAxis], cache: cache)[0..., -1, 0...]
        eval(last, next)
        return (last, next)
    }

    /// Tranches : prepare puis le jeton rendu, puis le meme pas de decodage.
    private func chunked(
        _ model: any LanguageModel, tokens: MLXArray, cache: [KVCache], step: Int
    ) throws -> (last: MLXArray, next: MLXArray, remaining: Int) {
        guard case .tokens(let rest) = try model.prepare(
            LMInput(tokens: tokens), cache: cache, windowSize: step)
        else { throw CancellationError() }
        let last = model(rest.tokens[.newAxis], cache: cache)[0..., -1, 0...]
        let next = model(MLXArray([Int32(7)])[.newAxis], cache: cache)[0..., -1, 0...]
        eval(last, next)
        return (last, next, rest.tokens.size)
    }

    @Test("modele texte : tranches de 3 et 5 (< fenetre 8) = passe unique", arguments: [3, 5, 512])
    func testTextParity(step: Int) throws {
        let config = try JSONDecoder().decode(Gemma4TextConfig.self, from: Data(Self.textConfig.utf8))
        MLXRandom.seed(1)
        let model = Gemma4LLMModel(config: config)
        let tokens = MLXArray(Self.prompt)

        let ref = reference(model, tokens: tokens, cache: model.newCache(parameters: nil))
        let got = try chunked(model, tokens: tokens, cache: model.newCache(parameters: nil), step: step)

        #expect(got.remaining == 1, "le TokenIterator doit recevoir un seul jeton")
        #expect(relativeError(ref.last, got.last) < 1e-4, "dernier jeton")
        #expect(relativeError(ref.next, got.next) < 1e-4, "pas de decodage suivant")
    }

    @Test("modele multimodal avec image : tranches de 4 = passe unique")
    func testMultimodalParity() throws {
        let full = """
        {"model_type": "gemma4", "text_config": \(Self.textConfig),
         "image_token_id": 100, "audio_token_id": 101, "video_token_id": 102,
         "vision_soft_tokens_per_image": 4, "tie_word_embeddings": true}
        """
        let config = try JSONDecoder().decode(Gemma4Config.self, from: Data(full.utf8))
        MLXRandom.seed(2)
        let model = Gemma4MultimodalLLMModel(config: config)
        // Texte, 4 jetons image a cheval sur deux tranches, texte.
        let ids: [Int32] = [2, 9, 13, 100, 100, 100, 100, 21, 34, 55, 8, 3]
        let tokens = MLXArray(ids)
        let image = MLXRandom.normal([1, 4, 32])

        model.pendingImageEmbeddings = image
        let ref = reference(model, tokens: tokens, cache: model.newCache(parameters: nil))

        model.pendingImageEmbeddings = image
        let got = try chunked(model, tokens: tokens, cache: model.newCache(parameters: nil), step: 4)

        #expect(got.remaining == 1)
        #expect(model.pendingImageEmbeddings == nil, "embeddings image consommes par prepare")
        #expect(relativeError(ref.last, got.last) < 1e-4, "dernier jeton")
        #expect(relativeError(ref.next, got.next) < 1e-4, "pas de decodage suivant")
    }

    /// Vrai modele (GEMMA4_INTEGRATION_MODEL_PATH), prompt de ~1 000 jetons : ecart des
    /// logits du premier jeton entre passe unique et tranches de 512, et marge top-1/top-2.
    /// Diagnostic : un ecart de l'ordre de la precision bf16 et une marge faible
    /// expliquent qu'un argmax puisse basculer sans bug.
    @Test("vrai modele : ecart passe unique / tranches sur le premier jeton",
          .enabled(if: ProcessInfo.processInfo.environment["GEMMA4_INTEGRATION_MODEL_PATH"] != nil))
    func testRealModelFirstTokenLogits() async throws {
        let url = URL(fileURLWithPath: ProcessInfo.processInfo.environment["GEMMA4_INTEGRATION_MODEL_PATH"]!)
        let container = try await Gemma4Registration.loadContainer(from: url, multimodal: false)
        let report = try await container.perform { context -> String in
            let sentence = "The history of the bicycle spans more than two centuries, from the wooden draisine of Karl Drais in 1817 to the carbon racing frames of today. "
            // Meme forme que `generate` : gabarit de chat, tour systeme par defaut.
            let ids = try Gemma4Processor.textChatIds(
                userPrompt: String(repeating: sentence, count: 36) + "Summarize this history in detail, step by step.",
                systemPrompt: "Tu es un assistant utile.", tokenizer: context.tokenizer)
            let tokens = MLXArray(ids.map { Int32($0) })

            let single = context.model(tokens[.newAxis], cache: context.model.newCache(parameters: nil))[0..., -1, 0...]
            let cache = context.model.newCache(parameters: nil)
            guard case .tokens(let rest) = try context.model.prepare(
                LMInput(tokens: tokens), cache: cache, windowSize: 512) else { return "prepare: logits" }
            let chunked = context.model(rest.tokens[.newAxis], cache: cache)[0..., -1, 0...]
            let a = single.asType(.float32), b = chunked.asType(.float32)
            let rel = (sqrt(sum(square(a - b))) / sqrt(sum(square(a)))).item(Float.self)
            let topA = argMax(a, axis: -1).item(Int32.self), topB = argMax(b, axis: -1).item(Int32.self)
            let sortedA = sorted(a, axis: -1)
            let margin = (sortedA[0..., -1] - sortedA[0..., -2]).item(Float.self)
            return "rel=\(rel) top1 unique=\(topA) tranches=\(topB) marge(top1-top2)=\(margin) dtype=\(single.dtype)"
        }
        print("PARITE-REEL", report)
        // Mesure du 2026-09-27 (E2B, ~1 000 jetons, gabarit de chat) : rel 2,1 % en
        // 6 bits, 0,6 % en bf16, meme top-1, marge top-1/top-2 de 0,5 logit. L'ordre des
        // reductions change avec le decoupage : un argmax peut basculer sur une
        // quasi-egalite, sans bug (la parite fp32 ci-dessus est exacte).
        let rel = Float(report.split(separator: " ").first?.dropFirst(4) ?? "") ?? .infinity
        #expect(rel < 0.05, "\(report)")
    }
}
