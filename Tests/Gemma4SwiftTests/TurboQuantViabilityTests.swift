// Tests de regression sur turboQuantViable() : le compte de KV heads doit
// utiliser num_global_key_value_heads sur les modeles MQA full attention
// (ex: 12B Unified), sinon on surestime massivement le gain et active TQ
// alors qu'il est COUTeux en perf et n'economise quasi rien.

import Testing
import Foundation
@testable import Gemma4Swift

@Suite("TurboQuant Viability — heuristique gain RAM")
struct TurboQuantViabilityTests {

    /// Construit un TextConfig minimaliste depuis un dict JSON, en partant
    /// d'un squelette commun.
    private static func textConfig(json: String) throws -> Gemma4TextConfig {
        let data = json.data(using: .utf8)!
        return try JSONDecoder().decode(Gemma4TextConfig.self, from: data)
    }

    @Test("12B Unified (MQA kv_heads=1, head_dim=512, 8 full_attn) — TQ4 doit etre desactive")
    func test12BMQADisabled() throws {
        // Pattern : 5 sliding + 1 full repete 8 fois = 48 layers, 8 full_attn.
        var pattern: [String] = []
        for _ in 0..<8 { pattern.append(contentsOf: Array(repeating: "sliding_attention", count: 5) + ["full_attention"]) }
        let layerTypesJson = pattern.map { "\"\($0)\"" }.joined(separator: ",")

        let json = """
        {
            "model_type": "gemma4_unified_text",
            "hidden_size": 3840,
            "num_hidden_layers": 48,
            "intermediate_size": 15360,
            "num_attention_heads": 16,
            "head_dim": 256,
            "global_head_dim": 512,
            "vocab_size": 262144,
            "num_key_value_heads": 8,
            "num_global_key_value_heads": 1,
            "num_kv_shared_layers": 0,
            "hidden_size_per_layer_input": 0,
            "sliding_window": 1024,
            "max_position_embeddings": 131072,
            "attention_k_eq_v": true,
            "layer_types": [\(layerTypesJson)],
            "tie_word_embeddings": true
        }
        """
        let cfg = try Self.textConfig(json: json)
        let (viable, fullCount, headDim, reason) = Gemma4LanguageModel.turboQuantViability(config: cfg, bits: 4)
        #expect(fullCount == 8)
        #expect(headDim == 512)
        #expect(viable == false, "12B MQA(1) full_attn ne devrait PAS passer le check : \(reason)")
        #expect(reason.contains("kv_heads=1"))
    }

    @Test("Hypothetique GQA kv_heads=8 sur full_attn — TQ4 doit etre active")
    func testGQA8HeadsEnabled() throws {
        // Meme pattern que 12B mais sans MQA : attention_k_eq_v=false (kv_heads=8 sur full).
        var pattern: [String] = []
        for _ in 0..<8 { pattern.append(contentsOf: Array(repeating: "sliding_attention", count: 5) + ["full_attention"]) }
        let layerTypesJson = pattern.map { "\"\($0)\"" }.joined(separator: ",")

        let json = """
        {
            "model_type": "gemma4_text",
            "hidden_size": 3840,
            "num_hidden_layers": 48,
            "intermediate_size": 15360,
            "num_attention_heads": 16,
            "head_dim": 256,
            "global_head_dim": 512,
            "vocab_size": 262144,
            "num_key_value_heads": 8,
            "num_kv_shared_layers": 0,
            "hidden_size_per_layer_input": 0,
            "sliding_window": 1024,
            "max_position_embeddings": 131072,
            "attention_k_eq_v": false,
            "layer_types": [\(layerTypesJson)],
            "tie_word_embeddings": true
        }
        """
        let cfg = try Self.textConfig(json: json)
        let (viable, _, _, reason) = Gemma4LanguageModel.turboQuantViability(config: cfg, bits: 4)
        #expect(viable == true, "8 KV heads sur full_attn => gain large, devrait etre actif : \(reason)")
        #expect(reason.contains("kv_heads=8"))
    }

    @Test("Modele a 0 full_attn — TQ desactive avec raison explicite")
    func testNoFullAttnDisabled() throws {
        let json = """
        {
            "model_type": "gemma4_text",
            "hidden_size": 2048,
            "num_hidden_layers": 20,
            "intermediate_size": 8192,
            "num_attention_heads": 8,
            "head_dim": 256,
            "global_head_dim": 0,
            "vocab_size": 262144,
            "num_key_value_heads": 4,
            "num_kv_shared_layers": 0,
            "hidden_size_per_layer_input": 0,
            "sliding_window": 1024,
            "max_position_embeddings": 131072,
            "attention_k_eq_v": false,
            "layer_types": ["sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention"],
            "tie_word_embeddings": true
        }
        """
        let cfg = try Self.textConfig(json: json)
        let (viable, fullCount, _, reason) = Gemma4LanguageModel.turboQuantViability(config: cfg, bits: 4)
        #expect(fullCount == 0)
        #expect(viable == false)
        #expect(reason.contains("trop peu de couches full attention"))
    }
}
