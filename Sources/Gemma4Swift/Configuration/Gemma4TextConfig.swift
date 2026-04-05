// Port de mlx-vlm/models/gemma4/config.py TextConfig

import Foundation

/// Parametres RoPE par type d'attention
public struct RoPEParameters: Codable {
    public let ropeTheta: Float
    public let ropeType: String
    public let partialRotaryFactor: Float?

    enum CodingKeys: String, CodingKey {
        case ropeTheta = "rope_theta"
        case ropeType = "rope_type"
        case partialRotaryFactor = "partial_rotary_factor"
    }
}

/// Configuration du modele texte Gemma 4
public struct Gemma4TextConfig: Codable {
    public let modelType: String
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let intermediateSize: Int
    public let numAttentionHeads: Int
    public let headDim: Int
    public let globalHeadDim: Int
    public let rmsNormEps: Float
    public let vocabSize: Int
    public let numKeyValueHeads: Int
    public let numGlobalKeyValueHeads: Int?
    public let numKvSharedLayers: Int
    public let hiddenSizePerLayerInput: Int
    public let vocabSizePerLayerInput: Int
    public let slidingWindow: Int
    public let slidingWindowPattern: Int
    public let maxPositionEmbeddings: Int
    public let ropeParameters: [String: RoPEParameters]?
    public let finalLogitSoftcapping: Float
    public let layerTypes: [String]?
    public let attentionBias: Bool
    public let attentionKEqV: Bool
    public let useDoubleWideMlp: Bool
    public let enableMoeBlock: Bool
    public let numExperts: Int?
    public let topKExperts: Int?
    public let tieWordEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case headDim = "head_dim"
        case globalHeadDim = "global_head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabSize = "vocab_size"
        case numKeyValueHeads = "num_key_value_heads"
        case numGlobalKeyValueHeads = "num_global_key_value_heads"
        case numKvSharedLayers = "num_kv_shared_layers"
        case hiddenSizePerLayerInput = "hidden_size_per_layer_input"
        case vocabSizePerLayerInput = "vocab_size_per_layer_input"
        case slidingWindow = "sliding_window"
        case slidingWindowPattern = "sliding_window_pattern"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeParameters = "rope_parameters"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case layerTypes = "layer_types"
        case attentionBias = "attention_bias"
        case attentionKEqV = "attention_k_eq_v"
        case useDoubleWideMlp = "use_double_wide_mlp"
        case enableMoeBlock = "enable_moe_block"
        case numExperts = "num_experts"
        case topKExperts = "top_k_experts"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decode(String.self, forKey: .modelType)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        numAttentionHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
        headDim = try c.decode(Int.self, forKey: .headDim)
        globalHeadDim = try c.decodeIfPresent(Int.self, forKey: .globalHeadDim) ?? 0
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        numKeyValueHeads = try c.decode(Int.self, forKey: .numKeyValueHeads)
        numGlobalKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numGlobalKeyValueHeads)
        numKvSharedLayers = try c.decodeIfPresent(Int.self, forKey: .numKvSharedLayers) ?? 0
        hiddenSizePerLayerInput = try c.decodeIfPresent(Int.self, forKey: .hiddenSizePerLayerInput) ?? 0
        vocabSizePerLayerInput = try c.decodeIfPresent(Int.self, forKey: .vocabSizePerLayerInput) ?? 0
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 512
        slidingWindowPattern = try c.decodeIfPresent(Int.self, forKey: .slidingWindowPattern) ?? 5
        maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131072
        ropeParameters = try c.decodeIfPresent([String: RoPEParameters].self, forKey: .ropeParameters)
        finalLogitSoftcapping = try c.decodeIfPresent(Float.self, forKey: .finalLogitSoftcapping) ?? 30.0
        layerTypes = try c.decodeIfPresent([String].self, forKey: .layerTypes)
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        attentionKEqV = try c.decodeIfPresent(Bool.self, forKey: .attentionKEqV) ?? false
        useDoubleWideMlp = try c.decodeIfPresent(Bool.self, forKey: .useDoubleWideMlp) ?? false
        enableMoeBlock = try c.decodeIfPresent(Bool.self, forKey: .enableMoeBlock) ?? false
        numExperts = try c.decodeIfPresent(Int.self, forKey: .numExperts)
        topKExperts = try c.decodeIfPresent(Int.self, forKey: .topKExperts)
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
    }

    /// Types de couches resolus (genere le pattern si absent)
    public var resolvedLayerTypes: [String] {
        if let lt = layerTypes { return lt }
        var pattern = Array(repeating: "sliding_attention", count: slidingWindowPattern - 1)
        pattern.append("full_attention")
        var result: [String] = []
        while result.count < numHiddenLayers {
            result.append(contentsOf: pattern)
        }
        return Array(result.prefix(numHiddenLayers))
    }

    /// Index de la premiere couche KV partagee
    public var firstKvSharedLayerIdx: Int {
        numHiddenLayers - numKvSharedLayers
    }

    /// Theta RoPE pour un type d'attention donne
    public func ropeTheta(forLayerType type: String) -> Float {
        let key = type == "full_attention" ? "full_attention" : "sliding_attention"
        return ropeParameters?[key]?.ropeTheta ?? 10000.0
    }

    /// Type RoPE pour un type d'attention
    public func ropeType(forLayerType type: String) -> String {
        let key = type == "full_attention" ? "full_attention" : "sliding_attention"
        return ropeParameters?[key]?.ropeType ?? "default"
    }

    /// Partial rotary factor pour full attention
    public var fullAttentionPartialRotaryFactor: Float {
        ropeParameters?["full_attention"]?.partialRotaryFactor ?? 1.0
    }
}
