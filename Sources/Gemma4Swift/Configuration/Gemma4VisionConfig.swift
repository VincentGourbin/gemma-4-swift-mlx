// Port de mlx-vlm/models/gemma4/config.py VisionConfig

import Foundation

/// Configuration de l'encodeur vision Gemma 4
public struct Gemma4VisionConfig: Codable, Sendable {
    public let modelType: String
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let globalHeadDim: Int
    public let rmsNormEps: Float
    public let maxPositionEmbeddings: Int
    public let patchSize: Int
    public let poolingKernelSize: Int
    public let positionEmbeddingSize: Int
    public let defaultOutputLength: Int
    public let useClippedLinears: Bool
    public let standardize: Bool
    public let ropeParameters: [String: AnyCodable]?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case globalHeadDim = "global_head_dim"
        case rmsNormEps = "rms_norm_eps"
        case maxPositionEmbeddings = "max_position_embeddings"
        case patchSize = "patch_size"
        case poolingKernelSize = "pooling_kernel_size"
        case positionEmbeddingSize = "position_embedding_size"
        case defaultOutputLength = "default_output_length"
        case useClippedLinears = "use_clipped_linears"
        case standardize
        case ropeParameters = "rope_parameters"
    }

    public var ropeTheta: Float {
        if let params = ropeParameters,
           let theta = params["rope_theta"] {
            return (theta.value as? NSNumber)?.floatValue ?? 100.0
        }
        return 100.0
    }

    /// Nombre maximum de patches (avant pooling)
    public var maxPatches: Int {
        defaultOutputLength * poolingKernelSize * poolingKernelSize
    }

    /// Config par defaut pour E2B/E4B
    public static let defaultConfig = Gemma4VisionConfig(
        modelType: "gemma4_vision", hiddenSize: 768, intermediateSize: 3072,
        numHiddenLayers: 16, numAttentionHeads: 12, numKeyValueHeads: 12,
        headDim: 64, globalHeadDim: 64, rmsNormEps: 1e-6,
        maxPositionEmbeddings: 131072, patchSize: 16, poolingKernelSize: 3,
        positionEmbeddingSize: 10240, defaultOutputLength: 280,
        useClippedLinears: true, standardize: false, ropeParameters: nil
    )
}

/// Type helper pour decoder des valeurs JSON heterogenes
public struct AnyCodable: Codable, @unchecked Sendable {
    public let value: Any

    public init(_ value: Any) { self.value = value }

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let v = try? container.decode(Double.self) { value = v }
        else if let v = try? container.decode(String.self) { value = v }
        else if let v = try? container.decode(Bool.self) { value = v }
        else if let v = try? container.decode(Int.self) { value = v }
        else { value = "unknown" }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        if let v = value as? Double { try container.encode(v) }
        else if let v = value as? String { try container.encode(v) }
        else if let v = value as? Bool { try container.encode(v) }
        else if let v = value as? Int { try container.encode(v) }
        else { try container.encode("unknown") }
    }
}
