// Port de mlx-vlm/models/gemma4/config.py AudioConfig

import Foundation

/// Configuration de l'encodeur audio Gemma 4 (Conformer)
public struct Gemma4AudioConfig: Codable, Sendable {
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let hiddenAct: String
    public let subsamplingConvChannels: [Int]
    public let convKernelSize: Int
    public let residualWeight: Float
    public let attentionChunkSize: Int
    public let attentionContextLeft: Int
    public let attentionContextRight: Int
    public let attentionLogitCap: Float
    public let attentionInvalidLogitsValue: Float
    public let useClippedLinears: Bool
    public let rmsNormEps: Float
    public let gradientClipping: Float
    public let outputProjDims: Int?

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case hiddenAct = "hidden_act"
        case subsamplingConvChannels = "subsampling_conv_channels"
        case convKernelSize = "conv_kernel_size"
        case residualWeight = "residual_weight"
        case attentionChunkSize = "attention_chunk_size"
        case attentionContextLeft = "attention_context_left"
        case attentionContextRight = "attention_context_right"
        case attentionLogitCap = "attention_logit_cap"
        case attentionInvalidLogitsValue = "attention_invalid_logits_value"
        case useClippedLinears = "use_clipped_linears"
        case rmsNormEps = "rms_norm_eps"
        case gradientClipping = "gradient_clipping"
        case outputProjDims = "output_proj_dims"
    }

    /// Taille du contexte d'attention
    public var contextSize: Int {
        attentionChunkSize + max(0, attentionContextLeft - 1) + attentionContextRight
    }

    /// Horizon passe maximum
    public var maxPastHorizon: Int {
        max(0, attentionContextLeft - 1)
    }

    /// Config par defaut pour E2B/E4B
    public static let defaultConfig = Gemma4AudioConfig(
        hiddenSize: 1024, numHiddenLayers: 12, numAttentionHeads: 8,
        hiddenAct: "silu", subsamplingConvChannels: [128, 32],
        convKernelSize: 5, residualWeight: 0.5, attentionChunkSize: 12,
        attentionContextLeft: 13, attentionContextRight: 0,
        attentionLogitCap: 50.0, attentionInvalidLogitsValue: -1e9,
        useClippedLinears: true, rmsNormEps: 1e-6,
        gradientClipping: 1e10, outputProjDims: 1536
    )
}
