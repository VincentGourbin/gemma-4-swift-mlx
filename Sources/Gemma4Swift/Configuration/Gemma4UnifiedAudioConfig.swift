// Config audio pour gemma4_unified (12B) — projection directe sans Conformer

import Foundation

/// Configuration de l'audio embedder de gemma4_unified.
///
/// Contraste avec [[Gemma4AudioConfig]] (E2B/E4B Conformer) : ici il n'y a
/// pas d'encodeur, juste une projection lineaire depuis des chunks de samples
/// PCM bruts (audioSamplesPerToken = 640 samples a 16kHz = 40ms par token).
public struct Gemma4UnifiedAudioConfig: Codable, Sendable {
    public let modelType: String
    /// Dimension d'entree du projector (640 par defaut).
    public let audioEmbedDim: Int
    /// Nombre de samples PCM bruts groupes en un token audio (640 -> 40ms a 16kHz).
    public let audioSamplesPerToken: Int
    public let hiddenSize: Int
    public let outputProjDims: Int
    public let rmsNormEps: Float

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case audioEmbedDim = "audio_embed_dim"
        case audioSamplesPerToken = "audio_samples_per_token"
        case hiddenSize = "hidden_size"
        case outputProjDims = "output_proj_dims"
        case rmsNormEps = "rms_norm_eps"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "gemma4_unified_audio"
        audioEmbedDim = try c.decodeIfPresent(Int.self, forKey: .audioEmbedDim) ?? 640
        audioSamplesPerToken = try c.decodeIfPresent(Int.self, forKey: .audioSamplesPerToken) ?? 640
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? audioEmbedDim
        outputProjDims = try c.decodeIfPresent(Int.self, forKey: .outputProjDims) ?? audioEmbedDim
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
    }
}
