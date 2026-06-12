// Top-level config pour gemma4_unified (12B)
// Garde Gemma4TextConfig en commun avec gemma4 — seuls vision/audio changent.

import Foundation

/// Configuration top-level de gemma4_unified (modele 12B).
///
/// Reutilise [[Gemma4TextConfig]] (le decoder texte est identique a gemma4)
/// mais swap vision/audio par les variantes encoder-free unified.
public struct Gemma4UnifiedConfig: Decodable, @unchecked Sendable {
    public let modelType: String
    public let textConfig: Gemma4TextConfig
    public let visionConfig: Gemma4UnifiedVisionConfig?
    public let audioConfig: Gemma4UnifiedAudioConfig?
    public let imageTokenId: Int
    public let audioTokenId: Int
    public let videoTokenId: Int
    public let boiTokenId: Int
    public let eoiTokenId: Int
    public let boaTokenId: Int
    public let eoaTokenId: Int
    public let visionSoftTokensPerImage: Int
    public let visionSoftTokensPerVideoFrame: Int
    public let tieWordEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfig = "text_config"
        case visionConfig = "vision_config"
        case audioConfig = "audio_config"
        case imageTokenId = "image_token_id"
        case audioTokenId = "audio_token_id"
        case videoTokenId = "video_token_id"
        case boiTokenId = "boi_token_id"
        case eoiTokenId = "eoi_token_id"
        case boaTokenId = "boa_token_id"
        case eoaTokenIdRaw = "eoa_token_id"
        case eoaTokenIndex = "eoa_token_index"
        case visionSoftTokensPerImage = "vision_soft_tokens_per_image"
        case visionSoftTokensPerVideoFrame = "vision_soft_tokens_per_video_frame"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "gemma4_unified"
        textConfig = try c.decode(Gemma4TextConfig.self, forKey: .textConfig)
        visionConfig = try c.decodeIfPresent(Gemma4UnifiedVisionConfig.self, forKey: .visionConfig)
        audioConfig = try c.decodeIfPresent(Gemma4UnifiedAudioConfig.self, forKey: .audioConfig)
        imageTokenId = try c.decodeIfPresent(Int.self, forKey: .imageTokenId) ?? 258880
        audioTokenId = try c.decodeIfPresent(Int.self, forKey: .audioTokenId) ?? 258881
        videoTokenId = try c.decodeIfPresent(Int.self, forKey: .videoTokenId) ?? 258884
        boiTokenId = try c.decodeIfPresent(Int.self, forKey: .boiTokenId) ?? 255999
        eoiTokenId = try c.decodeIfPresent(Int.self, forKey: .eoiTokenId) ?? 258882
        boaTokenId = try c.decodeIfPresent(Int.self, forKey: .boaTokenId) ?? 256000
        // gemma4_unified utilise "eoa_token_index" si "eoa_token_id" absent
        let eoaRaw = try c.decodeIfPresent(Int.self, forKey: .eoaTokenIdRaw)
        let eoaIdx = try c.decodeIfPresent(Int.self, forKey: .eoaTokenIndex)
        eoaTokenId = eoaRaw ?? eoaIdx ?? 258883
        visionSoftTokensPerImage = try c.decodeIfPresent(Int.self, forKey: .visionSoftTokensPerImage) ?? 280
        visionSoftTokensPerVideoFrame = try c.decodeIfPresent(Int.self, forKey: .visionSoftTokensPerVideoFrame) ?? 70
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
    }
}
