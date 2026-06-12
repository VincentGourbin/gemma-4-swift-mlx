// Port de configuration_diffusion_gemma.DiffusionGemmaConfig
//
// Top-level config pour `diffusion_gemma` (le model_type du checkpoint 26B-A4B).
// Tres proche de Gemma4UnifiedConfig : meme vision encoder SigLIP, memes special
// tokens, meme tie_word_embeddings.

import Foundation

/// Top-level config DiffusionGemma (model_type = "diffusion_gemma").
public struct DiffusionGemmaConfig: Decodable, @unchecked Sendable {
    public let modelType: String
    public let textConfig: DiffusionGemmaTextConfig
    public let visionConfig: Gemma4UnifiedVisionConfig?

    // Tokens speciaux (identiques a gemma4_unified)
    public let imageTokenId: Int
    public let boiTokenId: Int
    public let eoiTokenId: Int
    public let visionSoftTokensPerImage: Int
    public let tieWordEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfig = "text_config"
        case visionConfig = "vision_config"
        case imageTokenId = "image_token_id"
        case boiTokenId = "boi_token_id"
        case eoiTokenId = "eoi_token_id"
        case visionSoftTokensPerImage = "vision_soft_tokens_per_image"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "diffusion_gemma"
        textConfig = try c.decode(DiffusionGemmaTextConfig.self, forKey: .textConfig)
        visionConfig = try c.decodeIfPresent(Gemma4UnifiedVisionConfig.self, forKey: .visionConfig)
        imageTokenId = try c.decodeIfPresent(Int.self, forKey: .imageTokenId) ?? 258880
        boiTokenId = try c.decodeIfPresent(Int.self, forKey: .boiTokenId) ?? 255999
        eoiTokenId = try c.decodeIfPresent(Int.self, forKey: .eoiTokenId) ?? 258882
        visionSoftTokensPerImage = try c.decodeIfPresent(Int.self, forKey: .visionSoftTokensPerImage) ?? 280
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
    }
}
