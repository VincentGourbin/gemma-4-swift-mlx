// Port de mlx-vlm/models/gemma4/config.py ModelConfig

import Foundation

/// Configuration top-level du modele Gemma 4 (multimodal)
public struct Gemma4Config: Codable {
    public let modelType: String
    public let textConfig: Gemma4TextConfig
    public let visionConfig: Gemma4VisionConfig?
    public let audioConfig: Gemma4AudioConfig?
    public let imageTokenId: Int
    public let audioTokenId: Int
    public let videoTokenId: Int
    public let boiTokenId: Int
    public let eoiTokenId: Int
    public let boaTokenId: Int
    public let eoaTokenId: Int
    public let visionSoftTokensPerImage: Int
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
        case eoaTokenId = "eoa_token_id"
        case visionSoftTokensPerImage = "vision_soft_tokens_per_image"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try container.decode(String.self, forKey: .modelType)

        // text_config peut etre imbriquee ou au niveau racine
        if container.contains(.textConfig) {
            textConfig = try container.decode(Gemma4TextConfig.self, forKey: .textConfig)
        } else {
            // Decoder directement depuis la racine
            textConfig = try Gemma4TextConfig(from: decoder)
        }

        visionConfig = try container.decodeIfPresent(Gemma4VisionConfig.self, forKey: .visionConfig)
        audioConfig = try container.decodeIfPresent(Gemma4AudioConfig.self, forKey: .audioConfig)
        imageTokenId = try container.decodeIfPresent(Int.self, forKey: .imageTokenId) ?? 258880
        audioTokenId = try container.decodeIfPresent(Int.self, forKey: .audioTokenId) ?? 258881
        videoTokenId = try container.decodeIfPresent(Int.self, forKey: .videoTokenId) ?? 258884
        boiTokenId = try container.decodeIfPresent(Int.self, forKey: .boiTokenId) ?? 255999
        eoiTokenId = try container.decodeIfPresent(Int.self, forKey: .eoiTokenId) ?? 258882
        boaTokenId = try container.decodeIfPresent(Int.self, forKey: .boaTokenId) ?? 256000
        eoaTokenId = try container.decodeIfPresent(Int.self, forKey: .eoaTokenId) ?? 258883
        visionSoftTokensPerImage = try container.decodeIfPresent(Int.self, forKey: .visionSoftTokensPerImage) ?? 280
        tieWordEmbeddings = try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
    }
}
