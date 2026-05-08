// Configuration du modele drafter Gemma 4 Assistant (MTP)
//
// Le modele Assistant est un drafter de speculative decoding qui se branche sur
// un modele cible (gemma-4-e2b-it / gemma-4-e4b-it). Il consomme les hidden
// states du target via pre_projection (2*backbone -> hidden) et reprojette ses
// sorties vers l'espace du target via post_projection (hidden -> backbone).
//
// Reference: google/gemma-4-E2B-it-assistant config.json
// Implementation Python de reference: mlx-vlm/speculative/drafters/gemma4_assistant

import Foundation

/// Configuration du modele drafter Gemma 4 Assistant
public struct Gemma4AssistantConfig: Codable {
    public let modelType: String
    public let textConfig: Gemma4TextConfig

    /// Hidden size du modele cible (backbone). Le drafter projette
    /// concat(target_embed, target_hidden) [2 * backboneHiddenSize] -> textConfig.hiddenSize.
    public let backboneHiddenSize: Int

    /// Nombre de centroides pour le MaskedEmbedder (sparse softmax sur le vocab).
    public let numCentroids: Int

    /// Top-K centroides selectionnes pour le calcul de logits sparse.
    public let centroidIntermediateTopK: Int

    /// Si true, utiliser le MaskedEmbedder a centroides plutot que tied/lm_head pour les logits.
    public let useOrderedEmbeddings: Bool

    /// Word embeddings partages entre l'embedding et la projection de sortie (top-level).
    public let tieWordEmbeddings: Bool

    /// Tokens speciaux multimodaux (passes par le target, pas utilises par le drafter
    /// directement, mais conserves pour symetrie avec Gemma4Config).
    public let imageTokenId: Int
    public let audioTokenId: Int
    public let boiTokenId: Int
    public let eoiTokenId: Int
    public let boaTokenId: Int
    public let eoaTokenId: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfig = "text_config"
        case backboneHiddenSize = "backbone_hidden_size"
        case numCentroids = "num_centroids"
        case centroidIntermediateTopK = "centroid_intermediate_top_k"
        case useOrderedEmbeddings = "use_ordered_embeddings"
        case tieWordEmbeddings = "tie_word_embeddings"
        case imageTokenId = "image_token_id"
        case audioTokenId = "audio_token_id"
        case boiTokenId = "boi_token_id"
        case eoiTokenId = "eoi_token_id"
        case boaTokenId = "boa_token_id"
        case eoaTokenId = "eoa_token_id"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decode(String.self, forKey: .modelType)
        textConfig = try c.decode(Gemma4TextConfig.self, forKey: .textConfig)
        backboneHiddenSize = try c.decode(Int.self, forKey: .backboneHiddenSize)
        numCentroids = try c.decodeIfPresent(Int.self, forKey: .numCentroids) ?? 0
        centroidIntermediateTopK = try c.decodeIfPresent(Int.self, forKey: .centroidIntermediateTopK) ?? 0
        useOrderedEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .useOrderedEmbeddings) ?? false
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        imageTokenId = try c.decodeIfPresent(Int.self, forKey: .imageTokenId) ?? 258880
        audioTokenId = try c.decodeIfPresent(Int.self, forKey: .audioTokenId) ?? 258881
        boiTokenId = try c.decodeIfPresent(Int.self, forKey: .boiTokenId) ?? 255999
        eoiTokenId = try c.decodeIfPresent(Int.self, forKey: .eoiTokenId) ?? 258882
        boaTokenId = try c.decodeIfPresent(Int.self, forKey: .boaTokenId) ?? 256000
        eoaTokenId = try c.decodeIfPresent(Int.self, forKey: .eoaTokenId) ?? 258883
    }
}
