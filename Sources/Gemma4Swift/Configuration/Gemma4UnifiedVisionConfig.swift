// Config vision pour gemma4_unified (12B) — encoder-free patch embedder

import Foundation

/// Configuration du vision_embedder de gemma4_unified.
///
/// Contraste avec [[Gemma4VisionConfig]] (E2B/E4B) qui decrit un SigLIP complet :
/// ici l'encodeur est un simple patch-projector (LN -> Linear -> LN -> +pos -> LN).
public struct Gemma4UnifiedVisionConfig: Codable, Sendable {
    public let modelType: String
    /// Taille du patch "model" en pixels (48 par defaut).
    /// L'image est decoupee en patches de modelPatchSize x modelPatchSize x 3.
    public let modelPatchSize: Int
    /// Taille du patch fin (16) — utilise par le preprocessor pour aspect-ratio + pooling.
    public let patchSize: Int
    /// Kernel de pooling (3) — modelPatchSize = patchSize * poolingKernelSize.
    public let poolingKernelSize: Int
    /// Dimension d'embedding interne (3840 pour 12B).
    public let mmEmbedDim: Int
    /// Taille du tableau de pos_embedding par axe (1120) — supporte x ou y jusqu'a 1120.
    public let mmPosembSize: Int
    /// Nombre de soft tokens par image dans la sequence texte (280).
    public let numSoftTokens: Int
    public let outputProjDims: Int
    public let rmsNormEps: Float

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case modelPatchSize = "model_patch_size"
        case patchSize = "patch_size"
        case poolingKernelSize = "pooling_kernel_size"
        case mmEmbedDim = "mm_embed_dim"
        case mmPosembSize = "mm_posemb_size"
        case numSoftTokens = "num_soft_tokens"
        case outputProjDims = "output_proj_dims"
        case rmsNormEps = "rms_norm_eps"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "gemma4_unified_vision"
        modelPatchSize = try c.decodeIfPresent(Int.self, forKey: .modelPatchSize) ?? 48
        patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 16
        poolingKernelSize = try c.decodeIfPresent(Int.self, forKey: .poolingKernelSize) ?? 3
        mmEmbedDim = try c.decodeIfPresent(Int.self, forKey: .mmEmbedDim) ?? 3840
        mmPosembSize = try c.decodeIfPresent(Int.self, forKey: .mmPosembSize) ?? 1120
        numSoftTokens = try c.decodeIfPresent(Int.self, forKey: .numSoftTokens) ?? 280
        outputProjDims = try c.decodeIfPresent(Int.self, forKey: .outputProjDims) ?? (mmEmbedDim)
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
    }

    /// Dimension d'un patch flatten : modelPatchSize * modelPatchSize * 3
    public var patchDim: Int { modelPatchSize * modelPatchSize * 3 }

    /// Budget de patches **fins** (`patchSize`, 16 px), exprime dans la meme unite
    /// que le SigLIP de [[Gemma4VisionConfig]].
    ///
    /// Ne sert qu'a calculer le budget de pixels du resize
    /// (`maxPatches * patchSize^2`). **Ce n'est pas un nombre de lignes de
    /// tenseur** : les patches produits par [[Gemma4UnifiedImageProcessor]] sont
    /// des patches *modele* de `modelPatchSize` (48 px), soit 9 patches fins
    /// chacun. Pour un nombre de lignes, utiliser ``maxModelPatches``.
    public var maxPatches: Int { numSoftTokens * poolingKernelSize * poolingKernelSize }

    /// Nombre maximum de patches **modele** (`modelPatchSize`, 48 px) produits
    /// pour une image, donc le nombre de lignes du tenseur de patches et le
    /// nombre de soft tokens correspondants — un patch modele = un soft token.
    ///
    /// Vaut exactement `numSoftTokens` : le budget de pixels du resize est
    /// `maxPatches * patchSize^2`, soit `numSoftTokens * modelPatchSize^2` pixels,
    /// soit au plus `numSoftTokens` cellules de `modelPatchSize^2`.
    public var maxModelPatches: Int { numSoftTokens }
}
