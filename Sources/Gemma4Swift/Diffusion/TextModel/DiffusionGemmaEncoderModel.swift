// Port de modular_diffusion_gemma.DiffusionGemmaEncoderModel (lignes 813-944)
//
// Wrapper multimodal de l'encoder DiffusionGemma : vision_tower + embed_vision
// + language_model (text encoder).
//
// Forward :
//   1. image_mask = (input_ids == image_token_id)
//   2. inputs_embeds = embed_tokens(input_ids_avec_pad_au_lieu_d_image_token)
//   3. Si pixel_values fourni :
//        vision_features = vision_tower(pixel_values)  // [B, 280, 1152]
//        mm_features = embed_vision(vision_features)   // [B, 280, 2816]
//        inputs_embeds = masked_scatter(inputs_embeds, image_mask, mm_features)
//   4. language_model(inputsEmbeds) -> EncoderKVCache + last_hidden_state

import Foundation
import MLX
import MLXNN

/// Encoder multimodal de DiffusionGemma (vision + text).
public class DiffusionGemmaEncoderModel: Module {
    public let config: DiffusionGemmaConfig
    public let imageTokenId: Int

    @ModuleInfo(key: "vision_tower") public var visionTower: VisionModel?
    @ModuleInfo(key: "embed_vision") public var embedVision: MultimodalEmbedder?
    @ModuleInfo(key: "language_model") public var languageModel: DiffusionGemmaEncoderTextModel

    public init(_ config: DiffusionGemmaConfig) {
        self.config = config
        self.imageTokenId = config.imageTokenId

        if let vcfg = config.visionConfig {
            self._visionTower.wrappedValue = VisionModel(vcfg)
            self._embedVision.wrappedValue = MultimodalEmbedder(
                embeddingDim: vcfg.hiddenSize,
                textHiddenSize: config.textConfig.base.hiddenSize,
                eps: config.textConfig.base.rmsNormEps
            )
        } else {
            self._visionTower.wrappedValue = nil
            self._embedVision.wrappedValue = nil
        }

        self._languageModel.wrappedValue = DiffusionGemmaEncoderTextModel(config.textConfig)
        super.init()
    }

    /// Decharge les modules vision (vision_tower + embed_vision) une fois que
    /// les soft-tokens ont ete encodes dans l'EncoderKVCache. Pattern LTX
    /// `unloadAfterUse`. Libere ~600 MB (vision SigLIP 27 layers hidden 1152).
    ///
    /// A appeler APRES le 1er forward (qui place les soft-tokens vision dans
    /// le cache). Les forwards suivants peuvent etre incrementaux sur du texte
    /// pur sans vision.
    public func unloadVision() {
        self._visionTower.wrappedValue = nil
        self._embedVision.wrappedValue = nil
        MLX.Memory.clearCache()
    }

    /// True si le vision_tower est encore charge.
    public var hasVisionLoaded: Bool {
        visionTower != nil
    }

    /// Forward de l'encoder.
    ///
    /// - Parameters:
    ///   - inputIds : `[B, T]` int. Tokens du prompt (avec `imageTokenId` aux
    ///     positions a remplacer par les soft-tokens vision).
    ///   - pixelValues : `[B*nImages, 3, H, W]` float ou nil. Images preprocessees.
    ///   - priorCache : si fourni, encode SEULEMENT les nouveaux tokens. Le
    ///     traitement vision (vision_tower) n'est exécuté que si priorCache est nil
    ///     (les soft-tokens vision sont supposés déjà encodés dans le cache).
    public func callAsFunction(
        inputIds: MLXArray,
        pixelValues: MLXArray? = nil,
        priorCache: EncoderKVCache? = nil
    ) -> DiffusionEncoderOutput {
        // En mode incremental (priorCache != nil) : on suppose que la vision a
        // ete encodee au premier appel, donc on traite les inputIds comme du
        // texte pur (les nouveaux tokens sont du canvas argmax, pas d'image).
        let useVision = pixelValues != nil && priorCache == nil

        // 1) Mask des positions image_token AVANT de remplacer par pad
        let imageMask = inputIds .== MLXArray(Int32(imageTokenId))

        // 2) Remplace image_token par pad pour eviter l'OOV d'embedding
        let padTokenId = config.textConfig.base.vocabSize > imageTokenId ? imageTokenId : 0
        let safeInputIds: MLXArray
        if useVision {
            safeInputIds = MLX.where(imageMask, MLXArray(Int32(padTokenId)), inputIds)
        } else {
            safeInputIds = inputIds
        }

        // 3) Embeddings text
        var inputsEmbeds = languageModel.embedTokens(safeInputIds)
        inputsEmbeds = inputsEmbeds * MLXArray(languageModel.embedScale, dtype: inputsEmbeds.dtype)

        // 4) Vision : extrait soft-tokens et splice via masked_scatter (1er appel uniquement)
        if useVision,
           let pixelValues = pixelValues,
           let visionTower = visionTower,
           let embedVision = embedVision {
            let visionFeatures = visionTower(pixelValues)             // [B, 280, 1152]
            let mmFeatures = embedVision(visionFeatures)              // [B, 280, 2816]

            let maskExpanded = expandedDimensions(imageMask, axis: -1)
            let maskBroadcast = broadcast(maskExpanded, to: inputsEmbeds.shape)

            inputsEmbeds = maskedScatter(
                inputsEmbeds: inputsEmbeds,
                mask: maskBroadcast,
                source: mmFeatures
            )
        }

        // 5) Forward du language_model avec priorCache si fourni
        return languageModel(inputsEmbeds: inputsEmbeds, priorCache: priorCache)
    }

    /// Splice les valeurs `source` aux positions `mask=True` dans `inputsEmbeds`.
    /// Equivalent de `inputsEmbeds.masked_scatter(mask, source)` PyTorch.
    private func maskedScatter(
        inputsEmbeds: MLXArray,
        mask: MLXArray,
        source: MLXArray
    ) -> MLXArray {
        // source shape : [B, K, H] avec K = nb soft-tokens par image
        // mask shape : [B, T, H]
        // On veut placer source.flatten([B, K, H] -> [B*K, H]) aux positions mask=True.
        //
        // Approche simple : reshape source pour matcher [B, T, H] avec zeros ailleurs.
        // Mais c'est complique en MLX. On utilise un trick :
        //   1. Source aplati a [N_mask_true, H] (N_mask_true == B*K)
        //   2. positions_dans_T = cumsum(mask_bool) - 1 (indice cumul des positions True)
        //   3. Indexation : source[positions_dans_T] gather sur axe 0
        //   4. where(mask, gathered, inputsEmbeds)
        //
        // Pour Phase 5 vision initial : version simplifiee qui suppose
        // que mask est consecutif (tous les image_tokens sont contigus
        // dans la sequence), ce qui est le cas pour un seul block d'image.
        // Cas multi-images / mixe sera traite plus tard.

        let B = inputsEmbeds.dim(0)
        let H = inputsEmbeds.dim(2)
        // source aplati : [B, K*H] puis on broadcast aux positions mask
        // Trick : on ecrit source.reshape(B, -1, H) sur les positions mask.
        // Mais la maniere la plus robuste : utiliser une boucle batch + indexation.

        // Pour Phase 5 minimale, on traite batch=1 et on suppose tous les soft-tokens
        // sont a la suite. Cela couvre le cas "un prompt avec une image".
        precondition(B == 1, "maskedScatter multi-batch a implementer (Phase 6)")

        // Trouve les indices ou mask est True dans la sequence T (axis=1)
        // mask shape : [1, T, H] -> on regarde mask[0, :, 0] pour les positions
        let maskCol = mask[0, 0..., 0]  // [T] bool
        let positions = argSort(MLXArray(-1) * maskCol.asType(.int32), axis: 0)
        // positions[:K] = indices True, dans l'ordre. Puis on tronque a K.
        let K = source.dim(1)
        let targetIndices = positions[0 ..< K]  // [K]

        // Update inputsEmbeds[0, targetIndices, :] = source[0, :, :]
        // En MLX : putAlong
        let inputs0 = inputsEmbeds[0]  // [T, H]
        let source0 = source[0]        // [K, H]

        // Pour chaque k dans 0..K, ecrire inputs0[targetIndices[k], :] = source0[k, :]
        // Approche : utiliser putAlong sur axe 0 avec broadcast des indices.
        let idxExpanded = expandedDimensions(targetIndices, axis: -1)  // [K, 1]
        let idxBroadcast = broadcast(idxExpanded, to: [K, H])  // [K, H]
        let updated = putAlong(inputs0, idxBroadcast, values: source0, axis: 0)

        return expandedDimensions(updated, axis: 0)
    }
}
