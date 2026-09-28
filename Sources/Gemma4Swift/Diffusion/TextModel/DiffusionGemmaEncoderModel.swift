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
        // Affecter nil a la propriete @ModuleInfo est fatal dans MLX (« please use
        // Model.update(modules:) rather than mutating the Module property directly ») :
        // c'est ce qui faisait planter le canvas suivant (D-10). On garde les modules et
        // on remplace leurs parametres par des tableaux vides, ce qui libere les tampons.
        for module in [visionTower as Module?, embedVision as Module?].compactMap({ $0 }) {
            let empty = module.parameters().flattened().map { ($0.0, MLXArray.zeros([0])) }
            module.update(parameters: ModuleParameters.unflattened(empty))
        }
        visionUnloaded = true
        MLX.Memory.clearCache()
    }

    /// Vrai apres `unloadVision()` : la tour vision ne doit plus etre appelee.
    public private(set) var visionUnloaded = false

    /// True si le vision_tower est encore charge.
    public var hasVisionLoaded: Bool {
        visionTower != nil && !visionUnloaded
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
        let useVision = pixelValues != nil && priorCache == nil && !visionUnloaded

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

            // maskedScatter generique (chemin AR) : positions remplies dans l'ordre,
            // plusieurs images et B > 1 compris. L'ancienne version locale exigeait B == 1
            // et ne placait que la premiere image (D-09) ; la coherence nombre de jetons
            // image / nombre d'images est verifiee en tete de generate.
            inputsEmbeds = Gemma4Swift.maskedScatter(
                input: inputsEmbeds, mask: maskBroadcast, source: mmFeatures.asType(inputsEmbeds.dtype))
        }

        // 5) Forward du language_model avec priorCache si fourni
        // Blocs image bidirectionnels en mode "vision" (D-06).
        return languageModel(
            inputsEmbeds: inputsEmbeds, priorCache: priorCache,
            visionTokenMask: useVision ? imageMask : nil)
    }
}
