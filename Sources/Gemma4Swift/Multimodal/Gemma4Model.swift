// Port de gemma4.py Model — Modele multimodal complet (texte + vision + audio)

import Foundation
import MLX
import MLXNN
import MLXLMCommon

/// Modele Gemma 4 multimodal complet
public class Gemma4MultimodalModel: Module {
    let config: Gemma4Config

    @ModuleInfo(key: "language_model") var languageModel: Gemma4LanguageModel
    @ModuleInfo(key: "vision_tower") var visionTower: VisionModel?
    @ModuleInfo(key: "embed_vision") var embedVision: MultimodalEmbedder?

    // Audio (a implementer en Phase 4)
    // @ModuleInfo(key: "audio_tower") var audioTower: AudioEncoder?
    // @ModuleInfo(key: "embed_audio") var embedAudio: MultimodalEmbedder?

    public init(_ config: Gemma4Config) {
        self.config = config

        self._languageModel.wrappedValue = Gemma4LanguageModel(config.textConfig)

        // Vision
        if let visionConfig = config.visionConfig {
            self._visionTower.wrappedValue = VisionModel(visionConfig)
            self._embedVision.wrappedValue = MultimodalEmbedder(
                embeddingDim: visionConfig.hiddenSize,
                textHiddenSize: config.textConfig.hiddenSize,
                eps: visionConfig.rmsNormEps
            )
        }

        super.init()
    }

    /// Construit les embeddings d'entree en fusionnant texte + vision + audio
    public func getInputEmbeddings(
        inputIds: MLXArray,
        pixelValues: MLXArray? = nil,
        audioFeatures: MLXArray? = nil,
        audioMask: MLXArray? = nil
    ) -> (inputsEmbeds: MLXArray, perLayerInputs: MLXArray?) {
        // Token embeddings texte
        var inputsEmbeds = languageModel.model.embedTokens(inputIds)
        inputsEmbeds = inputsEmbeds * MLXArray(languageModel.model.embedScale, dtype: .float32)

        // Per-layer inputs (masquer les tokens image/audio)
        var perLayerInputs: MLXArray? = nil
        if languageModel.model.hiddenSizePerLayerInput > 0 {
            let imageMask = inputIds .== Int32(config.imageTokenId)
            let audioMaskIds = inputIds .== Int32(config.audioTokenId)
            let textMask = logicalNot(imageMask .|| audioMaskIds)
            let maskedIds = MLX.where(textMask, inputIds, MLXArray.zeros(like: inputIds))
            perLayerInputs = languageModel.model.getPerLayerInputs(maskedIds)
        }

        // Vision: scatter image features aux positions image_token_id
        if let pixelValues = pixelValues, let tower = visionTower, let embedder = embedVision {
            var imageFeatures = tower(pixelValues)
            imageFeatures = embedder(imageFeatures)
            imageFeatures = imageFeatures.asType(inputsEmbeds.dtype)

            let imageMask = inputIds .== Int32(config.imageTokenId)
            let imageMaskExpanded = broadcast(expandedDimensions(imageMask, axis: -1), to: inputsEmbeds.shape)

            inputsEmbeds = maskedScatter(input: inputsEmbeds, mask: imageMaskExpanded, source: imageFeatures)
        }

        return (inputsEmbeds, perLayerInputs)
    }

    /// Forward pass complet
    public func callAsFunction(
        inputIds: MLXArray,
        pixelValues: MLXArray? = nil,
        cache: [KVCache?]? = nil
    ) -> MLXArray {
        let (inputsEmbeds, perLayerInputs) = getInputEmbeddings(
            inputIds: inputIds,
            pixelValues: pixelValues
        )

        return languageModel(
            inputsEmbeds: inputsEmbeds,
            cache: cache,
            perLayerInputs: perLayerInputs
        )
    }
}
