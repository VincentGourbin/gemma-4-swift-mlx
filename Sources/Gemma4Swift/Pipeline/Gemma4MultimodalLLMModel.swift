// Modele multimodal LLM conforme au protocol pour chargement via mlx-swift-lm

import Foundation
import MLX
import MLXFast
import MLXNN
import MLXLMCommon
import MLXLLM

/// Modele Gemma 4 multimodal complet (texte + vision + audio)
/// Conforme a LLMModel pour l'enregistrement dans mlx-swift-lm.
public class Gemma4MultimodalLLMModel: Module, LLMModel {
    public let config: Gemma4Config

    @ModuleInfo(key: "language_model") var languageModel: Gemma4LanguageModel
    @ModuleInfo(key: "vision_tower") var visionTower: VisionModel
    @ModuleInfo(key: "embed_vision") var embedVision: MultimodalEmbedder
    @ModuleInfo(key: "audio_tower") var audioTower: AudioEncoder?
    @ModuleInfo(key: "embed_audio") var embedAudio: MultimodalEmbedder?

    public let modelType: String
    public var kvHeads: [Int]

    // Stockage temporaire des inputs multimodaux pour le forward pass
    // (le protocol LLMModel ne permet pas de passer des pixel_values directement)
    public var pendingPixelValues: MLXArray?
    public var pendingAudioFeatures: MLXArray?
    public var pendingAudioMask: MLXArray?

    public init(config: Gemma4Config) {
        self.config = config
        self.modelType = config.modelType

        let textConfig = config.textConfig
        self._languageModel.wrappedValue = Gemma4LanguageModel(textConfig)
        self.kvHeads = Array(repeating: textConfig.numKeyValueHeads, count: textConfig.numHiddenLayers)

        // Vision
        let visionConfig = config.visionConfig ?? Gemma4VisionConfig.defaultConfig
        self._visionTower.wrappedValue = VisionModel(visionConfig)
        self._embedVision.wrappedValue = MultimodalEmbedder(
            embeddingDim: visionConfig.hiddenSize,
            textHiddenSize: textConfig.hiddenSize,
            eps: visionConfig.rmsNormEps
        )

        // Audio (optionnel — 26B-A4B et 31B n'ont pas d'audio)
        if let audioConfig = config.audioConfig {
            let audioOutputDim = audioConfig.outputProjDims ?? audioConfig.hiddenSize
            self._audioTower.wrappedValue = AudioEncoder(audioConfig)
            self._embedAudio.wrappedValue = MultimodalEmbedder(
                embeddingDim: audioOutputDim,
                textHiddenSize: textConfig.hiddenSize,
                eps: audioConfig.rmsNormEps
            )
        } else {
            self._audioTower.wrappedValue = nil
            self._embedAudio.wrappedValue = nil
        }

        super.init()
    }

    // MARK: - LLMModel conformance

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let cacheArray: [KVCache?]? = cache?.map { $0 as KVCache? }

        // Si pas de media en attente, mode texte simple
        guard pendingPixelValues != nil || pendingAudioFeatures != nil else {
            return languageModel(inputs: inputs, cache: cacheArray)
        }

        // Mode multimodal: construire les embeddings fusionnes
        var inputsEmbeds = languageModel.model.embedTokens(inputs)
        inputsEmbeds = inputsEmbeds * MLXArray(languageModel.model.embedScale, dtype: .float32)

        // Per-layer inputs (masquer tokens image/audio)
        var perLayerInputs: MLXArray? = nil
        if languageModel.model.hiddenSizePerLayerInput > 0 {
            let imageMask = inputs .== Int32(config.imageTokenId)
            let audioMaskIds = inputs .== Int32(config.audioTokenId)
            let textMask = logicalNot(imageMask .|| audioMaskIds)
            let maskedIds = MLX.where(textMask, inputs, MLXArray.zeros(like: inputs))
            perLayerInputs = languageModel.model.getPerLayerInputs(maskedIds)
        }

        // Vision: traiter chaque image/frame individuellement puis scatter
        if let pixelValues = pendingPixelValues {
            let numImages = pixelValues.dim(0)
            var allFeatures: [MLXArray] = []

            // Traiter chaque image/frame separement dans le vision encoder
            for i in 0 ..< numImages {
                let singleImage = pixelValues[i ..< (i + 1)] // [1, C, H, W]
                var features = visionTower(singleImage) // [1, 280, dim]
                features = embedVision(features)
                allFeatures.append(features)
            }

            // Concatener: [1, numImages*280, dim]
            var imageFeatures = concatenated(allFeatures, axis: 1)
            imageFeatures = imageFeatures.asType(inputsEmbeds.dtype)

            let imageMask = inputs .== Int32(config.imageTokenId)
            let imageMaskExpanded = broadcast(expandedDimensions(imageMask, axis: -1), to: inputsEmbeds.shape)

            inputsEmbeds = maskedScatter(input: inputsEmbeds, mask: imageMaskExpanded, source: imageFeatures)
            pendingPixelValues = nil
        }

        // Audio: scatter audio features (seulement si le modele a un audio tower)
        if let audioFeatures = pendingAudioFeatures, let tower = audioTower, let embedder = embedAudio {
            let mask = pendingAudioMask ?? MLXArray.zeros([audioFeatures.dim(0), audioFeatures.dim(1)], type: Bool.self)
            let (audioEncodings, _) = tower(audioFeatures, audioMelMask: mask)
            var audioEmbeds = embedder(audioEncodings)
            audioEmbeds = audioEmbeds.asType(inputsEmbeds.dtype)

            let audioTokenMask = inputs .== Int32(config.audioTokenId)
            let numAudioTokens = audioTokenMask.sum().item(Int.self)
            let numAudioEmbeds = audioEmbeds.dim(1)

            // Ajuster si le nombre d'embeds ne correspond pas aux tokens
            if numAudioEmbeds != numAudioTokens && numAudioTokens > 0 && numAudioEmbeds > numAudioTokens {
                audioEmbeds = audioEmbeds[0..., 0 ..< numAudioTokens]
            }

            let audioMaskExpanded = broadcast(expandedDimensions(audioTokenMask, axis: -1), to: inputsEmbeds.shape)
            inputsEmbeds = maskedScatter(input: inputsEmbeds, mask: audioMaskExpanded, source: audioEmbeds)
            pendingAudioFeatures = nil
            pendingAudioMask = nil
        }

        return languageModel(
            inputsEmbeds: inputsEmbeds,
            cache: cacheArray,
            perLayerInputs: perLayerInputs
        )
    }

    public func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        languageModel.makeCache()
    }

    public var loraLayers: [Module] {
        languageModel.model.layers.map { $0 as Module }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // Auto-detect clipped linears: si les poids contiennent output_max dans le vision tower,
        // on les garde meme si la config dit false (modeles MLX community pre-quantises)
        let hasClippedWeights = weights.keys.contains { $0.contains("vision_tower") && $0.contains("output_max") }
        let useClipped = hasClippedWeights || (config.visionConfig?.useClippedLinears ?? false)

        return WeightSanitizer.sanitize(
            weights: weights,
            hasVision: true,
            hasAudio: config.audioConfig != nil,
            useClippedLinears: useClipped
        )
    }

    public func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int? = nil) throws -> PrepareResult {
        let promptTokens = input.text.tokens
        guard promptTokens.shape[0] > 0 else {
            let emptyToken = MLXArray(Int32(0))[0 ..< 0]
            return .tokens(.init(tokens: emptyToken))
        }
        return .tokens(input.text)
    }
}
