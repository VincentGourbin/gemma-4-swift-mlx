// Modele multimodal LLM conforme au protocol pour chargement via mlx-swift-lm

import Foundation
import MLX
import MLXFast
import MLXNN
import MLXLMCommon
import MLXLLM

/// Modele Gemma 4 multimodal complet (texte + vision + audio)
/// Conforme a LLMModel pour l'enregistrement dans mlx-swift-lm.
public class Gemma4MultimodalLLMModel: Module, LLMModel, LoRAModel {
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

    // Embeddings pre-calculees (pour le training — evite de tracer les towers dans valueAndGrad)
    public var pendingImageEmbeddings: MLXArray?
    public var pendingAudioEmbeddings: MLXArray?

    // Video: frames separees des images, avec truncation a softTokensPerFrame
    public var pendingVideoFrames: MLXArray?
    public var pendingVideoSoftTokensPerFrame: Int?

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
        guard pendingPixelValues != nil || pendingVideoFrames != nil || pendingAudioFeatures != nil
              || pendingImageEmbeddings != nil || pendingAudioEmbeddings != nil else {
            return languageModel(inputs: inputs, cache: cacheArray)
        }

        // Mode multimodal: construire les embeddings fusionnes
        var inputsEmbeds = languageModel.model.embedTokens(inputs)
        inputsEmbeds = inputsEmbeds * MLXArray(languageModel.model.embedScale, dtype: .float32)

        // Per-layer inputs (masquer tokens image/audio)
        var perLayerInputs: MLXArray? = nil
        if languageModel.model.hiddenSizePerLayerInput > 0 {
            let imageMask = inputs .== Int32(config.imageTokenId)
            let videoMaskIds = inputs .== Int32(config.videoTokenId)
            let audioMaskIds = inputs .== Int32(config.audioTokenId)
            let textMask = logicalNot(imageMask .|| videoMaskIds .|| audioMaskIds)
            let maskedIds = MLX.where(textMask, inputs, MLXArray.zeros(like: inputs))
            perLayerInputs = languageModel.model.getPerLayerInputs(maskedIds)
        }

        // Vision: utiliser les embeddings pre-calculees si disponibles (training)
        // ou encoder via le vision tower (inference)
        if let precomputed = pendingImageEmbeddings {
            var imageFeatures = precomputed.asType(inputsEmbeds.dtype)
            let imageMask = inputs .== Int32(config.imageTokenId)
            let imageMaskExpanded = broadcast(expandedDimensions(imageMask, axis: -1), to: inputsEmbeds.shape)
            inputsEmbeds = maskedScatter(input: inputsEmbeds, mask: imageMaskExpanded, source: imageFeatures)
            pendingImageEmbeddings = nil
        } else if let pixelValues = pendingPixelValues {
            let numImages = pixelValues.dim(0)
            var allFeatures: [MLXArray] = []

            for i in 0 ..< numImages {
                let singleImage = pixelValues[i ..< (i + 1)] // [1, C, H, W]
                var features = visionTower(singleImage) // [1, 280, dim]
                features = embedVision(features)
                allFeatures.append(features)
            }

            // Concatener: [1, numImages*280, dim]
            var imageFeatures = concatenated(allFeatures, axis: 1)
            // stopGradient: le vision tower est frozen, pas besoin de backprop
            imageFeatures = stopGradient(imageFeatures)
            imageFeatures = imageFeatures.asType(inputsEmbeds.dtype)


            let imageMask = inputs .== Int32(config.imageTokenId)
            let imageMaskExpanded = broadcast(expandedDimensions(imageMask, axis: -1), to: inputsEmbeds.shape)

            inputsEmbeds = maskedScatter(input: inputsEmbeds, mask: imageMaskExpanded, source: imageFeatures)

            pendingPixelValues = nil
        }

        // Video: traiter chaque frame via vision encoder, tronquer a softTokensPerFrame (70)
        if let videoFrames = pendingVideoFrames {
            let softTokens = pendingVideoSoftTokensPerFrame ?? 70
            let numFrames = videoFrames.dim(0)
            var allVideoFeatures: [MLXArray] = []

            for i in 0 ..< numFrames {
                let singleFrame = videoFrames[i ..< (i + 1)] // [1, C, H, W]
                var features = visionTower(singleFrame) // [1, 280, dim]
                features = embedVision(features)
                // Tronquer a softTokensPerFrame (70) — seuls les premiers tokens sont valides
                features = features[0..., 0 ..< softTokens]
                allVideoFeatures.append(features)
            }

            // Concatener: [1, numFrames*softTokens, dim]
            var videoFeatures = concatenated(allVideoFeatures, axis: 1)
            videoFeatures = stopGradient(videoFeatures)
            videoFeatures = videoFeatures.asType(inputsEmbeds.dtype)

            let videoMask = inputs .== Int32(config.videoTokenId)
            let videoMaskExpanded = broadcast(expandedDimensions(videoMask, axis: -1), to: inputsEmbeds.shape)

            inputsEmbeds = maskedScatter(input: inputsEmbeds, mask: videoMaskExpanded, source: videoFeatures)
            pendingVideoFrames = nil
            pendingVideoSoftTokensPerFrame = nil
        }

        // Audio: utiliser les embeddings pre-calculees si disponibles (training)
        // ou encoder via l'audio tower (inference)
        if let precomputed = pendingAudioEmbeddings {
            var audioEmbeds = precomputed.asType(inputsEmbeds.dtype)
            let audioTokenMask = inputs .== Int32(config.audioTokenId)
            let numAudioTokens = audioTokenMask.sum().item(Int.self)
            let numAudioEmbeds = audioEmbeds.dim(1)
            if numAudioEmbeds != numAudioTokens && numAudioTokens > 0 && numAudioEmbeds > numAudioTokens {
                audioEmbeds = audioEmbeds[0..., 0 ..< numAudioTokens]
            }
            let audioMaskExpanded = broadcast(expandedDimensions(audioTokenMask, axis: -1), to: inputsEmbeds.shape)
            inputsEmbeds = maskedScatter(input: inputsEmbeds, mask: audioMaskExpanded, source: audioEmbeds)
            pendingAudioEmbeddings = nil
        } else if let audioFeatures = pendingAudioFeatures, let tower = audioTower, let embedder = embedAudio {
            let mask = pendingAudioMask ?? MLXArray.zeros([audioFeatures.dim(0), audioFeatures.dim(1)], type: Bool.self)
            let (audioEncodings, _) = tower(audioFeatures, audioMelMask: mask)
            var audioEmbeds = embedder(audioEncodings)
            // stopGradient: l'audio tower est frozen, pas besoin de backprop
            audioEmbeds = stopGradient(audioEmbeds)
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
