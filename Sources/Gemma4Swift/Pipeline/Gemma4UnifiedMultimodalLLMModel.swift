// Modele multimodal gemma4_unified (12B) : language_model + vision_embedder + embed_vision + embed_audio.
//
// Difference avec [[Gemma4MultimodalLLMModel]] (E2B/E4B) :
//   - Pas de SigLIP : vision_embedder est un patch projector simple.
//   - Pas de Conformer : embed_audio prend directement les chunks PCM bruts.
//   - get_image_features : besoin de image_position_ids ; on slice les patches
//     par "validPatches" avant le masked_scatter pour ne pas injecter du padding.

import Foundation
import MLX
import MLXFast
import MLXNN
import MLXLMCommon
import MLXLLM

/// Modele Gemma 4 Unified multimodal (texte + vision + audio).
public class Gemma4UnifiedMultimodalLLMModel: Module, LLMModel, LoRAModel {
    public let config: Gemma4UnifiedConfig

    @ModuleInfo(key: "language_model") var languageModel: Gemma4LanguageModel
    @ModuleInfo(key: "vision_embedder") var visionEmbedder: Gemma4UnifiedVisionEmbedder?
    @ModuleInfo(key: "embed_vision") var embedVision: MultimodalEmbedder?
    @ModuleInfo(key: "embed_audio") var embedAudio: MultimodalEmbedder?

    public let modelType: String
    public var kvHeads: [Int]

    // --- Pending state injecte par le pipeline avant l'appel forward ---
    /// [B, N_pad, patchDim] patches flatten (B = nombre d'images stackees).
    public var pendingPixelPatches: MLXArray?
    /// [B, N_pad, 2] position ids (x, y), -1 = padding.
    public var pendingImagePositionIds: MLXArray?
    /// Nombre de patches VALIDES par image (== imageTokenCount par image).
    public var pendingValidPatches: [Int]?

    /// [F, N_pad, patchDim] frames video stackees (F = num frames).
    public var pendingVideoFramePatches: MLXArray?
    /// [F, N_pad, 2] position ids.
    public var pendingVideoFramePositionIds: MLXArray?
    /// Nombre de patches valides par frame.
    public var pendingVideoValidPatches: [Int]?

    /// [1, T, samplesPerToken] chunks PCM bruts.
    public var pendingAudioFeatures: MLXArray?
    /// [1, T] mask : true = valide, false = padding.
    public var pendingAudioMask: MLXArray?

    public init(config: Gemma4UnifiedConfig) {
        self.config = config
        self.modelType = config.modelType

        let textConfig = config.textConfig
        self._languageModel.wrappedValue = Gemma4LanguageModel(textConfig)
        self.kvHeads = Array(repeating: textConfig.numKeyValueHeads, count: textConfig.numHiddenLayers)

        // Vision (optionnelle)
        if let visionConfig = config.visionConfig {
            self._visionEmbedder.wrappedValue = Gemma4UnifiedVisionEmbedder(visionConfig)
            self._embedVision.wrappedValue = MultimodalEmbedder(
                embeddingDim: visionConfig.outputProjDims,
                textHiddenSize: textConfig.hiddenSize,
                eps: visionConfig.rmsNormEps
            )
        } else {
            self._visionEmbedder.wrappedValue = nil
            self._embedVision.wrappedValue = nil
        }

        // Audio (optionnelle)
        if let audioConfig = config.audioConfig {
            self._embedAudio.wrappedValue = MultimodalEmbedder(
                embeddingDim: audioConfig.outputProjDims,
                textHiddenSize: textConfig.hiddenSize,
                eps: audioConfig.rmsNormEps
            )
        } else {
            self._embedAudio.wrappedValue = nil
        }

        super.init()
    }

    // MARK: - LLMModel conformance

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let cacheArray: [KVCache?]? = cache?.map { $0 as KVCache? }
        // visionTokenMask : pour bidirectional overlay sur les blocs vision (T > 1).
        // Calcule TOUJOURS depuis inputs : meme apres prefill, le pattern image/video
        // reste exact (pas d'audio dans le mask = comportement Python identique).
        let visionMask: MLXArray? = computeVisionTokenMask(inputs)
        if let inputsEmbeds = prepareMultimodalEmbeds(inputs) {
            return languageModel(
                inputsEmbeds: inputsEmbeds,
                cache: cacheArray,
                visionTokenMask: visionMask
            )
        }
        return languageModel(
            inputs: inputs,
            cache: cacheArray,
            visionTokenMask: visionMask
        )
    }

    /// Variante de `callAsFunction` qui retourne logits + hidden states (pre-norm) +
    /// intermediates K/V — utilise par le path MTP speculative decoding.
    /// Symetrique a Gemma4MultimodalLLMModel.forwardWithIntermediates pour les
    /// futurs drafters publiables sur 12B Unified.
    public func forwardWithIntermediates(
        _ inputs: MLXArray,
        cache: [KVCache]?
    ) -> LanguageForwardOutput {
        let cacheArray: [KVCache?]? = cache?.map { $0 as KVCache? }
        let visionMask: MLXArray? = computeVisionTokenMask(inputs)
        if let inputsEmbeds = prepareMultimodalEmbeds(inputs) {
            return languageModel.forwardWithIntermediates(
                inputsEmbeds: inputsEmbeds,
                cache: cacheArray,
                visionTokenMask: visionMask
            )
        }
        return languageModel.forwardWithIntermediates(
            inputs: inputs,
            cache: cacheArray,
            visionTokenMask: visionMask
        )
    }

    /// Calcule un mask [B, T] bool : true ou le token est image OU video.
    /// Retourne nil si T==1 (decodage), ou si AUCUN token vision ni AUCUN audio
    /// (path text pur, pas de masking custom necessaire).
    ///
    /// Note : Si la sequence contient des tokens audio, on retourne nil pour ne pas
    /// appliquer l'overlay (cf. Python : "keep mixed image+audio prompts causal to
    /// avoid the vision block overlay dominating quantized unified models").
    private func computeVisionTokenMask(_ inputs: MLXArray) -> MLXArray? {
        let shape = inputs.shape
        let inputs2D = shape.count == 1 ? inputs.reshaped(1, -1) : inputs
        let T = inputs2D.dim(1)
        if T <= 1 { return nil }

        let isImage = inputs2D .== Int32(config.imageTokenId)
        let isVideo = inputs2D .== Int32(config.videoTokenId)
        let isAudio = inputs2D .== Int32(config.audioTokenId)

        let hasAudio = isAudio.asType(.int32).sum().item(Int.self) > 0
        let hasVision = (isImage .|| isVideo).asType(.int32).sum().item(Int.self) > 0
        if !hasVision || hasAudio { return nil }

        return isImage .|| isVideo
    }

    /// Construit les embeddings fusionnes (vision + audio) si du media est en attente.
    /// Retourne nil = mode text-only.
    private func prepareMultimodalEmbeds(_ inputs: MLXArray) -> MLXArray? {
        let hasImage = pendingPixelPatches != nil
        let hasVideo = pendingVideoFramePatches != nil
        let hasAudio = pendingAudioFeatures != nil
        guard hasImage || hasVideo || hasAudio else { return nil }

        var inputsEmbeds = languageModel.model.embedTokens(inputs)
        inputsEmbeds = inputsEmbeds * MLXArray(languageModel.model.embedScale, dtype: .float32)

        // --- Vision ---
        if let patches = pendingPixelPatches,
           let posIds = pendingImagePositionIds,
           let embedder = visionEmbedder,
           let proj = embedVision {
            let numImages = patches.dim(0)
            let valid = pendingValidPatches ?? Array(repeating: patches.dim(1), count: numImages)

            // Embedder s'applique sur le tenseur complet (padding inclus, ignore via valid mask).
            var features = embedder(patches, imagePositionIds: posIds) // [B, N_pad, mmEmbedDim]
            features = proj(features)                                    // [B, N_pad, textHidden]
            features = stopGradient(features)
            features = features.asType(inputsEmbeds.dtype)

            // Compacter : pour chaque image, garder uniquement valid[i] patches.
            var compactedRows: [MLXArray] = []
            for i in 0 ..< numImages {
                let v = valid[i]
                if v <= 0 { continue }
                compactedRows.append(features[i, 0 ..< v]) // [v, D]
            }
            if !compactedRows.isEmpty {
                let merged = concatenated(compactedRows, axis: 0) // [sum(v), D]
                // [1, sum(v), D] pour scatter sur input batch=1.
                let asBatch = expandedDimensions(merged, axis: 0)

                let imageMask = inputs .== Int32(config.imageTokenId)
                let imageMaskExp = broadcast(expandedDimensions(imageMask, axis: -1), to: inputsEmbeds.shape)
                inputsEmbeds = maskedScatter(input: inputsEmbeds, mask: imageMaskExp, source: asBatch)
            }

            pendingPixelPatches = nil
            pendingImagePositionIds = nil
            pendingValidPatches = nil
        }

        // --- Video frames (meme embedder, scatter sur video_token_id) ---
        if let patches = pendingVideoFramePatches,
           let posIds = pendingVideoFramePositionIds,
           let embedder = visionEmbedder,
           let proj = embedVision {
            let numFrames = patches.dim(0)
            let valid = pendingVideoValidPatches ?? Array(repeating: patches.dim(1), count: numFrames)

            var features = embedder(patches, imagePositionIds: posIds)
            features = proj(features)
            features = stopGradient(features)
            features = features.asType(inputsEmbeds.dtype)

            var compactedRows: [MLXArray] = []
            for i in 0 ..< numFrames {
                let v = valid[i]
                if v <= 0 { continue }
                compactedRows.append(features[i, 0 ..< v])
            }
            if !compactedRows.isEmpty {
                let merged = concatenated(compactedRows, axis: 0)
                let asBatch = expandedDimensions(merged, axis: 0)

                let videoMask = inputs .== Int32(config.videoTokenId)
                let videoMaskExp = broadcast(expandedDimensions(videoMask, axis: -1), to: inputsEmbeds.shape)
                inputsEmbeds = maskedScatter(input: inputsEmbeds, mask: videoMaskExp, source: asBatch)
            }

            pendingVideoFramePatches = nil
            pendingVideoFramePositionIds = nil
            pendingVideoValidPatches = nil
        }

        // --- Audio ---
        if let audioFeatures = pendingAudioFeatures,
           let proj = embedAudio {
            var projected = proj(audioFeatures) // [1, T, textHidden]
            projected = stopGradient(projected)
            projected = projected.asType(inputsEmbeds.dtype)

            // Compacter via mask (true = valide). Clamping defensif au cas ou
            // le mask aurait plus de samples valides que la projection (shape
            // mismatch silencieux).
            if let mask = pendingAudioMask {
                let rawCount = mask.asType(.int32).sum().item(Int.self)
                let validCount = min(rawCount, projected.dim(1))
                if validCount < projected.dim(1) {
                    projected = projected[0..., 0 ..< validCount]
                }
            }

            let audioMask = inputs .== Int32(config.audioTokenId)
            let audioMaskExp = broadcast(expandedDimensions(audioMask, axis: -1), to: inputsEmbeds.shape)
            inputsEmbeds = maskedScatter(input: inputsEmbeds, mask: audioMaskExp, source: projected)

            pendingAudioFeatures = nil
            pendingAudioMask = nil
        }

        return inputsEmbeds
    }

    public func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        languageModel.makeCache()
    }

    public var loraLayers: [Module] {
        languageModel.model.layers.map { $0 as Module }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        WeightSanitizer.sanitize(
            weights: weights,
            hasVision: config.visionConfig != nil,
            hasAudio: config.audioConfig != nil,
            useClippedLinears: false
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
