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
    @ModuleInfo(key: "vision_tower") public var visionTower: VisionModel
    @ModuleInfo(key: "embed_vision") var embedVision: MultimodalEmbedder
    @ModuleInfo(key: "audio_tower") public var audioTower: AudioEncoder?
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

    // Residence par etape (K-43) : tours liberees apres le prefill, rechargees a la demande.
    /// Liberer les tours vision/audio une fois un prefill avec media termine.
    public var releaseEncodersAfterPrefill = false
    /// Dossier du checkpoint (pose par `Gemma4Registration.loadContainer`), pour recharger.
    public var weightsDirectory: URL?
    /// Vrai quand les tours ont ete liberees et pas encore rechargees.
    public private(set) var encodersReleased = false

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
        let (inputsEmbeds, perLayerInputs) = prepareMultimodalEmbeds(inputs)
        if let inputsEmbeds = inputsEmbeds {
            return languageModel(
                inputsEmbeds: inputsEmbeds,
                cache: cacheArray,
                perLayerInputs: perLayerInputs
            )
        }
        return languageModel(inputs: inputs, cache: cacheArray)
    }

    /// Logits des positions `from...` seulement (entrainement masque, K-30 a).
    public func logits(_ inputs: MLXArray, from: Int) -> MLXArray {
        let (inputsEmbeds, perLayerInputs) = prepareMultimodalEmbeds(inputs)
        if let inputsEmbeds {
            return languageModel(inputsEmbeds: inputsEmbeds, cache: nil, perLayerInputs: perLayerInputs, logitsFrom: from)
        }
        return languageModel(inputs: inputs, cache: nil, logitsFrom: from)
    }

    /// Variante de `callAsFunction` qui retourne logits + hidden states (pre-norm) +
    /// intermediates K/V — utilise par le path MTP speculative decoding.
    public func forwardWithIntermediates(
        _ inputs: MLXArray,
        cache: [KVCache]?
    ) -> LanguageForwardOutput {
        let cacheArray: [KVCache?]? = cache?.map { $0 as KVCache? }
        let (inputsEmbeds, perLayerInputs) = prepareMultimodalEmbeds(inputs)
        if let inputsEmbeds = inputsEmbeds {
            return languageModel.forwardWithIntermediates(
                inputsEmbeds: inputsEmbeds,
                cache: cacheArray,
                perLayerInputs: perLayerInputs
            )
        }
        return languageModel.forwardWithIntermediates(inputs: inputs, cache: cacheArray)
    }

    /// Construit les embeddings fusionnes (vision/video/audio) si du media est en attente.
    /// Retourne (nil, nil) si pas de media — signal pour utiliser le path text-only.
    /// Mute pendingX en nil apres consommation.
    private func prepareMultimodalEmbeds(_ inputs: MLXArray) -> (MLXArray?, MLXArray?) {
        guard pendingPixelValues != nil || pendingVideoFrames != nil || pendingAudioFeatures != nil
              || pendingImageEmbeddings != nil || pendingAudioEmbeddings != nil else {
            return (nil, nil)
        }

        // Mode multimodal: construire les embeddings fusionnes
        var inputsEmbeds = languageModel.model.embedTokens(inputs)
        inputsEmbeds = inputsEmbeds * MLXArray(languageModel.model.embedScale, dtype: inputsEmbeds.dtype)

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
            let imageFeatures = precomputed.asType(inputsEmbeds.dtype)
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

        return (inputsEmbeds, perLayerInputs)
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
            useClippedLinears: useClipped,
            firstKvSharedLayerIdx: config.textConfig.firstKvSharedLayerIdx
        )
    }

    /// Prefill par tranches (voir `Gemma4ChunkedPrefill`). Avec un media en attente,
    /// les embeddings fusionnes (tours + masked_scatter) sont calcules une fois sur
    /// tout le prompt, puis le modele de langage avance par tranches d'embeddings ;
    /// le dernier jeton (texte : fin du gabarit) est rendu au `TokenIterator`.
    /// Exigence du protocole depuis mlx-swift-lm > 3.31.4 (mlx-swift 0.32) : sans elle, le
    /// `prepare` par defaut de `LLMModel` prendrait la place du notre (et sauterait les medias).
    public func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        // Sans tour audio (chargement `audio: false`), l'audio en attente serait ignore
        // en silence par `prepareMultimodalEmbeds` : refuser plutot que repondre sans.
        if pendingAudioFeatures != nil && audioTower == nil {
            pendingAudioFeatures = nil
            pendingAudioMask = nil
            throw Gemma4PipelineError.audioTowerUnavailable
        }
        let promptTokens = input.text.tokens
        let promptCount = promptTokens.shape[0]
        guard promptCount > 0 else {
            let emptyToken = MLXArray(Int32(0))[0 ..< 0]
            return .tokens(.init(tokens: emptyToken))
        }

        let cacheArray: [KVCache?] = cache.map { $0 as KVCache? }
        if hasPendingMedia { try restoreEncodersIfNeeded() }
        let (inputsEmbeds, perLayerInputs) = prepareMultimodalEmbeds(promptTokens[.newAxis])
        if let inputsEmbeds {
            try Gemma4ChunkedPrefill.run(count: promptCount, prefill: prefill, cache: cache) { range in
                _ = languageModel(
                    inputsEmbeds: inputsEmbeds[0..., range],
                    cache: cacheArray,
                    perLayerInputs: perLayerInputs?[0..., range]
                )
            }
            if releaseEncodersAfterPrefill {
                // Les soft tokens sont dans les caches : evaluer, puis liberer les tours.
                eval(cache)
                releaseEncoders()
            }
        } else {
            try Gemma4ChunkedPrefill.run(count: promptCount, prefill: prefill, cache: cache) { range in
                _ = languageModel(inputs: promptTokens[range][.newAxis], cache: cacheArray)
            }
        }
        return .tokens(input.text[(promptCount - 1)...])
    }

    /// Ancienne signature (pas seul), gardee pour les appelants directs.
    public func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int? = nil) throws -> PrepareResult {
        try prepare(input, cache: cache, state: nil, prefill: .init(stepSize: windowSize))
    }

    // MARK: - Residence par etape (K-43)

    private var hasPendingMedia: Bool {
        pendingPixelValues != nil || pendingVideoFrames != nil || pendingAudioFeatures != nil
    }

    private var encoderModules: [Module] {
        [visionTower, embedVision, audioTower, embedAudio].compactMap { $0 }
    }

    private static let encoderPrefixes = ["vision_tower.", "embed_vision.", "audio_tower.", "embed_audio."]

    /// Libere les tours (poids remplaces par des tableaux vides). Sans effet si une tour est
    /// quantifiee (quantification a la volee : on ne saurait pas la recharger a l'identique)
    /// ou si le dossier du modele est inconnu.
    public func releaseEncoders() {
        guard !encodersReleased, let directory = weightsDirectory else { return }
        guard encodersReloadable(from: directory) else { return }
        for module in encoderModules {
            let empty = module.parameters().flattened().map { ($0.0, MLXArray.zeros([0])) }
            module.update(parameters: ModuleParameters.unflattened(empty))
        }
        encodersReleased = true
        Memory.clearCache()
    }

    /// Les tours se rechargent a l'identique si chaque module quantifie en memoire l'est
    /// aussi sur disque (pack pre-quantifie : `embed_vision.embedding_projection` du pack
    /// E2B 4 bits) ; pas une tour quantifiee a la volee depuis un checkpoint bf16.
    /// Lecture des seules cles (chargement paresseux).
    private var reloadableCache: Bool?
    private func encodersReloadable(from directory: URL) -> Bool {
        if let reloadableCache { return reloadableCache }
        let quantizedPaths = [("vision_tower", visionTower as Module?), ("embed_vision", embedVision as Module?),
                              ("audio_tower", audioTower as Module?), ("embed_audio", embedAudio as Module?)]
            .flatMap { name, module -> [String] in
                guard let module else { return [] }
                return module.leafModules().flattened().filter { $0.1 is Quantized }.map { "\(name).\($0.0)" }
            }
        var reloadable = true
        if !quantizedPaths.isEmpty {
            let files = (try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
                .filter { $0.pathExtension == "safetensors" }) ?? []
            var keys = Set<String>()
            for file in files { keys.formUnion((try? loadArrays(url: file).keys).map(Array.init) ?? []) }
            reloadable = quantizedPaths.allSatisfy { keys.contains("\($0).scales") }
        }
        reloadableCache = reloadable
        return reloadable
    }

    /// Recharge les tours liberees depuis `weightsDirectory` (lecture paresseuse : seuls
    /// leurs tenseurs sont lus).
    public func restoreEncodersIfNeeded() throws {
        guard encodersReleased else { return }
        guard let directory = weightsDirectory else {
            throw Gemma4PipelineError.invalidInput("tours liberees et dossier du modele inconnu : impossible de les recharger")
        }
        var raw: [String: MLXArray] = [:]
        let files = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }
        for file in files {
            for (key, value) in try loadArrays(url: file)
            where key.contains("vision_tower") || key.contains("embed_vision")
                || key.contains("audio_tower") || key.contains("embed_audio") {
                raw[key] = value
            }
        }
        let weights = sanitize(weights: raw).filter { key, _ in
            Self.encoderPrefixes.contains { key.hasPrefix($0) }
        }
        // Pas de .shapeMismatch : les poids liberes sont des tableaux vides, toute forme differe.
        try update(parameters: ModuleParameters.unflattened(weights), verify: [.noUnusedKeys])
        for module in encoderModules { eval(module) }
        encodersReleased = false
    }
}

