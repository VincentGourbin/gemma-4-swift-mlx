// Port de language.py Gemma4TextModel — Modele texte complet

import Foundation
import MLX
import MLXFast
import MLXNN
import MLXLMCommon

/// Sortie d'un forward du TextModel avec les intermediaires (K/V par couche concrete).
/// Utilise par le path MTP speculative decoding pour exposer les K/V partages
/// du target au drafter.
public struct LayerIntermediate {
    public let keys: MLXArray
    public let values: MLXArray
    public let offset: Int
}

public struct TextForwardOutput {
    /// Sortie du dernier decoder layer APRES le final RMSNorm.
    /// Utilise pour calculer les logits (lm_head sur cette valeur).
    public let hidden: MLXArray

    /// Sortie du dernier decoder layer AVANT le final RMSNorm.
    /// IMPORTANT: c'est cette valeur que le drafter Assistant attend en entree
    /// (le `pre_projection` du drafter a ete entraine contre cette hidden pre-norm).
    /// Voir mlx_vlm/models/gemma4/language.py: "captured BEFORE the final RMSNorm".
    public let preNormHidden: MLXArray

    public let intermediates: [LayerIntermediate?]
}

/// Linear avec scaling integre (pour per_layer_model_projection)
class ScaledLinear: Module {
    @ModuleInfo var weight: MLXArray
    let scalar: Float

    init(inFeatures: Int, outFeatures: Int, scalar: Float) {
        self._weight.wrappedValue = MLXArray.zeros([outFeatures, inFeatures])
        self.scalar = scalar
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        (matmul(x, weight.T)) * MLXArray(scalar, dtype: x.dtype)
    }
}

/// Modele texte Gemma 4 (sans le head de logits)
public class Gemma4TextModel: Module {
    let config: Gemma4TextConfig
    let windowSize: Int
    let numHiddenLayers: Int
    let hiddenSizePerLayerInput: Int
    let firstKvSharedLayerIdx: Int
    let layerIdxToCacheIdx: [Int]
    let firstFullCacheIdx: Int
    let firstSlidingCacheIdx: Int

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [Gemma4DecoderLayer]
    @ModuleInfo var norm: RMSNorm

    // Per-layer input embeddings
    @ModuleInfo(key: "embed_tokens_per_layer") var embedTokensPerLayer: Embedding?
    @ModuleInfo(key: "per_layer_model_projection") var perLayerModelProjection: ScaledLinear?
    @ModuleInfo(key: "per_layer_projection_norm") var perLayerProjectionNorm: RMSNormZeroShift?

    let embedScale: Float
    let embedTokensPerLayerScale: Float
    let perLayerInputScale: Float

    public init(_ config: Gemma4TextConfig) {
        self.config = config
        self.windowSize = config.slidingWindow
        self.numHiddenLayers = config.numHiddenLayers
        self.hiddenSizePerLayerInput = config.hiddenSizePerLayerInput
        self.firstKvSharedLayerIdx = config.firstKvSharedLayerIdx
        self.embedScale = pow(Float(config.hiddenSize), 0.5)
        self.embedTokensPerLayerScale = pow(Float(config.hiddenSizePerLayerInput), 0.5)
        self.perLayerInputScale = pow(2.0, -0.5)

        // Compute layer_idx -> cache_idx mapping
        let layerTypes = config.resolvedLayerTypes
        let concreteLayers = Array(layerTypes[..<firstKvSharedLayerIdx])

        var mapping = Array(0 ..< firstKvSharedLayerIdx)
        if firstKvSharedLayerIdx < config.numHiddenLayers {
            let sharedFullIdx = concreteLayers.lastIndex(of: "full_attention") ?? 0
            let sharedSlidingIdx = concreteLayers.lastIndex(of: "sliding_attention") ?? 0

            for i in firstKvSharedLayerIdx ..< config.numHiddenLayers {
                if layerTypes[i] == "full_attention" {
                    mapping.append(sharedFullIdx)
                } else {
                    mapping.append(sharedSlidingIdx)
                }
            }
        }
        self.layerIdxToCacheIdx = mapping

        // Trouver les premiers index par type de cache
        self.firstFullCacheIdx = concreteLayers.firstIndex(of: "full_attention") ?? 0
        self.firstSlidingCacheIdx = concreteLayers.firstIndex(of: "sliding_attention") ?? 0

        // Embeddings
        self._embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)

        // Layers
        self._layers.wrappedValue = (0 ..< config.numHiddenLayers).map { i in
            Gemma4DecoderLayer(config, layerIdx: i)
        }

        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        // Per-layer input embeddings (pour modeles 2B/4B)
        if hiddenSizePerLayerInput > 0 {
            self._embedTokensPerLayer.wrappedValue = Embedding(
                embeddingCount: config.vocabSizePerLayerInput,
                dimensions: config.numHiddenLayers * config.hiddenSizePerLayerInput
            )
            self._perLayerModelProjection.wrappedValue = ScaledLinear(
                inFeatures: config.hiddenSize,
                outFeatures: config.numHiddenLayers * config.hiddenSizePerLayerInput,
                scalar: pow(Float(config.hiddenSize), -0.5)
            )
            self._perLayerProjectionNorm.wrappedValue = RMSNormZeroShift(
                dimensions: config.hiddenSizePerLayerInput,
                eps: config.rmsNormEps
            )
        } else {
            self._embedTokensPerLayer.wrappedValue = nil
            self._perLayerModelProjection.wrappedValue = nil
            self._perLayerProjectionNorm.wrappedValue = nil
        }

        super.init()
    }

    // MARK: - Per-layer inputs

    func getPerLayerInputs(_ inputIds: MLXArray) -> MLXArray {
        guard let embed = embedTokensPerLayer else {
            fatalError("embed_tokens_per_layer non disponible")
        }
        var result = embed(inputIds)
        result = result * MLXArray(embedTokensPerLayerScale, dtype: result.dtype)
        let shape = inputIds.shape + [config.numHiddenLayers, hiddenSizePerLayerInput]
        return result.reshaped(shape)
    }

    func projectPerLayerInputs(_ inputsEmbeds: MLXArray, perLayerInputs: MLXArray?) -> MLXArray {
        guard let proj = perLayerModelProjection, let projNorm = perLayerProjectionNorm else {
            fatalError("per_layer_model_projection non disponible")
        }
        var perLayerProjection = proj(inputsEmbeds)
        let shape = Array(inputsEmbeds.shape.dropLast()) + [config.numHiddenLayers, hiddenSizePerLayerInput]
        perLayerProjection = perLayerProjection.reshaped(shape)
        perLayerProjection = projNorm(perLayerProjection)

        guard let perLayerInputs = perLayerInputs else {
            return perLayerProjection
        }

        return (perLayerProjection + perLayerInputs) * MLXArray(perLayerInputScale, dtype: inputsEmbeds.dtype)
    }

    // MARK: - Forward

    public func callAsFunction(
        inputs: MLXArray? = nil,
        inputsEmbeds: MLXArray? = nil,
        cache: [KVCache?]? = nil,
        perLayerInputs: MLXArray? = nil,
        visionTokenMask: MLXArray? = nil
    ) -> MLXArray {
        // Fast path : modeles SANS KV-sharing (12B Unified, 31B). Bypass la
        // collecte d'intermediates qui retient ~150 MB de K/V refs par forward
        // et augmente la pression memoire pendant le prefill.
        // Pour E2B/E4B (avec KV-sharing), on garde le path complet.
        let collect = firstKvSharedLayerIdx < numHiddenLayers
        if collect {
            return runForward(
                inputs: inputs, inputsEmbeds: inputsEmbeds, cache: cache,
                perLayerInputs: perLayerInputs, visionTokenMask: visionTokenMask,
                collectIntermediates: true
            ).hidden
        }
        return runForward(
            inputs: inputs, inputsEmbeds: inputsEmbeds, cache: cache,
            perLayerInputs: perLayerInputs, visionTokenMask: visionTokenMask,
            collectIntermediates: false
        ).hidden
    }

    /// Variante de `callAsFunction` qui retourne aussi les K/V intermediaires
    /// par couche. Utilise par le path MTP pour exposer les K/V partages au drafter.
    /// Le `callAsFunction` standard reste inchange (delegate vers cette methode).
    ///
    /// - Parameter visionTokenMask : `[B, T]` bool, true ou le token est vision (image/video).
    ///   Si fourni ET prefill (T > 1), active l'overlay bidirectionnel sur les blocs vision
    ///   (port du Python `_apply_blockwise_bidirectional_overlay`). Utilise par
    ///   [[Gemma4UnifiedMultimodalLLMModel]] quand `use_bidirectional_attention=vision`.
    public func forwardCollectingIntermediates(
        inputs: MLXArray? = nil,
        inputsEmbeds: MLXArray? = nil,
        cache: [KVCache?]? = nil,
        perLayerInputs: MLXArray? = nil,
        visionTokenMask: MLXArray? = nil
    ) -> TextForwardOutput {
        runForward(
            inputs: inputs, inputsEmbeds: inputsEmbeds, cache: cache,
            perLayerInputs: perLayerInputs, visionTokenMask: visionTokenMask,
            collectIntermediates: true
        )
    }

    /// Path partage entre `callAsFunction` (fast-path sans intermediates) et
    /// `forwardCollectingIntermediates` (avec intermediates pour KV-sharing /
    /// MTP). La logique embedding, masks, layer loop et norm est identique ;
    /// seul l'enregistrement des K/V dans un array `intermediates[]` differe.
    private func runForward(
        inputs: MLXArray?,
        inputsEmbeds: MLXArray?,
        cache: [KVCache?]?,
        perLayerInputs: MLXArray?,
        visionTokenMask: MLXArray?,
        collectIntermediates: Bool
    ) -> TextForwardOutput {
        var h: MLXArray
        if let inputsEmbeds = inputsEmbeds {
            h = inputsEmbeds
        } else if let inputs = inputs {
            h = embedTokens(inputs)
            h = h * MLXArray(embedScale, dtype: h.dtype)
        } else {
            fatalError("inputs ou inputsEmbeds requis")
        }

        // Per-layer inputs
        var finalPerLayerInputs: MLXArray? = nil
        if hiddenSizePerLayerInput > 0 {
            var pli = perLayerInputs
            if inputs != nil && pli == nil {
                pli = getPerLayerInputs(inputs!)
            }
            if pli != nil || inputs != nil {
                finalPerLayerInputs = projectPerLayerInputs(h, perLayerInputs: pli)
            }
        }

        // Caches
        let cacheArray = cache ?? Array(repeating: nil as KVCache?, count: firstKvSharedLayerIdx)

        // Masques d'attention — utilise createAttentionMask() de MLXLMCommon
        // Pour les single tokens (T=1) : retourne .none (pas de masque materialise)
        // Pour les multi-tokens (prefill) : retourne .causal ou .array selon le cas
        var globalMask = MLXLMCommon.createAttentionMask(
            h: h,
            cache: firstFullCacheIdx < cacheArray.count ? cacheArray[firstFullCacheIdx] : nil
        )
        var slidingWindowMask = MLXLMCommon.createAttentionMask(
            h: h,
            cache: firstSlidingCacheIdx < cacheArray.count ? cacheArray[firstSlidingCacheIdx] : nil,
            windowSize: windowSize
        )

        // Overlay bidirectionnel pour les blocs vision (gemma4_unified) :
        // materialise les masques causaux + OR avec same_block.
        let T = h.dim(1)
        if let visionMask = visionTokenMask, T > 1 {
            let blockIds = Gemma4BidirectionalMask.blockSequenceIds(visionMask: visionMask)
            let overlay = Gemma4BidirectionalMask.overlay(blockSequenceIds: blockIds)

            let causalGlobal = MLXLMCommon.createCausalMask(n: T, offset: 0)
            let causalSliding = MLXLMCommon.createCausalMask(n: T, offset: 0, windowSize: windowSize)

            let mergedGlobal = Gemma4BidirectionalMask.compose(causal: causalGlobal, overlay: overlay)
            let mergedSliding = Gemma4BidirectionalMask.compose(causal: causalSliding, overlay: overlay)

            globalMask = .array(mergedGlobal)
            slidingWindowMask = .array(mergedSliding)
        }

        // Forward a travers les layers — avec ou sans suivi des intermediaires.
        // Le suivi est necessaire pour le KV sharing (E2B/E4B) et pour MTP.
        // Sans suivi : on jette les K/V retournes (~150 MB economises par
        // forward sur prefill long, gain x2.3 mesure sur MMLU Pro CoT 12B).
        let layerTypes = config.resolvedLayerTypes
        var intermediates: [(kv: (keys: MLXArray, values: MLXArray), offset: Int)?] =
            collectIntermediates
                ? Array(repeating: nil, count: numHiddenLayers)
                : []

        for (i, layer) in layers.enumerated() {
            let cacheIdx = layerIdxToCacheIdx[i]
            let c = cacheIdx < cacheArray.count ? cacheArray[cacheIdx] : nil
            let isGlobal = layerTypes[i] == "full_attention"

            let localMask = isGlobal ? globalMask : slidingWindowMask

            let perLayerInput: MLXArray?
            if let fpli = finalPerLayerInputs {
                perLayerInput = fpli[0..., 0..., i, 0...]
            } else {
                perLayerInput = nil
            }

            // KV sharing: passer les K/V de la couche source aux couches partagees
            // (seulement quand pas de cache — le cache gere deja le sharing a l'inference).
            // Quand collectIntermediates est false, on n'a pas de KV-sharing par definition.
            let sharedKV: (keys: MLXArray, values: MLXArray)?
            let sharedOffset: Int?
            if collectIntermediates,
               i >= firstKvSharedLayerIdx && firstKvSharedLayerIdx > 0 && cache == nil,
               let prev = intermediates[cacheIdx] {
                sharedKV = prev.kv
                sharedOffset = prev.offset
            } else {
                sharedKV = nil
                sharedOffset = nil
            }

            if collectIntermediates {
                let (output, kv, offset) = layer(
                    h, mask: localMask, cache: c, perLayerInput: perLayerInput,
                    sharedKV: sharedKV, sharedOffset: sharedOffset
                )
                h = output
                intermediates[i] = (kv: kv, offset: offset)
            } else {
                // Fast path : on jette les K/V (le cache les a deja captures
                // pour ses propres besoins).
                let (output, _, _) = layer(
                    h, mask: localMask, cache: c, perLayerInput: perLayerInput,
                    sharedKV: nil, sharedOffset: nil
                )
                h = output
            }
        }

        // Capture h pre-norm pour le drafter MTP, puis applique norm pour les logits standard.
        let preNormHidden = h
        let normedHidden = norm(h)

        let publicIntermediates: [LayerIntermediate?]
        if collectIntermediates {
            publicIntermediates = intermediates.map { entry in
                guard let entry = entry else { return nil }
                return LayerIntermediate(
                    keys: entry.kv.keys, values: entry.kv.values, offset: entry.offset
                )
            }
        } else {
            publicIntermediates = []
        }

        return TextForwardOutput(
            hidden: normedHidden,
            preNormHidden: preNormHidden,
            intermediates: publicIntermediates
        )
    }
}
