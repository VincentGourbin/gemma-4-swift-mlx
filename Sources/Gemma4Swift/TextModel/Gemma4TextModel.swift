// Port de language.py Gemma4TextModel — Modele texte complet

import Foundation
import MLX
import MLXFast
import MLXNN
import MLXLMCommon

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
        perLayerInputs: MLXArray? = nil
    ) -> MLXArray {
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
        let globalMask = MLXLMCommon.createAttentionMask(
            h: h,
            cache: firstFullCacheIdx < cacheArray.count ? cacheArray[firstFullCacheIdx] : nil
        )
        let slidingWindowMask = MLXLMCommon.createAttentionMask(
            h: h,
            cache: firstSlidingCacheIdx < cacheArray.count ? cacheArray[firstSlidingCacheIdx] : nil,
            windowSize: windowSize
        )

        // Forward a travers les layers — avec suivi des intermediaires pour le KV sharing
        // Ref: Python mlx-lm gemma4_text.py utilise intermediates[] + previous_kvs
        // pour passer les K/V des couches non-partagees aux couches partagees.
        let layerTypes = config.resolvedLayerTypes
        var intermediates: [(kv: (keys: MLXArray, values: MLXArray), offset: Int)?] =
            Array(repeating: nil, count: numHiddenLayers)

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
            // (seulement quand pas de cache — le cache gere deja le sharing a l'inference)
            let sharedKV: (keys: MLXArray, values: MLXArray)?
            let sharedOffset: Int?
            if i >= firstKvSharedLayerIdx && firstKvSharedLayerIdx > 0 && cache == nil,
               let prev = intermediates[cacheIdx] {
                sharedKV = prev.kv
                sharedOffset = prev.offset
            } else {
                sharedKV = nil
                sharedOffset = nil
            }

            let (output, kv, offset) = layer(
                h, mask: localMask, cache: c, perLayerInput: perLayerInput,
                sharedKV: sharedKV, sharedOffset: sharedOffset
            )
            h = output
            intermediates[i] = (kv: kv, offset: offset)
        }

        return norm(h)
    }
}
