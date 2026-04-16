// Port de language.py Attention — Attention multi-tete avec global_head_dim, K=V, partial RoPE

import Foundation
import MLX
import MLXFast
import MLXNN
import MLXLMCommon

/// Attention multi-tete Gemma 4
/// - global_head_dim pour full attention, head_dim pour sliding
/// - K=V optionnel (values = raw k_proj avant k_norm)
/// - KV sharing pour les couches tardives
/// - RoPE par type d'attention (standard ou proportional)
/// - Utilise attentionWithCacheUpdate() pour le support quantized KV cache
public class Gemma4Attention: Module {
    let config: Gemma4TextConfig
    let layerIdx: Int
    let layerType: String
    let isSliding: Bool
    let headDim: Int
    let numHeads: Int
    let numKVHeads: Int
    let useKEqV: Bool
    let isKvSharedLayer: Bool
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear?
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    @ModuleInfo(key: "v_norm") var vNorm: RMSNormNoScale

    let rope: RoPEWrapper

    public init(_ config: Gemma4TextConfig, layerIdx: Int) {
        self.config = config
        self.layerIdx = layerIdx

        let layerTypes = config.resolvedLayerTypes
        self.layerType = layerTypes[layerIdx]
        self.isSliding = layerType == "sliding_attention"

        // head_dim dynamique: global_head_dim pour full attention
        if !isSliding && config.globalHeadDim > 0 {
            self.headDim = config.globalHeadDim
        } else {
            self.headDim = config.headDim
        }

        let dim = config.hiddenSize
        self.numHeads = config.numAttentionHeads

        // K=V pour full attention (modeles 26B/31B)
        self.useKEqV = config.attentionKEqV && !isSliding
        if useKEqV, let globalKvHeads = config.numGlobalKeyValueHeads {
            self.numKVHeads = globalKvHeads
        } else {
            self.numKVHeads = config.numKeyValueHeads
        }

        self.scale = 1.0

        self._qProj.wrappedValue = Linear(dim, numHeads * headDim, bias: false)
        self._kProj.wrappedValue = Linear(dim, numKVHeads * headDim, bias: false)
        if !useKEqV {
            self._vProj.wrappedValue = Linear(dim, numKVHeads * headDim, bias: false)
        } else {
            self._vProj.wrappedValue = nil
        }
        self._oProj.wrappedValue = Linear(numHeads * headDim, dim, bias: false)

        self._qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self._vNorm.wrappedValue = RMSNormNoScale(eps: config.rmsNormEps)

        // KV sharing
        let firstKvSharedLayerIdx = config.firstKvSharedLayerIdx
        self.isKvSharedLayer = layerIdx >= firstKvSharedLayerIdx && firstKvSharedLayerIdx > 0

        // RoPE adapte au type d'attention
        let ropeTheta = config.ropeTheta(forLayerType: layerType)
        let ropeType = config.ropeType(forLayerType: layerType)
        let partialRotaryFactor = ropeType == "proportional" ? config.fullAttentionPartialRotaryFactor : 1.0

        self.rope = RoPEFactory.create(
            dims: headDim,
            base: ropeTheta,
            traditional: false,
            ropeType: ropeType,
            partialRotaryFactor: partialRotaryFactor
        )

        super.init()
    }

    /// Forward pass avec support du KV sharing entre couches.
    ///
    /// Quand `sharedKV` est fourni (couches KV-shared sans cache, i.e. training),
    /// les K/V partages sont reutilises au lieu d'etre recalcules via k_proj/v_proj.
    /// Cela reproduit le mecanisme `shared_kv` de Python mlx-lm.
    ///
    /// Retourne `(output, kv, offset)` pour permettre le suivi des intermediaires
    /// dans le forward pass du TextModel.
    public func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
        cache: KVCache? = nil,
        sharedKV: (keys: MLXArray, values: MLXArray)? = nil,
        sharedOffset: Int? = nil
    ) -> (output: MLXArray, kv: (keys: MLXArray, values: MLXArray), offset: Int) {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        var queries = qProj(x).reshaped(B, L, numHeads, headDim)
        queries = qNorm(queries)
        queries = queries.transposed(0, 2, 1, 3)

        var keys: MLXArray
        var values: MLXArray
        var effectiveOffset: Int

        if let (sharedKeys, sharedValues) = sharedKV {
            // KV sharing sans cache (training): reutiliser les K/V d'une couche precedente
            // Les K/V sont deja normalises, transposes et RoPE'd
            keys = sharedKeys
            values = sharedValues
            effectiveOffset = sharedOffset ?? 0
            queries = rope(queries, offset: effectiveOffset)

            let output = MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: keys,
                values: values,
                scale: scale,
                mask: mask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)
            return (oProj(output), (keys, values), effectiveOffset)

        } else if isKvSharedLayer, let cache = cache {
            // KV sharing avec cache (inference): reutiliser le cache existant
            effectiveOffset = cache.offset
            queries = rope(queries, offset: effectiveOffset)

            // TurboQuant shared
            if let turboCache = cache as? TurboQuantKVCache {
                let output = turboCache.quantizedAttention(
                    queries: queries, scale: scale, mask: mask
                )
                .transposed(0, 2, 1, 3)
                .reshaped(B, L, -1)
                let state = cache.state
                return (oProj(output), (state[0], state[1]), effectiveOffset)
            }

            // Standard shared: lire les K/V decompresses du cache
            let state = cache.state
            if state.count >= 2 {
                let output = MLXFast.scaledDotProductAttention(
                    queries: queries,
                    keys: state[0],
                    values: state[1],
                    scale: scale,
                    mask: mask
                )
                .transposed(0, 2, 1, 3)
                .reshaped(B, L, -1)
                return (oProj(output), (state[0], state[1]), effectiveOffset)
            }
            // Fallback: compute own KV (ne devrait pas arriver)
            let kv = computeKV(x: x, B: B, L: L)
            keys = kv.keys; values = kv.values
            keys = rope(keys, offset: effectiveOffset)

            let output = attentionWithCacheUpdate(
                queries: queries, keys: keys, values: values,
                cache: cache, scale: scale, mask: mask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)
            return (oProj(output), (keys, values), effectiveOffset)
        }

        // Non-shared: calculer ses propres K/V
        let kv = computeKV(x: x, B: B, L: L)
        keys = kv.keys; values = kv.values

        // Lire l'offset AVANT que attentionWithCacheUpdate() l'incremente
        effectiveOffset = cache?.offset ?? 0

        // Appliquer RoPE aux queries ET aux keys
        queries = rope(queries, offset: effectiveOffset)
        keys = rope(keys, offset: effectiveOffset)

        // TurboQuant path
        if let turboCache = cache as? TurboQuantKVCache {
            turboCache.update(keys: keys, values: values)
            let output = turboCache.quantizedAttention(
                queries: queries, scale: scale, mask: mask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)
            return (oProj(output), (keys, values), effectiveOffset)
        }

        // Standard path: attentionWithCacheUpdate() gere l'update du cache
        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return (oProj(output), (keys, values), effectiveOffset)
    }

    private func computeKV(
        x: MLXArray, B: Int, L: Int
    ) -> (keys: MLXArray, values: MLXArray) {
        var keys = kProj(x).reshaped(B, L, numKVHeads, headDim)

        // K=V: values sont le raw k_proj output (avant k_norm)
        var values: MLXArray
        if useKEqV {
            values = keys
        } else {
            values = vProj!(x).reshaped(B, L, numKVHeads, headDim)
        }

        keys = kNorm(keys)
        values = vNorm(values)
        values = values.transposed(0, 2, 1, 3)

        // RoPE est applique par l'appelant avec l'offset correct du cache
        keys = keys.transposed(0, 2, 1, 3)

        return (keys, values)
    }
}
