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

    public func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray? = nil,
        cache: KVCache? = nil
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        var queries = qProj(x).reshaped(B, L, numHeads, headDim)
        queries = qNorm(queries)

        var offset = 0
        var keys: MLXArray
        var values: MLXArray

        if isKvSharedLayer, let cache = cache {
            // Couche partagee: reutiliser le cache existant
            let state = cache.state
            if state.count >= 2 {
                keys = state[0]
                values = state[1]
                offset = cache.offset
            } else {
                (keys, values, offset) = computeKV(x: x, B: B, L: L, cache: cache)
            }
        } else {
            (keys, values, offset) = computeKV(x: x, B: B, L: L, cache: cache)
        }

        queries = queries.transposed(0, 2, 1, 3)
        queries = rope(queries, offset: offset)

        // Ajuster le masque si necessaire
        var adjustedMask = mask
        if let mask = mask {
            let keysSeqLen = keys.shape[keys.ndim - 2]
            if mask.shape.last! != keysSeqLen {
                adjustedMask = mask[.ellipsis, (mask.shape.last! - keysSeqLen)...]
            }
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: adjustedMask.map { .array($0) } ?? .none
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return oProj(output)
    }

    private func computeKV(
        x: MLXArray, B: Int, L: Int, cache: KVCache?
    ) -> (keys: MLXArray, values: MLXArray, offset: Int) {
        let offset = cache?.offset ?? 0

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

        keys = keys.transposed(0, 2, 1, 3)
        keys = rope(keys, offset: offset)

        if let cache = cache {
            (keys, values) = cache.update(keys: keys, values: values)
        }

        return (keys, values, offset)
    }
}
