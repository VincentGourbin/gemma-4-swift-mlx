// Port de modular_diffusion_gemma.DiffusionGemmaEncoderTextAttention (lignes 201-295)
//
// Variante de Gemma4Attention pour l'encoder DiffusionGemma :
//   1. Retire la logique KV-sharing
//   2. is_causal = config.use_bidirectional_attention != "all"
//      (par defaut "vision" -> causal sur tokens texte, bidir sur vision)
//   3. Calcule ses propres K/V puis met a jour le cache standard (DynamicCache)
//
// Implementation Swift : on retourne (output, keys, values) explicitement
// pour permettre a l'encoder model d'assembler un EncoderKVCache a la fin.
// Pas de KVCache live cote attention : tout est externalise.

import Foundation
import MLX
import MLXFast
import MLXNN

/// Attention de l'encoder DiffusionGemma. Calcule K/V et les expose pour collecte.
public class DiffusionGemmaEncoderTextAttention: Module {
    let config: Gemma4TextConfig
    let layerIdx: Int
    let layerType: String
    let isSliding: Bool
    let headDim: Int
    let numHeads: Int
    let numKVHeads: Int
    let useKEqV: Bool
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

        if !isSliding && config.globalHeadDim > 0 {
            self.headDim = config.globalHeadDim
        } else {
            self.headDim = config.headDim
        }

        let dim = config.hiddenSize
        self.numHeads = config.numAttentionHeads
        self.useKEqV = !isSliding  // v_proj=None sur full -> K=V
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

    /// - Parameters:
    ///   - x : `[B, T, H]`, hidden states post-input_layernorm.
    ///   - mask : masque encoder (causal ou bidir selon config).
    ///   - positionOffset : offset RoPE (== past_seen_tokens en cas de cache continu).
    /// - Returns: tuple (output, keys, values) — K/V deja transposes en
    ///   shape `[B, numKVHeads, T, headDim]` pour collecte par l'encoder model.
    public func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
        positionOffset: Int = 0
    ) -> (output: MLXArray, keys: MLXArray, values: MLXArray) {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        var queries = qProj(x).reshaped(B, L, numHeads, headDim)
        queries = qNorm(queries)
        queries = queries.transposed(0, 2, 1, 3)
        queries = rope(queries, offset: positionOffset)

        var keys = kProj(x).reshaped(B, L, numKVHeads, headDim)
        var values: MLXArray
        if useKEqV {
            values = keys
        } else {
            values = vProj!(x).reshaped(B, L, numKVHeads, headDim)
        }

        keys = kNorm(keys)
        keys = keys.transposed(0, 2, 1, 3)
        keys = rope(keys, offset: positionOffset)

        values = vNorm(values)
        values = values.transposed(0, 2, 1, 3)

        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return (oProj(output), keys, values)
    }
}
