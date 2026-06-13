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
    ///   - x : `[B, T_new, H]`, hidden states post-input_layernorm pour les nouveaux tokens uniquement.
    ///   - mask : masque encoder. Si priorKV est fourni, doit etre shape
    ///     `[T_new, T_prior + T_new]` (causal avec offset).
    ///   - positionOffset : offset RoPE pour les nouveaux tokens (== T_prior si cache).
    ///   - priorKV : K/V des tokens precedents (de cache). Si fourni, concatene
    ///     avec les nouveaux K/V avant attention. Les queries (nouvelles uniquement)
    ///     attendent ainsi sur tout l'historique cumule.
    /// - Returns: (output, newKeys, newValues) — newKeys/newValues sont uniquement
    ///   ceux des NOUVEAUX tokens, prets a etre appended au EncoderKVCache.
    public func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
        positionOffset: Int = 0,
        priorKV: (keys: MLXArray, values: MLXArray)? = nil
    ) -> (output: MLXArray, keys: MLXArray, values: MLXArray) {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        var queries = qProj(x).reshaped(B, L, numHeads, headDim)
        queries = qNorm(queries)
        queries = queries.transposed(0, 2, 1, 3)
        queries = rope(queries, offset: positionOffset)

        var newKeys = kProj(x).reshaped(B, L, numKVHeads, headDim)
        var newValues: MLXArray
        if useKEqV {
            newValues = newKeys
        } else {
            newValues = vProj!(x).reshaped(B, L, numKVHeads, headDim)
        }

        newKeys = kNorm(newKeys)
        newKeys = newKeys.transposed(0, 2, 1, 3)
        newKeys = rope(newKeys, offset: positionOffset)

        newValues = vNorm(newValues)
        newValues = newValues.transposed(0, 2, 1, 3)

        // Attention : si priorKV fourni, on concat avec les nouveaux K/V pour que
        // les nouvelles queries voient tout le contexte cumule.
        let attentionKeys: MLXArray
        let attentionValues: MLXArray
        if let prior = priorKV {
            attentionKeys = concatenated([prior.keys, newKeys], axis: 2)
            attentionValues = concatenated([prior.values, newValues], axis: 2)
        } else {
            attentionKeys = newKeys
            attentionValues = newValues
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: attentionKeys,
            values: attentionValues,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        // On retourne UNIQUEMENT les nouveaux K/V (a appender au cache)
        return (oProj(output), newKeys, newValues)
    }
}
