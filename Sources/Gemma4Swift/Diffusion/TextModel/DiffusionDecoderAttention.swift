// Port de modular_diffusion_gemma.DiffusionGemmaDecoderTextAttention (lignes 298-402)
//
// Variante du Gemma4Attention pour le decoder DiffusionGemma :
//   1. is_causal = False (bidirectionnel total sur le canvas)
//   2. Pas de KV sharing (drasticment retire dans DiffusionGemma)
//   3. Calcule ses propres K/V pour le canvas, puis fait
//        cat([encoder_KV, canvas_KV], dim=2)
//      a partir d'un EncoderKVCache passe par couche.
//   4. Ne met JAMAIS a jour l'encoder cache.
//
// K=V suit la meme regle que Gemma4Attention :
//   - couches full_attention : v_proj=None, values = raw k_proj output (avant k_norm)
//   - couches sliding_attention : v_proj present, values = vNorm(vProj(x))
//
// Python ref (ligne 365) :
//   value_states = self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else key_states
//
// Note RoPE : ici on n'a pas de cache K/V mutable cote decoder, donc l'offset
// RoPE pour les queries du canvas est `encoderCacheLength`. Cela correspond
// au `decoder_position_ids = arange(cache_len, cache_len + canvas_len)` cote
// Python (lignes 1069-1078).

import Foundation
import MLX
import MLXFast
import MLXNN

/// Attention bidirectionnelle du decoder DiffusionGemma.
public class DiffusionDecoderAttention: Module {
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

        // K=V pour full attention (Python : v_proj=None sur full)
        // Equivalent a Gemma4Attention quand attentionKEqV=true.
        self.useKEqV = !isSliding
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

    /// Forward bidirectionnel : pas de update de cache, juste cat([encoder, canvas]).
    ///
    /// - Parameters:
    ///   - x : `[B, T_canvas, H]`, embeddings normalises du canvas (input_layernorm deja applique).
    ///   - mask : masque pre-calcule par DiffusionAttentionMask. .none si pas de padding.
    ///   - encoderEntry : K/V de l'encoder pour cette couche (extraits du EncoderKVCache).
    ///     Si nil, attention est purement self-attention canvas-only.
    ///   - encoderCacheLength : offset RoPE pour les queries du canvas.
    public func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
        encoderEntry: EncoderKVCache.Entry? = nil,
        encoderCacheLength: Int = 0
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        // Queries
        var queries = qProj(x).reshaped(B, L, numHeads, headDim)
        queries = qNorm(queries)
        queries = queries.transposed(0, 2, 1, 3)
        // RoPE sur les positions decoder_position_ids = arange(cacheLen, cacheLen + canvasLen)
        queries = rope(queries, offset: encoderCacheLength)

        // K/V du canvas (toujours calcule, pas de partage)
        var keys = kProj(x).reshaped(B, L, numKVHeads, headDim)
        var values: MLXArray
        if useKEqV {
            // values = raw k_proj output (avant k_norm)
            values = keys
        } else {
            values = vProj!(x).reshaped(B, L, numKVHeads, headDim)
        }

        keys = kNorm(keys)
        keys = keys.transposed(0, 2, 1, 3)
        // RoPE sur les K du canvas : memes positions que les queries
        keys = rope(keys, offset: encoderCacheLength)

        values = vNorm(values)
        values = values.transposed(0, 2, 1, 3)

        // Cat avec encoder cache (read-only)
        let finalKeys: MLXArray
        let finalValues: MLXArray
        if let encoderEntry = encoderEntry {
            finalKeys = concatenated([encoderEntry.keys, keys], axis: 2)
            finalValues = concatenated([encoderEntry.values, values], axis: 2)
        } else {
            finalKeys = keys
            finalValues = values
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: finalKeys,
            values: finalValues,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return oProj(output)
    }
}
