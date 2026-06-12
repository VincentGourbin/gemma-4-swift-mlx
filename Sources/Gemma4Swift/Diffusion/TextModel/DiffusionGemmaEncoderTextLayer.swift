// Port de modular_diffusion_gemma.DiffusionGemmaEncoderTextLayer (lignes 446-517)
//
// Identique structurellement a DiffusionGemmaDecoderTextLayer, sauf que :
//   1. Utilise DiffusionGemmaEncoderTextAttention (cache mise a jour, masque controlable)
//   2. Expose K/V vers l'encoder model via le tuple de retour
//
// Le forward FFN+MoE est strictement identique.

import Foundation
import MLX
import MLXFast
import MLXNN

/// Couche encoder DiffusionGemma : encoder attention + MLP + MoE + layer_scalar.
public class DiffusionGemmaEncoderTextLayer: Module {
    let config: Gemma4TextConfig
    let layerIdx: Int
    let layerType: String

    @ModuleInfo(key: "self_attn") var selfAttn: DiffusionGemmaEncoderTextAttention
    @ModuleInfo var mlp: Gemma4MLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayernorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayernorm: RMSNorm

    @ModuleInfo(key: "router") var router: Gemma4Router
    @ModuleInfo(key: "experts") var experts: Gemma4Experts
    @ModuleInfo(key: "post_feedforward_layernorm_1") var postFeedforwardLayernorm1: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm_2") var preFeedforwardLayernorm2: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm_2") var postFeedforwardLayernorm2: RMSNorm

    @ModuleInfo(key: "layer_scalar") var layerScalar: MLXArray

    public init(_ config: Gemma4TextConfig, layerIdx: Int) {
        self.config = config
        self.layerIdx = layerIdx
        self.layerType = config.resolvedLayerTypes[layerIdx]

        self._selfAttn.wrappedValue = DiffusionGemmaEncoderTextAttention(config, layerIdx: layerIdx)
        self._mlp.wrappedValue = Gemma4MLP(config, layerIdx: layerIdx)

        self._inputLayernorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayernorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._preFeedforwardLayernorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postFeedforwardLayernorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        self._router.wrappedValue = Gemma4Router(config)
        self._experts.wrappedValue = Gemma4Experts(config)
        self._postFeedforwardLayernorm1.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._preFeedforwardLayernorm2.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postFeedforwardLayernorm2.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        self._layerScalar.wrappedValue = MLXArray.ones([1])

        super.init()
    }

    /// - Returns: tuple (output, keys, values) — K/V transposes prets pour collecte.
    public func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
        positionOffset: Int = 0
    ) -> (output: MLXArray, keys: MLXArray, values: MLXArray) {
        var residual = x

        var h = inputLayernorm(x)
        let (attnOut, keys, values) = selfAttn(h, mask: mask, positionOffset: positionOffset)
        h = postAttentionLayernorm(attnOut)
        h = residual + h

        residual = h

        var h1 = preFeedforwardLayernorm(h)
        h1 = mlp(h1)
        h1 = postFeedforwardLayernorm1(h1)

        let (topKIndices, topKWeights) = router(h)
        var h2 = preFeedforwardLayernorm2(h)
        h2 = experts(h2, topKIndices: topKIndices, topKWeights: topKWeights)
        h2 = postFeedforwardLayernorm2(h2)

        h = h1 + h2
        h = postFeedforwardLayernorm(h)
        h = residual + h

        h = h * layerScalar

        return (h, keys, values)
    }
}
