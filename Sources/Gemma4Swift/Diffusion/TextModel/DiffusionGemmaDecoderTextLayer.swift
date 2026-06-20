// Port de modular_diffusion_gemma.DiffusionGemmaDecoderTextLayer (lignes 520-592)
//
// Couche decoder de DiffusionGemma. Identique a [[Gemma4DecoderLayer]] sauf :
//   1. Utilise DiffusionDecoderAttention (bidirectionnel + cat encoder_KV)
//   2. PAS de per-layer input gate (PLE) -> jamais
//   3. Pas de pipe sharedKV
//
// L'architecture FFN+MoE est identique :
//   residual -> input_layernorm -> self_attn -> post_attention_layernorm -> +residual
//   residual -> pre_feedforward_layernorm -> MLP -> post_feedforward_layernorm_1 -> h1
//   residual (flat, NOT normed) -> router -> (top_k_weights, top_k_index)
//   residual -> pre_feedforward_layernorm_2 -> experts(top_k_*) -> post_feedforward_layernorm_2 -> h2
//   h = h1 + h2 -> post_feedforward_layernorm -> +residual -> *= layer_scalar
//
// Python ref ligne 577-578 : `hidden_states_2_for_routing = hidden_states_flat`
// (= residual non-normalise, pour le routing seul). Same comme dans Gemma4DecoderLayer.

import Foundation
import MLX
import MLXFast
import MLXNN

/// Couche decoder DiffusionGemma : DiffusionDecoderAttention + MLP + MoE + layer_scalar.
public class DiffusionGemmaDecoderTextLayer: Module {
    let config: Gemma4TextConfig
    let layerIdx: Int
    let layerType: String

    @ModuleInfo(key: "self_attn") var selfAttn: DiffusionDecoderAttention
    @ModuleInfo var mlp: Gemma4MLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayernorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayernorm: RMSNorm

    // MoE : tous les layers DiffusionGemma sont MoE
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

        self._selfAttn.wrappedValue = DiffusionDecoderAttention(config, layerIdx: layerIdx)
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

    public func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
        encoderEntry: EncoderKVCache.Entry? = nil,
        encoderCacheLength: Int = 0
    ) -> MLXArray {
        var residual = x

        // Self-attention bidirectionnelle
        var h = inputLayernorm(x)
        let attnOut = selfAttn(
            h, mask: mask,
            encoderEntry: encoderEntry,
            encoderCacheLength: encoderCacheLength
        )
        h = postAttentionLayernorm(attnOut)
        h = residual + h

        // Feedforward : MLP dense + MoE en parallele
        residual = h

        // Branche 1 : MLP dense
        var h1 = preFeedforwardLayernorm(h)
        h1 = mlp(h1)
        h1 = postFeedforwardLayernorm1(h1)

        // Branche 2 : MoE experts
        // Routing prend h NON-normalisee (cf Python ligne 578 : hidden_states_2_for_routing = hidden_states_flat)
        let (topKIndices, topKWeights) = router(h)
        var h2 = preFeedforwardLayernorm2(h)
        h2 = experts(h2, topKIndices: topKIndices, topKWeights: topKWeights)
        h2 = postFeedforwardLayernorm2(h2)

        h = h1 + h2
        h = postFeedforwardLayernorm(h)
        h = residual + h

        // Layer scalar (1,)
        h = h * layerScalar

        return h
    }
}
