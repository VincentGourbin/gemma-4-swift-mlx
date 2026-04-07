// Port de language.py DecoderLayer — Couche decoder complete

import Foundation
import MLX
import MLXFast
import MLXNN
import MLXLMCommon

/// Couche decoder Gemma 4
/// Combine: attention + MLP (+ MoE parallele pour 26B-A4B) + per-layer input gating + layer_scalar
public class Gemma4DecoderLayer: Module {
    let config: Gemma4TextConfig
    let layerIdx: Int
    let layerType: String
    let hiddenSizePerLayerInput: Int
    let enableMoe: Bool

    @ModuleInfo(key: "self_attn") var selfAttn: Gemma4Attention
    @ModuleInfo var mlp: Gemma4MLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayernorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayernorm: RMSNorm

    // MoE (26B-A4B) : MLP dense + experts en parallele
    @ModuleInfo(key: "router") var router: Gemma4Router?
    @ModuleInfo(key: "experts") var experts: Gemma4Experts?
    @ModuleInfo(key: "post_feedforward_layernorm_1") var postFeedforwardLayernorm1: RMSNorm?
    @ModuleInfo(key: "pre_feedforward_layernorm_2") var preFeedforwardLayernorm2: RMSNorm?
    @ModuleInfo(key: "post_feedforward_layernorm_2") var postFeedforwardLayernorm2: RMSNorm?

    // Per-layer input gating (modeles 2B/4B)
    @ModuleInfo(key: "per_layer_input_gate") var perLayerInputGate: Linear?
    @ModuleInfo(key: "per_layer_projection") var perLayerProjection: Linear?
    @ModuleInfo(key: "post_per_layer_input_norm") var postPerLayerInputNorm: RMSNorm?

    // Layer scalar
    @ModuleInfo(key: "layer_scalar") var layerScalar: MLXArray

    public init(_ config: Gemma4TextConfig, layerIdx: Int) {
        self.config = config
        self.layerIdx = layerIdx
        self.layerType = config.resolvedLayerTypes[layerIdx]
        self.hiddenSizePerLayerInput = config.hiddenSizePerLayerInput
        self.enableMoe = config.enableMoeBlock

        self._selfAttn.wrappedValue = Gemma4Attention(config, layerIdx: layerIdx)
        self._mlp.wrappedValue = Gemma4MLP(config, layerIdx: layerIdx)

        self._inputLayernorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayernorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._preFeedforwardLayernorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postFeedforwardLayernorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        // MoE : router + experts + 3 layernorms supplementaires
        if enableMoe {
            self._router.wrappedValue = Gemma4Router(config)
            self._experts.wrappedValue = Gemma4Experts(config)
            self._postFeedforwardLayernorm1.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._preFeedforwardLayernorm2.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._postFeedforwardLayernorm2.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        } else {
            self._router.wrappedValue = nil
            self._experts.wrappedValue = nil
            self._postFeedforwardLayernorm1.wrappedValue = nil
            self._preFeedforwardLayernorm2.wrappedValue = nil
            self._postFeedforwardLayernorm2.wrappedValue = nil
        }

        // Per-layer input gating (si le modele a des per-layer inputs)
        if hiddenSizePerLayerInput > 0 {
            self._perLayerInputGate.wrappedValue = Linear(config.hiddenSize, hiddenSizePerLayerInput, bias: false)
            self._perLayerProjection.wrappedValue = Linear(hiddenSizePerLayerInput, config.hiddenSize, bias: false)
            self._postPerLayerInputNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        } else {
            self._perLayerInputGate.wrappedValue = nil
            self._perLayerProjection.wrappedValue = nil
            self._postPerLayerInputNorm.wrappedValue = nil
        }

        self._layerScalar.wrappedValue = MLXArray.ones([1])

        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
        cache: KVCache? = nil,
        perLayerInput: MLXArray? = nil
    ) -> MLXArray {
        var residual = x

        // Self-attention
        var h = inputLayernorm(x)
        h = selfAttn(h, mask: mask, cache: cache)
        h = postAttentionLayernorm(h)
        h = residual + h

        // Feedforward : MLP dense (+ MoE experts en parallele si 26B-A4B)
        residual = h

        if enableMoe,
           let router = router,
           let experts = experts,
           let norm1 = postFeedforwardLayernorm1,
           let preNorm2 = preFeedforwardLayernorm2,
           let postNorm2 = postFeedforwardLayernorm2 {
            // Branche 1 : MLP dense
            var h1 = preFeedforwardLayernorm(h)
            h1 = mlp(h1)
            h1 = norm1(h1)

            // Branche 2 : MoE experts
            let (topKIndices, topKWeights) = router(h)
            var h2 = preNorm2(h)
            h2 = experts(h2, topKIndices: topKIndices, topKWeights: topKWeights)
            h2 = postNorm2(h2)

            // Combiner les deux branches
            h = h1 + h2
        } else {
            // MLP simple (E2B, E4B, 31B)
            h = preFeedforwardLayernorm(h)
            h = mlp(h)
        }

        h = postFeedforwardLayernorm(h)
        h = residual + h

        // Per-layer input gating
        if let gate = perLayerInputGate,
           let proj = perLayerProjection,
           let norm = postPerLayerInputNorm,
           let pli = perLayerInput {
            residual = h
            var gateOutput = gate(h)
            gateOutput = geluApproximate(gateOutput)
            gateOutput = gateOutput * pli
            gateOutput = proj(gateOutput)
            gateOutput = norm(gateOutput)
            h = residual + gateOutput
        }

        // Layer scalar
        h = h * layerScalar

        return h
    }
}
