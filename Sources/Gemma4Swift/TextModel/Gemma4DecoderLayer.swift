// Port de language.py DecoderLayer — Couche decoder complete

import Foundation
import MLX
import MLXNN
import MLXLMCommon

/// Couche decoder Gemma 4
/// Combine: attention + MLP + per-layer input gating + layer_scalar
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
        mask: MLXArray? = nil,
        cache: KVCache? = nil,
        perLayerInput: MLXArray? = nil
    ) -> MLXArray {
        var residual = x

        // Self-attention
        var h = inputLayernorm(x)
        h = selfAttn(h, mask: mask, cache: cache)
        h = postAttentionLayernorm(h)
        h = residual + h

        // MLP
        residual = h
        h = preFeedforwardLayernorm(h)
        h = mlp(h)
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
