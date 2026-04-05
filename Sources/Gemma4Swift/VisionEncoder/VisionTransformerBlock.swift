// Port de vision.py VisionTransformerBlock + VisionMLP

import MLX
import MLXNN

/// MLP de l'encodeur vision
public class VisionMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: ClippableLinear
    @ModuleInfo(key: "up_proj") var upProj: ClippableLinear
    @ModuleInfo(key: "down_proj") var downProj: ClippableLinear

    public init(_ config: Gemma4VisionConfig) {
        let clip = config.useClippedLinears
        self._gateProj.wrappedValue = ClippableLinear(inFeatures: config.hiddenSize, outFeatures: config.intermediateSize, useClipping: clip)
        self._upProj.wrappedValue = ClippableLinear(inFeatures: config.hiddenSize, outFeatures: config.intermediateSize, useClipping: clip)
        self._downProj.wrappedValue = ClippableLinear(inFeatures: config.intermediateSize, outFeatures: config.hiddenSize, useClipping: clip)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

/// Block transformer de l'encodeur vision (attention bidirectionnelle + MLP)
public class VisionTransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: VisionAttention
    @ModuleInfo var mlp: VisionMLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayernorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayernorm: RMSNorm

    public init(_ config: Gemma4VisionConfig) {
        self._selfAttn.wrappedValue = VisionAttention(config)
        self._mlp.wrappedValue = VisionMLP(config)
        self._inputLayernorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayernorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._preFeedforwardLayernorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postFeedforwardLayernorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, positions: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let normed = inputLayernorm(x)
        let attnOut = selfAttn(normed, positions: positions, mask: mask)
        let attnNormed = postAttentionLayernorm(attnOut)
        let h = x + attnNormed

        let ffNormed = preFeedforwardLayernorm(h)
        let ffOut = mlp(ffNormed)
        let ffNormed2 = postFeedforwardLayernorm(ffOut)
        return h + ffNormed2
    }
}
