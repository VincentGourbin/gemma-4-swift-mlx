// Port de audio.py ConformerBlock — Block Conformer Macaron complet

import MLX
import MLXNN

/// Block Conformer Macaron: FFW → Attn → LightConv → FFW → Norm
public class ConformerBlock: Module {
    let gradientClipping: Float

    @ModuleInfo(key: "feed_forward1") var feedForward1: ConformerFeedForward
    @ModuleInfo(key: "self_attn") var selfAttn: AudioAttention
    @ModuleInfo var lconv1d: ConformerLightConv1d
    @ModuleInfo(key: "feed_forward2") var feedForward2: ConformerFeedForward
    @ModuleInfo(key: "norm_pre_attn") var normPreAttn: AudioRMSNorm
    @ModuleInfo(key: "norm_post_attn") var normPostAttn: AudioRMSNorm
    @ModuleInfo(key: "norm_out") var normOut: AudioRMSNorm

    public init(_ config: Gemma4AudioConfig) {
        self.gradientClipping = config.gradientClipping

        self._feedForward1.wrappedValue = ConformerFeedForward(config)
        self._selfAttn.wrappedValue = AudioAttention(config)
        self._lconv1d.wrappedValue = ConformerLightConv1d(config)
        self._feedForward2.wrappedValue = ConformerFeedForward(config)
        self._normPreAttn.wrappedValue = AudioRMSNorm(dimensions: config.hiddenSize)
        self._normPostAttn.wrappedValue = AudioRMSNorm(dimensions: config.hiddenSize)
        self._normOut.wrappedValue = AudioRMSNorm(dimensions: config.hiddenSize)

        super.init()
    }

    public func callAsFunction(_ x: MLXArray, mask: MLXArray, causalValidMask: MLXArray, positionEmbeddings: MLXArray) -> MLXArray {
        var h = feedForward1(x)

        // Attention avec pre/post norm et residual
        let residual = h
        h = clip(h, min: MLXArray(-gradientClipping), max: MLXArray(gradientClipping))
        h = normPreAttn(h)
        h = selfAttn(h, mask: mask, causalValidMask: causalValidMask, positionEmbeddings: positionEmbeddings)
        h = clip(h, min: MLXArray(-gradientClipping), max: MLXArray(gradientClipping))
        h = residual + normPostAttn(h)

        // Zero les positions invalides avant lconv1d
        let validityMask = expandedDimensions(logicalNot(mask), axis: -1).asType(h.dtype)
        h = h * validityMask

        h = lconv1d(h)
        h = feedForward2(h)
        h = clip(h, min: MLXArray(-gradientClipping), max: MLXArray(gradientClipping))
        return normOut(h)
    }
}
