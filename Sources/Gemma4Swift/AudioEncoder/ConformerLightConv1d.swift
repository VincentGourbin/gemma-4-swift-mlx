// Port de audio.py ConformerLightConv1d — GLU + depthwise conv1d causale

import MLX
import MLXNN

/// Conteneur pour le poids depthwise conv1d (shape [channels, kernelSize, 1])
class DepthwiseConv1dWeight: Module {
    @ModuleInfo var weight: MLXArray

    init(channels: Int, kernelSize: Int) {
        self._weight.wrappedValue = MLXArray.zeros([channels, kernelSize, 1])
        super.init()
    }
}

/// Light convolution: norm → linear(2x) → GLU → depthwise_conv1d(causal) → norm → SiLU → linear + residual
public class ConformerLightConv1d: Module {
    let gradientClipping: Float
    let causalPadding: Int

    @ModuleInfo(key: "pre_layer_norm") var preLayerNorm: AudioRMSNorm
    @ModuleInfo(key: "linear_start") var linearStart: ClippableLinear
    @ModuleInfo(key: "depthwise_conv1d") var depthwiseConv1d: DepthwiseConv1dWeight
    @ModuleInfo(key: "conv_norm") var convNorm: AudioRMSNorm
    @ModuleInfo(key: "linear_end") var linearEnd: ClippableLinear

    public init(_ config: Gemma4AudioConfig) {
        self.gradientClipping = config.gradientClipping
        self.causalPadding = config.convKernelSize - 1

        self._preLayerNorm.wrappedValue = AudioRMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._linearStart.wrappedValue = ClippableLinear(inFeatures: config.hiddenSize, outFeatures: config.hiddenSize * 2)
        self._depthwiseConv1d.wrappedValue = DepthwiseConv1dWeight(channels: config.hiddenSize, kernelSize: config.convKernelSize)
        self._convNorm.wrappedValue = AudioRMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._linearEnd.wrappedValue = ClippableLinear(inFeatures: config.hiddenSize, outFeatures: config.hiddenSize)

        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        var h = preLayerNorm(x)
        h = linearStart(h)

        // GLU: split et gate
        let half = h.dim(-1) / 2
        let x1 = h[.ellipsis, 0 ..< half]
        let x2 = h[.ellipsis, half...]
        h = x1 * sigmoid(x2)

        // Padding causal pour Conv1d
        h = padded(h, widths: [.init(0), .init((causalPadding, 0)), .init(0)])

        // Depthwise conv1d via conv_general (groups = channels)
        h = convGeneral(h, depthwiseConv1d.weight, strides: 1, padding: 0, groups: h.dim(-1))

        h = clip(h, min: MLXArray(-gradientClipping), max: MLXArray(gradientClipping))
        h = convNorm(h)
        h = silu(h)
        h = linearEnd(h)

        return h + residual
    }
}
