// Port de audio.py ConformerFeedForward — FFW Macaron avec residual scaling

import MLX
import MLXNN

/// Feed-forward Macaron-style avec clipping de gradient et residual scaling
public class ConformerFeedForward: Module {
    let gradientClipping: Float
    let residualWeight: Float

    @ModuleInfo(key: "pre_layer_norm") var preLayerNorm: AudioRMSNorm
    @ModuleInfo(key: "ffw_layer_1") var ffwLayer1: ClippableLinear
    @ModuleInfo(key: "ffw_layer_2") var ffwLayer2: ClippableLinear
    @ModuleInfo(key: "post_layer_norm") var postLayerNorm: AudioRMSNorm

    public init(_ config: Gemma4AudioConfig) {
        self.gradientClipping = config.gradientClipping
        self.residualWeight = config.residualWeight

        self._preLayerNorm.wrappedValue = AudioRMSNorm(dimensions: config.hiddenSize)
        self._ffwLayer1.wrappedValue = ClippableLinear(inFeatures: config.hiddenSize, outFeatures: config.hiddenSize * 4)
        self._ffwLayer2.wrappedValue = ClippableLinear(inFeatures: config.hiddenSize * 4, outFeatures: config.hiddenSize)
        self._postLayerNorm.wrappedValue = AudioRMSNorm(dimensions: config.hiddenSize)

        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        var h = clip(x, min: MLXArray(-gradientClipping), max: MLXArray(gradientClipping))
        h = preLayerNorm(h)
        h = ffwLayer1(h)
        h = silu(h)
        h = ffwLayer2(h)
        h = clip(h, min: MLXArray(-gradientClipping), max: MLXArray(gradientClipping))
        h = postLayerNorm(h)
        return residual + h * MLXArray(residualWeight)
    }
}
