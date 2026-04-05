// Port de audio.py SubSampleConvProjection — Sous-echantillonnage convolutif

import MLX
import MLXNN

/// Block Conv2d + LayerNorm + ReLU pour le sous-echantillonnage audio
class SSCPConvBlock: Module {
    let timeStride: Int = 2

    @ModuleInfo var conv: Conv2d
    @ModuleInfo var norm: LayerNorm

    init(_ config: Gemma4AudioConfig, idx: Int) {
        let inChannels = idx == 0 ? 1 : config.subsamplingConvChannels[idx - 1]
        let outChannels = config.subsamplingConvChannels[idx]

        self._conv.wrappedValue = Conv2d(
            inputChannels: inChannels, outputChannels: outChannels,
            kernelSize: .init((3, 3)), stride: .init((2, 2)), padding: .init(0),
            bias: false
        )
        self._norm.wrappedValue = LayerNorm(dimensions: outChannels, eps: config.rmsNormEps, bias: false)

        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> (MLXArray, MLXArray) {
        // x: [B, T, F, C], mask: [B, T]
        var result = MLX.where(expandedDimensions(expandedDimensions(mask, axis: -1), axis: -1), MLXArray(Float(0.0)), x)

        // Padding manuel [B, T+2, F+2, C]
        result = padded(result, widths: [.init(0), .init(1), .init(1), .init(0)])

        result = conv(result)
        let tOut = result.dim(1)
        let outputMask = mask[0..., .stride(by: timeStride)][0..., 0 ..< tOut]

        result = norm(result)
        result = relu(result)

        return (result, outputMask)
    }
}

/// SSCP: 2 Conv2d blocks → flatten(F, C) → Linear vers hidden_size
public class SubSampleConvProjection: Module {
    static let inputFeatSize = 128

    @ModuleInfo var layer0: SSCPConvBlock
    @ModuleInfo var layer1: SSCPConvBlock
    @ModuleInfo(key: "input_proj_linear") var inputProjLinear: Linear

    public init(_ config: Gemma4AudioConfig) {
        self._layer0.wrappedValue = SSCPConvBlock(config, idx: 0)
        self._layer1.wrappedValue = SSCPConvBlock(config, idx: 1)

        // Calculer la dimension apres 2 convolutions
        var freq = Self.inputFeatSize
        for _ in 0 ..< 2 {
            freq = (freq + 2 - 3) / 2 + 1
        }
        let projInputDim = freq * config.subsamplingConvChannels.last!

        self._inputProjLinear.wrappedValue = Linear(projInputDim, config.hiddenSize, bias: false)
        super.init()
    }

    public func callAsFunction(_ audioMel: MLXArray, mask: MLXArray) -> (MLXArray, MLXArray) {
        // audioMel: [B, T, F_in] → ajouter channel dim: [B, T, F, 1]
        var x = expandedDimensions(audioMel, axis: -1)
        var currentMask = mask

        (x, currentMask) = layer0(x, mask: currentMask)
        (x, currentMask) = layer1(x, mask: currentMask)

        // Flatten F*C → [B, T, F*C]
        let B = x.dim(0)
        let T = x.dim(1)
        x = x.reshaped(B, T, -1)

        // Projection
        x = inputProjLinear(x)
        return (x, currentMask)
    }
}
