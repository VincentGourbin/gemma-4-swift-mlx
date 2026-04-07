// Port de language.py Experts — Sparse MoE via SwitchGLU (26B-A4B)

import Foundation
import MLX
import MLXNN
import MLXLMCommon

/// Experts MoE Gemma 4 : wrapper autour de SwitchGLU avec activation GeGLU (gelu_approx)
public class Gemma4Experts: Module {
    @ModuleInfo(key: "switch_glu") var switchGLU: SwitchGLU

    public init(_ config: Gemma4TextConfig) {
        let numExperts = config.numExperts ?? 128

        self._switchGLU.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: numExperts,
            activation: geluApproximate,
            bias: false
        )

        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray,
        topKIndices: MLXArray,
        topKWeights: MLXArray
    ) -> MLXArray {
        let (B, S, H) = (x.dim(0), x.dim(1), x.dim(2))
        let K = topKIndices.dim(-1)

        let xFlat = x.reshaped(B * S, H)
        let indicesFlat = topKIndices.reshaped(B * S, K)

        let expertOut = switchGLU(xFlat, indicesFlat)

        let weights = topKWeights.reshaped(B * S, K)[.ellipsis, .newAxis]
        return (expertOut * weights).sum(axis: -2).reshaped(B, S, H)
    }
}
