// Norms specifiques a l'encodeur vision (float32 computation)

import MLX
import MLXNN

/// RMSNorm avec poids apprenables, calcul en float32
public class VisionRMSNorm: Module {
    @ModuleInfo var weight: MLXArray
    let eps: Float

    public init(dimensions: Int, eps: Float = 1e-6) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xFloat = x.asType(.float32)
        let variance = mean(xFloat * xFloat, axis: -1, keepDims: true)
        let normed = xFloat * rsqrt(variance + MLXArray(eps))
        let result = normed * weight.asType(.float32)
        return result.asType(x.dtype)
    }
}

/// RMSNorm sans poids (parameter-free), calcul en float32
public class VisionRMSNormNoScale: Module {
    let eps: Float

    public init(eps: Float = 1e-6) {
        self.eps = eps
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xFloat = x.asType(.float32)
        let variance = mean(xFloat * xFloat, axis: -1, keepDims: true)
        return (xFloat * rsqrt(variance + MLXArray(eps))).asType(x.dtype)
    }
}
