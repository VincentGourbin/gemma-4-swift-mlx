// Norms specifiques a l'encodeur audio

import MLX
import MLXFast
import MLXNN

/// RMSNorm audio (poids directs, pas de +1 offset)
public class AudioRMSNorm: Module {
    @ModuleInfo var weight: MLXArray
    let eps: Float

    public init(dimensions: Int, eps: Float = 1e-6) {
        self._weight.wrappedValue = MLXArray.ones([dimensions])
        self.eps = eps
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight, eps: eps)
    }
}
