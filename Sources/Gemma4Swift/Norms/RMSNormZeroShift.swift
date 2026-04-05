// Port de language.py RMSNormZeroShift — RMSNorm avec poids directs (pas de +1 offset)

import MLX
import MLXFast
import MLXNN

/// RMSNorm Gemma4 : le poids est applique directement (scale_shift=0.0)
/// Contrairement au RMSNorm standard de MLXNN qui fait weight*(1+x), ici c'est weight*x
public class RMSNormZeroShift: Module {
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
