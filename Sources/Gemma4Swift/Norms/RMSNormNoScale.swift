// Port de language.py RMSNormNoScale — RMSNorm sans poids apprenables

import MLX
import MLXFast
import MLXNN

/// RMSNorm sans parametre de scale (parameter-free)
public class RMSNormNoScale: Module {
    let eps: Float

    public init(eps: Float = 1e-6) {
        self.eps = eps
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: MLXArray.mlxNone, eps: eps)
    }
}
