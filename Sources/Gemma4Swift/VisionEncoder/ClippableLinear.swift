// Port de vision.py ClippableLinear — Linear avec clamping optionnel input/output

import MLX
import MLXNN

/// Linear layer avec clamping optionnel sur les entrees et sorties.
/// Les bornes sont stockees comme buffers dans le checkpoint (scalaires).
/// Initialises a ±inf (no-op) jusqu'au chargement des vraies valeurs.
public class ClippableLinear: Module {
    @ModuleInfo var linear: Linear
    let useClipping: Bool

    @ModuleInfo(key: "input_min") var inputMin: MLXArray
    @ModuleInfo(key: "input_max") var inputMax: MLXArray
    @ModuleInfo(key: "output_min") var outputMin: MLXArray
    @ModuleInfo(key: "output_max") var outputMax: MLXArray

    public init(inFeatures: Int, outFeatures: Int, bias: Bool = false, useClipping: Bool = true) {
        self.useClipping = useClipping
        self._linear.wrappedValue = Linear(inFeatures, outFeatures, bias: bias)

        self._inputMin.wrappedValue = MLXArray(Float(-Float.infinity))
        self._inputMax.wrappedValue = MLXArray(Float.infinity)
        self._outputMin.wrappedValue = MLXArray(Float(-Float.infinity))
        self._outputMax.wrappedValue = MLXArray(Float.infinity)

        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var result = x
        if useClipping {
            result = clip(result, min: inputMin, max: inputMax)
        }
        result = linear(result)
        if useClipping {
            result = clip(result, min: outputMin, max: outputMax)
        }
        return result
    }
}
