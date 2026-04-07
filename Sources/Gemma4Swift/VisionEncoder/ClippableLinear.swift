// Port de vision.py ClippableLinear — Linear avec clamping optionnel input/output

import MLX
import MLXNN

/// Linear layer avec clamping optionnel sur les entrees et sorties.
/// Les bornes sont stockees comme buffers dans le checkpoint (scalaires).
/// Initialises a ±inf (no-op) jusqu'au chargement des vraies valeurs.
public class ClippableLinear: Module {
    @ModuleInfo var linear: Linear
    let useClipping: Bool

    // Clipping params optionnels : declares seulement si useClipping == true
    // Sinon verify: .all crash car les poids ne contiennent pas ces cles
    @ModuleInfo(key: "input_min") var inputMin: MLXArray?
    @ModuleInfo(key: "input_max") var inputMax: MLXArray?
    @ModuleInfo(key: "output_min") var outputMin: MLXArray?
    @ModuleInfo(key: "output_max") var outputMax: MLXArray?

    public init(inFeatures: Int, outFeatures: Int, bias: Bool = false, useClipping: Bool = true) {
        self.useClipping = useClipping
        self._linear.wrappedValue = Linear(inFeatures, outFeatures, bias: bias)

        if useClipping {
            self._inputMin.wrappedValue = MLXArray(Float(-Float.infinity))
            self._inputMax.wrappedValue = MLXArray(Float.infinity)
            self._outputMin.wrappedValue = MLXArray(Float(-Float.infinity))
            self._outputMax.wrappedValue = MLXArray(Float.infinity)
        } else {
            self._inputMin.wrappedValue = nil
            self._inputMax.wrappedValue = nil
            self._outputMin.wrappedValue = nil
            self._outputMax.wrappedValue = nil
        }

        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var result = x
        if useClipping, let inMin = inputMin, let inMax = inputMax {
            result = clip(result, min: inMin, max: inMax)
        }
        result = linear(result)
        if useClipping, let outMin = outputMin, let outMax = outputMax {
            result = clip(result, min: outMin, max: outMax)
        }
        return result
    }
}
