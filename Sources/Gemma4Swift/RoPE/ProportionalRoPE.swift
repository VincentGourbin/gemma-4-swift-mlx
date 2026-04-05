// Port de rope_utils.py ProportionalRoPE — RoPE avec rotation partielle pour full attention

import MLX
import MLXFast
import MLXNN

/// ProportionalRoPE pour les couches full_attention de Gemma 4.
/// Applique la rotation seulement sur une fraction des dimensions du head (partial_rotary_factor).
/// Les frequences sont calculees relativement a la dimension complete du head.
/// N'herite PAS de Module car freqs n'est pas un parametre apprenable.
public final class ProportionalRoPE {
    let dims: Int
    let traditional: Bool
    let rotatedDims: Int
    let freqs: MLXArray?

    public init(
        dims: Int,
        traditional: Bool = false,
        base: Float = 10000.0,
        factor: Float = 1.0,
        partialRotaryFactor: Float = 1.0
    ) {
        self.dims = dims
        self.traditional = traditional

        let ropeAngles = Int(partialRotaryFactor * Float(dims) / 2.0)
        self.rotatedDims = 2 * ropeAngles

        if rotatedDims > 0 {
            let exponents = MLXArray(stride(from: Float(0), to: Float(rotatedDims), by: 2)) / Float(dims)
            self.freqs = factor * pow(MLXArray(base), exponents)
        } else {
            self.freqs = nil
        }
    }

    public func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        guard rotatedDims > 0, let freqs = freqs else { return x }

        let head = x[.ellipsis, 0 ..< dims]
        let half = dims / 2

        // Split en deux moities gauche/droite
        let left = head[.ellipsis, 0 ..< half]
        let right = head[.ellipsis, half ..< dims]

        let rotHalf = rotatedDims / 2

        // Extraire les portions a rotater de chaque moitie
        let leftRot = left[.ellipsis, 0 ..< rotHalf]
        let rightRot = right[.ellipsis, 0 ..< rotHalf]

        // Concatener pour former le vecteur a rotater
        let toRotate = concatenated([leftRot, rightRot], axis: -1)

        // Appliquer RoPE via MLXFast
        let rotated = MLXFast.RoPE(
            toRotate,
            dimensions: rotatedDims,
            traditional: traditional,
            base: nil as Float?,
            scale: 1.0,
            offset: offset,
            freqs: freqs
        )

        // Recombiner : gauche = [rotated_left, left_passthrough], droite = [rotated_right, right_passthrough]
        let newLeft: MLXArray
        if rotHalf < half {
            let rotLeft = rotated[.ellipsis, 0 ..< rotHalf]
            let passLeft = left[.ellipsis, rotHalf...]
            newLeft = concatenated([rotLeft, passLeft], axis: -1)
        } else {
            newLeft = rotated[.ellipsis, 0 ..< rotHalf]
        }

        let newRight: MLXArray
        if rotHalf < half {
            let rotRight = rotated[.ellipsis, rotHalf...]
            let passRight = right[.ellipsis, rotHalf...]
            newRight = concatenated([rotRight, passRight], axis: -1)
        } else {
            newRight = rotated[.ellipsis, rotHalf...]
        }

        let newHead = concatenated([newLeft, newRight], axis: -1)

        // Si le tensor original avait des dims au-dela de dims, les reattacher
        if x.shape.last! > dims {
            let tail = x[.ellipsis, dims...]
            return concatenated([newHead, tail], axis: -1)
        }

        return newHead
    }
}
