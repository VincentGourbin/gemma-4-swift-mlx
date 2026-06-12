// Port de mlx-lm/models/rope_utils.py ProportionalRoPE (version optimisee).
//
// Astuce du code mlx-lm Python : au lieu de slicer/concatener x pour ne rotater
// qu'une partie des dimensions, on padde le tableau de frequences avec des `inf`
// pour les positions non rotees. `mx.fast.rope` interprete cos(t*inf)/sin(t*inf)
// comme identite (cos -> 1, sin -> 0), ce qui laisse ces dimensions inchangees.
//
// Resultat : 1 seul appel MLXFast.RoPE au lieu de 6 slice/concat. Critique pour
// le throughput sur 12B (8 full-attn layers x 2 (Q+K) par token).

import MLX
import MLXFast
import MLXNN

/// ProportionalRoPE pour les couches full_attention de Gemma 4.
/// Tient compte de `partial_rotary_factor` via le padding `inf` du tableau de
/// frequences plutot que par slicing.
public final class ProportionalRoPE {
    let dims: Int
    let traditional: Bool
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

        // rotated_dims aligne sur 2 (paires reelles/imaginaires).
        let rotatedDims = 2 * Int(partialRotaryFactor * Float(dims) / 2.0)

        if rotatedDims > 0 {
            let realCount = rotatedDims / 2
            let exponents = MLXArray(stride(from: Float(0), to: Float(rotatedDims), by: 2)) / Float(dims)
            let realFreqs = factor * pow(MLXArray(base), exponents)

            let padCount = (dims - rotatedDims) / 2
            if padCount > 0 {
                // Pad avec +inf -> rotation identite sur les dimensions non rotees.
                let pad = MLXArray(Array(repeating: Float.infinity, count: padCount))
                self.freqs = concatenated([realFreqs, pad], axis: 0)
            } else {
                self.freqs = realFreqs
            }
            _ = realCount // (silenced unused; kept for readability above)
        } else {
            self.freqs = nil
        }
    }

    public func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        guard let freqs = freqs else { return x }

        return MLXFast.RoPE(
            x,
            dimensions: dims,
            traditional: traditional,
            base: nil as Float?,
            scale: 1.0,
            offset: offset,
            freqs: freqs
        )
    }
}
