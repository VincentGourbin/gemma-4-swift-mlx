import Testing
import MLX
import MLXFast
import MLXNN
@testable import Gemma4Swift

/// K-41 : RoPE (proportionnel et standard) sur un lot de 2 lignes identiques, une position
/// (decodage), entree transposee comme dans l'attention : les deux lignes doivent rester egales.
/// Avant MLX 0.32, l'entree contigue rendait des lignes fausses sur GPU (ecart 5,5-6,
/// ml-explore/mlx#3494) ; garde-fou contre une regression amont.
@Suite("RoPE en lot")
struct RoPEBatchTests {
    private func gap(_ out: MLXArray) -> Float {
        abs(out[0] - out[1]).max().item(Float.self)
    }

    @Test("une position, B = 2 : lignes identiques, egales au calcul seul (GPU)")
    func testDecodeRowSymmetry() {
        MLXRandom.seed(3)
        let one = MLXRandom.normal([1, 1, 8, 256])                       // [B, L, H, D]
        let x = tiled(one, repetitions: [2, 1, 1, 1]).transposed(0, 2, 1, 3)   // [B, H, L, D] vue
        let proportional = ProportionalRoPE(dims: 256, base: 1_000_000, partialRotaryFactor: 0.25)
        let standard = MLXNN.RoPE(dimensions: 256, traditional: false, base: 10_000)
        let single = proportional(contiguous(one.transposed(0, 2, 1, 3)), offset: 37)
        for input in [x, contiguous(x)] {
            #expect(gap(proportional(input, offset: 37)) == 0)
            #expect(gap(standard.callAsFunction(input, offset: 37)) == 0)
            let wrapped = RoPEWrapper(proportional)(input, offset: 37)
            #expect(gap(wrapped) == 0)
            #expect(abs(wrapped[0] - single[0]).max().item(Float.self) == 0)
        }
    }
}
