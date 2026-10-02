import Testing
import MLX
import MLXFast
import MLXNN
@testable import Gemma4Swift

/// K-41 : RoPE (proportionnel et standard) sur un lot de 2 lignes identiques, une position
/// (decodage), entree transposee comme dans l'attention : les deux lignes doivent rester egales.
@Suite("RoPE en lot")
struct RoPEBatchTests {
    private func gap(_ out: MLXArray) -> Float {
        abs(out[0] - out[1]).max().item(Float.self)
    }

    @Test("une position, entree transposee, B = 2 : lignes identiques (GPU)")
    func testDecodeRowSymmetry() {
        MLXRandom.seed(3)
        let one = MLXRandom.normal([1, 1, 8, 256])                       // [B, L, H, D]
        let x = tiled(one, repetitions: [2, 1, 1, 1]).transposed(0, 2, 1, 3)   // [B, H, L, D] vue
        let proportional = ProportionalRoPE(dims: 256, base: 1_000_000, partialRotaryFactor: 0.25)
        let standard = MLXNN.RoPE(dimensions: 256, traditional: false, base: 10_000)
        let p = proportional(x, offset: 37), s = standard.callAsFunction(x, offset: 37)
        let pc = proportional(contiguous(x), offset: 37)
        let sc = standard.callAsFunction(contiguous(x), offset: 37)
        // Contournement : lot replie dans l'axe des tetes.
        let c = contiguous(x)
        let folded = proportional(c.reshaped(1, 2 * 8, 1, 256), offset: 37).reshaped(2, 8, 1, 256)
        let single = proportional(contiguous(one.transposed(0, 2, 1, 3)), offset: 37)
        let foldErr = abs(folded[0] - single[0]).max().item(Float.self)
        print("ROPE-GAP proportionnel \(gap(p)) standard \(gap(s)) proportionnel-contigu \(gap(pc)) standard-contigu \(gap(sc)) replie \(gap(folded)) replie-vs-seul \(foldErr)")
        #expect(gap(p) == 0)
        #expect(gap(s) == 0)
        #expect(gap(folded) == 0 && foldErr == 0)
        // Le RoPEWrapper de la bibliotheque replie le lot : contigu ou non, lignes egales.
        let wrapper = RoPEWrapper(proportional)
        #expect(gap(wrapper(contiguous(x), offset: 37)) == 0)
        #expect(abs(wrapper(contiguous(x), offset: 37)[0] - single[0]).max().item(Float.self) == 0)
    }
}
