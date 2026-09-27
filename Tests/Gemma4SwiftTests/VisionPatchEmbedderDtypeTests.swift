import Testing
import Foundation
import MLX
import MLXNN
@testable import Gemma4Swift

/// Garde-fous P-03 (audit du 2026-09-27) sur la patchification vision :
/// - une constante fp32 dans le masquage des positions promouvait tout
///   l'encodeur en fp32 ;
/// - `asType(inputProj.weight.dtype)` convertissait les patches en uint32 quand
///   `input_proj` etait quantifie a la volee (poids packes).
@Suite("Dtype de la patchification vision")
struct VisionPatchEmbedderDtypeTests {

    /// 4 patches de 16×16 (image 32×32), le dernier marque comme padding.
    private func inputs() -> (pixels: MLXArray, positions: MLXArray, padding: MLXArray) {
        let pixels = MLXRandom.uniform(0.0 ..< 1.0, [1, 3, 32, 32]).asType(.bfloat16)
        let positions = MLXArray([0, 0, 0, 1, 1, 0, 1, 1] as [Int32]).reshaped(1, 4, 2)
        let padding = MLXArray([false, false, false, true]).reshaped(1, 4)
        return (pixels, positions, padding)
    }

    private func makeEmbedder() -> VisionPatchEmbedder {
        MLXRandom.seed(0)
        let embedder = VisionPatchEmbedder(Gemma4VisionConfig.defaultConfig)
        embedder.apply { $0.dtype.isFloatingPoint ? $0.asType(.bfloat16) : $0 }
        return embedder
    }

    @Test("poids bf16 : la sortie reste en bf16")
    func testKeepsBf16() {
        let embedder = makeEmbedder()
        let (pixels, positions, padding) = inputs()
        let out = embedder(pixelValues: pixels, patchPositions: positions, paddingPositions: padding)
        #expect(out.dtype == .bfloat16, "sortie en \(out.dtype)")
    }

    @Test("input_proj quantifie : sortie proche de la version pleine precision")
    func testQuantizedInputProj() {
        let reference = makeEmbedder()
        let quantized = makeEmbedder()
        quantize(model: quantized, groupSize: 64, bits: 8)
        #expect(quantized.inputProj is QuantizedLinear, "input_proj non quantifie : le test ne prouve rien")

        let (pixels, positions, padding) = inputs()
        let expected = reference(pixelValues: pixels, patchPositions: positions, paddingPositions: padding)
        let actual = quantized(pixelValues: pixels, patchPositions: positions, paddingPositions: padding)
        let relative = (sqrt(sum(square((expected - actual).asType(.float32))))
            / sqrt(sum(square(expected.asType(.float32))))).item(Float.self)
        #expect(relative < 0.05, "erreur relative L2 \(relative)")
    }
}
