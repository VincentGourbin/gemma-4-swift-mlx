import XCTest
import MLX
import MLXNN
@testable import Gemma4Swift

final class RoPETests: XCTestCase {

    // MARK: - RoPEFactory

    func testRoPEFactoryCreatesStandardRoPEByDefault() {
        let wrapper = RoPEFactory.create(dims: 64, base: 10000.0)
        // With ropeType "default", inner should not be a ProportionalRoPE
        XCTAssertFalse(wrapper.inner is ProportionalRoPE, "Default ropeType should not produce a ProportionalRoPE")
    }

    func testRoPEFactoryCreatesProportionalRoPE() {
        let wrapper = RoPEFactory.create(
            dims: 64,
            base: 10000.0,
            ropeType: "proportional",
            partialRotaryFactor: 0.25
        )
        XCTAssertTrue(wrapper.inner is ProportionalRoPE, "ropeType 'proportional' should produce a ProportionalRoPE")
    }

    // MARK: - ProportionalRoPE output shape

    func testProportionalRoPEOutputShape() {
        let rope = ProportionalRoPE(dims: 64, base: 10000.0, partialRotaryFactor: 1.0)
        // Input shape: [batch=1, seqLen=4, heads=1, headDim=64]
        let input = MLXArray.zeros([1, 4, 1, 64])
        let output = rope(input, offset: 0)
        eval(output)
        XCTAssertEqual(output.shape, input.shape, "ProportionalRoPE must preserve input shape")
    }

    // MARK: - Partial rotation

    func testProportionalRoPEPartialRotationLeavesUnrotatedDimsUnchanged() {
        // partialRotaryFactor=0.25 on dims=64 => rotatedDims = 2*(0.25*64/2) = 16
        // The head is split into left=[0..31] and right=[32..63].
        // rotHalf = 16/2 = 8, so left[8..31] and right[8..31] are pass-through (unchanged).
        let rope = ProportionalRoPE(dims: 64, base: 10000.0, partialRotaryFactor: 0.25)

        // Use a recognizable non-zero tensor so we can detect unchanged values
        var floats = [Float](repeating: 0.0, count: 64)
        for i in 0..<64 { floats[i] = Float(i + 1) }
        // Shape [1, 1, 1, 64]: batch=1, seqLen=1, heads=1, headDim=64
        let input = MLXArray(floats).reshaped([1, 1, 1, 64])
        // Use offset > 0 so RoPE actually rotates (at offset=0, cos(0)=1 sin(0)=0 → identity)
        let output = rope(input, offset: 5)
        eval(input)
        eval(output)

        let inValues = input.reshaped([-1]).asArray(Float.self)
        let outValues = output.reshaped([-1]).asArray(Float.self)

        // rotHalf = 8
        // left half = dims [0..31], right half = dims [32..63]
        // Pass-through within left:  indices 8..31
        // Pass-through within right: indices 40..63
        let passthroughIndices = Array(8..<32) + Array(40..<64)

        for idx in passthroughIndices {
            XCTAssertEqual(
                outValues[idx], inValues[idx], accuracy: 1e-5,
                "Index \(idx) should be unchanged (pass-through, not rotated)"
            )
        }

        // Sanity: the rotated region (indices 0..7 and 32..39) should differ from input
        let rotatedIndices = Array(0..<8) + Array(32..<40)
        let anyRotated = rotatedIndices.contains { idx in
            abs(outValues[idx] - inValues[idx]) > 1e-5
        }
        XCTAssertTrue(anyRotated, "At least some of the rotated dims should differ from input")
    }
}
