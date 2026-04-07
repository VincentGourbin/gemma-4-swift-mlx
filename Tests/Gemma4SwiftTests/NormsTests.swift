import XCTest
import MLX
import MLXNN
@testable import Gemma4Swift

final class NormsTests: XCTestCase {

    // MARK: - RMSNormNoScale

    func testRMSNormNoScaleOutputShape() {
        let norm = RMSNormNoScale(eps: 1e-6)
        let input = MLXArray.zeros([2, 4, 8])
        let output = norm(input)
        XCTAssertEqual(output.shape, [2, 4, 8])
    }

    func testRMSNormNoScaleNormalization() {
        let norm = RMSNormNoScale(eps: 1e-6)
        // Input: [1, 1, 4] with values [1.0, 2.0, 3.0, 4.0]
        let input = MLXArray([Float(1.0), Float(2.0), Float(3.0), Float(4.0)]).reshaped([1, 1, 4])
        let output = norm(input)

        // RMS of [1, 2, 3, 4] = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386
        // After normalization each value / RMS, so RMS of output should be ≈ 1.0
        let outputFlat = output.reshaped([-1])
        eval(outputFlat)

        let values = outputFlat.asArray(Float.self)
        let sumSquares = values.reduce(Float(0)) { $0 + $1 * $1 }
        let rms = sqrt(sumSquares / Float(values.count))

        XCTAssertEqual(rms, 1.0, accuracy: 1e-5, "RMS of normalized output should be approximately 1.0")
    }

    // MARK: - RMSNormZeroShift

    func testRMSNormZeroShiftOutputShape() {
        let norm = RMSNormZeroShift(dimensions: 8)
        let input = MLXArray.zeros([2, 4, 8])
        let output = norm(input)
        XCTAssertEqual(output.shape, [2, 4, 8])
    }

    func testRMSNormZeroShiftHasWeights() {
        let norm = RMSNormZeroShift(dimensions: 8)
        let weight = norm.weight
        XCTAssertEqual(weight.shape, [8], "weight parameter should have shape [8]")
    }
}
