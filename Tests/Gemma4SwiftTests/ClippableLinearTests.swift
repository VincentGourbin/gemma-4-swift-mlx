import XCTest
import MLX
import MLXNN
@testable import Gemma4Swift

final class ClippableLinearTests: XCTestCase {

    func testForwardPassShape() {
        let layer = ClippableLinear(inFeatures: 4, outFeatures: 8)
        let input = MLXArray.zeros([1, 4])
        let output = layer(input)
        XCTAssertEqual(output.shape, [1, 8])
    }

    func testClippingEnabled() {
        // When useClipping=true, the layer initializes inputMin/Max/outputMin/Max to ±infinity (no-op bounds).
        // Verify those properties are non-nil, confirming the clipping infrastructure is set up.
        let layer = ClippableLinear(inFeatures: 4, outFeatures: 8, useClipping: true)

        XCTAssertNotNil(layer.inputMin, "inputMin should be set when useClipping is true")
        XCTAssertNotNil(layer.inputMax, "inputMax should be set when useClipping is true")
        XCTAssertNotNil(layer.outputMin, "outputMin should be set when useClipping is true")
        XCTAssertNotNil(layer.outputMax, "outputMax should be set when useClipping is true")

        // Default bounds are ±infinity — forward pass should work without NaN or inf clipping side-effects.
        let input = MLXArray(Array(repeating: Float(1.0), count: 4)).reshaped([1, 4])
        let output = layer(input)
        eval(output)
        XCTAssertEqual(output.shape, [1, 8], "Output shape should be [1, 8] regardless of clipping bounds")
    }

    func testClippingDisabled() {
        let layer = ClippableLinear(inFeatures: 4, outFeatures: 8, useClipping: false)
        XCTAssertNil(layer.inputMin, "inputMin should be nil when useClipping is false")
        XCTAssertNil(layer.inputMax, "inputMax should be nil when useClipping is false")
        XCTAssertNil(layer.outputMin, "outputMin should be nil when useClipping is false")
        XCTAssertNil(layer.outputMax, "outputMax should be nil when useClipping is false")
    }

    func testLinearSubmoduleWeightShape() {
        let layer = ClippableLinear(inFeatures: 4, outFeatures: 8, bias: false, useClipping: false)
        // Linear stores weight as [outFeatures, inFeatures]
        let weightShape = layer.linear.weight.shape
        XCTAssertEqual(weightShape, [8, 4], "linear.weight should have shape [outFeatures, inFeatures] = [8, 4]")
    }
}
