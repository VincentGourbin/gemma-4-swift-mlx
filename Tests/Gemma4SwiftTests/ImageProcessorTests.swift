// Tests pour Gemma4ImageProcessor — couverture cross-platform (macOS + iOS)

import CoreGraphics
import Foundation
import MLX
import Testing

@testable import Gemma4Swift

@Suite("ImageProcessor Tests")
struct ImageProcessorTests {

    /// Cree un CGImage synthetique de taille donnee (rouge uni)
    private func makeTestCGImage(width: Int, height: Int) -> CGImage {
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        var pixelData = [UInt8](repeating: 0, count: height * bytesPerRow)
        // Remplir en rouge
        for i in 0 ..< width * height {
            pixelData[i * 4] = 255     // R
            pixelData[i * 4 + 1] = 0   // G
            pixelData[i * 4 + 2] = 0   // B
            pixelData[i * 4 + 3] = 255 // A
        }
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(
            data: &pixelData,
            width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        )!
        return context.makeImage()!
    }

    @Test("processImage retourne [1, 3, H, W] avec dimensions divisibles par 48")
    func testOutputShape() throws {
        let cgImage = makeTestCGImage(width: 640, height: 480)
        let result = try Gemma4ImageProcessor.processImage(cgImage)

        #expect(result.ndim == 4)
        #expect(result.dim(0) == 1)
        #expect(result.dim(1) == 3) // RGB channels
        #expect(result.dim(2) % 48 == 0) // H divisible par 48
        #expect(result.dim(3) % 48 == 0) // W divisible par 48
    }

    @Test("Les valeurs sont normalisees entre 0 et 1")
    func testValueRange() throws {
        let cgImage = makeTestCGImage(width: 200, height: 200)
        let result = try Gemma4ImageProcessor.processImage(cgImage)

        let minVal = result.min().item(Float.self)
        let maxVal = result.max().item(Float.self)
        #expect(minVal >= 0.0)
        #expect(maxVal <= 1.0)
    }

    @Test("Le nombre de patches respecte le budget maxSoftTokens")
    func testPatchBudget() throws {
        let cgImage = makeTestCGImage(width: 1920, height: 1080)
        let maxSoftTokens = 280
        let patchSize = 16
        let poolingKernelSize = 3

        let result = try Gemma4ImageProcessor.processImage(
            cgImage, maxSoftTokens: maxSoftTokens, patchSize: patchSize, poolingKernelSize: poolingKernelSize
        )

        let h = result.dim(2)
        let w = result.dim(3)
        let numPatches = (w / patchSize) * (h / patchSize)
        let maxPatches = maxSoftTokens * poolingKernelSize * poolingKernelSize
        #expect(numPatches <= maxPatches)
    }

    @Test("Image carree produit une sortie carree")
    func testSquareImage() throws {
        let cgImage = makeTestCGImage(width: 512, height: 512)
        let result = try Gemma4ImageProcessor.processImage(cgImage)

        #expect(result.dim(2) == result.dim(3))
    }

    @Test("Petite image (< 48px) produit quand meme une sortie minimale 48x48")
    func testSmallImage() throws {
        let cgImage = makeTestCGImage(width: 32, height: 32)
        let result = try Gemma4ImageProcessor.processImage(cgImage)

        #expect(result.dim(2) >= 48)
        #expect(result.dim(3) >= 48)
    }

    @Test("processImage avec URL invalide lance une erreur")
    func testInvalidURL() throws {
        let badURL = URL(fileURLWithPath: "/nonexistent/image.png")
        #expect(throws: ImageProcessingError.self) {
            try Gemma4ImageProcessor.processImage(url: badURL)
        }
    }

    @Test("Differents maxSoftTokens produisent des tailles differentes")
    func testDifferentTokenBudgets() throws {
        let cgImage = makeTestCGImage(width: 1024, height: 768)
        let result280 = try Gemma4ImageProcessor.processImage(cgImage, maxSoftTokens: 280)
        let result70 = try Gemma4ImageProcessor.processImage(cgImage, maxSoftTokens: 70)

        let area280 = result280.dim(2) * result280.dim(3)
        let area70 = result70.dim(2) * result70.dim(3)
        #expect(area280 >= area70)
    }
}
