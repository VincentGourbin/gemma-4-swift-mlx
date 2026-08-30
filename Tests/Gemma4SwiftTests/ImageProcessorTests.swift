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

    /// CGImage non uniforme : R croit avec x, G croit avec y, B constant.
    /// Contrairement au rouge uni, une transposition d'axes ou une permutation
    /// de canaux change le resultat.
    private func makeGradientCGImage(width: Int, height: Int) -> CGImage {
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        var pixelData = [UInt8](repeating: 0, count: height * bytesPerRow)
        for y in 0 ..< height {
            for x in 0 ..< width {
                let i = y * width + x
                pixelData[i * 4] = UInt8((x * 255) / max(width - 1, 1))
                pixelData[i * 4 + 1] = UInt8((y * 255) / max(height - 1, 1))
                pixelData[i * 4 + 2] = 77
                pixelData[i * 4 + 3] = 255
            }
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

    @Test("La conversion vectorisee preserve l'ordre des canaux RGB")
    func testChannelOrder() throws {
        // Image rouge unie : canal 0 a 1.0, canaux 1 et 2 a 0.0.
        let cgImage = makeTestCGImage(width: 96, height: 96)
        let result = try Gemma4ImageProcessor.processImage(cgImage)

        let r = result[0, 0].mean().item(Float.self)
        let g = result[0, 1].mean().item(Float.self)
        let b = result[0, 2].mean().item(Float.self)
        #expect(abs(r - 1.0) < 1e-5)
        #expect(abs(g) < 1e-5)
        #expect(abs(b) < 1e-5)
    }

    @Test("La surcharge async produit exactement le meme resultat que la synchrone")
    func testAsyncMatchesSync() async throws {
        let cgImage = makeTestCGImage(width: 640, height: 480)
        let sync = try Gemma4ImageProcessor.processImage(cgImage)
        let async = try await Gemma4ImageProcessor.processImage(cgImage, priority: .userInitiated)

        #expect(sync.shape == async.shape)
        let maxDiff = abs(sync - async).max().item(Float.self)
        #expect(maxDiff == 0.0)
    }

    @Test("La surcharge async depuis une URL invalide lance la meme erreur")
    func testAsyncInvalidURL() async throws {
        let badURL = URL(fileURLWithPath: "/nonexistent/image.png")
        await #expect(throws: ImageProcessingError.self) {
            try await Gemma4ImageProcessor.processImage(url: badURL, priority: .utility)
        }
    }

    @Test("La conversion vectorisee est bit-a-bit identique a la formule scalaire d'origine")
    func testGoldenAgainstScalarReference() throws {
        // Image non carree ET non uniforme : c'est ce qui rend le test sensible
        // a l'ordre des axes. Sur un carre rouge uni, transposer H et W passe
        // inapercu.
        let cgImage = makeGradientCGImage(width: 320, height: 200)
        let result = try Gemma4ImageProcessor.processImage(cgImage)

        let h = result.dim(2)
        let w = result.dim(3)
        #expect(h != w) // sinon le test ne prouve rien sur l'ordre des axes

        // Rejoue le meme rendu CoreGraphics a la taille effectivement choisie,
        // puis applique la formule scalaire par canal de l'implementation
        // d'origine (3 x [Float] + concatenated sur l'axe 1).
        let bytesPerRow = 4 * w
        var pixelData = [UInt8](repeating: 0, count: h * bytesPerRow)
        let context = CGContext(
            data: &pixelData,
            width: w, height: h,
            bitsPerComponent: 8, bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        )!
        context.interpolationQuality = .high
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: w, height: h))

        var expected = [Float](repeating: 0, count: 3 * h * w)
        for c in 0 ..< 3 {
            for i in 0 ..< h * w {
                expected[c * h * w + i] = Float(pixelData[i * 4 + c]) / 255.0
            }
        }
        let reference = MLXArray(expected).reshaped(1, 3, h, w)

        #expect(result.shape == reference.shape)
        #expect(abs(result - reference).max().item(Float.self) == 0.0)
    }

    @Test("Un ratio extreme conserve son aspect au lieu de s'ecraser en 48x48")
    func testExtremeAspectRatio() throws {
        // Bande type ligne d'OCR : 2 * 20 < 48, la recherche de taille ne
        // produit aucun candidat et retombait silencieusement sur 48x48.
        let strip = try Gemma4ImageProcessor.processImage(makeGradientCGImage(width: 1000, height: 20))
        #expect(strip.dim(2) == 48)
        #expect(strip.dim(3) > 48)
        #expect(strip.dim(3) % 48 == 0)

        // Meme chose dans l'autre sens.
        let tall = try Gemma4ImageProcessor.processImage(makeGradientCGImage(width: 20, height: 1000))
        #expect(tall.dim(3) == 48)
        #expect(tall.dim(2) > 48)

        // Le budget de patches reste respecte dans les deux cas.
        for out in [strip, tall] {
            #expect((out.dim(3) / 16) * (out.dim(2) / 16) <= 280 * 9)
        }
    }

    @Test("Differents maxSoftTokens produisent des tailles differentes")
    func testDifferentTokenBudgets() throws {
        let cgImage = makeTestCGImage(width: 1024, height: 768)
        let result280 = try Gemma4ImageProcessor.processImage(cgImage, maxSoftTokens: 280)
        let result70 = try Gemma4ImageProcessor.processImage(cgImage, maxSoftTokens: 70)

        let area280 = result280.dim(2) * result280.dim(3)
        let area70 = result70.dim(2) * result70.dim(3)
        #expect(area280 > area70)
    }
}
