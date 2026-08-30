// Tests pour Gemma4UnifiedImageProcessor (12B Unified, patch projector sans encoder).
//
// Complement de [[ImageProcessorTests]] : le chemin pixel est desormais partage
// (Gemma4CGImageLoader.rgbTensor), mais le decoupage en patches et les position
// ids sont propres a ce processeur. Cf. issue #45.

import CoreGraphics
import Foundation
import MLX
import Testing

@testable import Gemma4Swift

@Suite("Unified ImageProcessor Tests")
struct UnifiedImageProcessorTests {

    /// Le config n'a qu'un `init(from:)`, donc pas d'init memberwise : on passe
    /// par le decodeur, ce qui exerce au passage le chemin de chargement reel.
    private func makeConfig(numSoftTokens: Int = 280) throws -> Gemma4UnifiedVisionConfig {
        let json = """
        {
          "model_type": "gemma4_unified_vision",
          "model_patch_size": 48,
          "patch_size": 16,
          "pooling_kernel_size": 3,
          "num_soft_tokens": \(numSoftTokens)
        }
        """.data(using: .utf8)!
        return try JSONDecoder().decode(Gemma4UnifiedVisionConfig.self, from: json)
    }

    /// Image non uniforme : R croit avec x, G croit avec y, B constant. Une
    /// permutation d'axes ou un melange de patches devient detectable.
    private func makeGradientCGImage(width: Int, height: Int) -> CGImage {
        let bytesPerRow = 4 * width
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
        let context = CGContext(
            data: &pixelData,
            width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        )!
        return context.makeImage()!
    }

    /// Rejoue le rendu CoreGraphics du processeur a une taille donnee.
    private func renderRGBA(_ image: CGImage, width: Int, height: Int) -> [UInt8] {
        let bytesPerRow = 4 * width
        var pixelData = [UInt8](repeating: 0, count: height * bytesPerRow)
        let context = CGContext(
            data: &pixelData,
            width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        )!
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return pixelData
    }

    /// Grille de patches deduite des position ids, plutot que de dupliquer la
    /// formule de resize du processeur.
    private func grid(of processed: Gemma4UnifiedImageProcessor.ProcessedImage) -> (pW: Int, pH: Int) {
        let valid = processed.positionIds[0 ..< processed.validPatches]
        let pW = Int(valid[0..., 0].max().item(Int32.self)) + 1
        let pH = Int(valid[0..., 1].max().item(Int32.self)) + 1
        return (pW, pH)
    }

    // MARK: - Formes et contrat de padding

    @Test("Formes de sortie et nombre de patches valides")
    func testShapes() throws {
        let config = try makeConfig()
        let processed = try Gemma4UnifiedImageProcessor.processImage(
            makeGradientCGImage(width: 640, height: 320), config: config)

        #expect(processed.patches.ndim == 2)
        #expect(processed.patches.dim(0) == config.maxPatches)
        #expect(processed.patches.dim(1) == config.patchDim) // 48*48*3 = 6912
        #expect(processed.positionIds.shape == [config.maxPatches, 2])

        let (pW, pH) = grid(of: processed)
        #expect(pW * pH == processed.validPatches)
        #expect(processed.validPatches > 0)
        #expect(processed.validPatches <= config.numSoftTokens)
    }

    @Test("Les lignes de padding sont a zero et leurs positions a -1")
    func testPaddingContract() throws {
        let config = try makeConfig()
        let processed = try Gemma4UnifiedImageProcessor.processImage(
            makeGradientCGImage(width: 640, height: 320), config: config)

        let valid = processed.validPatches
        #expect(valid < config.maxPatches) // sinon le test ne prouve rien

        let padPatches = processed.patches[valid...]
        #expect(abs(padPatches).max().item(Float.self) == 0.0)

        let padPositions = processed.positionIds[valid...]
        #expect(padPositions.max().item(Int32.self) == -1)
    }

    @Test("Les position ids sont (x, y) en row-major")
    func testPositionIdsLayout() throws {
        let config = try makeConfig()
        let processed = try Gemma4UnifiedImageProcessor.processImage(
            makeGradientCGImage(width: 640, height: 320), config: config)

        let (pW, pH) = grid(of: processed)
        let positions = processed.positionIds[0 ..< processed.validPatches].asArray(Int32.self)

        for py in 0 ..< pH {
            for px in 0 ..< pW {
                let idx = py * pW + px
                #expect(positions[idx * 2] == Int32(px))
                #expect(positions[idx * 2 + 1] == Int32(py))
            }
        }
    }

    // MARK: - Contenu des patches

    @Test("Le patch k est bien le bloc (py, px) de l'image redimensionnee")
    func testPatchContentMatchesResizedImage() throws {
        let config = try makeConfig()
        let source = makeGradientCGImage(width: 640, height: 320)
        let processed = try Gemma4UnifiedImageProcessor.processImage(source, config: config)

        let (pW, pH) = grid(of: processed)
        let mp = config.modelPatchSize
        let w = pW * mp
        let h = pH * mp
        let rgba = renderRGBA(source, width: w, height: h)

        // Coins et centre : une permutation dans le transposed(0, 2, 1, 3, 4)
        // melangerait les patches sans changer aucune dimension.
        let probes = [(0, 0), (0, pW - 1), (pH - 1, 0), (pH - 1, pW - 1), (pH / 2, pW / 2)]
        for (py, px) in probes {
            var expected = [Float]()
            expected.reserveCapacity(config.patchDim)
            for y in 0 ..< mp {
                for x in 0 ..< mp {
                    for c in 0 ..< 3 {
                        let pixel = ((py * mp + y) * w + (px * mp + x)) * 4 + c
                        expected.append(Float(rgba[pixel]) / 255.0)
                    }
                }
            }
            let actual = processed.patches[py * pW + px].asArray(Float.self)
            #expect(actual == expected, "patch (\(py), \(px)) ne correspond pas au bloc source")
        }
    }

    @Test("Les valeurs sont dans [0, 1] et l'ordre des canaux est preserve")
    func testValueRangeAndChannelOrder() throws {
        let config = try makeConfig()
        // Image rouge unie : dans chaque triplet RGB aplati, R = 1, G = B = 0.
        let width = 96, height = 96
        let bytesPerRow = 4 * width
        var pixelData = [UInt8](repeating: 0, count: height * bytesPerRow)
        for i in 0 ..< width * height {
            pixelData[i * 4] = 255
            pixelData[i * 4 + 3] = 255
        }
        let cg = CGContext(
            data: &pixelData, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        )!.makeImage()!

        let processed = try Gemma4UnifiedImageProcessor.processImage(cg, config: config)
        let firstPatch = processed.patches[0].reshaped(config.modelPatchSize * config.modelPatchSize, 3)

        #expect(processed.patches.min().item(Float.self) >= 0.0)
        #expect(processed.patches.max().item(Float.self) <= 1.0)
        #expect(abs(firstPatch[0..., 0] - MLXArray(Float(1.0))).max().item(Float.self) < 1e-5)
        #expect(abs(firstPatch[0..., 1]).max().item(Float.self) < 1e-5)
        #expect(abs(firstPatch[0..., 2]).max().item(Float.self) < 1e-5)
    }

    // MARK: - Fallbacks aspect-ratio

    @Test("Une image minuscule produit au moins une cellule")
    func testTinyImage() throws {
        let config = try makeConfig()
        let processed = try Gemma4UnifiedImageProcessor.processImage(
            makeGradientCGImage(width: 8, height: 8), config: config)

        #expect(processed.validPatches >= 1)
        let (pW, pH) = grid(of: processed)
        #expect(pW >= 1 && pH >= 1)
    }

    @Test("Un ratio extreme passe par le fallback sans depasser le budget")
    func testExtremeAspectRatioFallback() throws {
        let config = try makeConfig()
        // W/H > 280 : floor(factor * H / 48) tombe a 0, le fallback bestH == 0
        // prend la main.
        let processed = try Gemma4UnifiedImageProcessor.processImage(
            makeGradientCGImage(width: 2810, height: 10), config: config)

        let (pW, pH) = grid(of: processed)
        #expect(pH == 1)                    // une seule rangee de cellules
        #expect(pW > 1)                     // le ratio n'est pas ecrase en carre
        #expect(processed.validPatches == pW * pH)
        #expect(processed.validPatches <= config.numSoftTokens)
    }

    // MARK: - Surcharges async

    @Test("La surcharge async produit exactement le meme resultat que la synchrone")
    func testAsyncMatchesSync() async throws {
        let config = try makeConfig()
        let source = makeGradientCGImage(width: 640, height: 320)

        let sync = try Gemma4UnifiedImageProcessor.processImage(source, config: config)
        let asyncResult = try await Gemma4UnifiedImageProcessor.processImage(
            source, config: config, priority: .userInitiated)

        #expect(asyncResult.validPatches == sync.validPatches)
        #expect(asyncResult.patches.shape == sync.patches.shape)
        #expect(abs(sync.patches - asyncResult.patches).max().item(Float.self) == 0.0)
        #expect((sync.positionIds - asyncResult.positionIds).abs().max().item(Int32.self) == 0)
    }

    @Test("La surcharge async depuis une URL invalide lance la meme erreur")
    func testAsyncInvalidURL() async throws {
        let config = try makeConfig()
        let badURL = URL(fileURLWithPath: "/nonexistent/image.png")
        await #expect(throws: ImageProcessingError.self) {
            try await Gemma4UnifiedImageProcessor.processImage(
                url: badURL, config: config, priority: .utility)
        }
    }
}
