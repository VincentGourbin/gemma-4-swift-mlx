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
    private func makeConfig(
        numSoftTokens: Int = 280,
        patchSize: Int = 16,
        poolingKernelSize: Int = 3,
        modelPatchSize: Int = 48
    ) throws -> Gemma4UnifiedVisionConfig {
        let json = """
        {
          "model_type": "gemma4_unified_vision",
          "model_patch_size": \(modelPatchSize),
          "patch_size": \(patchSize),
          "pooling_kernel_size": \(poolingKernelSize),
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
        return pixelData.withUnsafeMutableBytes { raw in
            CGContext(
                data: raw.baseAddress,
                width: width, height: height,
                bitsPerComponent: 8, bytesPerRow: bytesPerRow,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
            )!.makeImage()!
        }
    }

    /// Rejoue le rendu CoreGraphics du processeur a une taille donnee.
    /// Meme discipline que Gemma4CGImageLoader.rgbTensor : le bitmap est cede a
    /// CoreGraphics dans une portee `withUnsafeMutableBytes`. Passer `&pixelData`
    /// ferait echapper un pointeur temporaire hors de l'appel qui l'a cree, ce
    /// que la doc Swift classe en comportement indefini — et ce serait
    /// reintroduire dans le test le defaut que le loader de production a corrige.
    private func renderRGBA(_ image: CGImage, width: Int, height: Int) -> [UInt8] {
        let bytesPerRow = 4 * width
        var pixelData = [UInt8](repeating: 0, count: height * bytesPerRow)
        pixelData.withUnsafeMutableBytes { raw in
            let context = CGContext(
                data: raw.baseAddress,
                width: width, height: height,
                bitsPerComponent: 8, bytesPerRow: bytesPerRow,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
            )!
            context.interpolationQuality = .high
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        }
        return pixelData
    }

    /// Image unie — beaucoup moins couteuse que le degrade, pour les tests qui
    /// ne lisent que des formes et des compteurs.
    private func makeSolidCGImage(width: Int, height: Int) -> CGImage {
        let bytesPerRow = 4 * width
        var pixelData = [UInt8](repeating: 200, count: height * bytesPerRow)
        return pixelData.withUnsafeMutableBytes { raw in
            CGContext(
                data: raw.baseAddress,
                width: width, height: height,
                bitsPerComponent: 8, bytesPerRow: bytesPerRow,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
            )!.makeImage()!
        }
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
        // Le tenseur est indexe en patches MODELE (48 px), pas en patches fins
        // (16 px) : une ligne = un patch de 48 px = un soft token.
        #expect(processed.patches.dim(0) == config.maxModelPatches)
        #expect(config.maxModelPatches == config.numSoftTokens)
        #expect(config.maxPatches == config.maxModelPatches * 9) // unites distinctes
        #expect(processed.patches.dim(1) == config.patchDim) // 48*48*3 = 6912
        #expect(processed.positionIds.shape == [config.maxModelPatches, 2])

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

        // `try #require` et non `#expect` : #expect ne halte pas, et une slice MLX
        // de zero ligne juste apres avorterait tout le process xctest au lieu de
        // faire echouer ce seul test.
        let valid = processed.validPatches
        try #require(valid < config.maxModelPatches) // sinon le test ne prouve rien

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
            // Tolerance tres serree plutot qu'egalite exacte : la division par 255
            // est faite par un kernel Metal d'un cote et par le CPU de l'autre.
            // 1e-6 reste 3900x plus fin qu'un pas d'octet (1/255), donc toute
            // erreur d'axe ou de canal est toujours attrapee.
            let actual = processed.patches[py * pW + px].asArray(Float.self)
            #expect(actual.count == expected.count)
            let maxDelta = zip(actual, expected).map { abs($0 - $1) }.max() ?? 0
            #expect(maxDelta < 1e-6, "patch (\(py), \(px)) ne correspond pas au bloc source")
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

        // Sur TOUS les patches valides, pas seulement le premier : l'image est
        // unie, donc le reechantillonnage ne peut rien produire d'autre que du
        // rouge, y compris sur les bords. Et on exclut le padding — l'assertion
        // `min() >= 0` sur le tenseur entier etait vacue, le padding etant a zero.
        let mp = config.modelPatchSize
        let valid = processed.patches[0 ..< processed.validPatches]
            .reshaped(processed.validPatches * mp * mp, 3)

        #expect(abs(valid[0..., 0] - MLXArray(Float(1.0))).max().item(Float.self) < 1e-6)
        #expect(abs(valid[0..., 1]).max().item(Float.self) < 1e-6)
        #expect(abs(valid[0..., 2]).max().item(Float.self) < 1e-6)
    }

    // MARK: - Fallbacks aspect-ratio

    @Test("Une image plus petite qu'une cellule est agrandie jusqu'au budget")
    func testTinyImageIsUpscaledToBudget() throws {
        // Le nom precedent ("produit au moins une cellule") decrivait la branche
        // de fallback bestH == 0 && bestW == 0, qui est inatteignable, et
        // n'assertait que des proprietes vraies pour toute entree. Le vrai
        // comportement est un agrandissement : le resize vise le budget de pixels
        // quelle que soit la taille source.
        let config = try makeConfig()
        let processed = try Gemma4UnifiedImageProcessor.processImage(
            makeGradientCGImage(width: 8, height: 8), config: config)

        let (pW, pH) = grid(of: processed)
        #expect(pW == pH)                                   // source carree
        #expect(processed.validPatches == pW * pH)
        #expect(processed.validPatches > config.maxModelPatches / 2,
                "8x8 devrait saturer le budget par agrandissement, pas rester minuscule")
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

    @Test("validPatches ne depasse jamais numSoftTokens, quelle que soit la taille source")
    func testPatchBudgetInvariantAcrossSizes() throws {
        // L'invariant qui autorise a padder a numSoftTokens plutot qu'a
        // maxPatches : le resize vise numSoftTokens * modelPatchSize^2 pixels,
        // donc au plus numSoftTokens cellules de 48x48. Verifie sur un balayage
        // plutot que sur un seul cas, les arrondis Float du calcul de facteur
        // etant le seul endroit ou l'invariant pourrait ceder.
        let config = try makeConfig()

        // Petites tailles croisees : bornes de cellule (47/48/49), carres et
        // ratios modestes. Les grandes images sont testees a part, leur
        // generation scalaire dominant le temps du test.
        let sides = [1, 7, 47, 48, 49, 96, 337]
        var cases = sides.flatMap { w in sides.map { h in (w, h) } }

        // Cas larges et ratios extremes, ou l'invariant est le plus tendu :
        // budget sature, et les deux branches de fallback.
        cases += [(640, 320), (1920, 1080), (1081, 1080), (2810, 10), (10, 2810), (4001, 3)]

        for (w, h) in cases {
            let processed = try Gemma4UnifiedImageProcessor.processImage(
                makeSolidCGImage(width: w, height: h), config: config)

            #expect(
                processed.validPatches <= config.maxModelPatches,
                "\(w)x\(h) produit \(processed.validPatches) patches > \(config.maxModelPatches)")
            #expect(processed.validPatches >= 1, "\(w)x\(h) ne produit aucun patch")
            // Forme constante quelle que soit l'image : c'est ce qui permet
            // d'empiler plusieurs images sur l'axe batch.
            #expect(processed.patches.dim(0) == config.maxModelPatches)
        }
    }

    @Test("Un budget video reduit padde a son propre numSoftTokens")
    func testReducedTokenBudget() throws {
        // Gemma4UnifiedVideoProcessor derive une config a 70 soft tokens par
        // frame : le padding doit suivre ce budget, pas celui des images fixes.
        let config = try makeConfig(numSoftTokens: 70)
        let processed = try Gemma4UnifiedImageProcessor.processImage(
            makeGradientCGImage(width: 640, height: 320), config: config)

        #expect(processed.patches.dim(0) == 70)
        #expect(processed.validPatches <= 70)
    }

    @Test("Une config incoherente est rejetee au decodage")
    func testInconsistentConfigRejected() throws {
        // C'est l'identite qui rend `maxModelPatches == numSoftTokens` vraie.
        // Sans elle la borne reelle est numSoftTokens * (patchSize *
        // poolingKernelSize / modelPatchSize)^2 : 498 patches modele pour
        // pooling_kernel_size=4, contre 280 emplacements de position ids.
        #expect(throws: DecodingError.self) { _ = try makeConfig(poolingKernelSize: 4) }
        #expect(throws: DecodingError.self) { _ = try makeConfig(modelPatchSize: 32) }
        #expect(throws: DecodingError.self) { _ = try makeConfig(patchSize: 8) }

        // Dimensions degenerees : padTarget == 0 faisait ecrire l'indice 0 dans
        // un tableau vide (atteignable via un softTokensPerFrame nul cote video).
        #expect(throws: DecodingError.self) { _ = try makeConfig(numSoftTokens: 0) }
        #expect(throws: DecodingError.self) { _ = try makeConfig(poolingKernelSize: 0) }

        // Et les combinaisons coherentes non standard restent acceptees.
        for (ps, pk) in [(16, 3), (16, 2), (8, 3), (16, 4), (32, 2)] {
            let config = try makeConfig(
                patchSize: ps, poolingKernelSize: pk, modelPatchSize: ps * pk)
            #expect(config.maxModelPatches == config.numSoftTokens)
        }
    }

    @Test("L'invariant tient aussi quand on fait varier les tailles de patch")
    func testInvariantAcrossPatchGeometries() throws {
        // Le balayage par taille d'image ne bouge que l'axe prouvablement sur.
        // L'axe qui casse reellement la borne, c'est la geometrie des patches.
        for (ps, pk) in [(16, 3), (16, 2), (8, 3), (16, 4), (32, 2), (8, 6)] {
            let config = try makeConfig(
                numSoftTokens: 280, patchSize: ps, poolingKernelSize: pk,
                modelPatchSize: ps * pk)

            for (w, h) in [(640, 320), (1920, 1080), (2810, 10), (48, 48)] {
                let processed = try Gemma4UnifiedImageProcessor.processImage(
                    makeSolidCGImage(width: w, height: h), config: config)

                #expect(
                    processed.validPatches <= config.maxModelPatches,
                    "ps=\(ps) pk=\(pk) sur \(w)x\(h) : \(processed.validPatches) > \(config.maxModelPatches)")
                #expect(processed.patches.dim(0) == config.maxModelPatches)
                #expect(processed.patches.dim(1) == config.patchDim)
            }
        }
    }

    @Test("Le resize sature effectivement le budget de pixels")
    func testResizeMagnitude() throws {
        // Rien ne fixait l'echelle absolue du resize : diviser targetPx par 9
        // laissait toute la suite verte, alors que chaque image perdrait 9x sa
        // resolution. Une grande image doit saturer le budget.
        let config = try makeConfig()
        let processed = try Gemma4UnifiedImageProcessor.processImage(
            makeSolidCGImage(width: 1920, height: 1280), config: config)

        #expect(processed.validPatches >= config.maxModelPatches * 9 / 10,
                "1920x1280 ne remplit que \(processed.validPatches) / \(config.maxModelPatches) cellules")
        #expect(processed.validPatches <= config.maxModelPatches)
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
