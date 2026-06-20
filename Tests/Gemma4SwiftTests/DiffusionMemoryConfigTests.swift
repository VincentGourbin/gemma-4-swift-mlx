// Tests des presets memoire DiffusionGemma — verifie que les configs
// sont coherentes et que recommended(forRAMGB:) retourne le bon preset.

import XCTest
@testable import Gemma4Swift

final class DiffusionMemoryConfigTests: XCTestCase {

    // MARK: - Presets statiques

    func testDisabledPreset() {
        let cfg = DiffusionMemoryConfig.disabled
        XCTAssertNil(cfg.mixedPrecision, "disabled doit etre en bf16 (mixedPrecision nil)")
        XCTAssertFalse(cfg.unloadVisionAfterFirstCanvas)
        XCTAssertFalse(cfg.clearCacheBetweenCanvases)
    }

    func testLightPreset() {
        let cfg = DiffusionMemoryConfig.light
        XCTAssertNil(cfg.mixedPrecision)
        XCTAssertTrue(cfg.unloadVisionAfterFirstCanvas)
        XCTAssertTrue(cfg.clearCacheBetweenCanvases)
    }

    func testModeratePreset() {
        let cfg = DiffusionMemoryConfig.moderate
        XCTAssertNotNil(cfg.mixedPrecision, "moderate doit avoir mixed precision")
        XCTAssertEqual(cfg.mixedPrecision?.highPrecisionBits, 8)
        XCTAssertEqual(cfg.mixedPrecision?.lowPrecisionBits, 4)
        XCTAssertTrue(cfg.unloadVisionAfterFirstCanvas)
        XCTAssertTrue(cfg.clearCacheBetweenCanvases)
    }

    func testAggressivePreset() {
        let cfg = DiffusionMemoryConfig.aggressive
        XCTAssertNotNil(cfg.mixedPrecision)
        let high = cfg.mixedPrecision?.highPrecisionLayers ?? []
        // default : 4 premiers + 4 derniers (8 layers)
        XCTAssertEqual(high.count, 8)
        XCTAssertTrue(cfg.unloadVisionAfterFirstCanvas)
        XCTAssertTrue(cfg.clearCacheBetweenCanvases)
    }

    func testExtremePreset() {
        let cfg = DiffusionMemoryConfig.extreme
        // aggressive mixed (2+2)
        XCTAssertEqual(cfg.mixedPrecision?.highPrecisionLayers.count, 4)
        XCTAssertTrue(cfg.unloadVisionAfterFirstCanvas)
        XCTAssertTrue(cfg.clearCacheBetweenCanvases)
    }

    func testDefaultIsLight() {
        let def = DiffusionMemoryConfig.default
        let light = DiffusionMemoryConfig.light
        XCTAssertEqual(def.unloadVisionAfterFirstCanvas, light.unloadVisionAfterFirstCanvas)
        XCTAssertEqual(def.clearCacheBetweenCanvases, light.clearCacheBetweenCanvases)
        // mixedPrecision est optional, on verifie qu'ils sont tous deux nil
        XCTAssertNil(def.mixedPrecision)
        XCTAssertNil(light.mixedPrecision)
    }

    // MARK: - recommended(forRAMGB:)

    func testRecommendedTinyRAM() {
        let cfg = DiffusionMemoryConfig.recommended(forRAMGB: 16)
        // extreme : mixed aggressive
        XCTAssertEqual(cfg.mixedPrecision?.highPrecisionLayers.count, 4)
    }

    func testRecommendedSmallRAM() {
        let cfg = DiffusionMemoryConfig.recommended(forRAMGB: 32)
        // aggressive : mixed default (8 layers high-prec)
        XCTAssertEqual(cfg.mixedPrecision?.highPrecisionLayers.count, 8)
    }

    func testRecommendedMediumRAM() {
        let cfg = DiffusionMemoryConfig.recommended(forRAMGB: 64)
        // moderate : mixed conservative (12 layers high-prec)
        XCTAssertEqual(cfg.mixedPrecision?.highPrecisionLayers.count, 12)
    }

    func testRecommendedLargeRAM() {
        let cfg = DiffusionMemoryConfig.recommended(forRAMGB: 96)
        // light : pas de mixed precision
        XCTAssertNil(cfg.mixedPrecision)
        XCTAssertTrue(cfg.unloadVisionAfterFirstCanvas)
    }

    func testRecommendedHugeRAM() {
        let cfg = DiffusionMemoryConfig.recommended(forRAMGB: 192)
        // disabled : tout en bf16, aucune optim
        XCTAssertNil(cfg.mixedPrecision)
        XCTAssertFalse(cfg.unloadVisionAfterFirstCanvas)
        XCTAssertFalse(cfg.clearCacheBetweenCanvases)
    }

    // MARK: - Monotonie : plus de RAM = moins d'optim

    func testRecommendedMonotony() {
        // Plus on a de RAM, moins on quantize
        let ram16 = DiffusionMemoryConfig.recommended(forRAMGB: 16)
        let ram32 = DiffusionMemoryConfig.recommended(forRAMGB: 32)
        let ram64 = DiffusionMemoryConfig.recommended(forRAMGB: 64)

        // 16 GB : layers high-prec minimaux (aggressive mixed = 4 layers)
        // 32 GB : layers high-prec moderes (default mixed = 8 layers)
        // 64 GB : layers high-prec maximaux (conservative mixed = 12 layers)
        let high16 = ram16.mixedPrecision?.highPrecisionLayers.count ?? 0
        let high32 = ram32.mixedPrecision?.highPrecisionLayers.count ?? 0
        let high64 = ram64.mixedPrecision?.highPrecisionLayers.count ?? 0

        XCTAssertLessThan(high16, high32, "Avec moins de RAM, on doit quantizer plus agressivement")
        XCTAssertLessThan(high32, high64, "Conservative > default")
    }
}
