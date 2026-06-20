// Tests pour DiffusionOnTheFlyQuantization.MixedPrecisionConfig.
// Verifie que les presets sont coherents avec les findings Q-DiT / ViDiT-Q.

import XCTest
@testable import Gemma4Swift

final class DiffusionMixedPrecisionTests: XCTestCase {

    typealias MPConfig = DiffusionOnTheFlyQuantization.MixedPrecisionConfig

    // MARK: - Configuration de base

    func testCustomConfig() {
        let cfg = MPConfig(
            highPrecisionLayers: Set([0, 1, 28, 29]),
            highPrecisionBits: 8,
            lowPrecisionBits: 4,
            groupSize: 64,
            quantizeSensitiveAtHighPrecision: true
        )
        XCTAssertEqual(cfg.highPrecisionLayers, [0, 1, 28, 29])
        XCTAssertEqual(cfg.highPrecisionBits, 8)
        XCTAssertEqual(cfg.lowPrecisionBits, 4)
        XCTAssertEqual(cfg.groupSize, 64)
        XCTAssertTrue(cfg.quantizeSensitiveAtHighPrecision)
    }

    // MARK: - Presets

    func testDefaultPresetLayers() {
        let cfg = MPConfig.default
        // 4 premiers (0..3) + 4 derniers (26..29) = 8 layers high-prec
        XCTAssertEqual(cfg.highPrecisionLayers, Set(0...3).union(Set(26...29)))
        XCTAssertEqual(cfg.highPrecisionLayers.count, 8)
        XCTAssertEqual(cfg.highPrecisionBits, 8)
        XCTAssertEqual(cfg.lowPrecisionBits, 4)
    }

    func testConservativePresetLayers() {
        let cfg = MPConfig.conservative
        // 6 premiers (0..5) + 6 derniers (24..29) = 12 layers high-prec
        XCTAssertEqual(cfg.highPrecisionLayers, Set(0...5).union(Set(24...29)))
        XCTAssertEqual(cfg.highPrecisionLayers.count, 12)
    }

    func testAggressivePresetLayers() {
        let cfg = MPConfig.aggressive
        // 2 premiers (0..1) + 2 derniers (28..29) = 4 layers high-prec
        XCTAssertEqual(cfg.highPrecisionLayers, Set(0...1).union(Set(28...29)))
        XCTAssertEqual(cfg.highPrecisionLayers.count, 4)
    }

    // MARK: - Invariants Q-DiT / ViDiT-Q

    func testHighPrecisionAlwaysContainsFirstAndLast() {
        // Les findings empiriques disent que les COUCHES D'ENTREE et SORTIE
        // sont les plus sensibles. Au minimum la couche 0 et la couche 29 (sur 30)
        // doivent toujours etre en high-precision.
        let presets: [MPConfig] = [.default, .conservative, .aggressive]
        for cfg in presets {
            XCTAssertTrue(cfg.highPrecisionLayers.contains(0), "Layer 0 doit toujours etre high-prec")
            XCTAssertTrue(cfg.highPrecisionLayers.contains(29), "Layer 29 doit toujours etre high-prec")
        }
    }

    func testNoOverlapAtMiddle() {
        // Le milieu (layer 15) ne doit JAMAIS etre high-precision (insensible)
        let presets: [MPConfig] = [.default, .conservative, .aggressive]
        for cfg in presets {
            XCTAssertFalse(cfg.highPrecisionLayers.contains(15), "Layer 15 (milieu) doit etre low-prec")
        }
    }

    func testConservativeHasMoreHighPrecThanDefault() {
        XCTAssertGreaterThan(
            MPConfig.conservative.highPrecisionLayers.count,
            MPConfig.default.highPrecisionLayers.count
        )
    }

    func testDefaultHasMoreHighPrecThanAggressive() {
        XCTAssertGreaterThan(
            MPConfig.default.highPrecisionLayers.count,
            MPConfig.aggressive.highPrecisionLayers.count
        )
    }

    // MARK: - groupSize valid pour DiffusionGemma

    func testGroupSizeDivisibleByDimensions() {
        // Le hidden_size de DiffusionGemma est 2816 et l'intermediate moe est 704.
        // group_size doit diviser tous les derniers axes des Linear quantizes.
        let groupSize = MPConfig.default.groupSize
        XCTAssertEqual(groupSize, 64, "Default group_size = 64 affine")
        XCTAssertEqual(2816 % groupSize, 0, "hidden_size 2816 doit etre divisible par groupSize")
        XCTAssertEqual(704 % groupSize, 0, "moe_intermediate 704 doit etre divisible par groupSize")
        XCTAssertEqual(1408 % groupSize, 0, "moe gate+up 1408 doit etre divisible par groupSize")
    }

    // MARK: - Stats vide par defaut

    func testMixedPrecisionStatsEmpty() {
        let stats = DiffusionOnTheFlyQuantization.MixedPrecisionStats()
        XCTAssertEqual(stats.quantizedHigh, 0)
        XCTAssertEqual(stats.quantizedLow, 0)
        XCTAssertEqual(stats.totalQuantized, 0)
        XCTAssertEqual(stats.skipped.count, 0)
    }
}
