// Tests pour DiffusionGenerationConfig — parsing JSON checkpoint Google.

import XCTest
@testable import Gemma4Swift

final class DiffusionGenerationConfigTests: XCTestCase {

    // MARK: - Defaults

    func testDefaultsMatchGoogleSpecs() {
        // Valeurs du generation_config.json officiel diffusiongemma-26B-A4B-it
        let cfg = DiffusionGenerationConfig()
        XCTAssertEqual(cfg.tMin, 0.4, accuracy: 1e-6, "t_min default Google = 0.4")
        XCTAssertEqual(cfg.tMax, 0.8, accuracy: 1e-6, "t_max default Google = 0.8")
        XCTAssertEqual(cfg.maxDenoisingSteps, 48, "max_denoising_steps Google = 48")
        XCTAssertEqual(cfg.entropyBound, 0.1, accuracy: 1e-6)
        XCTAssertEqual(cfg.stabilityThreshold, 1)
        XCTAssertEqual(cfg.confidenceThreshold, 0.005, accuracy: 1e-6)
        XCTAssertEqual(cfg.padTokenId, 0)
    }

    // MARK: - JSON parsing

    func testDecodeFromJsonObject() throws {
        let json = """
        {
            "t_min": 0.5,
            "t_max": 1.0,
            "max_denoising_steps": 32,
            "sampler_config": {
                "entropy_bound": 0.2
            },
            "stability_threshold": 3,
            "confidence_threshold": 0.01,
            "eos_token_id": [1, 106, 50],
            "pad_token_id": 0
        }
        """.data(using: .utf8)!

        let cfg = try JSONDecoder().decode(DiffusionGenerationConfig.self, from: json)
        XCTAssertEqual(cfg.tMin, 0.5, accuracy: 1e-6)
        XCTAssertEqual(cfg.tMax, 1.0, accuracy: 1e-6)
        XCTAssertEqual(cfg.maxDenoisingSteps, 32)
        XCTAssertEqual(cfg.entropyBound, 0.2, accuracy: 1e-6)
        XCTAssertEqual(cfg.stabilityThreshold, 3)
        XCTAssertEqual(cfg.confidenceThreshold, 0.01, accuracy: 1e-6)
        XCTAssertEqual(cfg.eosTokenIds, [1, 106, 50])
    }

    func testDecodeEosAsSingleInt() throws {
        // generation_config.json peut avoir eos_token_id soit Int soit [Int]
        let json = """
        {
            "t_min": 0.4,
            "t_max": 0.8,
            "max_denoising_steps": 48,
            "eos_token_id": 1
        }
        """.data(using: .utf8)!

        let cfg = try JSONDecoder().decode(DiffusionGenerationConfig.self, from: json)
        XCTAssertEqual(cfg.eosTokenIds, [1])
    }

    func testDecodeEmptyUsesDefaults() throws {
        // Si tous les champs sont absents, on doit avoir les defaults
        let json = "{}".data(using: .utf8)!
        let cfg = try JSONDecoder().decode(DiffusionGenerationConfig.self, from: json)
        XCTAssertEqual(cfg.tMin, 0.4, accuracy: 1e-6)
        XCTAssertEqual(cfg.tMax, 0.8, accuracy: 1e-6)
        XCTAssertEqual(cfg.maxDenoisingSteps, 48)
    }

    // MARK: - Temperature schedule

    func testLinearTemperatureScheduleStartAtTMin() {
        let sched = LinearTemperatureSchedule(tMin: 0.4, tMax: 0.8, maxDenoisingSteps: 48)
        XCTAssertEqual(sched.temperature(curStep: 0), 0.4, accuracy: 1e-5)
    }

    func testLinearTemperatureScheduleEndAtTMax() {
        let sched = LinearTemperatureSchedule(tMin: 0.4, tMax: 0.8, maxDenoisingSteps: 48)
        XCTAssertEqual(sched.temperature(curStep: 48), 0.8, accuracy: 1e-5)
    }

    func testLinearTemperatureScheduleMidpoint() {
        let sched = LinearTemperatureSchedule(tMin: 0.4, tMax: 0.8, maxDenoisingSteps: 48)
        XCTAssertEqual(sched.temperature(curStep: 24), 0.6, accuracy: 1e-5)
    }

    func testLinearTemperatureMonotonic() {
        let sched = LinearTemperatureSchedule(tMin: 0.4, tMax: 0.8, maxDenoisingSteps: 48)
        var prev: Float = -1.0
        for step in 0 ... 48 {
            let t = sched.temperature(curStep: step)
            XCTAssertGreaterThanOrEqual(t, prev, "temperature doit etre monotone croissant")
            prev = t
        }
    }
}
