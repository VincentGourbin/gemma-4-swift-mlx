import Testing
import Foundation
@testable import Gemma4Swift

/// Profils d'entrainement LoRA (K-33).
@Suite("Profils d'entrainement LoRA")
struct TrainingProfileTests {

    @Test("12 candidats, identifiants uniques, pack de base coherent avec les bits")
    func testMatrix() {
        let candidates = Gemma4TrainingProfile.candidates
        #expect(candidates.count == 12)
        #expect(Set(candidates.map(\.qualifiedID)).count == candidates.count)
        for profile in candidates {
            #expect(profile.model.family == profile.family, "\(profile.qualifiedID)")
            let bf16 = profile.model.rawValue.hasSuffix("-bf16")
            #expect(bf16 == (profile.bits == .sixteen), "\(profile.qualifiedID)")
        }
    }

    @Test("seuls les profils mesures sont publies")
    func testPublication() {
        #expect(Gemma4TrainingProfile.all.allSatisfy { $0.measurement != nil })
        #expect(Gemma4TrainingProfile.all.count == 11)
        // A diverge a lr 1e-4 : candidat, non publie.
        #expect(Gemma4TrainingProfile.named("b31b/lora-4bit-lean") == nil)
        for profile in Gemma4TrainingProfile.candidates where profile.measurement == nil {
            #expect(Gemma4TrainingProfile.named(profile.qualifiedID) == nil)
            #expect(Gemma4TrainingProfile.named(profile.qualifiedID, includingUnmeasured: true) == profile)
        }
    }

    @Test("lean = checkpointing et cache 1 Go ; fast = cache 2 Go, checkpointing au-dela d'E4B")
    func testKinds() {
        for profile in Gemma4TrainingProfile.candidates {
            let lean = profile.kind == .lean
            let small = profile.family == .e2b || profile.family == .e4b
            #expect(profile.gradientCheckpointing == (lean || !small), "\(profile.qualifiedID)")
            #expect(profile.memoryPolicy.cacheLimitMB == (lean ? 1024 : 2048))
        }
    }

    @Test("apply pose les reglages et laisse iterations, graine et troncature")
    func testApply() throws {
        let profile = try #require(
            Gemma4TrainingProfile.named("a4b/lora-4bit-lean", includingUnmeasured: true))
        var config = Gemma4LoRATrain.TrainingConfig(
            loraRank: 64, numLayers: 3, learningRate: 3e-3, batchSize: 4, iterations: 17,
            seed: 9, maxSeqLength: 512, responseOnlyHead: false)
        profile.apply(to: &config)
        #expect(config.modelFamily == .a4b)
        #expect(config.loraRank == 8 && config.loraScale == 20 && config.numLayers == 10)
        #expect(config.learningRate == 1e-4 && config.batchSize == 1)
        #expect(config.gradientCheckpointing && config.responseOnlyHead)
        #expect(config.memoryPolicy == Gemma4TrainingMemoryPolicy(cacheLimitMB: 1024))
        #expect(config.validationBatches == 25)
        #expect(config.iterations == 17 && config.seed == 9 && config.maxSeqLength == 512)
    }
}
