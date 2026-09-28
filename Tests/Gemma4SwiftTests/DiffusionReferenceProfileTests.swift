import Testing
import Foundation
@testable import Gemma4Swift

/// Profils de reference DiffusionGemma (K-D9).
@Suite("Profils de reference DiffusionGemma")
struct DiffusionReferenceProfileTests {

    @Test("6 profils a4bdiff/{16,8,4}bit-{fast,lean}, identifiants uniques")
    func testMatrix() {
        let all = DiffusionReferenceProfile.all
        #expect(all.count == 6)
        #expect(Set(all.map(\.qualifiedID)) == [
            "a4bdiff/16bit-fast", "a4bdiff/16bit-lean", "a4bdiff/8bit-fast",
            "a4bdiff/8bit-lean", "a4bdiff/4bit-fast", "a4bdiff/4bit-lean",
        ])
    }

    @Test("quantification par bits, vision dechargee en lean seulement")
    func testFields() throws {
        let bf16 = try #require(DiffusionReferenceProfile.named("a4bdiff/16bit-fast"))
        #expect(bf16.quantization == .none)
        #expect(!bf16.unloadVisionAfterFirstCanvas && !bf16.clearCacheBetweenCanvases)
        let q4 = try #require(DiffusionReferenceProfile.named("4bit-lean"))
        #expect(q4.quantization == .mixed(.default), "4 bits uniforme : 2,9x plus de passes")
        #expect(q4.unloadVisionAfterFirstCanvas && q4.clearCacheBetweenCanvases)
        #expect(q4.memoryLimitMB != nil)
        let lean16 = try #require(DiffusionReferenceProfile.named("16bit-lean"))
        #expect(lean16.memoryLimitMB == nil && lean16.cacheLimitMB == 4096, "16bit-lean garde les caches Mac")
        #expect(!q4.textOnlyVariant().includeVision)
    }

    @Test("recommended : 96 Go -> 8bit-fast, 64 -> 8bit-fast, 40 -> 4bit-fast, 32 et 16 -> 4bit-lean")
    func testRecommended() {
        #expect(DiffusionReferenceProfile.recommended(availableMB: 96 * 1024)?.qualifiedID == "a4bdiff/8bit-fast")
        #expect(DiffusionReferenceProfile.recommended(availableMB: 64 * 1024)?.qualifiedID == "a4bdiff/8bit-fast")
        #expect(DiffusionReferenceProfile.recommended(availableMB: 40 * 1024)?.qualifiedID == "a4bdiff/4bit-fast")
        #expect(DiffusionReferenceProfile.recommended(availableMB: 32 * 1024)?.qualifiedID == "a4bdiff/4bit-lean")
        #expect(DiffusionReferenceProfile.recommended(availableMB: 16 * 1024)?.qualifiedID == "a4bdiff/4bit-lean")
    }
}
