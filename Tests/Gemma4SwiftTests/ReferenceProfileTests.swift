import Testing
import Foundation
import MLXLMCommon
@testable import Gemma4Swift

/// Standard des profils de reference (K-20) : matrice, identifiants, regles fast/lean.
@Suite("Profils de reference")
struct ReferenceProfileTests {

    @Test("matrice : 5 familles x 4/8/16 bits x fast/lean, identifiants uniques")
    func testMatrix() {
        let all = Gemma4ReferenceProfile.all
        #expect(all.count == 30)
        #expect(Set(all.map(\.qualifiedID)).count == all.count)
        #expect(!all.contains { $0.model.isDiffusion })
        for profile in all {
            #expect(profile.id == "\(profile.bits.rawValue)bit-\(profile.kind.rawValue)")
            #expect(profile.model.family == profile.family)
        }
    }

    @Test("named : identifiant court dans une famille, ou identifiant complet")
    func testNamed() throws {
        let e2b = try #require(Gemma4ReferenceProfile.named("4bit-fast", family: .e2b))
        #expect(e2b.model == .e2b4bit)
        let b31b = try #require(Gemma4ReferenceProfile.named("b31b/8bit-lean"))
        #expect(b31b.model == .b31b8bit)
        #expect(Gemma4ReferenceProfile.named("3bit-fast", family: .e2b) == nil)
    }

    @Test("lean : KV 8 bits seulement pour 26B-A4B et 31B ; fast : jamais")
    func testKVBits() {
        for profile in Gemma4ReferenceProfile.all {
            let multiKVHeads = profile.family == .a4b || profile.family == .b31b
            #expect(profile.kvBits == (profile.kind == .lean && multiKVHeads ? 8 : nil), "\(profile.qualifiedID)")
        }
    }

    @Test("16bit-lean garde les caches Mac ; les autres lean ont une limite memoire")
    func testMemoryPolicy() {
        for profile in Gemma4ReferenceProfile.all where profile.kind == .lean {
            if profile.bits == .sixteen {
                #expect(profile.memoryLimitMB == nil && profile.cacheLimitMB == 4096, "\(profile.qualifiedID)")
            } else {
                #expect(profile.memoryLimitMB != nil, "\(profile.qualifiedID)")
                #expect((profile.cacheLimitMB ?? 0) <= 1024, "\(profile.qualifiedID)")
            }
            #expect(profile.clearCacheAfterAnswer)
            #expect(profile.prefillStepSize == (profile.family == .a4b ? 512 : 256), "\(profile.qualifiedID)")
        }
    }

    @Test("apply(to:) pose la tranche de prefill et le KV")
    func testApplyToParameters() throws {
        let profile = try #require(Gemma4ReferenceProfile.named("a4b/4bit-lean"))
        var params = GenerateParameters(maxTokens: 10)
        profile.apply(to: &params)
        #expect(params.prefillStepSize == 512)
        #expect(params.kvBits == 8)
    }

    @Test("recommended : le fast le plus large qui tient, sinon le lean le plus petit")
    func testRecommended() throws {
        let big = try #require(Gemma4ReferenceProfile.recommended(for: .e4b, availableMB: 96 * 1024))
        #expect(big.kind == .fast && big.bits == .sixteen)
        let small = try #require(Gemma4ReferenceProfile.recommended(for: .b31b, availableMB: 8 * 1024))
        #expect(small.kind == .lean && small.bits == .four)
    }

    @Test("textOnlyVariant garde tout sauf les tours")
    func testTextOnlyVariant() throws {
        let profile = try #require(Gemma4ReferenceProfile.named("e2b/8bit-fast"))
        let text = profile.textOnlyVariant()
        #expect(!text.multimodal)
        #expect(text.model == profile.model && text.prefillStepSize == profile.prefillStepSize)
    }

    @Test("lean : sans audio et tours liberees ; fast : tout resident ; variante avec audio")
    func testResidency() throws {
        for profile in Gemma4ReferenceProfile.all {
            #expect(profile.releaseEncodersAfterPrefill == (profile.kind == .lean), "\(profile.qualifiedID)")
            if profile.kind == .lean { #expect(!profile.audio, "\(profile.qualifiedID)") }
        }
        let fast = try #require(Gemma4ReferenceProfile.named("e2b/4bit-fast"))
        #expect(fast.audio)
        let lean = try #require(Gemma4ReferenceProfile.named("e2b/4bit-lean"))
        #expect(lean.withAudioVariant().audio && lean.withAudioVariant().releaseEncodersAfterPrefill)
        #expect(!lean.textOnlyVariant().audio && !lean.textOnlyVariant().multimodal)
    }
}

