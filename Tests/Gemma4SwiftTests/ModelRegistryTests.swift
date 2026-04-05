import Testing
import Foundation
@testable import Gemma4Swift

@Suite("Model Registry et Capabilities")
struct ModelRegistryTests {

    // MARK: - Capabilities

    @Test("E2B supporte toutes les modalites (any-to-any)")
    func testE2BCapabilities() {
        let model = Gemma4Pipeline.Model.e2bIT
        #expect(model.supportsImage == true)
        #expect(model.supportsAudio == true)
        #expect(model.supportsVideo == true)
        #expect(model.capabilities == .anyToAny)
    }

    @Test("E4B supporte toutes les modalites")
    func testE4BCapabilities() {
        #expect(Gemma4Pipeline.Model.e4bIT.supportsAudio == true)
        #expect(Gemma4Pipeline.Model.e4b.supportsImage == true)
    }

    @Test("26B-A4B ne supporte pas l'audio")
    func testA4BCapabilities() {
        let model = Gemma4Pipeline.Model.a4bIT
        #expect(model.supportsImage == true)
        #expect(model.supportsAudio == false)
        #expect(model.supportsVideo == true)
        #expect(model.capabilities == .imageTextToText)
    }

    @Test("31B ne supporte pas l'audio")
    func testB31BCapabilities() {
        let model = Gemma4Pipeline.Model.b31bIT
        #expect(model.supportsAudio == false)
        #expect(model.supportsImage == true)
        #expect(model.supportsVideo == true)
    }

    @Test("MLX community 4-bit sont any-to-any")
    func testMLXCommunityCapabilities() {
        #expect(Gemma4Pipeline.Model.e2b4bit.supportsAudio == true)
        #expect(Gemma4Pipeline.Model.e4b4bit.supportsAudio == true)
    }

    // MARK: - Metadata

    @Test("Parametres effectifs MoE vs dense")
    func testEffectiveParameters() {
        #expect(Gemma4Pipeline.Model.a4bIT.effectiveParameters == "3.8B")
        #expect(Gemma4Pipeline.Model.a4bIT.parameterCount == "25.8B")
        #expect(Gemma4Pipeline.Model.b31bIT.effectiveParameters == "31.3B")
        #expect(Gemma4Pipeline.Model.b31bIT.parameterCount == "31.3B")
    }

    @Test("isMoE uniquement pour 26B-A4B")
    func testIsMoE() {
        let moeModels = Gemma4Pipeline.Model.allCases.filter { $0.isMoE }
        #expect(moeModels.count == 2)
        #expect(moeModels.contains(.a4bIT))
        #expect(moeModels.contains(.a4b))
    }

    @Test("isBF16 pour tous les Google, pas pour MLX community")
    func testIsBF16() {
        #expect(Gemma4Pipeline.Model.e2b4bit.isBF16 == false)
        #expect(Gemma4Pipeline.Model.e4b4bit.isBF16 == false)
        #expect(Gemma4Pipeline.Model.e2bIT.isBF16 == true)
        #expect(Gemma4Pipeline.Model.b31b.isBF16 == true)
    }

    @Test("isInstructionTuned")
    func testIsInstructionTuned() {
        #expect(Gemma4Pipeline.Model.e2bIT.isInstructionTuned == true)
        #expect(Gemma4Pipeline.Model.e2b.isInstructionTuned == false)
        #expect(Gemma4Pipeline.Model.e2b4bit.isInstructionTuned == true)
    }

    @Test("RAM recommandee croissante avec la taille")
    func testRAMRecommendations() {
        let models: [Gemma4Pipeline.Model] = [.e2b4bit, .e4b4bit, .e2bIT, .e4bIT, .a4bIT, .b31bIT]
        for i in 0 ..< models.count - 1 {
            #expect(models[i].recommendedRAMGB <= models[i + 1].recommendedRAMGB)
        }
    }

    @Test("Taille estimee croissante avec les parametres")
    func testEstimatedSizes() {
        #expect(Gemma4Pipeline.Model.e2b4bit.estimatedSizeGB < Gemma4Pipeline.Model.e2bIT.estimatedSizeGB)
        #expect(Gemma4Pipeline.Model.e2bIT.estimatedSizeGB < Gemma4Pipeline.Model.e4bIT.estimatedSizeGB)
        #expect(Gemma4Pipeline.Model.e4bIT.estimatedSizeGB < Gemma4Pipeline.Model.a4bIT.estimatedSizeGB)
        #expect(Gemma4Pipeline.Model.a4bIT.estimatedSizeGB < Gemma4Pipeline.Model.b31bIT.estimatedSizeGB)
    }

    @Test("Raw values sont des IDs HuggingFace valides")
    func testRawValues() {
        for model in Gemma4Pipeline.Model.allCases {
            #expect(model.rawValue.contains("/"))
            let parts = model.rawValue.split(separator: "/")
            #expect(parts.count == 2)
        }
    }
}
