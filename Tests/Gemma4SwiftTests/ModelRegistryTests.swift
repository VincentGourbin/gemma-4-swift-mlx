import Testing
import Foundation
@testable import Gemma4Swift

@Suite("Model Registry et Capabilities")
struct ModelRegistryTests {

    // MARK: - Capabilities

    @Test("E2B supporte toutes les modalites (any-to-any)")
    func testE2BCapabilities() {
        let model = Gemma4Pipeline.Model.e2b4bit
        #expect(model.supportsImage == true)
        #expect(model.supportsAudio == true)
        #expect(model.supportsVideo == true)
        #expect(model.capabilities == .anyToAny)
    }

    @Test("E4B supporte toutes les modalites")
    func testE4BCapabilities() {
        #expect(Gemma4Pipeline.Model.e4b4bit.supportsAudio == true)
        #expect(Gemma4Pipeline.Model.e4bBf16.supportsImage == true)
    }

    @Test("26B-A4B ne supporte pas l'audio")
    func testA4BCapabilities() {
        let model = Gemma4Pipeline.Model.a4b4bit
        #expect(model.supportsImage == true)
        #expect(model.supportsAudio == false)
        #expect(model.supportsVideo == true)
        #expect(model.capabilities == .imageTextToText)
    }

    @Test("31B ne supporte pas l'audio")
    func testB31BCapabilities() {
        let model = Gemma4Pipeline.Model.b31b4bit
        #expect(model.supportsAudio == false)
        #expect(model.supportsImage == true)
        #expect(model.supportsVideo == true)
    }

    // MARK: - Metadata

    @Test("Parametres effectifs MoE vs dense")
    func testEffectiveParameters() {
        #expect(Gemma4Pipeline.Model.a4b4bit.effectiveParameters == "3.8B")
        #expect(Gemma4Pipeline.Model.a4b4bit.parameterCount == "25.8B")
        #expect(Gemma4Pipeline.Model.b31b4bit.effectiveParameters == "31.3B")
        #expect(Gemma4Pipeline.Model.b31b4bit.parameterCount == "31.3B")
    }

    @Test("isMoE uniquement pour 26B-A4B")
    func testIsMoE() {
        let moeModels = Gemma4Pipeline.Model.allCases.filter { $0.isMoE }
        #expect(moeModels.count == 4) // a4b 4-bit, 6-bit, 8-bit, bf16
        #expect(moeModels.allSatisfy { $0.family == .a4b })
    }

    @Test("Tous les modeles sont IT")
    func testIsInstructionTuned() {
        // Tous les modeles MLX community sont -it
        for model in Gemma4Pipeline.Model.allCases {
            #expect(model.isInstructionTuned == true)
        }
    }

    @Test("Quantization strings")
    func testQuantization() {
        #expect(Gemma4Pipeline.Model.e2b4bit.quantization == "4-bit")
        #expect(Gemma4Pipeline.Model.e2b8bit.quantization == "8-bit")
        #expect(Gemma4Pipeline.Model.e2b6bit.quantization == "6-bit")
        #expect(Gemma4Pipeline.Model.e2bBf16.quantization == "bf16")
    }

    @Test("Family classification")
    func testFamily() {
        #expect(Gemma4Pipeline.Model.e2b4bit.family == .e2b)
        #expect(Gemma4Pipeline.Model.e4bBf16.family == .e4b)
        #expect(Gemma4Pipeline.Model.b31b8bit.family == .b31b)
        #expect(Gemma4Pipeline.Model.a4b6bit.family == .a4b)
    }

    @Test("RAM recommandee croissante avec la taille")
    func testRAMRecommendations() {
        // Au sein d'une meme famille, BF16 > 8-bit > 6-bit > 4-bit
        #expect(Gemma4Pipeline.Model.e2b4bit.recommendedRAMGB < Gemma4Pipeline.Model.e2bBf16.recommendedRAMGB)
        // Entre familles, le plus gros demande plus
        #expect(Gemma4Pipeline.Model.e2b4bit.estimatedSizeGB < Gemma4Pipeline.Model.e4b4bit.estimatedSizeGB)
        #expect(Gemma4Pipeline.Model.e4b4bit.estimatedSizeGB < Gemma4Pipeline.Model.a4b4bit.estimatedSizeGB)
    }

    @Test("16 modeles au total (4 familles x 4 quantisations)")
    func testModelCount() {
        #expect(Gemma4Pipeline.Model.allCases.count == 16)
    }

    @Test("Raw values sont des IDs HuggingFace valides")
    func testRawValues() {
        for model in Gemma4Pipeline.Model.allCases {
            #expect(model.rawValue.contains("/"))
            #expect(model.rawValue.hasPrefix("mlx-community/"))
        }
    }

    @Test("Modeles recommandes pour RAM")
    func testRecommendedModels() {
        // 8 Go: seulement les plus petits 4-bit
        let for8GB = Gemma4Pipeline.Model.recommended(forRAMGB: 8)
        #expect(!for8GB.isEmpty)
        #expect(for8GB.allSatisfy { $0.recommendedRAMGB <= 8 })

        // 96 Go: beaucoup de modeles mais pas les BF16 les plus gros
        let for96GB = Gemma4Pipeline.Model.recommended(forRAMGB: 96)
        #expect(for96GB.count > for8GB.count)
    }
}
