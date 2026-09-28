import Testing
import Foundation
import MLX
import MLXNN
@testable import Gemma4Swift

/// Garde-fou D-02 (audit diffusion) : encodeur et decodeur partagent leurs poids en
/// bf16 (le sanitizer insere le meme tableau des deux cotes) ; la quantification doit
/// garder une seule copie, sinon un 4 bits pese le double.
@Suite("Poids partages encodeur / decodeur quantifies une fois")
struct DiffusionQuantizationTieTests {

    /// Modele minuscule dont le decodeur reprend les poids de l'encodeur, comme apres
    /// le sanitizer (hors self_conditioning et layer_scalar).
    private func tiedModel() throws -> DiffusionGemmaForBlockDiffusion {
        let model = try TinyDiffusion.model()
        // Comme le sanitizer : le decodeur reprend les tableaux bf16 de l'encodeur
        // (hors self_conditioning et layer_scalar).
        let encoder = Dictionary(model.encoder.languageModel.parameters().flattened(), uniquingKeysWith: { a, _ in a })
        let tied = model.decoder.parameters().flattened().compactMap { key, value -> (String, MLXArray)? in
            guard !key.hasPrefix("self_conditioning."), !key.hasSuffix("layer_scalar"),
                  let source = encoder[key], source.shape == value.shape else { return nil }
            return (key, source)
        }
        model.decoder.update(parameters: ModuleParameters.unflattened(tied))
        return model
    }

    /// Adresse du tampon sous-jacent (evalue, sans copie) : deux handles MLX distincts
    /// peuvent partager le meme tampon ; c'est ce partage qu'on verifie.
    private func buffer(_ a: MLXArray) -> UInt {
        eval(a)
        return a.asData(access: .noCopy).data.withUnsafeBytes { UInt(bitPattern: $0.baseAddress) }
    }

    private func qProjWeights(_ model: DiffusionGemmaForBlockDiffusion) -> (MLXArray, MLXArray) {
        let enc = Dictionary(model.encoder.languageModel.parameters().flattened(), uniquingKeysWith: { a, _ in a })
        let dec = Dictionary(model.decoder.parameters().flattened(), uniquingKeysWith: { a, _ in a })
        let key = "layers.0.self_attn.q_proj.weight"
        return (enc[key]!, dec[key]!)
    }

    @Test("apply(bits: 8) : q_proj encodeur et decodeur sont le meme tableau")
    func testUniformQuantizationSharesArrays() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try tiedModel()
            let (encBefore, decBefore) = qProjWeights(model)
            #expect(buffer(encBefore) == buffer(decBefore), "prerequis : poids lies avant quantification")
            DiffusionOnTheFlyQuantization.apply(to: model, bits: 8, groupSize: 32)
            let (enc, dec) = qProjWeights(model)
            #expect(enc.dtype == .uint32, "q_proj quantifie")
            #expect(buffer(enc) == buffer(dec), "une seule copie quantifiee")
        }
    }

    @Test("applyMixedPrecision : meme partage")
    func testMixedPrecisionSharesArrays() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try tiedModel()
            DiffusionOnTheFlyQuantization.applyMixedPrecision(
                to: model, config: .init(highPrecisionLayers: [0], highPrecisionBits: 8, lowPrecisionBits: 4, groupSize: 32))
            let (enc, dec) = qProjWeights(model)
            #expect(enc.dtype == .uint32)
            #expect(buffer(enc) == buffer(dec))
        }
    }
}
