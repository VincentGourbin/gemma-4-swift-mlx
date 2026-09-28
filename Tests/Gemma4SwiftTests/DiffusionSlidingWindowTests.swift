import Testing
import Foundation
import MLX
@testable import Gemma4Swift

/// Garde-fou D-05 (audit diffusion) : les couches glissantes de l'encodeur n'attendent
/// que sur `sliding_window` positions et leur cache ne garde que les
/// `sliding_window - 1` dernieres, comme la reference Python. Reference naive :
/// les memes couches, en une passe, avec des masques construits ici.
/// Sur CPU (voir DiffusionPipelineTests : le modele minuscule plante sur GPU).
@Suite("Fenetre glissante de la diffusion", .serialized)
struct DiffusionSlidingWindowTests {

    static let window = 8

    /// Encodeur minuscule a fenetre 8 : couche 0 glissante, couche 1 pleine.
    private func model() throws -> DiffusionGemmaForBlockDiffusion {
        let json = TinyDiffusion.configJSON.replacingOccurrences(
            of: "\"sliding_window\": 8", with: "\"sliding_window\": \(Self.window)")
        let config = try JSONDecoder().decode(DiffusionGemmaConfig.self, from: Data(json.utf8))
        MLXRandom.seed(21)
        return DiffusionGemmaForBlockDiffusion(config)
    }

    private func ids(_ n: Int) -> MLXArray {
        MLXArray((0 ..< n).map { Int32(($0 * 7 + 3) % 50) }).reshaped(1, n)
    }

    /// Reference naive : une passe, masque causal (pleine) ou causal + fenetre (glissante).
    private func reference(_ model: DiffusionGemmaForBlockDiffusion, _ tokens: MLXArray) -> MLXArray {
        let lm = model.encoder.languageModel
        let n = tokens.dim(1)
        var h = lm.embedTokens(tokens) * MLXArray(lm.embedScale)
        let q = MLXArray(0 ..< Int32(n))[0..., .newAxis]
        let k = MLXArray(0 ..< Int32(n))[.newAxis, 0...]
        let causal = k .<= q
        let sliding = causal .&& ((q - k) .< MLXArray(Int32(Self.window)))
        for (i, layer) in lm.layers.enumerated() {
            let isGlobal = lm.config.resolvedLayerTypes[i] == "full_attention"
            h = layer(h, mask: .array(isGlobal ? causal : sliding), positionOffset: 0, priorKV: nil).output
        }
        return lm.norm(h)
    }

    private func relativeError(_ a: MLXArray, _ b: MLXArray) -> Float {
        (sqrt(sum(square(a - b))) / sqrt(sum(square(a)))).item(Float.self)
    }

    @Test("20 jetons (> fenetre) : une passe = reference ; en deux appels (12 + 8) = reference")
    func testBeyondWindow() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try model()
            let lm = model.encoder.languageModel
            let tokens = ids(20)
            let ref = reference(model, tokens)

            let single = lm(inputs: tokens)
            #expect(relativeError(ref, single.lastHiddenState) < 1e-4, "une passe")

            let first = lm(inputs: tokens[0..., 0 ..< 12])
            let second = lm(inputs: tokens[0..., 12...], priorCache: first.kvCache)
            #expect(relativeError(ref[0..., 12...], second.lastHiddenState) < 1e-4, "deux appels")

            // Cache : couche glissante tronquee a fenetre - 1, couche pleine complete,
            // offset = positions totales.
            #expect(second.kvCache.entries[0]!.keys.dim(2) == Self.window - 1)
            #expect(second.kvCache.entries[1]!.keys.dim(2) == 20)
            #expect(second.kvCache.seqLength == 20)
        }
    }

    @Test("6 jetons (< fenetre) : identique a la reference")
    func testWithinWindow() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try model()
            let tokens = ids(6)
            let out = model.encoder.languageModel(inputs: tokens)
            #expect(relativeError(reference(model, tokens), out.lastHiddenState) < 1e-4)
        }
    }

    @Test("decodeur : cache tronque sans masque = masque explicite facon Python")
    func testDecoderMaskEquivalence() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try model()
            let cache = model.encoder.languageModel(inputs: ids(20)).kvCache
            let canvas = MLXArray([Int32(3), 4, 5, 6]).reshaped(1, 4)
            let implicit = model.denoiseStep(canvasIds: canvas, encoderCache: cache)
            let explicit = model.denoiseStep(
                canvasIds: canvas, encoderCache: cache,
                decoderAttentionMask: MLXArray.ones([1, 20 + 4], type: Int32.self))
            #expect(relativeError(explicit, implicit) < 1e-4)
        }
    }
}
