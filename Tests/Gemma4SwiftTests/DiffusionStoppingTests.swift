import Testing
import Foundation
import MLX
@testable import Gemma4Swift

/// Garde-fou D-04 (audit diffusion) : « stable » compare l'argmax courant aux argmax
/// PRECEDENTS, comme la reference Python. L'ancien code ajoutait d'abord l'argmax
/// courant a l'historique : avec le seuil 1, « stable » etait toujours vrai.
@Suite("Critere d'arret de la diffusion")
struct DiffusionStoppingTests {

    /// Logits tres piques : entropie quasi nulle, donc toujours « confiant ».
    private func confidentLogits(_ argmax: [Int32]) -> MLXArray {
        let vocab = 8
        var values = [Float](repeating: -30, count: argmax.count * vocab)
        for (i, a) in argmax.enumerated() { values[i * vocab + Int(a)] = 30 }
        return MLXArray(values).reshaped(1, argmax.count, vocab)
    }

    private func stop(_ s: StableConfidentStopping, _ argmax: [Int32]) -> Bool {
        s.shouldStop(argmaxCanvas: MLXArray(argmax).reshaped(1, argmax.count),
                     logits: confidentLogits(argmax)).all().item(Bool.self)
    }

    @Test("seuil 1 : jamais stable au premier pas, stable si deux argmax consecutifs egaux")
    func testThresholdOne() {
        let s = StableConfidentStopping(stabilityThreshold: 1, confidenceThreshold: 0.5)
        #expect(!stop(s, [1, 2, 3]), "premier pas : pas d'historique, donc pas stable")
        #expect(!stop(s, [1, 2, 4]), "argmax change : pas stable")
        #expect(stop(s, [1, 2, 4]), "meme argmax que le pas precedent et confiant : arret")
    }

    @Test("seuil 2 : il faut deux pas precedents identiques")
    func testThresholdTwo() {
        let s = StableConfidentStopping(stabilityThreshold: 2, confidenceThreshold: 0.5)
        #expect(!stop(s, [5, 6]))
        #expect(!stop(s, [5, 6]), "un seul pas precedent")
        #expect(stop(s, [5, 6]), "deux pas precedents identiques")
    }

    @Test("reset efface l'historique")
    func testReset() {
        let s = StableConfidentStopping(stabilityThreshold: 1, confidenceThreshold: 0.5)
        _ = stop(s, [1])
        s.reset()
        #expect(!stop(s, [1]))
    }

    @Test("entropie fournie : meme decision et meme canvas que le recalcul sur les logits (D-12)")
    func testSharedEntropyEquivalence() {
        MLXRandom.seed(11)
        let logits = MLXRandom.normal([1, 6, 16]) * 3
        let argmax = argMax(logits, axis: -1).asType(.int32)
        let a = StableConfidentStopping(stabilityThreshold: 1, confidenceThreshold: 2)
        let b = StableConfidentStopping(stabilityThreshold: 1, confidenceThreshold: 2)
        let sampler = EntropyBoundSampler(entropyBound: 0.5, vocabSize: 16, canvasLength: 6)
        let entropy = sampler.entropy(of: logits)
        for _ in 0 ..< 2 {
            let viaLogits = a.shouldStop(argmaxCanvas: argmax, logits: logits)
            let viaEntropy = b.shouldStop(argmaxCanvas: argmax, entropy: entropy)
            #expect(viaLogits.asArray(Bool.self) == viaEntropy.asArray(Bool.self))
        }
        let current = MLXArray.zeros([1, 6], type: Int32.self)
        let oldCanvas = sampler.accept(currentCanvas: current, denoiserCanvas: argmax, logits: logits)
        let newCanvas = sampler.accept(currentCanvas: current, denoiserCanvas: argmax, entropy: entropy)
        #expect(oldCanvas.asArray(Int32.self) == newCanvas.asArray(Int32.self))
    }
}
