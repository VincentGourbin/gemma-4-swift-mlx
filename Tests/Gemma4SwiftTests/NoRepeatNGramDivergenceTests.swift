import Testing
import Foundation
import MLX
import MLXLMCommon
@testable import Gemma4Swift

/// Porte K-15 sur vrai modele : on genere avec l'ancien chemin (historique CPU), puis on
/// rejoue la meme suite de jetons dans les deux processeurs ; ils doivent interdire les
/// memes jetons a chaque pas. Le prompt brut fait emettre des jetons speciaux (canal,
/// fin de tour) : c'est la qu'un premier A/B contre `includeThinkingInWindow: false`
/// divergeait (autre mode, pas l'ancien chemin).
private let modelPath = ProcessInfo.processInfo.environment["GEMMA4_INTEGRATION_MODEL_PATH"]

@Suite("n-gramme : parite GPU / CPU sur vrai modele", .serialized, .enabled(if: modelPath != nil))
struct NoRepeatNGramDivergenceTests {

    @Test("device et host interdisent les memes jetons sur une vraie generation")
    func testReplay() async throws {
        let container = try await Gemma4Registration.loadContainer(
            from: URL(fileURLWithPath: modelPath!), multimodal: false)
        let (prompt, generated) = try await container.perform { context -> ([Int32], [Int32]) in
            let text = String(repeating: "The quick brown fox jumps over the lazy dog near the river bank. ", count: 40)
            let ids = context.tokenizer.encode(text: text).prefix(512).map { Int32($0) }
            var iterator = try TokenIterator(
                input: LMInput(tokens: MLXArray(ids)), model: context.model, cache: nil,
                processor: NoRepeatNGramLogitProcessor(ngramSize: 5, forceHostHistory: true),
                sampler: ArgMaxSampler(), prefillStepSize: 512, maxTokens: 256)
            var out: [Int32] = []
            while out.count < 256, let t = iterator.next() { out.append(Int32(t)) }
            return (Array(ids), out)
        }
        var device = NoRepeatNGramLogitProcessor(ngramSize: 5)
        var host = NoRepeatNGramLogitProcessor(ngramSize: 5, forceHostHistory: true)
        device.prompt(MLXArray(prompt))
        host.prompt(MLXArray(prompt))
        let vocab = 262_144
        let zeros = MLXArray.zeros([1, vocab])
        for (step, token) in generated.enumerated() {
            let a = device.process(logits: zeros) .== -Float.infinity
            let b = host.process(logits: zeros) .== -Float.infinity
            let differs = any(a .!= b).item(Bool.self)
            if differs {
                let onlyDevice = argWhereIndices(a .&& logicalNot(b))
                let onlyHost = argWhereIndices(b .&& logicalNot(a))
                let context = generated[max(0, step - 6) ..< step].map(Int.init)
                Issue.record("pas \(step) : prefixe \(context), interdits device seuls \(onlyDevice.prefix(10)), host seuls \(onlyHost.prefix(10)), jeton genere \(token)")
                return
            }
            device.didSample(token: MLXArray([token]))
            host.didSample(token: MLXArray([token]))
        }
        print("DIAG aucune divergence sur \(generated.count) pas ; jetons speciaux generes : \(generated.filter { $0 < 256 || $0 > 255_000 })")
    }

    private func argWhereIndices(_ mask: MLXArray) -> [Int] {
        mask.reshaped(-1).asArray(Bool.self).enumerated().compactMap { $0.element ? $0.offset : nil }
    }
}
