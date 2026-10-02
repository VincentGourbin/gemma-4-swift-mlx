import Testing
import Foundation
import MLX
import MLXLMCommon
@testable import Gemma4Swift

/// K-41 sur un vrai modele (E2B) : un lot de copies du meme prompt doit generer, ligne par
/// ligne, ce que le prompt genere seul. Active par GEMMA4_INTEGRATION_MODEL_PATH.
@Suite("Generation par lot (integration)", .serialized,
       .enabled(if: ProcessInfo.processInfo.environment["GEMMA4_INTEGRATION_MODEL_PATH"] != nil))
struct BatchGenerationIntegrationTests {

    @Test("lignes identiques d'un lot de 2 : ecart ligne 0 / ligne 1 a chaque etape")
    func testRowSymmetry() async throws {
        let path = try #require(ProcessInfo.processInfo.environment["GEMMA4_INTEGRATION_MODEL_PATH"])
        let container = try await Gemma4Registration.loadContainer(from: URL(fileURLWithPath: path), multimodal: false)
        let report: String = try await container.perform { context in
            let ids = try Gemma4ChatEngine.promptIds(
                messages: [.init(role: .user, content: "Ecris un paragraphe descriptif de quatre phrases sur la mer.")],
                tools: [], enableThinking: false, tokenizer: context.tokenizer)
            let lm = try #require(Gemma4BatchGeneration.languageModel(of: context.model))
            let prompt = tiled(MLXArray(ids.map(Int32.init)).reshaped(1, -1), repetitions: [2, 1])
            func rowGap(_ logits: MLXArray) -> Float {
                let a = logits[0, -1].asType(.float32), b = logits[1, -1].asType(.float32)
                return (sqrt(sum(square(a - b))) / sqrt(sum(square(a)))).item(Float.self)
            }
            var lines: [String] = []
            lines.append("sans cache : \(rowGap(lm(inputs: prompt)))")
            for (name, cache) in [("makeCache()", lm.makeCache()), ("slidingCapacity", lm.makeCache(slidingCapacity: ids.count + 32))] {
                let c: [KVCache?] = cache.map { $0 }
                let prefill = lm(inputs: prompt, cache: c)
                let next = argMax(prefill[0..., -1], axis: -1).reshaped(2, 1)
                let step = lm(inputs: next, cache: c)
                lines.append("\(name) : prefill \(rowGap(prefill)), pas 1 \(rowGap(step))")
                lines.append("  types : \(cache.map { String(describing: type(of: $0)) }.joined(separator: ","))")
            }
            return lines.joined(separator: "\n")
        }
        print("K41-SYM\n" + report)
    }

    @Test("copies du meme prompt : memes jetons que seul (B = 2, 6) ; ecart des logits au 1er pas")
    func testIdenticalPrompts() async throws {
        let path = try #require(ProcessInfo.processInfo.environment["GEMMA4_INTEGRATION_MODEL_PATH"])
        let container = try await Gemma4Registration.loadContainer(from: URL(fileURLWithPath: path), multimodal: false)
        let report: String = try await container.perform { context in
            let ids = try Gemma4ChatEngine.promptIds(
                messages: [.init(role: .user, content: "Ecris un paragraphe descriptif de quatre phrases sur la mer.")],
                tools: [], enableThinking: false, tokenizer: context.tokenizer)
            func tokens(batch: Int) throws -> [[Int]] {
                var out = Array(repeating: [Int](), count: batch)
                _ = try Gemma4BatchGeneration.run(
                    model: context.model, requests: Array(repeating: .init(ids: ids, maxTokens: 24), count: batch)
                ) { row, token in out[row].append(token); return true }
                return out
            }
            let single = try tokens(batch: 1)[0]
            var lines: [String] = ["seul : \(single)"]
            for b in [2, 6] {
                let rows = try tokens(batch: b)
                // Lignes identiques entre elles (RoPE replie, `RoPEBatchTests`). Par rapport
                // a la generation seule, l'ecart d'arrondi du lot fait diverger plus loin.
                #expect(Set(rows).count == 1, "B=\(b) : lignes differentes")
                let firstDiff = rows.map { row in Array(zip(row, single)).firstIndex { $0.0 != $0.1 } ?? -1 }
                lines.append("B=\(b) premiere divergence par ligne : \(firstDiff)")
            }
            // Ecart des logits au premier pas, B = 1 contre B = 2 (meme prompt).
            let lm = try #require(Gemma4BatchGeneration.languageModel(of: context.model))
            func firstLogits(_ b: Int) -> MLXArray {
                let cache: [KVCache?] = lm.makeCache(slidingCapacity: ids.count + 32).map { $0 }
                let prompt = tiled(MLXArray(ids.map(Int32.init)).reshaped(1, -1), repetitions: [b, 1])
                return lm(inputs: prompt, cache: cache)[0, -1].asType(.float32)
            }
            let one = firstLogits(1), two = firstLogits(2)
            let rel = (sqrt(sum(square(one - two))) / sqrt(sum(square(one)))).item(Float.self)
            lines.append("logits 1er pas, B=1 vs B=2 ligne 0 : ecart relatif \(rel), argmax \(argMax(one).item(Int.self)) / \(argMax(two).item(Int.self))")
            return lines.joined(separator: "\n")
        }
        print("K41-INTEGRATION\n" + report)
        #expect(!report.contains(": [0"))
    }
}
