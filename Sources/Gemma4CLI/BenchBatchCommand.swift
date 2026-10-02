// gemma4-cli bench-batch : plafond theorique du decodage par lot (K-41, etape 1).
//
// Avant de construire un lot multi-clients dans le serveur, mesurer ce qu'il peut gagner :
// le temps d'un pas de decodage pour B sequences a la fois (meme prompt replique, caches
// au batch B) contre B = 1. Debit agrege = B / temps du pas. Si 8 sequences coutent
// presque 8 fois un pas seul, le lot ne rapporte rien et la fiche est abandonnee.

import ArgumentParser
import Foundation
import Gemma4Swift
import MLX
import MLXLMCommon

struct BenchBatch: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "bench-batch",
        abstract: "Plafond du decodage par lot : temps d'un pas pour B sequences (K-41)")

    @Option(name: .long, help: "Dossier du modele")
    var modelPath: String

    @Option(name: .long, help: "Tailles de lot, separees par des virgules")
    var batchSizes: String = "1,2,4,8"

    @Option(name: .long, help: "Longueur du prompt (jetons)")
    var promptTokens: Int = 256

    @Option(name: .long, help: "Pas de decodage mesures")
    var steps: Int = 64

    @Option(name: .long, help: "Fichier JSONL ou ajouter les lignes")
    var out: String?

    func run() async throws {
        let url = URL(fileURLWithPath: modelPath)
        let container = try await Gemma4Registration.loadContainer(from: url, multimodal: false)
        let sizes = batchSizes.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        let promptTokens = promptTokens
        let steps = steps
        let measured: [[String: Double]] = try await container.perform { context in
            let ids = try Bench.promptIds(size: promptTokens, filler: Bench.defaultFiller, tokenizer: context.tokenizer)
            var results: [[String: Double]] = []
            for b in sizes {
                let prompt = MLXArray(ids.map { Int32($0) }).reshaped(1, -1)
                let batch = tiled(prompt, repetitions: [b, 1])
                let cache = context.model.newCache(parameters: nil)
                // Prefill du lot entier.
                let prefillStart = Date()
                var logits = context.model(batch, cache: cache)
                var next = argMax(logits[0..., -1, 0...], axis: -1).reshaped(b, 1)
                eval(next)
                let prefill = Date().timeIntervalSince(prefillStart)
                // Echauffement de 4 pas, puis `steps` pas chronometres.
                var times: [Double] = []
                for step in 0 ..< steps + 4 {
                    let t0 = Date()
                    logits = context.model(next, cache: cache)
                    next = argMax(logits[0..., -1, 0...], axis: -1).reshaped(b, 1)
                    eval(next)
                    if step >= 4 { times.append(Date().timeIntervalSince(t0)) }
                }
                times.sort()
                let median = times[times.count / 2]
                results.append([
                    "batch": Double(b), "step_ms_median": median * 1000,
                    "aggregate_tok_s": Double(b) / median, "prefill_s": prefill,
                    "peak_mlx_mb": Double(Memory.peakMemory / 1_048_576),
                ])
                Memory.clearCache()
            }
            return results
        }
        let lines: [[String: Any]] = measured.map { m in
            [
                "kind": "batch_decode", "batch": Int(m["batch"] ?? 0), "prompt_tokens": promptTokens, "steps": steps,
                "step_ms_median": Bench.round(m["step_ms_median"] ?? 0),
                "aggregate_tok_s": Bench.round(m["aggregate_tok_s"] ?? 0),
                "prefill_s": Bench.round(m["prefill_s"] ?? 0), "peak_mlx_mb": Int(m["peak_mlx_mb"] ?? 0),
            ]
        }
        let context = BenchContext.collect(modelURL: url, loadSeconds: 0)
        let base = measured.first?["aggregate_tok_s"] ?? 1
        for (var line, m) in zip(lines, measured) {
            line.merge(context.fields) { current, _ in current }
            try Bench.writeLine(line, to: out)
            let agg = m["aggregate_tok_s"] ?? 0
            print(String(format: "B=%d : pas %.2f ms, %.1f tok/s agreges (x%.2f)",
                         Int(m["batch"] ?? 0), m["step_ms_median"] ?? 0, agg, agg / base))
        }
    }
}
