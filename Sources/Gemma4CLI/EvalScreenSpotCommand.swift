// eval-screenspot : qualite GUI grounding (ScreenSpot-100) d'un profil de diffusion.
//
// Meme prompt, meme extraction du clic et meme score que
// docs/examples/ui-grounding-bench/run_ss.py (reference bf16 : 79/100), mais le modele
// est charge UNE fois avec un profil (`--reference`, `--quant-variant`) au lieu d'un
// processus par cas. Echantillon : Scripts/quality/screenspot-sample.py.

import ArgumentParser
import Foundation
import Gemma4Swift
import MLX

struct EvalScreenSpot: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "eval-screenspot",
        abstract: "ScreenSpot-100 (clic dans la bbox) pour un profil a4bdiff/*"
    )

    @Option(name: .long, help: "Dossier du checkpoint bf16 officiel (google/diffusiongemma-26B-A4B-it)")
    var modelPath: String

    @Option(name: .long, help: "Profil de reference (a4bdiff/16bit-fast, a4bdiff/8bit-fast, …)")
    var reference: String

    @Option(name: .long, help: "Variante de quantification : 4bit-uniform, 4bit-sensitive8, 4bit-mixed")
    var quantVariant: String?

    @Option(name: .long, help: "meta.json produit par Scripts/quality/screenspot-sample.py")
    var meta: String

    @Option(name: .long, help: "Fichier JSONL : une ligne par cas, puis une ligne de synthese")
    var out: String

    @Option(name: .long, help: "Nombre de cas (defaut : tous)")
    var limit: Int?

    @Option(name: .long, help: "Etiquette libre")
    var label: String = ""

    @Option(name: .long, help: "K-D15 : seuil d'entropie d'acceptation (defaut : generation_config.json)")
    var entropyBound: Float?

    @Option(name: .long, help: "K-D15 : seuil de confiance de l'arret (defaut : generation_config.json)")
    var confidenceThreshold: Float?

    struct Case: Decodable {
        let idx: Int
        let image: String
        let width: Int
        let height: Int
        let instruction: String
        let bbox: [Double]
        let data_source: String
        let data_type: String
    }

    func run() async throws {
        guard var profile = DiffusionReferenceProfile.named(reference) else {
            throw ValidationError("profil inconnu : \(reference)")
        }
        switch quantVariant {
        case nil: break
        case "4bit-uniform": profile = profile.withQuantization(.uniform(bits: 4, groupSize: 64))
        case "4bit-sensitive8":
            profile = profile.withQuantization(.mixed(.init(highPrecisionLayers: [], quantizeSensitiveAtHighPrecision: true)))
        case "4bit-mixed": profile = profile.withQuantization(.mixed(.default))
        case "4bit-aggressive": profile = profile.withQuantization(.mixed(.aggressive))
        case let other?: throw ValidationError("variante inconnue : \(other)")
        }
        var cases = try JSONDecoder().decode([Case].self, from: Data(contentsOf: URL(fileURLWithPath: meta)))
        if let limit { cases = Array(cases.prefix(limit)) }

        let url = URL(fileURLWithPath: modelPath)
        let container = try await DiffusionGemmaRegistration.load(from: url, profile: profile)
        var context = BenchContext.collect(modelURL: url, loadSeconds: 0)
        context.fields["profile"] = profile.qualifiedID
        if let quantVariant { context.fields["quant_variant"] = quantVariant }

        var correct = 0
        var totalSteps = 0
        var byGroup: [String: (ok: Int, n: Int)] = [:]
        let start = Date()
        for (i, item) in cases.enumerated() {
            let caseStart = Date()
            let (output, steps) = try await generate(container: container, item: item)
            totalSteps += steps
            let click = Self.parseClick(output)
            let ok = click.map { Self.inside($0, item.bbox) } ?? false
            if ok { correct += 1 }
            for key in [item.data_source, item.data_type, "\(item.data_source)/\(item.data_type)"] {
                byGroup[key, default: (0, 0)].n += 1
                if ok { byGroup[key, default: (0, 0)].ok += 1 }
            }
            var line: [String: Any] = [
                "kind": "screenspot_case", "idx": item.idx, "data_source": item.data_source,
                "data_type": item.data_type, "correct": ok,
                "output": String(output.prefix(120)), "decoder_steps": steps,
                "seconds": Bench.round(Date().timeIntervalSince(caseStart)),
            ]
            if let click { line["click"] = [click.0, click.1] }
            line["profile"] = profile.qualifiedID
            line["label"] = label
            try Bench.writeLine(line, to: out)
            if (i + 1) % 10 == 0 || i == 0 {
                FileHandle.standardError.write(Data("[\(i + 1)/\(cases.count)] \(correct) justes\n".utf8))
            }
        }
        var summary: [String: Any] = [
            "kind": "screenspot_summary", "label": label, "cases": cases.count, "correct": correct,
            "score_pct": Bench.round(100 * Double(correct) / Double(max(1, cases.count))),
            "total_s": Bench.round(Date().timeIntervalSince(start)),
            "by_group": byGroup.mapValues { "\($0.ok)/\($0.n)" },
            "decoder_steps_mean": Bench.round(Double(totalSteps) / Double(max(1, cases.count))),
        ]
        if let entropyBound { summary["entropy_bound"] = entropyBound }
        if let confidenceThreshold { summary["confidence_threshold"] = confidenceThreshold }
        summary.merge(context.fields) { current, _ in current }
        try Bench.writeLine(summary, to: out)
        print("ScreenSpot \(profile.qualifiedID)\(quantVariant.map { " (\($0))" } ?? "") : \(correct)/\(cases.count)")
    }

    /// Prompt de run_ss.py, `<|image|>` en tete, 1 canvas, graine 0 (comme `diffusion`).
    private func generate(container: DiffusionGemmaContainer, item: Case) async throws -> (String, Int) {
        let prompt = """
            Look at this UI screenshot (\(item.width)x\(item.height) pixels).
            Goal: \(item.instruction)

            Where on the screen should I click to accomplish this goal? \
            Respond with the predicted click position in normalized coordinates between 0 and 1, \
            in the exact format:
              CLICK: (x=0.XX, y=0.XX)

            where x is the horizontal position (0=left, 1=right) and y is the vertical position \
            (0=top, 1=bottom). Reply with only the CLICK: line, no extra explanation.
            """
        let pixels = try Gemma4ImageProcessor.processImage(url: URL(fileURLWithPath: item.image))
        eval(pixels)
        let config = container.config
        var ids: [Int] = []
        for id in try container.tokenizer.applyChatTemplate(messages: [["role": "user", "content": "<|image|>\n\(prompt)"]]) {
            if id == config.imageTokenId {
                ids.append(config.boiTokenId)
                ids.append(contentsOf: Array(repeating: config.imageTokenId, count: config.visionSoftTokensPerImage))
                ids.append(config.eoiTokenId)
            } else {
                ids.append(id)
            }
        }
        let promptIds = MLXArray(ids.map { Int32($0) }).reshaped(1, -1)
        nonisolated(unsafe) let pixelsCapture = pixels
        let pipeline = container.makePipeline()
        if entropyBound != nil || confidenceThreshold != nil {
            await pipeline.configureStepping(entropyBound: entropyBound, confidenceThreshold: confidenceThreshold)
        }
        let result = await pipeline.generate(
            promptIds: promptIds, pixelValues: pixelsCapture, maxBlocks: 1, seed: 0)
        if case .invalidInput(let message) = result.stopReason {
            throw ValidationError("cas \(item.idx) : \(message)")
        }
        let generated = result.generatedIds.reshaped(-1).asArray(Int32.self).map(Int.init)
        return (container.tokenizer.decode(tokens: generated, skipSpecialTokens: true), result.totalDecoderSteps)
    }

    /// Meme ordre de formats que `parse_click` de run_ss.py.
    static func parseClick(_ text: String) -> (Double, Double)? {
        let patterns = [
            #"CLICK\s*:?\s*\(?\s*x\s*=\s*([\d.]+)\s*,\s*y\s*=\s*([\d.]+)"#,
            #"x\s*[=:]\s*([\d.]+)\s*,\s*y\s*[=:]\s*([\d.]+)"#,
        ]
        for pattern in patterns {
            if let pair = firstPair(pattern, in: text, options: [.caseInsensitive]) { return pair }
        }
        if let pair = firstPair(#"\(?\s*([\d.]+)\s*[,;]\s*([\d.]+)\s*\)?"#, in: text, options: []),
           pair.0 <= 1, pair.1 <= 1 {
            return pair
        }
        return nil
    }

    private static func firstPair(
        _ pattern: String, in text: String, options: NSRegularExpression.Options
    ) -> (Double, Double)? {
        guard let regex = try? NSRegularExpression(pattern: pattern, options: options),
              let match = regex.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)),
              let r1 = Range(match.range(at: 1), in: text), let r2 = Range(match.range(at: 2), in: text),
              let x = Double(text[r1]), let y = Double(text[r2])
        else { return nil }
        return (x, y)
    }

    static func inside(_ click: (Double, Double), _ bbox: [Double]) -> Bool {
        bbox.count == 4 && bbox[0] <= click.0 && click.0 <= bbox[2] && bbox[1] <= click.1 && click.1 <= bbox[3]
    }
}
