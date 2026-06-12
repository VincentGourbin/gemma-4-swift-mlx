// Eval MMLU 5-shot multi-choice style lm-eval-harness.
//
// Format standard (subject-aware few-shot) :
//   The following are multiple choice questions (with answers) about <subject>.
//
//   <dev_q1>
//   A. ...
//   B. ...
//   C. ...
//   D. ...
//   Answer: <gold_letter>
//
//   [5 exemples du dev split]
//
//   <test_q>
//   A. ...
//   B. ...
//   C. ...
//   D. ...
//   Answer:
//
// Le modele predit un token unique parmi " A"/" B"/" C"/" D" (avec espace
// leading). On compare les logits a ces 4 positions de vocab et on prend
// l'argmax.

import ArgumentParser
import Foundation
import Gemma4Swift
import MLX
import MLXLMCommon
import MLXLLM
import Tokenizers

struct EvalMmlu: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "eval-mmlu",
        abstract: "Eval MMLU 5-shot (style lm-eval-harness)"
    )

    @Option(name: .long, help: "Chemin local vers le modele")
    var modelPath: String

    @Option(name: .long, help: "Fichier JSON 5-shot (avec dev examples par sujet)")
    var dataset: String = "/tmp/mmlu_5shot.json"

    @Option(name: .customLong("quantize-bits"), help: "Quantification a la volee des poids (4/6/8)")
    var quantizeBits: Int?

    @Option(name: .customLong("quantize-group-size"), help: "Group size (defaut 64)")
    var quantizeGroupSize: Int = 64

    @Option(name: .customLong("quantize-mode"), help: "Mode : affine | mxfp4 | mxfp8")
    var quantizeMode: String = "affine"

    @Option(name: .long, help: "Bits de quantization KV cache TurboQuant (3 ou 4)")
    var kvBits: Int?

    @Option(name: .long, help: "Limite le nombre de questions")
    var limit: Int?

    @Option(name: .long, help: "Verbose : print chaque question")
    var verbose: Bool = false

    @Flag(name: .long, help: "Mode Chain-of-Thought : utilise cot_content pour 5-shot + generation libre + parse 'answer is (X)'")
    var cot: Bool = false

    @Option(name: .long, help: "Max tokens generes par question en mode CoT")
    var cotMaxTokens: Int = 512

    struct MmluItem: Codable {
        let subject: String
        let question: String
        let choices: [String]
        let answer: Int
    }

    struct DevItem: Codable {
        let q: String
        let c: [String]
        let a: Int
        let cot: String?
    }

    struct SubjectInfo: Codable {
        let dev: [DevItem]
    }

    struct MmluDataset: Codable {
        let subjects: [String: SubjectInfo]
        let questions: [MmluItem]
    }

    func run() async throws {
        // Support jusqu'a 10 choix pour MMLU Pro (A..J)
        let letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

        let data = try Data(contentsOf: URL(fileURLWithPath: dataset))
        let ds = try JSONDecoder().decode(MmluDataset.self, from: data)
        var items = ds.questions
        if let limit = limit { items = Array(items.prefix(limit)) }
        print("Loaded \(items.count) test questions, \(ds.subjects.count) subjects (5-shot dev)")

        // Pre-construit les prefixes 5-shot par sujet (CoT ou non).
        var prefixBySubject: [String: String] = [:]
        for (subj, info) in ds.subjects {
            let subjFmt = subj.replacingOccurrences(of: "_", with: " ")
            var p = "The following are multiple choice questions (with answers) about \(subjFmt).\n\n"
            for d in info.dev {
                p += "\(d.q)\n"
                for (i, c) in d.c.enumerated() where i < letters.count {
                    p += "\(letters[i]). \(c)\n"
                }
                if cot, let cotText = d.cot, !cotText.isEmpty {
                    // cotText commence souvent par "A: Let's think step by step..."
                    p += "\(cotText)\n\n"
                } else {
                    p += "Answer: \(letters[d.a])\n\n"
                }
            }
            prefixBySubject[subj] = p
        }

        // Load model
        await Gemma4Registration.register()
        let url = URL(fileURLWithPath: modelPath)
        let container = try await loadModelContainer(from: url, using: LocalTokenizerLoader())

        // OTF quant
        if let bits = quantizeBits {
            guard let mode = Gemma4OnTheFlyQuantization.Mode(rawValue: quantizeMode) else {
                print("Erreur: --quantize-mode invalide")
                throw ExitCode.failure
            }
            let count = await container.perform { context in
                Gemma4OnTheFlyQuantization.apply(
                    to: context.model, bits: bits,
                    groupSize: self.quantizeGroupSize, mode: mode
                )
            }
            print("OTF quant: \(count) modules (\(bits)-bit, \(mode.rawValue), g=\(quantizeGroupSize))")
        }

        // Resolve les 4 token IDs pour " A", " B", " C", " D"
        // (les exemples 5-shot conditionnent le modele a sortir un single letter)
        let letterTokens: [Int] = await container.perform { context in
            return letters.map { l in
                let ids = context.tokenizer.encode(text: " \(l)")
                // Skip BOS, prendre le dernier token de " A"
                return ids.last!
            }
        }
        print("Letter token IDs : A=\(letterTokens[0]) B=\(letterTokens[1]) C=\(letterTokens[2]) D=\(letterTokens[3])")

        let kvBitsParam = self.kvBits

        var correctByCfg = ["bf16-kv": 0]
        var totalByCfg = ["bf16-kv": 0]
        if kvBitsParam != nil { correctByCfg["TQ-kv"] = 0; totalByCfg["TQ-kv"] = 0 }
        var bySubjectCorrect: [String: Int] = [:]
        var bySubjectTotal: [String: Int] = [:]
        var sameAnswer = 0

        let startTime = Date()

        let cfgs: [(String, Int?)]
        if let kv = kvBitsParam {
            cfgs = [("bf16-kv", nil), ("TQ-kv", kv)]
        } else {
            cfgs = [("bf16-kv", nil)]
        }

        for (idx, item) in items.enumerated() {
            // Build le 5-shot prompt
            let prefix = prefixBySubject[item.subject] ?? ""
            var prompt = prefix
            prompt += "\(item.question)\n"
            for (i, c) in item.choices.enumerated() where i < letters.count {
                prompt += "\(letters[i]). \(c)\n"
            }
            // Format CoT : "A: Let's think step by step." declenche le raisonnement.
            // Format direct : "Answer:" + logit lookup.
            if cot {
                prompt += "A: Let's think step by step."
            } else {
                prompt += "Answer:"
            }

            // On limite l'argmax aux choix VALIDES de cette question
            let nChoices = min(item.choices.count, letters.count)
            let candidateTokens = Array(letterTokens.prefix(nChoices))

            var answers: [String: Int] = [:]
            for (cfgName, kv) in cfgs {
                let answer: Int
                if cot {
                    answer = await Self.predictAnswerCoT(
                        container: container,
                        prompt: prompt,
                        nChoices: nChoices,
                        maxTokens: self.cotMaxTokens,
                        kvBits: kv
                    )
                } else {
                    answer = await Self.predictAnswerLogits(
                        container: container,
                        prompt: prompt,
                        letterTokens: candidateTokens,
                        kvBits: kv
                    )
                }
                answers[cfgName] = answer
                totalByCfg[cfgName, default: 0] += 1
                if answer == item.answer {
                    correctByCfg[cfgName, default: 0] += 1
                    if cfgName == "bf16-kv" {
                        bySubjectCorrect[item.subject, default: 0] += 1
                    }
                }
            }
            bySubjectTotal[item.subject, default: 0] += 1

            if cfgs.count == 2 && answers["bf16-kv"] == answers["TQ-kv"] {
                sameAnswer += 1
            }

            if verbose || (idx + 1) % 10 == 0 {
                let elapsed = Date().timeIntervalSince(startTime)
                let eta = elapsed / Double(idx + 1) * Double(items.count - idx - 1)
                func ltr(_ i: Int?) -> String {
                    guard let i = i, i >= 0 && i < 4 else { return "?" }
                    return letters[i]
                }
                print("[\(idx + 1)/\(items.count)] subj=\(item.subject) gold=\(letters[item.answer]) " +
                    "bf16=\(ltr(answers["bf16-kv"]))" +
                    (kvBitsParam != nil ? " TQ=\(ltr(answers["TQ-kv"]))" : "") +
                    String(format: " | %.1fs elapsed, ETA %.0fs", elapsed, eta))
            }
        }

        let elapsed = Date().timeIntervalSince(startTime)
        print("\n=== Results (\(items.count) questions, 5-shot, \(String(format: "%.1f", elapsed))s) ===")
        for (cfg, total) in totalByCfg {
            let correct = correctByCfg[cfg] ?? 0
            let pct = Double(correct) / Double(total) * 100.0
            print("  \(cfg): \(correct)/\(total) = \(String(format: "%.1f%%", pct))")
        }
        if cfgs.count == 2 {
            let agreePct = Double(sameAnswer) / Double(items.count) * 100.0
            print("  Agreement bf16 vs TQ: \(sameAnswer)/\(items.count) = \(String(format: "%.1f%%", agreePct))")
        }
        print("\nBy subject (bf16-kv accuracy):")
        for subj in bySubjectTotal.keys.sorted() {
            let c = bySubjectCorrect[subj, default: 0]
            let t = bySubjectTotal[subj]!
            print("  \(subj): \(c)/\(t)")
        }
    }

    /// Mode Chain-of-Thought : greedy generation jusqu'a "The answer is (X)"
    /// ou EOS. Parse la lettre finale via regex.
    private static func predictAnswerCoT(
        container: ModelContainer,
        prompt: String,
        nChoices: Int,
        maxTokens: Int,
        kvBits: Int?
    ) async -> Int {
        let kvBitsArg = kvBits
        let generated: String = await container.perform { context in
            let promptIds = context.tokenizer.encode(text: prompt)
            let inputIds = MLXArray(promptIds.map { Int32($0) })
            let params = kvBitsArg != nil ? GenerateParameters(kvBits: kvBitsArg) : nil
            let cache = context.model.newCache(parameters: params)

            // Prefill
            let prefillOut = context.model(inputIds.reshaped(1, -1), cache: cache)
            var nextTok = argMax(prefillOut[0..., prefillOut.dim(1) - 1, 0...], axis: -1).item(Int32.self)
            var genIds: [Int] = [Int(nextTok)]

            for _ in 0 ..< maxTokens - 1 {
                if nextTok == 1 || nextTok == 106 { break }  // EOS
                let stepIn = MLXArray([nextTok]).reshaped(1, 1)
                let out = context.model(stepIn, cache: cache)
                nextTok = argMax(out[0..., 0, 0...], axis: -1).item(Int32.self)
                genIds.append(Int(nextTok))
                // Stop precoce si on detecte "The answer is" pour eviter de continuer.
                // (decode partiel toutes les ~32 tokens pour limiter le cout)
                if genIds.count % 32 == 0 {
                    let partial = context.tokenizer.decode(tokenIds: genIds).lowercased()
                    if partial.contains("the answer is") { break }
                }
            }
            return context.tokenizer.decode(tokenIds: genIds)
        }

        return parseLetterAnswer(generated, nChoices: nChoices)
    }

    /// Parse une reponse CoT pour extraire la lettre choisie.
    /// Strategie : cherche "answer is (X)" / "answer is X" / "(X)" / fallback first letter.
    static func parseLetterAnswer(_ text: String, nChoices: Int) -> Int {
        let letters = "ABCDEFGHIJ"
        let validLetters = Array(letters.prefix(nChoices))
        let lower = text.lowercased()

        // Pattern 1 : "answer is (X)" ou "answer is X"
        for marker in ["answer is (", "answer is *", "answer is **", "answer is "] {
            if let r = lower.range(of: marker) {
                let after = text[r.upperBound...]
                for ch in after.prefix(5) {
                    if let idx = validLetters.firstIndex(of: ch) { return idx }
                }
            }
        }

        // Pattern 2 : derniere occurrence de "(X)" dans le texte
        let chars = Array(text)
        for i in stride(from: chars.count - 1, through: 0, by: -1) {
            if chars[i] == ")" && i >= 2 && chars[i - 2] == "(" {
                let ch = chars[i - 1]
                if let idx = validLetters.firstIndex(of: ch) { return idx }
            }
        }

        // Pattern 3 : derniere lettre majuscule isolee (espace avant/apres ou fin)
        for i in stride(from: chars.count - 1, through: 0, by: -1) {
            let ch = chars[i]
            if let idx = validLetters.firstIndex(of: ch) {
                let prev = i > 0 ? chars[i - 1] : " "
                let next = i < chars.count - 1 ? chars[i + 1] : " "
                if !prev.isLetter && !next.isLetter { return idx }
            }
        }

        return -1
    }

    /// Forward complet sur le prompt 5-shot, compare logits a " A"/" B"/" C"/" D".
    private static func predictAnswerLogits(
        container: ModelContainer,
        prompt: String,
        letterTokens: [Int],
        kvBits: Int?
    ) async -> Int {
        let kvBitsArg = kvBits
        return await container.perform { context in
            let promptIds = context.tokenizer.encode(text: prompt)
            let inputIds = MLXArray(promptIds.map { Int32($0) }).reshaped(1, -1)
            let params = kvBitsArg != nil ? GenerateParameters(kvBits: kvBitsArg) : nil
            let cache = context.model.newCache(parameters: params)
            let logits = context.model(inputIds, cache: cache)
            // Logits du dernier token = distribution du prochain token (= " A"/" B"/...)
            let lastLogits = logits[0, logits.dim(1) - 1, 0...]
            // Gather des candidats en un seul tenseur + UN seul sync via asArray
            // (evite N appels .item() qui forcent N sync GPU->CPU).
            let candidateIdx = MLXArray(letterTokens.map { Int32($0) })
            let candidateLogits = lastLogits.take(candidateIdx, axis: 0).asArray(Float.self)
            var bestIdx = 0
            var bestLogit = Float(-Float.infinity)
            for (i, l) in candidateLogits.enumerated() {
                if l > bestLogit { bestLogit = l; bestIdx = i }
            }
            return bestIdx
        }
    }
}
