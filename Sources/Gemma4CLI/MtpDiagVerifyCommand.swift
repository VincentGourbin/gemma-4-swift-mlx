// Diagnostic: compare hidden states / logits produits par deux modes equivalents
// pour le MEME modele et la MEME sequence de tokens:
//   1. Sequential: autoregressive decode token-par-token (N forwards de longueur 1)
//   2. Parallel  : un forward de longueur N
//
// Si les hidden states divergent significativement, le verify forward du target
// produit un etat different du sequential equivalent, ce qui expliquerait pourquoi
// le drafter MTP (entraine sur sequential decode du target) drift sur les rounds.

import ArgumentParser
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Gemma4Swift

struct MtpDiagVerify: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "mtp-diag-verify",
        abstract: "Diagnostic: hidden states sequential vs parallel verify (target seul)"
    )

    @Option(name: .long, help: "Repo HuggingFace du target (BF16 recommande)")
    var target: String = "mlx-community/gemma-4-e2b-it-bf16"

    @Option(name: .long, help: "Prompt utilisateur")
    var prompt: String = "Tell me a one-line fact about the moon."

    @Option(name: .long, help: "Nombre de tokens a comparer apres le prompt")
    var nTokens: Int = 8

    @Option(name: .long, help: "Token HuggingFace")
    var hfToken: String?

    func run() async throws {
        print("[mtp-diag-verify] target=\(target)")
        print("[mtp-diag-verify] n_tokens=\(nTokens)")
        print()

        let targetDir = try await Gemma4ModelDownloader.download(
            modelId: target, token: resolveHFToken(hfToken)
        ) { _ in }

        await Gemma4Registration.register(multimodal: false)
        let container = try await loadModelContainer(from: targetDir, using: Gemma4TokenizerLoader())

        let n = nTokens
        let userPrompt = prompt

        try await container.perform { context in
            guard let llm = context.model as? Gemma4LLMModel else {
                fatalError("Expected Gemma4LLMModel, got \(type(of: context.model))")
            }
            let langModel = llm.languageModel

            // Tokenize avec chat template
            let messages: [[String: String]] = [["role": "user", "content": userPrompt]]
            let promptIds = try context.tokenizer.applyChatTemplate(messages: messages)
            let promptArr = MLXArray(promptIds.map { Int32($0) }).reshaped(1, -1)
            let promptLen = promptArr.dim(1)
            print("prompt_len = \(promptLen)")

            // ==============================================================
            // MODE A: Sequential autoregressive decode (greedy, N steps)
            // ==============================================================
            print("\n=== Mode A: SEQUENTIAL autoregressive decode ===")
            let cacheSeq = langModel.makeCache()
            let cacheSeqArr = cacheSeq.map { $0 as KVCache? }

            // Prefill
            let prefillSeq = langModel.forwardWithIntermediates(
                inputs: promptArr, cache: cacheSeqArr
            )
            eval(prefillSeq.logits, prefillSeq.preNormHiddenStates)

            // First bonus
            let firstBonusSeq = argMax(prefillSeq.logits[0, promptLen - 1, 0...], axis: -1).item(Int32.self)
            print("first_bonus = \(firstBonusSeq) -> \((try? context.tokenizer.decode(tokenIds: [Int(firstBonusSeq)])) ?? "?")")

            // Sequential N steps
            var seqTokens: [Int32] = [firstBonusSeq]
            var seqHiddens: [MLXArray] = []
            var seqLogits: [MLXArray] = []
            var lastTok = firstBonusSeq
            for step in 0 ..< n {
                let stepIn = MLXArray([lastTok]).reshaped(1, 1)
                let out = langModel.forwardWithIntermediates(inputs: stepIn, cache: cacheSeqArr)
                eval(out.logits, out.preNormHiddenStates)
                seqHiddens.append(out.preNormHiddenStates[0..., 0..<1, 0...])
                seqLogits.append(out.logits[0..., 0..<1, 0...])
                let nextTok = argMax(out.logits[0, 0, 0...], axis: -1).item(Int32.self)
                seqTokens.append(nextTok)
                lastTok = nextTok
                print("  step \(step): in=\(seqTokens[step]) → out=\(nextTok) (\((try? context.tokenizer.decode(tokenIds: [Int(nextTok)])) ?? "?"))")
            }

            // ==============================================================
            // MODE B: Parallel multi-token decode (le mode utilise par MTP verify)
            // ==============================================================
            print("\n=== Mode B: PARALLEL multi-token verify ===")
            let cachePar = langModel.makeCache()
            let cacheParArr = cachePar.map { $0 as KVCache? }

            // Re-prefill
            _ = langModel.forwardWithIntermediates(inputs: promptArr, cache: cacheParArr)

            // Input parallele = [first_bonus, then N-1 tokens generated sequentially]
            // (= ce que le verify recoit de cote MTP)
            let parInput = MLXArray(seqTokens.prefix(n).map { $0 }).reshaped(1, n)
            print("  input parallel = \(Array(seqTokens.prefix(n)))")

            let outPar = langModel.forwardWithIntermediates(inputs: parInput, cache: cacheParArr)
            eval(outPar.logits, outPar.preNormHiddenStates)

            // ==============================================================
            // COMPARISON
            // ==============================================================
            print("\n=== COMPARISON: seqHiddens[i] vs outPar.preNormHiddenStates[:, i, :] ===")
            print("(diff non-zero indique parallel verify produit un etat different du sequential)")
            print()
            for i in 0 ..< n {
                let seqH = seqHiddens[i]               // [1, 1, hidden]
                let parH = outPar.preNormHiddenStates[0..., i..<(i+1), 0...]  // [1, 1, hidden]

                let diff = abs(seqH - parH)
                let maxDiff = diff.max().item(Float.self)
                let meanDiff = diff.mean().item(Float.self)
                let normSeq = sqrt((seqH * seqH).sum().item(Float.self))
                let normPar = sqrt((parH * parH).sum().item(Float.self))
                let cosSim: Float = {
                    let dot = (seqH * parH).sum().item(Float.self)
                    return dot / (normSeq * normPar + 1e-9)
                }()

                // Compare logits aussi
                let seqL = seqLogits[i]
                let parL = outPar.logits[0..., i..<(i+1), 0...]
                let seqArgmax = argMax(seqL[0, 0, 0...], axis: -1).item(Int32.self)
                let parArgmax = argMax(parL[0, 0, 0...], axis: -1).item(Int32.self)
                let argmaxMatch = seqArgmax == parArgmax ? "OK" : "DIFF"

                print(String(format: "  pos %d: hidden max_diff=%.4e mean_diff=%.4e cosSim=%.6f  argmax seq=%d par=%d [%@]",
                             i, maxDiff, meanDiff, cosSim, seqArgmax, parArgmax, argmaxMatch))
            }

            // Also compare hidden state norms
            print("\n=== HIDDEN NORMS (sanity) ===")
            for i in 0 ..< n {
                let seqH = seqHiddens[i]
                let parH = outPar.preNormHiddenStates[0..., i..<(i+1), 0...]
                let normSeq = sqrt((seqH * seqH).sum().item(Float.self))
                let normPar = sqrt((parH * parH).sum().item(Float.self))
                print(String(format: "  pos %d: |seq|=%.4f  |par|=%.4f", i, normSeq, normPar))
            }

            // ==============================================================
            // MODE C: parallel forward avec L=1 (degenerate, equivalent au sequential)
            // Si meme avec L=1 le hidden differe du sequential, le bug est ailleurs.
            // ==============================================================
            print("\n=== Mode C: PARALLEL forward avec L=1 (chaque token un par un en mode parallele) ===")
            let cacheC = langModel.makeCache()
            let cacheCArr = cacheC.map { $0 as KVCache? }
            _ = langModel.forwardWithIntermediates(inputs: promptArr, cache: cacheCArr)

            for i in 0 ..< n {
                let stepIn = MLXArray([seqTokens[i]]).reshaped(1, 1)
                let outC = langModel.forwardWithIntermediates(inputs: stepIn, cache: cacheCArr)
                eval(outC.preNormHiddenStates)
                let cH = outC.preNormHiddenStates[0..., 0..<1, 0...]
                let seqH = seqHiddens[i]
                let diff = abs(cH - seqH)
                let maxDiff = diff.max().item(Float.self)
                let meanDiff = diff.mean().item(Float.self)
                print(String(format: "  pos %d (L=1 par): max_diff_vs_seq=%.4e mean_diff=%.4e", i, maxDiff, meanDiff))
            }
        }

        print("\n[mtp-diag-verify] DONE")
    }
}
