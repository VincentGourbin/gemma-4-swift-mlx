// Forward de bout en bout: target prefill -> drafter draft. Permet de valider que
// le drafter produit des tokens senses sur des hidden states / shared K/V REELS
// (vs synthetique random comme dans mtp-smoke).
//
// Usage:
//   gemma4-cli mtp-forward --target mlx-community/gemma-4-e2b-it-4bit \
//     --drafter google/gemma-4-E2B-it-assistant \
//     --prompt "Bonjour, comment vas-" \
//     --block-size 4

import ArgumentParser
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Gemma4Swift

struct MtpForward: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "mtp-forward",
        abstract: "Target prefill + drafter forward sur prompt reel (validation Jalon B)"
    )

    @Option(name: .long, help: "Repo HuggingFace du target (e.g. mlx-community/gemma-4-e2b-it-4bit)")
    var target: String = "mlx-community/gemma-4-e2b-it-bf16"

    @Option(name: .long, help: "Repo HuggingFace du drafter Assistant")
    var drafter: String = "google/gemma-4-E2B-it-assistant"

    @Option(name: .long, help: "Prompt a faire prefiller par le target")
    var prompt: String = "Tell me a one-line fact about the moon."

    @Option(name: .long, help: "Block size pour le draft (drafter genere blockSize-1 tokens)")
    var blockSize: Int = 4

    @Flag(name: .long, help: "Skip le chat template (utilise raw encode au lieu de applyChatTemplate)")
    var raw: Bool = false

    @Option(name: .long, help: "Token HuggingFace")
    var hfToken: String?

    func run() async throws {
        print("[mtp-forward] target=\(target)")
        print("[mtp-forward] drafter=\(drafter)")
        print("[mtp-forward] prompt=\(prompt.debugDescription)")
        print("[mtp-forward] block_size=\(blockSize)")
        print()

        // ============================================================
        // 1. Telecharger les deux modeles
        // ============================================================
        print("[1/5] Download target...")
        let targetDir = try await Gemma4ModelDownloader.download(
            modelId: target, token: resolveHFToken(hfToken)
        ) { p in
            print("\r  \(Int(p.fraction * 100))% — \(p.currentFile)              ", terminator: "")
            fflush(stdout)
        }
        print("\n  -> \(targetDir.path)")

        print("[2/5] Download drafter...")
        let drafterDir = try await Gemma4ModelDownloader.download(
            modelId: drafter, token: resolveHFToken(hfToken)
        ) { p in
            print("\r  \(Int(p.fraction * 100))% — \(p.currentFile)              ", terminator: "")
            fflush(stdout)
        }
        print("\n  -> \(drafterDir.path)")

        // ============================================================
        // 2. Charger le target via loadModelContainer (text-only)
        // ============================================================
        print("[3/5] Load target (text-only)...")
        await Gemma4Registration.register(multimodal: false)
        let container = try await loadModelContainer(from: targetDir, using: Gemma4TokenizerLoader())
        print("  target loaded")

        // ============================================================
        // 3. Charger le drafter manuellement
        // ============================================================
        print("[4/5] Load drafter...")
        let drafterConfigData = try Data(contentsOf: drafterDir.appendingPathComponent("config.json"))
        let drafterConfig = try JSONDecoder().decode(Gemma4AssistantConfig.self, from: drafterConfigData)
        let drafterModel = Gemma4AssistantDraftModel(drafterConfig)

        let drafterSafetensors = try FileManager.default
            .contentsOfDirectory(at: drafterDir, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }

        var drafterRawWeights: [String: MLXArray] = [:]
        for url in drafterSafetensors {
            for (k, v) in try MLX.loadArrays(url: url) {
                drafterRawWeights[k] = v
            }
        }
        let drafterSanitized = Gemma4AssistantWeightSanitizer.sanitize(
            weights: drafterRawWeights, tieWordEmbeddings: drafterConfig.tieWordEmbeddings
        )
        try drafterModel.update(parameters: ModuleParameters.unflattened(drafterSanitized), verify: .all)
        print("  drafter loaded (\(drafterSanitized.count) tensors)")
        print("  drafter backbone=\(drafterConfig.backboneHiddenSize), hidden=\(drafterConfig.textConfig.hiddenSize)")
        print()

        // ============================================================
        // 4. Forward de bout en bout dans une seule perform
        // ============================================================
        print("[5/5] Run target prefill + drafter draft...")
        nonisolated(unsafe) let drafterRef = drafterModel
        let drafterBlockSize = blockSize
        let userPrompt = prompt
        let useChatTemplate = !raw

        try await container.perform { context in
            guard let llm = context.model as? Gemma4LLMModel else {
                fatalError("Expected Gemma4LLMModel, got \(type(of: context.model))")
            }
            let langModel = llm.languageModel
            let textCfg = langModel.config

            // ---- Tokenize ----
            let promptIds: [Int]
            if useChatTemplate {
                let messages: [[String: String]] = [["role": "user", "content": userPrompt]]
                promptIds = try context.tokenizer.applyChatTemplate(messages: messages)
            } else {
                promptIds = try context.tokenizer.encode(text: userPrompt)
            }
            let inputArr = MLXArray(promptIds.map { Int32($0) }).reshaped(1, -1)
            let promptLen = inputArr.dim(1)
            print("  prompt len = \(promptLen) tokens")
            print("  first 5: \(Array(promptIds.prefix(5))), last 5: \(Array(promptIds.suffix(5)))")

            // ---- Target prefill avec collecte des intermediates ----
            let prefillOut = langModel.forwardWithIntermediates(inputs: inputArr)
            eval(prefillOut.logits, prefillOut.preNormHiddenStates)

            // Last position logits -> first bonus
            let lastLogits = prefillOut.logits[0, promptLen - 1, 0...]
            let firstBonus = argMax(lastLogits, axis: -1).item(Int32.self)
            let firstBonusStr = (try? context.tokenizer.decode(tokenIds: [Int(firstBonus)])) ?? "?"
            print("  target first bonus = \(firstBonus) -> \(firstBonusStr.debugDescription)")

            // Last hidden state [1, 1, hidden]
            let lastHidden = prefillOut.preNormHiddenStates[0..., (promptLen - 1) ..< promptLen, 0...]
            print("  last hidden shape = \(lastHidden.shape)")

            // ---- Extraire shared K/V (derniere couche concrete full + sliding) ----
            let layerTypes = textCfg.resolvedLayerTypes
            let firstSharedIdx = textCfg.firstKvSharedLayerIdx
            let concreteTypes = Array(layerTypes.prefix(firstSharedIdx))
            guard let lastFullIdx = concreteTypes.lastIndex(of: "full_attention"),
                  let lastSlidingIdx = concreteTypes.lastIndex(of: "sliding_attention"),
                  let fullKV = prefillOut.intermediates[lastFullIdx],
                  let slidingKV = prefillOut.intermediates[lastSlidingIdx] else {
                fatalError("Cannot extract shared K/V from intermediates (firstShared=\(firstSharedIdx))")
            }
            print("  shared full layer idx = \(lastFullIdx), K shape = \(fullKV.keys.shape)")
            print("  shared sliding layer idx = \(lastSlidingIdx), K shape = \(slidingKV.keys.shape)")

            let sharedKV: SharedKVStates = [
                "full_attention": (keys: fullKV.keys, values: fullKV.values),
                "sliding_attention": (keys: slidingKV.keys, values: slidingKV.values),
            ]

            // ---- Bind drafter au target ----
            drafterRef.bind(target: langModel)
            drafterRef.setSharedKV(sharedKV, kvOffset: promptLen)

            // ---- Drafter draftBlock ----
            print()
            print("  --- Drafter draftBlock(blockSize=\(drafterBlockSize)) ---")
            let drafts = drafterRef.draftBlock(
                lastBonus: firstBonus,
                hidden: lastHidden,
                blockSize: drafterBlockSize
            ) { logits in
                argMax(logits, axis: -1)
            }
            eval(drafts)
            let nDrafts = drafts.dim(1)
            var draftIds: [Int32] = []
            for i in 0 ..< nDrafts {
                draftIds.append(drafts[0, i].item(Int32.self))
            }
            let draftsStr = (try? context.tokenizer.decode(tokenIds: draftIds.map(Int.init))) ?? "?"
            print("  drafts (\(nDrafts) tokens) = \(draftIds)")
            print("  drafts decoded = \(draftsStr.debugDescription)")

            // ---- Comparaison avec target greedy autoregressive (golden) ----
            print()
            print("  --- Target greedy autoregressive (golden) ---")
            // Continue le target sans drafter pour comparer
            var targetCache = langModel.makeCache()
            // Re-prefill pour avoir un cache utilisable
            _ = langModel(inputs: inputArr, cache: targetCache.map { $0 as KVCache? })
            var lastTok = firstBonus
            var goldenIds: [Int32] = [firstBonus]
            for _ in 1 ... nDrafts {
                let stepIn = MLXArray([lastTok]).reshaped(1, 1)
                let stepOut = langModel(inputs: stepIn, cache: targetCache.map { $0 as KVCache? })
                lastTok = argMax(stepOut[0, 0, 0...], axis: -1).item(Int32.self)
                goldenIds.append(lastTok)
            }
            let goldenStr = (try? context.tokenizer.decode(tokenIds: goldenIds.map(Int.init))) ?? "?"
            print("  golden (\(goldenIds.count) tokens incl bonus) = \(goldenIds)")
            print("  golden decoded = \(goldenStr.debugDescription)")

            // Comparison
            let goldenDrafts = Array(goldenIds.dropFirst())  // skip the bonus
            print()
            print("  --- Comparaison drafts vs golden ---")
            var matches = 0
            for i in 0 ..< nDrafts {
                let m = draftIds[i] == goldenDrafts[i]
                if m { matches += 1 }
                let mark = m ? "OK" : "DIFF"
                let dStr = (try? context.tokenizer.decode(tokenIds: [Int(draftIds[i])])) ?? "?"
                let gStr = (try? context.tokenizer.decode(tokenIds: [Int(goldenDrafts[i])])) ?? "?"
                print("  pos \(i): drafter=\(draftIds[i])(\(dStr.debugDescription))  golden=\(goldenDrafts[i])(\(gStr.debugDescription))  [\(mark)]")
            }
            print()
            print("  -> \(matches)/\(nDrafts) drafts == target greedy")
            print("  Note: 100% est exceptionnel (drafter ~ target). 50%+ est deja le signe d'un drafter sain.")
        }

        print("\n[mtp-forward] DONE")
    }
}
