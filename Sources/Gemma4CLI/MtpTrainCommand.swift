// Fine-tune le drafter MTP par auto-distillation contre un target frozen.
//
// Usage:
//   gemma4-cli mtp-train \
//     --target /path/to/target \
//     --drafter google/gemma-4-E2B-it-assistant \
//     --data dataset.jsonl \
//     --iterations 200 \
//     --output drafter_finetuned/

import ArgumentParser
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXOptimizers
import Gemma4Swift

struct MtpTrain: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "mtp-train",
        abstract: "Fine-tune le drafter MTP par auto-distillation contre le target"
    )

    @Option(name: .long, help: "Chemin local vers le target")
    var target: String

    @Option(name: .long, help: "Repo HF du drafter de depart (Google Assistant)")
    var drafter: String = "google/gemma-4-E2B-it-assistant"

    @Option(name: .long, help: "Token HuggingFace")
    var hfToken: String?

    @Option(name: .long, help: "Fichier JSONL avec un champ 'text' par ligne (ou 'messages' pour chat)")
    var data: String

    @Option(name: .long, help: "Optionnel: fichier JSONL de validation. Si fourni, eval loss tous les --steps-per-valid")
    var validData: String?

    @Option(name: .long, help: "Frequence de validation (en iterations). 0 = pas de valid (default)")
    var stepsPerValid: Int = 0

    @Option(name: .long, help: "Nombre de batches a evaluer sur la valid")
    var validBatches: Int = 8

    @Option(name: .long, help: "Repertoire de sortie pour les poids fine-tunes")
    var output: String

    @Option(name: .long, help: "Nombre d'iterations de training")
    var iterations: Int = 100

    @Option(name: .long, help: "Sequence length par chunk")
    var seqLen: Int = 256

    @Option(name: .long, help: "Batch size (chunks per step)")
    var batchSize: Int = 1

    @Option(name: .long, help: "Learning rate")
    var lr: Float = 1e-4

    @Option(name: .long, help: "Steps entre les rapports")
    var stepsPerReport: Int = 10

    @Option(name: .long, help: "Steps entre les checkpoints")
    var saveEvery: Int = 50

    func run() async throws {
        print("[mtp-train] target=\(target)")
        print("[mtp-train] drafter=\(drafter)")
        print("[mtp-train] data=\(data)")
        print("[mtp-train] output=\(output)")
        print("[mtp-train] iterations=\(iterations) seq_len=\(seqLen) lr=\(lr)")
        print()

        // 1. Load target (text-only)
        print("[1/4] Load target...")
        await Gemma4Registration.register(multimodal: false)
        let targetURL = URL(fileURLWithPath: target)
        let container = try await loadModelContainer(from: targetURL, using: Gemma4TokenizerLoader())

        // 2. Load drafter (depuis HF + sanitize + verify)
        print("[2/4] Load drafter...")
        let drafterModel = try await loadDrafter(repo: drafter, hfToken: hfToken)

        // 3. Charger + tokenizer le dataset (train + optionnel valid)
        print("[3/4] Tokenize dataset...")
        let dataURL = URL(fileURLWithPath: data)
        let lines = try String(contentsOf: dataURL, encoding: .utf8)
            .split(separator: "\n")
            .map { String($0) }
            .filter { !$0.isEmpty }
        guard !lines.isEmpty else {
            print("Erreur: dataset vide")
            throw ExitCode.failure
        }
        print("  train: \(lines.count) lignes")

        var validLines: [String] = []
        if let vd = validData {
            let validURL = URL(fileURLWithPath: vd)
            validLines = try String(contentsOf: validURL, encoding: .utf8)
                .split(separator: "\n")
                .map { String($0) }
                .filter { !$0.isEmpty }
            print("  valid: \(validLines.count) lignes")
        }

        // Setup output dir
        let outputURL = URL(fileURLWithPath: output)
        try FileManager.default.createDirectory(at: outputURL, withIntermediateDirectories: true)
        let weightsURL = outputURL.appendingPathComponent("drafter.safetensors")

        // 4. Training loop
        print("[4/4] Training...")
        let it = iterations
        let sl = seqLen
        let bs = batchSize
        let lrate = lr
        let spr = stepsPerReport
        let spv = stepsPerValid
        let vb = validBatches
        let sve = saveEvery
        let wURL = weightsURL
        nonisolated(unsafe) let validLinesRef = validLines
        nonisolated(unsafe) let drafterRef = drafterModel
        nonisolated(unsafe) let textLines = lines

        try await container.perform { context in
            guard let llm = context.model as? Gemma4LLMModel else {
                fatalError("Expected Gemma4LLMModel for training")
            }
            let langModel = llm.languageModel
            let textCfg = langModel.config

            // Indices des dernieres couches concretes par type
            let layerTypes = textCfg.resolvedLayerTypes
            let concreteTypes = Array(layerTypes.prefix(textCfg.firstKvSharedLayerIdx))
            guard let lastFullIdx = concreteTypes.lastIndex(of: "full_attention"),
                  let lastSlidingIdx = concreteTypes.lastIndex(of: "sliding_attention") else {
                fatalError("Cannot find concrete full_attention and sliding_attention layers")
            }

            // Bind drafter au target (pour input_embed dans le path d'inference)
            drafterRef.bind(target: langModel)

            // Tokeniser chaque ligne. Supporte 2 formats JSONL:
            //  - {"text": "..."} : texte brut (concatene les chunks de seqLen)
            //  - {"messages": [{"role": ..., "content": ...}, ...]} : chat,
            //    rendu via applyChatTemplate, regroupe pour atteindre seqLen
            print("  tokenizing \(textLines.count) lines...")
            var allTokens: [Int] = []
            var tokenizedSamples: [[Int]] = []
            var failedLines = 0
            for (lineIdx, line) in textLines.enumerated() {
                let tokens: [Int]
                do {
                    tokens = try parseAndTokenizeJsonlLine(line, tokenizer: context.tokenizer)
                } catch {
                    failedLines += 1
                    if failedLines <= 3 {
                        print("  line \(lineIdx + 1) failed: \(error.localizedDescription)")
                        print("  preview: \(line.prefix(120))")
                    }
                    continue
                }
                if tokens.isEmpty { continue }

                // Si la ligne fait deja >= seqLen, en faire un sample.
                if tokens.count >= sl {
                    tokenizedSamples.append(tokens)
                } else {
                    // Sinon accumuler dans un buffer global, decouper en chunks de seqLen
                    allTokens.append(contentsOf: tokens)
                    while allTokens.count >= sl {
                        tokenizedSamples.append(Array(allTokens.prefix(sl)))
                        allTokens.removeFirst(sl)
                    }
                }
            }
            print("  train: \(tokenizedSamples.count) samples (chunks de \(sl)), \(failedLines) lignes echouees")

            guard !tokenizedSamples.isEmpty else {
                fatalError("Aucun sample n'a au moins \(sl) tokens")
            }

            // Tokeniser la valid (si fournie)
            var validSamples: [[Int]] = []
            var validBuffer: [Int] = []
            for line in validLinesRef {
                guard let toks = try? parseAndTokenizeJsonlLine(line, tokenizer: context.tokenizer) else { continue }
                if toks.count >= sl {
                    validSamples.append(toks)
                } else {
                    validBuffer.append(contentsOf: toks)
                    while validBuffer.count >= sl {
                        validSamples.append(Array(validBuffer.prefix(sl)))
                        validBuffer.removeFirst(sl)
                    }
                }
            }
            if !validSamples.isEmpty {
                print("  valid: \(validSamples.count) samples")
            }

            // Optimizer Adam
            let optimizer = Adam(learningRate: lrate)

            var config = Gemma4DrafterTraining.TrainConfig()
            config.iterations = it
            config.seqLen = sl
            config.batchSize = bs
            config.stepsPerReport = spr
            config.stepsPerValid = spv
            config.validBatches = vb
            config.saveEvery = sve
            config.weightsURL = wURL

            try Gemma4DrafterTraining.trainDrafter(
                drafter: drafterRef,
                target: langModel,
                tokenizedSamples: tokenizedSamples,
                validSamples: validSamples,
                lastFullCacheIdx: lastFullIdx,
                lastSlidingCacheIdx: lastSlidingIdx,
                optimizer: optimizer,
                config: config
            )
        }

        print("\n[mtp-train] DONE — drafter sauve dans \(weightsURL.path)")
    }

    /// Parse une ligne JSONL et la tokenize. Supporte:
    ///  - {"text": "..."} : tokenise via encode()
    ///  - {"messages": [...]} : tokenise via applyChatTemplate
    private func parseAndTokenizeJsonlLine(_ line: String, tokenizer: any Tokenizer) throws -> [Int] {
        guard let data = line.data(using: .utf8) else {
            throw NSError(domain: "MtpTrain", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid line"])
        }
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NSError(domain: "MtpTrain", code: 2, userInfo: [
                NSLocalizedDescriptionKey: "Not valid JSON: \(line.prefix(80))"
            ])
        }

        if let text = json["text"] as? String {
            return try tokenizer.encode(text: text)
        }
        if let messages = json["messages"] as? [[String: Any]] {
            // Convertir au format [String: String] requis par applyChatTemplate
            let strMessages: [[String: String]] = messages.compactMap { m in
                guard let role = m["role"] as? String,
                      let content = m["content"] as? String else { return nil }
                return ["role": role, "content": content]
            }
            return try tokenizer.applyChatTemplate(messages: strMessages)
        }
        throw NSError(domain: "MtpTrain", code: 3, userInfo: [
            NSLocalizedDescriptionKey: "Line missing 'text' or 'messages': \(line.prefix(80))"
        ])
    }
}
