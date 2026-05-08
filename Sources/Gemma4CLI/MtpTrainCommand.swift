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

    @Option(name: .long, help: "Fichier JSONL avec un champ 'text' par ligne")
    var data: String

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

        // 3. Charger + tokenizer le dataset
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
        print("  \(lines.count) lignes")

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
        let sve = saveEvery
        let wURL = weightsURL
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

            // Tokeniser chaque ligne (avec chat template? non, brut pour training general)
            print("  tokenizing \(textLines.count) lines...")
            var tokenizedSamples: [[Int]] = []
            for line in textLines {
                let parsed = try parseJsonlLine(line)
                let tokens = try context.tokenizer.encode(text: parsed)
                if tokens.count >= sl {
                    tokenizedSamples.append(tokens)
                }
            }
            print("  \(tokenizedSamples.count) samples avec >= \(sl) tokens")

            guard !tokenizedSamples.isEmpty else {
                fatalError("Aucun sample n'a au moins \(sl) tokens")
            }

            // Optimizer Adam
            let optimizer = Adam(learningRate: lrate)

            var config = Gemma4DrafterTraining.TrainConfig()
            config.iterations = it
            config.seqLen = sl
            config.batchSize = bs
            config.stepsPerReport = spr
            config.saveEvery = sve
            config.weightsURL = wURL

            try Gemma4DrafterTraining.trainDrafter(
                drafter: drafterRef,
                target: langModel,
                tokenizedSamples: tokenizedSamples,
                lastFullCacheIdx: lastFullIdx,
                lastSlidingCacheIdx: lastSlidingIdx,
                optimizer: optimizer,
                config: config
            )
        }

        print("\n[mtp-train] DONE — drafter sauve dans \(weightsURL.path)")
    }

    /// Parse une ligne JSONL avec un champ "text"
    private func parseJsonlLine(_ line: String) throws -> String {
        guard let data = line.data(using: .utf8) else {
            throw NSError(domain: "MtpTrain", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid line"])
        }
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let text = json["text"] as? String else {
            throw NSError(domain: "MtpTrain", code: 2, userInfo: [
                NSLocalizedDescriptionKey: "Line missing 'text' field: \(line.prefix(80))"
            ])
        }
        return text
    }
}
