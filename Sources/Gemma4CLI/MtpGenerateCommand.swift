// Generation MTP end-to-end: target + drafter, comparaison avec generation standard.
//
// Usage:
//   gemma4-cli mtp-generate --target ... --drafter ... --prompt "..." --max-tokens 64
//   gemma4-cli mtp-generate --compare      # active la comparaison cote-a-cote standard vs MTP
//
// Pour le mode --compare, le test d'equivalence est greedy strict: les sorties doivent
// etre identiques. Si elles divergent, MTP a un bug.

import ArgumentParser
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Gemma4Swift

struct MtpGenerate: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "mtp-generate",
        abstract: "Generation speculative decoding (MTP) avec equivalence greedy"
    )

    @Option(name: .long, help: "Repo HuggingFace du target (BF16 recommande)")
    var target: String = "mlx-community/gemma-4-e2b-it-bf16"

    @Option(name: .long, help: "Repo HuggingFace du drafter Assistant")
    var drafter: String = "google/gemma-4-E2B-it-assistant"

    @Option(name: .long, help: "Chemin vers un fichier safetensors de drafter fine-tune (override le drafter HF). Le config.json doit etre dans le meme repertoire ou hérité de --drafter.")
    var drafterPath: String?

    @Option(name: .long, help: "Prompt utilisateur")
    var prompt: String = "Tell me a one-line fact about the moon."

    @Option(name: .long, help: "Block size MTP (drafter genere blockSize-1 tokens par round)")
    var blockSize: Int = 4

    @Option(name: .long, help: "Nombre maximum de tokens a generer")
    var maxTokens: Int = 64

    @Flag(name: .long, help: "Compare la sortie MTP avec la generation standard greedy (test d'equivalence)")
    var compare: Bool = false

    @Flag(name: .long, help: "Skip le chat template")
    var raw: Bool = false

    @Flag(name: .long, help: "DIAGNOSTIC: utiliser sequential verify (1 token a la fois, lent mais numeriquement equivalent au sequential decode)")
    var sequentialVerify: Bool = false

    @Flag(name: .long, help: "Bypass le MaskedEmbedder a l'inference et utiliser le full lm head (necessaire si drafter est fine-tune sans toucher les centroides)")
    var fullLmHead: Bool = false

    @Option(name: .long, help: "Chemin vers un adapter LoRA a appliquer au target (e.g. pour benchmark MTP avec target fine-tune)")
    var adapterPath: String?

    @Option(name: .long, help: "Token HuggingFace")
    var hfToken: String?

    func run() async throws {
        print("[mtp-generate] target=\(target)")
        print("[mtp-generate] drafter=\(drafter)")
        print("[mtp-generate] prompt=\(prompt.debugDescription)")
        print("[mtp-generate] block_size=\(blockSize)  max_tokens=\(maxTokens)")
        print()

        // Download
        print("[1/3] Download...")
        let targetDir = try await Gemma4ModelDownloader.download(
            modelId: target, token: resolveHFToken(hfToken)
        ) { p in
            print("\r  target: \(Int(p.fraction * 100))% — \(p.currentFile)              ", terminator: "")
            fflush(stdout)
        }
        print()
        let drafterDir = try await Gemma4ModelDownloader.download(
            modelId: drafter, token: resolveHFToken(hfToken)
        ) { p in
            print("\r  drafter: \(Int(p.fraction * 100))% — \(p.currentFile)              ", terminator: "")
            fflush(stdout)
        }
        print()

        // Load
        print("[2/3] Load target + drafter...")
        await Gemma4Registration.register(multimodal: false)
        let container = try await loadModelContainer(from: targetDir, using: Gemma4TokenizerLoader())

        // Apply LoRA adapter if requested
        if let adapter = adapterPath {
            let adapterURL = URL(fileURLWithPath: adapter)
            try await Gemma4LoRAInference.loadAdapter(into: container, from: adapterURL)
            print("  LoRA adapter loaded: \(adapter)")
        }

        let drafterCfgData = try Data(contentsOf: drafterDir.appendingPathComponent("config.json"))
        let drafterCfg = try JSONDecoder().decode(Gemma4AssistantConfig.self, from: drafterCfgData)
        let drafterModel = Gemma4AssistantDraftModel(drafterCfg)

        // Choisir source des poids: override local (drafterPath) ou HF default (drafterDir)
        let weightURLs: [URL]
        if let pathStr = drafterPath {
            let pathURL = URL(fileURLWithPath: pathStr)
            // Si c'est un fichier .safetensors, juste celui-la. Sinon repertoire.
            var isDir: ObjCBool = false
            FileManager.default.fileExists(atPath: pathURL.path, isDirectory: &isDir)
            if isDir.boolValue {
                weightURLs = try FileManager.default
                    .contentsOfDirectory(at: pathURL, includingPropertiesForKeys: nil)
                    .filter { $0.pathExtension == "safetensors" }
            } else {
                weightURLs = [pathURL]
            }
            print("  drafter weights from local path: \(weightURLs.map { $0.lastPathComponent })")
        } else {
            weightURLs = try FileManager.default
                .contentsOfDirectory(at: drafterDir, includingPropertiesForKeys: nil)
                .filter { $0.pathExtension == "safetensors" }
        }

        var rawWeights: [String: MLXArray] = [:]
        for url in weightURLs {
            for (k, v) in try MLX.loadArrays(url: url) { rawWeights[k] = v }
        }
        let sanitized = Gemma4AssistantWeightSanitizer.sanitize(
            weights: rawWeights, tieWordEmbeddings: drafterCfg.tieWordEmbeddings
        )
        try drafterModel.update(parameters: ModuleParameters.unflattened(sanitized), verify: .all)
        drafterModel.useFullLMHead = fullLmHead
        print("  loaded\(fullLmHead ? " (full lm head)" : " (masked_embedding)")")

        // Generate via MTP pipeline
        print()
        print("[3/3] MTP generation...")
        let pipeline = Gemma4MTPPipeline(target: container, drafter: drafterModel)
        let mtpStream = await pipeline.mtpStream(
            prompt: prompt, blockSize: blockSize, maxTokens: maxTokens,
            useChatTemplate: !raw, sequentialVerify: sequentialVerify
        )

        var mtpOutput = ""
        let mtpStart = Date()
        print("MTP> ", terminator: "")
        fflush(stdout)
        for try await piece in mtpStream {
            print(piece, terminator: "")
            fflush(stdout)
            mtpOutput += piece
        }
        let mtpElapsed = Date().timeIntervalSince(mtpStart)
        print()
        let stats = await pipeline.lastStats
        print()
        print("[MTP] \(stats.emittedTokens) tokens en \(String(format: "%.2f", mtpElapsed))s = \(String(format: "%.1f", Double(stats.emittedTokens) / mtpElapsed)) tok/s")
        print("[MTP] rounds=\(stats.rounds), drafts acceptes=\(stats.acceptedDrafts)/\(stats.totalDrafts) (\(Int(stats.acceptRate * 100))%)")

        if !compare {
            print("\n[mtp-generate] DONE")
            return
        }

        // Comparaison: generation standard via ChatSession greedy
        print()
        print("[compare] Generation standard greedy pour equivalence...")
        let stdParams = GenerateParameters(maxTokens: maxTokens, temperature: 0)
        let session = ChatSession(container, instructions: nil, generateParameters: stdParams)
        var stdOutput = ""
        let stdStart = Date()
        print("STD> ", terminator: "")
        fflush(stdout)
        for try await piece in session.streamResponse(to: prompt) {
            print(piece, terminator: "")
            fflush(stdout)
            stdOutput += piece
        }
        let stdElapsed = Date().timeIntervalSince(stdStart)
        print()
        print()
        print("[STD] \(stdOutput.count) chars en \(String(format: "%.2f", stdElapsed))s")

        // Comparison
        print()
        print("=" .padding(toLength: 60, withPad: "=", startingAt: 0))
        print("Comparaison MTP vs STD:")
        if mtpOutput == stdOutput {
            print("  IDENTIQUE (\(mtpOutput.count) chars)")
            print("  Speedup: \(String(format: "%.2fx", stdElapsed / mtpElapsed))")
        } else {
            print("  DIFFERENT")
            print("  MTP: \(mtpOutput.debugDescription)")
            print("  STD: \(stdOutput.debugDescription)")
            // Trouver le 1er point de divergence
            let common = mtpOutput.commonPrefix(with: stdOutput).count
            print("  Prefix commun: \(common) chars")
            if common < mtpOutput.count {
                let idx = mtpOutput.index(mtpOutput.startIndex, offsetBy: common)
                print("  MTP diverge a: \(mtpOutput[idx...].prefix(40).debugDescription)")
            }
            if common < stdOutput.count {
                let idx = stdOutput.index(stdOutput.startIndex, offsetBy: common)
                print("  STD diverge a: \(stdOutput[idx...].prefix(40).debugDescription)")
            }
        }
    }
}
