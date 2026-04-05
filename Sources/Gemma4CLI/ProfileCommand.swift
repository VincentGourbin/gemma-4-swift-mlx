// Commandes de profiling pour Gemma 4 CLI

import ArgumentParser
import Foundation
import Gemma4Swift
import MLX
import MLXHuggingFace
import MLXLMCommon
import MLXLLM
import HuggingFace
import Tokenizers

struct Profile: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Profiling et benchmark de l'inference Gemma 4",
        subcommands: [ProfileRun.self],
        defaultSubcommand: ProfileRun.self
    )
}

// MARK: - Profile Run

struct ProfileRun: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "run",
        abstract: "Run profile avec trace Chrome Trace et rapport memoire"
    )

    @Option(name: .long, help: "ID HuggingFace du modele")
    var model: String = "mlx-community/gemma-4-e2b-it-4bit"

    @Option(name: .long, help: "Chemin local vers le modele")
    var modelPath: String?

    @Option(name: .long, help: "Token HuggingFace")
    var hfToken: String?

    @Option(name: .long, help: "Prompt a profiler")
    var prompt: String = "Explain the theory of relativity in simple terms."

    @Option(name: .long, help: "Prompt systeme")
    var system: String = "You are a helpful assistant. Be concise."

    @Option(name: .long, help: "Nombre maximum de tokens")
    var maxTokens: Int = 100

    @Option(name: .long, help: "Temperature")
    var temperature: Float = 0.1

    @Flag(name: .long, help: "Tracker la memoire par token")
    var perStepMemory: Bool = false

    @Flag(name: .long, help: "Desactiver l'export Chrome Trace")
    var noChromeTrace: Bool = false

    @Option(name: .long, help: "Bits de quantisation KV cache TurboQuant (3, 4)")
    var kvBits: Int?

    @Option(name: .long, help: "Repertoire de sortie pour les traces")
    var output: String?

    func run() async throws {
        let config = ProfilingConfig(
            trackMemory: true,
            trackPerStepMemory: perStepMemory,
            exportChromeTrace: !noChromeTrace,
            printSummary: true
        )
        let session = ProfilingSession(config: config)

        // Metadata
        let modelId = modelPath ?? model
        session.modelVariant = modelId.split(separator: "/").last.map(String.init) ?? modelId
        session.maxTokens = maxTokens

        // 1. Chargement du modele
        if let kvBits = kvBits {
            session.quantization = "TurboQuant \(kvBits)-bit KV"
        }

        print("Profiling Gemma 4: \(modelId)\(kvBits != nil ? " (TurboQuant \(kvBits!)-bit KV)" : "")")
        session.beginPhase("1. Model Loading", category: .modelLoad)

        await Gemma4Registration.register()

        let container: ModelContainer
        if let path = modelPath {
            let url = URL(fileURLWithPath: path)
            container = try await loadModelContainer(from: url, using: #huggingFaceTokenizerLoader())
        } else {
            container = try await loadModelContainer(
                from: #hubDownloader(makeHubClient(token: hfToken)),
                using: #huggingFaceTokenizerLoader(),
                id: model
            ) { progress in
                print("\rChargement... \(Int(progress.fractionCompleted * 100))%", terminator: "")
                fflush(stdout)
            }
            print()
        }

        session.endPhase("1. Model Loading", category: .modelLoad)

        // 2. Tokenization (via chat template)
        session.beginPhase("2. Tokenization", category: .tokenization)
        let messages: [[String: String]] = [
            ["role": "system", "content": system],
            ["role": "user", "content": prompt],
        ]
        let tokenIds: [Int] = try await container.perform { context in
            try context.tokenizer.applyChatTemplate(messages: messages)
        }
        session.promptTokenCount = tokenIds.count
        session.endPhase("2. Tokenization", category: .tokenization)

        print("Prompt tokens: \(tokenIds.count)")

        // 3. Prefill + Generation
        let inputIds = MLXArray(tokenIds.map { Int32($0) })
        nonisolated(unsafe) let capturedInputIds = inputIds

        let generatedTokens: [Int] = try await container.perform { context in
            var tokens: [Int] = []

            // 3. KV Cache allocation
            session.beginPhase("3. KV Cache Allocation", category: .kvCache)
            let params = self.kvBits != nil
                ? GenerateParameters(kvBits: self.kvBits)
                : nil
            let cache = context.model.newCache(parameters: params)
            session.endPhase("3. KV Cache Allocation", category: .kvCache)

            // 4. Prefill
            session.beginPhase("4. Prefill", category: .prefill)
            let prefillOutput = context.model(capturedInputIds.reshaped(1, -1), cache: cache)
            eval(prefillOutput)
            var nextToken = argMax(prefillOutput[0..., prefillOutput.dim(1) - 1, 0...], axis: -1).item(Int32.self)
            session.endPhase("4. Prefill", category: .prefill)

            // 5. Token generation
            session.beginPhase("5. Token Generation", category: .generation)
            for i in 0 ..< self.maxTokens {
                tokens.append(Int(nextToken))

                // EOS check
                if nextToken == 1 || nextToken == 106 { break }

                let stepStart = CFAbsoluteTimeGetCurrent()

                let nextInput = MLXArray([nextToken]).reshaped(1, 1)
                let output = context.model(nextInput, cache: cache)
                if self.temperature <= 0.01 {
                    nextToken = argMax(output[0..., 0, 0...], axis: -1).item(Int32.self)
                } else {
                    let logits = output[0..., 0, 0...] / self.temperature
                    let probs = softmax(logits, axis: -1)
                    nextToken = MLXRandom.categorical(log(probs)).item(Int32.self)
                }

                let stepDurationUs = UInt64((CFAbsoluteTimeGetCurrent() - stepStart) * 1_000_000)
                session.recordGenerationStep(index: i + 1, total: self.maxTokens, durationUs: stepDurationUs)
            }
            session.endPhase("5. Token Generation", category: .generation)

            return tokens
        }

        // Decoder la reponse
        nonisolated(unsafe) let capturedTokens = generatedTokens
        let response: String = await container.perform { context in
            context.tokenizer.decode(tokenIds: capturedTokens)
        }
        print("\nReponse (\(generatedTokens.count) tokens):\n\(response)\n")

        // Rapport
        if config.printSummary {
            print(session.generateReport())
        }

        // Export Chrome Trace
        if config.exportChromeTrace {
            let traceData = ChromeTraceExporter.export(session: session)
            let outputDir = output.map { URL(fileURLWithPath: $0) }
                ?? URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            let fileName = "gemma4_\(session.modelVariant)_trace.json"
            let traceURL = outputDir.appendingPathComponent(fileName)
            try traceData.write(to: traceURL)
            print("Chrome Trace: \(traceURL.path)")
            print("Ouvrir dans https://ui.perfetto.dev/")
        }
    }
}
