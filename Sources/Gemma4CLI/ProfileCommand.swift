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
        subcommands: [ProfileRun.self, ProfileSweep.self],
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
        session.kvBits = kvBits != nil ? Float(kvBits!) : nil

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

        // Enregistrer le nombre de tokens generes
        session.generatedTokenCount = generatedTokens.count

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

// MARK: - Profile Sweep (context size scaling)

struct ProfileSweep: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "sweep",
        abstract: "Benchmark TurboQuant vs Standard a differentes tailles de contexte"
    )

    @Option(name: .long, help: "ID HuggingFace du modele")
    var model: String = "mlx-community/gemma-4-e2b-it-4bit"

    @Option(name: .long, help: "Chemin local vers le modele")
    var modelPath: String?

    @Option(name: .long, help: "Token HuggingFace")
    var hfToken: String?

    @Option(name: .long, help: "Tailles de contexte (tokens), separees par des virgules")
    var contextSizes: String = "500,1000,2000,4000,8000,16000"

    @Option(name: .long, help: "Configurations KV bits (0=standard), separees par des virgules")
    var kvBitsList: String = "0,4"

    @Option(name: .long, help: "Tokens a generer par run")
    var generatedTokens: Int = 200

    @Option(name: .long, help: "Fichier texte pour remplir le contexte")
    var fillerText: String?

    @Option(name: .long, help: "Fichier CSV de sortie")
    var output: String?

    func run() async throws {
        let sizes = contextSizes.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        let kvConfigs = kvBitsList.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }

        guard !sizes.isEmpty, !kvConfigs.isEmpty else {
            print("Erreur: context-sizes et kv-bits-list ne peuvent pas etre vides")
            throw ExitCode.failure
        }

        // Charger le filler text
        let fillerContent: String
        if let path = fillerText {
            fillerContent = try String(contentsOfFile: path, encoding: .utf8)
        } else {
            // Default: texte repete
            fillerContent = "The development of artificial intelligence has been a long and winding road, spanning decades of research, multiple paradigm shifts, and countless breakthroughs that have transformed how we think about computation, cognition, and the nature of intelligence itself. From the earliest symbolic AI systems to modern transformer architectures, the field has evolved dramatically. "
        }

        let modelId = modelPath ?? model
        let modelName = modelId.split(separator: "/").last.map(String.init) ?? modelId

        print("Enregistrement Gemma 4...")
        await Gemma4Registration.register()

        print("Chargement du modele: \(modelId)")
        let container: ModelContainer
        if let path = modelPath {
            container = try await loadModelContainer(from: URL(fileURLWithPath: path), using: #huggingFaceTokenizerLoader())
        } else {
            container = try await loadModelContainer(
                from: #hubDownloader(makeHubClient(token: hfToken)),
                using: #huggingFaceTokenizerLoader(), id: model
            ) { p in print("\rChargement... \(Int(p.fractionCompleted * 100))%", terminator: ""); fflush(stdout) }
            print()
        }

        print("Modele charge.\n")

        // Resultats
        struct SweepResult {
            let contextTokens: Int
            let kvBits: Int
            let kvConfigName: String
            let throughput: Double
            let ttftMs: Double
            let peakMLXMB: Double
            let peakProcessMB: Double
            let generatedTokens: Int
        }
        var results: [SweepResult] = []

        // Header
        let separator = String(repeating: "\u{2500}", count: 82)
        print("\u{256D}\(separator)\u{256E}")
        print("\u{2502}  TURBOQUANT CONTEXT SWEEP \u{2014} \(modelName)".padding(toLength: 83, withPad: " ", startingAt: 0) + "\u{2502}")
        print("\u{251C}\(separator)\u{2524}")
        print("  Context    Config          tok/s     TTFT      MLX Peak    Process Peak  Tokens")
        print("  \(separator)")

        for targetSize in sizes {
            for kvBits in kvConfigs {
                // Construire le prompt pour atteindre la taille cible
                let prompt = await buildPrompt(
                    targetTokens: targetSize,
                    fillerText: fillerContent,
                    container: container
                )

                let kvBitsFloat: Float? = kvBits > 0 ? Float(kvBits) : nil
                let configName = kvBits > 0 ? "TQ \(kvBits)-bit" : "Standard"

                // Profiler
                let session = ProfilingSession(config: ProfilingConfig(trackMemory: true, exportChromeTrace: false, printSummary: false))
                session.modelVariant = modelName
                session.kvBits = kvBitsFloat

                // Tokeniser
                let messages: [[String: String]] = [
                    ["role": "user", "content": prompt],
                ]
                let tokenIds: [Int] = try await container.perform { context in
                    try context.tokenizer.applyChatTemplate(messages: messages)
                }
                session.promptTokenCount = tokenIds.count

                let inputIds = MLXArray(tokenIds.map { Int32($0) })
                nonisolated(unsafe) let capturedInputIds = inputIds

                // Generation profilee
                let genTokens: [Int] = try await container.perform { context in
                    var tokens: [Int] = []

                    let params = kvBitsFloat != nil ? GenerateParameters(kvBits: Int(kvBitsFloat!)) : nil
                    session.beginPhase("Prefill", category: .prefill)
                    let cache = context.model.newCache(parameters: params)
                    let prefillOutput = context.model(capturedInputIds.reshaped(1, -1), cache: cache)
                    eval(prefillOutput)
                    var nextToken = argMax(prefillOutput[0..., prefillOutput.dim(1) - 1, 0...], axis: -1).item(Int32.self)
                    session.endPhase("Prefill", category: .prefill)

                    session.beginPhase("Generation", category: .generation)
                    for _ in 0 ..< self.generatedTokens {
                        tokens.append(Int(nextToken))
                        if nextToken == 1 || nextToken == 106 { break }
                        let nextInput = MLXArray([nextToken]).reshaped(1, 1)
                        let output = context.model(nextInput, cache: cache)
                        nextToken = argMax(output[0..., 0, 0...], axis: -1).item(Int32.self)
                    }
                    session.endPhase("Generation", category: .generation)
                    return tokens
                }

                session.generatedTokenCount = genTokens.count

                // Extraire les metriques
                let events = session.getEvents()
                let timeline = session.getMemoryTimeline()

                var prefillMs: Double = 0
                var genMs: Double = 0
                var beginTs: [String: UInt64] = [:]
                for event in events {
                    if event.phase == .begin { beginTs[event.name] = event.timestampUs }
                    if event.phase == .end, let bTs = beginTs[event.name] {
                        let dur = Double(event.timestampUs - bTs) / 1000.0
                        if event.name == "Prefill" { prefillMs = dur }
                        if event.name == "Generation" { genMs = dur }
                    }
                }

                let throughput = genMs > 0 ? Double(genTokens.count) / (genMs / 1000.0) : 0
                let peakMLX = timeline.map(\.mlxActiveMB).max() ?? 0
                let peakProcess = timeline.map(\.processFootprintMB).max() ?? 0

                let result = SweepResult(
                    contextTokens: tokenIds.count,
                    kvBits: kvBits,
                    kvConfigName: configName,
                    throughput: throughput,
                    ttftMs: prefillMs,
                    peakMLXMB: peakMLX,
                    peakProcessMB: peakProcess,
                    generatedTokens: genTokens.count
                )
                results.append(result)

                // Afficher la ligne
                let ctxStr = String(format: "%7d", result.contextTokens)
                let cfgStr = configName.padding(toLength: 12, withPad: " ", startingAt: 0)
                let tpsStr = String(format: "%7.1f", result.throughput)
                let ttftStr = formatSweepMs(result.ttftMs)
                let mlxStr = String(format: "%8.1f Go", result.peakMLXMB / 1024)
                let procStr = String(format: "%8.1f Go", result.peakProcessMB / 1024)
                let tokStr = String(format: "%5d", result.generatedTokens)
                print("  \(ctxStr)    \(cfgStr)  \(tpsStr)   \(ttftStr)  \(mlxStr)  \(procStr)  \(tokStr)")

                // Liberer le cache GPU entre les runs
                MLX.GPU.clearCache()
            }
        }

        print("  \(separator)")
        print("\u{2570}\(separator)\u{256F}")

        // Export CSV
        if let csvPath = output {
            var csv = "model,context_tokens,kv_config,kv_bits,throughput_toks,ttft_ms,peak_mlx_mb,peak_process_mb,generated_tokens\n"
            for r in results {
                csv += "\(modelName),\(r.contextTokens),\(r.kvConfigName),\(r.kvBits),"
                csv += "\(String(format: "%.1f", r.throughput)),\(String(format: "%.1f", r.ttftMs)),"
                csv += "\(String(format: "%.1f", r.peakMLXMB)),\(String(format: "%.1f", r.peakProcessMB)),"
                csv += "\(r.generatedTokens)\n"
            }
            try csv.write(toFile: csvPath, atomically: true, encoding: .utf8)
            print("\nCSV: \(csvPath)")
        }
    }

    // Construit un prompt qui approche le nombre de tokens cible
    private func buildPrompt(targetTokens: Int, fillerText: String, container: ModelContainer) async -> String {
        // Tokeniser le filler une fois pour connaitre sa taille
        let fillerTokenCount = await container.perform { context in
            context.tokenizer.encode(text: fillerText).count
        }

        guard fillerTokenCount > 0 else { return fillerText }

        let suffix = "\n\nBased on the above text, provide a comprehensive summary of the key themes and findings."
        let suffixTokens = await container.perform { context in
            context.tokenizer.encode(text: suffix).count
        }

        let repeats = max(1, (targetTokens - suffixTokens) / fillerTokenCount)
        var prompt = ""
        for _ in 0 ..< repeats {
            prompt += fillerText
        }
        prompt += suffix
        return prompt
    }
}

private func formatSweepMs(_ ms: Double) -> String {
    if ms < 1000 {
        return String(format: "%7.0fms", ms)
    } else {
        return String(format: "%6.1fs ", ms / 1000)
    }
}
