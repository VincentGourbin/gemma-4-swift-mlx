// Commandes de profiling pour Gemma 4 CLI

import ArgumentParser
import Foundation
import Gemma4Swift
import MLX
import MLXLMCommon
import MLXLLM
import MLXProfiler
import Tokenizers
import SQLite3

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
        let modelName = modelId.split(separator: "/").last.map(String.init) ?? modelId
        session.metadata["model"] = modelName
        session.metadata["maxTokens"] = "\(maxTokens)"
        if let kvBits = kvBits {
            session.metadata["kvBits"] = "\(kvBits)"
            session.metadata["quantization"] = "TurboQuant \(kvBits)-bit KV"
        }

        print("Profiling Gemma 4: \(modelId)\(kvBits != nil ? " (TurboQuant \(kvBits!)-bit KV)" : "")")
        session.beginPhase("1. Model Loading", category: .modelLoad)

        guard let path = modelPath else {
            print("Erreur: --model-path requis.")
            throw ExitCode.failure
        }
        let container = try await loadLocalModel(path: path)

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
        session.metadata["promptTokenCount"] = "\(tokenIds.count)"
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
                session.recordStep(index: i + 1, total: self.maxTokens, durationUs: stepDurationUs)
            }
            session.endPhase("5. Token Generation", category: .generation)

            return tokens
        }

        // Enregistrer le nombre de tokens generes
        session.metadata["generatedTokenCount"] = "\(generatedTokens.count)"

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
            let fileName = "gemma4_\(session.metadata["model"] ?? "gemma4")_trace.json"
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

    @Option(name: .long, help: "Fichier JSONL de sortie (append-safe, reprend ou il en etait)")
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

        guard let path = modelPath else {
            print("Erreur: --model-path requis.")
            throw ExitCode.failure
        }
        print("Chargement du modele: \(path)")
        let container = try await loadLocalModel(path: path)

        // Lire le max_position_embeddings depuis la config du modele
        let maxContextSize = await container.perform { context -> Int in
            if let gemmaModel = context.model as? Gemma4LLMModel {
                return gemmaModel.config.maxPositionEmbeddings
            }
            if let multimodalModel = context.model as? Gemma4MultimodalLLMModel {
                return multimodalModel.config.textConfig.maxPositionEmbeddings
            }
            return 131072 // fallback
        }

        // Memoire systeme totale (unified memory = GPU memory sur Apple Silicon)
        let systemRAMBytes = ProcessInfo.processInfo.physicalMemory
        let systemRAMGB = String(format: "%.0f", Double(systemRAMBytes) / 1_073_741_824)

        print("Modele charge.")
        print("  Contexte max: \(maxContextSize) tokens")
        print("  RAM systeme: \(systemRAMGB) Go (unified memory)")
        print()

        // Base SQLite pour les resultats (resume-safe, queryable)
        let db = output != nil ? SweepDatabase(path: output!) : nil
        let completedRuns = db?.loadCompletedRuns(model: modelName) ?? []
        if !completedRuns.isEmpty {
            print("  Reprise: \(completedRuns.count) runs deja effectues, skip")
        }

        // Lire le max context et la memoire
        let maxCtx = await container.perform { context -> Int in
            if let m = context.model as? Gemma4LLMModel { return m.config.maxPositionEmbeddings }
            if let m = context.model as? Gemma4MultimodalLLMModel { return m.config.textConfig.maxPositionEmbeddings }
            return 131072
        }
        let modelBaseMemMB = Double(MLX.GPU.activeMemory) / (1024 * 1024)

        // Header
        let separator = String(repeating: "\u{2500}", count: 82)
        print("\u{256D}\(separator)\u{256E}")
        let headerLine = "\u{2502}  TURBOQUANT CONTEXT SWEEP \u{2014} \(modelName) (max ctx: \(maxCtx))"
        print(headerLine.padding(toLength: 83, withPad: " ", startingAt: 0) + "\u{2502}")
        print("\u{251C}\(separator)\u{2524}")
        print("  Context    Config          tok/s     TTFT      MLX Peak    Process Peak  Gen   Total")
        print("  \(separator)")

        // Metal buffer max = RAM totale (Apple Silicon unified memory)
        let metalMaxBytes = ProcessInfo.processInfo.physicalMemory
        let metalMaxGB = Double(metalMaxBytes) / 1_073_741_824
        // Garder 20% de marge pour l'OS et les process overheads
        let safeMaxGB = metalMaxGB * 0.80
        var lastPeakProcessMB: Double = 0

        var skipRemaining = false
        for targetSize in sizes {
            if skipRemaining { break }

            // Protection: verifier que le contexte ne depasse pas le max du modele
            if targetSize > maxCtx {
                print("  \(String(format: "%7d", targetSize))    SKIP (depasse max_position_embeddings = \(maxCtx))")
                continue
            }

            // Protection memoire basee sur le pic reel du run precedent
            // Si le dernier run a utilise >70% de la RAM, on ne tente pas un contexte plus grand
            if lastPeakProcessMB > 0 {
                let lastPeakGB = lastPeakProcessMB / 1024
                if lastPeakGB > safeMaxGB {
                    print("  \(String(format: "%7d", targetSize))    SKIP (pic precedent \(String(format: "%.1f", lastPeakGB)) Go > safe max \(String(format: "%.1f", safeMaxGB)) Go)")
                    skipRemaining = true
                    continue
                }
            }

            for kvBits in kvConfigs {
                // Reprise : skip si deja fait (tolerance 10% sur la taille du contexte)
                let alreadyDone = completedRuns.contains { r in
                    r.kvBits == kvBits && abs(r.contextTokens - targetSize) < max(50, targetSize / 10)
                }
                if alreadyDone {
                    let cfgLabel = kvBits > 0 ? "TQ \(kvBits)-bit" : "Standard"
                    print("  \(String(format: "%7d", targetSize))    \(cfgLabel.padding(toLength: 12, withPad: " ", startingAt: 0))  [deja fait, skip]")
                    continue
                }

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
                session.metadata["model"] = modelName
                if let kvBitsFloat {
                    session.metadata["kvBits"] = "\(kvBitsFloat)"
                }

                // Tokeniser
                let messages: [[String: String]] = [
                    ["role": "user", "content": prompt],
                ]
                let tokenIds: [Int] = try await container.perform { context in
                    try context.tokenizer.applyChatTemplate(messages: messages)
                }
                session.metadata["promptTokenCount"] = "\(tokenIds.count)"

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

                session.metadata["generatedTokenCount"] = "\(genTokens.count)"

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

                // Inserer dans la base SQLite (transactionnel, resume-safe)
                db?.insert(
                    model: modelName,
                    contextTokens: tokenIds.count,
                    generatedTokens: genTokens.count,
                    totalTokens: tokenIds.count + genTokens.count,
                    kvBits: kvBits,
                    kvConfigName: configName,
                    throughput: throughput,
                    ttftMs: prefillMs,
                    peakMLXMB: peakMLX,
                    peakProcessMB: peakProcess,
                    deviceArchitecture: GPU.deviceInfo().architecture,
                    systemRAMGB: Int(ProcessInfo.processInfo.physicalMemory / (1024 * 1024 * 1024))
                )

                // Mettre a jour le pic memoire reel pour la protection du prochain run
                lastPeakProcessMB = max(lastPeakProcessMB, peakProcess)

                // Afficher la ligne
                let ctxStr = String(format: "%7d", tokenIds.count)
                let cfgStr = configName.padding(toLength: 12, withPad: " ", startingAt: 0)
                let tpsStr = String(format: "%7.1f", throughput)
                let ttftStr = formatSweepMs(prefillMs)
                let mlxStr = String(format: "%8.1f Go", peakMLX / 1024)
                let procStr = String(format: "%8.1f Go", peakProcess / 1024)
                let genStr = String(format: "%4d", genTokens.count)
                let totalStr = String(format: "%6d", tokenIds.count + genTokens.count)
                print("  \(ctxStr)    \(cfgStr)  \(tpsStr)   \(ttftStr)  \(mlxStr)  \(procStr)  \(genStr)  \(totalStr)")

                // Liberer le cache GPU entre les runs
                MLX.GPU.clearCache()
            }
        }

        print("  \(separator)")
        print("\u{2570}\(separator)\u{256F}")

        // Résumé
        if let outputPath = output {
            let count = db?.totalCount() ?? 0
            print("\nSQLite: \(outputPath) (\(count) runs total)")
            print("Query: sqlite3 \(outputPath) \"SELECT * FROM sweep_results ORDER BY context_tokens, kv_bits\"")
        }
    }

    // Construit un prompt qui approche le nombre de tokens cible
    private func buildPrompt(targetTokens: Int, fillerText: String, container: ModelContainer) async -> String {
        let suffix = "\n\nBased on the above text, provide a comprehensive summary of the key themes and findings."

        // Tokeniser le filler et le suffix
        let fillerTokens: [Int] = await container.perform { context in
            context.tokenizer.encode(text: fillerText)
        }
        let suffixTokens: Int = await container.perform { context in
            context.tokenizer.encode(text: suffix).count
        }

        guard !fillerTokens.isEmpty else { return fillerText + suffix }

        let availableForFiller = max(1, targetTokens - suffixTokens)

        if fillerTokens.count <= availableForFiller {
            // Filler trop court : repliquer
            let repeats = max(1, availableForFiller / fillerTokens.count)
            var prompt = ""
            for _ in 0 ..< repeats { prompt += fillerText }
            prompt += suffix
            return prompt
        } else {
            // Filler trop long : tronquer au nombre de tokens cible
            let truncatedTokens = Array(fillerTokens.prefix(availableForFiller))
            let truncatedText: String = await container.perform { context in
                context.tokenizer.decode(tokenIds: truncatedTokens)
            }
            return truncatedText + suffix
        }
    }
}

private func formatSweepMs(_ ms: Double) -> String {
    if ms < 1000 {
        return String(format: "%7.0fms", ms)
    } else {
        return String(format: "%6.1fs ", ms / 1000)
    }
}

// MARK: - SQLite Storage for Sweep Results

/// Base SQLite pour stocker les resultats de benchmark (resume-safe, queryable)
final class SweepDatabase {
    private var db: OpaquePointer?

    struct CompletedRun {
        let contextTokens: Int
        let kvBits: Int
    }

    init(path: String) {
        guard sqlite3_open(path, &db) == SQLITE_OK else {
            print("  ⚠ SQLite: impossible d'ouvrir \(path)")
            return
        }
        createTable()
    }

    deinit {
        sqlite3_close(db)
    }

    private func createTable() {
        let sql = """
            CREATE TABLE IF NOT EXISTS sweep_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model TEXT NOT NULL,
                context_tokens INTEGER NOT NULL,
                generated_tokens INTEGER NOT NULL,
                total_tokens INTEGER NOT NULL,
                kv_bits INTEGER NOT NULL,
                kv_config_name TEXT NOT NULL,
                throughput_toks REAL NOT NULL,
                ttft_ms REAL NOT NULL,
                peak_mlx_mb REAL NOT NULL,
                peak_process_mb REAL NOT NULL,
                device_architecture TEXT NOT NULL,
                system_ram_gb INTEGER NOT NULL,
                timestamp TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """
        exec(sql)
    }

    func insert(
        model: String, contextTokens: Int, generatedTokens: Int, totalTokens: Int,
        kvBits: Int, kvConfigName: String, throughput: Double, ttftMs: Double,
        peakMLXMB: Double, peakProcessMB: Double, deviceArchitecture: String, systemRAMGB: Int
    ) {
        let sql = """
            INSERT INTO sweep_results
            (model, context_tokens, generated_tokens, total_tokens, kv_bits, kv_config_name,
             throughput_toks, ttft_ms, peak_mlx_mb, peak_process_mb, device_architecture, system_ram_gb)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { return }
        defer { sqlite3_finalize(stmt) }

        sqlite3_bind_text(stmt, 1, (model as NSString).utf8String, -1, nil)
        sqlite3_bind_int(stmt, 2, Int32(contextTokens))
        sqlite3_bind_int(stmt, 3, Int32(generatedTokens))
        sqlite3_bind_int(stmt, 4, Int32(totalTokens))
        sqlite3_bind_int(stmt, 5, Int32(kvBits))
        sqlite3_bind_text(stmt, 6, (kvConfigName as NSString).utf8String, -1, nil)
        sqlite3_bind_double(stmt, 7, throughput)
        sqlite3_bind_double(stmt, 8, ttftMs)
        sqlite3_bind_double(stmt, 9, peakMLXMB)
        sqlite3_bind_double(stmt, 10, peakProcessMB)
        sqlite3_bind_text(stmt, 11, (deviceArchitecture as NSString).utf8String, -1, nil)
        sqlite3_bind_int(stmt, 12, Int32(systemRAMGB))

        sqlite3_step(stmt)
    }

    /// Charge les runs deja effectues pour un modele (pour la reprise)
    func loadCompletedRuns(model: String) -> [CompletedRun] {
        let sql = "SELECT context_tokens, kv_bits FROM sweep_results WHERE model = ?"
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { return [] }
        defer { sqlite3_finalize(stmt) }

        sqlite3_bind_text(stmt, 1, (model as NSString).utf8String, -1, nil)

        var runs: [CompletedRun] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            runs.append(CompletedRun(
                contextTokens: Int(sqlite3_column_int(stmt, 0)),
                kvBits: Int(sqlite3_column_int(stmt, 1))
            ))
        }
        return runs
    }

    func totalCount() -> Int {
        let sql = "SELECT COUNT(*) FROM sweep_results"
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { return 0 }
        defer { sqlite3_finalize(stmt) }
        return sqlite3_step(stmt) == SQLITE_ROW ? Int(sqlite3_column_int(stmt, 0)) : 0
    }

    private func exec(_ sql: String) {
        sqlite3_exec(db, sql, nil, nil, nil)
    }
}
