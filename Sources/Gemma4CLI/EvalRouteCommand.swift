// gemma4-cli eval-route — classification d'intention par generation (banc de routage).
//
// Entree JSON :
//   {"instruction": "...liste des intentions...", "items": [{"id": "...", "text": "..."},
//    {"id": "...", "audio": "/chemin.wav"}, {"id": "...", "text": "...", "instruction": "..."}]}
// Message utilisateur : instruction, ligne vide, puis la requete (texte ou <|audio|>).
// Une ligne JSON par item : sortie brute (greedy), latence, jetons. L'analyse des
// etiquettes est faite par le script du banc.
//
// Texte : l'instruction commune est prefillee une fois, chaque item repart d'une copie
// des caches (comme eval-mmlu). Audio et instructions propres a un item : prefill complet.

import ArgumentParser
import Foundation
import Gemma4Swift
import MLX
import MLXLMCommon

struct EvalRoute: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "eval-route",
        abstract: "Classification d'intention par un LLM Gemma 4 (sortie brute JSONL, banc de routage)"
    )

    @Option(name: .long, help: "Chemin local vers le modele Gemma 4")
    var modelPath: String

    @Option(name: .long, help: "Fichier JSON {instruction, items}")
    var input: String

    @Option(name: .long, help: "Fichier JSONL de sortie (ecrase)")
    var output: String

    @Option(name: .long, help: "Jetons generes au plus par item")
    var maxTokens: Int = 16

    @Option(name: .long, help: "Limite le nombre d'items")
    var limit: Int?

    @Flag(name: .long, help: "Prefill complet a chaque item (pas de cache du prefixe commun)")
    var noPrefixCache = false

    struct Item: Decodable {
        let id: String
        let text: String?
        let audio: String?
        let instruction: String?
    }

    struct Input: Decodable {
        let instruction: String
        let items: [Item]
    }

    func run() async throws {
        let spec = try JSONDecoder().decode(Input.self, from: Data(contentsOf: URL(fileURLWithPath: input)))
        var items = spec.items
        if let limit { items = Array(items.prefix(limit)) }
        let multimodal = items.contains { $0.audio != nil }
        Memory.cacheLimit = 1 << 30

        let start = Date()
        let container = multimodal
            ? try await loadLocalMultimodalModel(path: modelPath) : try await loadLocalModel(path: modelPath)
        let loadSeconds = Date().timeIntervalSince(start)

        FileManager.default.createFile(atPath: output, contents: nil)
        let handle = try FileHandle(forWritingTo: URL(fileURLWithPath: output))
        defer { try? handle.close() }
        func emit(_ fields: [String: Any]) throws {
            var data = try JSONSerialization.data(withJSONObject: fields, options: [.sortedKeys])
            data.append(0x0A)
            try handle.write(contentsOf: data)
        }

        let box = EvalMmlu.PrefixCache()
        var prefixMs = 0.0
        if !multimodal && !noPrefixCache {
            // Prefixe commun = plus long prefixe partage par deux requetes fictives.
            let a = try await tokens(container, content: Self.content(spec.instruction, query: "A"))
            let b = try await tokens(container, content: Self.content(spec.instruction, query: "Z"))
            let common = zip(a, b).prefix { $0 == $1 }.count
            let t0 = Date()
            await container.perform { context in
                guard let llm = context.model as? Gemma4LLMModel else { return }
                let ids = Array(a.prefix(max(1, common - 1)))
                let caches = llm.newCache(parameters: nil)
                _ = llm.languageModel(
                    inputs: MLXArray(ids.map { Int32($0) })[.newAxis], cache: caches.map { $0 as KVCache? },
                    logitsFrom: ids.count - 1)
                eval(caches)
                box.ids = ids
                box.caches = caches
            }
            prefixMs = Date().timeIntervalSince(t0) * 1000
        }
        try emit([
            "point": "header", "model": URL(fileURLWithPath: modelPath).lastPathComponent,
            "load_s": Bench.round(loadSeconds), "prefix_tokens": box.ids.count, "prefix_ms": Bench.round(prefixMs),
            "items": items.count, "prefix_cache": !box.ids.isEmpty,
        ])

        let t0 = Date()
        for (index, item) in items.enumerated() {
            let instruction = item.instruction ?? spec.instruction
            let itemStart = Date()
            let result: (text: String, promptTokens: Int, generated: Int, cached: Bool)
            if let audio = item.audio {
                result = try await generateAudio(container, instruction: instruction, url: URL(fileURLWithPath: audio))
            } else {
                let ids = try await tokens(container, content: Self.content(instruction, query: item.text ?? ""))
                let useCache = item.instruction == nil && !box.ids.isEmpty
                result = try await generateText(container, ids: ids, prefix: useCache ? box : nil)
            }
            let ms = Date().timeIntervalSince(itemStart) * 1000
            try emit([
                "id": item.id, "output": result.text, "ms": Bench.round(ms), "prompt_tokens": result.promptTokens,
                "generated_tokens": result.generated, "cached": result.cached,
            ])
            if index < 3 || (index + 1) % 100 == 0 {
                let elapsed = Date().timeIntervalSince(t0)
                FileHandle.standardError.write(Data(String(
                    format: "[%d/%d] %.0f ms %@ | %.0f s, ETA %.0f s\n", index + 1, items.count, ms,
                    result.text.debugDescription, elapsed,
                    elapsed / Double(index + 1) * Double(items.count - index - 1)).utf8))
            }
        }
    }

    static func content(_ instruction: String, query: String) -> String {
        "\(instruction)\n\nRequete : \(query)"
    }

    private func tokens(_ container: ModelContainer, content: String) async throws -> [Int] {
        let messages: [[String: String]] = [["role": "user", "content": content]]
        return try await container.perform { context in try context.tokenizer.applyChatTemplate(messages: messages) }
    }

    /// Greedy depuis une copie des caches du prefixe commun quand la tokenisation le prolonge.
    private func generateText(
        _ container: ModelContainer, ids: [Int], prefix box: EvalMmlu.PrefixCache?
    ) async throws -> (text: String, promptTokens: Int, generated: Int, cached: Bool) {
        let maxTokens = self.maxTokens
        return try await container.perform { context in
            guard let llm = context.model as? Gemma4LLMModel else { throw ValidationError("modele texte attendu") }
            var caches: [any KVCache]
            var suffix: [Int]
            var cached = false
            if let box, ids.count > box.ids.count, ids.starts(with: box.ids) {
                caches = box.caches.map { $0.copy() }
                suffix = Array(ids[box.ids.count...])
                cached = true
            } else {
                caches = llm.newCache(parameters: nil)
                suffix = ids
            }
            var generated: [Int] = []
            for _ in 0 ..< maxTokens {
                let logits = llm.languageModel(
                    inputs: MLXArray(suffix.map { Int32($0) })[.newAxis], cache: caches.map { $0 as KVCache? },
                    logitsFrom: suffix.count - 1)
                let next = argMax(logits[0, logits.dim(1) - 1, 0...], axis: -1).item(Int32.self)
                if Gemma4Processor.eosTokenIds.contains(next) { break }
                generated.append(Int(next))
                suffix = [Int(next)]
            }
            return (context.tokenizer.decode(tokenIds: generated), ids.count, generated.count, cached)
        }
    }

    /// Instruction puis l'audio, prefill complet (chemin de `describe`).
    private func generateAudio(
        _ container: ModelContainer, instruction: String, url: URL
    ) async throws -> (text: String, promptTokens: Int, generated: Int, cached: Bool) {
        let features = try await Gemma4AudioProcessor.processAudio(url: url)
        let template = try await tokens(container, content: "\(instruction)\n\nRequete : <|audio|>")
        var ids: [Int] = []
        for id in template {
            if id == Int(Gemma4Processor.audioTokenId) {
                ids.append(Int(Gemma4Processor.boaTokenId))
                ids += Array(repeating: Int(Gemma4Processor.audioTokenId), count: features.numTokens)
                ids.append(Int(Gemma4Processor.eoaTokenId))
            } else {
                ids.append(id)
            }
        }
        let promptIds = ids
        let maxTokens = self.maxTokens
        nonisolated(unsafe) let audio = features
        return try await container.perform { context in
            guard let model = context.model as? Gemma4MultimodalLLMModel else {
                throw ValidationError("modele multimodal attendu")
            }
            model.pendingAudioFeatures = audio.features
            model.pendingAudioMask = audio.mask
            let cache = try context.model.newCache(parameters: nil)
            var input = MLXArray(promptIds.map { Int32($0) }).reshaped(1, -1)
            var generated: [Int] = []
            for _ in 0 ..< maxTokens {
                let logits = context.model(input, cache: cache)
                let next = argMax(logits[0, logits.dim(1) - 1, 0...], axis: -1).item(Int32.self)
                if Gemma4Processor.eosTokenIds.contains(next) { break }
                generated.append(Int(next))
                input = MLXArray([next]).reshaped(1, 1)
            }
            return (context.tokenizer.decode(tokenIds: generated), promptIds.count, generated.count, false)
        }
    }
}
