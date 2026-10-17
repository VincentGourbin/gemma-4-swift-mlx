// gemma4-cli bench-embed — latence et memoire de Gemma4Embedder par modalite (une ligne JSON par point)

import ArgumentParser
import Foundation
import Gemma4Swift
import MLX

struct BenchEmbed: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "bench-embed",
        abstract: "Mesure chargement, latence a chaud (mediane, p90) et memoire des embeddings EmbeddingGemma 2"
    )

    @Option(name: .long, help: "Dossier local du checkpoint EmbeddingGemma 2")
    var modelPath: String

    @Option(name: .long, help: "Dossier des medias : photo.jpg, audio_{2,10,30}s.wav, video_{4,16,32}s.mp4")
    var mediaDir: String

    @Option(name: .long, help: "Encodeurs : vision,audio (vide = texte seul, points medias ignores)")
    var encoders: String = "vision,audio"

    @Option(name: .long, help: "Points (virgules) : text,image,audio,video,mixed,throughput ; defaut tous")
    var points: String = "text,image,audio,video,mixed,throughput"

    @Option(name: .long, help: "Passes de chauffe par point (la premiere est rapportee en cold_ms)")
    var warmup: Int = 2

    @Option(name: .customLong("repeat"), help: "Passes mesurees par point")
    var repetitions: Int = 10

    @Option(name: .long, help: "Dimension Matryoshka (defaut native)")
    var dimensions: Int?

    @Option(name: .long, help: "Memory.cacheLimit en Mo (defaut MLX : non plafonne)")
    var cacheLimitMb: Int?

    @Option(name: .long, help: "Etiquette libre recopiee dans chaque ligne (passe, variante)")
    var label: String?

    @Option(name: .long, help: "Fichier JSONL (ajout) ; sinon stdout")
    var output: String?

    func run() async throws {
        let selected = Set(encoders.split(separator: ",").compactMap { name -> Gemma4Embedder.Encoder? in
            switch name.trimmingCharacters(in: .whitespaces) {
            case "vision": return .vision
            case "audio": return .audio
            default: return nil
            }
        })
        let modelURL = URL(fileURLWithPath: modelPath)
        if let cacheLimitMb { Memory.cacheLimit = cacheLimitMb * 1_048_576 }
        Memory.peakMemory = 0
        let start = Date()
        let embedder = try await Gemma4Embedder.load(from: modelURL, encoders: selected)
        let loadSeconds = Date().timeIntervalSince(start)
        var context = BenchContext.collect(modelURL: modelURL, loadSeconds: loadSeconds).fields
        context["encoders"] = embedder.encoders.map { "\($0)" }.sorted().joined(separator: ",")
        context["dimensions"] = dimensions ?? embedder.embeddingDimension
        context["cache_limit_mb"] = cacheLimitMb ?? -1
        if let label { context["label"] = label }
        let footprint = BenchContext.physFootprint()
        try emit(context.merging([
            "point": "load",
            "active_mb": Memory.activeMemory / 1_048_576,
            "peak_mlx_mb": Memory.peakMemory / 1_048_576,
            "phys_footprint_mb": footprint.current,
        ]) { _, new in new })

        let media = URL(fileURLWithPath: mediaDir)
        let wanted = Set(points.split(separator: ",").map { String($0) })
        func file(_ name: String) -> URL { media.appendingPathComponent(name) }

        var cases: [(String, String, [Gemma4Embedder.Part])] = []
        if wanted.contains("text") {
            let sentence = "Le routage deterministe d'actions repose sur quelques exemples par intention. "
            cases.append(("text", "text-short", [.text("Eteins la lumiere du salon.")]))
            // ~15 jetons par phrase ; la longueur exacte est rapportee dans `tokens`.
            for target in [128, 512, 2048, 8000] {
                let text = String(repeating: sentence, count: max(1, target / 15))
                cases.append(("text", "text-\(target)", [.text(text)]))
            }
        }
        if wanted.contains("image"), embedder.encoders.contains(.vision) {
            cases.append(("image", "image-280", [.image(file("photo.jpg"))]))
        }
        if wanted.contains("audio"), embedder.encoders.contains(.audio) {
            for seconds in [2, 10, 30] {
                cases.append(("audio", "audio-\(seconds)s", [.audio(file("audio_\(seconds)s.wav"))]))
            }
        }
        if wanted.contains("video"), embedder.encoders.contains(.vision) {
            for seconds in [4, 16, 32] {
                cases.append(("video", "video-\(seconds)s", [.video(file("video_\(seconds)s.mp4"))]))
            }
        }
        if wanted.contains("mixed"), embedder.encoders == [.vision, .audio] {
            cases.append(("mixed", "text+image+audio", [
                .text("Fiche produit : "), .image(file("photo.jpg")), .text(" Commentaire : "),
                .audio(file("audio_10s.wav")),
            ]))
        }

        for (modality, name, parts) in cases {
            try await measure(embedder, context: context, modality: modality, name: name, items: [parts])
        }
        if wanted.contains("throughput") {
            let verbs = ["Allume", "Eteins", "Ouvre", "Ferme", "Montre", "Lance", "Arrete", "Classe"]
            let objects = ["la lumiere", "le calendrier", "la musique", "la facture", "le minuteur", "les volets",
                           "la meteo", "les messages"]
            let items = (0 ..< 64).map { i -> [Gemma4Embedder.Part] in
                [.text("\(verbs[i % 8]) \(objects[(i / 8) % 8]) s'il te plait")]
            }
            try await measure(embedder, context: context, modality: "text", name: "throughput-64", items: items)
        }
    }

    /// Une passe = tous les `items` embarques a la suite. Les temps sont par item.
    private func measure(
        _ embedder: Gemma4Embedder, context: [String: Any], modality: String, name: String,
        items: [[Gemma4Embedder.Part]]
    ) async throws {
        func pass() async throws -> Double {
            let t0 = Date()
            for item in items {
                _ = try await embedder.embed(item, task: .classification, dimensions: dimensions)
            }
            return Date().timeIntervalSince(t0) * 1000 / Double(items.count)
        }
        Memory.peakMemory = 0
        var cold: Double?
        for _ in 0 ..< warmup {
            let ms = try await pass()
            if cold == nil { cold = ms }
        }
        var samples: [Double] = []
        for _ in 0 ..< repetitions { samples.append(try await pass()) }
        samples.sort()
        let median = samples[samples.count / 2]
        let p90 = samples[min(samples.count - 1, Int((Double(samples.count) * 0.9).rounded(.up)) - 1)]
        let tokens = await embedder.lastTokenCount
        let footprint = BenchContext.physFootprint()
        var line = context
        line["point"] = name
        line["modality"] = modality
        line["items"] = items.count
        line["tokens"] = tokens
        line["cold_ms"] = Bench.round(cold ?? 0)
        line["median_ms"] = Bench.round(median)
        line["p90_ms"] = Bench.round(p90)
        line["min_ms"] = Bench.round(samples.first ?? 0)
        line["items_per_s"] = Bench.round(1000 / median)
        line["tokens_per_s"] = Bench.round(Double(tokens) * 1000 / median)
        line["peak_mlx_mb"] = Memory.peakMemory / 1_048_576
        line["active_mb"] = Memory.activeMemory / 1_048_576
        line["cache_mb"] = Memory.cacheMemory / 1_048_576
        line["phys_footprint_mb"] = footprint.current
        line["phys_footprint_peak_mb"] = footprint.peak
        line["warmup"] = warmup
        line["repeat"] = repetitions
        try emit(line)
    }

    private func emit(_ fields: [String: Any]) throws {
        var data = try JSONSerialization.data(withJSONObject: fields, options: [.sortedKeys])
        data.append(0x0A)
        if let output {
            let url = URL(fileURLWithPath: output)
            if !FileManager.default.fileExists(atPath: output) {
                FileManager.default.createFile(atPath: output, contents: nil)
            }
            let handle = try FileHandle(forWritingTo: url)
            try handle.seekToEnd()
            try handle.write(contentsOf: data)
            try handle.close()
        } else {
            FileHandle.standardOutput.write(data)
        }
        let point = fields["point"] as? String ?? "?"
        let median = fields["median_ms"].map { "\($0) ms" } ?? "\(fields["load_s"] ?? "") s"
        FileHandle.standardError.write(Data("\(point) \(median)\n".utf8))
    }
}
