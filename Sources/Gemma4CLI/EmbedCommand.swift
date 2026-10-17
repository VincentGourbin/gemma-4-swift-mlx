// gemma4-cli embed — embeddings EmbeddingGemma 2 (texte, image, audio, video, entrelaces)

import ArgumentParser
import Foundation
import Gemma4Swift
import MLX

struct Embed: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Embeddings EmbeddingGemma 2 : une entree en ligne de commande, ou un lot JSON"
    )

    @Option(name: .long, help: "Dossier local du checkpoint (google/embeddinggemma-2 ou conversion mlx-community)")
    var modelPath: String

    @Option(name: .long, help: "Texte (premier morceau de l'entree)")
    var text: String?

    @Option(name: .long, help: "Image (morceaux apres le texte, dans l'ordre image, video, audio)")
    var image: [String] = []

    @Option(name: .long, help: "Video")
    var video: [String] = []

    @Option(name: .long, help: "Audio (mono 16 kHz de preference)")
    var audio: [String] = []

    @Option(name: .long, help: "Tache : none, classification, clustering, similarity, query, qa, fact, code, document")
    var task: String = "none"

    @Option(name: .long, help: "Dimension Matryoshka (128, 256, 512, 768)")
    var dimensions: Int?

    @Option(name: .long, help: "Encodeurs a charger : vision,audio (vide = texte seul)")
    var encoders: String = "vision,audio"

    @Option(name: .long, help: "Lot JSON {nom: {task, parts: [{text|image|audio|video: ...}]}} ; ecrit --output")
    var batch: String?

    @Option(name: .long, help: "Sortie JSON du lot {nom: [Float]}, plus _timings_ms")
    var output: String?

    @Option(name: .long, help: "Diagnostic : ecrit les pixels de la premiere --image, sinon les log-mel du premier --audio (float32 brut), et sort")
    var dumpPreprocessed: String?

    func run() async throws {
        if let dumpPreprocessed, let first = image.first {
            let pixels = try Gemma4ImageProcessor.processImage(
                url: URL(fileURLWithPath: first), maxSoftTokens: Gemma4Embedder.imageSoftTokens,
                resampling: .pillowBicubic)
            let values = pixels.asType(.float32).asArray(Float.self)
            try values.withUnsafeBufferPointer { Data(buffer: $0) }.write(to: URL(fileURLWithPath: dumpPreprocessed))
            print("pixels \(pixels.shape)")
            return
        }
        if let dumpPreprocessed, let first = audio.first {
            let features = try await Gemma4AudioProcessor.processAudio(url: URL(fileURLWithPath: first)).features
            let values = features.asType(.float32).asArray(Float.self)
            try values.withUnsafeBufferPointer { Data(buffer: $0) }.write(to: URL(fileURLWithPath: dumpPreprocessed))
            print("mel \(features.shape)")
            return
        }
        let selected = Set(encoders.split(separator: ",").compactMap { name -> Gemma4Embedder.Encoder? in
            switch name.trimmingCharacters(in: .whitespaces) {
            case "vision": return .vision
            case "audio": return .audio
            default: return nil
            }
        })
        let start = Date()
        let embedder = try await Gemma4Embedder.load(from: URL(fileURLWithPath: modelPath), encoders: selected)
        let loadMs = Date().timeIntervalSince(start) * 1000
        Self.log(
            "charge en \(Int(loadMs)) ms, encodeurs \(embedder.encoders.map { "\($0)" }.sorted()), "
                + "GPU actif \(Memory.activeMemory / 1_048_576) Mo\n")

        if let batch {
            try await runBatch(embedder, spec: batch, loadMs: loadMs)
            return
        }
        var parts: [Gemma4Embedder.Part] = []
        if let text { parts.append(.text(text)) }
        parts += image.map { .image(URL(fileURLWithPath: $0)) }
        parts += video.map { .video(URL(fileURLWithPath: $0)) }
        parts += audio.map { .audio(URL(fileURLWithPath: $0)) }
        let t0 = Date()
        let vector = try await embedder.embed(parts, task: try Self.task(task), dimensions: dimensions)
        let ms = Date().timeIntervalSince(t0) * 1000
        let head = vector.prefix(6).map { String(format: "%.4f", $0) }.joined(separator: ", ")
        print("dim \(vector.count), \(String(format: "%.1f", ms)) ms, [\(head), ...]")
    }

    private func runBatch(_ embedder: Gemma4Embedder, spec path: String, loadMs: Double) async throws {
        struct Entry: Decodable {
            let task: String?
            let parts: [[String: String]]
        }
        let entries = try JSONDecoder().decode(
            [String: Entry].self, from: Data(contentsOf: URL(fileURLWithPath: path)))
        var vectors: [String: [Float]] = [:]
        var timings: [String: Double] = ["_load": loadMs]
        for name in entries.keys.sorted() {
            let entry = entries[name]!
            let parts = try entry.parts.map { part -> Gemma4Embedder.Part in
                guard let (kind, value) = part.first else { throw ValidationError("\(name) : morceau vide") }
                switch kind {
                case "text": return .text(value)
                case "image": return .image(URL(fileURLWithPath: value))
                case "audio": return .audio(URL(fileURLWithPath: value))
                case "video": return .video(URL(fileURLWithPath: value))
                default: throw ValidationError("\(name) : morceau inconnu \(kind)")
                }
            }
            let t0 = Date()
            vectors[name] = try await embedder.embed(parts, task: try Self.task(entry.task ?? "none"), dimensions: dimensions)
            timings[name] = Date().timeIntervalSince(t0) * 1000
            Self.log("\(name) \(String(format: "%.1f", timings[name]!)) ms\n")
        }
        var json: [String: Any] = vectors
        json["_timings_ms"] = timings
        json["_peak_mb"] = Memory.peakMemory / 1_048_576
        let data = try JSONSerialization.data(withJSONObject: json)
        if let output {
            try data.write(to: URL(fileURLWithPath: output))
        } else {
            FileHandle.standardOutput.write(data)
        }
    }

    static func log(_ message: String) {
        FileHandle.standardError.write(Data(message.utf8))
    }

    static func task(_ name: String) throws -> Gemma4Embedder.Task {
        switch name {
        case "none": return .none
        case "classification": return .classification
        case "clustering": return .clustering
        case "similarity": return .sentenceSimilarity
        case "query": return .searchQuery
        case "qa": return .questionAnswering
        case "fact": return .factChecking
        case "code": return .codeRetrieval
        case "document": return .document(title: nil)
        default: throw ValidationError("tache inconnue : \(name)")
        }
    }
}
