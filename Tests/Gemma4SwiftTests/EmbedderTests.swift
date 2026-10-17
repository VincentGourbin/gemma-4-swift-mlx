import Testing
import Foundation
import MLX
@testable import Gemma4Swift

@Suite("Embedder : logique pure")
struct EmbedderLogicTests {

    @Test("prefixe de tache rejoint le premier texte")
    func promptedJoinsFirstText() {
        let parts = Gemma4Embedder.prompted(
            [.text("Eteins la lumiere"), .image(URL(fileURLWithPath: "/x.png"))], task: .classification)
        guard case .text(let first) = parts[0] else { Issue.record("premier morceau non textuel"); return }
        #expect(first == "task: classification | query: Eteins la lumiere")
        #expect(parts.count == 2)
    }

    @Test("media seul : pas de prefixe")
    func mediaOnlyHasNoPrefix() {
        let parts = Gemma4Embedder.prompted([.audio(URL(fileURLWithPath: "/a.wav"))], task: .searchQuery)
        #expect(parts.count == 1)
        if case .text = parts[0] { Issue.record("prefixe ajoute a un media seul") }
    }

    @Test("media puis texte : le prefixe ouvre la sequence")
    func mediaThenTextGetsLeadingPrefix() {
        let parts = Gemma4Embedder.prompted(
            [.image(URL(fileURLWithPath: "/x.png")), .text("legende")], task: .classification)
        #expect(parts.count == 3)
        if case .text(let first) = parts[0] { #expect(first == "task: classification | query: ") }
    }

    @Test("texte deja prefixe inchange")
    func existingPrefixKept() {
        let parts = Gemma4Embedder.prompted([.text("title: Doc | text: corps")], task: .classification)
        if case .text(let first) = parts[0] { #expect(first == "title: Doc | text: corps") }
    }

    @Test("prefixes de la model card")
    func modelCardPrefixes() {
        #expect(Gemma4Embedder.Task.searchQuery.prefix() == "task: search result | query: ")
        #expect(Gemma4Embedder.Task.document(title: nil).prefix() == "title: none | text: ")
        #expect(Gemma4Embedder.Task.document(title: "  Mon   titre ").prefix() == "title: Mon titre | text: ")
        #expect(Gemma4Embedder.Task.none.prefix() == "")
    }

    @Test("frames a 1 fps, plafond uniforme a 32 comme np.linspace")
    func videoFrameTimes() {
        #expect(Gemma4Embedder.frameTimes(duration: 4.0) == [0, 1, 2, 3])
        #expect(Gemma4Embedder.frameTimes(duration: 0.4) == [0])
        let long = Gemma4Embedder.frameTimes(duration: 100)
        #expect(long.count == 32)
        #expect(long.first == 0)
        #expect(long.last == 99)
        // np.linspace(0, 99, 32, dtype=int)[1] == 3
        #expect(long[1] == 3)
    }

    @Test("troncature Matryoshka renormalisee")
    func matryoshkaTruncation() throws {
        let values = (0 ..< 768).map { Float($0 % 7) - 3 }
        let cut = try Gemma4Embedder.truncated(values, to: 256)
        #expect(cut.count == 256)
        let norm = cut.reduce(0) { $0 + Double($1) * Double($1) }.squareRoot()
        #expect(abs(norm - 1) < 1e-5)
        #expect(throws: Gemma4Embedder.EmbedderError.invalidEmbedding) {
            try Gemma4Embedder.truncated([Float](repeating: 0, count: 8), to: 8)
        }
        #expect(throws: Gemma4Embedder.EmbedderError.invalidEmbedding) {
            try Gemma4Embedder.truncated([.nan, 1], to: 2)
        }
    }

    @Test("jetons speciaux lus dans config.json (eoa_token_index)")
    func specialTokensDecoding() throws {
        let json = """
            {"image_token_id": 1, "video_token_id": 2, "audio_token_id": 3, "boi_token_id": 4,
             "eoi_token_id": 5, "boa_token_id": 6, "eoa_token_index": 7,
             "vision_config": {"hidden_size": 768}, "audio_config": null}
            """
        let tokens = try JSONDecoder().decode(Gemma4Embedder.SpecialTokens.self, from: Data(json.utf8))
        #expect(tokens.eoaTokenId == 7)
        #expect(tokens.hasVision)
        #expect(!tokens.hasAudio)
        #expect(tokens.media == [1, 2, 3])
    }
}

/// Parite contre sentence-transformers (fp32, `Fixtures/embeddinggemma2-reference.json`) et
/// proprietes de l'espace. Vrai modele :
///   GEMMA4_EMBEDDING_MODEL_PATH=/Volumes/Lexar/models/google/embeddinggemma-2 \
///     Scripts/run-tests.sh -only-testing:Gemma4SwiftTests/EmbedderIntegrationTests
private let embeddingModelPath = ProcessInfo.processInfo.environment["GEMMA4_EMBEDDING_MODEL_PATH"]

@Suite("Embedder : EmbeddingGemma 2", .serialized, .enabled(if: embeddingModelPath != nil))
struct EmbedderIntegrationTests {

    struct Reference: Decodable {
        struct Case: Decodable {
            let name: String
            let text: String
            let vector: [Float]
        }
        let cases: [Case]
    }

    static func cosine(_ a: [Float], _ b: [Float]) -> Double {
        let dot = zip(a, b).reduce(0.0) { $0 + Double($1.0) * Double($1.1) }
        let na = a.reduce(0.0) { $0 + Double($1) * Double($1) }.squareRoot()
        let nb = b.reduce(0.0) { $0 + Double($1) * Double($1) }.squareRoot()
        return dot / (na * nb)
    }

    @Test("texte : parite sentence-transformers et determinisme")
    func textParity() async throws {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appendingPathComponent("Fixtures/embeddinggemma2-reference.json")
        let reference = try JSONDecoder().decode(Reference.self, from: Data(contentsOf: url))
        let embedder = try await Gemma4Embedder.load(
            from: URL(fileURLWithPath: embeddingModelPath!), encoders: [])
        #expect(embedder.encoders.isEmpty)
        for item in reference.cases {
            let vector = try await embedder.embed([.text(item.text)], task: .none)
            let again = try await embedder.embed([.text(item.text)], task: .none)
            #expect(vector == again, "\(item.name) non deterministe")
            // bf16 : >= 0,9997 mesure ; 8 bits >= 0,9993 ; 4 bits ~0,97 (seuil non tenu).
            #expect(Self.cosine(vector, item.vector) > 0.999, "\(item.name)")
        }
    }

    @Test("espace : paraphrase plus proche qu'une autre action, en 768 et 256 dims")
    func routingGeometry() async throws {
        let embedder = try await Gemma4Embedder.load(
            from: URL(fileURLWithPath: embeddingModelPath!), encoders: [])
        for dimensions in [768, 256] {
            let action = try await embedder.embed(
                [.text("Quels sont mes rendez-vous demain ?")], task: .classification, dimensions: dimensions)
            let paraphrase = try await embedder.embed(
                [.text("J'ai quoi au programme jeudi ?")], task: .classification, dimensions: dimensions)
            let other = try await embedder.embed(
                [.text("Mets un minuteur de 10 minutes")], task: .classification, dimensions: dimensions)
            #expect(action.count == dimensions)
            #expect(Self.cosine(action, paraphrase) > Self.cosine(action, other) + 0.1)
        }
        await #expect(throws: Gemma4Embedder.EmbedderError.invalidDimension(300)) {
            _ = try await embedder.embed([.text("x")], task: .none, dimensions: 300)
        }
        await #expect(throws: Gemma4Embedder.EmbedderError.unsupportedModality("image")) {
            _ = try await embedder.embed([.image(URL(fileURLWithPath: "/nonexistent.png"))], task: .none)
        }
    }

    @Test("image : vecteur unitaire, document image plus proche d'une capture d'ecran que d'un minuteur")
    func imageEmbedding() async throws {
        let embedder = try await Gemma4Embedder.load(
            from: URL(fileURLWithPath: embeddingModelPath!), encoders: [.vision])
        let ui = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("docs/examples/vision-image-description/UI.png")
        let image = try await embedder.embed([.image(ui)], task: .none)
        let norm = image.reduce(0.0) { $0 + Double($1) * Double($1) }.squareRoot()
        #expect(abs(norm - 1) < 1e-4)
        let screenshot = try await embedder.embed(
            [.text("a screenshot of a software user interface")], task: .classification)
        let timer = try await embedder.embed([.text("set a timer for ten minutes")], task: .classification)
        #expect(Self.cosine(image, screenshot) > Self.cosine(image, timer))
    }
}
