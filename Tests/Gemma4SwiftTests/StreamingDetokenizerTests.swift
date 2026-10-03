import Testing
import Foundation
import CoreGraphics
import MLXLMCommon
@testable import Gemma4Swift

/// Garde-fou S-11 (audit du 2026-09-27) : la detokenisation incrementale doit
/// reconstituer exactement le decodage complet. Ne charge que le tokenizer du
/// modele d'integration (pas de poids, pas de GPU).
private let integrationModelPath = ProcessInfo.processInfo
    .environment["GEMMA4_INTEGRATION_MODEL_PATH"]

@Suite("Detokenisation incrementale", .enabled(if: integrationModelPath != nil))
struct StreamingDetokenizerTests {

    /// Repli octet par octet (𠀋, 𝔘𝔫𝔦), drapeau (2 indicateurs regionaux), sequence
    /// ZWJ, accent combinant (e + U+0301), texte courant et fins de ligne.
    static let text = "Voiture 🇫🇷 👩‍👩‍👧 𠀋 𝔘𝔫𝔦 cafe\u{301} naïveté\nligne 2 🍓\nfin"

    private func tokenizer() async throws -> any Tokenizer {
        try await Gemma4TokenizerLoader().load(from: URL(fileURLWithPath: integrationModelPath!))
    }

    @Test("Gemma4StreamingDetokenizer reconstitue le decodage complet")
    func testReconstructsFullDecode() async throws {
        let tokenizer = try await tokenizer()
        let ids = tokenizer.encode(text: Self.text, addSpecialTokens: false)
        var detokenizer = Gemma4StreamingDetokenizer(tokenizer: tokenizer)
        let streamed = ids.compactMap { detokenizer.append(token: $0) }.joined()
        #expect(streamed == tokenizer.decode(tokenIds: ids))
    }

    @Test("decoder token par token perd les caracteres en repli octet par octet")
    func testPerTokenDecodeIsLossy() async throws {
        let tokenizer = try await tokenizer()
        let ids = tokenizer.encode(text: Self.text, addSpecialTokens: false)
        let perToken = ids.map { tokenizer.decode(tokenIds: [$0]) }.joined()
        #expect(perToken != tokenizer.decode(tokenIds: ids))
    }

    /// Bug amont (mlx-swift-lm, `NaiveStreamingDetokenizer.next()` : difference par
    /// `String.count`, donc par graphemes), corrige en 3.32 (mlx-swift-lm#613, prefixe
    /// commun par scalaires). Garde-fou de non-regression amont.
    @Test("NaiveStreamingDetokenizer amont garde les scalaires qui fusionnent en grapheme")
    func testUpstreamDetokenizerKeepsScalars() async throws {
        let tokenizer = try await tokenizer()
        let ids = tokenizer.encode(text: Self.text, addSpecialTokens: false)
        var upstream = NaiveStreamingDetokenizer(tokenizer: tokenizer)
        var streamed = ""
        for id in ids {
            upstream.append(token: id)
            if let piece = upstream.next() { streamed += piece }
        }
        #expect(streamed == tokenizer.decode(tokenIds: ids))
    }

    /// Bout en bout : les chemins du pipeline qui generent via TokenIterator
    /// (multimodal, n-gramme) detokenisent eux-memes et rendent le texte intact.
    @Test("chatStreamMultimodal et chemin n-gramme restituent drapeau, ZWJ et repli octet")
    @MainActor
    func testPipelinePathsKeepScalars() async throws {
        let pipeline = Gemma4Pipeline()
        try await pipeline.load(from: URL(fileURLWithPath: integrationModelPath!), multimodal: true)
        let target = "🇫🇷 👩‍👩‍👧 𠀋 𝔘𝔫𝔦"
        let prompt = "Recopie exactement, sans rien d'autre : \(target)"

        let ctx = try #require(CGContext(
            data: nil, width: 64, height: 64, bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
        let pixels = try Gemma4ImageProcessor.processImage(try #require(ctx.makeImage()))
        var multimodal = ""
        for try await chunk in try pipeline.chatStreamMultimodal(
            prompt: prompt, pixelValues: pixels, temperature: 0, maxTokens: 40) { multimodal += chunk }
        #expect(multimodal.contains(target), "multimodal : \(multimodal)")

        var ngram = ""
        for try await chunk in try pipeline.chatStream(
            prompt: prompt, temperature: 0, maxTokens: 40, noRepeatNGramSize: 8,
            // Fenetre hors prompt : sinon le n-gramme interdit justement la recopie.
            noRepeatNGramIncludesPrompt: false) { ngram += chunk }
        // Ce qui departage les detokeniseurs : l'amont rendrait « 🇫 👩 ». La fidelite de
        // la recopie complete depend du modele (𠀋 parfois omis sur ce chemin).
        #expect(ngram.contains("🇫🇷") && ngram.contains("👩‍👩‍👧"), "n-gramme : \(ngram)")
    }
}
