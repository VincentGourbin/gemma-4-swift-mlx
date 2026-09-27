import Testing
import Foundation
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
    /// `String.count`, donc par graphemes). Quand il sera corrige, ce test le signalera
    /// (« known issue not recorded ») : `chatStream` en beneficiera alors aussi.
    @Test("NaiveStreamingDetokenizer amont perd les scalaires qui fusionnent en grapheme")
    func testUpstreamDetokenizerKnownIssue() async throws {
        let tokenizer = try await tokenizer()
        let ids = tokenizer.encode(text: Self.text, addSpecialTokens: false)
        var upstream = NaiveStreamingDetokenizer(tokenizer: tokenizer)
        var streamed = ""
        for id in ids {
            upstream.append(token: id)
            if let piece = upstream.next() { streamed += piece }
        }
        withKnownIssue("mlx-swift-lm NaiveStreamingDetokenizer : difference par graphemes") {
            #expect(streamed == tokenizer.decode(tokenIds: ids))
        }
    }
}
