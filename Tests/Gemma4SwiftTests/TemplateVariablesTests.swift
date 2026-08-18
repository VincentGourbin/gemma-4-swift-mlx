import Testing
import Foundation
import CoreGraphics
import MLX
import MLXLMCommon
@testable import Gemma4Swift

/// Variables de chat template (`additionalContext`), dont `enable_thinking`.
/// Gatees sur le meme chemin de modele local que les autres tests
/// d'integration ; la premiere suite n'a besoin que du tokenizer.
private let integrationModelPath = ProcessInfo.processInfo
    .environment["GEMMA4_INTEGRATION_MODEL_PATH"]

@Suite("Variables de chat template")
struct TemplateVariablesTests {

    private func loadTokenizer() async throws -> any Tokenizer {
        try await Gemma4TokenizerLoader().load(
            from: URL(fileURLWithPath: integrationModelPath!))
    }

    private func expandingImageMarkers(_ ids: [Int]) -> [Int] {
        let imageTokenId = Int(Gemma4Processor.imageTokenId)
        var expanded: [Int] = []
        for tid in ids {
            if tid == imageTokenId {
                expanded.append(Int(Gemma4Processor.boiTokenId))
                for _ in 0 ..< 280 { expanded.append(imageTokenId) }
                expanded.append(Int(Gemma4Processor.eoiTokenId))
            } else {
                expanded.append(tid)
            }
        }
        return expanded
    }

    // Rendus HF du meme chat_template.jinja (jinja2, add_generation_prompt=true)
    // pour le tour utilisateur "<|image|>\nHi." — marqueur image non expanse.
    private static let hfThinkingNoSystem = [
        2, 105, 9731, 107, 98, 107, 106, 107,
        105, 2364, 107, 258880, 107, 10979, 236761, 106, 107, 105, 4368, 107,
    ]
    private static let hfThinkingWithSystem = [
        2, 105, 9731, 107, 98, 107, 3912, 17514, 236761, 106, 107,
        105, 2364, 107, 258880, 107, 10979, 236761, 106, 107, 105, 4368, 107,
    ]

    @Test("enable_thinking sans tour systeme : le template en cree un avec <|think|>",
          .enabled(if: integrationModelPath != nil))
    func testThinkingCreatesSystemTurn() async throws {
        let tokenizer = try await loadTokenizer()

        let ids = try Gemma4Processor.multimodalChatIds(
            userPrompt: "Hi.", systemPrompt: nil, tokenizer: tokenizer,
            templateVariables: ["enable_thinking": true])

        #expect(ids == expandingImageMarkers(Self.hfThinkingNoSystem))
        #expect(ids.contains(Int(Gemma4Processor.thinkTokenId)))
    }

    @Test("enable_thinking avec tour systeme : <|think|> precede les instructions",
          .enabled(if: integrationModelPath != nil))
    func testThinkingWithSystemPrompt() async throws {
        let tokenizer = try await loadTokenizer()

        let ids = try Gemma4Processor.multimodalChatIds(
            userPrompt: "Hi.", systemPrompt: "Be terse.", tokenizer: tokenizer,
            templateVariables: ["enable_thinking": true])

        #expect(ids == expandingImageMarkers(Self.hfThinkingWithSystem))
    }

    @Test("enable_thinking: false se comporte comme l'absence de variable",
          .enabled(if: integrationModelPath != nil))
    func testThinkingFalseIsNoop() async throws {
        let tokenizer = try await loadTokenizer()

        let off = try Gemma4Processor.multimodalChatIds(
            userPrompt: "Hi.", tokenizer: tokenizer,
            templateVariables: ["enable_thinking": false])
        let absent = try Gemma4Processor.multimodalChatIds(
            userPrompt: "Hi.", tokenizer: tokenizer)

        #expect(off == absent)
        #expect(!off.contains(Int(Gemma4Processor.thinkTokenId)))
    }

    @Test("templateVariables: nil — non-regression stricte des ids",
          .enabled(if: integrationModelPath != nil))
    func testNilTemplateVariablesIsUnchanged() async throws {
        let tokenizer = try await loadTokenizer()
        let prompt = "user prompt: a 2CV on a coastal road"

        for system in [nil, "Be terse."] as [String?] {
            let withNil = try Gemma4Processor.multimodalChatIds(
                userPrompt: prompt, systemPrompt: system, tokenizer: tokenizer,
                templateVariables: nil)
            let withoutArg = try Gemma4Processor.multimodalChatIds(
                userPrompt: prompt, systemPrompt: system, tokenizer: tokenizer)
            #expect(withNil == withoutArg)
        }
    }

    @Test("Chemin texte : meme variable, meme effet",
          .enabled(if: integrationModelPath != nil))
    func testTextPathCarriesTemplateVariables() async throws {
        let tokenizer = try await loadTokenizer()

        let thinking = try Gemma4Processor.textChatIds(
            userPrompt: "Hi.", systemPrompt: "Be terse.", tokenizer: tokenizer,
            templateVariables: ["enable_thinking": true])
        let plain = try Gemma4Processor.textChatIds(
            userPrompt: "Hi.", systemPrompt: "Be terse.", tokenizer: tokenizer)

        #expect(thinking.contains(Int(Gemma4Processor.thinkTokenId)))
        #expect(!plain.contains(Int(Gemma4Processor.thinkTokenId)))
        // Le marqueur image n'a rien a faire sur le chemin texte.
        #expect(!thinking.contains(Int(Gemma4Processor.imageTokenId)))
    }
}

/// Verifie sur le modele reel que le raisonnement est bien emis et que ses
/// delimiteurs traversent le detokenizer streaming intacts.
@Suite("Chat template : thinking (integration)", .serialized)
struct TemplateVariablesIntegrationTests {

    @MainActor
    private func loadPipeline() async throws -> Gemma4Pipeline {
        let pipeline = Gemma4Pipeline()
        try await pipeline.load(
            from: URL(fileURLWithPath: integrationModelPath!), multimodal: true)
        return pipeline
    }

    private func collect(_ stream: AsyncThrowingStream<String, Error>) async throws -> String {
        var text = ""
        for try await chunk in stream { text += chunk }
        return text
    }

    private func syntheticImage(width: Int = 224, height: Int = 224) throws -> CGImage {
        let space = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(
            data: nil, width: width, height: height, bitsPerComponent: 8,
            bytesPerRow: 0, space: space,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else {
            throw Gemma4PipelineError.invalidInput("CGContext indisponible")
        }
        ctx.setFillColor(CGColor(red: 0.1, green: 0.2, blue: 0.6, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
        ctx.setFillColor(CGColor(red: 0.9, green: 0.7, blue: 0.1, alpha: 1))
        ctx.fillEllipse(in: CGRect(
            x: width / 4, y: height / 4, width: width / 2, height: height / 2))
        guard let image = ctx.makeImage() else {
            throw Gemma4PipelineError.invalidInput("makeImage a echoue")
        }
        return image
    }

    @Test("Multimodal : le canal de pensee est emis, delimiteurs intacts",
          .enabled(if: integrationModelPath != nil))
    @MainActor
    func testThinkingChannelSurvivesStreaming() async throws {
        let pipeline = try await loadPipeline()
        defer { pipeline.unload() }

        let pixels = try Gemma4ImageProcessor.processImage(try syntheticImage())
        let thinking = try await collect(pipeline.chatStreamMultimodal(
            prompt: "Combien de formes vois-tu ? Reponds brievement.",
            // Le raisonnement est verbeux : ~350 tokens ici. Sous ~300, la
            // generation est coupee avant <channel|> et le test mesurerait la
            // troncature, pas la survie des delimiteurs.
            pixelValues: pixels, temperature: 0.0, maxTokens: 400,
            templateVariables: ["enable_thinking": true]))

        // C'est la garantie demandee : les delimiteurs arrivent tels quels dans
        // le flux, le detokenizer ne les avale pas.
        #expect(thinking.contains("<|channel>"))
        #expect(thinking.contains("<channel|>"))
        #expect(thinking.contains("thought"))
        // Et il reste du texte hors canal de pensee.
        let afterThought = thinking.components(separatedBy: "<channel|>").last ?? ""
        #expect(!afterThought.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
    }

    @Test("Sans la variable, aucun canal de pensee",
          .enabled(if: integrationModelPath != nil))
    @MainActor
    func testNoThinkingChannelByDefault() async throws {
        let pipeline = try await loadPipeline()
        defer { pipeline.unload() }

        let pixels = try Gemma4ImageProcessor.processImage(try syntheticImage())
        let plain = try await collect(pipeline.chatStreamMultimodal(
            prompt: "Combien de formes vois-tu ? Reponds brievement.",
            pixelValues: pixels, temperature: 0.0, maxTokens: 120))

        #expect(!plain.contains("<|channel>thought"))
    }

    @Test("Le raisonnement genere compte dans la fenetre n-gramme",
          .enabled(if: integrationModelPath != nil))
    @MainActor
    func testThinkingTokensFeedNGramWindow() async throws {
        let pipeline = try await loadPipeline()
        defer { pipeline.unload() }

        let pixels = try Gemma4ImageProcessor.processImage(try syntheticImage())
        let text = try await collect(pipeline.chatStreamMultimodal(
            prompt: "Decris cette image.", pixelValues: pixels,
            temperature: 0.0, maxTokens: 150,
            noRepeatNGramSize: 5, noRepeatNGramIncludesPrompt: false,
            templateVariables: ["enable_thinking": true]))

        // didSample recoit tous les tokens echantillonnes, canal de pensee
        // compris : aucun 5-gramme de mots ne doit se repeter, y compris entre
        // le raisonnement et la reponse.
        let words = text.split(whereSeparator: { $0.isWhitespace || $0.isNewline })
            .map(String.init)
        var seen = Set<String>()
        var repeated = false
        if words.count >= 5 {
            for start in 0 ... (words.count - 5) {
                let gram = words[start ..< (start + 5)].joined(separator: " ")
                if !seen.insert(gram).inserted { repeated = true; break }
            }
        }
        #expect(!repeated)
    }
}
