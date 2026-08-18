import Testing
import Foundation
import CoreGraphics
import MLXLMCommon
@testable import Gemma4Swift

/// La premiere suite n'a besoin que du tokenizer et du chat_template.jinja —
/// pas des poids ; la seconde charge le modele et verifie bout-en-bout que le
/// tour system est bien suivi. Les deux sont gatees sur le meme chemin de
/// modele local que les autres tests d'integration :
///
/// ```
/// GEMMA4_INTEGRATION_MODEL_PATH=~/Library/Caches/models/mlx-community/gemma-4-e4b-it-4bit \
///   Scripts/run-tests.sh -only-testing:Gemma4SwiftTests/MultimodalSystemPromptTests
/// ```
private let integrationModelPath = ProcessInfo.processInfo
    .environment["GEMMA4_INTEGRATION_MODEL_PATH"]

@Suite("Prompt systeme sur le chemin multimodal")
struct MultimodalSystemPromptTests {

    private func loadTokenizer() async throws -> any Tokenizer {
        try await Gemma4TokenizerLoader().load(
            from: URL(fileURLWithPath: integrationModelPath!))
    }

    /// Construction historique (avant l'ajout de `systemPrompt`), recopiee ici
    /// pour servir d'oracle de non-regression.
    private func legacyIds(prompt: String, tokenizer: any Tokenizer) throws -> [Int] {
        let content = "<|image|>\n\(prompt)"
        let ids = try tokenizer.applyChatTemplate(
            messages: [["role": "user", "content": content]])
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

    private func imageTokenCount(_ ids: [Int]) -> Int {
        ids.filter { $0 == Int(Gemma4Processor.imageTokenId) }.count
    }

    @Test("Sans systemPrompt : ids strictement identiques a avant",
          .enabled(if: integrationModelPath != nil))
    func testNoSystemPromptIsUnchanged() async throws {
        let tokenizer = try await loadTokenizer()
        let prompt = "user prompt: a 2CV driving along a coastal road"

        let ids = try Gemma4Processor.multimodalChatIds(
            userPrompt: prompt, systemPrompt: nil, tokenizer: tokenizer)

        #expect(ids == (try legacyIds(prompt: prompt, tokenizer: tokenizer)))
    }

    @Test("Avec systemPrompt : tour system distinct, different de la concatenation",
          .enabled(if: integrationModelPath != nil))
    func testSystemPromptRendersOwnTurn() async throws {
        let tokenizer = try await loadTokenizer()
        let system = "You are a prompt enhancer. Answer with the caption only."
        let prompt = "user prompt: a 2CV driving along a coastal road"

        let withSystem = try Gemma4Processor.multimodalChatIds(
            userPrompt: prompt, systemPrompt: system, tokenizer: tokenizer)
        // Ce que fait le consommateur aujourd'hui, faute de parametre : tout
        // dans le tour utilisateur.
        let concatenated = try Gemma4Processor.multimodalChatIds(
            userPrompt: "\(system)\n\n\(prompt)", systemPrompt: nil, tokenizer: tokenizer)

        #expect(withSystem != concatenated)

        // Le template Gemma 4 rend un tour system distinct — c'est lui qui
        // decide du placement, on ne le prefixe pas a la main.
        let rendered = tokenizer.decode(tokenIds: withSystem, skipSpecialTokens: false)
        #expect(rendered.contains("<|turn>system"))
        #expect(rendered.contains("<|turn>user"))
        // Le tour system precede le tour utilisateur.
        let systemRange = try #require(rendered.range(of: "<|turn>system"))
        let userRange = try #require(rendered.range(of: "<|turn>user"))
        #expect(systemRange.lowerBound < userRange.lowerBound)
        // Et le systeme ne porte pas le marqueur image.
        #expect(!rendered.contains("<|turn>system\n<|image>"))
    }

    @Test("Le budget de tokens image est inchange, avec ou sans systemPrompt",
          .enabled(if: integrationModelPath != nil))
    func testImageTokenBudgetUnchanged() async throws {
        let tokenizer = try await loadTokenizer()
        let prompt = "Describe this image."
        let system = String(repeating: "Instruction line.\n", count: 200)

        let without = try Gemma4Processor.multimodalChatIds(
            userPrompt: prompt, systemPrompt: nil, tokenizer: tokenizer)
        let with = try Gemma4Processor.multimodalChatIds(
            userPrompt: prompt, systemPrompt: system, tokenizer: tokenizer)

        #expect(imageTokenCount(without) == 280)
        #expect(imageTokenCount(with) == 280)
        #expect(with.filter { $0 == Int(Gemma4Processor.boiTokenId) }.count == 1)
        #expect(with.filter { $0 == Int(Gemma4Processor.eoiTokenId) }.count == 1)
        // Seul le tour system s'ajoute : le reste du prompt ne bouge pas.
        #expect(with.count > without.count)
    }

    @Test("Un marqueur image dans le systemPrompt est refuse",
          .enabled(if: integrationModelPath != nil))
    func testImageMarkerInSystemPromptThrows() async throws {
        let tokenizer = try await loadTokenizer()

        #expect(throws: Gemma4PipelineError.self) {
            _ = try Gemma4Processor.multimodalChatIds(
                userPrompt: "Describe this image.",
                systemPrompt: "Voici l'image de reference : <|image|>",
                tokenizer: tokenizer)
        }
    }

    @Test("Plusieurs images restent possibles via le tour utilisateur",
          .enabled(if: integrationModelPath != nil))
    func testMultipleImagesInUserTurnStillExpand() async throws {
        let tokenizer = try await loadTokenizer()

        // Le premier marqueur est ajoute par multimodalChatIds, le second vient
        // du prompt : deux images empilees sur l'axe batch de pixelValues.
        let ids = try Gemma4Processor.multimodalChatIds(
            userPrompt: "<|image|>\nCompare these two images.",
            systemPrompt: "Be concise.",
            tokenizer: tokenizer)

        #expect(imageTokenCount(ids) == 560)
    }
}

/// Verifie que le tour system n'est pas seulement rendu dans le template, mais
/// effectivement suivi par le modele sur le chemin multimodal.
@Suite("Prompt systeme multimodal (integration)", .serialized)
struct MultimodalSystemPromptIntegrationTests {

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

    /// Image synthetique (degrade) — evite un asset binaire dans le repo.
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

    @Test("Le tour system est suivi (instruction absente du tour utilisateur)",
          .enabled(if: integrationModelPath != nil))
    @MainActor
    func testSystemTurnIsObeyed() async throws {
        let pipeline = try await loadPipeline()
        defer { pipeline.unload() }

        let pixels = try Gemma4ImageProcessor.processImage(try syntheticImage())
        // Le mot-cle n'apparait que dans le tour system : s'il ressort, c'est
        // que ce tour a bien atteint le modele.
        let system = "Commence imperativement ta reponse par le mot exact BANANE."
        let prompt = "Decris cette image en une phrase."

        let guided = try await collect(pipeline.chatStreamMultimodal(
            prompt: prompt, pixelValues: pixels, systemPrompt: system,
            temperature: 0.0, maxTokens: 40))
        let plain = try await collect(pipeline.chatStreamMultimodal(
            prompt: prompt, pixelValues: pixels,
            temperature: 0.0, maxTokens: 40))

        #expect(guided.contains("BANANE"))
        #expect(!plain.contains("BANANE"))
    }
}
