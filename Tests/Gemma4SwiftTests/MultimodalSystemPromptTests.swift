import Testing
import Foundation
import CoreGraphics
import MLX
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

    /// Expanse les marqueurs image d'une sequence d'ids de reference, pour
    /// pouvoir comparer a la sortie de `multimodalChatIds` sans recopier 280
    /// ids a la main.
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

    private func imageTokenCount(_ ids: [Int]) -> Int {
        ids.filter { $0 == Int(Gemma4Processor.imageTokenId) }.count
    }

    // Ids de reference : rendu du meme chat_template.jinja par HF
    // (jinja2 ImmutableSandboxedEnvironment(trim_blocks: true,
    // lstrip_blocks: true), add_generation_prompt=true), pour le prompt
    // "<|image|>\nHi." — marqueur image non encore expanse.
    private static let hfUserOnly = [
        2, 105, 2364, 107, 258880, 107, 10979, 236761, 106, 107, 105, 4368, 107,
    ]
    private static let hfSystemAndUser = [
        2, 105, 9731, 107, 3912, 17514, 236761, 106, 107,
        105, 2364, 107, 258880, 107, 10979, 236761, 106, 107, 105, 4368, 107,
    ]

    @Test("Sans systemPrompt : parite token a token avec le rendu HF",
          .enabled(if: integrationModelPath != nil))
    func testNoSystemPromptMatchesReference() async throws {
        let tokenizer = try await loadTokenizer()

        let ids = try Gemma4Processor.multimodalChatIds(
            userPrompt: "Hi.", systemPrompt: nil, tokenizer: tokenizer)

        #expect(ids == expandingImageMarkers(Self.hfUserOnly))
    }

    @Test("Avec systemPrompt : parite token a token avec le rendu HF",
          .enabled(if: integrationModelPath != nil))
    func testSystemPromptMatchesReference() async throws {
        let tokenizer = try await loadTokenizer()

        let ids = try Gemma4Processor.multimodalChatIds(
            userPrompt: "Hi.", systemPrompt: "Be terse.", tokenizer: tokenizer)

        #expect(ids == expandingImageMarkers(Self.hfSystemAndUser))
    }

    @Test("Les sauts de ligne parasites de swift-jinja sont reparés",
          .enabled(if: integrationModelPath != nil))
    func testJinjaWhitespaceArtifactsAreStripped() async throws {
        let tokenizer = try await loadTokenizer()
        let bos = Int(Gemma4Processor.bosTokenId)
        let turnStart = Int(Gemma4Processor.turnStartTokenId)
        let turnEnd = Int(Gemma4Processor.turnEndTokenId)
        let newline = Int(Gemma4Processor.newlineTokenId)
        let doubleNewline = Int(Gemma4Processor.doubleNewlineTokenId)

        // Ce que rend swift-jinja aujourd'hui : \n parasite apres <bos> quand il
        // n'y a pas de tour systeme, \n\n entre les tours quand il y en a un.
        let rawNoSystem = try tokenizer.applyChatTemplate(
            messages: [["role": "user", "content": "Hi."]])
        let rawWithSystem = try tokenizer.applyChatTemplate(messages: [
            ["role": "system", "content": "Be terse."],
            ["role": "user", "content": "Hi."],
        ])
        #expect(rawNoSystem.count >= 2 && rawNoSystem[1] == newline)
        #expect(rawWithSystem.contains(doubleNewline))

        // Apres reparation, plus aucun des deux.
        let fixedNoSystem = Gemma4Processor.strippingTemplateArtifacts(rawNoSystem)
        let fixedWithSystem = Gemma4Processor.strippingTemplateArtifacts(rawWithSystem)
        #expect(fixedNoSystem[0] == bos && fixedNoSystem[1] == turnStart)
        #expect(!fixedWithSystem.contains(doubleNewline))
        #expect(fixedNoSystem.count == rawNoSystem.count - 1)
        #expect(fixedWithSystem.count == rawWithSystem.count)
        // Et le contenu utile n'a pas bouge.
        #expect(fixedWithSystem.filter { $0 != newline } == rawWithSystem.filter {
            $0 != newline && $0 != doubleNewline
        })
        #expect(fixedWithSystem.contains(turnEnd))
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

    @Test("Un marqueur audio ou video dans le systemPrompt est refuse aussi",
          .enabled(if: integrationModelPath != nil))
    func testOtherModalityMarkersInSystemPromptThrow() async throws {
        let tokenizer = try await loadTokenizer()

        // Cas realiste : un long prompt systeme qui documente les marqueurs du
        // modele. Sans garde, ces ids speciaux partent bruts dans la sequence.
        for marker in [
            Gemma4Processor.audioToken, Gemma4Processor.videoToken,
            Gemma4Processor.boiToken, Gemma4Processor.eoiToken,
        ] {
            #expect(throws: Gemma4PipelineError.self) {
                _ = try Gemma4Processor.multimodalChatIds(
                    userPrompt: "Describe this image.",
                    systemPrompt: "Never emit \(marker) yourself.",
                    tokenizer: tokenizer)
            }
        }
    }

    @Test("buildMultimodalPrompt tokenise comme le chat template du modele",
          .enabled(if: integrationModelPath != nil))
    func testBuildMultimodalPromptMatchesChatTemplate() async throws {
        let tokenizer = try await loadTokenizer()
        let userPrompt = "What is the capital of France?"

        // Oracle : le rendu du chat_template.jinja du modele, debarrasse des
        // sauts de ligne parasites de swift-jinja.
        let expected = Gemma4Processor.strippingTemplateArtifacts(
            try tokenizer.applyChatTemplate(
                messages: [["role": "user", "content": userPrompt]]))
        // La construction manuelle doit tomber sur les memes ids : c'est ce qui
        // garantit que l'evaluation LoRA voit le meme format que l'inference.
        let built = tokenizer.encode(
            text: Gemma4Processor.buildMultimodalPrompt(userPrompt: userPrompt),
            addSpecialTokens: false)

        #expect(built == expected)
    }

    @Test("buildMultimodalPrompt : le tour systeme suit aussi le template",
          .enabled(if: integrationModelPath != nil))
    func testBuildMultimodalPromptSystemMatchesChatTemplate() async throws {
        let tokenizer = try await loadTokenizer()
        let system = "You are terse."
        let userPrompt = "Hello."

        let expected = Gemma4Processor.strippingTemplateArtifacts(
            try tokenizer.applyChatTemplate(messages: [
                ["role": "system", "content": system],
                ["role": "user", "content": userPrompt],
            ]))
        let built = tokenizer.encode(
            text: Gemma4Processor.buildMultimodalPrompt(
                userPrompt: userPrompt, systemPrompt: system),
            addSpecialTokens: false)

        #expect(built == expected)
    }

    @Test("applyGemma4ChatTemplate porte les marqueurs de tour Gemma 4",
          .enabled(if: integrationModelPath != nil))
    func testLoRAChatTemplateUsesGemma4Markers() async throws {
        let tokenizer = try await loadTokenizer()
        let text = applyGemma4ChatTemplate(messages: [
            ChatMessage(role: "user", content: "Decris."),
            ChatMessage(role: "assistant", content: "Une image."),
        ])
        let ids = tokenizer.encode(text: text, addSpecialTokens: false)

        // Le preprocessing LoRA multimodal cherche <|turn> user \n (105, 2364,
        // 107) pour y injecter boi + image_token × 280 + eoi, et <|turn> model
        // (105, 4368) pour masquer le prompt. Avec des marqueurs Gemma 3, ces
        // ancres n'existent pas et l'image n'est jamais injectee.
        let turn = Int(105), user = Int(2364), newline = Int(107), model = Int(4368)
        let hasInjectionPoint = (0 ..< max(0, ids.count - 2)).contains {
            ids[$0] == turn && ids[$0 + 1] == user && ids[$0 + 2] == newline
        }
        let hasMaskAnchor = (0 ..< max(0, ids.count - 1)).contains {
            ids[$0] == turn && ids[$0 + 1] == model
        }
        #expect(hasInjectionPoint)
        #expect(hasMaskAnchor)
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

    @Test("Desaccord entre nombre d'images et de marqueurs : erreur, pas de silence",
          .enabled(if: integrationModelPath != nil))
    @MainActor
    func testImageCountMismatchThrows() async throws {
        let pipeline = try await loadPipeline()
        defer { pipeline.unload() }

        // Deux images empilees, un seul marqueur : maskedScatter indexe modulo
        // la taille de la source et laisserait passer en silence.
        let one = try Gemma4ImageProcessor.processImage(try syntheticImage())
        let two = concatenated([one, one], axis: 0)

        var caught: Error? = nil
        do {
            let stream = try pipeline.chatStreamMultimodal(
                prompt: "Compare.", pixelValues: two, maxTokens: 8)
            for try await _ in stream {}
        } catch {
            caught = error
        }
        #expect(caught is Gemma4PipelineError)
    }
}
