import Testing
@testable import Gemma4Swift

/// K-33 : invite de generation retiree quelle que soit sa longueur selon la famille.
@Suite("Invite de generation")
struct GenerationPromptTests {
    // <bos> <|turn>user\n 500 <turn|>\n <|turn>model\n 600 601 <turn|>\n
    let conversation = [2, 105, 2364, 107, 500, 106, 107, 105, 4368, 107, 600, 601, 106, 107]

    @Test("E2B/E4B : <|turn>model\\n retire, comme avant")
    func testShortPrompt() {
        #expect(Gemma4Processor.droppingGenerationPrompt(conversation + [105, 4368, 107]) == conversation)
    }

    @Test("12B/26B/31B : canal de pensee vide retire aussi")
    func testThinkingChannelPrompt() {
        // <|channel>thought\n<channel|> : ids quelconques apres <|turn>model\n
        let prompt = [105, 4368, 107, 100, 45518, 107, 101]
        #expect(Gemma4Processor.droppingGenerationPrompt(conversation + prompt) == conversation)
    }

    @Test("sans invite ouverte : inchange (le dernier tour model est ferme)")
    func testNoPrompt() {
        #expect(Gemma4Processor.droppingGenerationPrompt(conversation) == conversation)
    }

    @Test("le masque de reponse retrouve la vraie reponse apres retrait")
    func testResponseMask() throws {
        let prompt = [105, 4368, 107, 100, 45518, 107, 101]
        let ids = Gemma4Processor.droppingGenerationPrompt(conversation + prompt)
        let sample = try #require(Gemma4LoRATrain.trainingSample(ids, maskPrompt: true))
        #expect(sample.tokens.count - sample.promptOffset >= 4)  // 600 601 <turn|> \n
    }
}
