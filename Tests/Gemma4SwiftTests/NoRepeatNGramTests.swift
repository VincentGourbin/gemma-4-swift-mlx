import Testing
import Foundation
import MLX
@testable import Gemma4Swift

@Suite("No-repeat n-gram")
struct NoRepeatNGramTests {

    /// Logits [1, vocab] a zero — un token banni se lit a -inf.
    private func zeroLogits(vocab: Int = 8, dtype: DType = .float32) -> MLXArray {
        MLXArray.zeros([1, vocab]).asType(dtype)
    }

    private func values(_ logits: MLXArray) -> [Float] {
        logits.asType(.float32).asArray(Float.self)
    }

    @Test("n=2 : le token qui a deja suivi le prefixe est banni, les autres intacts")
    func testBigramBlocking() {
        var processor = NoRepeatNGramLogitProcessor(ngramSize: 2)
        // Historique [a=1, b=2, a=1] : le bigramme (1, 2) existe deja,
        // le prochain token ne peut donc pas etre 2.
        processor.prompt(MLXArray([Int32(1), 2, 1]))

        let out = values(processor.process(logits: zeroLogits()))
        #expect(out[2] == -Float.infinity)
        for token in [0, 1, 3, 4, 5, 6, 7] {
            #expect(out[token] == 0)
        }
    }

    @Test("n=5 : sans repetition du prefixe, les logits sont inchanges")
    func testNoMatchIsIdentity() {
        var processor = NoRepeatNGramLogitProcessor(ngramSize: 5)
        processor.prompt(MLXArray([Int32(1), 2, 3, 4, 5, 6, 7]))

        let out = values(processor.process(logits: zeroLogits()))
        #expect(out.allSatisfy { $0 == 0 })
    }

    @Test("Prefixe incomplet (debut de sequence) : aucun blocage")
    func testShortHistoryIsIdentity() {
        var processor = NoRepeatNGramLogitProcessor(ngramSize: 5)
        processor.prompt(MLXArray([Int32(1), 2]))

        let out = values(processor.process(logits: zeroLogits()))
        #expect(out.allSatisfy { $0 == 0 })
    }

    @Test("didSample alimente l'historique comme le prompt")
    func testSampledTokensExtendHistory() {
        var processor = NoRepeatNGramLogitProcessor(ngramSize: 3)
        // Prompt [1, 2, 3] → trigramme (1, 2) → 3.
        processor.prompt(MLXArray([Int32(1), 2, 3]))
        // Rien a bloquer apres (2, 3).
        #expect(values(processor.process(logits: zeroLogits())).allSatisfy { $0 == 0 })

        // On rejoue 1 puis 2 : le prefixe redevient (1, 2), donc 3 est banni.
        processor.didSample(token: MLXArray([Int32(1)]))
        processor.didSample(token: MLXArray([Int32(2)]))

        let out = values(processor.process(logits: zeroLogits()))
        #expect(out[3] == -Float.infinity)
        #expect(out[1] == 0)
        #expect(out[2] == 0)
    }

    @Test("n=5 : cas LTX — le 5-gramme repete est coupe a la derniere position")
    func testFiveGramBlocking() {
        var processor = NoRepeatNGramLogitProcessor(ngramSize: 5)
        // (1,2,3,4) → 5 est deja vu ; l'historique se termine par (1,2,3,4).
        processor.prompt(MLXArray([Int32(1), 2, 3, 4, 5, 7, 1, 2, 3, 4]))

        let out = values(processor.process(logits: zeroLogits()))
        #expect(out[5] == -Float.infinity)
        #expect(out[7] == 0)
    }

    @Test("Plusieurs continuations bannies pour le meme prefixe")
    func testMultipleBannedContinuations() {
        var processor = NoRepeatNGramLogitProcessor(ngramSize: 2)
        // (1)→2 et (1)→3 vus tous les deux.
        processor.prompt(MLXArray([Int32(1), 2, 1, 3, 1]))

        let out = values(processor.process(logits: zeroLogits()))
        #expect(out[2] == -Float.infinity)
        #expect(out[3] == -Float.infinity)
        #expect(out[4] == 0)
    }

    @Test("n=1 : tout token deja vu est banni")
    func testUnigramBansEverythingSeen() {
        var processor = NoRepeatNGramLogitProcessor(ngramSize: 1)
        processor.prompt(MLXArray([Int32(2), 5]))

        let out = values(processor.process(logits: zeroLogits()))
        #expect(out[2] == -Float.infinity)
        #expect(out[5] == -Float.infinity)
        #expect(out[0] == 0)
    }

    @Test("Le dtype des logits est preserve (bf16)")
    func testDtypePreserved() {
        var processor = NoRepeatNGramLogitProcessor(ngramSize: 2)
        processor.prompt(MLXArray([Int32(1), 2, 1]))

        let out = processor.process(logits: zeroLogits(dtype: .bfloat16))
        #expect(out.dtype == .bfloat16)
        #expect(values(out)[2] == -Float.infinity)
    }

    // MARK: - Fenetre d'interdiction configurable

    @Test("includePromptInWindow=false : le prompt reste citable verbatim")
    func testPromptExcludedAllowsVerbatimQuote() {
        // Prompt [a=1, b=2, c=3, d=4, e=5], n=5 : regenerer a,b,c,d ne doit pas
        // interdire e (la citation fidele du prompt reste possible).
        var processor = NoRepeatNGramLogitProcessor(
            ngramSize: 5, includePromptInWindow: false)
        processor.prompt(MLXArray([Int32(1), 2, 3, 4, 5]))
        for token: Int32 in [1, 2, 3, 4] {
            processor.didSample(token: MLXArray([token]))
        }

        let out = values(processor.process(logits: zeroLogits()))
        #expect(out[5] == 0)
        #expect(out.allSatisfy { $0 == 0 })
    }

    @Test("includePromptInWindow=true (defaut) : le meme cas bannit bien e")
    func testPromptIncludedBansVerbatimQuote() {
        var processor = NoRepeatNGramLogitProcessor(ngramSize: 5)
        processor.prompt(MLXArray([Int32(1), 2, 3, 4, 5]))
        for token: Int32 in [1, 2, 3, 4] {
            processor.didSample(token: MLXArray([token]))
        }

        let out = values(processor.process(logits: zeroLogits()))
        #expect(out[5] == -Float.infinity)
        #expect(out[6] == 0)
    }

    @Test("includePromptInWindow=false : une boucle interne au genere est tuee")
    func testPromptExcludedStillKillsGeneratedLoop() {
        var processor = NoRepeatNGramLogitProcessor(
            ngramSize: 5, includePromptInWindow: false)
        processor.prompt(MLXArray([Int32(6), 7]))
        // Le genere boucle : a,b,c,d,e puis a,b,c,d → e doit etre banni.
        for token: Int32 in [1, 2, 3, 4, 5, 1, 2, 3, 4] {
            processor.didSample(token: MLXArray([token]))
        }

        let out = values(processor.process(logits: zeroLogits()))
        #expect(out[5] == -Float.infinity)
        #expect(out[6] == 0)
        #expect(out[7] == 0)
    }

    @Test("includePromptInWindow=false, n=1 : seuls les tokens generes sont bannis")
    func testPromptExcludedUnigram() {
        var processor = NoRepeatNGramLogitProcessor(
            ngramSize: 1, includePromptInWindow: false)
        processor.prompt(MLXArray([Int32(2), 5]))
        // Rien de genere encore : le prompt n'interdit rien.
        #expect(values(processor.process(logits: zeroLogits())).allSatisfy { $0 == 0 })

        processor.didSample(token: MLXArray([Int32(3)]))

        let out = values(processor.process(logits: zeroLogits()))
        #expect(out[3] == -Float.infinity)
        #expect(out[2] == 0)
        #expect(out[5] == 0)
    }

    @Test("Le defaut du parametre est la parite HF")
    func testDefaultIncludesPrompt() {
        #expect(NoRepeatNGramLogitProcessor(ngramSize: 5).includePromptInWindow)
    }

    // MARK: - Canal de pensee

    /// Ids reels du tokenizer Gemma 4 : `<|channel>thought\n` = [100, 45518, 107],
    /// `<channel|>` = 101.
    private static let channelStart = Gemma4Processor.channelStartTokenId
    private static let channelEnd = Gemma4Processor.channelEndTokenId
    private static let thoughtName = Gemma4Processor.thoughtChannelNameTokenId
    private static let responseName = Gemma4Processor.responseChannelNameTokenId

    private func feed(_ processor: inout NoRepeatNGramLogitProcessor, _ tokens: [Int32]) {
        for token in tokens {
            processor.didSample(token: MLXArray([token]))
        }
    }

    @Test("includesThinking=false : le raisonnement ne s'interdit pas dans la reponse")
    func testThinkingExcludedAllowsReuseInAnswer() {
        // <|channel>thought a b c d e <channel|> a b c d → e reste autorise.
        var processor = NoRepeatNGramLogitProcessor(
            ngramSize: 5, includeThinkingInWindow: false)
        feed(&processor, [Self.channelStart, Self.thoughtName, 1, 2, 3, 4, 5, Self.channelEnd])
        feed(&processor, [1, 2, 3, 4])

        let out = values(processor.process(logits: zeroLogits()))
        #expect(out[5] == 0)
        #expect(out.allSatisfy { $0 == 0 })
    }

    @Test("includesThinking=true (defaut) : le meme cas interdit e")
    func testThinkingIncludedBansReuseInAnswer() {
        var processor = NoRepeatNGramLogitProcessor(ngramSize: 5)
        feed(&processor, [Self.channelStart, Self.thoughtName, 1, 2, 3, 4, 5, Self.channelEnd])
        feed(&processor, [1, 2, 3, 4])

        let out = values(processor.process(logits: zeroLogits()))
        #expect(out[5] == -Float.infinity)
    }

    @Test("includesThinking=false : une boucle interne a la reponse reste tuee")
    func testThinkingExcludedStillKillsAnswerLoop() {
        var processor = NoRepeatNGramLogitProcessor(
            ngramSize: 5, includeThinkingInWindow: false)
        feed(&processor, [Self.channelStart, Self.thoughtName, 6, 7, Self.channelEnd])
        // Hors canal, la reponse boucle : a,b,c,d,e puis a,b,c,d → e banni.
        feed(&processor, [1, 2, 3, 4, 5, 1, 2, 3, 4])

        let out = values(processor.process(logits: zeroLogits()))
        #expect(out[5] == -Float.infinity)
        #expect(out[6] == 0)
        #expect(out[7] == 0)
    }

    @Test("includesThinking=true : la boucle interne a la reponse est tuee aussi")
    func testThinkingIncludedStillKillsAnswerLoop() {
        var processor = NoRepeatNGramLogitProcessor(ngramSize: 5)
        feed(&processor, [1, 2, 3, 4, 5, 1, 2, 3, 4])

        let out = values(processor.process(logits: zeroLogits()))
        #expect(out[5] == -Float.infinity)
    }

    @Test("Canal jamais referme : rien n'est interdit retroactivement")
    func testUnclosedThoughtChannelBansNothing() {
        // Generation coupee par maxTokens en plein raisonnement : l'automate
        // reste dans le canal, le mode degrade est « pas de blocage ».
        var processor = NoRepeatNGramLogitProcessor(
            ngramSize: 5, includeThinkingInWindow: false)
        feed(&processor, [Self.channelStart, Self.thoughtName])
        feed(&processor, [1, 2, 3, 4, 5, 1, 2, 3, 4])

        let out = values(processor.process(logits: zeroLogits()))
        #expect(out.allSatisfy { $0 == 0 })
    }

    @Test("Canal response : son contenu compte, contrairement au canal thought")
    func testResponseChannelStillFeedsWindow() {
        var processor = NoRepeatNGramLogitProcessor(
            ngramSize: 5, includeThinkingInWindow: false)
        feed(&processor, [Self.channelStart, Self.responseName])
        feed(&processor, [1, 2, 3, 4, 5, 1, 2, 3, 4])

        let out = values(processor.process(logits: zeroLogits()))
        #expect(out[5] == -Float.infinity)
    }

    @Test("<|think|> du prompt n'ouvre pas de canal : le prompt reste dans la fenetre")
    func testThinkMarkerDoesNotOpenChannel() {
        // Le template emet <|think|> au sommet du tour system, sans <channel|>
        // en face. L'y faire ouvrir un canal exclurait tout le prompt.
        var processor = NoRepeatNGramLogitProcessor(
            ngramSize: 5, includeThinkingInWindow: false)
        processor.prompt(MLXArray([Gemma4Processor.thinkTokenId, 1, 2, 3, 4, 5]))
        feed(&processor, [1, 2, 3, 4])

        let out = values(processor.process(logits: zeroLogits()))
        #expect(out[5] == -Float.infinity)
    }

    @Test("Le prompt aussi est filtre : un thought d'un tour precedent ne compte pas")
    func testThoughtChannelInPromptIsExcluded() {
        var processor = NoRepeatNGramLogitProcessor(
            ngramSize: 5, includeThinkingInWindow: false)
        processor.prompt(
            MLXArray([Self.channelStart, Self.thoughtName, 1, 2, 3, 4, 5, Self.channelEnd]))
        feed(&processor, [1, 2, 3, 4])

        let out = values(processor.process(logits: zeroLogits()))
        #expect(out.allSatisfy { $0 == 0 })
    }

    @Test("Dans le canal, aucun blocage : le prefixe gele ne rebannit pas")
    func testNoBanningWhileInsideThought() {
        // Historique gele pendant la pensee : sans garde, `process` rebannirait
        // les continuations du prefixe d'avant l'ouverture a chaque pas.
        var processor = NoRepeatNGramLogitProcessor(
            ngramSize: 5, includeThinkingInWindow: false)
        processor.prompt(MLXArray([Int32(1), 2, 3, 4, 5, 1, 2, 3, 4]))
        // Hors canal, le prefixe (1,2,3,4) bannit bien 5.
        #expect(values(processor.process(logits: zeroLogits()))[5] == -Float.infinity)

        feed(&processor, [Self.channelStart, Self.thoughtName])
        // Dans le canal : plus rien n'est bloque, a chaque pas du raisonnement.
        #expect(values(processor.process(logits: zeroLogits())).allSatisfy { $0 == 0 })
        feed(&processor, [42, 43, 44])
        #expect(values(processor.process(logits: zeroLogits())).allSatisfy { $0 == 0 })

        // A la sortie du canal, le blocage reprend a l'identique.
        feed(&processor, [Self.channelEnd])
        #expect(values(processor.process(logits: zeroLogits()))[5] == -Float.infinity)
    }

    @Test("Nom de canal inattendu : le token suivant reste du contenu compte")
    func testUnknownChannelNameStillCounts() {
        // `<|channel>` egare en pleine reponse : ne pas affaiblir la protection
        // anti-boucle la ou le modele part en vrille.
        var processor = NoRepeatNGramLogitProcessor(
            ngramSize: 5, includeThinkingInWindow: false)
        feed(&processor, [Self.channelStart, 1, 2, 3, 4, 5, 1, 2, 3, 4])

        let out = values(processor.process(logits: zeroLogits()))
        #expect(out[5] == -Float.infinity)
    }

    @Test("Non-regression : sans thinking, les deux modes sont identiques")
    func testNoThinkingTokensMeansIdenticalBehavior() {
        let sequence: [Int32] = [1, 2, 3, 4, 5, 1, 2, 3, 4]
        var included = NoRepeatNGramLogitProcessor(ngramSize: 5)
        var excluded = NoRepeatNGramLogitProcessor(
            ngramSize: 5, includeThinkingInWindow: false)
        included.prompt(MLXArray([Int32(9), 8, 7]))
        excluded.prompt(MLXArray([Int32(9), 8, 7]))
        feed(&included, sequence)
        feed(&excluded, sequence)

        #expect(values(included.process(logits: zeroLogits()))
            == values(excluded.process(logits: zeroLogits())))
    }

    @Test("Le defaut du parametre est le comportement historique")
    func testDefaultIncludesThinking() {
        #expect(NoRepeatNGramLogitProcessor(ngramSize: 5).includeThinkingInWindow)
    }

    @Test("Logits 1D acceptes (rang preserve)")
    func testOneDimensionalLogits() {
        var processor = NoRepeatNGramLogitProcessor(ngramSize: 2)
        processor.prompt(MLXArray([Int32(1), 2, 1]))

        let out = processor.process(logits: MLXArray.zeros([8]))
        #expect(out.shape == [8])
        #expect(values(out)[2] == -Float.infinity)
    }
}
