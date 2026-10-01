import Testing
import Foundation
import MLX
import MLXLMCommon
@testable import Gemma4Swift

/// Budget de pensee (K-42) : au-dela de N jetons dans le canal de pensee, seul `<channel|>`
/// reste echantillonnable ; hors du canal, les logits ne sont jamais touches.
@Suite("Budget de pensee")
struct ThinkingBudgetTests {
    let vocab = 46_000  // > 45518, id du nom de canal « thought »

    private func logits() -> MLXArray { MLXArray.zeros([1, vocab]) }

    private func forced(_ p: Gemma4ThinkingBudgetProcessor) -> Bool {
        let out = p.process(logits: logits())
        let end = Int(Gemma4Processor.channelEndTokenId)
        return out[0, end].item(Float.self) == 0 && out[0, 0].item(Float.self) == -Float.infinity
    }

    private func sample(_ p: inout Gemma4ThinkingBudgetProcessor, _ tokens: [Int32]) {
        for t in tokens { p.didSample(token: MLXArray([t])) }
    }

    @Test("dans la pensee : libre jusqu'au budget, puis seul <channel|>")
    func testForcesEndAfterBudget() {
        Device.withDefaultDevice(.cpu) {
            var p = Gemma4ThinkingBudgetProcessor(budget: 3)
            p.prompt(MLXArray([Int32(2)]))
            sample(&p, [Gemma4Processor.channelStartTokenId, Gemma4Processor.thoughtChannelNameTokenId])
            #expect(!forced(p))                 // 0 jeton de pensee
            sample(&p, [7, 8])
            #expect(!forced(p))                 // 2 < 3
            sample(&p, [9])
            #expect(forced(p))                  // 3 = budget atteint
            sample(&p, [Gemma4Processor.channelEndTokenId])
            #expect(!forced(p))                 // canal ferme : reponse libre
            sample(&p, [10, 11, 12, 13])
            #expect(!forced(p))
        }
    }

    @Test("hors du canal de pensee, ou canal « response » : jamais force")
    func testNeverForcedOutsideThought() {
        Device.withDefaultDevice(.cpu) {
            var p = Gemma4ThinkingBudgetProcessor(budget: 0)
            p.prompt(MLXArray([Int32(2)]))
            #expect(!forced(p))
            sample(&p, [5, 6, 7])
            #expect(!forced(p))
            sample(&p, [Gemma4Processor.channelStartTokenId, Gemma4Processor.responseChannelNameTokenId, 8])
            #expect(!forced(p))
        }
    }

    @Test("un second passage dans la pensee repart de zero")
    func testCounterResetsOnReentry() {
        Device.withDefaultDevice(.cpu) {
            var p = Gemma4ThinkingBudgetProcessor(budget: 2)
            p.prompt(MLXArray([Int32(2)]))
            sample(&p, [Gemma4Processor.channelStartTokenId, Gemma4Processor.thoughtChannelNameTokenId, 7, 8])
            #expect(forced(p))
            sample(&p, [Gemma4Processor.channelEndTokenId, 20,
                        Gemma4Processor.channelStartTokenId, Gemma4Processor.thoughtChannelNameTokenId])
            #expect(!forced(p))
            sample(&p, [7])
            #expect(!forced(p))
            sample(&p, [8])
            #expect(forced(p))
        }
    }

    @Test("chaine : le budget s'applique apres l'autre processeur")
    func testChained() {
        Device.withDefaultDevice(.cpu) {
            var chain = Gemma4ChainedLogitProcessor(
                NoRepeatNGramLogitProcessor(ngramSize: 3), Gemma4ThinkingBudgetProcessor(budget: 1))
            chain.prompt(MLXArray([Int32(2)]))
            for t in [Gemma4Processor.channelStartTokenId, Gemma4Processor.thoughtChannelNameTokenId, 7] {
                chain.didSample(token: MLXArray([t]))
            }
            let out = chain.process(logits: logits())
            #expect(out[0, Int(Gemma4Processor.channelEndTokenId)].item(Float.self) == 0)
            #expect(out[0, 0].item(Float.self) == -Float.infinity)
        }
    }
}
