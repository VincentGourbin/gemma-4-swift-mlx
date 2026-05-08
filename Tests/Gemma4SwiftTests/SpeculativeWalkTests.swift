import Testing
@testable import Gemma4Swift

@Suite("Speculative walk")
struct SpeculativeWalkTests {

    @Test("Mismatch immediat: accepted=0, emit [target[0]]")
    func testMismatchAtZero() {
        let r = SpeculativeWalk.walk(
            drafts: [10, 20, 30],
            targets: [99, 50, 60, 70],
            budget: 10
        )
        #expect(r.accepted == 0)
        #expect(r.newTokens == [99])
    }

    @Test("Mismatch a position 1: accepted=1, emit [d0, target[1]]")
    func testMismatchAtOne() {
        let r = SpeculativeWalk.walk(
            drafts: [10, 20, 30],
            targets: [10, 99, 60, 70],
            budget: 10
        )
        #expect(r.accepted == 1)
        #expect(r.newTokens == [10, 99])
    }

    @Test("Tous les drafts acceptes: emit drafts + target bonus final")
    func testAllAccepted() {
        let r = SpeculativeWalk.walk(
            drafts: [10, 20, 30],
            targets: [10, 20, 30, 99],
            budget: 10
        )
        #expect(r.accepted == 3)
        #expect(r.newTokens == [10, 20, 30, 99])
    }

    @Test("Budget tronque les tokens emis")
    func testBudgetTruncation() {
        let r = SpeculativeWalk.walk(
            drafts: [10, 20, 30],
            targets: [10, 20, 30, 99],
            budget: 2
        )
        #expect(r.accepted == 3)
        #expect(r.newTokens == [10, 20])
    }

    @Test("Budget = 0: aucun token emis")
    func testZeroBudget() {
        let r = SpeculativeWalk.walk(
            drafts: [10, 20],
            targets: [99, 88, 77],
            budget: 0
        )
        #expect(r.accepted == 0)
        #expect(r.newTokens.isEmpty)
    }

    @Test("Mismatch a la derniere position")
    func testMismatchAtLast() {
        let r = SpeculativeWalk.walk(
            drafts: [10, 20, 30],
            targets: [10, 20, 99, 70],
            budget: 10
        )
        #expect(r.accepted == 2)
        #expect(r.newTokens == [10, 20, 99])
    }

    @Test("Drafts vides: emit juste target[0]")
    func testEmptyDrafts() {
        let r = SpeculativeWalk.walk(
            drafts: [],
            targets: [42],
            budget: 10
        )
        #expect(r.accepted == 0)
        #expect(r.newTokens == [42])
    }

    @Test("Equivalence avec algo Python sur cas crafted")
    func testParityWithPython() {
        // Reference Python: d=[1,2,3], t=[1,2,99,77]
        //   accepted = next(i for i in range(3) if d[i] != t[i]) = 2
        //   new_tokens = (d[:2] + [t[2]])[:budget] = [1, 2, 99]
        let r = SpeculativeWalk.walk(
            drafts: [1, 2, 3],
            targets: [1, 2, 99, 77],
            budget: 10
        )
        #expect(r.accepted == 2)
        #expect(r.newTokens == [1, 2, 99])
    }
}
