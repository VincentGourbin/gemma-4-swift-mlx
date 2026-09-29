import Testing
import Foundation
@testable import Gemma4Swift

/// K-40 : magasin LRU des instantanes de conversation (sans modele : caches vides).
@Suite("Magasin de conversations")
struct ConversationStoreTests {

    private func snapshot(_ ids: [Int], images: [Data] = []) -> ConversationStore.Snapshot {
        .init(ids: ids, caches: [], imageDigests: images)
    }

    @Test("le plus long prefixe strict gagne ; deux conversations cohabitent")
    func testBestPrefix() {
        let store = ConversationStore(capacity: 4)
        store.put(snapshot([1, 2, 3]))
        store.put(snapshot([9, 8]))
        #expect(store.bestPrefix(of: [1, 2, 3, 4], digests: [])?.ids == [1, 2, 3])
        #expect(store.bestPrefix(of: [9, 8, 7], digests: [])?.ids == [9, 8])
        #expect(store.bestPrefix(of: [1, 2, 3], digests: []) == nil, "prefixe strict seulement")
        #expect(store.bestPrefix(of: [5, 6], digests: []) == nil)
    }

    @Test("un tour qui prolonge sa conversation remplace son instantane")
    func testReplaceExtended() {
        let store = ConversationStore(capacity: 4)
        store.put(snapshot([1, 2]))
        store.put(snapshot([1, 2, 3, 4]))
        #expect(store.count == 1)
        #expect(store.bestPrefix(of: [1, 2, 3, 4, 5], digests: [])?.ids == [1, 2, 3, 4])
    }

    @Test("capacite : le moins recemment utilise part")
    func testLRU() {
        let store = ConversationStore(capacity: 2)
        store.put(snapshot([1]))
        store.put(snapshot([2]))
        _ = store.bestPrefix(of: [1, 5], digests: [])  // [1] redevient recent
        store.put(snapshot([3]))
        #expect(store.bestPrefix(of: [2, 5], digests: []) == nil, "[2] evince")
        #expect(store.bestPrefix(of: [1, 5], digests: []) != nil)
    }

    @Test("images : l'instantane n'est pris que si ses images sont en tete de la requete")
    func testImages() {
        let store = ConversationStore()
        let a = Data([1]), b = Data([2])
        store.put(snapshot([1, 2], images: [a]))
        #expect(store.bestPrefix(of: [1, 2, 3], digests: [a, b]) != nil)
        #expect(store.bestPrefix(of: [1, 2, 3], digests: [b]) == nil)
    }
}
