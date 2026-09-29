import Testing
import Foundation
@testable import Gemma4Swift

/// A-11 : `--drafter-path <dossier>` ne doit plus melanger drafter.safetensors et
/// drafter.best.safetensors dans l'ordre du systeme de fichiers.
@Suite("Poids du drafter")
struct DrafterWeightsTests {

    private func dir(_ files: [String]) throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("gemma4-drafter-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        for f in files { try Data([0]).write(to: url.appendingPathComponent(f)) }
        return url
    }

    @Test("dossier ambigu refuse")
    func testAmbiguous() throws {
        let url = try dir(["drafter.safetensors", "drafter.best.safetensors", "config.json"])
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(throws: DrafterTrainingError.self) { try Gemma4DrafterWeights.files(at: url) }
    }

    @Test("dossier de shards : ordre trie ; fichier : lui seul ; absent : erreur")
    func testResolve() throws {
        let url = try dir(["model-00002-of-00002.safetensors", "model-00001-of-00002.safetensors", "config.json"])
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(try Gemma4DrafterWeights.files(at: url).map(\.lastPathComponent)
            == ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"])
        let file = url.appendingPathComponent("model-00001-of-00002.safetensors")
        #expect(try Gemma4DrafterWeights.files(at: file) == [file])
        #expect(throws: DrafterTrainingError.self) {
            try Gemma4DrafterWeights.files(at: url.appendingPathComponent("absent"))
        }
    }
}
