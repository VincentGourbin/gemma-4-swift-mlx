import Testing
import Foundation
@testable import Gemma4Swift

/// Garde-fou S-02 (audit du 2026-09-27) : un telechargement multi-shards coupe
/// ne doit plus passer pour complet. Travaille sur `hasModelFiles(at:)` dans des
/// dossiers temporaires, sans toucher a `customModelsDirectory`.
@Suite("Completude d'un modele telecharge")
struct ModelCompletenessTests {

    private func makeModelDir() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("gemma4-completeness-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: dir.appendingPathComponent("config.json"))
        return dir
    }

    private func writeIndex(_ shards: [String], in dir: URL) throws {
        var weightMap: [String: String] = [:]
        for (i, shard) in shards.enumerated() { weightMap["layer.\(i).weight"] = shard }
        let data = try JSONSerialization.data(withJSONObject: ["metadata": [:], "weight_map": weightMap])
        try data.write(to: dir.appendingPathComponent("model.safetensors.index.json"))
    }

    private func touch(_ name: String, in dir: URL) throws {
        try Data([0]).write(to: dir.appendingPathComponent(name))
    }

    static let shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]

    @Test("index a 2 shards, un seul present : incomplet")
    func testMissingShard() throws {
        let dir = try makeModelDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        try writeIndex(Self.shards, in: dir)
        try touch(Self.shards[0], in: dir)
        #expect(!Gemma4ModelCache.hasModelFiles(at: dir))
    }

    @Test("index a 2 shards, les deux presents : complet")
    func testAllShards() throws {
        let dir = try makeModelDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        try writeIndex(Self.shards, in: dir)
        for shard in Self.shards { try touch(shard, in: dir) }
        #expect(Gemma4ModelCache.hasModelFiles(at: dir))
    }

    @Test("shard en lien symbolique pendant (disque externe demonte) : complet")
    func testDanglingSymlinkCountsAsPresent() throws {
        let dir = try makeModelDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        try writeIndex(Self.shards, in: dir)
        try touch(Self.shards[0], in: dir)
        try FileManager.default.createSymbolicLink(
            atPath: dir.appendingPathComponent(Self.shards[1]).path,
            withDestinationPath: "/Volumes/gemma4-absent-\(UUID().uuidString)/shard.safetensors")
        #expect(Gemma4ModelCache.hasModelFiles(at: dir))
    }

    @Test("sans index, un seul fichier de poids : complet")
    func testSingleFileModel() throws {
        let dir = try makeModelDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        try touch("model.safetensors", in: dir)
        #expect(Gemma4ModelCache.hasModelFiles(at: dir))
    }

    @Test("index illisible : incomplet")
    func testCorruptIndex() throws {
        let dir = try makeModelDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        try Data("{".utf8).write(to: dir.appendingPathComponent("model.safetensors.index.json"))
        try touch(Self.shards[0], in: dir)
        #expect(!Gemma4ModelCache.hasModelFiles(at: dir))
    }

    @Test("sans config.json : incomplet")
    func testMissingConfig() throws {
        let dir = try makeModelDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        try FileManager.default.removeItem(at: dir.appendingPathComponent("config.json"))
        try touch("model.safetensors", in: dir)
        #expect(!Gemma4ModelCache.hasModelFiles(at: dir))
    }
}
