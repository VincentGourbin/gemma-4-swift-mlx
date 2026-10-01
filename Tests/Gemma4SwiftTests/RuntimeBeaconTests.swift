import Testing
import Foundation
@testable import Gemma4Swift

/// Beacon de runtime (contrat SiliconScope, schema v1). Repertoire temporaire : les tests
/// n'ecrivent jamais dans ~/Library/Application Support/ai-runtime-beacons.
@Suite("Beacon de runtime (SiliconScope)", .serialized)
struct RuntimeBeaconTests {

    /// Lance `body` avec un repertoire jetable et un etat d'activation connu, puis remet
    /// tout en place.
    private func withDirectory(enabled: Bool, _ body: (URL) throws -> Void) rethrows {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("beacon-tests-\(UUID().uuidString)", isDirectory: true)
        unsetenv("GEMMA4_RUNTIME_BEACON")
        RuntimeBeacon.directoryOverride = dir
        RuntimeBeacon.isEnabled = enabled
        defer {
            RuntimeBeacon.isEnabled = false
            RuntimeBeacon.directoryOverride = nil
            try? FileManager.default.removeItem(at: dir)
        }
        try body(dir)
    }

    private func manifests(_ dir: URL) -> [URL] {
        ((try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? [])
            .filter { $0.pathExtension == "json" }
    }

    @Test("(a) desactive par defaut : aucun fichier")
    func testDisabledByDefault() {
        withDirectory(enabled: false) { dir in
            let session = RuntimeBeacon.begin(task: "generate", model: "e2b")
            #expect(session == nil)
            #expect(manifests(dir).isEmpty)
        }
    }

    @Test("(b) active : fichier <pid>-<id>.json conforme pendant l'operation, supprime apres")
    func testManifestLifecycle() throws {
        try withDirectory(enabled: true) { dir in
            let session = try #require(RuntimeBeacon.begin(task: "generate", model: "gemma-4-e2b-it-4bit"))
            session.update(phase: "decode", step: 3, totalSteps: 10)
            let files = manifests(dir)
            #expect(files.count == 1)
            let url = try #require(files.first)
            let pid = ProcessInfo.processInfo.processIdentifier
            let parts = url.deletingPathExtension().lastPathComponent.split(separator: "-")
            #expect(parts.count == 2 && parts[0] == "\(pid)")
            let json = try #require(try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
            #expect(json["version"] as? Int == 1)
            #expect(json["pid"] as? Int == Int(pid))
            #expect(json["runtime"] as? String == "gemma-4-swift-mlx")
            #expect(json["displayName"] as? String == "Gemma 4")
            #expect(json["task"] as? String == "generate")
            #expect(json["model"] as? String == "gemma-4-e2b-it-4bit")
            #expect(json["phase"] as? String == "decode")
            #expect(json["step"] as? Int == 3 && json["totalSteps"] as? Int == 10)
            // ISO-8601 UTC sans fractions de seconde.
            let date = try #require(json["updatedAt"] as? String)
            #expect(date.wholeMatch(of: /\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z/) != nil)
            session.end()
            #expect(manifests(dir).isEmpty)
        }
    }

    @Test("(b bis) GEMMA4_RUNTIME_BEACON=1 suffit, lue a l'appel")
    func testEnvironmentVariable() {
        withDirectory(enabled: false) { dir in
            setenv("GEMMA4_RUNTIME_BEACON", "1", 1)
            defer { unsetenv("GEMMA4_RUNTIME_BEACON") }
            let session = RuntimeBeacon.begin(task: "train")
            #expect(session != nil)
            #expect(manifests(dir).count == 1)
            session?.end()
        }
    }

    @Test("(c) supprime quand l'operation leve une erreur")
    func testRemovedOnThrow() {
        struct Failure: Error {}
        withDirectory(enabled: true) { dir in
            func operation() throws {
                let beacon = RuntimeBeacon.begin(task: "generate")
                defer { beacon?.end() }
                #expect(manifests(dir).count == 1)
                throw Failure()
            }
            #expect(throws: Failure.self) { try operation() }
            #expect(manifests(dir).isEmpty)
        }
    }

    @Test("(d) update apres end ne recree pas le fichier")
    func testUpdateAfterEnd() throws {
        try withDirectory(enabled: true) { dir in
            let session = try #require(RuntimeBeacon.begin(task: "generate"))
            session.end()
            session.update(phase: "decode", step: 1, totalSteps: 2)
            session.update(phase: "other")
            #expect(manifests(dir).isEmpty)
        }
    }

    @Test("(e) le manifeste d'un pid mort est nettoye au debut d'une session, pas celui d'un pid vivant")
    func testStaleManifestCleanup() throws {
        try withDirectory(enabled: true) { dir in
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            // pid tres grand : aucun processus ne l'a (kill renvoie ESRCH).
            let dead = dir.appendingPathComponent("99999999-deadbeef.json")
            try Data(#"{"version":1,"pid":99999999,"runtime":"x"}"#.utf8).write(to: dead)
            // pid 1 (launchd) est vivant : son fichier reste.
            let alive = dir.appendingPathComponent("1-cafebabe.json")
            try Data(#"{"version":1,"pid":1,"runtime":"x"}"#.utf8).write(to: alive)
            let session = RuntimeBeacon.begin(task: "generate")
            #expect(!FileManager.default.fileExists(atPath: dead.path))
            #expect(FileManager.default.fileExists(atPath: alive.path))
            session?.end()
        }
    }
}
