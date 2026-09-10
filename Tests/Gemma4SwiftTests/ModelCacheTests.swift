import Testing
import Foundation
@testable import Gemma4Swift

@Suite("Model Cache")
struct ModelCacheTests {

    @Test("modelsDirectory par defaut pointe vers ~/Library/Caches/models/")
    func testDefaultDirectory() {
        // Reset custom directory
        Gemma4ModelCache.customModelsDirectory = nil
        let dir = Gemma4ModelCache.modelsDirectory
        #expect(dir.path.contains("Library/Caches/models"))
    }

    @Test("customModelsDirectory override")
    func testCustomDirectory() {
        let custom = URL(fileURLWithPath: "/tmp/test-gemma4-models")
        Gemma4ModelCache.customModelsDirectory = custom
        #expect(Gemma4ModelCache.modelsDirectory == custom)
        // Reset
        Gemma4ModelCache.customModelsDirectory = nil
    }

    @Test("systemRAMGB retourne une valeur raisonnable")
    func testSystemRAM() {
        let ram = Gemma4ModelCache.systemRAMGB
        #expect(ram >= 4)  // Au moins 4 Go
        #expect(ram <= 512)  // Pas plus de 512 Go
    }

    @Test("isDownloaded retourne false pour un modele inexistant")
    func testNotDownloaded() {
        // Le modele E4B base n'est probablement pas telecharge
        #expect(Gemma4ModelCache.isDownloaded(modelId: "google/gemma-4-fake-model") == false)
    }

    @Test("localPath retourne nil pour un modele absent")
    func testLocalPathNil() {
        let model = Gemma4Pipeline.Model.b31bBf16  // 31B BF16, peu probable d'etre la
        // On ne peut pas garantir qu'il est absent, mais on verifie que la methode ne crashe pas
        _ = Gemma4ModelCache.localPath(for: model)
    }

    @Test("diskSize retourne nil pour un modele absent")
    func testDiskSizeNil() {
        #expect(Gemma4ModelCache.diskSize(for: .b31bBf16) == nil)
    }

    @Test("diskSize suit un fichier de poids symlinke vers une cible externe")
    func testDiskSizeFollowsSymlinkedWeights() throws {
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let modelDir = root.appendingPathComponent("mlx-community/gemma-4-31b-it-bf16")
        try fm.createDirectory(at: modelDir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: root) }

        let configData = Data("{}".utf8)
        try configData.write(to: modelDir.appendingPathComponent("config.json"))

        // Cible "externe" simulee, en dehors de modelDir, avec une taille connue.
        let externalTarget = root.appendingPathComponent("external-weights.safetensors")
        let payload = Data(repeating: 0x42, count: 12_345)
        try payload.write(to: externalTarget)

        try fm.createSymbolicLink(
            at: modelDir.appendingPathComponent("model.safetensors"),
            withDestinationURL: externalTarget
        )

        Gemma4ModelCache.customModelsDirectory = root
        defer { Gemma4ModelCache.customModelsDirectory = nil }

        let size = Gemma4ModelCache.diskSize(for: .b31bBf16)
        #expect(size == Int64(payload.count + configData.count))
    }

    @Test("diskSize traite un symlink casse (cible absente) comme 0, pas comme une erreur")
    func testDiskSizeBrokenSymlinkContributesZero() throws {
        // Regression: URL.resolvingSymlinksInPath() ne signale pas l'echec quand la
        // cible est absente (disque externe non monte) — elle retourne silencieusement
        // le chemin du lien lui-meme, ce qui faisait fuiter la taille lstat du lien
        // (quelques octets) au lieu de contribuer 0. Signale par l'equipe LTX apres
        // avoir rencontre exactement ce piege dans leur propre fix (ask #4).
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let modelDir = root.appendingPathComponent("mlx-community/gemma-4-31b-it-bf16")
        try fm.createDirectory(at: modelDir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: root) }

        let configData = Data("{}".utf8)
        try configData.write(to: modelDir.appendingPathComponent("config.json"))

        // Symlink pointant vers une cible qui n'existe pas (disque externe debranche).
        try fm.createSymbolicLink(
            atPath: modelDir.appendingPathComponent("model.safetensors").path,
            withDestinationPath: "/nonexistent/external-disk/model.safetensors"
        )

        Gemma4ModelCache.customModelsDirectory = root
        defer { Gemma4ModelCache.customModelsDirectory = nil }

        let size = Gemma4ModelCache.diskSize(for: .b31bBf16)
        #expect(size == Int64(configData.count))
    }
}
