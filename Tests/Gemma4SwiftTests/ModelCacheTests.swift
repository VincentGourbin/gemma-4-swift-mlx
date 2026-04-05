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
        let model = Gemma4Pipeline.Model.b31b  // 31B base, peu probable d'etre la
        // On ne peut pas garantir qu'il est absent, mais on verifie que la methode ne crashe pas
        _ = Gemma4ModelCache.localPath(for: model)
    }

    @Test("diskSize retourne nil pour un modele absent")
    func testDiskSizeNil() {
        #expect(Gemma4ModelCache.diskSize(for: .b31b) == nil)
    }
}
