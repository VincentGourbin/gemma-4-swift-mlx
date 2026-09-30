import Testing
import Foundation
@testable import Gemma4Swift

/// A-06 : famille lue dans config.json, 12B comprise ; modele inconnu = nil (plus de repli E2B).
@Suite("Famille LoRA depuis config.json")
struct LoRAFamilyDetectionTests {

    private func family(_ json: String) throws -> Gemma4LoRADefaults.ModelFamily? {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("gemma4-family-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try Data(json.utf8).write(to: dir.appendingPathComponent("config.json"))
        return Gemma4LoRADefaults.ModelFamily.from(directory: dir)
    }

    @Test("E2B, E4B, 12B, 26B-A4B, 31B ; inconnu = nil ; nombres de couches justes")
    func testDetection() throws {
        #expect(try family(#"{"model_type":"gemma4","text_config":{"num_hidden_layers":35}}"#) == .e2b)
        #expect(try family(#"{"model_type":"gemma4","text_config":{"num_hidden_layers":42}}"#) == .e4b)
        #expect(try family(#"{"model_type":"gemma4_unified","text_config":{"num_hidden_layers":48}}"#) == .b12b)
        #expect(try family(#"{"model_type":"gemma4","text_config":{"num_hidden_layers":30,"enable_moe_block":true}}"#) == .a4b)
        #expect(try family(#"{"model_type":"gemma4","text_config":{"num_hidden_layers":60}}"#) == .dense31b)
        #expect(try family(#"{"model_type":"gemma4","text_config":{"num_hidden_layers":13}}"#) == nil)
        #expect(Gemma4LoRADefaults.ModelFamily.a4b.totalLayers == 30)
        #expect(Gemma4LoRADefaults.ModelFamily.dense31b.totalLayers == 60)
    }
}
