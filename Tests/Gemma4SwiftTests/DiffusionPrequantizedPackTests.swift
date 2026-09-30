import Testing
import Foundation
import MLX
import MLXNN
@testable import Gemma4Swift

/// K-D12 : un pack exporte puis recharge donne les memes sorties que le modele quantifie
/// a la volee, sans lire de bf16, et garde une seule copie des modules partages.
@Suite("Pack pre-quantifie DiffusionGemma", .serialized)
struct DiffusionPrequantizedPackTests {

    /// Modele minuscule aux poids lies encodeur/decodeur, comme apres le sanitizer.
    private func tiedModel() throws -> DiffusionGemmaForBlockDiffusion {
        let model = try TinyDiffusion.model()
        let encoder = Dictionary(model.encoder.languageModel.parameters().flattened(), uniquingKeysWith: { a, _ in a })
        let tied = model.decoder.parameters().flattened().compactMap { key, value -> (String, MLXArray)? in
            guard !key.hasPrefix("self_conditioning."), !key.hasSuffix("layer_scalar"),
                  let source = encoder[key], source.shape == value.shape else { return nil }
            return (key, source)
        }
        model.decoder.update(parameters: ModuleParameters.unflattened(tied))
        return model
    }

    private func tempDir(_ name: String) throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("gemma4-pack-\(name)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private func logits(_ model: DiffusionGemmaForBlockDiffusion) -> MLXArray {
        let enc = model.encodePrompt(promptIds: TinyDiffusion.prompt(), pixelValues: nil, priorCache: nil)
        let out = model.denoiseStep(canvasIds: MLXArray([Int32(1), 2, 3, 4]).reshaped(1, 4), encoderCache: enc.kvCache)
        eval(out)
        return out
    }

    @Test("export puis chargement : memes logits, modules partages, structure quantifiee")
    func testRoundTrip() throws {
        try Device.withDefaultDevice(.cpu) {
            let source = try tempDir("src")
            let pack = try tempDir("pack")
            defer {
                try? FileManager.default.removeItem(at: source)
                try? FileManager.default.removeItem(at: pack)
            }
            try Data(TinyDiffusion.configJSON.utf8).write(to: source.appendingPathComponent("config.json"))

            let model = try tiedModel()
            let config = DiffusionOnTheFlyQuantization.MixedPrecisionConfig(
                highPrecisionLayers: [0], highPrecisionBits: 8, lowPrecisionBits: 4, groupSize: 32)
            DiffusionOnTheFlyQuantization.applyMixedPrecision(to: model, config: config)
            let expected = logits(model)

            let manifest = try DiffusionPrequantizedPack.export(
                model: model, quantization: "test", sourceDirectory: source, to: pack, shardBytes: 4096)
            #expect(manifest.files.count > 1, "petits shards : plusieurs fichiers")
            #expect(FileManager.default.fileExists(atPath: pack.appendingPathComponent("config.json").path))
            #expect(DiffusionPrequantizedPack.isPack(pack))

            // Une seule copie : aucune cle decodeur partagee dans les fichiers.
            var stored = Set<String>()
            for name in manifest.files.keys {
                stored.formUnion(try loadArrays(url: pack.appendingPathComponent(name)).keys)
            }
            #expect(!stored.contains("decoder.layers.0.self_attn.q_proj.weight"))
            #expect(stored.contains("encoder.language_model.layers.0.self_attn.q_proj.weight"))

            let loaded = try DiffusionPrequantizedPack.load(from: pack, includeVision: false, verifyChecksums: true)
            let q0 = loaded.model.decoder.layers[0].leafModules().flattened().first { $0.0 == "self_attn.q_proj" }?.1
            let q1 = loaded.model.decoder.layers[1].leafModules().flattened().first { $0.0 == "self_attn.q_proj" }?.1
            #expect((q0 as? QuantizedLinear)?.bits == 8, "couche 0 en 8 bits")
            #expect((q1 as? QuantizedLinear)?.bits == 4, "couche 1 en 4 bits")
            let encQ0 = loaded.model.encoder.languageModel.layers[0].leafModules().flattened()
                .first { $0.0 == "self_attn.q_proj" }?.1
            #expect(encQ0 != nil && q0 != nil && encQ0! === q0!, "module partage avec l'encodeur")

            let got = logits(loaded.model)
            #expect(got.shape == expected.shape)
            #expect(allClose(got, expected, rtol: 0, atol: 0).item(Bool.self), "memes logits, bit pour bit")
        }
    }

    @Test("fichier modifie : la verification SHA-256 echoue")
    func testChecksumMismatch() throws {
        try Device.withDefaultDevice(.cpu) {
            let source = try tempDir("src")
            let pack = try tempDir("pack")
            defer {
                try? FileManager.default.removeItem(at: source)
                try? FileManager.default.removeItem(at: pack)
            }
            try Data(TinyDiffusion.configJSON.utf8).write(to: source.appendingPathComponent("config.json"))
            let model = try tiedModel()
            DiffusionOnTheFlyQuantization.apply(to: model, bits: 8, groupSize: 32)
            let manifest = try DiffusionPrequantizedPack.export(
                model: model, quantization: "test", sourceDirectory: source, to: pack)
            let file = pack.appendingPathComponent(manifest.files.keys.sorted()[0])
            let handle = try FileHandle(forWritingTo: file)
            try handle.seekToEnd()
            try handle.write(contentsOf: Data([0]))
            try handle.close()
            #expect(throws: DiffusionPrequantizedPack.PackError.self) {
                _ = try DiffusionPrequantizedPack.load(from: pack, includeVision: false, verifyChecksums: true)
            }
        }
    }

    @Test("signature de quantification stable")
    func testSignature() {
        #expect(DiffusionReferenceProfile.Quantization.uniform(bits: 8, groupSize: 64).signature == "uniform-8bit-g64")
        #expect(DiffusionReferenceProfile.Quantization.mixed(.default).signature
            == "mixed-4/8bit-g64-layers[0,1,2,3,26,27,28,29]-sensitive")
    }
}
