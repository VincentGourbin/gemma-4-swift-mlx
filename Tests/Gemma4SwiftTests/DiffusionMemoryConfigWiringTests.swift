import Testing
import Foundation
import MLX
@testable import Gemma4Swift

/// D-10 (audit diffusion) : DiffusionMemoryConfig etait a moitie cable, et le pipeline
/// affirmait qu'unloadVision() fait planter le canvas suivant. Sur CPU (voir
/// DiffusionPipelineTests).
@Suite("Politique memoire de la diffusion", .serialized)
struct DiffusionMemoryConfigWiringTests {

    /// Modele minuscule avec une tour vision (config par defaut, jamais evaluee ici).
    private func modelWithVision() throws -> DiffusionGemmaForBlockDiffusion {
        let json = TinyDiffusion.configJSON.replacingOccurrences(
            of: "\"image_token_id\": 60,", with: """
            "vision_config": {"model_type": "gemma4_vision", "hidden_size": 32, "intermediate_size": 64,
              "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": 16,
              "global_head_dim": 16, "rms_norm_eps": 1e-6, "max_position_embeddings": 64, "patch_size": 16,
              "pooling_kernel_size": 2, "position_embedding_size": 64, "default_output_length": 4,
              "use_clipped_linears": false, "standardize": false},
            "image_token_id": 60,
            """)
        let config = try JSONDecoder().decode(DiffusionGemmaConfig.self, from: Data(json.utf8))
        MLXRandom.seed(5)
        return DiffusionGemmaForBlockDiffusion(config)
    }

    @Test("unloadVision puis encodage incremental et pas de debruitage : pas de plantage")
    func testUnloadVisionThenContinue() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try modelWithVision()
            #expect(model.encoder.hasVisionLoaded)
            let first = model.encodePrompt(promptIds: TinyDiffusion.prompt(), pixelValues: nil, priorCache: nil)
            model.encoder.unloadVision()
            #expect(!model.encoder.hasVisionLoaded)
            let second = model.encodePrompt(
                promptIds: MLXArray([Int32(7), 8]).reshaped(1, 2), pixelValues: nil, priorCache: first.kvCache)
            let logits = model.denoiseStep(
                canvasIds: MLXArray([Int32(1), 2, 3, 4]).reshaped(1, 4), encoderCache: second.kvCache)
            eval(logits)
            #expect(second.kvCache.seqLength == 6)
        }
    }

    @Test("makePipeline transmet la politique memoire du container")
    func testMakePipelinePassesConfig() async throws {
        let config = DiffusionMemoryConfig(
            mixedPrecision: nil, unloadVisionAfterFirstCanvas: true, clearCacheBetweenCanvases: false)
        let pipeline = DiffusionGemmaPipeline(
            model: try TinyDiffusion.model(), genConfig: DiffusionGenerationConfig(), memoryConfig: config)
        let applied = await pipeline.memoryConfig
        #expect(applied.unloadVisionAfterFirstCanvas)
        #expect(!applied.clearCacheBetweenCanvases)
    }

    /// Poids vision au format sanitise (`encoder.vision_tower.*`, `encoder.embed_vision.*`).
    private func visionWeights(_ model: DiffusionGemmaForBlockDiffusion) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (key, value) in model.encoder.parameters().flattened()
            where key.hasPrefix("vision_tower.") || key.hasPrefix("embed_vision.") {
            out["encoder." + key] = value
        }
        return out
    }

    @Test("reloadVision restitue les poids apres unloadVision")
    func testReloadVision() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try modelWithVision()
            let saved = visionWeights(model).mapValues { $0 + 0 }
            eval(Array(saved.values))
            model.encoder.unloadVision()
            #expect(visionWeights(model).values.allSatisfy { $0.size == 0 })
            try model.encoder.reloadVision(from: saved)
            #expect(model.encoder.hasVisionLoaded)
            let reloaded = visionWeights(model)
            #expect(reloaded.count == saved.count)
            for (key, value) in saved {
                #expect(reloaded[key].map { allClose($0, value).item(Bool.self) } ?? false, "\(key)")
            }
        }
    }

    @Test("image apres dechargement sans dossier du modele : refusee, pas ignoree")
    func testImageAfterUnloadWithoutDirectoryIsRejected() async throws {
        let model = try modelWithVision()
        model.encoder.unloadVision()
        let pipeline = DiffusionGemmaPipeline(model: model, genConfig: DiffusionGenerationConfig())
        let count = model.config.visionSoftTokensPerImage
        let prompt = MLXArray(Array(repeating: Int32(model.config.imageTokenId), count: count)).reshaped(1, count)
        nonisolated(unsafe) let pixels = MLXArray.zeros([1, 3, 32, 32])
        let result = await pipeline.generate(promptIds: prompt, pixelValues: pixels, maxBlocks: 1, seed: 0)
        guard case .invalidInput(let message) = result.stopReason else {
            Issue.record("attendu invalidInput, obtenu \(result.stopReason)")
            return
        }
        #expect(message.contains("vision dechargee"))
    }
}
