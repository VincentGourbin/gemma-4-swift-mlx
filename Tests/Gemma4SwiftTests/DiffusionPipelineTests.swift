import Testing
import Foundation
import MLX
@testable import Gemma4Swift

/// Modele DiffusionGemma minuscule (poids aleatoires, graine fixe) pour tester le
/// pipeline sans GPU lourd.
enum TinyDiffusion {
    static let configJSON = """
    {
        "model_type": "diffusion_gemma",
        "text_config": {
            "model_type": "gemma4_text", "hidden_size": 32, "num_hidden_layers": 2,
            "intermediate_size": 64, "num_attention_heads": 2, "head_dim": 16,
            "global_head_dim": 16, "rms_norm_eps": 1e-6, "vocab_size": 64,
            "num_key_value_heads": 1, "num_kv_shared_layers": 0, "sliding_window": 8,
            "sliding_window_pattern": 2, "max_position_embeddings": 256,
            "attention_bias": false, "use_double_wide_mlp": false, "enable_moe_block": false,
            "tie_word_embeddings": true, "final_logit_softcapping": 30.0,
            "rope_parameters": {
                "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
                "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}
            },
            "layer_types": ["sliding_attention", "full_attention"],
            "canvas_length": 4
        },
        "image_token_id": 60, "boi_token_id": 61, "eoi_token_id": 62,
        "vision_soft_tokens_per_image": 4, "tie_word_embeddings": true
    }
    """

    static func model(seed: UInt64 = 7) throws -> DiffusionGemmaForBlockDiffusion {
        let config = try JSONDecoder().decode(DiffusionGemmaConfig.self, from: Data(configJSON.utf8))
        MLXRandom.seed(seed)
        return DiffusionGemmaForBlockDiffusion(config)
    }

    /// Pas de debruitage fixes, pas d'EOS atteignable (ids hors vocabulaire).
    static func pipeline(steps: Int = 6) throws -> DiffusionGemmaPipeline {
        DiffusionGemmaPipeline(
            model: try model(),
            genConfig: DiffusionGenerationConfig(
                maxDenoisingSteps: steps, confidenceThreshold: -1, eosTokenIds: [9_999]))
    }

    static func prompt() -> MLXArray { MLXArray([Int32(2), 5, 9, 11]).reshaped(1, 4) }
}

/// Garde-fous D-07 / D-08 (audit diffusion) : generation annulable, refusee pendant
/// un entrainement, resultats evalues avant de sortir de l'actor.
///
/// Sur CPU : sur GPU, le modele minuscule fait planter une fonction compilee de MLX
/// (`Compiled::eval_gpu`, tampon d'entree nul) des l'encodeur, ancien code compris ;
/// le checkpoint reel tourne sur GPU. Non investigue plus avant (2026-09-28).
@Suite("Pipeline DiffusionGemma", .serialized)
struct DiffusionPipelineTests {

    @Test("generation complete sur le modele minuscule")
    func testCompletes() async throws {
        try await Device.withDefaultDevice(.cpu) {
            let pipeline = try TinyDiffusion.pipeline()
            let result = await pipeline.generate(promptIds: TinyDiffusion.prompt(), maxBlocks: 2)
            #expect(result.stopReason == .completed)
            #expect(result.canvases == 2)
            #expect(result.generatedIds.shape == [1, 8])
        }
    }

    @Test("annulation pendant le premier canvas : arret au pas suivant, rien de commite")
    func testCancellation() async throws {
        try await Device.withDefaultDevice(.cpu) {
            let pipeline = try TinyDiffusion.pipeline(steps: 20)
            let result = await Task {
                await pipeline.generate(
                    promptIds: TinyDiffusion.prompt(), maxBlocks: 3,
                    onStep: { _, _, _ in
                        // Premier pas observe : l'appelant abandonne.
                        withUnsafeCurrentTask { $0?.cancel() }
                    })
            }.value
            #expect(result.stopReason == .cancelled)
            #expect(result.totalDecoderSteps == 1, "\(result.totalDecoderSteps) pas executes")
            #expect(result.canvases == 0)
            #expect(result.generatedIds.dim(1) == 0)
        }
    }

    @Test("entrainement en cours : generation refusee sans aucun pas")
    func testRefusedDuringTraining() async throws {
        try await Device.withDefaultDevice(.cpu) {
            let pipeline = try TinyDiffusion.pipeline()
            try Gemma4ComputeGate.shared.beginTraining()
            defer { Gemma4ComputeGate.shared.endTraining() }
            let result = await pipeline.generate(promptIds: TinyDiffusion.prompt(), maxBlocks: 2)
            #expect(result.stopReason == .trainingInProgress)
            #expect(result.totalDecoderSteps == 0)
        }
    }
}
