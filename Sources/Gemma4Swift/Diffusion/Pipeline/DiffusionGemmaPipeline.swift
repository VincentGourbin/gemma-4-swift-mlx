// Stub Phase 2 — pipeline de generation block-AR
//
// L'algorithme complet :
//   while not all_blocks_done:
//       encode_prompt_to_kv_cache()          // 1 forward encoder
//       canvas = sampler.initialize_canvas() // random
//       prev_logits = None                    // self-conditioning
//       for step in 0..<max_denoising_steps:
//           logits = decoder(canvas, encoder_kv, self_cond=prev_logits)
//           logits = apply_logit_softcapping(logits)
//           logits = temperature_schedule.apply(logits, step)
//           denoiser_canvas = argmax_or_sample(logits)
//           canvas = sampler.accept(current=canvas, denoiser=denoiser_canvas, logits=logits)
//           if stopping.should_stop(argmax(canvas), logits).all(): break
//           canvas = sampler.renoise(canvas)
//           prev_logits = logits
//       commit_canvas_to_prompt()
//       check_eos_or_max_length()

import Foundation
import MLX

/// Pipeline de generation DiffusionGemma block-AR. STUB Phase 2.
public actor DiffusionGemmaPipeline {
    public let model: DiffusionGemmaForBlockDiffusion
    public let genConfig: DiffusionGenerationConfig
    public let sampler: EntropyBoundSampler
    public let temperatureSchedule: LinearTemperatureSchedule
    public let stopping: StableConfidentStopping

    public init(
        model: DiffusionGemmaForBlockDiffusion,
        genConfig: DiffusionGenerationConfig
    ) {
        self.model = model
        self.genConfig = genConfig
        let vocab = model.config.textConfig.base.vocabSize
        let canvas = model.config.textConfig.canvasLength
        self.sampler = EntropyBoundSampler(
            entropyBound: genConfig.entropyBound,
            vocabSize: vocab,
            canvasLength: canvas
        )
        self.temperatureSchedule = LinearTemperatureSchedule(
            tMin: genConfig.tMin,
            tMax: genConfig.tMax,
            maxDenoisingSteps: genConfig.maxDenoisingSteps
        )
        self.stopping = StableConfidentStopping(
            stabilityThreshold: genConfig.stabilityThreshold,
            confidenceThreshold: genConfig.confidenceThreshold
        )
    }

    // TODO Phase 4-5: implementer generate(promptIds:maxBlocks:)
}
