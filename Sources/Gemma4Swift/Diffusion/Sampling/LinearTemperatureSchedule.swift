// Port de LinearTemperatureScheduleLogitsProcessor
//
// Python ref (generation_diffusion_gemma.py) :
//   temperature = t_min + (t_max - t_min) * (cur_step / max_denoising_steps)
//   return scores / temperature
//
// Pour cur_step=0 -> tMin (denoising "froid", deterministe)
// Pour cur_step=max -> tMax (denoising "chaud", exploration)

import Foundation
import MLX

/// Schedule lineaire de temperature appliquee aux logits avant sampling.
public struct LinearTemperatureSchedule: Sendable {
    public let tMin: Float
    public let tMax: Float
    public let maxDenoisingSteps: Int

    public init(tMin: Float, tMax: Float, maxDenoisingSteps: Int) {
        self.tMin = tMin
        self.tMax = tMax
        self.maxDenoisingSteps = maxDenoisingSteps
    }

    /// Temperature scalaire au step donne.
    public func temperature(curStep: Int) -> Float {
        let frac = Float(curStep) / Float(maxDenoisingSteps)
        return tMin + (tMax - tMin) * frac
    }

    /// Applique la temperature aux logits : `logits / T`.
    public func apply(_ logits: MLXArray, curStep: Int) -> MLXArray {
        let t = temperature(curStep: curStep)
        return logits / MLXArray(t)
    }
}
