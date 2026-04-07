// Port de modeling_gemma4.py Gemma4AudioRelPositionalEncoding
// Encodage positionnel relatif sinusoidal pour l'attention audio par chunks

import Foundation
import MLX
import MLXNN

/// Genere des embeddings positionnels sinusoidaux [context_size, hidden_size]
/// pour l'attention relative dans le conformer audio.
/// Positions: context_size-1 → 0 (descending), layout [sin..., cos...]
public class AudioRelPositionalEncoding {
    let hiddenSize: Int
    let maxPastHorizon: Int
    let invTimescales: MLXArray

    public init(_ config: Gemma4AudioConfig) {
        self.hiddenSize = config.hiddenSize
        // Ref Google: torch.arange(12, -1, -1) → max_past_horizon=context_left-1=12
        self.maxPastHorizon = max(0, config.attentionContextLeft - 1)

        let minTimescale: Float = 1.0
        let maxTimescale: Float = 10000.0
        let numTimescales = hiddenSize / 2
        let logIncrement = log(maxTimescale / minTimescale) / Float(max(numTimescales - 1, 1))

        // inv_timescales: [1, numTimescales]
        var scales = [Float](repeating: 0, count: numTimescales)
        for i in 0 ..< numTimescales {
            scales[i] = minTimescale * exp(Float(i) * -logIncrement)
        }
        self.invTimescales = MLXArray(scales).reshaped(1, numTimescales)
    }

    /// Genere les position embeddings: [maxPastHorizon+1, hiddenSize]
    /// Positions: maxPastHorizon → 0 (ref: torch.arange(12, -1, -1))
    public func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        let numPositions = maxPastHorizon + 1 // 13 for context_left=13
        // Position IDs: [numPositions, 1] (descending: maxPastHorizon ... 0)
        let positionIds = MLXArray((0 ..< numPositions).reversed().map { Float($0) }).reshaped(numPositions, 1)

        // scaled_time: [numPositions, numTimescales]
        let scaledTime = matmul(positionIds, invTimescales.asType(hiddenStates.dtype))

        // Concat sin + cos: [numPositions, hiddenSize]
        return concatenated([sin(scaledTime), cos(scaledTime)], axis: -1).asType(hiddenStates.dtype)
    }
}
