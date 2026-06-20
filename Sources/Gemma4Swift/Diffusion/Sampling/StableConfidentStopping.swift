// Port de StableAndConfidentStoppingCriteria
//
// Python ref (generation_diffusion_gemma.py) :
//   class StableAndConfidentStoppingCriteria:
//       def __call__(self, argmax_canvas, logits):
//           stable = (argmax_history == argmax_canvas[None]).all(dim=-1).all(dim=0)
//           confident = mean(token_entropy, dim=-1) < confidence_threshold
//           return stable & confident
//
// "Stable" = l'argmax du canvas n'a pas bouge pendant `stabilityThreshold` etapes consecutives.
// "Confident" = l'entropie moyenne par exemple est sous le seuil.

import Foundation
import MLX

/// Critere d'arret combine : stable (argmax inchange) ET confiant (entropie basse).
public final class StableConfidentStopping: @unchecked Sendable {
    public let stabilityThreshold: Int
    public let confidenceThreshold: Float

    /// Historique des argmax recents. Capacite = stabilityThreshold.
    /// Chaque element shape : `[B, T]` int.
    private var argmaxHistory: [MLXArray] = []

    public init(stabilityThreshold: Int, confidenceThreshold: Float) {
        self.stabilityThreshold = stabilityThreshold
        self.confidenceThreshold = confidenceThreshold
    }

    public func reset() {
        argmaxHistory.removeAll()
    }

    /// Verifie si l'arret est decide pour chaque exemple du batch.
    /// - Parameters:
    ///   - argmaxCanvas : `[B, T]`, int. Argmax du canvas actuel.
    ///   - logits : `[B, T, V]`, float. Logits du denoiser cette etape.
    /// - Returns: `[B]`, bool. True pour les exemples qui ont converge.
    public func shouldStop(argmaxCanvas: MLXArray, logits: MLXArray) -> MLXArray {
        argmaxHistory.append(argmaxCanvas)
        if argmaxHistory.count > stabilityThreshold {
            argmaxHistory.removeFirst()
        }

        // Stable : tous les argmax dans l'historique == argmax courant.
        let stable: MLXArray
        if argmaxHistory.count < stabilityThreshold {
            // Pas assez d'historique : on ne peut pas dire stable.
            stable = MLXArray.zeros([argmaxCanvas.shape[0]]).asType(.bool)
        } else {
            // AND sur toute la fenetre, AND sur l'axe T
            var allEqual: MLXArray = argmaxHistory[0] .== argmaxCanvas
            for i in 1..<argmaxHistory.count {
                allEqual = allEqual .&& (argmaxHistory[i] .== argmaxCanvas)
            }
            stable = allEqual.all(axis: -1)  // [B]
        }

        // Confident : mean(entropy, axis=-1) < confidence_threshold
        let entropy = EntropyBoundSampler.tokenEntropy(logits)  // [B, T]
        let meanEntropy = entropy.mean(axis: -1)                // [B]
        let confident = meanEntropy .< MLXArray(confidenceThreshold)

        return stable .&& confident
    }
}
