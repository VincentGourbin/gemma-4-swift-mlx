// Port de vision.py VisionPooler — Average pooling spatial

import MLX
import MLXNN

/// Pooler spatial : reduit le nombre de tokens par average pooling basee sur les positions
public class VisionPooler: Module {
    let hiddenSize: Int
    let defaultOutputLength: Int
    let rootHiddenSize: Float

    public init(_ config: Gemma4VisionConfig) {
        self.hiddenSize = config.hiddenSize
        self.defaultOutputLength = config.defaultOutputLength
        self.rootHiddenSize = Float(config.hiddenSize).squareRoot()
        super.init()
    }

    func avgPoolByPositions(
        _ x: MLXArray, patchPositions: MLXArray, length: Int, kernelSize: Int? = nil
    ) -> (MLXArray, MLXArray) {
        // Sans `kernelSize`, deduit de la longueur d'entree : suppose l'entree completee
        // a length * k^2 positions (ancien chemin paddé).
        let k = kernelSize ?? Int((Float(x.dim(1)) / Float(length)).squareRoot())
        let kSquared = Float(k * k)

        let clamped = clip(patchPositions, min: Int32(0))
        let maxX = clamped[.ellipsis, 0].max(axis: -1, keepDims: true) + 1
        let kernelIdxs = floor(clamped.asType(.float32) / Float(k)).asType(.int32)
        let linearIdxs = kernelIdxs[.ellipsis, 0] + (maxX / Int32(k)) * kernelIdxs[.ellipsis, 1]

        let weights = oneHot(linearIdxs, numClasses: length).asType(.float32) / kSquared
        // [B, L, length] x [B, L, D] → [B, length, D]
        let output = matmul(weights.transposed(0, 2, 1), x).asType(x.dtype)
        // Masque: True = valide
        let mask = logicalNot(all(weights .== 0, axis: 1))
        return (output, mask)
    }

    public func callAsFunction(
        hiddenStates: MLXArray,
        patchPositions: MLXArray,
        paddingPositions: MLXArray,
        outputLength: Int? = nil,
        kernelSize: Int? = nil
    ) -> (MLXArray, MLXArray) {
        // Zero out padding (au dtype des etats : une constante fp32 promouvait la sortie)
        var states = MLX.where(
            expandedDimensions(paddingPositions, axis: -1),
            MLXArray(Float(0.0)).asType(hiddenStates.dtype), hiddenStates)

        let length = outputLength ?? defaultOutputLength
        let mask: MLXArray
        // Noyau impose (entree non paddee) : toujours pooler, meme si le nombre de
        // patches reels vaut `length` (l'entree paddee ne l'egalait jamais).
        if kernelSize == nil && states.dim(1) == length {
            mask = paddingPositions
        } else {
            (states, mask) = avgPoolByPositions(
                states, patchPositions: patchPositions, length: length, kernelSize: kernelSize)
        }
        states = states * MLXArray(rootHiddenSize).asType(states.dtype)
        return (states, mask)
    }
}
