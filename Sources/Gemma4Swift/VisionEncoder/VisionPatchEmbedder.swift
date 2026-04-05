// Port de vision.py VisionPatchEmbedder — Patchification + position embeddings

import MLX
import MLXNN

/// Convertit les pixels en patches et ajoute les embeddings de position 2D.
public class VisionPatchEmbedder: Module {
    let hiddenSize: Int
    let patchSize: Int
    let positionEmbeddingSize: Int

    @ModuleInfo(key: "input_proj") var inputProj: Linear
    @ModuleInfo(key: "position_embedding_table") var positionEmbeddingTable: MLXArray

    public init(_ config: Gemma4VisionConfig) {
        self.hiddenSize = config.hiddenSize
        self.patchSize = config.patchSize
        self.positionEmbeddingSize = config.positionEmbeddingSize

        // 3 canaux * patchSize^2 → hiddenSize
        self._inputProj.wrappedValue = Linear(3 * patchSize * patchSize, hiddenSize, bias: false)
        // Table de position: [2, positionEmbeddingSize, hiddenSize]
        self._positionEmbeddingTable.wrappedValue = MLXArray.ones([2, positionEmbeddingSize, hiddenSize])

        super.init()
    }

    /// Calcule les embeddings de position a partir des coordonnees de patches
    func positionEmbeddings(patchPositions: MLXArray, paddingPositions: MLXArray) -> MLXArray {
        // one-hot: [B, numPatches, 2, posSize]
        let oh = oneHot(patchPositions, numClasses: positionEmbeddingSize)
        // [B, 2, numPatches, posSize]
        let ohT = oh.transposed(0, 2, 1, 3).asType(positionEmbeddingTable.dtype)
        // matmul: [B, 2, numPatches, hiddenSize]
        var posEmb = matmul(ohT, positionEmbeddingTable)
        // Somme sur la dim spatiale (2 axes: x, y) → [B, numPatches, hiddenSize]
        posEmb = posEmb.sum(axis: 1)
        // Masquer les positions de padding
        let mask = expandedDimensions(paddingPositions, axis: -1)
        posEmb = MLX.where(mask, MLXArray(Float(0.0)), posEmb)
        return posEmb
    }

    /// Decoupe les pixels en patches et projette
    func patchify(_ pixelValues: MLXArray) -> MLXArray {
        // pixelValues: [B, C, H, W] (channel-first du processeur)
        let B = pixelValues.dim(0)
        let C = pixelValues.dim(1)
        let H = pixelValues.dim(2)
        let W = pixelValues.dim(3)
        let pH = H / patchSize
        let pW = W / patchSize

        // [B, C, pH, p, pW, p] → [B, pH, pW, p, p, C] → [B, pH*pW, p*p*C]
        var patches = pixelValues.reshaped(B, C, pH, patchSize, pW, patchSize)
        patches = patches.transposed(0, 2, 4, 3, 5, 1) // [B, pH, pW, p, p, C]
        patches = patches.reshaped(B, pH * pW, C * patchSize * patchSize)
        // Normalise vers [-1, 1]
        patches = 2 * (patches - 0.5)
        return inputProj(patches.asType(inputProj.weight.dtype))
    }

    public func callAsFunction(
        pixelValues: MLXArray,
        patchPositions: MLXArray,
        paddingPositions: MLXArray
    ) -> MLXArray {
        let hidden = patchify(pixelValues)
        let posEmb = positionEmbeddings(patchPositions: patchPositions, paddingPositions: paddingPositions)
        return hidden + posEmb
    }
}

/// One-hot encoding
func oneHot(_ indices: MLXArray, numClasses: Int) -> MLXArray {
    let expanded = expandedDimensions(indices, axis: -1)
    let classes = MLXArray(0 ..< Int32(numClasses))
    return (expanded .== classes).asType(.float32)
}
