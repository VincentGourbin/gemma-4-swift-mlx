// Port de gemma4.py masked_scatter — Fusion embeddings texte/vision/audio

import MLX

/// Remplace dans inputTensor les positions ou mask==true par les valeurs de source.
/// Equivalent de PyTorch masked_scatter pour MLX.
public func maskedScatter(input: MLXArray, mask: MLXArray, source: MLXArray) -> MLXArray {
    let maskFlat = mask.flattened().asType(.int32)
    let indices = cumsum(maskFlat, axis: 0) - 1
    let sourceFlat = source.flattened()
    let aligned = sourceFlat[remainder(indices, Int32(source.size))]
    return MLX.where(maskFlat, aligned, input.flattened()).reshaped(input.shape)
}
