// Vision embedder encoder-free pour gemma4_unified.
//
// Architecture (sans transformer) :
//   pixel_patches [B, N, patchDim] (patchDim = 48*48*3)
//   -> patch_ln1  (LayerNorm sur patchDim)
//   -> patch_dense (Linear patchDim -> mmEmbedDim, bias=True)
//   -> patch_ln2  (LayerNorm sur mmEmbedDim)
//   -> + pos_embedding[x_idx, 0] + pos_embedding[y_idx, 1]  (additif si position_ids fournis)
//   -> pos_norm  (LayerNorm)
//
// Port direct du VisionEmbedder Python mlx-vlm/gemma4_unified.

import Foundation
import MLX
import MLXNN

/// Encoder-free patch embedder (Gemma 4 Unified 12B).
public class Gemma4UnifiedVisionEmbedder: Module {
    public let config: Gemma4UnifiedVisionConfig
    public let patchDim: Int

    @ModuleInfo(key: "patch_ln1") var patchLN1: LayerNorm
    @ModuleInfo(key: "patch_dense") var patchDense: Linear
    @ModuleInfo(key: "patch_ln2") var patchLN2: LayerNorm
    @ModuleInfo(key: "pos_norm") var posNorm: LayerNorm

    /// pos_embedding shape : [mmPosembSize, 2, mmEmbedDim].
    /// Axe 1 : 0 = embedding x, 1 = embedding y.
    @ParameterInfo(key: "pos_embedding") var posEmbedding: MLXArray

    public init(_ config: Gemma4UnifiedVisionConfig) {
        self.config = config
        self.patchDim = config.patchDim

        self._patchLN1.wrappedValue = LayerNorm(dimensions: patchDim)
        self._patchDense.wrappedValue = Linear(patchDim, config.mmEmbedDim, bias: true)
        self._patchLN2.wrappedValue = LayerNorm(dimensions: config.mmEmbedDim)
        self._posNorm.wrappedValue = LayerNorm(dimensions: config.mmEmbedDim)
        self._posEmbedding.wrappedValue = MLXArray.zeros(
            [config.mmPosembSize, 2, config.mmEmbedDim]
        )
        super.init()
    }

    /// - Parameters:
    ///   - pixelValues : [B, N, patchDim] (patches flattenes) OU [B, N, _, patchDim] qui sera reshape.
    ///   - imagePositionIds : [B, N, 2] (x_idx, y_idx) avec -1 pour padding. nil = pas de pos embedding.
    /// - Returns: [B, N, mmEmbedDim]
    public func callAsFunction(
        _ pixelValues: MLXArray,
        imagePositionIds: MLXArray? = nil
    ) -> MLXArray {
        // Accept [B, ?, ?, patchDim] et flatten en [B, N, patchDim]
        var x = pixelValues
        if x.ndim == 4 && x.dim(-1) == patchDim {
            x = x.reshaped(x.dim(0), -1, patchDim)
        }

        var hidden = patchLN1(x)
        hidden = patchDense(hidden)
        hidden = patchLN2(hidden)

        if let posIds = imagePositionIds {
            // clamped = max(posIds, 0).int32() ; valid = (posIds != -1).dtype(hidden)
            let clamped = maximum(posIds, MLXArray(Int32(0))).asType(.int32)
            let valid = (posIds .!= MLXArray(Int32(-1))).asType(hidden.dtype)

            // x_pos = pos_embedding[clamped[..., 0], 0]  -> [B, N, mmEmbedDim]
            // y_pos = pos_embedding[clamped[..., 1], 1]
            let xIdx = clamped[.ellipsis, 0]
            let yIdx = clamped[.ellipsis, 1]

            // pos_embedding [P, 2, D] -> [P, D] pour axe x ou y.
            let xLookup = posEmbedding[0..., 0]  // [P, D]
            let yLookup = posEmbedding[0..., 1]  // [P, D]

            // Gather sur axe 0 : prend xLookup[xIdx[b,n]] -> [B, N, D]
            let xPos = xLookup.take(xIdx, axis: 0)
            let yPos = yLookup.take(yIdx, axis: 0)

            // valid shape [B, N, 2]; mask sur axe -1.
            let validX = expandedDimensions(valid[.ellipsis, 0], axis: -1)
            let validY = expandedDimensions(valid[.ellipsis, 1], axis: -1)

            hidden = hidden + (xPos * validX + yPos * validY)
        }

        return posNorm(hidden)
    }
}
