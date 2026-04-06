// TurboQuant MSE Codec — Quantisation vectorielle par rotation + codebook optimal
// Port de turboquant.py : _TurboQuantMSECodec

import Foundation
import MLX
import MLXFast

/// Codec MSE TurboQuant : normalise → rotation (RHT ou dense) → quantise → pack
/// Decodage : unpack → codebook lookup → inverse rotation → denormalise
public final class TurboQuantMSECodec: @unchecked Sendable {
    public let dim: Int
    public let bits: Int
    public let useRHT: Bool

    /// Matrice de rotation dense [D, D] (toujours disponible)
    public let rotation: MLXArray
    /// Rotation transposee [D, D]
    public let rotationT: MLXArray
    /// Vecteur de signes pour RHT (nil si dense rotation)
    public let signs: MLXArray?
    /// Codebook [2^bits] centroides optimaux
    public let codebook: MLXArray
    /// Midpoints entre centroides pour binary search [(2^bits)-1]
    let midpoints: MLXArray

    public init(dim: Int, bits: Int, seed: UInt64) {
        self.dim = dim
        self.bits = bits
        // RHT desactive — le butterfly Swift ne produit pas les bons resultats.
        // La rotation dense est mathematiquement correcte pour toutes les dimensions.
        // TODO: reimplementer le WHT via Metal kernel pour O(D log D)
        self.useRHT = false

        if useRHT {
            self.signs = turboQuantRHTSignVector(dim: dim, seed: seed)
        } else {
            self.signs = nil
        }

        self.rotation = turboQuantRotationMatrix(dim: dim, seed: seed)
        self.rotationT = dim > 0 ? rotation.transposed() : rotation
        self.codebook = turboQuantCodebook(dim: dim, bits: bits)

        if bits > 0 && codebook.shape[0] > 1 {
            self.midpoints = (codebook[0 ..< codebook.shape[0] - 1] + codebook[1...]) / 2
        } else {
            self.midpoints = MLXArray.zeros([0])
        }
    }

    // MARK: - Rotation

    func rotateForward(_ x: MLXArray) -> MLXArray {
        if useRHT, let signs = signs {
            return rhtForward(x, signs: signs)
        }
        return matmul(x, rotationT)
    }

    func rotateInverse(_ x: MLXArray) -> MLXArray {
        if useRHT, let signs = signs {
            return rhtInverse(x, signs: signs)
        }
        return matmul(x, rotation)
    }

    // MARK: - Quantize / Dequantize

    /// Quantise unit vectors via binary search sur les midpoints
    func quantizeUnit(_ unitVectors: MLXArray) -> MLXArray {
        guard bits > 0 else {
            return MLXArray.zeros(Array(unitVectors.shape.dropLast()) + [0], dtype: .uint32)
        }
        let rotated = rotateForward(unitVectors)
        // Binary search: accumulate comparisons vs midpoints
        var indices = MLXArray.zeros(rotated.shape, dtype: .uint32)
        for m in 0 ..< midpoints.shape[0] {
            indices = indices + (rotated .> midpoints[m]).asType(.uint32)
        }
        eval(indices)
        return turboQuantPackLowbit(indices, bits: bits)
    }

    /// Quantise avec retour de l'estimation (pour le codec Prod)
    func quantizeUnitWithEstimate(_ unitVectors: MLXArray) -> (packed: MLXArray, estimate: MLXArray) {
        guard bits > 0 else {
            return (
                MLXArray.zeros(Array(unitVectors.shape.dropLast()) + [0], dtype: .uint32),
                MLXArray.zeros(unitVectors.shape)
            )
        }
        let rotated = rotateForward(unitVectors)
        var indices = MLXArray.zeros(rotated.shape, dtype: .uint32)
        for m in 0 ..< midpoints.shape[0] {
            indices = indices + (rotated .> midpoints[m]).asType(.uint32)
        }
        eval(indices)
        let packed = turboQuantPackLowbit(indices, bits: bits)
        let estimatedRotated = codebook[indices.asType(.int32)]
        return (packed, rotateInverse(estimatedRotated))
    }

    /// Dequantise : unpack → lookup → inverse rotate
    func dequantizeUnit(_ packedIndices: MLXArray) -> MLXArray {
        guard bits > 0 else {
            return MLXArray.zeros(Array(packedIndices.shape.dropLast()) + [dim])
        }
        let indices = turboQuantUnpackLowbit(packedIndices, bits: bits, length: dim).asType(.int32)
        let rotated = codebook[indices]
        return rotateInverse(rotated)
    }

    // MARK: - Public API

    /// Quantise des vecteurs KV complets → TurboQuantMSEState
    public func quantize(_ vectors: MLXArray) -> TurboQuantMSEState {
        let vectorsF32 = vectors.asType(.float32)
        let norms = norm(vectorsF32, axis: -1)
        let unitVectors = vectorsF32 / maximum(norms[.ellipsis, .newAxis], MLXArray(TURBOQUANT_EPS))
        return TurboQuantMSEState(
            norms: norms.asType(.float16),
            indices: quantizeUnit(unitVectors)
        )
    }

    /// Dequantise un state → vecteurs approximes
    public func dequantize(_ state: TurboQuantMSEState) -> MLXArray {
        let unitVectors = dequantizeUnit(state.indices)
        return state.norms[.ellipsis, .newAxis].asType(unitVectors.dtype) * unitVectors
    }

    /// Prepare les queries en les tournant dans l'espace du codebook
    public func prepareQueries(_ queries: MLXArray) -> MLXArray {
        rotateForward(queries)
    }

    /// Score : dot product entre queries preparees et keys quantisees
    public func scorePrepared(_ preparedQueries: MLXArray, state: TurboQuantMSEState) -> MLXArray {
        // Fallback MLX pur (pas de Metal kernel pour l'instant)
        let indices = turboQuantUnpackLowbit(state.indices, bits: bits, length: dim).asType(.int32)
        let rotated = codebook[indices]
        // preparedQueries: [B, nKVHeads, R, L, D], rotated: [B, nKVHeads, T, D]
        let dots = MLX.einsum("bhmld,bhtd->bhmlt", preparedQueries, rotated)
        return dots * state.norms.asType(.float32)[0..., 0..., .newAxis, .newAxis, 0...]
    }

    /// Score queries non-preparees
    public func score(_ queries: MLXArray, state: TurboQuantMSEState) -> MLXArray {
        scorePrepared(prepareQueries(queries), state: state)
    }

    /// Somme ponderee des valeurs quantisees
    public func weightedSum(_ weights: MLXArray, state: TurboQuantMSEState) -> MLXArray {
        let indices = turboQuantUnpackLowbit(state.indices, bits: bits, length: dim).asType(.int32)
        let rotated = codebook[indices]
        // weights: [B, H, R, L, T], norms: [B, H, T], rotated: [B, H, T, D]
        let weightedRot = MLX.einsum(
            "bhmlt,bht,bhtd->bhmld",
            weights,
            state.norms.asType(.float32),
            rotated
        )
        return rotateInverse(weightedRot)
    }

    /// Somme ponderee depuis des scores (applique softmax d'abord)
    public func weightedSumFromScores(_ scores: MLXArray, state: TurboQuantMSEState) -> MLXArray {
        weightedSum(softmax(scores, axis: -1), state: state)
    }
}

// MARK: - Metal Score Kernel

/// Kernel Metal pour le scoring MSE : dot product query × packed codebook entries
/// Utilise simd_sum pour la reduction parallele
let turboQuantMSEScoreKernel: MLXFastKernel = MLXFast.metalKernel(
    name: "turboquant_mse_score",
    inputNames: ["q_rot", "norms", "packed", "codebook"],
    outputNames: ["out"],
    source: """
        auto lane = thread_position_in_grid.x;
        auto repeat_idx = thread_position_in_grid.y;
        auto n = thread_position_in_grid.z;

        auto token_count = norms_shape[2];
        auto kv_heads = norms_shape[1];
        auto repeat_count = q_rot_shape[2];
        if (repeat_idx >= repeat_count) {
            return;
        }

        auto b = n / (kv_heads * token_count);
        auto rem = n % (kv_heads * token_count);
        auto h = rem / token_count;
        auto t = rem % token_count;

        auto q_ptr = q_rot + ((b * kv_heads + h) * repeat_count + repeat_idx) * Dim;
        auto packed_ptr = packed + ((b * kv_heads + h) * token_count + t) * PackedWidth;

        float acc = 0.0f;
        for (int d = lane; d < Dim; d += 32) {
            int bit_offset = d * Bits;
            int word_idx = bit_offset / 32;
            int offset = bit_offset % 32;
            uint value = packed_ptr[word_idx] >> offset;
            int spill = offset + Bits - 32;
            if (spill > 0) {
                value |= packed_ptr[word_idx + 1] << (Bits - spill);
            }
            value &= ((1u << Bits) - 1u);
            acc += static_cast<float>(q_ptr[d]) * codebook[value];
        }

        acc = simd_sum(acc);
        if (thread_index_in_simdgroup == 0) {
            out[((b * kv_heads + h) * repeat_count + repeat_idx) * token_count + t] =
                acc * static_cast<float>(norms[(b * kv_heads + h) * token_count + t]);
        }
    """
)

/// Kernel Metal pour weighted sum en espace tourne
let turboQuantMSEWeightedRotKernel: MLXFastKernel = MLXFast.metalKernel(
    name: "turboquant_mse_weighted_rot",
    inputNames: ["weights", "norms", "packed", "codebook"],
    outputNames: ["out"],
    source: """
        auto lane = thread_position_in_grid.x;
        auto dim_idx = thread_position_in_grid.y;
        auto n = thread_position_in_grid.z;

        if (dim_idx >= Dim) {
            return;
        }

        auto token_count = norms_shape[2];
        auto kv_heads = norms_shape[1];
        auto repeat_count = weights_shape[2];
        auto b = n / (kv_heads * repeat_count);
        auto rem = n % (kv_heads * repeat_count);
        auto h = rem / repeat_count;
        auto repeat_idx = rem % repeat_count;

        auto weights_ptr = weights + ((b * kv_heads + h) * repeat_count + repeat_idx) * token_count;
        auto norms_ptr = norms + (b * kv_heads + h) * token_count;
        auto packed_ptr = packed + ((b * kv_heads + h) * token_count) * PackedWidth;

        float acc = 0.0f;
        for (int t = lane; t < token_count; t += 32) {
            auto token_ptr = packed_ptr + t * PackedWidth;
            int bit_offset = dim_idx * Bits;
            int word_idx = bit_offset / 32;
            int offset = bit_offset % 32;
            uint value = token_ptr[word_idx] >> offset;
            int spill = offset + Bits - 32;
            if (spill > 0) {
                value |= token_ptr[word_idx + 1] << (Bits - spill);
            }
            value &= ((1u << Bits) - 1u);
            acc += static_cast<float>(weights_ptr[t])
                * static_cast<float>(norms_ptr[t])
                * codebook[value];
        }

        acc = simd_sum(acc);
        if (thread_index_in_simdgroup == 0) {
            out[((b * kv_heads + h) * repeat_count + repeat_idx) * Dim + dim_idx] = acc;
        }
    """
)
