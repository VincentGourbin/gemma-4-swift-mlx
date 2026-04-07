// TurboQuant Decode Kernels — Kernels fusionnes pour attention quantisee
// Port de turboquant.py : _fused_mse_decode_kernel, _mse_score_tiled_kernel

import Foundation
import MLX
import MLXFast

// MARK: - Fused MSE Decode (score + online softmax + weighted value sum)

/// Genere le kernel Metal fusionne pour le decode MSE single-token.
/// 32 simdgroups × 32 lanes : chaque simdgroup traite des tokens differents,
/// chaque lane traite un sous-ensemble des dimensions.
/// Online softmax : pas besoin de materialiser les scores en memoire.
func makeFusedMSEDecodeKernel(keyBits: Int, valBits: Int) -> MLXFastKernel {
    let kMask = (1 << keyBits) - 1
    let vMask = (1 << valBits) - 1

    return MLXFast.metalKernel(
        name: "turboquant_fused_mse_sdpa_k\(keyBits)_v\(valBits)",
        inputNames: [
            "queries", "key_norms", "key_packed", "key_codebook",
            "val_norms", "val_packed", "val_codebook",
        ],
        outputNames: ["out"],
        source: """
            constexpr int BN = 32;
            constexpr int BD = 32;
            constexpr int elems_per_lane = Dim / BD;
            constexpr uint k_mask = \(kMask)u;
            constexpr uint v_mask = \(vMask)u;
            constexpr int k_bits = \(keyBits);
            constexpr int v_bits = \(valBits);

            auto bqh = threadgroup_position_in_grid.x;
            auto sg = simdgroup_index_in_threadgroup;
            auto lane = thread_index_in_simdgroup;

            auto T = key_norms_shape[2];
            auto kv_heads = key_norms_shape[1];
            auto bh = bqh / RepeatCount;

            auto k_nm = key_norms + bh * T;
            auto k_pk = key_packed + bh * T * KPackedWidth;
            auto v_nm = val_norms + bh * T;
            auto v_pk = val_packed + bh * T * VPackedWidth;

            // Shared memory for cross-simdgroup reduction
            threadgroup float max_scores[BN];
            threadgroup float sum_exp_scores[BN];
            threadgroup float shared_mem[BN * BD];

            // Load pre-rotated query into registers
            float q[elems_per_lane];
            auto qr = queries + bqh * Dim + lane * elems_per_lane;
            for (int i = 0; i < elems_per_lane; i++)
                q[i] = static_cast<float>(qr[i]);

            // Accumulators
            float o[elems_per_lane] = {};
            float max_score = -INFINITY;
            float sum_exp = 0;

            // KV loop: each simdgroup handles tokens sg, sg+32, ...
            for (int t = sg; t < (int)T; t += BN) {
                float kn = static_cast<float>(k_nm[t]);
                auto k_ptr = key_packed + (bh * T + t) * KPackedWidth;

                // Key score: dot product with query via packed codebook
                float score = 0.0f;
                for (int i = 0; i < elems_per_lane; i++) {
                    int d = lane * elems_per_lane + i;
                    int bit_off = d * k_bits;
                    int word_idx = bit_off / 32;
                    int offset = bit_off % 32;
                    uint val = k_ptr[word_idx] >> offset;
                    int spill = offset + k_bits - 32;
                    if (spill > 0) val |= k_ptr[word_idx + 1] << (k_bits - spill);
                    val &= k_mask;
                    score += q[i] * key_codebook[val];
                }
                score = simd_sum(score) * kn;

                // Online softmax + value accumulation
                auto v_ptr = val_packed + (bh * T + t) * VPackedWidth;
                float vn = static_cast<float>(v_nm[t]);

                float new_max = max(max_score, score);
                float factor = fast::exp(max_score - new_max);
                float exp_score = fast::exp(score - new_max);
                max_score = new_max;
                sum_exp = sum_exp * factor + exp_score;

                for (int i = 0; i < elems_per_lane; i++) {
                    int d = lane * elems_per_lane + i;
                    int bit_off = d * v_bits;
                    int word_idx = bit_off / 32;
                    int offset = bit_off % 32;
                    uint val = v_ptr[word_idx] >> offset;
                    int spill = offset + v_bits - 32;
                    if (spill > 0) val |= v_ptr[word_idx + 1] << (v_bits - spill);
                    val &= v_mask;
                    o[i] = o[i] * factor + exp_score * val_codebook[val] * vn;
                }
            }

            // Cross-simdgroup reduction
            if (lane == 0) {
                max_scores[sg] = max_score;
                sum_exp_scores[sg] = sum_exp;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float sg_max = max_scores[lane < BN ? lane : 0];
            float new_max = simd_max(sg_max);
            float factor = fast::exp(sg_max - new_max);
            float total_sum = simd_sum((lane < BN ? sum_exp_scores[lane] : 0.0f) * factor);

            float my_factor = fast::exp(max_score - new_max);

            // Transpose-reduce outputs
            for (int i = 0; i < elems_per_lane; i++) {
                shared_mem[lane * BD + sg] = o[i] * my_factor;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                o[i] = simd_sum(shared_mem[sg * BD + lane]);
                o[i] = total_sum > 0 ? o[i] / total_sum : 0;
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            // Write output (rotated space)
            if (lane == 0) {
                for (int i = 0; i < elems_per_lane; i++) {
                    out[bqh * Dim + sg * elems_per_lane + i] = o[i];
                }
            }
        """
    )
}

// MARK: - Cached kernel instances

nonisolated(unsafe) private var _fusedMSEDecodeKernels: [String: MLXFastKernel] = [:]
private let _kernelLock = NSLock()

func getFusedMSEDecodeKernel(keyBits: Int, valBits: Int) -> MLXFastKernel {
    let key = "\(keyBits)_\(valBits)"
    _kernelLock.lock()
    if let cached = _fusedMSEDecodeKernels[key] {
        _kernelLock.unlock()
        return cached
    }
    _kernelLock.unlock()

    let kernel = makeFusedMSEDecodeKernel(keyBits: keyBits, valBits: valBits)

    _kernelLock.lock()
    _fusedMSEDecodeKernels[key] = kernel
    _kernelLock.unlock()

    return kernel
}

// MARK: - Fused MSE Decode Dispatch

/// Execute l'attention fusionnee sur KV quantises (single-token decode).
/// Score + online softmax + weighted value sum en un seul dispatch Metal.
/// Les sorties sont en espace tourne — l'appelant applique la rotation inverse.
///
/// - Parameters:
///   - queries: Pre-rotated queries [B*nQHeads, D]
///   - keyState: Quantised key state (norms + packed indices)
///   - valueState: Quantised value state
///   - keyBits: Bits pour les keys
///   - valBits: Bits pour les values
///   - keyCodebook: Codebook des keys
///   - valCodebook: Codebook des values
///   - nRepeats: GQA repeat count (nQHeads / nKVHeads)
/// - Returns: Output en espace tourne [B*nQHeads, D], ou nil si fallback necessaire
func fusedMSEDecode(
    queries: MLXArray,
    keyState: TurboQuantMSEState,
    valueState: TurboQuantMSEState,
    keyBits: Int,
    valBits: Int,
    keyCodebook: MLXArray,
    valCodebook: MLXArray,
    nRepeats: Int
) -> MLXArray? {
    let D = queries.shape.last!
    let T = keyState.length

    // Le kernel requiert D multiple de 32 et T <= 32768 pour le single-pass
    guard D >= 32, D % 32 == 0, T > 0, T <= 32768 else { return nil }

    let BQH = queries.dim(0) // B * nQHeads
    let kPackedWidth = keyState.indices.shape.last!
    let vPackedWidth = valueState.indices.shape.last!

    let kernel = getFusedMSEDecodeKernel(keyBits: keyBits, valBits: valBits)

    let result = kernel(
        [queries, keyState.norms, keyState.indices, keyCodebook,
         valueState.norms, valueState.indices, valCodebook],
        template: [
            ("Dim", D),
            ("RepeatCount", nRepeats),
            ("KPackedWidth", kPackedWidth),
            ("VPackedWidth", vPackedWidth),
        ],
        grid: (BQH * 1024, 1, 1),
        threadGroup: (1024, 1, 1),  // 32 simdgroups × 32 lanes
        outputShapes: [[BQH, D]],
        outputDTypes: [.float32]
    )[0]

    return result
}
