// BitPacking — Pack/unpack N-bit indices dans uint32 avec Metal kernels
// Port de turboquant.py : _pack_lowbit, _unpack_lowbit, _packed_width

import Foundation
import MLX
import MLXFast

// MARK: - Packed Width

/// Nombre de mots uint32 necessaires pour stocker `length` valeurs a `bits` bits
func turboQuantPackedWidth(length: Int, bits: Int) -> Int {
    guard length > 0, bits > 0 else { return 0 }
    return (length * bits + 31) / 32
}

// MARK: - Metal Kernels (lazy init)

private let _packKernel: MLXFastKernel = MLXFast.metalKernel(
    name: "turboquant_pack_lowbit",
    inputNames: ["values"],
    outputNames: ["out"],
    source: """
        auto word = thread_position_in_grid.x;
        auto row = thread_position_in_grid.y;

        if (row >= values_shape[0] || word >= PackedWidth) {
            return;
        }

        auto values_ptr = values + row * Length;
        uint packed_word = 0u;
        int start = max(0, (int(word) * 32 - (Bits - 1)) / Bits);
        int end = min(Length, ((int(word) + 1) * 32 + (Bits - 1)) / Bits);

        for (int idx = start; idx < end; ++idx) {
            int bit_offset = idx * Bits;
            int word_idx = bit_offset / 32;
            int offset = bit_offset % 32;
            uint value = values_ptr[idx] & ((1u << Bits) - 1u);
            if (word_idx == int(word)) {
                packed_word |= value << offset;
            }
            if (word_idx + 1 == int(word)) {
                int spill = offset + Bits - 32;
                if (spill > 0) {
                    packed_word |= value >> (Bits - spill);
                }
            }
        }

        out[row * PackedWidth + word] = packed_word;
    """
)

private let _unpackKernel: MLXFastKernel = MLXFast.metalKernel(
    name: "turboquant_unpack_lowbit",
    inputNames: ["packed"],
    outputNames: ["out"],
    source: """
        auto idx = thread_position_in_grid.x;
        auto row = thread_position_in_grid.y;

        if (row >= packed_shape[0] || idx >= Length) {
            return;
        }

        auto packed_ptr = packed + row * PackedWidth;
        int bit_offset = idx * Bits;
        int word_idx = bit_offset / 32;
        int offset = bit_offset % 32;
        uint value = packed_ptr[word_idx] >> offset;
        int spill = offset + Bits - 32;
        if (spill > 0) {
            value |= packed_ptr[word_idx + 1] << (Bits - spill);
        }
        out[row * Length + idx] = value & ((1u << Bits) - 1u);
    """
)

// MARK: - Pack / Unpack

/// Pack des valeurs N-bit dans des mots uint32 via Metal kernel
/// - Parameter values: [..., length] valeurs uint32 (seuls les `bits` LSB sont utilises)
/// - Parameter bits: nombre de bits par valeur (1-8)
/// - Returns: [..., packedWidth] mots uint32
func turboQuantPackLowbit(_ values: MLXArray, bits: Int) -> MLXArray {
    guard bits > 0 else {
        return MLXArray.zeros(Array(values.shape.dropLast()) + [0], dtype: .uint32)
    }

    let values32 = values.asType(.uint32)
    let length = values32.shape.last!
    let packedWidth = turboQuantPackedWidth(length: length, bits: bits)
    let flat = values32.reshaped(-1, length)
    let rows = flat.shape[0]

    let packed = _packKernel(
        [flat],
        template: [("Bits", bits), ("Length", length), ("PackedWidth", packedWidth)],
        grid: (packedWidth, rows, 1),
        threadGroup: (min(32, packedWidth), 1, 1),
        outputShapes: [[rows, packedWidth]],
        outputDTypes: [.uint32]
    )[0]

    return packed.reshaped(Array(values.shape.dropLast()) + [packedWidth])
}

/// Unpack des mots uint32 en valeurs N-bit via Metal kernel
/// - Parameter packed: [..., packedWidth] mots uint32
/// - Parameter bits: nombre de bits par valeur
/// - Parameter length: nombre de valeurs originales
/// - Returns: [..., length] valeurs uint32
func turboQuantUnpackLowbit(_ packed: MLXArray, bits: Int, length: Int) -> MLXArray {
    guard bits > 0 else {
        return MLXArray.zeros(Array(packed.shape.dropLast()) + [0], dtype: .uint32)
    }

    let packed32 = packed.asType(.uint32)
    let packedWidth = packed32.shape.last!
    let flat = packed32.reshaped(-1, packedWidth)
    let rows = flat.shape[0]

    let unpacked = _unpackKernel(
        [flat],
        template: [("Bits", bits), ("Length", length), ("PackedWidth", packedWidth)],
        grid: (length, rows, 1),
        threadGroup: (min(32, length), 1, 1),
        outputShapes: [[rows, length]],
        outputDTypes: [.uint32]
    )[0]

    return unpacked.reshaped(Array(packed.shape.dropLast()) + [length])
}
