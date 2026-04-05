// TurboQuant KV Cache — Cache compresse conforme au protocol KVCache de MLXLMCommon
// Port de turboquant.py : TurboQuantKVCache(_BaseCache)

import Foundation
import MLX
import MLXFast
import MLXLMCommon

/// KV Cache compresse via TurboQuant MSE Codec.
/// Drop-in replacement pour KVCacheSimple avec compression 3-7x du KV cache.
public final class TurboQuantKVCache: @unchecked Sendable, KVCache {
    public let bits: Float
    public let seed: UInt64
    let cacheStep: Int

    private var keyCodec: TurboQuantMSECodec?
    private var valueCodec: TurboQuantMSECodec?
    private var keyStore: TurboQuantMSEState?
    private var valueStore: TurboQuantMSEState?
    private var _offset: Int = 0
    private var _cachedSliced: (TurboQuantMSEState, TurboQuantMSEState)?
    private var _cachedSlicedOffset: Int = -1

    public init(bits: Float = 4.0, seed: UInt64 = 0, cacheStep: Int = 256) {
        self.bits = bits
        self.seed = seed
        self.cacheStep = cacheStep
    }

    // MARK: - Codec Initialization

    private func ensureCodecs(keys: MLXArray, values: MLXArray) {
        if keyCodec == nil {
            let keyBits = Int(floor(bits))
            let valBits = (bits - floor(bits)) > 0.01 ? Int(ceil(bits)) : Int(bits)
            keyCodec = TurboQuantMSECodec(dim: keys.shape.last!, bits: keyBits, seed: seed)
            valueCodec = TurboQuantMSECodec(dim: values.shape.last!, bits: valBits, seed: seed &+ 1)
        }
    }

    // MARK: - KVCache Protocol

    public var offset: Int {
        get { _offset }
        set { _offset = newValue }
    }

    public var maxSize: Int? { nil }

    /// state retourne les K/V decompresses pour la compatibilite avec les couches KV-shared
    /// qui lisent cache.state[0] / cache.state[1] comme des tenseurs rank-4
    public var state: [MLXArray] {
        get {
            guard let ks = currentKeyState, let vs = currentValueState,
                  let kCodec = keyCodec, let vCodec = valueCodec else { return [] }
            return [kCodec.dequantize(ks), vCodec.dequantize(vs)]
        }
        set {
            // Serialization: re-quantiser les K/V
            guard newValue.count >= 2 else { return }
            if keyCodec == nil {
                ensureCodecs(keys: newValue[0], values: newValue[1])
            }
            keyStore = keyCodec!.quantize(newValue[0])
            valueStore = valueCodec!.quantize(newValue[1])
            _offset = newValue[0].dim(2)
            invalidateCache()
        }
    }

    public var metaState: [String] {
        get { ["\(_offset)", "\(bits)", "\(seed)"] }
        set {
            guard newValue.count >= 1 else { return }
            _offset = Int(newValue[0]) ?? 0
        }
    }

    public var isTrimmable: Bool { true }

    @discardableResult
    public func trim(_ n: Int) -> Int {
        let trimmed = min(_offset, n)
        _offset -= trimmed
        invalidateCache()
        return trimmed
    }

    public func copy() -> any KVCache {
        let c = TurboQuantKVCache(bits: bits, seed: seed, cacheStep: cacheStep)
        c._offset = _offset
        c.keyCodec = keyCodec
        c.valueCodec = valueCodec
        c.keyStore = keyStore
        c.valueStore = valueStore
        return c
    }

    public func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        if n == 1 { return .none }
        if returnArray || (windowSize != nil && n > windowSize!) {
            return .array(createCausalMask(n: n, offset: _offset, windowSize: windowSize))
        }
        return .causal
    }

    public func innerState() -> [MLXArray] {
        state
    }

    @discardableResult
    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        ensureCodecs(keys: keys, values: values)

        let newKeyState = keyCodec!.quantize(keys)
        let newValueState = valueCodec!.quantize(values)
        let newEnd = _offset + keys.dim(2)

        if keyStore == nil {
            keyStore = allocate(like: newKeyState, length: max(newEnd, cacheStep))
            valueStore = allocate(like: newValueState, length: max(newEnd, cacheStep))
        } else {
            keyStore = reserve(keyStore!, used: _offset, needed: newEnd)
            valueStore = reserve(valueStore!, used: _offset, needed: newEnd)
        }

        write(dst: &keyStore!, src: newKeyState, start: _offset)
        write(dst: &valueStore!, src: newValueState, start: _offset)

        _offset = newEnd
        invalidateCache()

        if keys.dim(2) > 1 || (_offset % 50 == 0) {
            if let ks = keyStore, let vs = valueStore {
                eval(ks.norms, ks.indices, vs.norms, vs.indices)
            }
        }

        // Decompresser pour la compatibilite avec l'attention standard
        return (keyCodec!.dequantize(currentKeyState!), valueCodec!.dequantize(currentValueState!))
    }

    // MARK: - Quantized Attention (fast path)

    /// Attention directe sur K/V quantises (sans decompression complete)
    /// Utilise le kernel fusionne si possible (single-token decode, D multiple de 32)
    public func quantizedAttention(
        queries: MLXArray,
        scale: Float = 1.0,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .none
    ) -> MLXArray {
        guard let ks = currentKeyState, let vs = currentValueState,
              let kCodec = keyCodec, let vCodec = valueCodec else {
            fatalError("TurboQuantKVCache not initialized")
        }

        let B = queries.dim(0)
        let nQHeads = queries.dim(1)
        let L = queries.dim(2)
        let D = queries.dim(3)
        let nKVHeads = ks.norms.dim(1)
        let nRepeats = nQHeads / nKVHeads

        let groupedQueries = (queries * scale).reshaped(B, nKVHeads, nRepeats, L, D)

        // Fast path: fused Metal kernel pour single-token decode
        if L == 1 && D >= 32 && D % 32 == 0 {
            let qRot = kCodec.prepareQueries(groupedQueries)
            // Flatten: [B, nKVHeads, nRepeats, 1, D] → [B*nQHeads, D]
            let qFlat = qRot.reshaped(B * nQHeads, D)

            if let fusedOut = fusedMSEDecode(
                queries: qFlat,
                keyState: ks,
                valueState: vs,
                keyBits: kCodec.bits,
                valBits: vCodec.bits,
                keyCodebook: kCodec.codebook,
                valCodebook: vCodec.codebook,
                nRepeats: nRepeats
            ) {
                // Output est en espace tourne — appliquer rotation inverse des values
                let outRotated = fusedOut.reshaped(B, nKVHeads, nRepeats, D)
                let output = vCodec.rotateInverse(outRotated)
                return output.reshaped(B, nQHeads, L, D).asType(queries.dtype)
            }
        }

        // Fallback: MLX pur (prefill ou dimensions non-alignees)
        let preparedQueries = kCodec.prepareQueries(groupedQueries)
        let scores = kCodec.scorePrepared(preparedQueries, state: ks)
        let output = vCodec.weightedSumFromScores(scores, state: vs)

        return output.reshaped(B, nQHeads, L, D).asType(queries.dtype)
    }

    // MARK: - Compression Stats

    /// Taille memoire compressée en octets
    public var compressedNbytes: Int {
        (currentKeyState?.nbytes ?? 0) + (currentValueState?.nbytes ?? 0)
    }

    /// Ratio de compression effectif
    public var effectiveCompressionRatio: Float {
        guard _offset > 0, let kCodec = keyCodec else { return 1.0 }
        let bfSize = _offset * kCodec.dim * 2 * 2 // T * D * 2 (K+V) * 2 bytes (float16)
        let compSize = compressedNbytes
        return compSize > 0 ? Float(bfSize) / Float(compSize) : 1.0
    }

    // MARK: - Private Helpers

    private func invalidateCache() {
        _cachedSliced = nil
        _cachedSlicedOffset = -1
    }

    private var currentKeyState: TurboQuantMSEState? {
        guard let ks = keyStore else { return nil }
        if _cachedSlicedOffset == _offset, let cached = _cachedSliced { return cached.0 }
        let kSliced = slice(ks, end: _offset)
        let vSliced = slice(valueStore!, end: _offset)
        _cachedSliced = (kSliced, vSliced)
        _cachedSlicedOffset = _offset
        return kSliced
    }

    private var currentValueState: TurboQuantMSEState? {
        guard valueStore != nil else { return nil }
        if _cachedSlicedOffset == _offset, let cached = _cachedSliced { return cached.1 }
        _ = currentKeyState
        return _cachedSliced?.1
    }

    private func allocate(like state: TurboQuantMSEState, length: Int) -> TurboQuantMSEState {
        TurboQuantMSEState(
            norms: MLXArray.zeros([state.norms.dim(0), state.norms.dim(1), length], dtype: state.norms.dtype),
            indices: MLXArray.zeros([state.indices.dim(0), state.indices.dim(1), length, state.indices.shape.last!], dtype: state.indices.dtype)
        )
    }

    private func reserve(_ state: TurboQuantMSEState, used: Int, needed: Int) -> TurboQuantMSEState {
        let capacity = state.norms.dim(2)
        guard needed > capacity else { return state }
        let newCap = max(needed, capacity + cacheStep)
        var newState = allocate(like: state, length: newCap)
        if used > 0 {
            newState.norms[0..., 0..., 0 ..< used] = state.norms[0..., 0..., 0 ..< used]
            newState.indices[0..., 0..., 0 ..< used, 0...] = state.indices[0..., 0..., 0 ..< used, 0...]
        }
        return newState
    }

    private func write(dst: inout TurboQuantMSEState, src: TurboQuantMSEState, start: Int) {
        let end = start + src.length
        dst.norms[0..., 0..., start ..< end] = src.norms
        dst.indices[0..., 0..., start ..< end, 0...] = src.indices
    }

    private func slice(_ state: TurboQuantMSEState, end: Int) -> TurboQuantMSEState {
        TurboQuantMSEState(
            norms: state.norms[0..., 0..., 0 ..< end],
            indices: state.indices[0..., 0..., 0 ..< end, 0...]
        )
    }
}
