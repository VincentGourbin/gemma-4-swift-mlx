// TurboQuant KV Cache — Cache compresse avec quantisation immediate et attention chunked
// Port de turboquant.py : TurboQuantKVCache(_BaseCache)
// Quantise les K/V IMMEDIATEMENT dans update() (pas de buffer BF16)
// Attention chunked avec online softmax pendant le prefill

import Foundation
import MLX
import MLXFast
import MLXLMCommon

/// KV Cache compresse via TurboQuant MSE Codec.
/// Quantise immediatement chaque K/V dans update().
/// Attention chunked (query blocks × key chunks) avec online softmax pour le prefill.
/// Fused Metal kernel pour le decode single-token.
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

    /// Tailles de chunk pour l'attention quantisee pendant le prefill
    let queryBlockSize: Int = 16
    let keyChunkSize: Int = 2048

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

    /// Retourne les normes quantisees comme proxy.
    /// Les couches KV-shared utilisent quantizedAttention() directement.
    public var state: [MLXArray] {
        get {
            guard let ks = currentKeyState, let vs = currentValueState else { return [] }
            return [ks.norms, vs.norms]
        }
        set { /* serialization: pas de support */ }
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

    public func innerState() -> [MLXArray] { state }

    // MARK: - Update (quantisation immediate)

    @discardableResult
    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        ensureCodecs(keys: keys, values: values)
        let nNew = keys.dim(2)

        // Quantiser immediatement — pas de buffer BF16
        let newKeyState = keyCodec!.quantize(keys)
        let newValueState = valueCodec!.quantize(values)
        let newEnd = _offset + nNew

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

        // Eval periodique pour eviter l'explosion du graph
        if nNew > 1 || (_offset % 50 == 0) {
            eval(keyStore!.norms, keyStore!.indices, valueStore!.norms, valueStore!.indices)
        }

        let dummy = MLXArray.zeros([1])
        return (dummy, dummy)
    }

    // MARK: - Quantized Attention

    /// Attention quantisee — route vers fused kernel (L=1) ou chunked (L>1)
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
            let qFlat = qRot.reshaped(B * nQHeads, D)

            if let fusedOut = fusedMSEDecode(
                queries: qFlat, keyState: ks, valueState: vs,
                keyBits: kCodec.bits, valBits: vCodec.bits,
                keyCodebook: kCodec.codebook, valCodebook: vCodec.codebook,
                nRepeats: nRepeats
            ) {
                let outRotated = fusedOut.reshaped(B, nKVHeads, nRepeats, D)
                let output = vCodec.rotateInverse(outRotated)
                return output.reshaped(B, nQHeads, L, D).asType(queries.dtype)
            }
        }

        // Chunked path pour prefill (L > 1) et fallback decode
        return chunkedQuantizedAttention(
            groupedQueries: groupedQueries,
            keyState: ks, valueState: vs,
            kCodec: kCodec, vCodec: vCodec,
            B: B, nKVHeads: nKVHeads, nRepeats: nRepeats, L: L, D: D
        ).asType(queries.dtype)
    }

    // MARK: - Chunked Quantized Attention (online softmax)

    /// Attention chunked : score par blocs de queries × chunks de keys
    /// Online softmax pour la stabilite numerique sans materialiser [L, T] complet
    private func chunkedQuantizedAttention(
        groupedQueries: MLXArray,
        keyState: TurboQuantMSEState,
        valueState: TurboQuantMSEState,
        kCodec: TurboQuantMSECodec,
        vCodec: TurboQuantMSECodec,
        B: Int, nKVHeads: Int, nRepeats: Int, L: Int, D: Int
    ) -> MLXArray {
        let T = keyState.length
        let nQHeads = nKVHeads * nRepeats

        // Tourner toutes les queries d'un coup (une matmul)
        let preparedQueries = kCodec.prepareQueries(groupedQueries)

        // Offset des queries dans la sequence (pour le masque causal)
        // Les queries couvrent les positions [offset - L, offset), les keys [0, T)
        let queryOffset = _offset - L

        var outputBlocks: [MLXArray] = []

        for qStart in stride(from: 0, to: L, by: queryBlockSize) {
            let qEnd = min(qStart + queryBlockSize, L)

            // Slice du bloc de queries : [B, nKVHeads, nRepeats, QB, D]
            let qBlock = preparedQueries[0..., 0..., 0..., qStart ..< qEnd, 0...]

            // Accumulateurs online softmax (float32)
            var maxScore = MLXArray.full([B, nKVHeads, nRepeats, qEnd - qStart], values: MLXArray(Float(-1e30)))
            var sumExp = MLXArray.zeros([B, nKVHeads, nRepeats, qEnd - qStart])
            var outputAccum = MLXArray.zeros([B, nKVHeads, nRepeats, qEnd - qStart, D])

            for kStart in stride(from: 0, to: T, by: keyChunkSize) {
                let kEnd = min(kStart + keyChunkSize, T)

                // Slice des states pour ce chunk
                let keyChunk = sliceStateRange(keyState, start: kStart, end: kEnd)
                let valChunk = sliceStateRange(valueState, start: kStart, end: kEnd)

                // Score : [B, H, R, QB, KC]
                var scores = kCodec.scorePreparedChunk(qBlock, state: keyChunk)

                // Masque causal : query a position absolue (queryOffset + qStart + qi)
                // peut voir les keys a position <= sa position
                scores = applyCausalMask(
                    scores: scores,
                    qAbsStart: queryOffset + qStart,
                    kAbsStart: kStart,
                    QB: qEnd - qStart,
                    KC: kEnd - kStart
                )

                // Online softmax update
                let chunkMax = scores.max(axis: -1)
                let newMax = maximum(maxScore, chunkMax)
                let correction = MLX.exp(maxScore - newMax)
                let expScores = MLX.exp(scores - expandedDimensions(newMax, axis: -1))
                let chunkSum = expScores.sum(axis: -1)

                // Rescale et accumule
                outputAccum = outputAccum * expandedDimensions(correction, axis: -1)
                    + vCodec.weightedSumRotatedChunk(expScores, state: valChunk)
                sumExp = sumExp * correction + chunkSum
                maxScore = newMax
            }

            // Normaliser et rotation inverse
            let normalized = outputAccum / maximum(expandedDimensions(sumExp, axis: -1), MLXArray(1e-10))
            let blockFinal = vCodec.rotateInverse(normalized)
            outputBlocks.append(blockFinal)

            // Eval pour eviter l'explosion du graph
            eval(outputBlocks.last!)
        }

        // Concatener tous les blocs : [B, nKVHeads, nRepeats, L, D]
        let fullOutput: MLXArray
        if outputBlocks.count == 1 {
            fullOutput = outputBlocks[0]
        } else {
            fullOutput = concatenated(outputBlocks, axis: 3)
        }
        return fullOutput.reshaped(B, nQHeads, L, D)
    }

    /// Applique le masque causal pour un chunk query×key
    private func applyCausalMask(
        scores: MLXArray, qAbsStart: Int, kAbsStart: Int, QB: Int, KC: Int
    ) -> MLXArray {
        // Optimisation : si tout le chunk de keys est avant le debut des queries, pas de masque
        if kAbsStart + KC <= qAbsStart { return scores }

        // Construire un petit masque [QB, KC]
        let qPositions = MLXArray(Int32(qAbsStart) ..< Int32(qAbsStart + QB))[0..., .newAxis]
        let kPositions = MLXArray(Int32(kAbsStart) ..< Int32(kAbsStart + KC))[.newAxis, 0...]
        let mask = qPositions .>= kPositions // [QB, KC] bool

        return MLX.where(mask, scores, MLXArray(Float(-1e30)))
    }

    /// Slice un state sur la dimension T (axe 2)
    private func sliceStateRange(_ state: TurboQuantMSEState, start: Int, end: Int) -> TurboQuantMSEState {
        TurboQuantMSEState(
            norms: state.norms[0..., 0..., start ..< end],
            indices: state.indices[0..., 0..., start ..< end, 0...]
        )
    }

    // MARK: - Compression Stats

    public var compressedNbytes: Int {
        (currentKeyState?.nbytes ?? 0) + (currentValueState?.nbytes ?? 0)
    }

    public var effectiveCompressionRatio: Float {
        guard _offset > 0, let kCodec = keyCodec else { return 1.0 }
        let bfSize = _offset * kCodec.dim * 2 * 2
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
        let newState = allocate(like: state, length: newCap)
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
