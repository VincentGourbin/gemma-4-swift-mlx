// Port de audio.py AudioAttention — Attention locale par chunks avec position relative

import Foundation
import MLX
import MLXNN

/// Attention locale par chunks avec embeddings de position relative et logit softcapping
public class AudioAttention: Module {
    let numHeads: Int
    let hiddenSize: Int
    let headDim: Int
    let chunkSize: Int
    let maxFutureHorizon: Int
    let maxPastHorizon: Int
    let contextSize: Int
    let invalidLogitsValue: Float
    let softcap: Float
    let qScale: Float
    let kScale: Float

    @ModuleInfo(key: "q_proj") var qProj: ClippableLinear
    @ModuleInfo(key: "k_proj") var kProj: ClippableLinear
    @ModuleInfo(key: "v_proj") var vProj: ClippableLinear
    @ModuleInfo var post: ClippableLinear
    @ModuleInfo(key: "relative_k_proj") var relativeKProj: Linear
    @ModuleInfo(key: "per_dim_scale") var perDimScale: MLXArray

    public init(_ config: Gemma4AudioConfig) {
        self.numHeads = config.numAttentionHeads
        self.hiddenSize = config.hiddenSize
        self.headDim = config.hiddenSize / config.numAttentionHeads
        self.chunkSize = config.attentionChunkSize
        self.maxFutureHorizon = config.attentionContextRight
        self.maxPastHorizon = max(0, config.attentionContextLeft - 1)
        self.contextSize = chunkSize + maxPastHorizon + maxFutureHorizon
        self.invalidLogitsValue = config.attentionInvalidLogitsValue
        self.softcap = config.attentionLogitCap

        self.qScale = pow(Float(hiddenSize / numHeads), -0.5) / log(2.0)
        self.kScale = log(1.0 + exp(1.0)) / log(2.0)

        self._qProj.wrappedValue = ClippableLinear(inFeatures: hiddenSize, outFeatures: numHeads * headDim)
        self._kProj.wrappedValue = ClippableLinear(inFeatures: hiddenSize, outFeatures: numHeads * headDim)
        self._vProj.wrappedValue = ClippableLinear(inFeatures: hiddenSize, outFeatures: numHeads * headDim)
        self._post.wrappedValue = ClippableLinear(inFeatures: hiddenSize, outFeatures: hiddenSize)
        self._relativeKProj.wrappedValue = Linear(hiddenSize, numHeads * headDim, bias: false)
        self._perDimScale.wrappedValue = MLXArray.zeros([headDim])

        super.init()
    }

    /// Decoupe la sequence en blocks de taille chunkSize
    func convertToBlock(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0)
        let T = x.dim(1)
        let rest = Array(x.shape[2...])
        let numBlocks = (T + chunkSize - 1) / chunkSize
        let padLen = numBlocks * chunkSize - T
        var result = x
        if padLen > 0 {
            result = padded(result, widths: [.init(0), .init((0, padLen))] + rest.map { _ in .init(0) })
        }
        return result.reshaped([B, numBlocks, chunkSize] + rest)
    }

    /// Extrait le contexte local pour chaque block
    func extractBlockContext(_ x: MLXArray) -> MLXArray {
        let padLeft = maxPastHorizon
        let padRight = maxFutureHorizon + chunkSize - 1
        let rest = Array(x.shape[2...])
        var result = padded(x, widths: [.init(0), .init((padLeft, padRight))] + rest.map { _ in .init(0) })

        let TPadded = result.dim(1)
        let numBlocks = (TPadded - contextSize) / chunkSize + 1

        // Construire les indices pour extraire les fenetres de contexte
        let starts = MLXArray(stride(from: 0, to: numBlocks * chunkSize, by: chunkSize))
        let offsets = MLXArray(0 ..< Int32(contextSize))

        // [numBlocks, contextSize]
        let indices = expandedDimensions(starts, axis: 1) + expandedDimensions(offsets, axis: 0)
        return result[0..., indices]
    }

    /// Relative position shift (ref: Transformer-XL appendix B, huggingface/papers/1901.02860)
    func relShift(_ x: MLXArray) -> MLXArray {
        let shape = x.shape
        let B = shape[0], N = shape[1], U = shape[2], W = shape[3], posLen = shape[4]
        let C = contextSize

        // Pad, reshape, trim to align relative positions with context windows
        var result = padded(x, widths: [.init(0), .init(0), .init(0), .init(0), .init((0, C + 1 - posLen))])
        result = result.reshaped(B, N, U, W * (C + 1))
        result = result[0..., 0..., 0..., 0 ..< W * C]
        return result.reshaped(B, N, U, W, C)
    }

    public func callAsFunction(_ hiddenStates: MLXArray, mask: MLXArray, causalValidMask: MLXArray, positionEmbeddings: MLXArray) -> MLXArray {
        let B = hiddenStates.dim(0)
        let T = hiddenStates.dim(1)

        var q = qProj(hiddenStates).asType(.float32).reshaped(B, T, numHeads, headDim)
        var k = kProj(hiddenStates).asType(.float32).reshaped(B, T, numHeads, headDim)
        let v = vProj(hiddenStates).asType(.float32).reshaped(B, T, numHeads, headDim)

        let dimScale = softplus(perDimScale)
        q = q * (qScale * dimScale)
        k = k * kScale

        let queryBlocks = convertToBlock(q)
        let keyBlocks = extractBlockContext(k)
        let valueBlocks = extractBlockContext(v)
        let numBlocks = queryBlocks.dim(1)

        // matrix_ac: standard Q·K^T
        let queries = queryBlocks.transposed(0, 3, 1, 2, 4)  // [B, N, U, W, H]
        let kT = keyBlocks.transposed(0, 3, 1, 4, 2)          // [B, N, U, H, C]
        let matrixAC = matmul(queries, kT)                     // [B, N, U, W, C]

        // matrix_bd: Q · RelativeK^T (relative position bias)
        // positionEmbeddings: [posLen, hidden_size]
        var relativeKeyStates = relativeKProj(positionEmbeddings.asType(hiddenStates.dtype))
        // [posLen, numHeads * headDim] → [posLen, numHeads, headDim]
        let posLen = relativeKeyStates.dim(0)
        relativeKeyStates = relativeKeyStates.reshaped(posLen, numHeads, headDim)
        // Permute to [numHeads, headDim, posLen]
        let relKT = relativeKeyStates.transposed(1, 2, 0)

        // queries_flat: [B, N, U*W, H]
        let queriesFlat = queries.reshaped(B, numHeads, -1, headDim)
        // matmul: [B, N, U*W, H] x [1, N, H, posLen] → [B, N, U*W, posLen]
        var matrixBD = matmul(queriesFlat, expandedDimensions(relKT, axis: 0))
        // Reshape back: [B, N, U, W, posLen]
        matrixBD = matrixBD.reshaped(B, numHeads, numBlocks, chunkSize, -1)
        matrixBD = relShift(matrixBD)

        // Combine: attn_weights = matrix_ac + matrix_bd
        var logits = matrixAC + matrixBD

        // Softcap
        logits = tanh(logits / softcap) * softcap

        // 1. Masque causal+validite
        let causalExpanded = expandedDimensions(expandedDimensions(expandedDimensions(causalValidMask, axis: 0), axis: 0), axis: 0)
        logits = MLX.where(causalExpanded, logits, MLXArray(invalidLogitsValue))

        // 2. Masque de padding
        let validMask = logicalNot(mask)
        let extractedValid = extractBlockContext(expandedDimensions(validMask, axis: -1)).squeezed(axis: -1)
        let paddingCondition = expandedDimensions(expandedDimensions(extractedValid, axis: 1), axis: 3)
        logits = MLX.where(paddingCondition, logits, MLXArray(invalidLogitsValue))

        // Softmax + attention
        let probs = softmax(logits, axis: -1)

        // probs [B, N, U, W, C] x V [B, N, U, C, H] → [B, N, U, W, H]
        let vT = valueBlocks.transposed(0, 3, 1, 2, 4)
        var context = matmul(probs, vT)

        // Reshape back: [B, N, U, W, H] → [B, U*W, N*H] → [B, T, D]
        context = context.transposed(0, 2, 3, 1, 4) // [B, U, W, N, H]
        let U = context.dim(1)
        context = context.reshaped(B, U * chunkSize, numHeads, headDim)
        context = context[0..., 0 ..< T]

        context = context.reshaped(B, T, numHeads * headDim)
        return post(context)
    }
}
