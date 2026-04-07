// Port de vision.py VisionAttention — Attention bidirectionnelle + RoPE 2D multidimensionnel

import MLX
import MLXFast
import MLXNN

/// Attention bidirectionnelle pour l'encodeur vision avec RoPE 2D multidimensionnel
public class VisionAttention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let hiddenSize: Int
    let ropeBaseFrequency: Float

    @ModuleInfo(key: "q_proj") var qProj: ClippableLinear
    @ModuleInfo(key: "k_proj") var kProj: ClippableLinear
    @ModuleInfo(key: "v_proj") var vProj: ClippableLinear
    @ModuleInfo(key: "o_proj") var oProj: ClippableLinear
    @ModuleInfo(key: "q_norm") var qNorm: VisionRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: VisionRMSNorm
    @ModuleInfo(key: "_v_norm") var vNorm: VisionRMSNormNoScale

    public init(_ config: Gemma4VisionConfig) {
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.hiddenSize = config.hiddenSize
        self.ropeBaseFrequency = config.ropeTheta

        let clip = config.useClippedLinears
        self._qProj.wrappedValue = ClippableLinear(inFeatures: hiddenSize, outFeatures: numHeads * headDim, useClipping: clip)
        self._kProj.wrappedValue = ClippableLinear(inFeatures: hiddenSize, outFeatures: numKVHeads * headDim, useClipping: clip)
        self._vProj.wrappedValue = ClippableLinear(inFeatures: hiddenSize, outFeatures: numKVHeads * headDim, useClipping: clip)
        self._oProj.wrappedValue = ClippableLinear(inFeatures: numHeads * headDim, outFeatures: hiddenSize, useClipping: clip)

        self._qNorm.wrappedValue = VisionRMSNorm(dimensions: headDim)
        self._kNorm.wrappedValue = VisionRMSNorm(dimensions: headDim)
        self._vNorm.wrappedValue = VisionRMSNormNoScale()

        super.init()
    }

    public func callAsFunction(_ x: MLXArray, positions: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        var q = qProj(x).reshaped(B, L, numHeads, headDim)
        var k = kProj(x).reshaped(B, L, numKVHeads, headDim)
        var v = vProj(x).reshaped(B, L, numKVHeads, headDim)

        q = qNorm(q)
        k = kNorm(k)
        v = vNorm(v)

        // Appliquer RoPE 2D multidimensionnel
        q = applyMultidimensionalRoPE(q, positions: positions, baseFrequency: ropeBaseFrequency)
        k = applyMultidimensionalRoPE(k, positions: positions, baseFrequency: ropeBaseFrequency)

        // [B, L, H, D] → [B, H, L, D]
        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        v = v.transposed(0, 2, 1, 3)

        // Pad head_dim a une taille supportee par le fused SDPA (64, 80, 128)
        // pour eviter les NaN sur les lignes all-masked (tokens padding)
        // head_dim=72 (26B/31B) doit etre padde a 80
        let needsPad = headDim != 64 && headDim != 80 && headDim != 128 && headDim != 256
        var qPad = q, kPad = k, vPad = v
        let targetDim: Int
        if needsPad {
            targetDim = headDim <= 64 ? 64 : headDim <= 80 ? 80 : headDim <= 128 ? 128 : 256
            let padSize = targetDim - headDim
            let padShape = [q.dim(0), q.dim(1), q.dim(2), padSize]
            let zeros = MLXArray.zeros(padShape, dtype: q.dtype)
            qPad = concatenated([q, zeros], axis: -1)
            kPad = concatenated([k, zeros], axis: -1)
            vPad = concatenated([v, zeros], axis: -1)
        } else {
            targetDim = headDim
        }

        var output = MLXFast.scaledDotProductAttention(
            queries: qPad, keys: kPad, values: vPad,
            scale: 1.0,
            mask: mask.map { .array($0) } ?? .none
        )

        // Unpad si on a padde
        if needsPad {
            output = output[.ellipsis, 0 ..< headDim]
        }

        // [B, H, L, D] → [B, L, H*D]
        let result = output.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        return oProj(result)
    }
}

// MARK: - RoPE 2D multidimensionnel

/// Rotation de moitie: [-x2, x1]
private func rotateHalf(_ x: MLXArray) -> MLXArray {
    let half = x.dim(-1) / 2
    let x1 = x[.ellipsis, 0 ..< half]
    let x2 = x[.ellipsis, half...]
    return concatenated([-x2, x1], axis: -1)
}

/// Applique le RoPE multidimensionnel 2D pour les patches d'image.
/// Split le head_dim en ndim parties et applique rotate_half independamment par dimension spatiale.
func applyMultidimensionalRoPE(_ inputs: MLXArray, positions: MLXArray, baseFrequency: Float) -> MLXArray {
    let headDim = inputs.dim(-1)

    // Si positions est 1D: fallback standard
    if positions.ndim == 2 {
        let half = headDim / 2
        let freqExponents = MLXArray(stride(from: Float(0), to: Float(half), by: 1)) * (2.0 / Float(headDim))
        let timescale = pow(MLXArray(baseFrequency), freqExponents)
        let sinInput = expandedDimensions(positions, axis: -1).asType(.float32) / timescale
        var cosVal = cos(sinInput)
        var sinVal = sin(sinInput)
        cosVal = concatenated([cosVal, cosVal], axis: -1).asType(inputs.dtype)
        sinVal = concatenated([sinVal, sinVal], axis: -1).asType(inputs.dtype)
        cosVal = expandedDimensions(cosVal, axis: 2)
        sinVal = expandedDimensions(sinVal, axis: 2)
        return inputs * cosVal + rotateHalf(inputs) * sinVal
    }

    // 2D: split par dimension spatiale
    let ndim = positions.dim(-1)
    let channelsPerDim = 2 * (headDim / (2 * ndim))
    let halfPerDim = channelsPerDim / 2

    var resultParts: [MLXArray] = []
    for d in 0 ..< ndim {
        let xPart = inputs[.ellipsis, (d * channelsPerDim) ..< ((d + 1) * channelsPerDim)]

        let freqExponents = MLXArray(stride(from: Float(0), to: Float(halfPerDim), by: 1)) * (2.0 / Float(channelsPerDim))
        let timescale = pow(MLXArray(baseFrequency), freqExponents)
        let posD = positions[.ellipsis, d ..< (d + 1)].asType(.float32)
        let sinInput = posD / timescale
        var cosD = cos(sinInput)
        var sinD = sin(sinInput)
        cosD = concatenated([cosD, cosD], axis: -1).asType(inputs.dtype)
        sinD = concatenated([sinD, sinD], axis: -1).asType(inputs.dtype)
        cosD = expandedDimensions(cosD, axis: 2)
        sinD = expandedDimensions(sinD, axis: 2)

        let yPart = xPart * cosD + rotateHalf(xPart) * sinD
        resultParts.append(yPart)
    }

    return concatenated(resultParts, axis: -1)
}
