// Port simplifie de turboquant.py MSE Codec — Quantification vectorielle apprise

import MLX

/// Codec MSE (Mean Squared Error) pour la compression du KV cache.
/// Utilise un codebook de centroides pour quantifier les vecteurs K/V.
/// Version simplifiee: operations MLX pures (pas de kernels Metal custom).
public struct MSECodec {
    let config: TurboQuantConfig
    /// Codebook: [numCentroids] valeurs normalisees dans [-1, 1]
    let codebook: MLXArray

    public init(config: TurboQuantConfig) {
        self.config = config
        // Generer le codebook via distribution beta-weighted (comme turboquant.py)
        self.codebook = Self.generateCodebook(bits: config.bits)
    }

    /// Genere un codebook de centroides via distribution beta
    static func generateCodebook(bits: Int) -> MLXArray {
        let numCentroids = 1 << bits
        // Centroides uniformement espaces dans [-1, 1]
        // (version simplifiee; la version complete utilise k-means)
        var values: [Float] = []
        for i in 0 ..< numCentroids {
            let t = Float(i) / Float(numCentroids - 1)
            values.append(2.0 * t - 1.0) // [-1, 1]
        }
        return MLXArray(values)
    }

    /// Quantifie un vecteur: normalise → trouve le centroide le plus proche → retourne l'index
    /// - Parameter x: [B, seqLen, headDim] ou [B, numHeads, seqLen, headDim]
    /// - Returns: (norms: [B, ...], indices: [B, ...]) pour la reconstruction
    public func quantize(_ x: MLXArray) -> (norms: MLXArray, indices: MLXArray) {
        // Norme par token
        let norms = sqrt(sum(x * x, axis: -1, keepDims: true))
        let eps = MLXArray(Float(1e-8))
        let normalized = x / maximum(norms, eps)

        // Trouver le centroide le plus proche pour chaque element
        // normalized: [..., D], codebook: [numCentroids]
        // distances: [..., D, numCentroids]
        let expanded = expandedDimensions(normalized, axis: -1) // [..., D, 1]
        let distances = (expanded - codebook) * (expanded - codebook) // [..., D, numCentroids]
        let indices = argMin(distances, axis: -1).asType(.uint8)   // [..., D]

        return (norms: norms.squeezed(axis: -1), indices: indices)
    }

    /// Dequantifie: lookup codebook + rescale par norme
    public func dequantize(norms: MLXArray, indices: MLXArray) -> MLXArray {
        // Lookup: indices [..., D] → values [..., D]
        let values = codebook[indices.asType(.int32)]
        // Rescale
        return values * expandedDimensions(norms, axis: -1)
    }
}
