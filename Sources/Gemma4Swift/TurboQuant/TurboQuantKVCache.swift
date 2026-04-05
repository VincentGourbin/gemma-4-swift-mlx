// TurboQuant KV Cache — Cache compresse (experimental, pas encore conforme KVCache)

import MLX

/// KV Cache compresse via TurboQuant MSE Codec.
/// Stocke les K/V sous forme quantifiee (norms + indices) pour reduire la memoire.
///
/// NOTE: Version experimentale. Pour l'utiliser, decompresser les K/V avant l'attention.
/// L'integration complete avec le protocol KVCache de mlx-swift-lm viendra en phase avancee.
public class TurboQuantKVCache {
    let codec: MSECodec
    let config: TurboQuantConfig

    // Stockage quantifie par segments
    private var keyNorms: [MLXArray] = []
    private var keyIndices: [MLXArray] = []
    private var valueNorms: [MLXArray] = []
    private var valueIndices: [MLXArray] = []

    // Cache non-quantifie pour les tokens recents (meilleure qualite)
    private var recentKeys: MLXArray?
    private var recentValues: MLXArray?
    private let recentWindowSize: Int

    private var _offset: Int = 0
    public var offset: Int { _offset }

    public init(config: TurboQuantConfig = .default, recentWindowSize: Int = 64) {
        self.config = config
        self.codec = MSECodec(config: config)
        self.recentWindowSize = recentWindowSize
    }

    /// Ajoute de nouvelles K/V au cache, quantifie si necessaire, retourne tout decompresse
    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        _offset += keys.dim(keys.ndim - 2)

        // Ajouter au cache recent
        if let existingKeys = recentKeys {
            recentKeys = concatenated([existingKeys, keys], axis: -2)
            recentValues = concatenated([recentValues!, values], axis: -2)
        } else {
            recentKeys = keys
            recentValues = values
        }

        // Si le cache recent depasse la fenetre, quantifier les anciens tokens
        if let rk = recentKeys, rk.dim(rk.ndim - 2) > recentWindowSize {
            let overflow = rk.dim(rk.ndim - 2) - recentWindowSize
            let oldKeys = rk[.ellipsis, 0 ..< overflow, 0...]
            let oldValues = recentValues![.ellipsis, 0 ..< overflow, 0...]

            let (kNorms, kIndices) = codec.quantize(oldKeys)
            let (vNorms, vIndices) = codec.quantize(oldValues)

            keyNorms.append(kNorms)
            keyIndices.append(kIndices)
            valueNorms.append(vNorms)
            valueIndices.append(vIndices)

            recentKeys = rk[.ellipsis, overflow..., 0...]
            recentValues = recentValues![.ellipsis, overflow..., 0...]
        }

        return (decompressAll(keyNorms, keyIndices, recentKeys),
                decompressAll(valueNorms, valueIndices, recentValues))
    }

    private func decompressAll(_ norms: [MLXArray], _ indices: [MLXArray], _ recent: MLXArray?) -> MLXArray {
        var parts: [MLXArray] = []
        for (n, idx) in zip(norms, indices) {
            parts.append(codec.dequantize(norms: n, indices: idx))
        }
        if let recent = recent { parts.append(recent) }
        if parts.isEmpty { return MLXArray.zeros([1, 0, 1]) }
        if parts.count == 1 { return parts[0] }
        return concatenated(parts, axis: -2)
    }

    /// Ratio de compression estime
    public var compressionRatio: Float { config.compressionRatio }
}
