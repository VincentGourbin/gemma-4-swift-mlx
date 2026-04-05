// TurboQuant KV Cache — Cache compresse via TurboQuant MSE/Prod Codecs
// TODO: Port complet de turboquant.py TurboQuantKVCache(_BaseCache)
// Pour l'instant: stub avec les types de base

import MLX
import MLXLMCommon

/// Validation du nombre de bits pour TurboQuant
public func turboQuantValidateBits(_ bits: Float) -> Float {
    let valid: Set<Float> = [1, 1.5, 2, 2.5, 3, 3.5, 4]
    precondition(valid.contains(bits), "TurboQuant bits must be one of: \(valid.sorted())")
    return bits
}

/// Verifie si TurboQuant est active
public func turboQuantEnabled(bits: Float?, scheme: String? = nil) -> Bool {
    guard let bits = bits, bits > 0 else { return false }
    if let scheme = scheme {
        return scheme.lowercased() == "turboquant"
    }
    return true
}
