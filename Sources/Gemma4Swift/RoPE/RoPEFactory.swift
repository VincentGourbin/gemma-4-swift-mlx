// Port de rope_utils.py initialize_rope()

import MLX
import MLXNN

/// Type-erased wrapper pour les differentes implementations de RoPE
public protocol RoPELayer {
    func callAsFunction(_ x: MLXArray, offset: Int) -> MLXArray
}

extension RoPE: RoPELayer {}

extension ProportionalRoPE: RoPELayer {}

/// Wrapper leger pour stocker un RoPELayer (PAS un Module pour eviter l'enregistrement de parametres)
public final class RoPEWrapper {
    let inner: any RoPELayer

    public init(_ inner: any RoPELayer) {
        self.inner = inner
    }

    public func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        inner(x, offset: offset)
    }
}

/// Factory pour creer le bon type de RoPE selon la config
public enum RoPEFactory {

    /// Initialise le RoPE adapte au type d'attention
    /// - Parameters:
    ///   - dims: dimension du head
    ///   - base: frequence de base (theta)
    ///   - traditional: mode traditionnel
    ///   - ropeType: "default" ou "proportional"
    ///   - partialRotaryFactor: fraction des dims a rotater (1.0 = tout)
    ///   - factor: facteur de scaling
    public static func create(
        dims: Int,
        base: Float,
        traditional: Bool = false,
        ropeType: String = "default",
        partialRotaryFactor: Float = 1.0,
        factor: Float = 1.0
    ) -> RoPEWrapper {
        if ropeType == "proportional" {
            return RoPEWrapper(ProportionalRoPE(
                dims: dims,
                traditional: traditional,
                base: base,
                factor: factor,
                partialRotaryFactor: partialRotaryFactor
            ))
        }
        // Default: RoPE standard
        return RoPEWrapper(RoPE(dimensions: dims, traditional: traditional, base: base))
    }
}
