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

    /// Lot de plus d'une ligne : le lot est replie dans l'axe des tetes avant le RoPE.
    /// `MLXFast.RoPE` (standard comme a frequences explicites) rend des lignes fausses au-dela
    /// de la premiere sur GPU pour une entree contigue `[B > 1, H, 1, D]` (decodage par lot,
    /// K-41 : ecart 5,5-6 sur deux lignes identiques, `RoPEBatchTests`). Le RoPE ne depend
    /// que de la position et toutes les lignes ont le meme `offset` : le repli est exact.
    public func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        guard x.ndim == 4, x.dim(0) > 1 else { return inner(x, offset: offset) }
        let (b, h, l, d) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        return inner(x.reshaped(1, b * h, l, d), offset: offset).reshaped(b, h, l, d)
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
