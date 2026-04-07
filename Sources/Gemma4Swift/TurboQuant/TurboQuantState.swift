// TurboQuant State Types — Structures de donnees pour les etats quantifies
// Port de turboquant.py : TurboQuantMSEState, TurboQuantProdState

import MLX

/// Etat MSE quantifie : normes float16 + indices packed uint32
public struct TurboQuantMSEState {
    /// Normes par token: [B, nKVHeads, T] en float16
    public var norms: MLXArray
    /// Indices packed: [B, nKVHeads, T, packedWidth] en uint32
    public var indices: MLXArray

    public init(norms: MLXArray, indices: MLXArray) {
        self.norms = norms
        self.indices = indices
    }

    /// Nombre de tokens stockes
    public var length: Int {
        norms.ndim >= 3 ? norms.shape[2] : 0
    }

    /// Taille memoire approximative en octets
    public var nbytes: Int {
        norms.nbytes + indices.nbytes
    }
}

/// Etat Prod quantifie : MSE + residuel QJL
public struct TurboQuantProdState {
    /// Normes par token: [B, nKVHeads, T] en float16
    public var norms: MLXArray
    /// Indices MSE packed: [B, nKVHeads, T, msePackedWidth] en uint32
    public var mseIndices: MLXArray
    /// Normes residuelles: [B, nKVHeads, T] en float16
    public var residualNorms: MLXArray
    /// Signes QJL packed: [B, nKVHeads, T, signPackedWidth] en uint32
    public var qjlSigns: MLXArray

    public init(norms: MLXArray, mseIndices: MLXArray, residualNorms: MLXArray, qjlSigns: MLXArray) {
        self.norms = norms
        self.mseIndices = mseIndices
        self.residualNorms = residualNorms
        self.qjlSigns = qjlSigns
    }

    public var length: Int {
        norms.ndim >= 3 ? norms.shape[2] : 0
    }

    public var nbytes: Int {
        norms.nbytes + mseIndices.nbytes + residualNorms.nbytes + qjlSigns.nbytes
    }
}
