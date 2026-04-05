// TurboQuant Configuration — Compression KV cache

import Foundation

/// Configuration de la compression TurboQuant du KV cache
public struct TurboQuantConfig: Sendable {
    /// Nombre de bits pour la quantification MSE (2-4)
    public var bits: Int
    /// Taille du groupe pour la quantification
    public var groupSize: Int
    /// Activer le codec residuel QJL (plus precis, plus lent)
    public var useResidual: Bool

    public static let `default` = TurboQuantConfig(bits: 4, groupSize: 32, useResidual: false)
    public static let highQuality = TurboQuantConfig(bits: 4, groupSize: 32, useResidual: true)
    public static let compact = TurboQuantConfig(bits: 2, groupSize: 64, useResidual: false)

    public init(bits: Int = 4, groupSize: Int = 32, useResidual: Bool = false) {
        self.bits = bits
        self.groupSize = groupSize
        self.useResidual = useResidual
    }

    /// Nombre de centroides dans le codebook
    public var numCentroids: Int { 1 << bits }

    /// Facteur de compression approximatif
    public var compressionRatio: Float {
        Float(16) / Float(bits) // par rapport a float16
    }
}
