// TurboQuant Utilitaires — Rotation, codebook Beta-optimise, Hadamard transform
// Port de turboquant.py : _rotation_matrix, _rht_*, _codebook, _beta_pdf

import Foundation
import MLX
import MLXFast
import MLXRandom

let TURBOQUANT_DEFAULT_SEED: UInt64 = 0
let TURBOQUANT_EPS: Float = 1e-6

// MARK: - Caches (equivalent de @lru_cache Python)

nonisolated(unsafe) private var _rotationMatrixCache: [String: MLXArray] = [:]
nonisolated(unsafe) private var _codebookCache: [String: MLXArray] = [:]
nonisolated(unsafe) private var _rhtSignCache: [String: MLXArray] = [:]
nonisolated(unsafe) private var _projectionMatrixCache: [String: MLXArray] = [:]
private let _cacheLock = NSLock()

// MARK: - Rotation Matrix

/// Genere une matrice de rotation orthogonale aleatoire via QR decomposition
/// Deterministe pour un (dim, seed) donne
func turboQuantRotationMatrix(dim: Int, seed: UInt64) -> MLXArray {
    let key = "\(dim)_\(seed)"
    _cacheLock.lock()
    if let cached = _rotationMatrixCache[key] {
        _cacheLock.unlock()
        return cached
    }
    _cacheLock.unlock()

    guard dim > 0 else { return MLXArray.zeros([0, 0]) }
    guard dim > 1 else { return MLXArray.ones([1, 1]) }

    // Generer une matrice gaussienne deterministe
    let rngKey = MLXRandom.key(seed &+ UInt64(dim) &* 7919)
    let matrix = MLXRandom.normal([dim, dim], key: rngKey)

    // QR decomposition → Q est orthogonale
    let (q, r) = MLXLinalg.qr(matrix, stream: .cpu)

    // Corriger le signe: Q *= sign(diag(R))
    let diagR = MLX.diag(r)
    let signs = MLX.sign(diagR)
    let result = q * signs

    eval(result)

    _cacheLock.lock()
    _rotationMatrixCache[key] = result
    _cacheLock.unlock()

    return result
}

// MARK: - Randomized Hadamard Transform (RHT)

/// Vecteur de signes deterministe pour le RHT
func turboQuantRHTSignVector(dim: Int, seed: UInt64) -> MLXArray {
    let key = "\(dim)_\(seed)"
    _cacheLock.lock()
    if let cached = _rhtSignCache[key] {
        _cacheLock.unlock()
        return cached
    }
    _cacheLock.unlock()

    guard dim > 0 else { return MLXArray.zeros([0]) }

    let rngKey = MLXRandom.key(seed &+ UInt64(dim) &* 7919)
    // Generer des 0/1 puis convertir en -1/+1
    let bits = MLXRandom.bernoulli(0.5, [dim], key: rngKey)
    let result = (bits.asType(.float32) * 2 - 1)
    eval(result)

    _cacheLock.lock()
    _rhtSignCache[key] = result
    _cacheLock.unlock()

    return result
}

func nextPowerOfTwo(_ n: Int) -> Int {
    guard n > 1 else { return 1 }
    var v = n - 1
    v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16
    return v + 1
}

func rhtPaddedDim(_ dim: Int) -> Int {
    nextPowerOfTwo(dim)
}

/// Walsh-Hadamard Transform forward via matmul (fallback sans mx.hadamard_transform)
/// O(D^2) au lieu de O(D log D), mais fonctionne pour toutes les dimensions
func rhtForward(_ x: MLXArray, signs: MLXArray) -> MLXArray {
    let dim = signs.shape[0]
    let padded = rhtPaddedDim(dim)
    var y = x * signs

    // Padding si necessaire
    if padded > dim {
        let padShape = Array(y.shape.dropLast()) + [padded - dim]
        let padding = MLXArray.zeros(padShape, dtype: y.dtype)
        y = concatenated([y, padding], axis: -1)
    }

    // Walsh-Hadamard via operations butterfly iteratives
    var stride = 1
    let n = padded
    while stride < n {
        let half = stride
        stride *= 2
        // Pour chaque paire de blocs, appliquer le butterfly
        let reshaped = y.reshaped(Array(y.shape.dropLast()) + [n / stride, stride])
        let left = reshaped[.ellipsis, 0 ..< half]
        let right = reshaped[.ellipsis, half...]
        let newLeft = left + right
        let newRight = left - right
        y = concatenated([newLeft, newRight], axis: -1).reshaped(Array(y.shape.dropLast()) + [n])
    }

    let scale = MLXArray(1.0 / sqrt(Float(padded)), dtype: y.dtype)
    y = y * scale

    if padded > dim {
        y = y[.ellipsis, 0 ..< dim]
    }
    return y
}

/// Walsh-Hadamard Transform inverse
func rhtInverse(_ x: MLXArray, signs: MLXArray) -> MLXArray {
    let dim = signs.shape[0]
    let padded = rhtPaddedDim(dim)
    var y = x

    if padded > dim {
        let padShape = Array(y.shape.dropLast()) + [padded - dim]
        let padding = MLXArray.zeros(padShape, dtype: y.dtype)
        y = concatenated([y, padding], axis: -1)
    }

    // Hadamard est sa propre inverse (a un facteur pres)
    var stride = 1
    let n = padded
    while stride < n {
        let half = stride
        stride *= 2
        let reshaped = y.reshaped(Array(y.shape.dropLast()) + [n / stride, stride])
        let left = reshaped[.ellipsis, 0 ..< half]
        let right = reshaped[.ellipsis, half...]
        let newLeft = left + right
        let newRight = left - right
        y = concatenated([newLeft, newRight], axis: -1).reshaped(Array(y.shape.dropLast()) + [n])
    }

    let scale = MLXArray(1.0 / sqrt(Float(padded)), dtype: y.dtype)
    y = y * scale

    if padded > dim {
        y = y[.ellipsis, 0 ..< dim]
    }
    return y * signs
}

// MARK: - Projection Matrix (pour QJL)

/// Matrice de projection aleatoire pour le codec residuel QJL
func turboQuantProjectionMatrix(dim: Int, seed: UInt64) -> MLXArray {
    let key = "\(dim)_\(seed)"
    _cacheLock.lock()
    if let cached = _projectionMatrixCache[key] {
        _cacheLock.unlock()
        return cached
    }
    _cacheLock.unlock()

    guard dim > 0 else { return MLXArray.zeros([0, 0]) }

    let rngKey = MLXRandom.key(seed &+ UInt64(dim) &* 2971 &+ 17)
    let result = MLXRandom.normal([dim, dim], key: rngKey)
    eval(result)

    _cacheLock.lock()
    _projectionMatrixCache[key] = result
    _cacheLock.unlock()

    return result
}

// MARK: - Codebook Generation (Beta-distribution + k-means)

/// PDF Beta pour les composantes d'un vecteur unitaire en dimension D
private func betaPdf(grid: [Float], dim: Int) -> [Float] {
    guard dim > 1 else {
        return [Float](repeating: 1.0 / Float(grid.count), count: grid.count)
    }

    let logCoeff = lgammaf(Float(dim) / 2.0)
        - 0.5 * logf(Float.pi)
        - lgammaf(Float(dim - 1) / 2.0)

    var logPdf = [Float](repeating: 0, count: grid.count)
    var maxLogPdf: Float = -.infinity

    for (i, g) in grid.enumerated() {
        let val = max(1.0 - g * g, 1e-30)
        logPdf[i] = logCoeff + (Float(dim - 3) / 2.0) * logf(val)
        if logPdf[i] > maxLogPdf { maxLogPdf = logPdf[i] }
    }

    // Normaliser pour eviter les overflows
    var pdf = [Float](repeating: 0, count: grid.count)
    var sum: Float = 0
    for i in 0 ..< grid.count {
        pdf[i] = expf(logPdf[i] - maxLogPdf)
        sum += pdf[i]
    }

    guard sum > 0 else {
        return [Float](repeating: 1.0 / Float(grid.count), count: grid.count)
    }

    for i in 0 ..< grid.count {
        pdf[i] /= sum
    }
    return pdf
}

/// Genere un codebook optimal pour la quantisation MSE
/// Utilise la distribution Beta pour initialiser puis 100 iterations de k-means
func turboQuantCodebook(dim: Int, bits: Int) -> MLXArray {
    let key = "\(dim)_\(bits)"
    _cacheLock.lock()
    if let cached = _codebookCache[key] {
        _cacheLock.unlock()
        return cached
    }
    _cacheLock.unlock()

    guard bits > 0 else { return MLXArray.zeros([0]) }

    let levels = 1 << bits

    guard dim > 1 else {
        var values = [Float](repeating: 0, count: levels)
        for i in 0 ..< levels {
            values[i] = -1.0 + 2.0 * Float(i) / Float(levels - 1)
        }
        let result = MLXArray(values)
        _cacheLock.lock()
        _codebookCache[key] = result
        _cacheLock.unlock()
        return result
    }

    // Grid fine pour la PDF
    let gridCount = 32768
    var grid = [Float](repeating: 0, count: gridCount)
    for i in 0 ..< gridCount {
        grid[i] = -1.0 + 1e-6 + (2.0 - 2e-6) * Float(i) / Float(gridCount - 1)
    }

    let weights = betaPdf(grid: grid, dim: dim)

    // CDF
    var cdf = [Float](repeating: 0, count: gridCount)
    cdf[0] = weights[0]
    for i in 1 ..< gridCount {
        cdf[i] = cdf[i - 1] + weights[i]
    }

    // Initialisation via quantiles de la CDF
    var centroids = [Float](repeating: 0, count: levels)
    for i in 0 ..< levels {
        let target = (Float(i) + 0.5) / Float(levels)
        // Interpolation lineaire dans la CDF
        var idx = 0
        for j in 0 ..< gridCount - 1 {
            if cdf[j + 1] >= target { idx = j; break }
            if j == gridCount - 2 { idx = j }
        }
        let frac = cdf[idx + 1] > cdf[idx]
            ? (target - cdf[idx]) / (cdf[idx + 1] - cdf[idx])
            : 0.0
        centroids[i] = grid[idx] + frac * (grid[idx + 1] - grid[idx])
    }

    // K-means refinement (100 iterations)
    for _ in 0 ..< 100 {
        var boundaries = [Float](repeating: 0, count: levels + 1)
        boundaries[0] = -1.0
        boundaries[levels] = 1.0
        for i in 0 ..< levels - 1 {
            boundaries[i + 1] = 0.5 * (centroids[i] + centroids[i + 1])
        }

        var newCentroids = centroids
        for i in 0 ..< levels {
            var weightedSum: Float = 0
            var totalWeight: Float = 0
            for j in 0 ..< gridCount {
                let inBucket: Bool
                if i == levels - 1 {
                    inBucket = grid[j] >= boundaries[i] && grid[j] <= boundaries[i + 1]
                } else {
                    inBucket = grid[j] >= boundaries[i] && grid[j] < boundaries[i + 1]
                }
                if inBucket {
                    weightedSum += weights[j] * grid[j]
                    totalWeight += weights[j]
                }
            }
            if totalWeight > 0 {
                newCentroids[i] = weightedSum / totalWeight
            }
        }

        // Check convergence
        var maxDiff: Float = 0
        for i in 0 ..< levels {
            maxDiff = max(maxDiff, abs(newCentroids[i] - centroids[i]))
        }
        centroids = newCentroids
        if maxDiff < 1e-6 { break }
    }

    let result = MLXArray(centroids)

    _cacheLock.lock()
    _codebookCache[key] = result
    _cacheLock.unlock()

    return result
}

// MARK: - Helpers

func isPowerOfTwo(_ value: Int) -> Bool {
    value > 0 && (value & (value - 1)) == 0
}
