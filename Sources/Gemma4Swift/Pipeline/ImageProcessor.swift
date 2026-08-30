// Processeur d'image pour Gemma 4 — resize aspect-ratio preserving + normalisation

#if canImport(AppKit)
import AppKit
#elseif canImport(UIKit)
import UIKit
#endif
import CoreGraphics
import Foundation
import MLX

/// Processeur d'image compatible Gemma 4
public enum Gemma4ImageProcessor {

    /// Charge et preprocesse une image depuis un fichier
    /// - Parameters:
    ///   - url: chemin de l'image
    ///   - maxSoftTokens: nombre max de soft tokens (280 par defaut)
    ///   - patchSize: taille du patch (16)
    ///   - poolingKernelSize: taille du kernel de pooling (3)
    /// - Returns: MLXArray [1, C, H, W] channel-first float32 [0, 1]
    public static func processImage(
        url: URL,
        maxSoftTokens: Int = 280,
        patchSize: Int = 16,
        poolingKernelSize: Int = 3
    ) throws -> MLXArray {
        let cgImage = try Gemma4CGImageLoader.load(from: url)
        return try processImage(cgImage, maxSoftTokens: maxSoftTokens, patchSize: patchSize, poolingKernelSize: poolingKernelSize)
    }

    /// Preprocesse un CGImage
    public static func processImage(
        _ image: CGImage,
        maxSoftTokens: Int = 280,
        patchSize: Int = 16,
        poolingKernelSize: Int = 3
    ) throws -> MLXArray {
        // Calculer la taille cible (aspect-ratio preserving, divisible par 48)
        let divisor = patchSize * poolingKernelSize // 48
        let maxPatches = maxSoftTokens * poolingKernelSize * poolingKernelSize // 2520

        let origW = Float(image.width)
        let origH = Float(image.height)
        let aspectRatio = origW / origH

        // Trouver la meilleure taille qui respecte le budget de patches
        var bestW = divisor
        var bestH = divisor
        var bestArea = 0

        for h in stride(from: divisor, through: Int(origH * 2), by: divisor) {
            let w = Int(round(Float(h) * aspectRatio / Float(divisor))) * divisor
            if w < divisor { continue }
            let numPatches = (w / patchSize) * (h / patchSize)
            if numPatches <= maxPatches && w * h > bestArea {
                bestW = w
                bestH = h
                bestArea = w * h
            }
        }

        // Redimensionner
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * bestW
        var pixelData = [UInt8](repeating: 0, count: bestH * bytesPerRow)

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: &pixelData,
            width: bestW, height: bestH,
            bitsPerComponent: 8, bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ) else {
            throw ImageProcessingError.processingFailed
        }

        // Haute qualite
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: bestW, height: bestH))

        // Convertir en [1, 3, H, W] float32 [0, 1] en operations MLX vectorisees
        // (meme approche que [[Gemma4UnifiedImageProcessor]]). Au budget max
        // (2520 patches de 16px = ~645 k pixels) cela remplace ~1,9 M iterations
        // Swift scalaires et 3 allocations [Float] de 645 k elements par quelques
        // kernels MLX.
        let raw = MLXArray(pixelData).reshaped(bestH, bestW, 4)
        let rgb = raw[0..., 0..., 0 ..< 3].asType(.float32) / MLXArray(Float(255.0))

        return rgb.transposed(2, 0, 1).expandedDimensions(axis: 0) // [1, 3, H, W]
    }

    /// Variante asynchrone de ``processImage(url:maxSoftTokens:patchSize:poolingKernelSize:)``
    /// qui deporte tout le travail CPU (decodage ImageIO, resize CoreGraphics,
    /// lecture du buffer) sur une tache detachee.
    ///
    /// A utiliser depuis un contexte `@MainActor` : la version synchrone execute
    /// le decodage et le resize *sur le thread appelant*, donc sur le main thread,
    /// et son attente sur les workers internes de CoreGraphics declenche en prime
    /// le diagnostic runtime « User-initiated thread waiting on a lower QoS thread ».
    ///
    /// - Parameter priority: priorite de la tache detachee. Volontairement **sans
    ///   valeur par defaut** : cela distingue sans ambiguite cette surcharge de la
    ///   version synchrone de meme nom (les appels existants continuent de resoudre
    ///   vers la version synchrone), et force un choix explicite de QoS.
    ///
    /// Le graphe MLX reste paresseux — aucun `eval()` n'est force ici — donc la
    /// semantique est strictement identique a la version synchrone.
    public static func processImage(
        url: URL,
        maxSoftTokens: Int = 280,
        patchSize: Int = 16,
        poolingKernelSize: Int = 3,
        priority: TaskPriority
    ) async throws -> MLXArray {
        let transferred = try await Task.detached(priority: priority) {
            UncheckedTransfer(try processImage(
                url: url,
                maxSoftTokens: maxSoftTokens,
                patchSize: patchSize,
                poolingKernelSize: poolingKernelSize
            ))
        }.value
        return transferred.value
    }

    /// Variante asynchrone de ``processImage(_:maxSoftTokens:patchSize:poolingKernelSize:)``.
    /// Voir la surcharge `url:` pour le detail de `priority`.
    public static func processImage(
        _ image: CGImage,
        maxSoftTokens: Int = 280,
        patchSize: Int = 16,
        poolingKernelSize: Int = 3,
        priority: TaskPriority
    ) async throws -> MLXArray {
        let source = UncheckedTransfer(image)
        let transferred = try await Task.detached(priority: priority) {
            UncheckedTransfer(try processImage(
                source.value,
                maxSoftTokens: maxSoftTokens,
                patchSize: patchSize,
                poolingKernelSize: poolingKernelSize
            ))
        }.value
        return transferred.value
    }
}

public enum ImageProcessingError: LocalizedError {
    case cannotLoadImage(String)
    case processingFailed

    public var errorDescription: String? {
        switch self {
        case .cannotLoadImage(let p): return "Impossible de charger l'image: \(p)"
        case .processingFailed: return "Echec du traitement de l'image"
        }
    }
}
