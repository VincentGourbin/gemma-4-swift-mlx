// Processeur d'image pour Gemma 4 — resize aspect-ratio preserving + normalisation

import AppKit
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
        guard let nsImage = NSImage(contentsOf: url),
              let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            throw ImageProcessingError.cannotLoadImage(url.path)
        }

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
        let maxPixels = maxPatches * patchSize * patchSize // ~645K

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

        // Convertir en [1, 3, H, W] float32 [0, 1]
        var rChannel = [Float](repeating: 0, count: bestH * bestW)
        var gChannel = [Float](repeating: 0, count: bestH * bestW)
        var bChannel = [Float](repeating: 0, count: bestH * bestW)

        for i in 0 ..< bestH * bestW {
            rChannel[i] = Float(pixelData[i * 4]) / 255.0
            gChannel[i] = Float(pixelData[i * 4 + 1]) / 255.0
            bChannel[i] = Float(pixelData[i * 4 + 2]) / 255.0
        }

        let r = MLXArray(rChannel).reshaped(1, 1, bestH, bestW)
        let g = MLXArray(gChannel).reshaped(1, 1, bestH, bestW)
        let b = MLXArray(bChannel).reshaped(1, 1, bestH, bestW)

        return concatenated([r, g, b], axis: 1) // [1, 3, H, W]
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
