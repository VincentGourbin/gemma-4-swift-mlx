// Image preprocessor pour gemma4_unified (12B) — patches 48x48 sans encoder.
//
// Difference avec [[Gemma4ImageProcessor]] (E2B/E4B SigLIP) :
//   - Sortie : (pixelValues [N, patchDim], positionIds [N, 2])
//   - patchDim = modelPatchSize * modelPatchSize * 3 (par defaut 48*48*3 = 6912)
//   - Pas de [1, C, H, W] : les patches sont directement flattenes.
//   - Position IDs (x, y) pour chaque patch, -1 si padding.

#if canImport(AppKit)
import AppKit
#elseif canImport(UIKit)
import UIKit
#endif
import CoreGraphics
import Foundation
import MLX

public enum Gemma4UnifiedImageProcessor {

    /// Resultat du preprocessing d'une seule image.
    public struct ProcessedImage: @unchecked Sendable {
        /// [numPatches, patchDim] avec patchDim = modelPatchSize^2 * 3
        public let patches: MLXArray
        /// [numPatches, 2] = (x_idx, y_idx). -1 = padding.
        public let positionIds: MLXArray
        /// Nombre de patches valides (avant padding).
        public let validPatches: Int
    }

    /// Pas de normalisation pour gemma4_unified (verifie processor_config.json :
    /// do_normalize=false, image_mean=[0,0,0], image_std=[1,1,1]).
    /// Seul le rescale 1/255 vers [0,1] est applique.
    public static let imageMean: [Float] = [0.0, 0.0, 0.0]
    public static let imageStd: [Float] = [1.0, 1.0, 1.0]

    /// Pipeline complet : URL -> patches + positions ready for VisionEmbedder.
    public static func processImage(
        url: URL,
        config: Gemma4UnifiedVisionConfig
    ) throws -> ProcessedImage {
        let cgImage = try Gemma4CGImageLoader.load(from: url)
        return try processImage(cgImage, config: config)
    }

    /// Pipeline complet a partir d'un CGImage.
    public static func processImage(
        _ image: CGImage,
        config: Gemma4UnifiedVisionConfig
    ) throws -> ProcessedImage {
        let modelPatch = config.modelPatchSize       // 48 = pooling_kernel * patch_size
        let patchSize = config.patchSize             // 16
        let maxPatches = config.maxPatches            // 280*9 = 2520 (en patches de patch_size)

        // 1) aspect_ratio_preserving_resize (port verbatim du Python).
        //    target_px (a patch_size=16) bornne ; on aligne ensuite sur modelPatch.
        let origW = image.width
        let origH = image.height

        let targetPx = Float(maxPatches * patchSize * patchSize)
        let factor = (targetPx / Float(origH * origW)).squareRoot()
        let sideMult = modelPatch  // = pooling_kernel * patch_size

        var bestH = Int(floor(factor * Float(origH) / Float(sideMult))) * sideMult
        var bestW = Int(floor(factor * Float(origW) / Float(sideMult))) * sideMult

        // Fallbacks (image extreme).
        let maxSideLength = (maxPatches / (config.poolingKernelSize * config.poolingKernelSize)) * sideMult
        if bestH == 0 && bestW == 0 {
            // Image trop petite : on prend une seule cellule.
            bestH = sideMult
            bestW = sideMult
        } else if bestH == 0 {
            bestH = sideMult
            bestW = min(Int(floor(Float(origW) / Float(origH))) * sideMult, maxSideLength)
        } else if bestW == 0 {
            bestW = sideMult
            bestH = min(Int(floor(Float(origH) / Float(origW))) * sideMult, maxSideLength)
        }

        // 2) Redimensionner via CG (BGRA -> RGB float).
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
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: bestW, height: bestH))

        // 3-4) Tout le pipeline RGBA -> normaliser -> patches en operations MLX
        // vectorisees. Pour une image 2K (~4M pixels), on passe de ~16M
        // operations scalaires Swift a quelques kernels MLX (gain ~10-50x).
        let pH = bestH / modelPatch
        let pW = bestW / modelPatch
        let patchDim = config.patchDim
        let numPatches = pH * pW

        // (a) buffer brut UInt8 [H*W*4] -> [H, W, 4] -> [H, W, 3] (drop alpha)
        let raw = MLXArray(pixelData).reshaped(bestH, bestW, 4)
        let rgb = raw[0..., 0..., 0..<3].asType(.float32) / MLXArray(Float(255.0))

        // (b) normalisation par canal : (x - mean) / std (broadcast sur [3])
        let mean = MLXArray(Self.imageMean)
        let std = MLXArray(Self.imageStd)
        let normalized = (rgb - mean) / std

        // (c) decoupage en patches : [pH, mp, pW, mp, 3] -> [pH, pW, mp, mp, 3]
        //     -> [numPatches, patchDim]
        let patchesValid = normalized
            .reshaped(pH, modelPatch, pW, modelPatch, 3)
            .transposed(0, 2, 1, 3, 4)
            .reshaped(numPatches, patchDim)

        // 5) Pad jusqu'a maxPatches (positions -1 pour les paddings).
        let validCount = numPatches
        let padTarget = maxPatches
        let patchesMLX: MLXArray
        if validCount < padTarget {
            let padding = MLXArray.zeros([padTarget - validCount, patchDim], type: Float.self)
            patchesMLX = concatenated([patchesValid, padding], axis: 0)
        } else {
            patchesMLX = patchesValid
        }

        // Positions (x, y) : petit tableau, on garde en Swift (numPatches <= 2520).
        var positionsArr = [Int32](repeating: -1, count: padTarget * 2)
        for py in 0 ..< pH {
            for px in 0 ..< pW {
                let patchIdx = py * pW + px
                positionsArr[patchIdx * 2    ] = Int32(px)
                positionsArr[patchIdx * 2 + 1] = Int32(py)
            }
        }
        let positionsMLX = MLXArray(positionsArr).reshaped(padTarget, 2)

        return ProcessedImage(patches: patchesMLX, positionIds: positionsMLX, validPatches: validCount)
    }
}
