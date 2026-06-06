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
        #if canImport(AppKit)
        guard let nsImage = NSImage(contentsOf: url),
              let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            throw ImageProcessingError.cannotLoadImage(url.path)
        }
        #elseif canImport(UIKit)
        guard let data = try? Data(contentsOf: url),
              let uiImage = UIImage(data: data),
              let cgImage = uiImage.cgImage else {
            throw ImageProcessingError.cannotLoadImage(url.path)
        }
        #endif

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

        // 3) Normaliser en float32 [-1, 1] (mean=0.5, std=0.5).
        // Layout du buffer : (H, W, 4=BGRA). On veut (3, H, W) channel-first.
        let totalPixels = bestH * bestW
        var rChannel = [Float](repeating: 0, count: totalPixels)
        var gChannel = [Float](repeating: 0, count: totalPixels)
        var bChannel = [Float](repeating: 0, count: totalPixels)
        for i in 0 ..< totalPixels {
            // CGContext avec noneSkipLast est en BGR ordering sur certaines plateformes ;
            // mais avec CGColorSpaceCreateDeviceRGB + alphaInfo = noneSkipLast = RGBX.
            rChannel[i] = (Float(pixelData[i * 4    ]) / 255.0 - imageMean[0]) / imageStd[0]
            gChannel[i] = (Float(pixelData[i * 4 + 1]) / 255.0 - imageMean[1]) / imageStd[1]
            bChannel[i] = (Float(pixelData[i * 4 + 2]) / 255.0 - imageMean[2]) / imageStd[2]
        }

        // 4) Decouper en patches (modelPatch x modelPatch x 3) flattenes.
        let pH = bestH / modelPatch
        let pW = bestW / modelPatch
        let patchDim = config.patchDim
        let numPatches = pH * pW

        var patchesArr = [Float](repeating: 0, count: numPatches * patchDim)
        var positionsArr = [Int32](repeating: 0, count: numPatches * 2)

        for py in 0 ..< pH {
            for px in 0 ..< pW {
                let patchIdx = py * pW + px
                positionsArr[patchIdx * 2    ] = Int32(px) // x
                positionsArr[patchIdx * 2 + 1] = Int32(py) // y

                // Copier le patch (modelPatch lignes x modelPatch cols x 3 canaux).
                // Layout cible : (patchH, patchW, C) ligne-par-ligne.
                let baseY = py * modelPatch
                let baseX = px * modelPatch
                let patchOffset = patchIdx * patchDim
                for ly in 0 ..< modelPatch {
                    let srcY = baseY + ly
                    let rowSrcBase = srcY * bestW
                    let rowDstBase = patchOffset + ly * modelPatch * 3
                    for lx in 0 ..< modelPatch {
                        let srcIdx = rowSrcBase + baseX + lx
                        let dstIdx = rowDstBase + lx * 3
                        patchesArr[dstIdx    ] = rChannel[srcIdx]
                        patchesArr[dstIdx + 1] = gChannel[srcIdx]
                        patchesArr[dstIdx + 2] = bChannel[srcIdx]
                    }
                }
            }
        }

        // 5) Pad jusqu'a maxPatches (positions -1 pour les paddings).
        let validCount = numPatches
        let padTarget = maxPatches
        var finalPatches = patchesArr
        var finalPositions = positionsArr
        if validCount < padTarget {
            finalPatches.append(contentsOf: [Float](repeating: 0, count: (padTarget - validCount) * patchDim))
            for _ in validCount ..< padTarget {
                finalPositions.append(-1)
                finalPositions.append(-1)
            }
        }

        let patchesMLX = MLXArray(finalPatches).reshaped(padTarget, patchDim)
        let positionsMLX = MLXArray(finalPositions).reshaped(padTarget, 2)

        return ProcessedImage(patches: patchesMLX, positionIds: positionsMLX, validPatches: validCount)
    }
}
