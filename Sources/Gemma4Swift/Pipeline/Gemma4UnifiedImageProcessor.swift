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
        /// `[config.maxModelPatches, patchDim]`, patchDim = modelPatchSize^2 * 3.
        /// Les lignes au-dela de ``validPatches`` sont du padding a zero.
        public let patches: MLXArray
        /// `[config.maxModelPatches, 2]` = (x_idx, y_idx), -1 sur le padding.
        public let positionIds: MLXArray
        /// Nombre de lignes reellement issues de l'image, `<= maxModelPatches`.
        /// C'est aussi le nombre de soft tokens a reserver dans la sequence texte.
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
        guard image.width > 0, image.height > 0 else {
            throw ImageProcessingError.processingFailed
        }
        let origW = image.width
        let origH = image.height

        let targetPx = Float(maxPatches * patchSize * patchSize)
        let factor = (targetPx / Float(origH * origW)).squareRoot()
        let sideMult = modelPatch  // = pooling_kernel * patch_size

        var bestH = Int(floor(factor * Float(origH) / Float(sideMult))) * sideMult
        var bestW = Int(floor(factor * Float(origW) / Float(sideMult))) * sideMult

        // Fallbacks (image extreme).
        let maxSideLength = config.maxModelPatches * sideMult
        if bestH == 0 && bestW == 0 {
            // Defensif : inatteignable sous l'identite validee au decodage, les
            // deux conditions exigeant simultanement W/H > numSoftTokens et
            // H/W > numSoftTokens. Conserve pour que le calcul reste total.
            bestH = sideMult
            bestW = sideMult
        } else if bestH == 0 {
            bestH = sideMult
            bestW = min(Int(floor(Float(origW) / Float(origH))) * sideMult, maxSideLength)
        } else if bestW == 0 {
            bestW = sideMult
            bestH = min(Int(floor(Float(origH) / Float(origW))) * sideMult, maxSideLength)
        }

        // 2) Redimensionner via CG -> buffer RGB [H, W, 3] UInt8.
        let raw = try Gemma4CGImageLoader.rgbTensor(from: image, width: bestW, height: bestH)

        // 3-4) Tout le pipeline RGB -> normaliser -> patches en operations MLX
        // vectorisees. Pour une image 2K (~4M pixels), on passe de ~16M
        // operations scalaires Swift a quelques kernels MLX (gain ~10-50x).
        let pH = bestH / modelPatch
        let pW = bestW / modelPatch
        let patchDim = config.patchDim
        let numPatches = pH * pW

        // (a) rescale vers [0, 1]
        let rgb = raw.asType(.float32) / MLXArray(Float(255.0))

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

        // 5) Pad jusqu'au nombre de patches MODELE (positions -1 pour les
        // paddings), pas jusqu'a `maxPatches` qui compte des patches fins de
        // 16 px : une ligne de ce tenseur est un patch de 48 px, donc 9 patches
        // fins. Padder a 2520 au lieu de 280 gonflait le tenseur x9 (~70 Mo par
        // image au lieu de ~7,7) et, surtout, faisait tourner
        // [[Gemma4UnifiedVisionEmbedder]] sur 9x trop de lignes : il s'applique
        // au tenseur complet, padding inclus, et seul le resultat est compacte
        // par validPatches en aval.
        let validCount = numPatches
        let padTarget = config.maxModelPatches
        let patchesMLX: MLXArray
        if validCount < padTarget {
            let padding = MLXArray.zeros([padTarget - validCount, patchDim], type: Float.self)
            patchesMLX = concatenated([patchesValid, padding], axis: 0)
        } else {
            patchesMLX = patchesValid
        }

        // Positions (x, y) : petit tableau (<= numSoftTokens entrees), on garde en
        // Swift.
        //
        // `numPatches <= padTarget` decoule de l'identite
        // `modelPatchSize == patchSize * poolingKernelSize`, validee au decodage
        // de [[Gemma4UnifiedVisionConfig]]. La garde ci-dessous n'est donc pas
        // atteignable via le decodeur ; elle est la pour que la surete memoire de
        // cette boucle ne repose pas sur une invariante situee dans un autre
        // fichier — sans elle, une config incoherente deborderait le tableau.
        guard numPatches <= padTarget else {
            throw ImageProcessingError.processingFailed
        }
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

    /// Variante asynchrone de ``processImage(url:config:)`` : deporte le decodage
    /// ImageIO et le resize CoreGraphics sur une tache detachee, pour ne pas
    /// bloquer un appelant `@MainActor`.
    ///
    /// - Parameter priority: priorite de la tache detachee. Sans valeur par defaut,
    ///   pour distinguer cette surcharge de la version synchrone de meme nom et
    ///   forcer un choix explicite de QoS.
    ///
    /// - Important: les tableaux sont evalues avant de franchir la frontiere de
    ///   tache (`MLXArray` n'est pas thread-safe), et `Task.detached` n'herite pas
    ///   des task-locals de MLX. Details sur
    ///   ``Gemma4ImageProcessor/processImage(url:maxSoftTokens:patchSize:poolingKernelSize:priority:)``.
    public static func processImage(
        url: URL,
        config: Gemma4UnifiedVisionConfig,
        priority: TaskPriority
    ) async throws -> ProcessedImage {
        try await Task.detached(priority: priority) {
            let processed = try processImage(url: url, config: config)
            eval(processed.patches, processed.positionIds)
            return processed
        }.value
    }

    /// Variante asynchrone de ``processImage(_:config:)``.
    /// Voir la surcharge `url:` pour le detail de `priority`.
    public static func processImage(
        _ image: CGImage,
        config: Gemma4UnifiedVisionConfig,
        priority: TaskPriority
    ) async throws -> ProcessedImage {
        try await Task.detached(priority: priority) {
            let processed = try processImage(image, config: config)
            eval(processed.patches, processed.positionIds)
            return processed
        }.value
    }
}
