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

    /// Budget de soft tokens par image (reference mlx-vlm / HF).
    public static let defaultMaxSoftTokens = 280
    /// Taille de patch du tour vision SigLIP.
    public static let defaultPatchSize = 16
    /// Taille du kernel de pooling : 3x3 patches par soft token.
    public static let defaultPoolingKernelSize = 3

    /// Charge et preprocesse une image depuis un fichier
    /// - Parameters:
    ///   - url: chemin de l'image
    ///   - maxSoftTokens: nombre max de soft tokens (280 par defaut)
    ///   - patchSize: taille du patch (16)
    ///   - poolingKernelSize: taille du kernel de pooling (3)
    /// - Returns: MLXArray [1, C, H, W] channel-first float32 [0, 1]
    ///
    /// - Note: synchrone — le decodage et le resize s'executent **sur le thread
    ///   appelant**. Depuis un contexte `@MainActor`, preferer
    ///   ``processImage(url:maxSoftTokens:patchSize:poolingKernelSize:priority:)``.
    public static func processImage(
        url: URL,
        maxSoftTokens: Int = defaultMaxSoftTokens,
        patchSize: Int = defaultPatchSize,
        poolingKernelSize: Int = defaultPoolingKernelSize
    ) throws -> MLXArray {
        let cgImage = try Gemma4CGImageLoader.load(from: url)
        return try processImage(cgImage, maxSoftTokens: maxSoftTokens, patchSize: patchSize, poolingKernelSize: poolingKernelSize)
    }

    /// Preprocesse un CGImage
    ///
    /// - Note: synchrone, cf. la remarque sur ``processImage(url:maxSoftTokens:patchSize:poolingKernelSize:)``.
    public static func processImage(
        _ image: CGImage,
        maxSoftTokens: Int = defaultMaxSoftTokens,
        patchSize: Int = defaultPatchSize,
        poolingKernelSize: Int = defaultPoolingKernelSize
    ) throws -> MLXArray {
        let (bestW, bestH) = try targetSize(
            for: image,
            maxSoftTokens: maxSoftTokens,
            patchSize: patchSize,
            poolingKernelSize: poolingKernelSize
        )

        // Buffer RGB [H, W, 3] UInt8 (decode + resize haute qualite).
        let rgb = try Gemma4CGImageLoader.rgbTensor(from: image, width: bestW, height: bestH)

        // Permuter les axes AVANT l'arithmetique : l'operation elementwise
        // materialise alors un buffer contigu. Transposer apres coup ne
        // renverrait qu'une vue stridee (Slice et Transpose sont zero-copy dans
        // MLX), ce qui ferait payer une copie General a tout consommateur ayant
        // besoin de contiguite — l'ancien `concatenated([r, g, b], axis: 1)`
        // materialisait, donc c'est le contrat que l'API doit conserver.
        let chw = rgb.transposed(2, 0, 1).asType(.float32) / MLXArray(Float(255.0))

        return chw.expandedDimensions(axis: 0) // [1, 3, H, W]
    }

    /// Taille cible aspect-ratio preserving, alignee sur `patchSize * poolingKernelSize`
    /// et tenant dans le budget de patches.
    private static func targetSize(
        for image: CGImage,
        maxSoftTokens: Int,
        patchSize: Int,
        poolingKernelSize: Int
    ) throws -> (width: Int, height: Int) {
        let divisor = patchSize * poolingKernelSize // 48
        let maxPatches = maxSoftTokens * poolingKernelSize * poolingKernelSize // 2520

        guard image.width > 0, image.height > 0 else {
            throw ImageProcessingError.processingFailed
        }

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

        if bestArea == 0 {
            // Aucun candidat. Deux causes : image plus courte que `divisor / 2`
            // (la boucle ne tourne pas du tout), ou ratio si extreme que la
            // largeur alignee tombe sous un patch / explose le budget. Sans
            // rattrapage on renvoyait 48x48, ce qui ecrase silencieusement le
            // ratio : une bande OCR 1000x20 devenait un carre illisible.
            //
            // On fixe le cote court a une cellule et on etire le cote long
            // jusqu'a la limite du budget, comme le font les fallbacks de
            // [[Gemma4UnifiedImageProcessor]].
            let maxSide = (maxPatches / (poolingKernelSize * poolingKernelSize)) * divisor
            let longRatio = aspectRatio >= 1 ? aspectRatio : 1 / aspectRatio
            let cells = min(max(longRatio.rounded(), 1), Float(maxSide / divisor))
            let longSide = Int(cells) * divisor

            if aspectRatio >= 1 {
                bestH = divisor
                bestW = longSide
            } else {
                bestW = divisor
                bestH = longSide
            }
        }

        return (bestW, bestH)
    }

    /// Variante asynchrone de ``processImage(url:maxSoftTokens:patchSize:poolingKernelSize:)``
    /// qui deporte le preprocessing sur une tache detachee.
    ///
    /// A utiliser depuis un contexte `@MainActor` : la version synchrone execute
    /// le decodage (AppKit/UIKit) et le resize CoreGraphics *sur le thread
    /// appelant*, donc sur le main thread.
    ///
    /// - Parameter priority: priorite de la tache detachee. Volontairement **sans
    ///   valeur par defaut** : cela distingue sans ambiguite cette surcharge de la
    ///   version synchrone de meme nom (les appels existants continuent de resoudre
    ///   vers la version synchrone), et force un choix explicite de QoS.
    ///
    /// - Important: le tableau est **evalue avant de franchir la frontiere de
    ///   tache**. `MLXArray` n'est pas thread-safe et la doc de mlx-swift est
    ///   explicite : « It is not safe to create `c` in one thread and
    ///   consume/evaluate it in another. » Renvoyer un graphe paresseux construit
    ///   ici pour le laisser evaluer par l'appelant corromprait l'etat global de
    ///   MLX. C'est donc un tableau materialise qui traverse, et l'appelant paie
    ///   zero calcul — mais le cout GPU est paye ici, pas plus tard.
    ///
    /// - Important: `Task.detached` **n'herite pas des task-locals**. Les portees
    ///   `withError`, `Device.withDefaultDevice`, `Stream.withNewDefaultStream` et
    ///   l'etat `MLXRandom` installes par l'appelant ne traversent pas : le
    ///   preprocessing s'execute sur le device global par defaut, et une erreur
    ///   MLX y declenche le handler global (a defaut, `fatalError`) au lieu du
    ///   handler local. Si vous en dependez, appelez la version synchrone dans
    ///   votre propre tache.
    public static func processImage(
        url: URL,
        maxSoftTokens: Int = defaultMaxSoftTokens,
        patchSize: Int = defaultPatchSize,
        poolingKernelSize: Int = defaultPoolingKernelSize,
        priority: TaskPriority
    ) async throws -> MLXArray {
        let transferred = try await Task.detached(priority: priority) {
            let pixels = try processImage(
                url: url,
                maxSoftTokens: maxSoftTokens,
                patchSize: patchSize,
                poolingKernelSize: poolingKernelSize
            )
            eval(pixels)
            return UncheckedTransfer(pixels)
        }.value
        return transferred.value
    }

    /// Variante asynchrone de ``processImage(_:maxSoftTokens:patchSize:poolingKernelSize:)``.
    /// Voir la surcharge `url:` pour `priority`, l'evaluation avant transfert et
    /// la perte des task-locals.
    public static func processImage(
        _ image: CGImage,
        maxSoftTokens: Int = defaultMaxSoftTokens,
        patchSize: Int = defaultPatchSize,
        poolingKernelSize: Int = defaultPoolingKernelSize,
        priority: TaskPriority
    ) async throws -> MLXArray {
        let transferred = try await Task.detached(priority: priority) {
            let pixels = try processImage(
                image,
                maxSoftTokens: maxSoftTokens,
                patchSize: patchSize,
                poolingKernelSize: poolingKernelSize
            )
            eval(pixels)
            return UncheckedTransfer(pixels)
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
