// Helper partage entre les processeurs d'image (Gemma4ImageProcessor pour
// E2B/E4B SigLIP et Gemma4UnifiedImageProcessor pour 12B Unified) — evite
// de dupliquer le pattern AppKit/UIKit -> CGImage, puis CGImage -> buffer RGB.

#if canImport(AppKit)
import AppKit
#elseif canImport(UIKit)
import UIKit
#endif
import CoreGraphics
import Foundation
import MLX

public enum Gemma4CGImageLoader {
    /// Charge une image depuis une URL et renvoie son CGImage en utilisant
    /// la pile graphique native (NSImage sur macOS, UIImage sur iOS).
    ///
    /// Note : `NSImage` differe le decode reel jusqu'au premier rendu, donc
    /// l'essentiel du cout se paie dans ``rgbTensor(from:width:height:)`` et non
    /// ici. `NSImage` est utilisable hors du main thread (« thread-unsafe but
    /// usable from one thread at a time » dans le Thread Safety Summary d'Apple),
    /// ce dont dependent les surcharges `async` des processeurs.
    public static func load(from url: URL) throws -> CGImage {
        #if canImport(AppKit)
        guard let nsImage = NSImage(contentsOf: url),
              let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            throw ImageProcessingError.cannotLoadImage(url.path)
        }
        return cgImage
        #elseif canImport(UIKit)
        guard let data = try? Data(contentsOf: url),
              let uiImage = UIImage(data: data),
              let cgImage = uiImage.cgImage else {
            throw ImageProcessingError.cannotLoadImage(url.path)
        }
        return cgImage
        #endif
    }

    /// Rend `image` dans un bitmap `width` x `height` (interpolation haute
    /// qualite) et renvoie les canaux RGB en `MLXArray` **[H, W, 3] UInt8**,
    /// alpha ecarte.
    ///
    /// Le dtype reste `UInt8` a dessein : chaque processeur applique ensuite sa
    /// propre mise a l'echelle et son propre reordonnancement d'axes, et faire
    /// l'arithmetique *apres* la permutation d'axes materialise un buffer
    /// contigu la ou une transposition posterieure ne rendrait qu'une vue stridee.
    ///
    /// Le bitmap est alloue et cede a CoreGraphics dans une portee
    /// `withUnsafeMutableBytes` : passer `&tableau` a `CGContext(data:)` ferait
    /// echapper un pointeur temporaire hors de l'appel qui l'a cree, ce que la
    /// doc Swift (`temporary-pointers`) classe en comportement indefini — le
    /// contexte conserve le pointeur et ecrit dedans au `draw` suivant.
    public static func rgbTensor(from image: CGImage, width: Int, height: Int) throws -> MLXArray {
        let bytesPerRow = 4 * width
        var pixelData = [UInt8](repeating: 0, count: height * bytesPerRow)

        let rgba: MLXArray = try pixelData.withUnsafeMutableBytes { raw in
            guard let base = raw.baseAddress,
                  let context = CGContext(
                    data: base,
                    width: width, height: height,
                    bitsPerComponent: 8, bytesPerRow: bytesPerRow,
                    space: CGColorSpaceCreateDeviceRGB(),
                    bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
                  )
            else {
                throw ImageProcessingError.processingFailed
            }

            context.interpolationQuality = .high
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

            // mlx_array_new_data copie immediatement : rien ne reference `raw`
            // une fois la portee refermee.
            return MLXArray(UnsafeRawBufferPointer(raw), [height, width, 4], type: UInt8.self)
        }

        return rgba[0..., 0..., 0 ..< 3]
    }
}
