// Helper partage entre les processeurs d'image (Gemma4ImageProcessor pour
// E2B/E4B SigLIP et Gemma4UnifiedImageProcessor pour 12B Unified) — evite
// de dupliquer le pattern AppKit/UIKit -> CGImage.

#if canImport(AppKit)
import AppKit
#elseif canImport(UIKit)
import UIKit
#endif
import CoreGraphics
import Foundation

public enum Gemma4CGImageLoader {
    /// Charge une image depuis une URL et renvoie son CGImage en utilisant
    /// la pile graphique native (NSImage sur macOS, UIImage sur iOS).
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
}
