// Phase 3: Video Processor — Extraction de frames + traitement via VisionEncoder

import AVFoundation
import CoreGraphics
import Foundation
import MLX

/// Processeur video Gemma 4: extrait des frames et les prepare pour le vision encoder.
/// La video est traitee comme une sequence de frames individuelles passees dans le meme
/// vision encoder que les images fixes. Chaque frame produit ~280 soft tokens.
public enum Gemma4VideoProcessor {

    /// Resultat du traitement video
    public struct VideoFrames: @unchecked Sendable {
        /// Pixel values empiles: [numFrames, C, H, W] (channel-first, float32, normalise [0,1])
        public let pixelValues: MLXArray
        /// Nombre de frames extraites
        public let frameCount: Int
        /// Nombre total de tokens video (frameCount * softTokensPerFrame)
        public let totalTokens: Int
    }

    /// Extrait N frames uniformement reparties depuis une video
    /// - Parameters:
    ///   - url: URL du fichier video
    ///   - maxFrames: nombre maximum de frames a extraire (defaut: 8)
    ///   - targetSize: taille cible pour chaque frame (hauteur et largeur divisibles par 48)
    ///   - softTokensPerFrame: nombre de soft tokens par frame (defaut: 280)
    /// - Returns: VideoFrames pret pour le vision encoder
    public static func processVideo(
        url: URL,
        maxFrames: Int = 8,
        targetSize: (width: Int, height: Int) = (336, 336),
        softTokensPerFrame: Int = 280
    ) async throws -> VideoFrames {
        let asset = AVAsset(url: url)
        let duration = try await asset.load(.duration)
        let durationSeconds = CMTimeGetSeconds(duration)

        guard durationSeconds > 0 else {
            throw VideoProcessingError.invalidVideo("Duree video invalide")
        }

        // Calculer les timestamps uniformement repartis
        let frameCount = min(maxFrames, max(1, Int(durationSeconds * 2))) // ~2 fps max
        var times: [CMTime] = []
        for i in 0 ..< frameCount {
            let t = durationSeconds * Double(i) / Double(max(1, frameCount - 1))
            times.append(CMTime(seconds: min(t, durationSeconds - 0.01), preferredTimescale: 600))
        }
        if frameCount == 1 {
            times = [CMTime(seconds: durationSeconds / 2, preferredTimescale: 600)]
        }

        // Extraire les frames via AVAssetImageGenerator
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.maximumSize = CGSize(width: targetSize.width, height: targetSize.height)
        generator.requestedTimeToleranceBefore = CMTime(seconds: 0.5, preferredTimescale: 600)
        generator.requestedTimeToleranceAfter = CMTime(seconds: 0.5, preferredTimescale: 600)

        var frames: [MLXArray] = []
        for time in times {
            let (image, _) = try await generator.image(at: time)
            let frameArray = cgImageToMLXArray(image, width: targetSize.width, height: targetSize.height)
            frames.append(frameArray)
        }

        guard !frames.isEmpty else {
            throw VideoProcessingError.noFramesExtracted
        }

        // Stack: [numFrames, C, H, W]
        let pixelValues = stacked(frames, axis: 0)

        return VideoFrames(
            pixelValues: pixelValues,
            frameCount: frames.count,
            totalTokens: frames.count * softTokensPerFrame
        )
    }

    /// Traite des frames pre-extraites (CGImage)
    public static func processFrames(
        _ images: [CGImage],
        targetSize: (width: Int, height: Int) = (336, 336),
        softTokensPerFrame: Int = 280
    ) -> VideoFrames {
        let frames = images.map { cgImageToMLXArray($0, width: targetSize.width, height: targetSize.height) }
        let pixelValues = stacked(frames, axis: 0)
        return VideoFrames(
            pixelValues: pixelValues,
            frameCount: frames.count,
            totalTokens: frames.count * softTokensPerFrame
        )
    }

    /// Convertit un CGImage en MLXArray [C, H, W] channel-first, float32 [0, 1]
    static func cgImageToMLXArray(_ image: CGImage, width: Int, height: Int) -> MLXArray {
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        var pixelData = [UInt8](repeating: 0, count: height * bytesPerRow)

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: &pixelData,
            width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ) else {
            return MLXArray.zeros([3, height, width])
        }

        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        // [H, W, 4] RGBX → [3, H, W] float32 normalise [0, 1]
        var rChannel = [Float](repeating: 0, count: height * width)
        var gChannel = [Float](repeating: 0, count: height * width)
        var bChannel = [Float](repeating: 0, count: height * width)

        for i in 0 ..< height * width {
            rChannel[i] = Float(pixelData[i * 4]) / 255.0
            gChannel[i] = Float(pixelData[i * 4 + 1]) / 255.0
            bChannel[i] = Float(pixelData[i * 4 + 2]) / 255.0
        }

        let r = MLXArray(rChannel).reshaped(1, height, width)
        let g = MLXArray(gChannel).reshaped(1, height, width)
        let b = MLXArray(bChannel).reshaped(1, height, width)

        return concatenated([r, g, b], axis: 0) // [3, H, W]
    }
}

public enum VideoProcessingError: LocalizedError {
    case invalidVideo(String)
    case noFramesExtracted

    public var errorDescription: String? {
        switch self {
        case .invalidVideo(let msg): return "Video invalide: \(msg)"
        case .noFramesExtracted: return "Aucune frame extraite de la video"
        }
    }
}
