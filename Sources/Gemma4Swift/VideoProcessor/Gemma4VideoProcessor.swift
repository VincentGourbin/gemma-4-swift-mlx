// Phase 3: Video Processor — Extraction de frames + traitement via VisionEncoder
// Aligné sur la référence Python (transformers): 1fps, 32 frames max, 70 soft tokens/frame, timestamps

import AVFoundation
import CoreGraphics
import Foundation
import MLX

/// Processeur video Gemma 4: extrait des frames et les prepare pour le vision encoder.
/// La video est traitee comme une sequence de frames individuelles passees dans le meme
/// vision encoder que les images fixes. Chaque frame produit 70 soft tokens (vs 280 pour les images).
public enum Gemma4VideoProcessor {

    /// Nombre de soft tokens par frame video (reference: 70, vs 280 pour les images)
    public static let defaultSoftTokensPerFrame = 70

    /// Nombre maximum de frames par defaut (reference: 32)
    public static let defaultMaxFrames = 32

    /// Duree maximale de video supportee (secondes)
    public static let maxVideoDurationSeconds: Double = 60.0

    /// Resultat du traitement video
    public struct VideoFrames: @unchecked Sendable {
        /// Pixel values empiles: [numFrames, C, H, W] (channel-first, float32, normalise [0,1])
        /// Chaque frame est redimensionnee pour tenir dans le budget de 70 soft tokens
        public let pixelValues: MLXArray
        /// Nombre de frames extraites
        public let frameCount: Int
        /// Nombre de soft tokens par frame (70 par defaut)
        public let softTokensPerFrame: Int
        /// Nombre total de tokens video (frameCount * softTokensPerFrame)
        public let totalTokens: Int
        /// Timestamps de chaque frame en secondes
        public let timestamps: [Double]
        /// FPS detecte de la video source
        public let sourceFPS: Float
    }

    /// Extrait N frames uniformement reparties depuis une video (~1 fps)
    /// - Parameters:
    ///   - url: URL du fichier video
    ///   - maxFrames: nombre maximum de frames a extraire (defaut: 32)
    ///   - softTokensPerFrame: nombre de soft tokens par frame (defaut: 70)
    /// - Returns: VideoFrames pret pour le vision encoder
    public static func processVideo(
        url: URL,
        maxFrames: Int = defaultMaxFrames,
        softTokensPerFrame: Int = defaultSoftTokensPerFrame
    ) async throws -> VideoFrames {
        let asset = AVAsset(url: url)
        let duration = try await asset.load(.duration)
        let durationSeconds = CMTimeGetSeconds(duration)

        guard durationSeconds > 0 else {
            throw VideoProcessingError.invalidVideo("Duree video invalide")
        }

        // Detecter le FPS source
        let tracks = try await asset.loadTracks(withMediaType: .video)
        let sourceFPS: Float
        if let videoTrack = tracks.first {
            sourceFPS = try await videoTrack.load(.nominalFrameRate)
        } else {
            sourceFPS = 24.0
        }

        // Limiter a maxVideoDurationSeconds
        let effectiveDuration = min(durationSeconds, maxVideoDurationSeconds)

        // Echantillonnage ~1 fps, uniforme sur la duree, plafonné à maxFrames
        let frameCount = min(maxFrames, max(1, Int(effectiveDuration)))
        var times: [CMTime] = []
        var timestamps: [Double] = []

        if frameCount == 1 {
            let t = effectiveDuration / 2
            times.append(CMTime(seconds: t, preferredTimescale: 600))
            timestamps.append(t)
        } else {
            for i in 0 ..< frameCount {
                let t = effectiveDuration * Double(i) / Double(frameCount - 1)
                let clampedT = min(t, durationSeconds - 0.01)
                times.append(CMTime(seconds: clampedT, preferredTimescale: 600))
                timestamps.append(clampedT)
            }
        }

        // Extraire les frames via AVAssetImageGenerator
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        // Ne pas limiter la taille ici — ImageProcessor gere le resize
        generator.requestedTimeToleranceBefore = CMTime(seconds: 0.5, preferredTimescale: 600)
        generator.requestedTimeToleranceAfter = CMTime(seconds: 0.5, preferredTimescale: 600)

        var frames: [MLXArray] = []
        for time in times {
            let (cgImage, _) = try await generator.image(at: time)
            // Reutiliser ImageProcessor avec le budget de soft tokens video (70)
            let frameArray = try Gemma4ImageProcessor.processImage(
                cgImage,
                maxSoftTokens: softTokensPerFrame
            )
            frames.append(frameArray)
        }

        guard !frames.isEmpty else {
            throw VideoProcessingError.noFramesExtracted
        }

        // Stack: [numFrames, C, H, W]
        // Les frames peuvent avoir des tailles differentes (aspect ratio preservé)
        // → padder a la taille max pour pouvoir stacker
        let maxH = frames.map { $0.dim(2) }.max()!
        let maxW = frames.map { $0.dim(3) }.max()!
        var padded: [MLXArray] = []
        for frame in frames {
            let h = frame.dim(2), w = frame.dim(3)
            if h == maxH && w == maxW {
                padded.append(frame)
            } else {
                let result = MLXArray.zeros([1, 3, maxH, maxW], dtype: frame.dtype)
                result[0..., 0..., 0 ..< h, 0 ..< w] = frame
                padded.append(result)
            }
        }
        let pixelValues = concatenated(padded, axis: 0) // [numFrames, C, H, W]

        return VideoFrames(
            pixelValues: pixelValues,
            frameCount: frames.count,
            softTokensPerFrame: softTokensPerFrame,
            totalTokens: frames.count * softTokensPerFrame,
            timestamps: timestamps,
            sourceFPS: sourceFPS
        )
    }

    /// Formate un timestamp en MM:SS pour insertion dans le prompt
    public static func formatTimestamp(_ seconds: Double) -> String {
        let mins = Int(seconds) / 60
        let secs = Int(seconds) % 60
        return String(format: "%02d:%02d", mins, secs)
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
