// Video preprocessor pour gemma4_unified (12B).
//
// Strategie : echantillonner N frames a ~1 fps, traiter chaque frame comme une image
// via [[Gemma4UnifiedImageProcessor]] avec un budget reduit de soft tokens
// (vision_soft_tokens_per_video_frame, typiquement 70).

import AVFoundation
import CoreGraphics
import Foundation
import MLX

public enum Gemma4UnifiedVideoProcessor {

    public struct ProcessedVideoFrame: @unchecked Sendable {
        public let patches: MLXArray      // [N_pad, patchDim]
        public let positionIds: MLXArray  // [N_pad, 2]
        public let validPatches: Int
    }

    public struct ProcessedVideo: @unchecked Sendable {
        public let frames: [ProcessedVideoFrame]
        public let timestamps: [Double]
        public let sourceFPS: Float
    }

    public static let defaultMaxFrames = 32

    /// Extrait des frames + les preprocesse au format patches.
    /// - Parameters:
    ///   - url : fichier video
    ///   - config : vision_config du modele
    ///   - softTokensPerFrame : budget de soft tokens par frame (defaut: 70).
    ///     Override le `numSoftTokens` du config pour les videos (frames plus petites).
    ///   - maxFrames : plafond du nombre de frames
    public static func processVideo(
        url: URL,
        config: Gemma4UnifiedVisionConfig,
        softTokensPerFrame: Int = 70,
        maxFrames: Int = defaultMaxFrames
    ) async throws -> ProcessedVideo {
        let asset = AVAsset(url: url)
        let duration = try await asset.load(.duration)
        let durationSeconds = CMTimeGetSeconds(duration)

        guard durationSeconds > 0 else {
            throw VideoProcessingError.invalidVideo("Duree video invalide")
        }

        let tracks = try await asset.loadTracks(withMediaType: .video)
        let sourceFPS: Float = (try? await tracks.first?.load(.nominalFrameRate)) ?? 24.0

        let effectiveDuration = min(durationSeconds, Gemma4VideoProcessor.maxVideoDurationSeconds)
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

        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = CMTime(seconds: 0.5, preferredTimescale: 600)
        generator.requestedTimeToleranceAfter = CMTime(seconds: 0.5, preferredTimescale: 600)

        // Reduit le budget de patches pour chaque frame (independent du config.numSoftTokens).
        // On construit une config locale avec numSoftTokens=softTokensPerFrame pour
        // que le processor d'image respecte ce budget.
        let frameConfig = Self.makeFrameConfig(base: config, numSoftTokens: softTokensPerFrame)

        var processed: [ProcessedVideoFrame] = []
        for time in times {
            let (cgImage, _) = try await generator.image(at: time)
            let p = try Gemma4UnifiedImageProcessor.processImage(cgImage, config: frameConfig)
            processed.append(ProcessedVideoFrame(
                patches: p.patches,
                positionIds: p.positionIds,
                validPatches: p.validPatches
            ))
        }

        return ProcessedVideo(
            frames: processed,
            timestamps: timestamps,
            sourceFPS: sourceFPS
        )
    }

    /// Construit une config derivee avec un budget de patches reduit (pour video).
    private static func makeFrameConfig(
        base: Gemma4UnifiedVisionConfig,
        numSoftTokens: Int
    ) -> Gemma4UnifiedVisionConfig {
        // On encode en JSON puis re-decode pour reutiliser le decoder existant
        // avec l'override de num_soft_tokens.
        let json: [String: Any] = [
            "model_type": base.modelType,
            "model_patch_size": base.modelPatchSize,
            "patch_size": base.patchSize,
            "pooling_kernel_size": base.poolingKernelSize,
            "mm_embed_dim": base.mmEmbedDim,
            "mm_posemb_size": base.mmPosembSize,
            "num_soft_tokens": numSoftTokens,
            "output_proj_dims": base.outputProjDims,
            "rms_norm_eps": base.rmsNormEps
        ]
        let data = try! JSONSerialization.data(withJSONObject: json)
        return try! JSONDecoder().decode(Gemma4UnifiedVisionConfig.self, from: data)
    }
}
