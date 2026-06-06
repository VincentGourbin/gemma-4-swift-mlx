// Audio preprocessor pour gemma4_unified (12B).
//
// Difference avec [[Gemma4AudioProcessor]] (E2B/E4B mel-spectrogram + Conformer) :
// le 12B prend des chunks PCM bruts (audioSamplesPerToken = 640 samples a 16kHz
// = 40ms par token). Pas de mel, pas de Conformer.

import AVFoundation
import Foundation
import MLX

public enum Gemma4UnifiedAudioProcessor {

    public struct ProcessedAudio: @unchecked Sendable {
        /// [1, T, audioSamplesPerToken] = chunks PCM bruts.
        public let features: MLXArray
        /// [1, T] mask : true pour chunks valides, false pour padding.
        public let mask: MLXArray
        /// Nombre de tokens audio (= chunks valides).
        public let numTokens: Int
        public let durationSeconds: Float
    }

    /// Sample rate fixe (16kHz, comme E2B/E4B).
    public static let sampleRate = 16_000

    /// Pipeline : URL audio -> chunks PCM prets pour embed_audio.
    /// - Parameters:
    ///   - url : fichier audio (n'importe quel format AVFoundation).
    ///   - config : audio_config du modele (samples_per_token = 640 typique).
    ///   - maxDurationSeconds : tronquage de securite (30s par defaut).
    public static func processAudio(
        url: URL,
        config: Gemma4UnifiedAudioConfig,
        maxDurationSeconds: Float = 30.0
    ) async throws -> ProcessedAudio {
        var pcm = try await Gemma4AudioProcessor.loadAudioPCM(url: url)

        let maxSamples = Int(maxDurationSeconds * Float(sampleRate))
        if pcm.count > maxSamples {
            pcm = Array(pcm.prefix(maxSamples))
        }
        let usedDuration = Float(pcm.count) / Float(sampleRate)

        let samplesPerToken = config.audioSamplesPerToken
        // Pad to multiple of samplesPerToken (constant 0, comme le Python).
        let padLen = (samplesPerToken - (pcm.count % samplesPerToken)) % samplesPerToken
        if padLen > 0 {
            pcm.append(contentsOf: [Float](repeating: 0, count: padLen))
        }

        let numTokens = pcm.count / samplesPerToken
        let features = MLXArray(pcm).reshaped(1, numTokens, samplesPerToken)
        let mask = MLXArray.ones([1, numTokens], type: Bool.self)

        return ProcessedAudio(
            features: features,
            mask: mask,
            numTokens: numTokens,
            durationSeconds: usedDuration
        )
    }
}
