// Processeur audio pour Gemma 4 — Extraction de features mel spectrogram
// Parametres alignes sur Gemma4AudioFeatureExtractor Python

import AVFoundation
import Accelerate
import Foundation
import MLX

/// Processeur audio compatible Gemma 4
/// Extrait les features log-mel spectrogram (128 bins) depuis un fichier audio
public enum Gemma4AudioProcessor {

    // Parametres alignes sur le Python Gemma4AudioFeatureExtractor
    static let numMelFilters = 128
    static let sampleRate = 16000
    static let frameLengthMs: Float = 20.0
    static let hopLengthMs: Float = 10.0
    static let minFrequency: Float = 0.0
    static let maxFrequency: Float = 8000.0
    static let melFloor: Float = 1e-3
    static let fftOverdrive = false // processor_config.json specifies fft_length=512 directly
    /// ms par token audio (de processor_config.json)
    static let msPerToken: Float = 40.0

    // Calculs derives
    static var frameLength: Int { Int(round(Float(sampleRate) * frameLengthMs / 1000.0)) } // 320
    static var hopLength: Int { Int(round(Float(sampleRate) * hopLengthMs / 1000.0)) }     // 160
    static var fftLength: Int {
        var n = 1
        while n < frameLength { n *= 2 }  // 512
        return fftOverdrive ? n * 2 : n    // 1024
    }

    /// Resultat du traitement audio
    public struct AudioFeatures: @unchecked Sendable {
        public let features: MLXArray  // [1, T, 128]
        public let mask: MLXArray      // [1, T]
        public let numTokens: Int
        public let durationSeconds: Float
    }

    /// Charge et preprocesse un fichier audio
    /// - Parameters:
    ///   - url: chemin du fichier audio
    ///   - maxDurationSeconds: duree max a traiter (defaut 30s)
    public static func processAudio(url: URL, maxDurationSeconds: Float = 30.0) async throws -> AudioFeatures {
        var pcmData = try await loadAudioPCM(url: url)

        // Tronquer le PCM si trop long
        let maxSamples = Int(maxDurationSeconds * Float(sampleRate))
        if pcmData.count > maxSamples {
            pcmData = Array(pcmData.prefix(maxSamples))
        }
        let usedDuration = Float(pcmData.count) / Float(sampleRate)

        // Calculer le mel spectrogram
        let melData = computeLogMelSpectrogram(pcm: pcmData)
        let T = melData.count / numMelFilters

        // Le conformer subsample par 4 (2x Conv2d stride 2)
        let conformerOutputTokens = T / 4

        let features = MLXArray(melData).reshaped(1, T, numMelFilters)
        let mask = MLXArray.zeros([1, T], type: Bool.self)

        return AudioFeatures(
            features: features,
            mask: mask,
            numTokens: conformerOutputTokens,
            durationSeconds: usedDuration
        )
    }

    /// Charge l'audio en PCM float32 mono 16kHz
    static func loadAudioPCM(url: URL) async throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        let origFormat = file.processingFormat
        let frameCount = AVAudioFrameCount(file.length)

        guard let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double(sampleRate),
            channels: 1,
            interleaved: false
        ) else {
            throw AudioProcessingError.formatError
        }

        guard let origBuffer = AVAudioPCMBuffer(pcmFormat: origFormat, frameCapacity: frameCount) else {
            throw AudioProcessingError.bufferError
        }
        try file.read(into: origBuffer)

        if origFormat.sampleRate == Double(sampleRate) && origFormat.channelCount == 1 {
            let data = origBuffer.floatChannelData![0]
            return Array(UnsafeBufferPointer(start: data, count: Int(origBuffer.frameLength)))
        }

        guard let converter = AVAudioConverter(from: origFormat, to: targetFormat) else {
            throw AudioProcessingError.conversionError
        }

        let ratio = Double(sampleRate) / origFormat.sampleRate
        let targetFrameCount = AVAudioFrameCount(Double(frameCount) * ratio) + 1024
        guard let targetBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: targetFrameCount) else {
            throw AudioProcessingError.bufferError
        }

        var isDone = false
        try converter.convert(to: targetBuffer, error: nil) { _, outStatus in
            if isDone {
                outStatus.pointee = .noDataNow
                return nil
            }
            outStatus.pointee = .haveData
            isDone = true
            return origBuffer
        }

        let data = targetBuffer.floatChannelData![0]
        return Array(UnsafeBufferPointer(start: data, count: Int(targetBuffer.frameLength)))
    }

    /// Calcule le log-mel spectrogram avec les parametres exacts de Gemma 4
    static func computeLogMelSpectrogram(pcm: [Float]) -> [Float] {
        let fl = frameLength   // 320
        let hl = hopLength     // 160
        let fftLen = fftLength // 1024
        let numFrames = max(1, (pcm.count - fl) / hl + 1)
        let numBins = fftLen / 2 + 1

        // Fenetre de Hanning
        var hanningWindow = [Float](repeating: 0, count: fl)
        for i in 0 ..< fl {
            hanningWindow[i] = 0.5 * (1.0 - cos(2.0 * Float.pi * Float(i) / Float(fl)))
        }

        // Mel filter bank
        let melFilters = createMelFilterBank(
            numFilters: numMelFilters, numBins: numBins,
            sampleRate: sampleRate, minFreq: minFrequency, maxFreq: maxFrequency
        )

        // FFT setup
        let log2n = vDSP_Length(log2(Float(fftLen)))
        guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
            return [Float](repeating: 0, count: numFrames * numMelFilters)
        }
        defer { vDSP_destroy_fftsetup(fftSetup) }

        var melOutput = [Float](repeating: 0, count: numFrames * numMelFilters)

        for frame in 0 ..< numFrames {
            let start = frame * hl

            // Fenetre + zero-padding a fftLen
            var windowed = [Float](repeating: 0, count: fftLen)
            for i in 0 ..< min(fl, pcm.count - start) {
                windowed[i] = pcm[start + i] * hanningWindow[i]
            }

            // FFT magnitude
            let magnitudes = computeFFTMagnitude(windowed, fftSetup: fftSetup, log2n: log2n)

            // Appliquer les filtres mel + log
            for m in 0 ..< numMelFilters {
                var energy: Float = 0
                for k in 0 ..< min(magnitudes.count, melFilters[m].count) {
                    energy += magnitudes[k] * melFilters[m][k]
                }
                // Log mel avec floor
                melOutput[frame * numMelFilters + m] = log(max(energy, melFloor))
            }
        }

        return melOutput
    }

    /// FFT magnitude via vDSP
    static func computeFFTMagnitude(_ signal: [Float], fftSetup: FFTSetup, log2n: vDSP_Length) -> [Float] {
        let n = signal.count
        let halfN = n / 2

        var realPart = [Float](repeating: 0, count: halfN)
        var imagPart = [Float](repeating: 0, count: halfN)

        signal.withUnsafeBufferPointer { ptr in
            ptr.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: halfN) { complexPtr in
                var splitComplex = DSPSplitComplex(realp: &realPart, imagp: &imagPart)
                vDSP_ctoz(complexPtr, 2, &splitComplex, 1, vDSP_Length(halfN))
                vDSP_fft_zrip(fftSetup, &splitComplex, 1, log2n, FFTDirection(FFT_FORWARD))
            }
        }

        // Power spectrum (magnitude squared)
        var magnitudes = [Float](repeating: 0, count: halfN + 1)
        // DC component
        magnitudes[0] = realPart[0] * realPart[0]
        // Nyquist
        magnitudes[halfN] = imagPart[0] * imagPart[0]
        // Remaining bins
        for i in 1 ..< halfN {
            magnitudes[i] = realPart[i] * realPart[i] + imagPart[i] * imagPart[i]
        }

        return magnitudes
    }

    /// Cree un banc de filtres mel triangulaires (HTK scale)
    static func createMelFilterBank(numFilters: Int, numBins: Int, sampleRate: Int, minFreq: Float, maxFreq: Float) -> [[Float]] {
        func hzToMel(_ hz: Float) -> Float { 2595.0 * log10(1.0 + hz / 700.0) }
        func melToHz(_ mel: Float) -> Float { 700.0 * (pow(10.0, mel / 2595.0) - 1.0) }

        let minMel = hzToMel(minFreq)
        let maxMel = hzToMel(maxFreq)

        // Points mel uniformement espaces
        var melPoints = [Float](repeating: 0, count: numFilters + 2)
        for i in 0 ..< numFilters + 2 {
            melPoints[i] = minMel + Float(i) * (maxMel - minMel) / Float(numFilters + 1)
        }

        // Convertir en indices de bins FFT
        let fftLen = (numBins - 1) * 2
        let binPoints = melPoints.map { mel -> Int in
            let hz = melToHz(mel)
            return Int(round(hz * Float(fftLen) / Float(sampleRate)))
        }

        // Filtres triangulaires
        var filters = [[Float]](repeating: [Float](repeating: 0, count: numBins), count: numFilters)
        for m in 0 ..< numFilters {
            let left = binPoints[m]
            let center = binPoints[m + 1]
            let right = binPoints[m + 2]

            for k in left ..< min(center, numBins) {
                if center > left {
                    filters[m][k] = Float(k - left) / Float(center - left)
                }
            }
            for k in center ..< min(right, numBins) {
                if right > center {
                    filters[m][k] = Float(right - k) / Float(right - center)
                }
            }
        }

        return filters
    }
}

public enum AudioProcessingError: LocalizedError {
    case formatError
    case bufferError
    case conversionError
    case invalidAudio(String)

    public var errorDescription: String? {
        switch self {
        case .formatError: return "Format audio non supporte"
        case .bufferError: return "Erreur de buffer audio"
        case .conversionError: return "Erreur de conversion audio"
        case .invalidAudio(let msg): return "Audio invalide: \(msg)"
        }
    }
}
