// Port de generation_config.json de google/diffusiongemma-26B-A4B-it
//
// Parametres de l'algorithme de denoising block-AR :
//  - tMin / tMax           : temperature schedule (lineaire de tMin a tMax)
//  - maxDenoisingSteps    : nb max d'iterations de denoising par bloc
//  - entropyBound         : seuil EntropyBoundSampler (mutual info bound)
//  - stabilityThreshold  : nb de steps consecutifs sans changement d'argmax
//  - confidenceThreshold : seuil sur l'entropie moyenne (stopping criterion)

import Foundation

/// Parametres de generation pour DiffusionGemma (block-AR diffusion).
public struct DiffusionGenerationConfig: Decodable, Sendable {
    public let tMin: Float
    public let tMax: Float
    public let maxDenoisingSteps: Int
    public let entropyBound: Float
    public let stabilityThreshold: Int
    public let confidenceThreshold: Float
    public let eosTokenIds: [Int]
    public let padTokenId: Int

    enum CodingKeys: String, CodingKey {
        case tMin = "t_min"
        case tMax = "t_max"
        case maxDenoisingSteps = "max_denoising_steps"
        case samplerConfig = "sampler_config"
        case stabilityThreshold = "stability_threshold"
        case confidenceThreshold = "confidence_threshold"
        case eosTokenId = "eos_token_id"
        case padTokenId = "pad_token_id"
    }

    enum SamplerKeys: String, CodingKey {
        case entropyBound = "entropy_bound"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        tMin = try c.decodeIfPresent(Float.self, forKey: .tMin) ?? 0.4
        tMax = try c.decodeIfPresent(Float.self, forKey: .tMax) ?? 0.8
        maxDenoisingSteps = try c.decodeIfPresent(Int.self, forKey: .maxDenoisingSteps) ?? 48
        stabilityThreshold = try c.decodeIfPresent(Int.self, forKey: .stabilityThreshold) ?? 1
        confidenceThreshold = try c.decodeIfPresent(Float.self, forKey: .confidenceThreshold) ?? 0.005
        padTokenId = try c.decodeIfPresent(Int.self, forKey: .padTokenId) ?? 0

        // eos_token_id : peut etre Int ou [Int] dans le JSON
        if let single = try? c.decode(Int.self, forKey: .eosTokenId) {
            eosTokenIds = [single]
        } else if let array = try? c.decode([Int].self, forKey: .eosTokenId) {
            eosTokenIds = array
        } else {
            eosTokenIds = [1, 106, 50] // valeurs par defaut du checkpoint 26B-A4B
        }

        // sampler_config.entropy_bound
        if let samplerC = try? c.nestedContainer(keyedBy: SamplerKeys.self, forKey: .samplerConfig) {
            entropyBound = try samplerC.decodeIfPresent(Float.self, forKey: .entropyBound) ?? 0.1
        } else {
            entropyBound = 0.1
        }
    }

    /// Init explicite (utile pour les tests).
    public init(
        tMin: Float = 0.4,
        tMax: Float = 0.8,
        maxDenoisingSteps: Int = 48,
        entropyBound: Float = 0.1,
        stabilityThreshold: Int = 1,
        confidenceThreshold: Float = 0.005,
        eosTokenIds: [Int] = [1, 106, 50],
        padTokenId: Int = 0
    ) {
        self.tMin = tMin
        self.tMax = tMax
        self.maxDenoisingSteps = maxDenoisingSteps
        self.entropyBound = entropyBound
        self.stabilityThreshold = stabilityThreshold
        self.confidenceThreshold = confidenceThreshold
        self.eosTokenIds = eosTokenIds
        self.padTokenId = padTokenId
    }
}
