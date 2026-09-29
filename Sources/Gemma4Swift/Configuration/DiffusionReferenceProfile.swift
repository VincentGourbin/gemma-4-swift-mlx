// Profils de reference DiffusionGemma <bits>bit-<fast|lean> (K-D9, audit diffusion §C)

import Foundation
import MLX

/// Configuration nommee de DiffusionGemma 26B-A4B qui fige tous les reglages qui
/// comptent : quantification (a la volee depuis le bf16 officiel), vision, politique
/// memoire. Type frere de `Gemma4ReferenceProfile` : memes `Bits`, `Kind`, format
/// d'identifiant et regle de memoire disponible, mais champs propres a la diffusion
/// (pas de `kvBits` ni de tranche de prefill, la diffusion n'utilise pas
/// `GenerateParameters`).
///
/// Le debruitage (`generation_config.json`) est identique dans les six profils : la
/// vitesse ne s'achete pas avec la qualite dans un profil de reference.
///
/// **Etat au 2026-09-28 : valeurs initiales, non mesurees** (audit diffusion, K-D14).
public struct DiffusionReferenceProfile: Sendable, Identifiable, Equatable {

    public typealias Bits = Gemma4ReferenceProfile.Bits
    public typealias Kind = Gemma4ReferenceProfile.Kind

    /// Quantification appliquee au chargement.
    public enum Quantization: Sendable, Equatable {
        /// bf16 du checkpoint.
        case none
        /// Tout le modele texte (experts compris, routeur en 8 bits), vision en bf16.
        case uniform(bits: Int, groupSize: Int)
        /// Precision mixte par couche (`applyMixedPrecision`), vision en bf16.
        case mixed(DiffusionOnTheFlyQuantization.MixedPrecisionConfig)

        /// Identifiant stable, ecrit dans un pack pre-quantifie et compare au chargement.
        public var signature: String {
            switch self {
            case .none: return "none"
            case .uniform(let bits, let groupSize): return "uniform-\(bits)bit-g\(groupSize)"
            case .mixed(let c):
                let layers = c.highPrecisionLayers.sorted().map(String.init).joined(separator: ",")
                return "mixed-\(c.lowPrecisionBits)/\(c.highPrecisionBits)bit-g\(c.groupSize)-layers[\(layers)]"
                    + (c.quantizeSensitiveAtHighPrecision ? "-sensitive" : "")
            }
        }
    }

    /// Checkpoint source (bf16 officiel, ~50 Go).
    public static let checkpointID = "google/diffusiongemma-26B-A4B-it"

    public let bits: Bits
    public let kind: Kind
    public let quantization: Quantization
    /// Tour vision chargee (`DiffusionGemmaRegistration.load(includeVision:)`).
    public let includeVision: Bool
    /// Decharger la vision apres le premier canvas (`DiffusionMemoryConfig`).
    public let unloadVisionAfterFirstCanvas: Bool
    /// `Memory.clearCache()` entre canvases (`DiffusionMemoryConfig`).
    public let clearCacheBetweenCanvases: Bool
    /// `Memory.cacheLimit` en Mo.
    public let cacheLimitMB: Int?
    /// `Memory.memoryLimit` en Mo (seuil de liberation du cache, pas un plafond dur).
    public let memoryLimitMB: Int?
    /// Taille estimee des poids en Go (calcul, a remplacer par la mesure).
    public let estimatedWeightsGB: Float
    public let summary: String

    public var id: String { "\(bits.rawValue)bit-\(kind.rawValue)" }
    public var qualifiedID: String { "a4bdiff/\(id)" }

    /// Politique memoire a passer au pipeline.
    public var memoryConfig: DiffusionMemoryConfig {
        DiffusionMemoryConfig(
            mixedPrecision: nil,
            unloadVisionAfterFirstCanvas: unloadVisionAfterFirstCanvas,
            clearCacheBetweenCanvases: clearCacheBetweenCanvases)
    }

    /// Pose les reglages de processus. A appeler apres chargement et quantification.
    public func applyGlobalPolicy() {
        if let cacheLimitMB { Memory.cacheLimit = cacheLimitMB * 1_048_576 }
        if let memoryLimitMB { Memory.memoryLimit = memoryLimitMB * 1_048_576 }
    }

    /// Meme profil avec une autre quantification (mesure de variantes).
    public func withQuantization(_ quantization: Quantization) -> DiffusionReferenceProfile {
        DiffusionReferenceProfile(
            bits: bits, kind: kind, quantization: quantization, includeVision: includeVision,
            unloadVisionAfterFirstCanvas: unloadVisionAfterFirstCanvas,
            clearCacheBetweenCanvases: clearCacheBetweenCanvases,
            cacheLimitMB: cacheLimitMB, memoryLimitMB: memoryLimitMB,
            estimatedWeightsGB: estimatedWeightsGB, summary: summary)
    }

    /// Meme profil sans tour vision.
    public func textOnlyVariant() -> DiffusionReferenceProfile {
        DiffusionReferenceProfile(
            bits: bits, kind: kind, quantization: quantization, includeVision: false,
            unloadVisionAfterFirstCanvas: false, clearCacheBetweenCanvases: clearCacheBetweenCanvases,
            cacheLimitMB: cacheLimitMB, memoryLimitMB: memoryLimitMB,
            estimatedWeightsGB: estimatedWeightsGB, summary: summary + " Variante texte seul.")
    }

    /// `a4bdiff/4bit-lean` ou `4bit-lean`.
    public static func named(_ id: String) -> DiffusionReferenceProfile? {
        all.first { $0.qualifiedID == id || $0.id == id }
    }

    /// Le `fast` le plus large dont les poids tiennent sous la moitie de la memoire
    /// disponible, sinon le `lean` le plus petit (meme regle que `Gemma4ReferenceProfile`).
    public static func recommended(
        availableMB: Int = Gemma4ReferenceProfile.availableMemoryMB()
    ) -> DiffusionReferenceProfile? {
        let budgetGB = Float(availableMB) / 1024 / 2
        if let fast = all.filter({ $0.kind == .fast && $0.estimatedWeightsGB <= budgetGB })
            .max(by: { $0.estimatedWeightsGB < $1.estimatedWeightsGB }) {
            return fast
        }
        return all.filter { $0.kind == .lean }.min { $0.estimatedWeightsGB < $1.estimatedWeightsGB }
    }

    public static let all: [DiffusionReferenceProfile] = Bits.allCases.flatMap { bits in
        Kind.allCases.map { make(bits: bits, kind: $0) }
    }

    private static func make(bits: Bits, kind: Kind) -> DiffusionReferenceProfile {
        let lean = kind == .lean
        let available = Gemma4ReferenceProfile.availableMemoryMB()
        // 16bit-lean garde les caches Mac (bf16 + limites serrees = thrash, YuE2).
        let macCaches = !lean || bits == .sixteen
        let quantization: Quantization
        let weightsGB: Float
        switch bits {
        case .sixteen: quantization = .none; weightsGB = 50
        case .eight: quantization = .uniform(bits: 8, groupSize: 64); weightsGB = 27
        // 4 bits uniforme : 43,5 passes/canvas contre 15 en bf16 (3x plus lent) ; couches
        // 0-3 et 26-29 en 8 bits : 14,5 passes, 38,5 tok/s, ~19 Go actifs (2026-09-28).
        case .four: quantization = .mixed(.default); weightsGB = 19
        }
        var notes = ["DiffusionGemma 26B-A4B."]
        if bits != .sixteen {
            notes.append("Quantification a la volee depuis le bf16 (experts compris, routeur 8 bits, vision bf16) : ~51 Go au chargement.")
            if bits == .four { notes.append("Couches 0-3 et 26-29 en 8 bits, le reste en 4.") }
        }
        notes.append(lean ? "Econome : vision dechargee apres le 1er canvas, limites memoire adaptees." : "Rapide : tout resident.")
        notes.append("Vitesse et memoire mesurees en texte (BENCHMARKS.md), qualite non mesuree.")
        return DiffusionReferenceProfile(
            bits: bits, kind: kind, quantization: quantization, includeVision: true,
            unloadVisionAfterFirstCanvas: lean, clearCacheBetweenCanvases: lean,
            cacheLimitMB: macCaches ? 4096 : min(1024, max(256, available / 6)),
            memoryLimitMB: macCaches ? nil : max(4096, available - 2048),
            estimatedWeightsGB: weightsGB, summary: notes.joined(separator: " "))
    }
}
