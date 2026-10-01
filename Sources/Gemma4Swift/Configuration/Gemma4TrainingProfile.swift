// Profils d'entrainement LoRA lora-<bits>bit-<fast|lean> (K-33, audit annexes § 1.4)

import Foundation

/// Configuration d'entrainement nommee, sur le modele des profils d'inference
/// (`Gemma4ReferenceProfile`) : chaque champ est un reglage de `TrainingConfig` qui
/// existe deja, fige et mesure.
///
/// - `bits` : largeur des poids de **base** (16 = LoRA sur bf16 ; 8/4 = QLoRA sur pack
///   quantifie).
/// - `fast` : politique memoire K-30 b (cache MLX 2 Go), sans gradient checkpointing sur
///   E2B/E4B.
/// - `lean` : gradient checkpointing par couche (K-32 : pic -45 %, temps +35 %, pertes
///   identiques), cache MLX 1 Go.
///
/// 12B, 26B-A4B et 31B ont le checkpointing **aussi en `fast`** : sans lui, E2B bf16 monte
/// deja a 38,6 Go de pic sur director, et les activations croissent avec largeur x couches
/// (x 3,4 pour le 12B) ; une machine de 96 Go n'y suffirait pas.
///
/// Communs : rang 8, echelle 20, lr 1e-4 (31B 4 bits : 3e-5), batch 1, tete sur la reponse seule (K-30 a),
/// 25 lots de validation au plus, **aucune troncature** (sur le dataset director, borner a
/// 2048 jetons coupait la fin des reponses : 30/30 -> 27/30). Les exemples longs font le
/// pic : le profil ne le borne pas, `maxSeqLength` reste au choix de l'appelant.
///
/// Un profil n'est publie (`all`, `named`) qu'une fois mesure ; les autres restent dans
/// `candidates` pour la campagne de mesure.
public struct Gemma4TrainingProfile: Sendable, Identifiable, Equatable {

    public enum Kind: String, CaseIterable, Sendable { case fast, lean }

    /// Mesure de reference (campagne K-33, `Scripts/bench-training-campaign.sh`) : dataset
    /// director de Fluxforge (898 exemples, jusqu'a 3 320 jetons), graine 0, `steps` pas,
    /// validation sur 10 lots, M3 Max 96 Go. Lignes brutes : `benchmarks/k33-*`.
    public struct Measurement: Sendable, Equatable {
        /// Pic MLX (`Memory.peakMemory`), en Go.
        public let peakMLXGB: Double
        /// Empreinte physique maximale du processus (`phys_footprint`), en Go.
        public let footprintGB: Double
        /// Jetons de reponse entraines par seconde, moyenne sur le run.
        public let trainedTokensPerSecond: Double
        /// Perte de validation au dernier pas (10 lots).
        public let validationLoss: Double
        public let steps: Int
        /// Porte qualite E7 (30 briefs tenus a l'ecart), si un run complet a ete evalue.
        public let e7Valid: Int?
        public let date: String
    }

    public let family: Gemma4Pipeline.Model.Family
    public let bits: Gemma4ReferenceProfile.Bits
    public let kind: Kind
    /// Poids de base (pack mlx-community ; bf16 pour `.sixteen`).
    public let model: Gemma4Pipeline.Model
    public let rank: Int
    public let scale: Float
    public let numLayers: Int
    public let learningRate: Float
    public let batchSize: Int
    public let gradientCheckpointing: Bool
    public let memoryPolicy: Gemma4TrainingMemoryPolicy
    public let validationBatches: Int
    public let measurement: Measurement?
    public let summary: String

    public var id: String { "lora-\(bits.rawValue)bit-\(kind.rawValue)" }

    /// Identifiant complet, unique : `e2b/lora-16bit-fast`.
    public var qualifiedID: String { "\(family.rawValue)/\(id)" }

    /// Famille au sens des defauts LoRA.
    public var loraFamily: Gemma4LoRADefaults.ModelFamily {
        switch family {
        case .e2b: return .e2b
        case .e4b: return .e4b
        case .b12b: return .b12b
        case .a4b, .a4bDiff: return .a4b
        case .b31b: return .dense31b
        }
    }

    /// Pose les reglages du profil ; ne touche ni aux iterations, ni au dossier de sortie,
    /// ni a la graine, ni a `maxSeqLength`.
    public func apply(to config: inout Gemma4LoRATrain.TrainingConfig) {
        config.fineTuneType = .lora
        config.loraRank = rank
        config.loraScale = scale
        config.numLayers = numLayers
        config.modelFamily = loraFamily
        config.learningRate = learningRate
        config.batchSize = batchSize
        config.gradientCheckpointing = gradientCheckpointing
        config.memoryPolicy = memoryPolicy
        config.validationBatches = validationBatches
        config.responseOnlyHead = true
    }

    /// Profil publie (mesure) par id complet (`e2b/lora-16bit-fast`) ou court avec famille.
    public static func named(
        _ id: String, family: Gemma4Pipeline.Model.Family? = nil, includingUnmeasured: Bool = false
    ) -> Gemma4TrainingProfile? {
        (includingUnmeasured ? candidates : all).first { profile in
            profile.qualifiedID == id || (profile.id == id && (family == nil || profile.family == family))
        }
    }

    /// Profils publies : ceux qui ont une mesure.
    public static var all: [Gemma4TrainingProfile] { candidates.filter { $0.measurement != nil } }

    /// Matrice proposee (audit annexes § 1.4), mesuree ou non.
    public static let candidates: [Gemma4TrainingProfile] = [
        make(.e2b, .sixteen, .fast, numLayers: 16),
        make(.e2b, .sixteen, .lean, numLayers: 16),
        make(.e2b, .four, .lean, numLayers: 16),
        make(.e4b, .sixteen, .fast, numLayers: 12),
        make(.e4b, .sixteen, .lean, numLayers: 12),
        make(.e4b, .eight, .lean, numLayers: 12),
        make(.b12b, .sixteen, .fast, numLayers: 16),
        make(.b12b, .eight, .lean, numLayers: 16),
        make(.a4b, .sixteen, .fast, numLayers: 10),
        make(.a4b, .four, .lean, numLayers: 10),
        make(.b31b, .eight, .fast, numLayers: 16),
        // lr 1e-4 diverge sur cette base (val 1,554 -> 1,777 en 50 pas) ; 1e-5 et 3e-5 convergent
        // (0,957 et 0,945), 3e-5 retenu (2026-10-01, `benchmarks/k33-31b4-lr-20261001`).
        make(.b31b, .four, .lean, numLayers: 16, learningRate: 3e-5),
    ].compactMap { $0 }

    /// Mesures de la campagne K-33 (2026-09-29/30), 50 pas. Pertes de validation au pas 1 :
    /// E2B 1,85 (4 bits 1,89), E4B 1,41, 12B 1,40-1,42, 26B-A4B 1,71 (4 bits 1,87), 31B 1,55.
    ///
    /// Les pertes « au pas 1 » de ces mesures ont ete prises **apres** la premiere mise a jour
    /// (corrige le 2026-10-01 : la validation initiale precede desormais le premier pas) ; les
    /// pertes au pas 50 ne sont pas concernees.
    static let measurements: [String: Measurement] = [
        "e2b/lora-16bit-fast": .init(peakMLXGB: 37.0, footprintGB: 13.9, trainedTokensPerSecond: 218, validationLoss: 1.276, steps: 50, e7Valid: nil, date: "2026-09-30"),
        "e2b/lora-16bit-lean": .init(peakMLXGB: 20.7, footprintGB: 12.8, trainedTokensPerSecond: 168, validationLoss: 1.276, steps: 50, e7Valid: nil, date: "2026-09-29"),
        "e2b/lora-4bit-lean": .init(peakMLXGB: 14.5, footprintGB: 6.7, trainedTokensPerSecond: 132, validationLoss: 1.339, steps: 50, e7Valid: nil, date: "2026-09-29"),
        // E7 : epoque complete (898 pas, val 0,988 au pas 600, pic 40,6 Go) -> 29/30 (base E4B 19/30).
        "e4b/lora-16bit-fast": .init(peakMLXGB: 35.7, footprintGB: 18.7, trainedTokensPerSecond: 144, validationLoss: 1.157, steps: 50, e7Valid: 29, date: "2026-09-29"),
        "e4b/lora-16bit-lean": .init(peakMLXGB: 24.1, footprintGB: 18.3, trainedTokensPerSecond: 115, validationLoss: 1.157, steps: 50, e7Valid: nil, date: "2026-09-29"),
        "e4b/lora-8bit-lean": .init(peakMLXGB: 16.8, footprintGB: 11.5, trainedTokensPerSecond: 85, validationLoss: 1.160, steps: 50, e7Valid: nil, date: "2026-09-29"),
        "b12b/lora-16bit-fast": .init(peakMLXGB: 37.9, footprintGB: 27.5, trainedTokensPerSecond: 35, validationLoss: 1.131, steps: 50, e7Valid: nil, date: "2026-09-30"),
        "b12b/lora-8bit-lean": .init(peakMLXGB: 27.9, footprintGB: 16.4, trainedTokensPerSecond: 30, validationLoss: 1.100, steps: 50, e7Valid: nil, date: "2026-09-30"),
        "a4b/lora-16bit-fast": .init(peakMLXGB: 57.6, footprintGB: 52.3, trainedTokensPerSecond: 67, validationLoss: 1.084, steps: 50, e7Valid: nil, date: "2026-09-30"),
        "a4b/lora-4bit-lean": .init(peakMLXGB: 20.4, footprintGB: 17.6, trainedTokensPerSecond: 68, validationLoss: 1.140, steps: 50, e7Valid: nil, date: "2026-09-30"),
        "b31b/lora-8bit-fast": .init(peakMLXGB: 54.8, footprintGB: 35.9, trainedTokensPerSecond: 14, validationLoss: 1.289, steps: 50, e7Valid: nil, date: "2026-09-30"),
        "b31b/lora-4bit-lean": .init(peakMLXGB: 41.2, footprintGB: 20.4, trainedTokensPerSecond: 14, validationLoss: 0.945, steps: 50, e7Valid: nil, date: "2026-10-01"),
    ]

    private static func make(
        _ family: Gemma4Pipeline.Model.Family, _ bits: Gemma4ReferenceProfile.Bits, _ kind: Kind,
        numLayers: Int, learningRate: Float = 1e-4
    ) -> Gemma4TrainingProfile? {
        guard let model = Gemma4ReferenceProfile.named("\(bits.rawValue)bit-fast", family: family)?.model
        else { return nil }
        let lean = kind == .lean
        let qualifiedID = "\(family.rawValue)/lora-\(bits.rawValue)bit-\(kind.rawValue)"
        let checkpointing = lean || [.b12b, .a4b, .b31b].contains(family)
        var notes: [String] = [lean
            ? "Econome : gradient checkpointing par couche, cache MLX 1 Go."
            : checkpointing
            ? "Rapide : cache MLX 2 Go ; checkpointing garde (activations trop grosses sans)."
            : "Rapide : sans checkpointing, cache MLX 2 Go."]
        if bits != .sixteen { notes.append("QLoRA sur pack \(bits.rawValue) bits.") }
        if family == .a4b { notes.append("Experts MoE non adaptes (SwitchLinear n'est pas un Linear) ; MLP dense et attention le sont.") }
        return Gemma4TrainingProfile(
            family: family, bits: bits, kind: kind, model: model,
            rank: 8, scale: 20, numLayers: numLayers, learningRate: learningRate, batchSize: 1,
            gradientCheckpointing: checkpointing,
            memoryPolicy: Gemma4TrainingMemoryPolicy(cacheLimitMB: lean ? 1024 : 2048),
            validationBatches: 25,
            measurement: measurements[qualifiedID],
            summary: notes.joined(separator: " "))
    }
}
