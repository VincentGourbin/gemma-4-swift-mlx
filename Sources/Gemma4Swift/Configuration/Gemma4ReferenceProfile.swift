// Profils de reference <bits>bit-<fast|lean> (K-20, standard des projets MLX)

import Foundation
import MLX
import MLXLMCommon
#if canImport(Darwin)
import Darwin
#endif

/// Configuration nommee qui fige **tous** les reglages qui comptent : poids,
/// cache KV, tranche de prefill, limites memoire, modalites. Meme profil + meme
/// graine = meme sortie, temps comparable sur materiel comparable.
///
/// Standard commun aux ports MLX de Vincent (YuE2, Qwen38) : chaque champ est un
/// reglage qui existe deja ; un profil n'introduit aucun cablage nouveau.
///
/// - `fast` : tout resident, cache MLX genereux, tranche de prefill 512.
/// - `lean` : limites memoire calees sur la memoire disponible, tranche 256, cache
///   MLX vide apres chaque reponse, KV 8 bits la ou il y a plusieurs tetes KV.
///
/// **Etat au 2026-09-27 : valeurs initiales, non mesurees.** La table mesuree
/// (temps, pic) viendra de `gemma4-cli bench` selon `docs/Benchmarks.md` ; un profil
/// dont une valeur change apres mesure garde son id.
public struct Gemma4ReferenceProfile: Sendable, Identifiable, Equatable {

    public enum Bits: String, CaseIterable, Sendable { case four = "4", eight = "8", sixteen = "16" }
    public enum Kind: String, CaseIterable, Sendable { case fast, lean }

    public let family: Gemma4Pipeline.Model.Family
    public let bits: Bits
    public let kind: Kind
    /// Poids recommandes (pack mlx-community ; bf16 pour `.sixteen`).
    public let model: Gemma4Pipeline.Model
    /// KV quantifie (`GenerateParameters.kvBits`), `nil` = bf16. Non applique sur les
    /// chemins avec interdiction de n-grammes (TokenIterator construit sans kvBits).
    public let kvBits: Int?
    /// Tranche de prefill (`GenerateParameters.prefillStepSize`).
    public let prefillStepSize: Int
    /// `Memory.cacheLimit` en Mo, `nil` = laisser MLX.
    public let cacheLimitMB: Int?
    /// `Memory.memoryLimit` en Mo : seuil de liberation du cache MLX, pas un plafond dur.
    public let memoryLimitMB: Int?
    /// `Memory.clearCache()` a la fin de chaque reponse.
    public let clearCacheAfterAnswer: Bool
    /// Tours vision/audio charges (`load(multimodal:)`). Variante : `textOnlyVariant()`.
    public let multimodal: Bool
    public let summary: String

    public var id: String { "\(bits.rawValue)bit-\(kind.rawValue)" }

    /// Identifiant complet, unique dans `all` : `e2b/4bit-fast`.
    public var qualifiedID: String { "\(family.rawValue)/\(id)" }

    /// Pose les reglages de processus. A appeler **apres** le chargement.
    public func applyGlobalPolicy() {
        if let cacheLimitMB { Memory.cacheLimit = cacheLimitMB * 1_048_576 }
        if let memoryLimitMB { Memory.memoryLimit = memoryLimitMB * 1_048_576 }
    }

    /// Applique les reglages par appel a des parametres de generation.
    public func apply(to parameters: inout GenerateParameters) {
        parameters.prefillStepSize = prefillStepSize
        parameters.kvBits = kvBits
        if kvBits != nil { parameters.kvGroupSize = 64 }
    }

    /// Meme profil sans tours vision/audio.
    public func textOnlyVariant() -> Gemma4ReferenceProfile {
        Gemma4ReferenceProfile(
            family: family, bits: bits, kind: kind, model: model, kvBits: kvBits,
            prefillStepSize: prefillStepSize, cacheLimitMB: cacheLimitMB,
            memoryLimitMB: memoryLimitMB, clearCacheAfterAnswer: clearCacheAfterAnswer,
            multimodal: false, summary: summary + " Variante texte seul.")
    }

    /// Profil par identifiant (`4bit-fast`) dans une famille, ou identifiant complet
    /// (`e2b/4bit-fast`).
    public static func named(_ id: String, family: Gemma4Pipeline.Model.Family? = nil) -> Gemma4ReferenceProfile? {
        all.first { profile in
            profile.qualifiedID == id || (profile.id == id && (family == nil || profile.family == family))
        }
    }

    /// Profils d'une famille.
    public static func profiles(for family: Gemma4Pipeline.Model.Family) -> [Gemma4ReferenceProfile] {
        all.filter { $0.family == family }
    }

    /// Profil conseille pour la memoire disponible : le plus large `fast` dont le
    /// pack tient sous la moitie de la memoire, sinon le `lean` le plus petit.
    public static func recommended(
        for family: Gemma4Pipeline.Model.Family, availableMB: Int = availableMemoryMB()
    ) -> Gemma4ReferenceProfile? {
        let candidates = profiles(for: family)
        let budgetGB = Float(availableMB) / 1024 / 2
        if let fast = candidates.filter({ $0.kind == .fast && $0.model.estimatedSizeGB <= budgetGB })
            .max(by: { $0.model.estimatedSizeGB < $1.model.estimatedSizeGB }) {
            return fast
        }
        return candidates.filter { $0.kind == .lean }
            .min { $0.model.estimatedSizeGB < $1.model.estimatedSizeGB }
    }

    /// Memoire disponible en Mo. iOS : `os_proc_available_memory()` ; macOS :
    /// physique - 8 Go. `GEMMA4_AVAILABLE_MB` la remplace (simuler un Mac 16 Go).
    public static func availableMemoryMB() -> Int {
        if let override = ProcessInfo.processInfo.environment["GEMMA4_AVAILABLE_MB"], let value = Int(override) {
            return value
        }
        #if os(iOS)
        return Int(os_proc_available_memory() / 1_048_576)
        #else
        return max(2048, Int(ProcessInfo.processInfo.physicalMemory / 1_048_576) - 8192)
        #endif
    }

    // MARK: - Matrice

    /// Toutes les familles hors DiffusionGemma, 4/8/16 bits x fast/lean.
    public static let all: [Gemma4ReferenceProfile] = Gemma4Pipeline.Model.Family.allCases
        .filter { $0 != .a4bDiff }
        .flatMap { family in
            Bits.allCases.flatMap { bits in
                Kind.allCases.compactMap { kind in make(family: family, bits: bits, kind: kind) }
            }
        }

    private static func pack(_ family: Gemma4Pipeline.Model.Family, _ bits: Bits) -> Gemma4Pipeline.Model? {
        switch (family, bits) {
        case (.e2b, .four): return .e2b4bit
        case (.e2b, .eight): return .e2b8bit
        case (.e2b, .sixteen): return .e2bBf16
        case (.e4b, .four): return .e4b4bit
        case (.e4b, .eight): return .e4b8bit
        case (.e4b, .sixteen): return .e4bBf16
        case (.b12b, .four): return .b12b4bit
        case (.b12b, .eight): return .b12b8bit
        case (.b12b, .sixteen): return .b12bBf16
        case (.a4b, .four): return .a4b4bit
        case (.a4b, .eight): return .a4b8bit
        case (.a4b, .sixteen): return .a4bBf16
        case (.b31b, .four): return .b31b4bit
        case (.b31b, .eight): return .b31b8bit
        case (.b31b, .sixteen): return .b31bBf16
        default: return nil
        }
    }

    private static func make(
        family: Gemma4Pipeline.Model.Family, bits: Bits, kind: Kind
    ) -> Gemma4ReferenceProfile? {
        guard let model = pack(family, bits) else { return nil }
        // KV 8 bits en lean seulement la ou il y a plusieurs tetes KV (26B-A4B, 31B) :
        // E2B/E4B/12B n'en ont qu'une, le gain y est negligeable (BENCHMARKS.md §3).
        let multiKVHeads = family == .a4b || family == .b31b
        let available = availableMemoryMB()
        let lean = kind == .lean
        // 16bit-lean garde les caches Mac : des limites serrees font thrasher un
        // working set bf16 (YuE2 : +73 % de temps).
        let macCaches = !lean || bits == .sixteen
        var notes: [String] = []
        if family == .b12b && bits == .four {
            notes.append("12B 4 bits : qualite degradee (MMLU 37 % contre 57 % en bf16, 100 questions) ; preferer 8 bits.")
        }
        if family == .a4b || family == .b31b { notes.append("Pas d'audio.") }
        notes.append(lean ? "Econome : limites memoire adaptees a la machine." : "Rapide : tout resident.")
        notes.append("Valeurs initiales non mesurees.")
        return Gemma4ReferenceProfile(
            family: family, bits: bits, kind: kind, model: model,
            kvBits: lean && multiKVHeads ? 8 : nil,
            prefillStepSize: lean ? 256 : 512,
            cacheLimitMB: macCaches ? 4096 : min(1024, max(256, available / 6)),
            memoryLimitMB: macCaches ? nil : max(4096, available - 2048),
            clearCacheAfterAnswer: lean,
            multimodal: true,
            summary: notes.joined(separator: " "))
    }
}
