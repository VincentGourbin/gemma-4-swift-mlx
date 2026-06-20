// Configuration centralisee des optimisations memoire pour DiffusionGemma.
//
// Inspire de LTX `MemoryOptimizationConfig` qui regroupe les presets par RAM.
// Les optimisations sont :
//   - mixedPrecision        : Q-DiT/ViDiT-Q (premiers/derniers layers high-prec)
//   - unloadVisionAfterUse  : vision_tower retire apres le 1er forward
//   - clearCacheBetweenCanvases : Memory.clearCache() entre canvas

import Foundation

/// Preset d'optimisations memoire pour DiffusionGemma.
public struct DiffusionMemoryConfig: Sendable {

    /// Configuration mixed precision a appliquer (nil = bf16 pur).
    public var mixedPrecision: DiffusionOnTheFlyQuantization.MixedPrecisionConfig?

    /// Decharger `vision_tower` + `embed_vision` apres le 1er canvas.
    /// (Les soft-tokens sont deja dans le KV cache.)
    public var unloadVisionAfterFirstCanvas: Bool

    /// Appeler `MLX.Memory.clearCache()` entre chaque canvas.
    public var clearCacheBetweenCanvases: Bool

    public init(
        mixedPrecision: DiffusionOnTheFlyQuantization.MixedPrecisionConfig? = nil,
        unloadVisionAfterFirstCanvas: Bool = true,
        clearCacheBetweenCanvases: Bool = true
    ) {
        self.mixedPrecision = mixedPrecision
        self.unloadVisionAfterFirstCanvas = unloadVisionAfterFirstCanvas
        self.clearCacheBetweenCanvases = clearCacheBetweenCanvases
    }

    // MARK: - Presets

    /// Aucune optimisation : bf16 partout, vision_tower garde, pas de clearCache.
    /// Recommande pour Mac Studio 128+ GB.
    public static let disabled = DiffusionMemoryConfig(
        mixedPrecision: nil,
        unloadVisionAfterFirstCanvas: false,
        clearCacheBetweenCanvases: false
    )

    /// Light : unload vision + clearCache, pas de quantization.
    /// Cible Mac 96 GB.
    public static let light = DiffusionMemoryConfig(
        mixedPrecision: nil,
        unloadVisionAfterFirstCanvas: true,
        clearCacheBetweenCanvases: true
    )

    /// Moderate : mixed precision conservative + unload + clearCache.
    /// Cible Mac 64 GB.
    public static let moderate = DiffusionMemoryConfig(
        mixedPrecision: .conservative,
        unloadVisionAfterFirstCanvas: true,
        clearCacheBetweenCanvases: true
    )

    /// Aggressive : mixed precision default + unload + clearCache.
    /// Cible Mac 32-48 GB.
    public static let aggressive = DiffusionMemoryConfig(
        mixedPrecision: .default,
        unloadVisionAfterFirstCanvas: true,
        clearCacheBetweenCanvases: true
    )

    /// Extreme : mixed precision aggressive + unload + clearCache.
    /// Cible Mac 16-24 GB (peut deborder).
    public static let extreme = DiffusionMemoryConfig(
        mixedPrecision: .aggressive,
        unloadVisionAfterFirstCanvas: true,
        clearCacheBetweenCanvases: true
    )

    /// Defaut : light.
    public static let `default` = DiffusionMemoryConfig.light

    /// Auto-select preset par RAM systeme (en GB).
    public static func recommended(forRAMGB ram: Int) -> DiffusionMemoryConfig {
        switch ram {
        case ...24: return .extreme
        case 25...48: return .aggressive
        case 49...80: return .moderate
        case 81...112: return .light
        default: return .disabled
        }
    }
}
