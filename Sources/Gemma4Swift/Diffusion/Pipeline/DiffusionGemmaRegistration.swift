// Registration pour DiffusionGemma — chargement natif via API publique.
//
// DiffusionGemma N'EST PAS un LanguageModel (block-AR diffusion != AR
// token-by-token), donc ne s'integre pas directement dans LLMTypeRegistry.
//
// Cette enum expose une API equivalente a Gemma4Registration mais dediee :
//   await DiffusionGemmaRegistration.load(modelId:) -> DiffusionGemmaContainer
//
// Le container packagine modele + tokenizer + config gen + memory config.

import Foundation
import MLX
import MLXNN
import Tokenizers

/// Container DiffusionGemma : tout ce qu'il faut pour lancer une generation.
public struct DiffusionGemmaContainer: @unchecked Sendable {
    public let model: DiffusionGemmaForBlockDiffusion
    public let config: DiffusionGemmaConfig
    public let generationConfig: DiffusionGenerationConfig
    public let tokenizer: Tokenizer
    public let memoryConfig: DiffusionMemoryConfig

    public init(
        model: DiffusionGemmaForBlockDiffusion,
        config: DiffusionGemmaConfig,
        generationConfig: DiffusionGenerationConfig,
        tokenizer: Tokenizer,
        memoryConfig: DiffusionMemoryConfig
    ) {
        self.model = model
        self.config = config
        self.generationConfig = generationConfig
        self.tokenizer = tokenizer
        self.memoryConfig = memoryConfig
    }

    /// Cree un pipeline pret a generer depuis ce container.
    public func makePipeline() -> DiffusionGemmaPipeline {
        DiffusionGemmaPipeline(model: model, genConfig: generationConfig)
    }
}

/// Registration / loading helper for DiffusionGemma.
public enum DiffusionGemmaRegistration {

    /// Erreurs de chargement.
    public enum LoadError: LocalizedError {
        case directoryNotFound(URL)
        case tokenizerLoadFailed(Error)
        case configLoadFailed(Error)

        public var errorDescription: String? {
            switch self {
            case .directoryNotFound(let url):
                return "Repertoire modele introuvable : \(url.path)"
            case .tokenizerLoadFailed(let e):
                return "Echec chargement tokenizer : \(e.localizedDescription)"
            case .configLoadFailed(let e):
                return "Echec chargement config : \(e.localizedDescription)"
            }
        }
    }

    /// Charge un DiffusionGemma depuis un repertoire local en appliquant
    /// automatiquement les optimisations memoire selon le preset.
    ///
    /// - Parameters:
    ///   - directory : repertoire contenant config.json + safetensors + tokenizer.json
    ///   - memoryConfig : preset d'optimisation (defaut : auto par RAM systeme)
    ///   - includeVision : charger vision_tower + embed_vision (defaut : true)
    /// - Returns: container pret a generer
    public static func load(
        from directory: URL,
        memoryConfig: DiffusionMemoryConfig = .recommended(forRAMGB: systemRAMGB),
        includeVision: Bool = true
    ) async throws -> DiffusionGemmaContainer {
        guard FileManager.default.fileExists(atPath: directory.path) else {
            throw LoadError.directoryNotFound(directory)
        }

        // 1) Modele + config
        let (model, config): (DiffusionGemmaForBlockDiffusion, DiffusionGemmaConfig)
        do {
            (model, config) = try DiffusionGemmaLoader.load(from: directory, includeVision: includeVision)
        } catch {
            throw LoadError.configLoadFailed(error)
        }

        // 2) Mixed precision si demandee
        if let mp = memoryConfig.mixedPrecision {
            _ = DiffusionOnTheFlyQuantization.applyMixedPrecision(to: model, config: mp)
        }

        // 3) Generation config (avec fallback)
        let genConfig: DiffusionGenerationConfig
        let genConfigURL = directory.appendingPathComponent("generation_config.json")
        if FileManager.default.fileExists(atPath: genConfigURL.path),
           let data = try? Data(contentsOf: genConfigURL),
           let parsed = try? JSONDecoder().decode(DiffusionGenerationConfig.self, from: data)
        {
            genConfig = parsed
        } else {
            genConfig = DiffusionGenerationConfig()
        }

        // 4) Tokenizer
        let tokenizer: Tokenizer
        do {
            tokenizer = try await AutoTokenizer.from(modelFolder: directory)
        } catch {
            throw LoadError.tokenizerLoadFailed(error)
        }

        return DiffusionGemmaContainer(
            model: model,
            config: config,
            generationConfig: genConfig,
            tokenizer: tokenizer,
            memoryConfig: memoryConfig
        )
    }

    /// RAM systeme en GB pour auto-selection du preset.
    public static var systemRAMGB: Int {
        Int(ProcessInfo.processInfo.physicalMemory / (1024 * 1024 * 1024))
    }

    /// Charge depuis l'ID HF (utilise le cache Gemma4ModelCache).
    /// - Parameters:
    ///   - modelId : ex. "google/diffusiongemma-26B-A4B-it"
    ///   - memoryConfig : preset (auto par defaut)
    ///   - includeVision : charger vision (defaut : true)
    public static func load(
        modelId: String,
        memoryConfig: DiffusionMemoryConfig = .recommended(forRAMGB: systemRAMGB),
        includeVision: Bool = true
    ) async throws -> DiffusionGemmaContainer {
        var dir = Gemma4ModelCache.modelsDirectory
        for part in modelId.split(separator: "/") {
            dir = dir.appendingPathComponent(String(part))
        }
        return try await load(from: dir, memoryConfig: memoryConfig, includeVision: includeVision)
    }
}
