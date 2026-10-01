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
    /// Dossier du checkpoint : permet de recharger la vision apres un dechargement.
    public let modelDirectory: URL?

    public init(
        model: DiffusionGemmaForBlockDiffusion,
        config: DiffusionGemmaConfig,
        generationConfig: DiffusionGenerationConfig,
        tokenizer: Tokenizer,
        memoryConfig: DiffusionMemoryConfig,
        modelDirectory: URL? = nil
    ) {
        self.model = model
        self.config = config
        self.generationConfig = generationConfig
        self.tokenizer = tokenizer
        self.memoryConfig = memoryConfig
        self.modelDirectory = modelDirectory
    }

    /// Cree un pipeline pret a generer depuis ce container.
    public func makePipeline() -> DiffusionGemmaPipeline {
        DiffusionGemmaPipeline(
            model: model, genConfig: generationConfig, memoryConfig: memoryConfig, modelDirectory: modelDirectory)
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
        let beacon = RuntimeBeacon.begin(task: "load-models", model: directory.lastPathComponent)
        defer { beacon?.end() }
        beacon?.update(phase: "loading-weights")

        // 1) Modele + config
        let (model, config): (DiffusionGemmaForBlockDiffusion, DiffusionGemmaConfig)
        do {
            (model, config) = try DiffusionGemmaLoader.load(from: directory, includeVision: includeVision)
        } catch {
            throw LoadError.configLoadFailed(error)
        }

        // 2) Mixed precision si demandee (un pack est deja quantifie)
        if let mp = memoryConfig.mixedPrecision, !DiffusionPrequantizedPack.isPack(directory) {
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
            memoryConfig: memoryConfig,
            modelDirectory: directory
        )
    }

    /// Charge le bf16 officiel et applique un profil de reference : quantification a la
    /// volee (experts compris, routeur 8 bits, vision bf16), vision, politique memoire.
    /// Additif : `load(from:memoryConfig:includeVision:)` reste.
    public static func load(
        from directory: URL,
        profile: DiffusionReferenceProfile
    ) async throws -> DiffusionGemmaContainer {
        // Pack pre-quantifie : sa quantification doit etre celle du profil.
        if DiffusionPrequantizedPack.isPack(directory) {
            let manifest = try DiffusionPrequantizedPack.readManifest(directory)
            guard manifest.quantization == profile.quantization.signature else {
                throw DiffusionPrequantizedPack.PackError.quantizationMismatch(
                    pack: manifest.quantization, profile: profile.quantization.signature)
            }
            let container = try await load(
                from: directory, memoryConfig: profile.memoryConfig, includeVision: profile.includeVision)
            profile.applyGlobalPolicy()
            return container
        }
        let container = try await load(
            from: directory, memoryConfig: profile.memoryConfig, includeVision: profile.includeVision)
        switch profile.quantization {
        case .none:
            break
        case .uniform(let bits, let groupSize):
            DiffusionOnTheFlyQuantization.apply(
                to: container.model, bits: bits, groupSize: groupSize,
                excludedPathPrefixes: DiffusionOnTheFlyQuantization.multimodalEncoderPrefixes)
        case .mixed(let config):
            DiffusionOnTheFlyQuantization.applyMixedPrecision(to: container.model, config: config)
        }
        profile.applyGlobalPolicy()
        return container
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
