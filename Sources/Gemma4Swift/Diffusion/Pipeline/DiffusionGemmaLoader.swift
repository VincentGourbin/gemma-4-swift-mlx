// Loader autonome pour DiffusionGemma.
//
// Phase 4 minimal :
//   1. Lit config.json depuis le repertoire local
//   2. Instancie DiffusionGemmaForBlockDiffusion(config)
//   3. Charge tous les *.safetensors
//   4. Applique DiffusionWeightSanitizer (decoder duplique vers encoder)
//   5. update(parameters:) sur le modele
//
// Pas de quantization ni de tokenizer cote loader pour Phase 4 — ces volets
// arriveront en Phase 5+ (quantization a la volee + Gemma4TokenizerLoader).

import Foundation
import MLX
import MLXNN

public enum DiffusionGemmaLoaderError: LocalizedError {
    case configNotFound(URL)
    case noSafetensorsFound(URL)
    case configDecodingFailed(String)

    public var errorDescription: String? {
        switch self {
        case .configNotFound(let url): return "config.json introuvable a \(url.path)"
        case .noSafetensorsFound(let url): return "Aucun fichier *.safetensors a \(url.path)"
        case .configDecodingFailed(let msg): return "Decoding config.json : \(msg)"
        }
    }
}

/// Loader autonome pour DiffusionGemma.
public enum DiffusionGemmaLoader {
    /// Charge un DiffusionGemma depuis un repertoire local.
    /// - Parameters:
    ///   - directory : repertoire contenant config.json + *.safetensors
    ///   - includeVision : si true, garde les poids vision (Phase 5+).
    /// - Returns: modele pret a l'inference (eval applique).
    public static func load(
        from directory: URL,
        includeVision: Bool = false
    ) throws -> (model: DiffusionGemmaForBlockDiffusion, config: DiffusionGemmaConfig) {
        let config = try loadConfig(from: directory)
        let model = DiffusionGemmaForBlockDiffusion(config)
        try loadWeights(into: model, from: directory, includeVision: includeVision)
        eval(model)
        return (model, config)
    }

    /// Lit config.json et le decode.
    public static func loadConfig(from directory: URL) throws -> DiffusionGemmaConfig {
        let configURL = directory.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw DiffusionGemmaLoaderError.configNotFound(configURL)
        }

        let data = try Data(contentsOf: configURL)
        do {
            return try JSONDecoder().decode(DiffusionGemmaConfig.self, from: data)
        } catch {
            throw DiffusionGemmaLoaderError.configDecodingFailed(String(describing: error))
        }
    }

    /// Charge les poids depuis le repertoire et les applique au modele.
    public static func loadWeights(
        into model: DiffusionGemmaForBlockDiffusion,
        from directory: URL,
        includeVision: Bool = false
    ) throws {
        // Collect tous les *.safetensors
        var weights = [String: MLXArray]()
        let enumerator = FileManager.default.enumerator(
            at: directory,
            includingPropertiesForKeys: nil
        )

        var fileCount = 0
        if let enumerator = enumerator {
            for case let url as URL in enumerator {
                if url.pathExtension == "safetensors" {
                    let (w, _) = try loadArraysAndMetadata(url: url)
                    for (key, value) in w {
                        weights[key] = value
                    }
                    fileCount += 1
                }
            }
        }

        guard fileCount > 0 else {
            throw DiffusionGemmaLoaderError.noSafetensorsFound(directory)
        }

        // Sanitize + remapping
        let sanitized = DiffusionWeightSanitizer.sanitize(weights, includeVision: includeVision)

        // Apply
        let parameters = ModuleParameters.unflattened(sanitized)
        try model.update(parameters: parameters, verify: [.all])
    }
}
