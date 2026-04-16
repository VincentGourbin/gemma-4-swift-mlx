// Chargement et gestion des adapters LoRA pour l'inference

import Foundation
import MLX
import MLXLMCommon

/// Utilitaires pour charger/fusionner/retirer des adapters LoRA sur un modele Gemma 4
public enum Gemma4LoRAInference {

    /// Charge un adapter LoRA depuis un repertoire et l'applique au modele
    ///
    /// Le repertoire doit contenir:
    /// - `adapter_config.json` — configuration LoRA
    /// - `adapters.safetensors` — poids de l'adapter
    ///
    /// - Parameters:
    ///   - container: le ModelContainer contenant le modele de base
    ///   - directory: repertoire contenant les fichiers de l'adapter
    public static func loadAdapter(
        into container: ModelContainer,
        from directory: URL
    ) async throws {
        let adapter = try LoRAContainer.from(directory: directory)
        try await container.perform { context in
            guard let model = context.model as? LanguageModel else {
                throw Gemma4LoRAError.incompatibleModel
            }
            try adapter.load(into: model)
        }
    }

    /// Fusionne definitivement un adapter LoRA dans les poids du modele
    ///
    /// Apres fusion, l'adapter n'est plus necessaire et le modele peut etre
    /// utilise normalement avec des performances d'inference identiques au modele de base.
    ///
    /// - Parameters:
    ///   - container: le ModelContainer contenant le modele
    ///   - directory: repertoire contenant les fichiers de l'adapter
    public static func fuseAdapter(
        into container: ModelContainer,
        from directory: URL
    ) async throws {
        let adapter = try LoRAContainer.from(directory: directory)
        try await container.perform { context in
            guard let model = context.model as? LanguageModel else {
                throw Gemma4LoRAError.incompatibleModel
            }
            try adapter.fuse(with: model)
        }
    }

    /// Retire un adapter LoRA et restaure les poids de base du modele
    ///
    /// - Parameters:
    ///   - container: le ModelContainer contenant le modele avec adapter
    ///   - directory: repertoire contenant la config de l'adapter (pour connaitre les layers)
    public static func unloadAdapter(
        from container: ModelContainer,
        directory: URL
    ) async throws {
        let adapter = try LoRAContainer.from(directory: directory)
        await container.perform { context in
            guard let model = context.model as? LanguageModel else { return }
            adapter.unload(from: model)
        }
    }
}

public enum Gemma4LoRAError: LocalizedError {
    case incompatibleModel
    case trainingFailed(String)
    case adapterNotFound(URL)

    public var errorDescription: String? {
        switch self {
        case .incompatibleModel:
            return "Le modele n'est pas compatible avec LoRA (doit conformer a LanguageModel + LoRAModel)"
        case .trainingFailed(let reason):
            return "Echec de l'entrainement: \(reason)"
        case .adapterNotFound(let url):
            return "Adapter introuvable a \(url.path())"
        }
    }
}
