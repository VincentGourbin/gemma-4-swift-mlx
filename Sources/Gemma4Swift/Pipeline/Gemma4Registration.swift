// Enregistrement du model type "gemma4_text" dans LLMTypeRegistry

import Foundation
import MLXLMCommon
import MLXLLM

/// Enregistre Gemma 4 dans le registre de types de modeles de mlx-swift-lm.
/// Doit etre appele AVANT tout chargement de modele Gemma 4.
///
/// Usage:
/// ```swift
/// await Gemma4Registration.register()
/// // Maintenant MLXLMCommon.loadModelContainer(id: "mlx-community/gemma-4-e2b-it-4bit") fonctionne
/// ```
public enum Gemma4Registration {

    /// Enregistre les types "gemma4_text", "gemma4", "gemma4_unified_text" et
    /// "gemma4_unified" dans LLMTypeRegistry.shared.
    ///
    /// `gemma4_unified*` correspond a la variante 12B (gemma-4-12B-it) qui
    /// reutilise la meme architecture decoder. Le path text-only est
    /// supporte out-of-the-box ; le path multimodal du 12B n'est pas encore
    /// branche (schema vision/audio different) et tombe en text-only.
    /// - Parameter multimodal: si true, charge le modele multimodal complet (vision+audio)
    ///   uniquement pour les variantes E2B/E4B (`gemma4`).
    public static func register(multimodal: Bool = false) async {
        let textFactory: @Sendable (Data) throws -> any LanguageModel = { configData in
            let fullConfig = try JSONDecoder().decode(Gemma4Config.self, from: configData)
            return Gemma4LLMModel(config: fullConfig.textConfig)
        }

        await LLMTypeRegistry.shared.registerModelType("gemma4_text", creator: textFactory)
        await LLMTypeRegistry.shared.registerModelType("gemma4_unified_text", creator: textFactory)

        await LLMTypeRegistry.shared.registerModelType("gemma4") { configData in
            let fullConfig = try JSONDecoder().decode(Gemma4Config.self, from: configData)
            if multimodal {
                return Gemma4MultimodalLLMModel(config: fullConfig)
            } else {
                return Gemma4LLMModel(config: fullConfig.textConfig)
            }
        }

        // gemma4_unified : utilise le wrapper multimodal dedie en mode multimodal,
        // sinon path text-only.
        await LLMTypeRegistry.shared.registerModelType("gemma4_unified") { configData in
            if multimodal {
                let unifiedConfig = try JSONDecoder().decode(Gemma4UnifiedConfig.self, from: configData)
                return Gemma4UnifiedMultimodalLLMModel(config: unifiedConfig)
            } else {
                let fullConfig = try JSONDecoder().decode(Gemma4Config.self, from: configData)
                return Gemma4LLMModel(config: fullConfig.textConfig)
            }
        }
    }
}
