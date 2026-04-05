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

    /// Enregistre les types "gemma4_text" et "gemma4" dans LLMTypeRegistry.shared
    /// - Parameter multimodal: si true, charge le modele multimodal complet (vision+audio)
    public static func register(multimodal: Bool = false) async {
        await LLMTypeRegistry.shared.registerModelType("gemma4_text") { configData in
            let fullConfig = try JSONDecoder().decode(Gemma4Config.self, from: configData)
            return Gemma4LLMModel(config: fullConfig.textConfig)
        }

        await LLMTypeRegistry.shared.registerModelType("gemma4") { configData in
            let fullConfig = try JSONDecoder().decode(Gemma4Config.self, from: configData)
            if multimodal {
                return Gemma4MultimodalLLMModel(config: fullConfig)
            } else {
                return Gemma4LLMModel(config: fullConfig.textConfig)
            }
        }
    }
}
