// Enregistrement du model type "gemma4_text" dans LLMTypeRegistry

import Foundation
import MLXLMCommon
import MLXLLM

/// Enregistre Gemma 4 dans le registre de types de modeles de mlx-swift-lm.
/// Doit etre appele AVANT tout chargement de modele Gemma 4.
///
/// Usage:
/// ```swift
/// let container = try await Gemma4Registration.loadContainer(from: modelDirectory)
/// ```
///
/// N'appelez `register()` a la main que si vous chargez vous-meme. Dans ce cas,
/// passez par `MLXLLM.LLMModelFactory.shared` et **pas** par la fonction libre
/// `MLXLMCommon.loadModelContainer(...)` : voir `loadContainer(from:using:multimodal:)`
/// ci-dessous pour le pourquoi.
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

    /// Enregistre les types Gemma 4 puis charge le modele **en forcant
    /// `LLMModelFactory`**.
    ///
    /// A preferer systematiquement a la fonction libre
    /// `MLXLMCommon.loadModelContainer(from:using:)`.
    ///
    /// Pourquoi : la fonction libre passe par `ModelFactoryRegistry.shared`, qui
    /// essaie `MLXVLM.VLMModelFactory` **avant** `MLXLLM.LLMModelFactory` et retient
    /// la premiere fabrique qui reussit. Or `register(multimodal:)` ne patche que
    /// `LLMTypeRegistry.shared` ; l'amont publie ses propres entrees `"gemma4"` et
    /// `"gemma4_unified"` dans `VLMTypeRegistry.shared`, hors de notre portee (le
    /// paquet ne lie pas `MLXVLM`). Des qu'un autre module du meme processus lie
    /// `MLXVLM`, la fabrique VLM gagne la course et renvoie `MLXVLM.Gemma4` au lieu
    /// de `Gemma4MultimodalLLMModel` — le `as?` de `chatStreamMultimodal` echoue
    /// alors avec `unsupportedModelFamily`, independamment du `multimodal:` demande.
    ///
    /// Court-circuiter `ModelFactoryRegistry` supprime la course a la source et
    /// reste chirurgical : on ne touche pas aux entrees VLM de l'amont, donc une app
    /// qui veut deliberement `MLXVLM.Gemma4` le garde.
    ///
    /// - Parameters:
    ///   - directory: repertoire contenant `config.json`, les safetensors et le tokenizer
    ///   - tokenizerLoader: chargeur de tokenizer (defaut : `Gemma4TokenizerLoader`)
    ///   - multimodal: si true, `"gemma4"` / `"gemma4_unified"` instancient le
    ///     wrapper multimodal (vision+audio) ; sinon le path text-only.
    public static func loadContainer(
        from directory: URL,
        using tokenizerLoader: any TokenizerLoader = Gemma4TokenizerLoader(),
        multimodal: Bool = true
    ) async throws -> ModelContainer {
        await register(multimodal: multimodal)
        return try await LLMModelFactory.shared.loadContainer(
            from: directory, using: tokenizerLoader)
    }

}
