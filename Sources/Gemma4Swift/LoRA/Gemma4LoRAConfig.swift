// Configuration LoRA avec defaults adaptes par famille de modele Gemma 4

import Foundation
import MLXLMCommon

/// Defaults LoRA par famille de modele Gemma 4
public enum Gemma4LoRADefaults {

    /// Famille de modele Gemma 4
    public enum ModelFamily: String, CaseIterable, Sendable {
        case e2b       // 2.3B effectif, 35 couches
        case e4b       // 4.5B effectif, 42 couches
        case b12b      // 12B unified, 48 couches
        case a4b       // 26B-A4B MoE, 30 couches
        case dense31b  // 31B dense, 60 couches

        /// Nombre total de couches decoder (`text_config.num_hidden_layers`).
        public var totalLayers: Int {
            switch self {
            case .e2b: return 35
            case .e4b: return 42
            case .b12b: return 48
            case .a4b: return 30
            case .dense31b: return 60
            }
        }

        /// Nombre de couches LoRA par defaut (suffixe du modele)
        public var defaultNumLayers: Int {
            switch self {
            case .e2b: return 8
            case .e4b: return 12
            case .b12b: return 16
            case .dense31b: return 16
            case .a4b: return 10
            }
        }

        /// Detecte la famille a partir d'un ID de modele HuggingFace (repli : E2B).
        /// Preferer `from(directory:)`, qui lit `config.json`.
        public static func from(modelId: String) -> ModelFamily {
            let id = modelId.lowercased()
            if id.contains("e2b") { return .e2b }
            if id.contains("e4b") { return .e4b }
            if id.contains("12b") { return .b12b }
            if id.contains("26b") || id.contains("a4b") { return .a4b }
            if id.contains("31b") { return .dense31b }
            // Default E2B pour les modeles inconnus
            return .e2b
        }

        /// Famille lue dans `config.json` (A-06 : le chemin ne dit rien d'un dossier renomme,
        /// et tout modele inconnu prenait les defauts d'E2B). `nil` si non reconnue.
        public static func from(directory: URL) -> ModelFamily? {
            guard let data = try? Data(contentsOf: directory.appendingPathComponent("config.json")),
                  let config = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            else { return nil }
            let text = config["text_config"] as? [String: Any] ?? config
            let modelType = (config["model_type"] as? String ?? "").lowercased()
            let layers = text["num_hidden_layers"] as? Int ?? 0
            if modelType.hasPrefix("gemma4_unified") { return .b12b }
            if text["enable_moe_block"] as? Bool == true { return .a4b }
            return allCases.first { $0 != .a4b && $0 != .b12b && $0.totalLayers == layers }
        }
    }

    /// Cree une configuration LoRA adaptee pour Gemma 4
    ///
    /// - Parameters:
    ///   - family: famille de modele (ou nil pour utiliser les defaults generaux)
    ///   - rank: rang LoRA (default: 8)
    ///   - scale: facteur d'echelle LoRA alpha (default: 20.0)
    ///   - numLayers: nombre de couches a adapter (nil = default par famille)
    ///   - keys: cles des Linear layers a cibler (nil = toutes les Linear, via default du protocol)
    /// - Returns: LoRAConfiguration prete a l'emploi
    public static func configuration(
        for family: ModelFamily = .e2b,
        rank: Int = 8,
        scale: Float = 20.0,
        numLayers: Int? = nil,
        keys: [String]? = nil,
        useDora: Bool = false
    ) -> LoRAConfiguration {
        LoRAConfiguration(
            numLayers: numLayers ?? family.defaultNumLayers,
            fineTuneType: useDora ? .dora : .lora,
            loraParameters: .init(rank: rank, scale: scale, keys: keys)
        )
    }
}
