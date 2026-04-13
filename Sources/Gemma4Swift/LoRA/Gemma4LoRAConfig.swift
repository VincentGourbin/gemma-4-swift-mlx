// Configuration LoRA avec defaults adaptes par famille de modele Gemma 4

import Foundation
import MLXLMCommon

/// Defaults LoRA par famille de modele Gemma 4
public enum Gemma4LoRADefaults {

    /// Famille de modele Gemma 4
    public enum ModelFamily: String, CaseIterable, Sendable {
        case e2b    // 2.3B effectif, 26 layers
        case e4b    // 4.5B effectif, 34 layers
        case dense31b  // 31B dense, 46 layers
        case a4b    // 26B-A4B MoE, 26 layers

        /// Nombre total de couches decoder
        public var totalLayers: Int {
            switch self {
            case .e2b: return 35
            case .e4b: return 42
            case .dense31b: return 50
            case .a4b: return 34
            }
        }

        /// Nombre de couches LoRA par defaut (suffixe du modele)
        public var defaultNumLayers: Int {
            switch self {
            case .e2b: return 8
            case .e4b: return 12
            case .dense31b: return 16
            case .a4b: return 10
            }
        }

        /// Detecte la famille a partir d'un ID de modele HuggingFace
        public static func from(modelId: String) -> ModelFamily {
            let id = modelId.lowercased()
            if id.contains("e2b") { return .e2b }
            if id.contains("e4b") { return .e4b }
            if id.contains("26b") || id.contains("a4b") { return .a4b }
            if id.contains("31b") { return .dense31b }
            // Default E2B pour les modeles inconnus
            return .e2b
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
        keys: [String]? = nil
    ) -> LoRAConfiguration {
        LoRAConfiguration(
            numLayers: numLayers ?? family.defaultNumLayers,
            fineTuneType: .lora,
            loraParameters: .init(rank: rank, scale: scale, keys: keys)
        )
    }
}
