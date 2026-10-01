// Branchement du beacon (SiliconScope) sur les operations lourdes de Gemma 4.

import Foundation
import MLXLMCommon

extension RuntimeBeacon {
    /// Nom de modele a publier : dernier composant de l'identifiant ou du dossier
    /// (`gemma-4-e2b-it-4bit`), jamais un chemin complet.
    static func modelName(_ configuration: ModelConfiguration) -> String {
        modelName(configuration.name)
    }

    static func modelName(_ identifier: String) -> String {
        URL(fileURLWithPath: identifier).lastPathComponent
    }
}
