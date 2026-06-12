// Port de WeightSanitizer pour DiffusionGemma
//
// Le checkpoint utilise des cles `model.decoder.X`, `model.encoder.X`, `model.encoder.vision_tower.X`.
// On doit les remapper vers les modules Swift correspondants.
//
// Observation cle (cf. safetensors index) : l'encoder language model NE CONTIENT QUE
// `layer_scalar` par couche. Cela CONFIRME le weight sharing :
//   - les linear de layers sont *physiquement partages* entre encoder et decoder
//   - seul `layer_scalar` (1 scalaire par couche) differe entre les deux modes
//
// Strategy :
//   - Charger les poids decoder dans Gemma4DecoderLayer standard
//   - Charger les `layer_scalar` de l'encoder dans un buffer separe
//   - Au forward encoder, swap le layer_scalar pour appliquer la bonne valeur

import Foundation
import MLX

/// Sanitizer de poids PyTorch -> module hierarchy Swift pour DiffusionGemma.
/// STUB Phase 2 : la table de remapping sera completee en Phase 3.
public enum DiffusionWeightSanitizer {
    /// Remappe et filtre les cles de poids depuis le checkpoint PyTorch.
    /// - Parameter weights : dict cle Python -> MLXArray.
    /// - Returns: dict cle Swift -> MLXArray.
    public static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        for (key, value) in weights {
            var newKey = key
            // TODO Phase 3:
            //   - "model.decoder.layers.N.X" -> "decoder.layers.N.X"
            //   - "model.decoder.self_conditioning.Y" -> "self_conditioning.Y"
            //   - "model.encoder.vision_tower.X" -> "vision_tower.X"
            //   - "model.encoder.language_model.layers.N.layer_scalar" -> stocker dans encoder_layer_scalars[N]
            //   - "model.decoder.embed_tokens" -> partage avec lm_head si tie_word_embeddings
            //   - "model.decoder.norm" -> "decoder.norm"
            //   - split MoE gate_up_proj si necessaire (deja gere ailleurs)
            newKey = newKey.replacingOccurrences(of: "model.", with: "", options: .anchored)
            result[newKey] = value
        }
        return result
    }
}
