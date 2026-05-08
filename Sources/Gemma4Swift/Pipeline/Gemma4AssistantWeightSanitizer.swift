// Sanitizer dedie aux poids du modele Assistant (drafter MTP)
//
// Structure des poids dans le checkpoint Google:
//   model.embed_tokens.weight
//   model.layers.{0..3}.{self_attn,mlp,...}.weight
//   model.norm.weight
//   pre_projection.weight
//   post_projection.weight
//   masked_embedding.centroids.weight
//   masked_embedding.token_ordering              (buffer, doit etre Int32)
//   lm_head.weight                               (optionnel, skip si tie_word_embeddings)
//
// La structure Swift de Gemma4AssistantDraftModel est concue pour matcher 1:1
// ces cles, donc aucun renaming n'est necessaire.

import Foundation
import MLX

public enum Gemma4AssistantWeightSanitizer {

    /// Nettoie les poids charges depuis le checkpoint du drafter Assistant.
    /// - Parameters:
    ///   - weights: dictionnaire brut des poids (cles au format PyTorch).
    ///   - tieWordEmbeddings: si true, skip `lm_head.weight` (les embeddings sont reutilisees).
    public static func sanitize(
        weights: [String: MLXArray],
        tieWordEmbeddings: Bool
    ) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]

        for (key, value) in weights {
            // Skip rotary embeddings pre-calculees (defensif: pas attendues dans les checkpoints
            // mais on s'aligne sur le sanitizer principal)
            if key.contains("rotary_emb") { continue }
            if key.contains(".rope.") && key.hasSuffix(".freqs") { continue }

            // Skip lm_head si word embeddings tied (le MaskedEmbedder ou les embeddings
            // partagees produisent les logits)
            if tieWordEmbeddings && key == "lm_head.weight" { continue }

            // Convertir le buffer token_ordering en Int32 — c'est un index, pas un float
            if key == "masked_embedding.token_ordering" {
                out[key] = value.asType(.int32)
                continue
            }

            out[key] = value
        }

        return out
    }
}
