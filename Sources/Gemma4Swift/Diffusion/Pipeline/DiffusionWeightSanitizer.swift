// Sanitizer de poids PyTorch -> module hierarchy Swift pour DiffusionGemma.
//
// Layout dans le checkpoint :
//   model.encoder.language_model.embed_tokens.weight    [tied -> decoder]
//   model.encoder.language_model.layers.N.layer_scalar  [SPECIFIC encoder]
//   model.encoder.language_model.norm.weight            [tied -> decoder]
//   model.encoder.vision_tower.*                        [skip Phase 3]
//   model.encoder.embed_vision.*                        [skip Phase 3]
//   model.decoder.embed_tokens.weight                   [tied: pas un duplicate, c'est LA source]
//   model.decoder.layers.N.{q_proj, k_proj, ..., layer_scalar}
//   model.decoder.layers.N.{experts.gate_up_proj, experts.down_proj}  [MoE -> split]
//   model.decoder.self_conditioning.{pre_norm, post_norm, gate_proj, up_proj, down_proj}
//   model.decoder.norm.weight
//   lm_head.weight                                      [skip : tied avec embed_tokens]
//
// Layout cible Swift (DiffusionGemmaForBlockDiffusion racine) :
//   encoder.embed_tokens.weight
//   encoder.layers.N.{X, layer_scalar specific encoder}
//   encoder.norm.weight
//   decoder.embed_tokens.weight
//   decoder.layers.N.{X, layer_scalar specific decoder}
//   decoder.self_conditioning.{*}
//   decoder.norm.weight
//
// Strategie :
//   1. Tous les poids "decoder.layers.N.X" sauf layer_scalar sont duplique vers encoder.layers.N.X
//   2. Les layer_scalar restent separes : decoder vs encoder.language_model
//   3. embed_tokens et norm : depuis decoder, duplique vers encoder
//   4. Vision skip (Phase 4+)
//   5. lm_head skip (tied)
//   6. Split MoE gate_up_proj reutilise la meme logique que WeightSanitizer general

import Foundation
import MLX

/// Sanitizer de poids PyTorch -> module hierarchy Swift pour DiffusionGemma.
public enum DiffusionWeightSanitizer {
    /// Remappe et filtre les cles de poids depuis le checkpoint PyTorch.
    /// - Parameters:
    ///   - weights : dict cle Python -> MLXArray.
    ///   - includeVision : si true, garde les cles vision_tower / embed_vision
    ///     (Phase 4+ ; pour Phase 3 on les saute).
    /// - Returns: dict cle Swift -> MLXArray, pret a etre charge dans
    ///   DiffusionGemmaForBlockDiffusion.update(parameters:).
    public static func sanitize(
        _ weights: [String: MLXArray],
        includeVision: Bool = false
    ) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]

        for (key, value) in weights {
            // Skip lm_head (tied avec decoder.embed_tokens)
            if key == "lm_head.weight" || key.hasSuffix(".lm_head.weight") {
                continue
            }

            // Skip rotary_emb pre-computed frequencies
            if key.contains("rotary_emb") { continue }

            // Skip vision (Phase 4+)
            if !includeVision {
                if key.contains("vision_tower") || key.contains("embed_vision") {
                    continue
                }
            }

            // Strip "model." prefix
            var stripped = key
            if stripped.hasPrefix("model.") {
                stripped = String(stripped.dropFirst("model.".count))
            }

            // Cas 1 : encoder.language_model.* -> encoder.*
            //   Seuls layer_scalar / embed_tokens / norm doivent etre maintenus
            //   pour encoder. layer_scalar est SPECIFIC encoder. Les autres
            //   (embed_tokens, norm) sont tied avec le decoder mais on les
            //   charge cote encoder aussi pour avoir une hierarchie complete.
            if stripped.hasPrefix("encoder.language_model.") {
                let rest = String(stripped.dropFirst("encoder.language_model.".count))
                let target = "encoder." + rest
                insert(into: &result, key: target, value: value)
                continue
            }

            // Cas 2 : decoder.* -> decoder.* + duplique vers encoder.* sauf layer_scalar
            if stripped.hasPrefix("decoder.") {
                let rest = String(stripped.dropFirst("decoder.".count))
                let decoderTarget = "decoder." + rest

                // self_conditioning : decoder-only, pas de duplication
                if rest.hasPrefix("self_conditioning.") {
                    insert(into: &result, key: decoderTarget, value: value)
                    continue
                }

                // layer_scalar : decoder-only, pas de duplication
                if rest.hasSuffix(".layer_scalar") {
                    insert(into: &result, key: decoderTarget, value: value)
                    continue
                }

                // Tous les autres poids (embed_tokens, norm, layers.N.{X != layer_scalar})
                // sont tied avec encoder. On insere les deux.
                insert(into: &result, key: decoderTarget, value: value)

                // Duplique vers encoder uniquement pour ce qui existe cote encoder :
                //  - embed_tokens.weight
                //  - norm.weight
                //  - layers.N.X
                let encoderTarget = "encoder." + rest
                insert(into: &result, key: encoderTarget, value: value)
                continue
            }

            // Cas 3 : tout le reste, on laisse passer apres strip "model."
            insert(into: &result, key: stripped, value: value)
        }

        return result
    }

    /// Insertion avec gestion du split MoE gate_up_proj.
    /// Reproduit la logique de [[WeightSanitizer]] mais en ciblant
    /// la cle apres remap.
    private static func insert(
        into result: inout [String: MLXArray],
        key: String,
        value: MLXArray
    ) {
        // MoE: experts.down_proj -> experts.switch_glu.down_proj.weight
        if key.hasSuffix(".experts.down_proj") {
            let newKey = key.replacingOccurrences(
                of: ".experts.down_proj",
                with: ".experts.switch_glu.down_proj.weight"
            )
            result[newKey] = value
            return
        }

        // MoE: experts.gate_up_proj -> split en gate_proj + up_proj
        if key.hasSuffix(".experts.gate_up_proj") {
            let gateKey = key.replacingOccurrences(
                of: ".experts.gate_up_proj",
                with: ".experts.switch_glu.gate_proj.weight"
            )
            let upKey = key.replacingOccurrences(
                of: ".experts.gate_up_proj",
                with: ".experts.switch_glu.up_proj.weight"
            )
            let swapped = value.swappedAxes(-1, -2)
            let midDim = swapped.shape.last! / 2
            result[gateKey] = swapped[.ellipsis, 0 ..< midDim].swappedAxes(-1, -2)
            result[upKey] = swapped[.ellipsis, midDim...].swappedAxes(-1, -2)
            return
        }

        result[key] = value
    }
}
