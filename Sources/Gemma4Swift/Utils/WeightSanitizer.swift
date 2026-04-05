// Port de gemma4.py sanitize() — Nettoyage et remapping des poids

import Foundation
import MLX

/// Sanitize les poids du modele pour correspondre a la structure Swift
public enum WeightSanitizer {

    /// Detecte si les poids viennent du format Google BF16 (vs MLX community pre-converti)
    /// Les poids Google ont le prefixe "model." sur les cles
    public static func isGoogleFormat(_ weights: [String: MLXArray]) -> Bool {
        weights.keys.contains { $0.hasPrefix("model.") }
    }

    /// Nettoie les poids charges depuis les safetensors
    /// - Supprime les prefixes "model."
    /// - Remappe "language_model.X" → "language_model.model.X"
    /// - Ignore les cles rotary_emb
    /// - Transpose les poids Conv2d/Conv1d PyTorch → MLX (format Google uniquement)
    /// - Split les poids MoE gate_up_proj
    public static func sanitize(
        weights: [String: MLXArray],
        hasVision: Bool = false,
        hasAudio: Bool = false,
        useClippedLinears: Bool = false
    ) -> [String: MLXArray] {
        let isGoogle = isGoogleFormat(weights)
        var sanitized: [String: MLXArray] = [:]

        for (key, value) in weights {
            // Skip clipping params si non utilises
            if key.contains("input_max") || key.contains("input_min")
                || key.contains("output_max") || key.contains("output_min") {
                if key.contains("vision_tower") && !useClippedLinears { continue }
                if !key.contains("vision_tower") && !key.contains("audio_tower") { continue }
            }

            // Skip rotary embeddings (pre-computed frequencies)
            if key.contains("rotary_emb") { continue }
            if key.contains(".rope.") && key.hasSuffix(".freqs") { continue }

            // Skip audio si pas d'audio tower
            if !hasAudio && (key.contains("audio_tower") || key.contains("embed_audio")) { continue }

            // Skip vision si pas de vision tower
            if !hasVision && (key.contains("vision_tower") || key.contains("embed_vision")) { continue }

            var newKey = key
            var newValue = value

            // Strip "model." prefix
            if newKey.hasPrefix("model.") {
                newKey = String(newKey.dropFirst("model.".count))
            }

            // Remap language_model paths (PyTorch → MLX module structure)
            // Les poids MLX ont deja "language_model.model.X" — ne pas re-mapper
            // Les poids PyTorch ont "language_model.X" (sans "model.") → ajouter "model."
            if newKey.hasPrefix("language_model.") && !newKey.hasPrefix("language_model.model.") {
                let rest = String(newKey.dropFirst("language_model.".count))
                newKey = "language_model.model." + rest
            }

            // Conv2d: transposer [out, in, H, W] → [out, H, W, in] pour les modeles Google BF16
            // Les modeles MLX community ont deja les poids au bon format
            if isGoogle && newKey.contains(".conv") && newKey.hasSuffix(".weight") && newValue.ndim == 4 {
                newValue = newValue.transposed(0, 2, 3, 1)
            }

            // Conv1d: transposer [out, in, K] → [out, K, in] pour les modeles Google BF16
            if isGoogle && newKey.contains(".conv") && newKey.hasSuffix(".weight") && newValue.ndim == 3 {
                newValue = newValue.transposed(0, 2, 1)
            }

            // MoE: experts.down_proj → experts.switch_glu.down_proj.weight
            if newKey.hasSuffix(".experts.down_proj") {
                newKey = newKey.replacingOccurrences(of: ".experts.down_proj", with: ".experts.switch_glu.down_proj.weight")
            }

            // MoE: experts.gate_up_proj → split en gate_proj + up_proj
            if newKey.hasSuffix(".experts.gate_up_proj") {
                let gateKey = newKey.replacingOccurrences(of: ".experts.gate_up_proj", with: ".experts.switch_glu.gate_proj.weight")
                let upKey = newKey.replacingOccurrences(of: ".experts.gate_up_proj", with: ".experts.switch_glu.up_proj.weight")

                let swapped = newValue.swappedAxes(-1, -2)
                let midDim = swapped.shape.last! / 2
                sanitized[gateKey] = swapped[.ellipsis, 0 ..< midDim].swappedAxes(-1, -2)
                sanitized[upKey] = swapped[.ellipsis, midDim...].swappedAxes(-1, -2)
                continue
            }

            sanitized[newKey] = newValue
        }

        return sanitized
    }
}
