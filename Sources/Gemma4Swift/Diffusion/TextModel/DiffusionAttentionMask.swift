// Port de DiffusionGemmaDecoderModel.create_diffusion_decoder_attention_mask
//
// Le decoder voit la concatenation `[encoder_KV ; canvas_KV]`. Le masque doit donc :
//   - sur la portion `encoder_KV` : reproduire le `decoder_attention_mask` (gestion padding)
//   - sur la portion `canvas_KV` : tout 1 (bidirectionnel total sur le canvas)
//
// Pour full_attention -> masque complet shape `[B, 1, T_canvas, T_cache + T_canvas]`
// Pour sliding_attention -> slice [start_idx, end_idx) sur la portion encoder,
//   puis pad True sur les `canvas_length` derniers tokens.
//
// Python ref (modular_diffusion_gemma.py l.1109-1247) :
//   batch_size, canvas_length, _ = inputs_embeds.shape
//   valid_cache_tokens = past_key_values.get_seq_length()
//   full_cache_kv_length = valid_cache_tokens  # cas non-compileable
//   full_kv_length = full_cache_kv_length + canvas_length
//   full_mask = decoder_attention_mask[:, None, None, :].bool().expand(B, 1, T_canvas, full_kv_length)
//   sliding_cache_is_full = valid_cache_tokens >= sliding_window
//   if full:
//       sliding_start = valid_cache_tokens - sliding_window + 1
//       sliding_end = valid_cache_tokens
//   else:
//       sliding_start = 0
//       sliding_end = valid_cache_tokens
//   sliding_mask = full_mask[..., sliding_start:sliding_end]
//   sliding_mask = pad(sliding_mask, (0, canvas_length), value=True)

import Foundation
import MLX

/// Utilitaires de construction des masques d'attention du decoder DiffusionGemma.
public enum DiffusionAttentionMask {
    /// Masque dictionnaire par layer_type ("full_attention" / "sliding_attention").
    public struct Mapping: @unchecked Sendable {
        public let fullAttention: MLXArray?
        public let slidingAttention: MLXArray?

        public init(fullAttention: MLXArray?, slidingAttention: MLXArray?) {
            self.fullAttention = fullAttention
            self.slidingAttention = slidingAttention
        }
    }

    /// Cree la paire de masques pour un batch de canvases + cache encoder.
    ///
    /// - Parameters:
    ///   - canvasLength : longueur du canvas (256).
    ///   - encoderCacheLength : longueur effective du cache encoder (`valid_cache_tokens`).
    ///   - slidingWindow : taille de la fenetre glissante.
    ///   - decoderAttentionMask : `[B, encoderCacheLength + canvasLength]`, bool/int.
    ///     Si nil -> shortcut (nil, nil) : pas de padding, le decoder se debrouille
    ///     (attention bidirectionnelle totale, aucun mask materialise).
    /// - Returns: paire (full, sliding) prete a etre injectee dans `.array(mask)`.
    public static func create(
        canvasLength: Int,
        encoderCacheLength: Int,
        slidingWindow: Int,
        decoderAttentionMask: MLXArray?
    ) -> Mapping {
        // Shortcut : pas de mask explicite -> pas de padding, pas besoin de materialiser.
        guard let attnMask = decoderAttentionMask else {
            return Mapping(fullAttention: nil, slidingAttention: nil)
        }

        let batchSize = attnMask.dim(0)
        let fullKvLength = encoderCacheLength + canvasLength
        precondition(
            attnMask.dim(1) == fullKvLength,
            "decoderAttentionMask length \(attnMask.dim(1)) != \(encoderCacheLength) + \(canvasLength)"
        )

        // Full attention mask : [B, 1, canvasLength, fullKvLength] bool
        // Python : full_mask = decoder_attention_mask[:, None, None, :].bool().expand(B, 1, T_canvas, full_kv)
        let attnBool = attnMask.asType(.bool)
        let fullMask = MLX.broadcast(
            attnBool[0..., .newAxis, .newAxis, 0...],
            to: [batchSize, 1, canvasLength, fullKvLength]
        )

        // Sliding mask : slice + pad True sur canvasLength
        let validCacheTokens = encoderCacheLength
        let slidingStart: Int
        let slidingEnd: Int
        if validCacheTokens >= slidingWindow {
            // Cache plein : on prend les derniers (sliding_window - 1) tokens + le current
            slidingStart = validCacheTokens - slidingWindow + 1
            slidingEnd = validCacheTokens
        } else {
            slidingStart = 0
            slidingEnd = validCacheTokens
        }

        let slidingSlice: MLXArray
        if slidingEnd > slidingStart {
            slidingSlice = fullMask[0..., 0..., 0..., slidingStart ..< slidingEnd]
        } else {
            // Cas degenere : encoder cache vide -> slice 0-large. On cree un slice vide.
            slidingSlice = MLXArray.ones([batchSize, 1, canvasLength, 0], dtype: .bool)
        }

        // Pad True sur les `canvasLength` derniers tokens (canvas bidirectionnel total).
        let canvasTrue = MLXArray.ones([batchSize, 1, canvasLength, canvasLength], dtype: .bool)
        let slidingMask = concatenated([slidingSlice, canvasTrue], axis: -1)

        return Mapping(fullAttention: fullMask, slidingAttention: slidingMask)
    }
}
