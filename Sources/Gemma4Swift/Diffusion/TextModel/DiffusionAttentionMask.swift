// Port de DiffusionGemmaDecoderModel.create_diffusion_decoder_attention_mask
//
// Le decoder voit la concatenation `[encoder_KV ; canvas_KV]`. Le masque doit donc :
//   - sur la portion `encoder_KV` : reproduire le `decoder_attention_mask` (gestion padding)
//   - sur la portion `canvas_KV` : tout 1 (bidirectionnel total sur le canvas)
//
// Pour full_attention -> masque complet shape [B, 1, T_canvas, T_cache + T_canvas]
// Pour sliding_attention -> meme chose mais on slice la portion encoder pour ne garder
//   que la fenetre [start_idx, end_idx], puis on rappend la portion canvas a 1.

import Foundation
import MLX

/// Utilitaires de construction des masques d'attention du decoder DiffusionGemma.
public enum DiffusionAttentionMask {
    /// Masque dictionnaire par layer_type ("full_attention" / "sliding_attention").
    public struct Mapping: @unchecked Sendable {
        public let fullAttention: MLXArray?
        public let slidingAttention: MLXArray?
    }

    /// Cree la paire de masques pour un batch de canvases + cache encoder.
    ///
    /// - Parameters:
    ///   - canvasLength : longueur du canvas (256).
    ///   - encoderCacheLength : longueur effective du cache encoder.
    ///   - slidingWindow : taille de la fenetre glissante.
    ///   - decoderAttentionMask : `[B, encoderCacheLength + canvasLength]`, bool/int.
    ///     Si nil et pas de padding -> retourne (nil, nil) (l'attention interne gere).
    public static func create(
        canvasLength: Int,
        encoderCacheLength: Int,
        slidingWindow: Int,
        decoderAttentionMask: MLXArray?
    ) -> Mapping {
        // TODO Phase 3: implementer ce qui est code dans modular ligne 1109-1247
        // - shortcut nil/nil si pas de padding
        // - full_mask = decoderMask[:, None, None, :].expand(B, 1, T_canvas, fullKv)
        // - sliding_mask = full_mask[..., start:end] + pad(canvasLength * True)
        _ = canvasLength
        _ = encoderCacheLength
        _ = slidingWindow
        _ = decoderAttentionMask
        return Mapping(fullAttention: nil, slidingAttention: nil)
    }
}
