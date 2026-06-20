// Port de configuration_diffusion_gemma.DiffusionGemmaTextConfig
//
// Sous-config texte de DiffusionGemma. 95% identique a Gemma4TextConfig :
// memes hidden_size, num_layers, num_experts, layer_types... mais expose en plus :
//  - use_bidirectional_attention: "vision" (controle l'encoder, pas le decoder)
//  - canvas_length (256 par defaut) : longueur du canvas pour le bloc-AR
//
// Strategy : on **reutilise** Gemma4TextConfig en interne et on stocke seulement les deltas.
// Le DecoderTextLayer cote MLX-Swift reutilise [[Gemma4DecoderLayer]] tel quel.

import Foundation

/// Sous-config texte specifique a DiffusionGemma.
///
/// Compose autour de [[Gemma4TextConfig]] + champs supplementaires pour la
/// diffusion :
/// - `useBidirectionalAttention` : "vision" | "all" | "none". Controle uniquement
///   l'encoder ; le decoder est *toujours* bidirectionnel.
/// - `canvasLength` : taille du bloc denoise simultanement (256).
public struct DiffusionGemmaTextConfig: Decodable, @unchecked Sendable {
    /// Config Gemma 4 texte sous-jacente (hidden_size, layers, MoE, etc.)
    public let base: Gemma4TextConfig

    /// "vision" pour le checkpoint 26B-A4B publie. Ne controle QUE l'encoder.
    /// Le decoder est toujours `is_causal=False` (bidirectionnel) cf. Python ligne 312.
    public let useBidirectionalAttention: String

    /// Longueur du canvas (block-AR). 256 dans le checkpoint publie.
    public let canvasLength: Int

    enum CodingKeys: String, CodingKey {
        case useBidirectionalAttention = "use_bidirectional_attention"
        case canvasLength = "canvas_length"
    }

    public init(from decoder: Decoder) throws {
        self.base = try Gemma4TextConfig(from: decoder)
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.useBidirectionalAttention = try c.decodeIfPresent(String.self, forKey: .useBidirectionalAttention) ?? "vision"
        self.canvasLength = try c.decodeIfPresent(Int.self, forKey: .canvasLength) ?? 256
    }
}
