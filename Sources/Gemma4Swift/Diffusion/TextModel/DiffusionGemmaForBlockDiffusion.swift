// Stub Phase 2 — sera complete en Phase 3+
//
// DiffusionGemmaForBlockDiffusion = wrapper top-level qui :
//   1. Encode le prompt (encoder text model) -> KV cache + last_hidden_state
//   2. Boucle de denoising du canvas (decoder model) -> logits par step
//   3. Sampling EntropyBound + temperature -> nouveau canvas
//   4. Stopping criterion -> early exit
//   5. Apply logit softcapping : tanh(logits / cap) * cap
//   6. LM head : reutilise embed_tokens.weight (tie_word_embeddings)
//
// Voir modular_diffusion_gemma.py ligne 1392-1482.

import Foundation
import MLX
import MLXNN

/// Wrapper top-level pour DiffusionGemma block diffusion.
///
/// STUB Phase 2 : seules les signatures sont en place. L'implementation suivra en Phase 3.
public class DiffusionGemmaForBlockDiffusion: Module {
    public let config: DiffusionGemmaConfig
    public let finalLogitSoftcapping: Float

    @ModuleInfo(key: "self_conditioning") var selfConditioning: DiffusionGemmaSelfConditioning

    public init(_ config: DiffusionGemmaConfig) {
        self.config = config
        self.finalLogitSoftcapping = config.textConfig.base.finalLogitSoftcapping
        self._selfConditioning.wrappedValue = DiffusionGemmaSelfConditioning(config.textConfig.base)
        super.init()
    }

    /// Applique le softcapping final aux logits (cf. config.text_config.final_logit_softcapping).
    /// `tanh(logits / cap) * cap` borne les logits dans [-cap, cap].
    public func applyLogitSoftcapping(_ logits: MLXArray) -> MLXArray {
        let cap = MLXArray(finalLogitSoftcapping)
        let scaled = logits / cap
        return MLX.tanh(scaled) * cap
    }

    // TODO Phase 3:
    //   public func forward(promptIds: MLXArray, canvasIds: MLXArray,
    //                       selfCondLogits: MLXArray?) -> MLXArray
}
