// Port de modular_diffusion_gemma.DiffusionGemmaForBlockDiffusion (lignes 1392-1482)
//
// Wrapper top-level pour DiffusionGemma block diffusion :
//   1. encoder.forward(prompt_ids) -> EncoderKVCache + last_hidden_state
//   2. decoder.forward(canvas_ids, encoder_cache, self_cond_logits) -> hidden
//   3. lm_head(hidden) -> logits
//   4. logits = tanh(logits / cap) * cap  (final logit softcapping)
//
// Note : tie_word_embeddings = true -> lm_head reutilise embed_tokens.weight.
// Pas de Linear separe. On expose `logits(_ hidden: MLXArray)` qui fait
// `hidden @ decoder.embed_tokens.weight.T` directement.

import Foundation
import MLX
import MLXNN

/// Sortie d'un step de denoising : logits + hidden states optionnels.
public struct DiffusionForwardOutput: @unchecked Sendable {
    public let logits: MLXArray
    public let encoderHidden: MLXArray?

    public init(logits: MLXArray, encoderHidden: MLXArray? = nil) {
        self.logits = logits
        self.encoderHidden = encoderHidden
    }
}

/// Wrapper top-level DiffusionGemma. Combine encoder + decoder + LM head softcap.
public class DiffusionGemmaForBlockDiffusion: Module {
    public let config: DiffusionGemmaConfig
    public let finalLogitSoftcapping: Float

    @ModuleInfo(key: "encoder") var encoder: DiffusionGemmaEncoderModel
    @ModuleInfo(key: "decoder") var decoder: DiffusionGemmaDecoderTextModel

    public init(_ config: DiffusionGemmaConfig) {
        self.config = config
        self.finalLogitSoftcapping = config.textConfig.base.finalLogitSoftcapping
        self._encoder.wrappedValue = DiffusionGemmaEncoderModel(config)
        self._decoder.wrappedValue = DiffusionGemmaDecoderTextModel(config.textConfig)
        super.init()
    }

    /// Applique le softcapping final aux logits.
    public func applyLogitSoftcapping(_ logits: MLXArray) -> MLXArray {
        let cap = MLXArray(finalLogitSoftcapping)
        let scaled = logits / cap
        return MLX.tanh(scaled) * cap
    }

    /// Calcule les logits depuis les hidden states du decoder.
    /// tie_word_embeddings = true : on reutilise decoder.embed_tokens via asLinear
    /// (override par QuantizedEmbedding pour gerer la dequantization au vol).
    public func computeLogits(_ hiddenStates: MLXArray) -> MLXArray {
        // asLinear fait x @ weight.T en interne, et fonctionne aussi bien
        // pour Embedding standard que pour QuantizedEmbedding (qui dequantize
        // automatiquement le pack 4-bit/8-bit lors du matmul).
        let logits = decoder.embedTokens.asLinear(hiddenStates)
        return applyLogitSoftcapping(logits.asType(.float32))
    }

    /// Encode un prompt et retourne le cache encoder + le last hidden state.
    /// - Parameters:
    ///   - promptIds : si priorCache nil, c'est le prompt complet. Sinon, ce sont
    ///     SEULEMENT les nouveaux tokens a appender (delta).
    ///   - pixelValues : ignore si priorCache != nil (vision deja dans le cache).
    ///   - priorCache : K/V cache encoder existant a etendre (mode incremental).
    public func encodePrompt(
        promptIds: MLXArray,
        pixelValues: MLXArray? = nil,
        priorCache: EncoderKVCache? = nil
    ) -> DiffusionEncoderOutput {
        encoder(inputIds: promptIds, pixelValues: pixelValues, priorCache: priorCache)
    }

    /// Forward d'un step de denoising : canvas -> logits.
    ///
    /// - Parameters:
    ///   - canvasIds : `[B, canvasLength]` int. Canvas a denoiser.
    ///   - encoderCache : K/V du prompt (produit par encodePrompt).
    ///   - selfConditioningLogits : logits du step precedent (nil = 1er step).
    ///   - decoderAttentionMask : padding optionnel (nil = pas de padding).
    /// - Returns: logits + softcap apres LM head.
    public func denoiseStep(
        canvasIds: MLXArray,
        encoderCache: EncoderKVCache,
        selfConditioningLogits: MLXArray? = nil,
        decoderAttentionMask: MLXArray? = nil
    ) -> MLXArray {
        let hidden = decoder(
            decoderInputIds: canvasIds,
            encoderCache: encoderCache,
            selfConditioningLogits: selfConditioningLogits,
            selfConditioningMask: nil,
            decoderAttentionMask: decoderAttentionMask
        )
        return computeLogits(hidden)
    }

    /// Forward complet : encode + denoise un step.
    public func callAsFunction(
        promptIds: MLXArray,
        canvasIds: MLXArray,
        selfConditioningLogits: MLXArray? = nil
    ) -> DiffusionForwardOutput {
        let enc = encodePrompt(promptIds: promptIds)
        let logits = denoiseStep(
            canvasIds: canvasIds,
            encoderCache: enc.kvCache,
            selfConditioningLogits: selfConditioningLogits
        )
        return DiffusionForwardOutput(logits: logits, encoderHidden: enc.lastHiddenState)
    }
}
