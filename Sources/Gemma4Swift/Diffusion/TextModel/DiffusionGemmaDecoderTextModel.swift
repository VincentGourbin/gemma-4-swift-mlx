// Port de modular_diffusion_gemma.DiffusionGemmaDecoderModel (lignes 976-1107)
//
// Forward du decoder :
//   1. inputs_embeds = embed_tokens(decoder_input_ids) * embed_scale
//   2. soft_embeddings = self_conditioning_logits != nil
//        ? softmax(self_conditioning_logits, fp32) @ embed_tokens.weight * embed_scale
//        : zeros_like(inputs_embeds)
//      Si self_conditioning_mask -> multiplie par mask[:, None, None]
//   3. inputs_embeds = self_conditioning(inputs_embeds, soft_embeddings)
//   4. decoder_position_ids = arange(cache_seq_length, cache_seq_length + canvas_length)
//      sauf si fourni explicitement.
//   5. mask_mapping = create_diffusion_decoder_attention_mask(...)
//   6. boucle layers (en passant le bon mask par layer_type) + encoder cache
//   7. norm finale

import Foundation
import MLX
import MLXFast
import MLXNN

/// Decoder text model DiffusionGemma : self_conditioning + layers + norm.
public class DiffusionGemmaDecoderTextModel: Module {
    let config: Gemma4TextConfig
    let canvasLength: Int

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [DiffusionGemmaDecoderTextLayer]
    @ModuleInfo var norm: RMSNorm
    @ModuleInfo(key: "self_conditioning") var selfConditioning: DiffusionGemmaSelfConditioning

    let embedScale: Float

    public init(_ textConfig: DiffusionGemmaTextConfig) {
        self.config = textConfig.base
        self.canvasLength = textConfig.canvasLength
        self.embedScale = pow(Float(config.hiddenSize), 0.5)

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize,
            dimensions: config.hiddenSize
        )

        let baseConfig = self.config
        self._layers.wrappedValue = (0 ..< config.numHiddenLayers).map { i in
            DiffusionGemmaDecoderTextLayer(baseConfig, layerIdx: i)
        }

        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._selfConditioning.wrappedValue = DiffusionGemmaSelfConditioning(config)

        super.init()
    }

    /// Forward de denoising d'un canvas.
    ///
    /// - Parameters:
    ///   - decoderInputIds : `[B, canvasLength]` int. Canvas a denoiser.
    ///   - encoderCache : K/V de l'encoder (read-only). Si nil, pas de cross-attention.
    ///   - selfConditioningLogits : `[B, canvasLength, vocabSize]` fp32, logits du step precedent.
    ///     Si nil -> soft_embeddings = 0 (1er step).
    ///   - selfConditioningMask : `[B]` bool. Masque per-example sur soft_embeddings.
    ///   - decoderAttentionMask : `[B, encoderLen + canvasLength]` bool/int. Padding encoder.
    ///   - decoderPositionIds : override des position_ids. Si nil -> arange(cacheLen, cacheLen + canvasLen).
    /// - Returns: `[B, canvasLength, hidden_size]` hidden states post-norm finale.
    public func callAsFunction(
        decoderInputIds: MLXArray,
        encoderCache: EncoderKVCache? = nil,
        selfConditioningLogits: MLXArray? = nil,
        selfConditioningMask: MLXArray? = nil,
        decoderAttentionMask: MLXArray? = nil
    ) -> MLXArray {
        // 1) Embeddings du canvas
        var inputsEmbeds = embedTokens(decoderInputIds)
        inputsEmbeds = inputsEmbeds * MLXArray(embedScale, dtype: inputsEmbeds.dtype)

        // 2) Soft-embeddings (self-conditioning)
        let softEmbeddings: MLXArray
        if let scLogits = selfConditioningLogits {
            // softmax fp32 puis cast vers weight dtype
            let weight = embedTokens.weight  // [vocab_size, hidden_size]
            let weightDtype = weight.dtype
            var probs = softmax(scLogits.asType(.float32), axis: -1)
            probs = probs.asType(weightDtype)
            // soft = probs @ weight : [B, canvas, vocab] @ [vocab, hidden] = [B, canvas, hidden]
            var soft = matmul(probs, weight) * MLXArray(embedScale, dtype: inputsEmbeds.dtype)
            // selfConditioningMask : [B] bool -> [B, 1, 1]
            if let mask = selfConditioningMask {
                let maskExpanded = mask.asType(soft.dtype)[0..., .newAxis, .newAxis]
                soft = soft * maskExpanded
            }
            softEmbeddings = soft.asType(inputsEmbeds.dtype)
        } else {
            softEmbeddings = MLXArray.zeros(like: inputsEmbeds)
        }

        // 3) Fusion self-conditioning + inputs
        var hiddenStates = selfConditioning(inputsEmbeds, selfConditioningSignal: softEmbeddings)

        // 4) Position_ids : continue apres l'encoder
        let encoderCacheLength = encoderCache?.seqLength ?? 0

        // 5) Masques par layer_type
        let mapping = DiffusionAttentionMask.create(
            canvasLength: hiddenStates.dim(1),
            encoderCacheLength: encoderCacheLength,
            slidingWindow: config.slidingWindow,
            decoderAttentionMask: decoderAttentionMask
        )

        // 6) Boucle layers
        let layerTypes = config.resolvedLayerTypes
        for (i, layer) in layers.enumerated() {
            let isGlobal = layerTypes[i] == "full_attention"
            let maskArray = isGlobal ? mapping.fullAttention : mapping.slidingAttention
            let mask: MLXFast.ScaledDotProductAttentionMaskMode =
                maskArray.map { .array($0) } ?? .none

            let encoderEntry = encoderCache?.entries[i]
            hiddenStates = layer(
                hiddenStates,
                mask: mask,
                encoderEntry: encoderEntry,
                encoderCacheLength: encoderCacheLength
            )
        }

        // 7) Norm finale
        return norm(hiddenStates)
    }
}
