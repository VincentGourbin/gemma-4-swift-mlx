// Port de modular_diffusion_gemma.DiffusionGemmaEncoderTextModel (lignes 711-803)
//
// Encoder text model DiffusionGemma. Encode un prompt vers :
//   - last_hidden_state (utile pour analyse / training)
//   - EncoderKVCache : K/V par couche pour le decoder
//
// Forward :
//   1. inputs_embeds = embed_tokens(input_ids) * embed_scale
//   2. position_ids = arange(0, T)
//   3. mask = bidir total ou causal selon use_bidirectional_attention
//      - "all"  -> mask = .none (= softmax sur tout, bidirectionnel total)
//      - sinon  -> mask causal
//      Note Phase 3 : pour "vision", l'overlay block-bidir n'est pas implemente
//      cote encoder (cela demande un mm_token_type_ids). Fallback causal.
//   4. Loop layers, collect K/V dans EncoderKVCache
//   5. Norm finale

import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

/// Sortie de l'encoder DiffusionGemma : hidden + cache K/V.
public struct DiffusionEncoderOutput: @unchecked Sendable {
    public let lastHiddenState: MLXArray
    public let kvCache: EncoderKVCache

    public init(lastHiddenState: MLXArray, kvCache: EncoderKVCache) {
        self.lastHiddenState = lastHiddenState
        self.kvCache = kvCache
    }
}

/// Encoder text model DiffusionGemma.
public class DiffusionGemmaEncoderTextModel: Module {
    let config: Gemma4TextConfig
    let useBidirectionalAttention: String

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [DiffusionGemmaEncoderTextLayer]
    @ModuleInfo var norm: RMSNorm

    let embedScale: Float

    public init(_ textConfig: DiffusionGemmaTextConfig) {
        self.config = textConfig.base
        self.useBidirectionalAttention = textConfig.useBidirectionalAttention
        self.embedScale = pow(Float(config.hiddenSize), 0.5)

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize,
            dimensions: config.hiddenSize
        )

        let baseConfig = self.config
        self._layers.wrappedValue = (0 ..< config.numHiddenLayers).map { i in
            DiffusionGemmaEncoderTextLayer(baseConfig, layerIdx: i)
        }

        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        super.init()
    }

    /// Forward avec inputs_embeds deja calcule (utile pour multimodal).
    public func callAsFunction(
        inputs: MLXArray? = nil,
        inputsEmbeds: MLXArray? = nil
    ) -> DiffusionEncoderOutput {
        var hiddenStates: MLXArray
        if let inputsEmbeds = inputsEmbeds {
            hiddenStates = inputsEmbeds
        } else if let inputs = inputs {
            hiddenStates = embedTokens(inputs)
            hiddenStates = hiddenStates * MLXArray(embedScale, dtype: hiddenStates.dtype)
        } else {
            fatalError("inputs ou inputsEmbeds requis")
        }

        let T = hiddenStates.dim(1)

        // Masque encoder :
        // - "all"   -> bidirectionnel total = .none
        // - autre   -> causal
        // TODO Phase 3+: "vision" demande l'overlay bidir sur les tokens d'image
        //   (cf. mm_token_type_ids cote Python). Pour l'instant on tombe sur causal.
        let mask: MLXFast.ScaledDotProductAttentionMaskMode
        if useBidirectionalAttention == "all" {
            mask = .none
        } else if T > 1 {
            mask = .array(MLXLMCommon.createCausalMask(n: T, offset: 0))
        } else {
            mask = .none
        }

        // Note : pour la fenetre glissante, MLXLMCommon.createAttentionMask gere
        // la window mais cote encoder DiffusionGemma, les layers sliding utilisent
        // le meme masque que les full (l'attention est meme causal partout). Le
        // SDPA ignore au-dela de window grace au flag `sliding_window` cote Python.
        // Cote Swift, on materialise simplement le mask causal complet.
        // TODO Phase 3+: utiliser sliding_window separe si necessaire.
        let slidingMask = mask

        var cache = EncoderKVCache(numLayers: layers.count)
        let layerTypes = config.resolvedLayerTypes

        for (i, layer) in layers.enumerated() {
            let isGlobal = layerTypes[i] == "full_attention"
            let layerMask = isGlobal ? mask : slidingMask

            let (output, keys, values) = layer(
                hiddenStates,
                mask: layerMask,
                positionOffset: 0
            )
            hiddenStates = output
            cache.set(layerIdx: i, keys: keys, values: values)
        }

        let normed = norm(hiddenStates)
        return DiffusionEncoderOutput(lastHiddenState: normed, kvCache: cache)
    }
}
