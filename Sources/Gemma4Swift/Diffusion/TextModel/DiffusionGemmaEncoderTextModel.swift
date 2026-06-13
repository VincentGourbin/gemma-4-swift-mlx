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
    public let config: Gemma4TextConfig
    public let useBidirectionalAttention: String

    @ModuleInfo(key: "embed_tokens") public var embedTokens: Embedding
    @ModuleInfo public var layers: [DiffusionGemmaEncoderTextLayer]
    @ModuleInfo public var norm: RMSNorm

    public let embedScale: Float

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

    /// Forward complet (sans cache) ou incremental (avec priorCache).
    ///
    /// - Parameters:
    ///   - inputs : nouveaux tokens (delta). Si priorCache nil, c'est tout le prompt.
    ///   - inputsEmbeds : alternative a inputs (utile multimodal).
    ///   - priorCache : si fourni, on encode SEULEMENT les nouveaux tokens.
    ///     Le K/V cache est etendu avec append. positionOffset = priorCache.seqLength.
    /// - Returns: DiffusionEncoderOutput avec cache mis a jour (priorCache + nouveaux).
    public func callAsFunction(
        inputs: MLXArray? = nil,
        inputsEmbeds: MLXArray? = nil,
        priorCache: EncoderKVCache? = nil
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

        let T_new = hiddenStates.dim(1)
        let positionOffset = priorCache?.seqLength ?? 0
        let T_total = positionOffset + T_new

        // Masque encoder avec offset pour les nouvelles queries vs cache total.
        // - "all"   -> bidir : .none (mais alors cache n'est pas semantique correct)
        // - causal  -> createCausalMask(n: T_new, offset: positionOffset)
        let mask: MLXFast.ScaledDotProductAttentionMaskMode
        if useBidirectionalAttention == "all" {
            mask = .none
        } else if T_total > 1 {
            mask = .array(MLXLMCommon.createCausalMask(n: T_new, offset: positionOffset))
        } else {
            mask = .none
        }
        let slidingMask = mask

        // Cache initial : copie du priorCache si fourni, sinon vide.
        var cache = priorCache ?? EncoderKVCache(numLayers: layers.count)
        let layerTypes = config.resolvedLayerTypes

        for (i, layer) in layers.enumerated() {
            let isGlobal = layerTypes[i] == "full_attention"
            let layerMask = isGlobal ? mask : slidingMask

            let priorKV = priorCache?.entries[i].map { (keys: $0.keys, values: $0.values) }

            let (output, newKeys, newValues) = layer(
                hiddenStates,
                mask: layerMask,
                positionOffset: positionOffset,
                priorKV: priorKV
            )
            hiddenStates = output

            // Cache : si priorKV present, append. Sinon, set direct.
            if let prior = priorKV {
                let mergedKeys = concatenated([prior.keys, newKeys], axis: 2)
                let mergedValues = concatenated([prior.values, newValues], axis: 2)
                cache.set(layerIdx: i, keys: mergedKeys, values: mergedValues)
            } else {
                cache.set(layerIdx: i, keys: newKeys, values: newValues)
            }
        }

        let normed = norm(hiddenStates)
        return DiffusionEncoderOutput(lastHiddenState: normed, kvCache: cache)
    }
}
