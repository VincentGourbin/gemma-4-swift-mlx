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
//      Pour "vision" : causal + overlay bidirectionnel dans chaque bloc image
//      contigu (visionTokenMask, meme overlay que le 12B Unified ; D-06).
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
    /// - Parameter visionTokenMask: `[B, T_new]` bool, positions image du delta. Avec
    ///   `use_bidirectional_attention == "vision"`, l'attention est bidirectionnelle a
    ///   l'interieur de chaque bloc image contigu (D-06, meme overlay que le 12B Unified).
    public func callAsFunction(
        inputs: MLXArray? = nil,
        inputsEmbeds: MLXArray? = nil,
        priorCache: EncoderKVCache? = nil,
        visionTokenMask: MLXArray? = nil
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
        // Couches glissantes : fenetre de `slidingWindow` positions (D-05). Leur cache
        // ne garde que les `slidingWindow - 1` dernieres positions (comme le DynamicCache
        // Python) : les cles vont de `positionOffset - priorSliding` a `T_total - 1`.
        let layerTypes = config.resolvedLayerTypes
        let window = config.slidingWindow
        let priorSliding = priorCache.flatMap { cache in
            layerTypes.indices.first { layerTypes[$0] != "full_attention" && cache.entries[$0] != nil }
                .map { cache.entries[$0]!.keys.dim(2) }
        } ?? 0
        let slidingMask: MLXFast.ScaledDotProductAttentionMaskMode
        if useBidirectionalAttention == "all" || T_total <= 1 {
            slidingMask = mask
        } else {
            let queryPositions = (MLXArray(Int32(positionOffset)) + MLXArray(0 ..< Int32(T_new)))[0..., .newAxis]
            let keyPositions = (MLXArray(Int32(positionOffset - priorSliding))
                + MLXArray(0 ..< Int32(priorSliding + T_new)))[.newAxis, 0...]
            slidingMask = .array((keyPositions .<= queryPositions)
                .&& ((queryPositions - keyPositions) .< MLXArray(Int32(window))))
        }

        // Blocs image bidirectionnels (mode "vision") : OU avec l'overlay « meme bloc ».
        // Les blocs sont dans le delta courant (l'image n'est encodee qu'au premier
        // appel) ; les colonnes du cache anterieur ne recoivent pas d'overlay.
        var fullMask = mask
        var windowMask = slidingMask
        if useBidirectionalAttention == "vision", let visionTokenMask, T_new > 1 {
            let blocks = Gemma4BidirectionalMask.blockSequenceIds(visionMask: visionTokenMask)
            var overlay = Gemma4BidirectionalMask.overlay(blockSequenceIds: blocks)  // [B, T_new, T_new]
            let B = overlay.dim(0)
            overlay = overlay[0..., .newAxis, 0..., 0...]                              // [B, 1, T_new, T_new]
            func compose(_ base: MLXFast.ScaledDotProductAttentionMaskMode, priorColumns: Int)
                -> MLXFast.ScaledDotProductAttentionMaskMode
            {
                let causal: MLXArray
                switch base {
                case .array(let array): causal = array
                default: causal = MLXLMCommon.createCausalMask(n: T_new, offset: priorColumns)
                }
                let padded = priorColumns > 0
                    ? concatenated([MLXArray.zeros([B, 1, T_new, priorColumns], type: Bool.self), overlay], axis: -1)
                    : overlay
                return .array(causal[.newAxis, .newAxis] .|| padded)
            }
            fullMask = compose(mask, priorColumns: positionOffset)
            windowMask = compose(slidingMask, priorColumns: priorSliding)
        }

        // Cache initial : copie du priorCache si fourni, sinon vide.
        var cache = priorCache ?? EncoderKVCache(numLayers: layers.count)

        for (i, layer) in layers.enumerated() {
            let isGlobal = layerTypes[i] == "full_attention"
            let layerMask = isGlobal ? fullMask : windowMask

            let priorKV = priorCache?.entries[i].map { (keys: $0.keys, values: $0.values) }

            let (output, newKeys, newValues) = layer(
                hiddenStates,
                mask: layerMask,
                positionOffset: positionOffset,
                priorKV: priorKV
            )
            hiddenStates = output

            // Cache : si priorKV present, append. Sinon, set direct.
            var mergedKeys = newKeys
            var mergedValues = newValues
            if let prior = priorKV {
                mergedKeys = concatenated([prior.keys, newKeys], axis: 2)
                mergedValues = concatenated([prior.values, newValues], axis: 2)
            }
            // Couche glissante : ne garder que les `window - 1` dernieres positions.
            let keep = window - 1
            if !isGlobal && mergedKeys.dim(2) > keep {
                let start = mergedKeys.dim(2) - keep
                mergedKeys = mergedKeys[0..., 0..., start...]
                mergedValues = mergedValues[0..., 0..., start...]
            }
            cache.set(layerIdx: i, keys: mergedKeys, values: mergedValues)
        }
        cache.offset = T_total

        let normed = norm(hiddenStates)
        return DiffusionEncoderOutput(lastHiddenState: normed, kvCache: cache)
    }
}
