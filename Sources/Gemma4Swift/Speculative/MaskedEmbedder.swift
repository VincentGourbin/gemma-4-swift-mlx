// Port de mlx-vlm/speculative/drafters/gemma4_assistant/masked_embedder.py
//
// Centroid-routed sparse softmax: au lieu de calculer hidden @ embed.T sur tout
// le vocab (262144 tokens, couteux pour un drafter avec hidden_size=256), le
// drafter score 2048 clusters de tokens via une couche `centroids`, selectionne
// les top-K (32) clusters, et calcule les logits seulement pour les tokens de
// ces clusters (32 * vocab_size_per_centroid = 32 * 128 = 4096 tokens).
// Les autres positions du vocab sont masquees a min - 1.

import Foundation
import MLX
import MLXNN

/// LM head sparse a centroides pour le drafter Gemma 4 Assistant.
public class MaskedEmbedder: Module {
    public let hiddenSize: Int
    public let vocabSize: Int
    public let numCentroids: Int
    public let topK: Int
    public let vocabSizePerCentroid: Int

    @ModuleInfo var centroids: Linear  // [num_centroids, hidden_size]
    @ParameterInfo(key: "token_ordering") var tokenOrdering: MLXArray  // [vocab_size], int32

    public init(_ config: Gemma4AssistantConfig) {
        self.hiddenSize = config.textConfig.hiddenSize
        self.vocabSize = config.textConfig.vocabSize
        self.numCentroids = config.numCentroids
        self.topK = config.centroidIntermediateTopK
        self.vocabSizePerCentroid = config.textConfig.vocabSize / config.numCentroids

        self._centroids.wrappedValue = Linear(hiddenSize, numCentroids, bias: false)
        self._tokenOrdering.wrappedValue = MLXArray.zeros([config.textConfig.vocabSize], type: Int32.self)

        super.init()
    }

    /// Calcule les logits sparses sur le vocab complet.
    ///
    /// - Parameters:
    ///   - hidden: hidden states `[B, L, hidden_size]`
    ///   - lmHeadWeight: matrice d'embedding tied `[vocab_size, hidden_size]`
    /// - Returns: logits `[B, L, vocab_size]`. Les positions non-selectionnees
    ///   sont masquees a `min(selected_logits) - 1`.
    public func callAsFunction(_ hidden: MLXArray, lmHeadWeight: MLXArray) -> MLXArray {
        let B = hidden.dim(0)
        let L = hidden.dim(1)

        // 1. Scores des centroides [B, L, num_centroids]
        let centroidLogits = centroids(hidden)

        // 2. Top-K indices de clusters [B, L, top_k]
        // argPartition(kth=N-K) place le (N-K)-eme element a sa position triee.
        // Les indices >= N-K sont donc les top-K plus grands.
        let partitioned = argPartition(centroidLogits, kth: numCentroids - topK, axis: -1)
        let topkIdx = partitioned[.ellipsis, (numCentroids - topK)...]  // [B, L, top_k]

        // 3. Reshape token_ordering en [num_centroids, vocab_size_per_centroid]
        let ordering = tokenOrdering.reshaped(numCentroids, vocabSizePerCentroid)

        // 4. Gather: pour chaque cluster selectionne, recuperer ses token IDs
        // ordering.take(topkIdx, axis: 0) → [B, L, top_k, vocab_size_per_centroid]
        let selectedCanonical = ordering.take(topkIdx, axis: 0)

        // 5. Embedding lookup pour les tokens selectionnes
        // flat_idx [B*L*top_k*vsc] → selectedEmb [B, L, top_k*vsc, hidden_size]
        let flatIdx = selectedCanonical.reshaped(-1)
        let selectedEmb = lmHeadWeight.take(flatIdx, axis: 0)
            .reshaped(B, L, topK * vocabSizePerCentroid, hiddenSize)

        // 6. selected_logits = h @ E.T
        // hidden[..., None, :] [B, L, 1, hidden] @ selectedEmb.T [B, L, hidden, top_k*vsc]
        // → [B, L, 1, top_k*vsc] → squeeze → [B, L, top_k*vsc]
        let hiddenExpanded = expandedDimensions(hidden, axis: -2)
        let embT = selectedEmb.swappedAxes(-1, -2)
        let selectedLogits = matmul(hiddenExpanded, embT).squeezed(axis: -2)

        // 7. Mask value = min des logits selectionnes - 1
        let maskValueArr = selectedLogits.min() - MLXArray(Float(1.0))

        // 8. Allouer le tenseur full-vocab rempli avec mask_value
        let scatterIdx = selectedCanonical.reshaped(B, L, -1)  // [B, L, top_k*vsc]
        let out = MLXArray.full(
            [B, L, vocabSize],
            values: maskValueArr.asType(hidden.dtype)
        )

        // 9. Scatter selected_logits aux positions canoniques du vocab
        return putAlong(out, scatterIdx, values: selectedLogits, axis: -1)
    }
}
