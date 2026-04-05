// Port de gemma4.py MultimodalEmbedder — Projection des tokens visuels/audio vers l'espace texte

import MLX
import MLXNN

/// Projette les soft tokens (vision/audio) vers la dimension du modele texte
public class MultimodalEmbedder: Module {
    @ModuleInfo(key: "embedding_projection") var embeddingProjection: Linear
    @ModuleInfo(key: "embedding_pre_projection_norm") var embeddingPreProjectionNorm: RMSNormNoScale

    public init(embeddingDim: Int, textHiddenSize: Int, eps: Float = 1e-6) {
        self._embeddingProjection.wrappedValue = Linear(embeddingDim, textHiddenSize, bias: false)
        self._embeddingPreProjectionNorm.wrappedValue = RMSNormNoScale(eps: eps)
        super.init()
    }

    public func callAsFunction(_ inputsEmbeds: MLXArray) -> MLXArray {
        let normed = embeddingPreProjectionNorm(inputsEmbeds)
        return embeddingProjection(normed)
    }
}
