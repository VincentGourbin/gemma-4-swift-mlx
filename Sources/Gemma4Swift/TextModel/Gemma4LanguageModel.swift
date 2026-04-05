// Port de language.py LanguageModel — Wrapper avec softcap et generation de cache

import Foundation
import MLX
import MLXFast
import MLXNN
import MLXLMCommon

/// Modele de langage Gemma 4 complet (text model + logit head + softcapping)
public class Gemma4LanguageModel: Module {
    let config: Gemma4TextConfig
    let finalLogitSoftcapping: Float?

    @ModuleInfo var model: Gemma4TextModel

    public init(_ config: Gemma4TextConfig) {
        self.config = config
        self.finalLogitSoftcapping = config.finalLogitSoftcapping > 0 ? config.finalLogitSoftcapping : nil

        self._model.wrappedValue = Gemma4TextModel(config)
        super.init()
    }

    public func callAsFunction(
        inputs: MLXArray? = nil,
        inputsEmbeds: MLXArray? = nil,
        mask: MLXArray? = nil,
        cache: [KVCache?]? = nil,
        perLayerInputs: MLXArray? = nil
    ) -> MLXArray {
        var out = model(
            inputs: inputs,
            inputsEmbeds: inputsEmbeds,
            mask: mask,
            cache: cache,
            perLayerInputs: perLayerInputs
        )

        // Tied word embeddings: utiliser embed_tokens comme linear
        out = model.embedTokens.asLinear(out)

        // Final logit softcapping
        if let softcap = finalLogitSoftcapping {
            out = tanh(out / softcap) * softcap
        }

        return out
    }

    /// Cree les caches KV pour chaque couche concrete (non-partagee)
    public func makeCache() -> [any KVCache] {
        var caches: [any KVCache] = []
        let layerTypes = config.resolvedLayerTypes
        let concreteLayers = Array(layerTypes[..<config.firstKvSharedLayerIdx])

        for layerType in concreteLayers {
            if layerType == "full_attention" {
                caches.append(KVCacheSimple())
            } else {
                caches.append(MLXLMCommon.RotatingKVCache(maxSize: config.slidingWindow, keep: 0))
            }
        }
        return caches
    }
}
