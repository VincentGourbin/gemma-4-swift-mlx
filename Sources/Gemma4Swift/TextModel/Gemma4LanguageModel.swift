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
        cache: [KVCache?]? = nil,
        perLayerInputs: MLXArray? = nil
    ) -> MLXArray {
        var out = model(
            inputs: inputs,
            inputsEmbeds: inputsEmbeds,
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

    /// Estime si TurboQuant est benefique pour cette architecture.
    /// TurboQuant compresse les couches full attention. L'overhead fixe
    /// (rotation matrices, codecs, graph MLX) n'est rentable que si le
    /// nombre de couches full attention et la taille des KV heads sont
    /// suffisants pour que le gain de compression depasse l'overhead.
    public func turboQuantViable(bits: Float) -> (viable: Bool, fullAttnLayers: Int, kvHeadDim: Int, reason: String) {
        let layerTypes = config.resolvedLayerTypes
        let concreteLayers = Array(layerTypes[..<config.firstKvSharedLayerIdx])
        let fullAttnCount = concreteLayers.filter { $0 == "full_attention" }.count
        let kvHeads = config.numKeyValueHeads
        let headDim = config.globalHeadDim > 0 ? config.globalHeadDim : config.headDim

        // Overhead fixe ~500 Mo process (graph MLX, codecs, rotation matrices)
        // Gain par couche full attention a 16K tokens:
        //   BF16: T * D * kvHeads * 2 (K+V) * 2 bytes
        //   TQ:   T * (2 + packedWidth*4) * kvHeads * 2
        //   Rotation: D * D * 4 * 2 (key+value codecs)
        let T = 16000 // reference context
        let packedWidth = (headDim * Int(bits) + 31) / 32
        let bf16PerLayer = T * headDim * kvHeads * 2 * 2
        let tqPerLayer = T * (2 + packedWidth * 4) * kvHeads * 2
        let rotPerLayer = headDim * headDim * 4 * 2
        let savingPerLayer = bf16PerLayer - tqPerLayer - rotPerLayer
        let totalSaving = savingPerLayer * fullAttnCount
        let overheadEstimate = 500 * 1024 * 1024 // ~500 Mo process overhead

        if fullAttnCount < 3 {
            return (false, fullAttnCount, headDim, "trop peu de couches full attention (\(fullAttnCount))")
        }
        if totalSaving < overheadEstimate / 2 {
            return (false, fullAttnCount, headDim, "gain estime \(totalSaving / 1024 / 1024) Mo < overhead ~500 Mo a 16K tokens")
        }
        return (true, fullAttnCount, headDim, "gain estime \(totalSaving / 1024 / 1024) Mo a 16K tokens sur \(fullAttnCount) couches")
    }

    /// Cree les caches KV pour chaque couche concrete (non-partagee)
    /// - Parameter kvBits: si specifie, utilise TurboQuant pour les couches full attention
    ///   Si le modele n'a pas assez de couches full attention, TurboQuant est desactive automatiquement
    public func makeCache(kvBits: Float? = nil) -> [any KVCache] {
        var caches: [any KVCache] = []
        let layerTypes = config.resolvedLayerTypes
        let concreteLayers = Array(layerTypes[..<config.firstKvSharedLayerIdx])

        // Guard: verifier si TurboQuant est viable pour cette architecture
        var effectiveKvBits = kvBits
        if let bits = kvBits, bits > 0 {
            let (viable, _, _, reason) = turboQuantViable(bits: bits)
            if !viable {
                print("[TurboQuant] Desactive: \(reason)")
                effectiveKvBits = nil
            }
        }

        for layerType in concreteLayers {
            if layerType == "full_attention" {
                if let bits = effectiveKvBits, bits > 0 {
                    caches.append(TurboQuantKVCache(bits: bits))
                } else {
                    caches.append(KVCacheSimple())
                }
            } else {
                caches.append(MLXLMCommon.RotatingKVCache(maxSize: config.slidingWindow, keep: 0))
            }
        }
        return caches
    }
}
