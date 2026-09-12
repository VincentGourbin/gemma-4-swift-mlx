// Conformance LLMModel pour integration avec mlx-swift-lm (ChatSession, ModelContainer, etc.)

import Foundation
import MLX
import MLXFast
import MLXNN
import MLXLMCommon
import MLXLLM

/// Modele Gemma 4 conforme au protocol LLMModel de mlx-swift-lm.
/// Permet l'utilisation via Gemma4Registration.loadContainer() et ChatSession.
public class Gemma4LLMModel: Module, LLMModel, LoRAModel {
    @ModuleInfo(key: "language_model") public var languageModel: Gemma4LanguageModel

    public let config: Gemma4TextConfig
    public let modelType: String

    public var kvHeads: [Int]

    public init(config: Gemma4TextConfig) {
        self.config = config
        self.modelType = config.modelType

        self._languageModel.wrappedValue = Gemma4LanguageModel(config)
        self.kvHeads = Array(repeating: config.numKeyValueHeads, count: config.numHiddenLayers)

        super.init()
    }

    // MARK: - LoRAModel conformance

    /// Couches exposees pour l'application des adaptateurs LoRA (toutes les couches du transformer).
    public var loraLayers: [Module] {
        languageModel.model.layers.map { $0 as Module }
    }

    // MARK: - LLMModel conformance

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let cacheArray: [KVCache?]? = cache?.map { $0 as KVCache? }
        return languageModel(inputs: inputs, cache: cacheArray)
    }

    /// Hidden states de toutes les couches, convention HuggingFace
    /// `output_hidden_states=True` : `num_hidden_layers + 1` tenseurs
    /// `[B, T, hidden_size]` (49 x `[B, T, 3840]` sur le 12B).
    ///
    /// C'est le point d'entree de l'usage "encodeur texte" : `Gemma4Registration`
    /// resout `gemma4`, `gemma4_text` et `gemma4_unified` vers ce type des lors que
    /// `register(multimodal:)` vaut `false`. Meme API que sur
    /// [[Gemma4UnifiedMultimodalLLMModel]], pour que le consommateur n'ait pas a
    /// savoir quel wrapper le registry lui a rendu.
    ///
    /// - Parameter cache: normalement `nil` pour un encodage one-shot.
    public func forwardCollectingHiddenStates(
        _ inputs: MLXArray,
        cache: [KVCache]? = nil
    ) -> [MLXArray] {
        let cacheArray: [KVCache?]? = cache?.map { $0 as KVCache? }
        return languageModel.forwardCollectingHiddenStates(inputs: inputs, cache: cacheArray)
    }

    public func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        let kvBits: Float? = parameters?.kvBits != nil ? Float(parameters!.kvBits!) : nil
        return languageModel.makeCache(kvBits: kvBits)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        WeightSanitizer.sanitize(
            weights: weights,
            firstKvSharedLayerIdx: config.firstKvSharedLayerIdx
        )
    }

    /// Prepare les tokens d'entree pour la generation
    public func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int? = nil) throws -> PrepareResult {
        let promptTokens = input.text.tokens
        let promptCount = promptTokens.shape[0]

        guard promptCount > 0 else {
            let emptyToken = MLXArray(Int32(0))[0 ..< 0]
            return .tokens(.init(tokens: emptyToken))
        }

        return .tokens(input.text)
    }
}
