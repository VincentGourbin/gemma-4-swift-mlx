// Conformance LLMModel pour integration avec mlx-swift-lm (ChatSession, ModelContainer, etc.)

import Foundation
import MLX
import MLXFast
import MLXNN
import MLXLMCommon
import MLXLLM

/// Modele Gemma 4 conforme au protocol LLMModel de mlx-swift-lm.
/// Permet l'utilisation via MLXLMCommon.loadModelContainer() et ChatSession.
public class Gemma4LLMModel: Module, LLMModel {
    @ModuleInfo(key: "language_model") public var languageModel: Gemma4LanguageModel

    let config: Gemma4TextConfig
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

    public func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        languageModel.makeCache()
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        WeightSanitizer.sanitize(weights: weights)
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
