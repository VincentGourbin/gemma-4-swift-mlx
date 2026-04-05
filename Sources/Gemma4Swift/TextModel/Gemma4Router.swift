// Port de language.py Router — Routage expert top-k pour MoE (26B-A4B)

import Foundation
import MLX
import MLXNN

/// Router MoE Gemma 4 : norm → scale → project → softmax → top-k → renormalize → per_expert_scale
public class Gemma4Router: Module {
    let numExperts: Int
    let topK: Int
    let rootSize: Float

    let norm: RMSNormNoScale
    @ModuleInfo var proj: Linear
    @ModuleInfo var scale: MLXArray
    @ModuleInfo(key: "per_expert_scale") var perExpertScale: MLXArray

    public init(_ config: Gemma4TextConfig) {
        self.numExperts = config.numExperts ?? 128
        self.topK = config.topKExperts ?? 8
        self.rootSize = pow(Float(config.hiddenSize), -0.5)

        self.norm = RMSNormNoScale(eps: config.rmsNormEps)
        self._proj.wrappedValue = Linear(config.hiddenSize, numExperts, bias: false)
        self._scale.wrappedValue = MLXArray.ones([config.hiddenSize])
        self._perExpertScale.wrappedValue = MLXArray.ones([numExperts])

        super.init()
    }

    /// Retourne (top_k_indices, top_k_weights)
    public func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        var h = norm(x)
        h = h * MLXArray(rootSize, dtype: h.dtype)
        h = h * scale

        let expertScores = proj(h)
        let routerProbs = softmax(expertScores, axis: -1)

        let topKIndices = MLX.argPartition(-expertScores, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]

        var topKWeights = MLX.takeAlong(routerProbs, topKIndices, axis: -1)
        topKWeights = topKWeights / MLX.sum(topKWeights, axis: -1, keepDims: true)
        topKWeights = topKWeights * perExpertScale[topKIndices]

        return (topKIndices, topKWeights)
    }
}
