// Port de modular_diffusion_gemma.DiffusionGemmaSelfConditioning (ligne 608-641)
//
// Module de self-conditioning : prend les soft-embeddings du step precedent
// (= softmax(logits_prev) @ embed_tokens.weight) et les fusionne dans
// l'embedding d'entree du decoder.
//
// Architecture : SwiGLU MLP avec pre/post RMSNorm. Important :
//   - pre_norm : RMSNorm avec scale (poids appris)
//   - post_norm : RMSNorm **sans scale** (with_scale=False) -> [[RMSNormNoScale]]
//
// Python ref :
//   normed = pre_norm(self_conditioning_signal)
//   sc = down_proj(act(gate_proj(normed)) * up_proj(normed))
//   combined = inputs_embeds + sc
//   return post_norm(combined)

import Foundation
import MLX
import MLXNN

/// Module de self-conditioning de DiffusionGemma.
public class DiffusionGemmaSelfConditioning: Module {
    @ModuleInfo(key: "pre_norm") var preNorm: RMSNorm
    @ModuleInfo(key: "post_norm") var postNorm: RMSNormNoScale
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    public init(_ config: Gemma4TextConfig) {
        let h = config.hiddenSize
        let i = config.intermediateSize
        self._preNorm.wrappedValue = RMSNorm(dimensions: h, eps: config.rmsNormEps)
        self._postNorm.wrappedValue = RMSNormNoScale(eps: config.rmsNormEps)
        self._gateProj.wrappedValue = Linear(h, i, bias: false)
        self._upProj.wrappedValue = Linear(h, i, bias: false)
        self._downProj.wrappedValue = Linear(i, h, bias: false)
        super.init()
    }

    /// - Parameters:
    ///   - inputsEmbeds : `[B, T, H]`, embeddings du canvas pour le step courant.
    ///   - selfConditioningSignal : `[B, T, H]`, soft-embeddings issus du step precedent.
    ///     A zero pour le 1er step.
    /// - Returns: `[B, T, H]`, embeddings combines + normalises.
    public func callAsFunction(
        _ inputsEmbeds: MLXArray,
        selfConditioningSignal: MLXArray
    ) -> MLXArray {
        let normed = preNorm(selfConditioningSignal)
        let scSignal = downProj(geluApprox(gateProj(normed)) * upProj(normed))
        let combined = inputsEmbeds + scSignal
        return postNorm(combined)
    }

    /// Activation Gemma : gelu approximee (cf. config.hidden_activation = "gelu_pytorch_tanh").
    @inline(__always)
    private func geluApprox(_ x: MLXArray) -> MLXArray {
        // tanh approx : 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        // MLXNN expose `geluApproximate` qui fait deja ca.
        MLXNN.geluApproximate(x)
    }
}
