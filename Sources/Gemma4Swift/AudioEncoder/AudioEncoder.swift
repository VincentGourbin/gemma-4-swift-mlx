// Port de audio.py AudioEncoder — Encodeur audio Conformer complet

import MLX
import MLXNN

/// Encodeur audio Gemma 4 base sur l'architecture Universal Speech Model.
/// SSCP → 12 ConformerBlocks → output projection
public class AudioEncoder: Module {
    let config: Gemma4AudioConfig
    let relPosEnc: AudioRelPositionalEncoding

    @ModuleInfo(key: "subsample_conv_projection") var subsampleConvProjection: SubSampleConvProjection
    @ModuleInfo var layers: [ConformerBlock]
    @ModuleInfo(key: "output_proj") var outputProj: Linear?

    public init(_ config: Gemma4AudioConfig) {
        self.config = config
        self.relPosEnc = AudioRelPositionalEncoding(config)

        self._subsampleConvProjection.wrappedValue = SubSampleConvProjection(config)
        self._layers.wrappedValue = (0 ..< config.numHiddenLayers).map { _ in
            ConformerBlock(config)
        }

        if let outputDims = config.outputProjDims {
            self._outputProj.wrappedValue = Linear(config.hiddenSize, outputDims, bias: true)
        }

        super.init()
    }

    /// Construit le masque causal+validite pour l'attention par chunks
    func buildCausalValidMask() -> MLXArray {
        let chunkSize = config.attentionChunkSize
        let maxFuture = config.attentionContextRight
        let maxPast = max(0, config.attentionContextLeft - 1)
        let upperDiag = maxPast + maxFuture
        let ctxSize = chunkSize + maxPast + maxFuture

        let lowerCausal = MLX.tril(MLXArray.ones([ctxSize, chunkSize])).T
        let upperCausal = MLX.tril(MLXArray.ones([chunkSize, ctxSize]), k: upperDiag)
        return (lowerCausal * upperCausal).asType(.bool)
    }

    public func callAsFunction(_ audioMel: MLXArray, audioMelMask: MLXArray) -> (MLXArray, MLXArray) {
        var (encodings, currentMask) = subsampleConvProjection(audioMel, mask: audioMelMask)
        let causalValidMask = buildCausalValidMask()
        let positionEmbeddings = relPosEnc(encodings)

        for block in layers {
            encodings = block(encodings, mask: currentMask, causalValidMask: causalValidMask, positionEmbeddings: positionEmbeddings)
        }

        if let proj = outputProj {
            encodings = proj(encodings)
        }

        // Ajuster le masque si la longueur a change
        if currentMask.dim(1) != encodings.dim(1) {
            currentMask = currentMask[0..., 0 ..< encodings.dim(1)]
        }

        // Zero les positions de padding
        encodings = MLX.where(expandedDimensions(currentMask, axis: -1), MLXArray(Float(0.0)), encodings)

        return (encodings, currentMask)
    }
}
