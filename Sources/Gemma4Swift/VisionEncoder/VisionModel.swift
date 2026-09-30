// Port de vision.py VisionModel — Pipeline vision complete

import Foundation
import MLX
import MLXNN

/// Encodeur vision Gemma 4 complet: patch embed → transformer → pool
public class VisionModel: Module {
    let config: Gemma4VisionConfig
    let patchSize: Int
    let poolingKernelSize: Int
    let defaultOutputLength: Int
    let maxPatches: Int

    @ModuleInfo(key: "patch_embedder") var patchEmbedder: VisionPatchEmbedder
    @ModuleInfo var encoder: VisionTransformerModel
    @ModuleInfo var pooler: VisionPooler

    // Standardisation optionnelle
    @ModuleInfo(key: "std_bias") var stdBias: MLXArray?
    @ModuleInfo(key: "std_scale") var stdScale: MLXArray?

    public init(_ config: Gemma4VisionConfig) {
        self.config = config
        self.patchSize = config.patchSize
        self.poolingKernelSize = config.poolingKernelSize
        self.defaultOutputLength = config.defaultOutputLength
        self.maxPatches = config.maxPatches

        self._patchEmbedder.wrappedValue = VisionPatchEmbedder(config)
        self._encoder.wrappedValue = VisionTransformerModel(config)
        self._pooler.wrappedValue = VisionPooler(config)

        if config.standardize {
            self._stdBias.wrappedValue = MLXArray.zeros([config.hiddenSize])
            self._stdScale.wrappedValue = MLXArray.ones([config.hiddenSize])
        }

        super.init()
    }

    /// Calcule les positions de patches et le masque de padding
    func patchPositions(for pixelValues: MLXArray) -> (MLXArray, MLXArray) {
        let B = pixelValues.dim(0)
        let H = pixelValues.dim(2)
        let W = pixelValues.dim(3)
        let pH = H / patchSize
        let pW = W / patchSize
        let numPatches = pH * pW
        let numPadding = maxPatches - numPatches

        // Grille de positions [numPatches, 2]
        var positions: [[Int32]] = []
        for y in 0 ..< pH {
            for x in 0 ..< pW {
                positions.append([Int32(x), Int32(y)])
            }
        }
        // Padding avec (-1, -1)
        for _ in 0 ..< numPadding {
            positions.append([-1, -1])
        }

        // [maxPatches, 2] → broadcast [B, maxPatches, 2]
        let posArray = MLXArray(positions.flatMap { $0 }).reshaped(maxPatches, 2)
        let batchPos = broadcast(expandedDimensions(posArray, axis: 0), to: [B, maxPatches, 2])

        // Padding mask: [B, maxPatches]
        var paddingMask = MLXArray.zeros([B, maxPatches], type: Bool.self)
        if numPadding > 0 {
            // Les derniers numPadding tokens sont du padding
            // On construit un masque: false pour real, true pour padding
            let mask = MLXArray((0 ..< maxPatches).map { $0 >= numPatches })
            paddingMask = broadcast(expandedDimensions(mask, axis: 0), to: [B, maxPatches])
        }

        return (batchPos, paddingMask)
    }

    /// Ancien chemin : completer a `maxPatches` (2 520) avec un masque `[B,1,L,L]`.
    /// Garde pour le test de parite (K-18) ; le chemin par defaut n'encode que les
    /// patches reels. Public pour la mesure A/B (`gemma4-cli bench --vision-padded`).
    public var padToMaxPatches = false

    public func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
        let B = pixelValues.dim(0)
        let H = pixelValues.dim(2)
        let W = pixelValues.dim(3)
        let numReal = (H / patchSize) * (W / patchSize)

        let (allPositions, allPadding) = patchPositions(for: pixelValues)
        let realPositions = allPositions[0..., 0 ..< numReal]
        let realPadding = allPadding[0..., 0 ..< numReal]

        // Embed les patches
        var inputsEmbeds = patchEmbedder(pixelValues: pixelValues, patchPositions: realPositions, paddingPositions: realPadding)

        let hiddenStates: MLXArray
        let poolPositions: MLXArray
        let poolPadding: MLXArray
        let numPadding = maxPatches - numReal
        if padToMaxPatches && numPadding > 0 {
            let padEmbeds = MLXArray.zeros([B, numPadding, config.hiddenSize], dtype: inputsEmbeds.dtype)
            inputsEmbeds = concatenated([inputsEmbeds, padEmbeds], axis: 1)

            // Masque d'attention bidirectionnel [B, 1, L, L]
            let validMask = logicalNot(allPadding)
            let attnMask2d = expandedDimensions(validMask, axis: 1) * expandedDimensions(validMask, axis: 2)
            let negInf = MLXArray(Float(-Float.infinity), dtype: inputsEmbeds.dtype)
            let zero = MLXArray(Float(0.0), dtype: inputsEmbeds.dtype)
            let attnMask = expandedDimensions(MLX.where(attnMask2d, zero, negInf), axis: 1)
            hiddenStates = encoder(inputsEmbeds, positions: allPositions, mask: attnMask)
            poolPositions = allPositions
            poolPadding = allPadding
        } else {
            // Patches reels seulement (K-18) : les requetes reelles n'attendaient deja que
            // les cles reelles (le masque mettait le padding a -inf), et les sorties du
            // padding etaient mises a zero par le pooler. Meme resultat aux arrondis pres,
            // sans les 2 520 x 2 520 scores (une frame video a 70 jetons = 630 patches).
            hiddenStates = encoder(inputsEmbeds, positions: realPositions, mask: nil)
            poolPositions = realPositions
            poolPadding = realPadding
        }

        // Pooling ; noyau fixe = pooling_kernel_size (sqrt(2520 / 280) avec le padding).
        let (pooled, _) = pooler(
            hiddenStates: hiddenStates, patchPositions: poolPositions,
            paddingPositions: poolPadding, kernelSize: poolingKernelSize
        )

        // Extraire les tokens valides (non-padding)
        // Pour simplifier: prendre les defaultOutputLength premiers tokens valides
        var output = pooled[0..., 0 ..< defaultOutputLength]

        if config.standardize, let bias = stdBias, let scale = stdScale {
            output = (output - bias) * scale
        }

        return output
    }
}

/// Conteneur pour les layers du transformer vision
public class VisionTransformerModel: Module {
    @ModuleInfo var layers: [VisionTransformerBlock]

    public init(_ config: Gemma4VisionConfig) {
        self._layers.wrappedValue = (0 ..< config.numHiddenLayers).map { _ in
            VisionTransformerBlock(config)
        }
        super.init()
    }

    public func callAsFunction(_ hiddenStates: MLXArray, positions: MLXArray, mask: MLXArray?) -> MLXArray {
        var h = hiddenStates
        for layer in layers {
            h = layer(h, positions: positions, mask: mask)
        }
        return h
    }
}
