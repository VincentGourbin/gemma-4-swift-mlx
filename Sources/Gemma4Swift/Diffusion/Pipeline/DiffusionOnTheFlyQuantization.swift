// Quantification a la volee post-chargement pour DiffusionGemma.
//
// Wrapper minimal autour de Gemma4OnTheFlyQuantization qui prend
// directement un Module (DiffusionGemmaForBlockDiffusion) au lieu
// d'un LanguageModel.
//
// Cible 4-bit : modele passe de 48 Go bf16 a ~14 Go, et chaque
// forward decoder devient 3-4x plus rapide (kernels MLX optimises
// pour 4-bit groupwise sur Apple Silicon).

import Foundation
import MLX
import MLXNN

public enum DiffusionOnTheFlyQuantization {

    public typealias Mode = Gemma4OnTheFlyQuantization.Mode

    /// Prefixes pour preserver les encoders multimodaux en bf16 (utile pour
    /// isoler l'impact qualite de la quantization sur le decoder texte).
    public static let multimodalEncoderPrefixes: [String] = [
        "encoder.vision_tower",
        "encoder.embed_vision",
    ]

    /// Applique la quantification au modele DiffusionGemma.
    /// - Parameters:
    ///   - model : `DiffusionGemmaForBlockDiffusion` (ou tout `Module`).
    ///   - bits  : 2, 3, 4, 6 ou 8.
    ///   - groupSize : 32 pour mxfp4/mxfp8, 64 par defaut affine.
    ///   - mode : affine | mxfp4 | mxfp8.
    ///   - excludedPathPrefixes : paths a NE PAS quantifier.
    /// - Returns: nombre de modules quantifies.
    @discardableResult
    public static func apply(
        to model: Module,
        bits: Int,
        groupSize: Int = 64,
        mode: Mode = .affine,
        excludedPathPrefixes: [String] = []
    ) -> Int {
        // mxfp4/mxfp8 imposent group_size=32
        var effectiveGroupSize = groupSize
        switch mode {
        case .mxfp4, .mxfp8:
            if effectiveGroupSize != 32 {
                let msg = "[DiffusionQuant] WARNING: \(mode.rawValue) requires group_size=32, " +
                    "overriding \(effectiveGroupSize) -> 32\n"
                FileHandle.standardError.write(Data(msg.utf8))
                effectiveGroupSize = 32
            }
        case .affine:
            break
        }

        var quantizedCount = 0
        let mlxMode = mode.mlxMode

        var skipped: [String] = []
        MLXNN.quantize(
            model: model,
            filter: { path, m -> (groupSize: Int, bits: Int, mode: QuantizationMode)? in
                for prefix in excludedPathPrefixes {
                    if path.hasPrefix(prefix) || path.contains(".\(prefix).") {
                        return nil
                    }
                }
                // MLX exige que la last_dim du weight soit divisible par group_size.
                // Pour DiffusionGemma c'est le cas pour le text model (last_dim=2816,
                // 1408, etc.) mais PAS pour vision_tower.mlp.down_proj qui a
                // weight=[1152, 4304] et 4304 % 64 != 0. On skip silencieusement.
                if let lin = m as? Linear {
                    let lastDim = lin.weight.shape.last ?? 0
                    if lastDim % effectiveGroupSize != 0 {
                        skipped.append("\(path) [last_dim=\(lastDim)]")
                        return nil
                    }
                }
                if let emb = m as? Embedding {
                    let lastDim = emb.weight.shape.last ?? 0
                    if lastDim % effectiveGroupSize != 0 {
                        skipped.append("\(path) [last_dim=\(lastDim)]")
                        return nil
                    }
                }
                if m is Linear || m is Embedding {
                    return (groupSize: effectiveGroupSize, bits: bits, mode: mlxMode)
                }
                return nil
            },
            apply: { layer, gs, b, qmode in
                quantizedCount += 1
                return quantizeSingle(layer: layer, groupSize: gs, bits: b, mode: qmode)
            }
        )
        if !skipped.isEmpty {
            let msg = "[DiffusionQuant] \(skipped.count) modules skip (last_dim non divisible par \(effectiveGroupSize)) :\n"
                + skipped.prefix(5).map { "  - \($0)" }.joined(separator: "\n")
                + (skipped.count > 5 ? "\n  ... +\(skipped.count - 5)" : "")
                + "\n"
            FileHandle.standardError.write(Data(msg.utf8))
        }

        // Materialise les nouveaux poids quantifies
        eval(model)
        // Libere les anciens weights bf16 maintenus en cache MLX
        MLX.Memory.clearCache()
        return quantizedCount
    }

    // MARK: - Mixed Precision Quantization
    //
    // Pattern issu de Q-DiT (CVPR 2025) et ViDiT-Q (ICLR 2025) : les premieres
    // et dernieres couches du transformer sont sensibles a la quantization,
    // les couches du milieu tolerent du 4-bit aggressif.
    //
    // Pour DiffusionGemma 30 layers x 2 (encoder + decoder) :
    //   - layers {0..3} U {26..29} en 8-bit (high precision)
    //   - layers {4..25} en 4-bit (low precision)
    //   - embed_tokens en 8-bit (vocabulaire critique)
    //   - self_conditioning en 8-bit (modulation soft signals)
    //   - vision_tower en bf16 (skip, sensible et petit)

    public struct MixedPrecisionConfig: Sendable {
        public var highPrecisionLayers: Set<Int>
        public var highPrecisionBits: Int
        public var lowPrecisionBits: Int
        public var groupSize: Int

        /// Si true, embed_tokens + self_conditioning + lm_head sont quantizes
        /// en `highPrecisionBits`. Sinon ils restent en bf16.
        public var quantizeSensitiveAtHighPrecision: Bool

        public init(
            highPrecisionLayers: Set<Int>,
            highPrecisionBits: Int = 8,
            lowPrecisionBits: Int = 4,
            groupSize: Int = 64,
            quantizeSensitiveAtHighPrecision: Bool = true
        ) {
            self.highPrecisionLayers = highPrecisionLayers
            self.highPrecisionBits = highPrecisionBits
            self.lowPrecisionBits = lowPrecisionBits
            self.groupSize = groupSize
            self.quantizeSensitiveAtHighPrecision = quantizeSensitiveAtHighPrecision
        }

        /// Default : 4 premiers + 4 derniers en 8-bit, le reste en 4-bit
        /// (sur 30 layers). Cible empirique Q-DiT/ViDiT-Q.
        public static let `default` = MixedPrecisionConfig(
            highPrecisionLayers: Set(0...3).union(Set(26...29)),
            highPrecisionBits: 8,
            lowPrecisionBits: 4
        )

        /// Conservative : 6 premiers + 6 derniers en 8-bit (qualite max).
        public static let conservative = MixedPrecisionConfig(
            highPrecisionLayers: Set(0...5).union(Set(24...29)),
            highPrecisionBits: 8,
            lowPrecisionBits: 4
        )

        /// Aggressive : seulement 2 premiers + 2 derniers en 8-bit (RAM min).
        public static let aggressive = MixedPrecisionConfig(
            highPrecisionLayers: Set(0...1).union(Set(28...29)),
            highPrecisionBits: 8,
            lowPrecisionBits: 4
        )
    }

    /// Stats retournees par applyMixedPrecision pour reporting.
    public struct MixedPrecisionStats {
        public var quantizedHigh: Int = 0
        public var quantizedLow: Int = 0
        public var skipped: [String] = []
        public var totalQuantized: Int { quantizedHigh + quantizedLow }
    }

    /// Applique une quantization mixed precision sur DiffusionGemmaForBlockDiffusion.
    ///
    /// Strategy :
    ///   1. Parcourt encoder.language_model.layers et decoder.layers.
    ///   2. Pour chaque layer index, choisit highPrecisionBits si dans la liste
    ///      `highPrecisionLayers`, sinon lowPrecisionBits.
    ///   3. embed_tokens / self_conditioning quantizes en highPrecisionBits si
    ///      `quantizeSensitiveAtHighPrecision`, sinon laisses en bf16.
    ///   4. vision_tower et embed_vision : laisses en bf16 (skip explicite).
    ///
    /// - Parameter model: DiffusionGemmaForBlockDiffusion
    /// - Returns: stats de quantization
    @discardableResult
    public static func applyMixedPrecision(
        to model: DiffusionGemmaForBlockDiffusion,
        config: MixedPrecisionConfig = .default
    ) -> MixedPrecisionStats {
        var stats = MixedPrecisionStats()

        // Helper : quantize un Module avec un certain nb de bits + filtre divisibilite
        func quantizeModule(_ module: Module, bits: Int, label: String) {
            MLXNN.quantize(
                model: module,
                filter: { path, m -> (groupSize: Int, bits: Int, mode: QuantizationMode)? in
                    if let lin = m as? Linear {
                        let lastDim = lin.weight.shape.last ?? 0
                        if lastDim % config.groupSize != 0 {
                            stats.skipped.append("\(label)/\(path) [last_dim=\(lastDim)]")
                            return nil
                        }
                    }
                    if let emb = m as? Embedding {
                        let lastDim = emb.weight.shape.last ?? 0
                        if lastDim % config.groupSize != 0 {
                            stats.skipped.append("\(label)/\(path) [last_dim=\(lastDim)]")
                            return nil
                        }
                    }
                    if m is Linear || m is Embedding {
                        return (groupSize: config.groupSize, bits: bits, mode: .affine)
                    }
                    return nil
                },
                apply: { layer, gs, b, qmode in
                    if b == config.highPrecisionBits {
                        stats.quantizedHigh += 1
                    } else {
                        stats.quantizedLow += 1
                    }
                    return quantizeSingle(layer: layer, groupSize: gs, bits: b, mode: qmode)
                }
            )
        }

        // 1) Encoder text layers
        let encoderTextLayers = model.encoder.languageModel.layers
        for (i, layer) in encoderTextLayers.enumerated() {
            let bits = config.highPrecisionLayers.contains(i) ? config.highPrecisionBits : config.lowPrecisionBits
            quantizeModule(layer, bits: bits, label: "encoder.language_model.layers.\(i)")
        }

        // 2) Decoder layers (memes index, partagent les poids tied avec encoder)
        let decoderLayers = model.decoder.layers
        for (i, layer) in decoderLayers.enumerated() {
            let bits = config.highPrecisionLayers.contains(i) ? config.highPrecisionBits : config.lowPrecisionBits
            quantizeModule(layer, bits: bits, label: "decoder.layers.\(i)")
        }

        // 3) Sensible : embed_tokens + self_conditioning en highPrecision si demande
        if config.quantizeSensitiveAtHighPrecision {
            quantizeModule(model.decoder.embedTokens, bits: config.highPrecisionBits, label: "decoder.embed_tokens")
            quantizeModule(model.encoder.languageModel.embedTokens, bits: config.highPrecisionBits, label: "encoder.embed_tokens")
            quantizeModule(model.decoder.selfConditioning, bits: config.highPrecisionBits, label: "decoder.self_conditioning")
        }

        // 4) vision_tower / embed_vision restent en bf16 (volontaire).

        // Materialise les nouveaux poids + clear cache pour liberer bf16
        eval(model)
        MLX.Memory.clearCache()

        return stats
    }
}

