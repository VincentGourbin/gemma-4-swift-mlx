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
}

