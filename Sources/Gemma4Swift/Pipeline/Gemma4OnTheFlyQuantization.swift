// Quantification a la volee post-chargement.
//
// Utilite : benchmarker une famille bf16 sous plusieurs configurations
// (4/6/8-bit, group_size, mode) sans avoir a telecharger une variante
// pre-quantisee par config. Equivalent du `nn.quantize(...)` Python.

import Foundation
import MLX
import MLXNN
import MLXLMCommon

public enum Gemma4OnTheFlyQuantization {

    public enum Mode: String, Sendable {
        case affine
        case mxfp4
        case mxfp8

        var mlxMode: QuantizationMode {
            switch self {
            case .affine: return .affine
            case .mxfp4: return .mxfp4
            case .mxfp8: return .mxfp8
            }
        }
    }

    /// Defaut : pas d'exclusion. Tous les Linear/Embedding sont quantifies,
    /// y compris les encoders vision/audio. Cela correspond au comportement
    /// des checkpoints pre-quantifies mlx-community (qui quantifient aussi
    /// `vision_embedder.patch_dense`, `embed_vision.embedding_projection`, etc.).
    ///
    /// Si tu veux preserver les encoders multimodaux en pleine precision (par
    /// exemple pour isoler l'impact de la quantization sur le decoder texte
    /// uniquement), passe explicitement les prefixes via `excludedPathPrefixes`.
    public static let defaultExcludedPathPrefixes: [String] = []

    /// Prefixes des encoders multimodaux — utile pour "decoder texte seul".
    /// A passer EXPLICITEMENT a `apply(...)` si on veut ce comportement.
    public static let multimodalEncoderPrefixes: [String] = [
        "vision_embedder",
        "embed_vision",
        "embed_audio",
        "vision_tower",      // E2B/E4B SigLIP
        "audio_tower",       // E2B/E4B Conformer
    ]

    /// Cherche `Module` sur le `LanguageModel` exposed via le protocol.
    /// Tous nos modeles concrets (Gemma4LLMModel, *Multimodal*) sont des `Module`.
    /// - Returns: nil si le cast echoue.
    public static func asModule(_ model: LanguageModel) -> Module? {
        return model as? Module
    }

    /// Applique la quantification a la volee sur `model`.
    /// - Parameters:
    ///   - model : `LanguageModel` du `ModelContext` (cast vers Module en interne).
    ///   - bits  : 2, 3, 4, 6 ou 8 (selon le mode).
    ///   - groupSize : taille des groupes (defaut 64).
    ///   - mode : affine | mxfp4 | mxfp8.
    ///   - excludedPathPrefixes : paths qui ne sont PAS quantifies (par defaut on saute les encoders multimodaux).
    /// - Returns: nombre de modules quantifies.
    @discardableResult
    public static func apply(
        to model: LanguageModel,
        bits: Int,
        groupSize: Int = 64,
        mode: Mode = .affine,
        excludedPathPrefixes: [String] = defaultExcludedPathPrefixes
    ) -> Int {
        guard let module = asModule(model) else {
            print("[OnTheFlyQuantization] modele non-Module, skip")
            return 0
        }

        // mxfp4/mxfp8 imposent group_size=32 cote MLX. On corrige silencieusement.
        var effectiveGroupSize = groupSize
        switch mode {
        case .mxfp4, .mxfp8:
            if effectiveGroupSize != 32 {
                print("[OnTheFlyQuantization] \(mode.rawValue) requiert group_size=32, override")
                effectiveGroupSize = 32
            }
        case .affine:
            break
        }

        var quantizedCount = 0
        let mlxMode = mode.mlxMode

        // Filter prend (path, Module) -> tuple? : retourne le tuple pour quantizer, nil pour skip.
        MLXNN.quantize(
            model: module,
            filter: { path, m -> (groupSize: Int, bits: Int, mode: QuantizationMode)? in
                // Skip modules exclus
                for prefix in excludedPathPrefixes {
                    if path.hasPrefix(prefix) || path.contains(".\(prefix).") {
                        return nil
                    }
                }
                // Quantize Linear + Embedding seulement (defaut MLX)
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

        // Materialiser les nouveaux poids quantifies pour que la 1ere inference
        // ne paye pas le cout.
        eval(module)

        return quantizedCount
    }
}
