// Quantification a la volee post-chargement pour DiffusionGemma.
//
// Wrapper minimal autour de Gemma4OnTheFlyQuantization qui prend
// directement un Module (DiffusionGemmaForBlockDiffusion) au lieu
// d'un LanguageModel.
//
// Taille estimee (calcul, non mesuree) : ~50 Go bf16, ~27 Go en 8 bits,
// ~15 Go en 4 bits, experts MoE compris (voir DiffusionReferenceProfile).
// Avant le correctif D-01 (2026-09-28) les experts SwitchLinear n'etaient
// PAS quantifies : les mesures anterieures (docs/examples/
// diffusion-optim-phases.md) ne portaient que sur ~12 % des poids. Aucun
// gain de vitesse n'est etabli ; a mesurer avec `gemma4-cli bench-diffusion`.

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
        func quantizeTree(_ root: Module, pathPrefix: String) {
        MLXNN.quantize(
            model: root,
            filter: { relativePath, m -> (groupSize: Int, bits: Int, mode: QuantizationMode)? in
                let path = pathPrefix + relativePath
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
                // Experts MoE compris (SwitchLinear n'herite pas de Linear, D-01) ;
                // routeur en 8 bits sous 8 bits comme les packs mlx-community (D-03).
                guard m is Quantizable, !(m is Quantized) else { return nil }
                if path.hasSuffix("router.proj") && bits < 8 {
                    return (groupSize: effectiveGroupSize, bits: 8, mode: .affine)
                }
                return (groupSize: effectiveGroupSize, bits: bits, mode: mlxMode)
            },
            apply: { layer, gs, b, qmode in
                quantizedCount += 1
                return quantizeSingle(layer: layer, groupSize: gs, bits: b, mode: qmode)
            }
        )
        }

        // Encodeur et decodeur partagent leurs poids (D-02) : quantifier l'encodeur,
        // le materialiser, puis donner au decodeur les MEMES instances de modules ;
        // ensuite seulement le reste (self_conditioning, tours…), les modules deja
        // quantifies etant ignores. Voir `shareEncoderModules`.
        if let diffusion = model as? DiffusionGemmaForBlockDiffusion {
            let shared = sharedDecoderLeafPaths(diffusion)
            let encoder = diffusion.encoder.languageModel
            quantizeSharedLayerwise(diffusion, shared: shared) { i, layer in
                quantizeTree(layer, pathPrefix: "encoder.language_model.layers.\(i).")
            }
            quantizeTree(encoder, pathPrefix: "encoder.language_model.")
            eval(encoder)
            shareEncoderModules(diffusion, paths: shared)
        }
        quantizeTree(model, pathPrefix: "")
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

    public struct MixedPrecisionConfig: Sendable, Equatable {
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
                    // Experts MoE compris (D-01) ; routeur en 8 bits (D-03).
                    guard m is Quantizable, !(m is Quantized) else { return nil }
                    if path.hasSuffix("router.proj") && bits < 8 {
                        return (groupSize: config.groupSize, bits: 8, mode: .affine)
                    }
                    return (groupSize: config.groupSize, bits: bits, mode: .affine)
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

        // Chemins du decodeur qui partagent leurs poids avec l'encodeur (avant quantif).
        let shared = sharedDecoderLeafPaths(model)

        // 1) Encoder text layers, chacune reprise aussitot par le decodeur
        quantizeSharedLayerwise(model, shared: shared) { i, layer in
            let bits = config.highPrecisionLayers.contains(i) ? config.highPrecisionBits : config.lowPrecisionBits
            quantizeModule(layer, bits: bits, label: "encoder.language_model.layers.\(i)")
        }

        if config.quantizeSensitiveAtHighPrecision {
            quantizeModule(model.encoder.languageModel.embedTokens, bits: config.highPrecisionBits, label: "encoder.embed_tokens")
        }

        // 2) Le decodeur reprend les modules quantifies de l'encodeur (memes instances,
        //    meme precision par couche) : une seule copie en memoire (D-02).
        eval(model.encoder.languageModel)
        shareEncoderModules(model, paths: shared)

        // 3) Ce qui reste propre au decodeur (modules deja quantifies ignores).
        let decoderLayers = model.decoder.layers
        for (i, layer) in decoderLayers.enumerated() {
            let bits = config.highPrecisionLayers.contains(i) ? config.highPrecisionBits : config.lowPrecisionBits
            quantizeModule(layer, bits: bits, label: "decoder.layers.\(i)")
        }
        if config.quantizeSensitiveAtHighPrecision {
            quantizeModule(model.decoder.embedTokens, bits: config.highPrecisionBits, label: "decoder.embed_tokens")
            quantizeModule(model.decoder.selfConditioning, bits: config.highPrecisionBits, label: "decoder.self_conditioning")
        }

        // 4) vision_tower / embed_vision restent en bf16 (volontaire).

        // Materialise les nouveaux poids + clear cache pour liberer bf16
        eval(model)
        MLX.Memory.clearCache()

        return stats
    }

    /// Chemins des modules feuilles du decodeur partages avec l'encodeur. Contrat du
    /// sanitizer (`DiffusionWeightSanitizer`) : le meme tableau est insere sous
    /// `decoder.X` et `encoder.language_model.X` pour tout sauf `self_conditioning` et
    /// `layer_scalar` (qui n'est pas un module feuille). Garde-fou : memes cles, formes
    /// et dtypes. A appeler sur le modele bf16 charge, avant toute quantification.
    static func sharedDecoderLeafPaths(_ model: DiffusionGemmaForBlockDiffusion) -> Set<String> {
        let encoderLeaves = Dictionary(
            model.encoder.languageModel.leafModules().flattened(), uniquingKeysWith: { a, _ in a })
        var shared = Set<String>()
        for (path, module) in model.decoder.leafModules().flattened() where !path.hasPrefix("self_conditioning") {
            guard let source = encoderLeaves[path], type(of: source) == type(of: module) else { continue }
            let mine = module.parameters().flattened()
            let theirs = Dictionary(source.parameters().flattened(), uniquingKeysWith: { a, _ in a })
            let compatible = !mine.isEmpty && mine.count == theirs.count && mine.allSatisfy { key, array in
                theirs[key].map { $0.shape == array.shape && $0.dtype == array.dtype } ?? false
            }
            if compatible { shared.insert(path) }
        }
        return shared
    }

    /// Quantifie les couches texte de l'encodeur une a une et donne chacune au decodeur
    /// des qu'elle est evaluee : le bf16 de la couche n'a alors plus de reference et
    /// part. Quantifier tout l'encodeur avant de partager gardait le bf16 complet et
    /// tout le quantifie en meme temps (pic de chargement mesure le 2026-09-28 : 77 Go
    /// en 8 bits, 65 Go en 4 bits, pour 49 Go de bf16).
    static func quantizeSharedLayerwise(
        _ model: DiffusionGemmaForBlockDiffusion,
        shared: Set<String>,
        quantizeLayer: (Int, Module) -> Void
    ) {
        let decoderLayers = model.decoder.layers
        for (i, layer) in model.encoder.languageModel.layers.enumerated() where i < decoderLayers.count {
            quantizeLayer(i, layer)
            eval(layer)
            // Mise a jour de la couche elle-meme : `unflattened` sur "layers.i.*" seul ne
            // reconstruit pas le tableau `layers` (unexpectedStructure).
            let prefix = "layers.\(i)."
            let leaves = Dictionary(layer.leafModules().flattened(), uniquingKeysWith: { a, _ in a })
            let replacements = shared.filter { $0.hasPrefix(prefix) }.sorted().compactMap { path -> (String, Module)? in
                let relative = String(path.dropFirst(prefix.count))
                return leaves[relative].map { (relative, $0) }
            }
            if !replacements.isEmpty {
                decoderLayers[i].update(modules: ModuleChildren.unflattened(replacements))
            }
            MLX.Memory.clearCache()
        }
    }

    /// Donne au decodeur les instances de modules (quantifies) de l'encodeur pour les
    /// chemins partages. Partager les **modules**, et non les tableaux : relier les
    /// tableaux quantifies gardait les 49 Go de bf16 d'origine en memoire (mesure sur le
    /// vrai checkpoint le 2026-09-28 : 74,8 Go actifs en 8 bits ; avec les modules
    /// partages : 26,7 Go). A appeler apres avoir quantifie ET evalue l'encodeur.
    @discardableResult
    static func shareEncoderModules(_ model: DiffusionGemmaForBlockDiffusion, paths: Set<String>) -> Int {
        let encoderLeaves = Dictionary(
            model.encoder.languageModel.leafModules().flattened(), uniquingKeysWith: { a, _ in a })
        let replacements = paths.sorted().compactMap { path in encoderLeaves[path].map { (path, $0) } }
        if !replacements.isEmpty {
            model.decoder.update(modules: ModuleChildren.unflattened(replacements))
        }
        return replacements.count
    }
}

