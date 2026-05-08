// Port de mlx-vlm/speculative/drafters/gemma4_assistant/gemma4_assistant.py
//
// Drafter de speculative decoding pour Gemma 4. Le drafter est un mini-transformer
// (4 couches, hidden 256) dont TOUTES les couches sont kv_shared_only — elles
// consomment les K/V des dernieres couches concretes du target (full + sliding).
//
// Pipeline d'un round MTP:
//   1. setSharedKV(...) avec les K/V du target apres prefill
//   2. draftBlock(...) genere block_size-1 tokens autoregressifs cote drafter
//   3. Le target verifie [bonus | drafts] en un seul forward parallele
//   4. Walk + rollback du cache target en cas de divergence

import Foundation
import MLX
import MLXFast
import MLXNN

/// Inner du drafter — mirror minimal de Gemma4TextModel sans per-layer inputs ni softcap.
public class Gemma4AssistantDraftInner: Module {
    public let config: Gemma4TextConfig

    @ModuleInfo(key: "embed_tokens") public var embedTokens: Embedding
    @ModuleInfo public var layers: [Gemma4DecoderLayer]
    @ModuleInfo public var norm: RMSNorm

    public init(_ config: Gemma4TextConfig) {
        self.config = config
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize,
            dimensions: config.hiddenSize
        )
        self._layers.wrappedValue = (0 ..< config.numHiddenLayers).map { i in
            Gemma4DecoderLayer(config, layerIdx: i, kvSharedOnly: true)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }
}

/// Etat partage du target — K/V des dernieres couches concretes par layer_type.
public typealias SharedKVStates = [String: (keys: MLXArray, values: MLXArray)]

/// Drafter MTP Gemma 4 Assistant.
///
/// Architecture:
///   - `model: Gemma4AssistantDraftInner` (4-layer text transformer kv-shared-only)
///   - `pre_projection: Linear(2 * backbone, hidden)`  — projete concat(target_embed, target_hidden)
///   - `post_projection: Linear(hidden, backbone)`     — re-projete vers l'espace du target
///   - `masked_embedding: MaskedEmbedder?`             — LM head sparse a centroides (si use_ordered_embeddings)
///   - `lm_head: Linear?`                              — fallback dense si !tie_word_embeddings && !use_ordered_embeddings
public class Gemma4AssistantDraftModel: Module {
    public let config: Gemma4AssistantConfig

    @ModuleInfo public var model: Gemma4AssistantDraftInner
    @ModuleInfo(key: "pre_projection") public var preProjection: Linear
    @ModuleInfo(key: "post_projection") public var postProjection: Linear
    @ModuleInfo(key: "masked_embedding") public var maskedEmbedding: MaskedEmbedder?
    @ModuleInfo(key: "lm_head") public var lmHead: Linear?

    // Etat lie au target via bind(...)
    private var inputEmbed: Embedding?
    private var inputEmbedScale: Float = 1.0

    // Etat lie au round courant via setSharedKV(...)
    public private(set) var sharedKV: SharedKVStates?
    public private(set) var kvOffset: Int = 0
    public private(set) var position: Int = 0

    public init(_ config: Gemma4AssistantConfig) {
        self.config = config
        let textCfg = config.textConfig

        self._model.wrappedValue = Gemma4AssistantDraftInner(textCfg)
        self._preProjection.wrappedValue = Linear(
            2 * config.backboneHiddenSize, textCfg.hiddenSize, bias: false
        )
        self._postProjection.wrappedValue = Linear(
            textCfg.hiddenSize, config.backboneHiddenSize, bias: false
        )

        if config.useOrderedEmbeddings {
            self._maskedEmbedding.wrappedValue = MaskedEmbedder(config)
        } else {
            self._maskedEmbedding.wrappedValue = nil
        }

        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(
                textCfg.hiddenSize, textCfg.vocabSize, bias: false
            )
        } else {
            self._lmHead.wrappedValue = nil
        }

        super.init()
    }

    // MARK: - Binding au target

    /// Lie le drafter au target — extrait l'embedding + scale du target pour
    /// les utiliser dans `draftBlock` (les tokens sont embeddes dans l'espace du target
    /// avant concat avec son hidden state).
    @discardableResult
    public func bind(target: Gemma4LanguageModel) -> Self {
        self.inputEmbed = target.model.embedTokens
        self.inputEmbedScale = pow(Float(target.config.hiddenSize), 0.5)
        return self
    }

    /// Reset l'etat de round (sharedKV, position) — appele entre rounds MTP.
    public func reset() {
        self.sharedKV = nil
        self.kvOffset = 0
        self.position = 0
    }

    // MARK: - Etat partage du target

    /// Stocke les K/V partages du target pour le round courant.
    /// - Parameters:
    ///   - sharedKVStates: dict layer_type -> (keys, values), shapes [B, numHeads, S, headDim]
    ///   - kvOffset: longueur valide du cache target a ce point (= position d'ecriture du prochain token)
    ///   - position: position des queries du drafter (default = kvOffset)
    public func setSharedKV(
        _ sharedKVStates: SharedKVStates,
        kvOffset: Int,
        position: Int? = nil
    ) {
        self.sharedKV = sharedKVStates
        self.kvOffset = kvOffset
        self.position = position ?? kvOffset
    }

    // MARK: - Forward

    /// Forward d'un pas drafter.
    /// - Parameters:
    ///   - inputsEmbeds: `[B, L, 2 * backbone_hidden]` — concat(target_embed(token), target_hidden)
    ///   - sharedKVStates: K/V partages du target par layer_type
    ///   - position: position des queries (single int pour B=1 unbatched)
    /// - Returns: `(lastHidden: [B, L, backbone_hidden], logits: [B, L, vocab_size])`
    public func callAsFunction(
        inputsEmbeds: MLXArray,
        sharedKVStates: SharedKVStates,
        position: Int
    ) -> (lastHidden: MLXArray, logits: MLXArray) {
        let textCfg = config.textConfig

        // 1. Pre-projection: 2*backbone -> hidden
        var h = preProjection(inputsEmbeds)

        // 2. Forward a travers les layers, chaque couche consomme la K/V partagee
        //    correspondant a son layer_type.
        let layerTypes = textCfg.resolvedLayerTypes
        for (i, layer) in model.layers.enumerated() {
            let layerType = layerTypes[i]
            guard let kv = sharedKVStates[layerType] else {
                fatalError("sharedKVStates manque pour layer_type=\(layerType)")
            }

            // Pour le cas unbatched/no-padding (Phase 2 MVP), pas de masque additif.
            // Les masques additifs (sliding window long, batch padding) seront ajoutes
            // si necessaire en Phase 3.
            let mask: MLXFast.ScaledDotProductAttentionMaskMode = .none

            let (output, _, _) = layer(
                h,
                mask: mask,
                cache: nil,
                perLayerInput: nil,
                sharedKV: kv,
                sharedOffset: position
            )
            h = output
        }

        // 3. Norm finale
        h = model.norm(h)

        // 4. Post-projection: hidden -> backbone (pour servir d'input au prochain pas drafter)
        let lastHidden = postProjection(h)

        // 5. LM head
        let logits: MLXArray
        if let masked = maskedEmbedding {
            // Sparse softmax via MaskedEmbedder, utilise les embeddings du drafter
            logits = masked(h, lmHeadWeight: model.embedTokens.weight)
        } else if config.tieWordEmbeddings {
            logits = model.embedTokens.asLinear(h)
        } else if let head = lmHead {
            logits = head(h)
        } else {
            fatalError("Aucun LM head disponible (ni masked_embedding, ni tied, ni lm_head)")
        }

        return (lastHidden, logits)
    }

    // MARK: - Draft block (autoregressive K-step drafting)

    /// Genere `blockSize - 1` tokens autoregressifs cote drafter.
    ///
    /// Pre-conditions: `bind(target:)` et `setSharedKV(...)` doivent avoir ete appelees.
    ///
    /// - Parameters:
    ///   - lastBonus: dernier token accepte/sample par le target (debut du round).
    ///   - hidden: hidden state du target a la position du `lastBonus`, shape `[B, 1, backbone_hidden]`.
    ///   - blockSize: taille totale du bloc verify cote target (drafter genere `blockSize - 1` tokens).
    ///   - sampler: fonction logits `[B, L, vocab]` -> tokens `[B, L]` (typiquement argmax pour greedy).
    /// - Returns: tokens drafts `[B, blockSize - 1]` (Int32).
    public func draftBlock(
        lastBonus: Int32,
        hidden: MLXArray,
        blockSize: Int,
        sampler: (MLXArray) -> MLXArray
    ) -> MLXArray {
        guard let sharedKVStates = sharedKV else {
            fatalError("setSharedKV(...) doit etre appele avant draftBlock(...)")
        }
        guard let inputEmbed = inputEmbed else {
            fatalError("bind(target:) doit etre appele avant draftBlock(...)")
        }
        precondition(blockSize >= 2, "blockSize doit etre >= 2 (sinon aucun draft)")

        var tok = MLXArray([Int32(lastBonus)]).reshaped(1, 1)  // [B=1, 1]
        var hPrev = hidden  // [B, 1, backbone]
        var tokens: [MLXArray] = []

        for _ in 0 ..< (blockSize - 1) {
            // Embed le token courant dans l'espace du target (avec son scale)
            var tokEmbed = inputEmbed(tok)
            tokEmbed = tokEmbed * MLXArray(inputEmbedScale, dtype: tokEmbed.dtype)

            // Concat avec hidden du target -> [B, 1, 2*backbone]
            let inputsEmbeds = concatenated([tokEmbed, hPrev], axis: -1)

            let out = self(
                inputsEmbeds: inputsEmbeds,
                sharedKVStates: sharedKVStates,
                position: position
            )
            hPrev = out.lastHidden
            tok = sampler(out.logits)  // [B, 1]
            tokens.append(tok)
        }

        return concatenated(tokens, axis: 1)  // [B, blockSize - 1]
    }
}
