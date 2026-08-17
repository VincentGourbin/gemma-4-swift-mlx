// Blocage des n-grammes repetes — port de NoRepeatNGramLogitsProcessor (HF transformers)

import Foundation
import MLX
@preconcurrency import MLXLMCommon

/// `LogitProcessor` qui interdit de completer un n-gramme deja present dans la
/// sequence, a la maniere de `no_repeat_ngram_size` de HF transformers.
///
/// A chaque etape, on regarde les `n - 1` derniers tokens : tout token qui a
/// deja suivi ce prefixe dans l'historique voit son logit mis a `-inf`.
///
/// La fenetre d'interdiction est configurable via `includePromptInWindow` :
/// - `true` (defaut) — historique = `prompt + genere`, parite HF.
/// - `false` — historique = tokens generes seulement. Les boucles de
///   generation restent tuees, mais le modele peut citer le prompt
///   verbatim. Utile quand le prompt contient du texte que la reponse doit
///   recopier a l'identique (timeline, timestamps, identifiants) : en mode
///   HF, un tel passage s'interdit lui-meme et le decodage greedy contourne
///   en graphies degradees.
///
/// Utilise par le prompt enhancer LTX-2.5 (`no_repeat_ngram_size = 5`), ou le
/// decodage greedy de longues captions derive sans ce blocage (repetitions,
/// tokens aberrants).
///
/// Note perf : l'historique est maintenu cote CPU, donc `didSample` force
/// l'evaluation du token echantillonne a chaque etape (sync GPU→CPU). Le cout
/// est negligeable aux longueurs visees (quelques centaines de tokens), mais
/// cela desactive de fait le pipelining `asyncEval` du `TokenIterator`.
public struct NoRepeatNGramLogitProcessor: LogitProcessor {

    /// Taille du n-gramme bloque (`n`). `1` interdit tout token deja vu.
    public let ngramSize: Int

    /// Si `true` (defaut), les tokens du prompt alimentent l'historique
    /// (parite HF). Si `false`, seuls les tokens generes comptent.
    public let includePromptInWindow: Bool

    /// Historique pris en compte : `prompt + genere`, ou `genere` seul selon
    /// `includePromptInWindow`.
    private var history: [Int32] = []

    /// Prefixe de taille `n - 1` → tokens qui l'ont deja suivi.
    private var continuations: [[Int32]: Set<Int32>] = [:]

    /// - Parameters:
    ///   - ngramSize: taille du n-gramme, >= 1.
    ///   - includePromptInWindow: inclure le prompt dans la fenetre
    ///     d'interdiction (defaut `true`, parite HF).
    public init(ngramSize: Int, includePromptInWindow: Bool = true) {
        precondition(ngramSize >= 1, "ngramSize doit etre >= 1 (recu \(ngramSize))")
        self.ngramSize = ngramSize
        self.includePromptInWindow = includePromptInWindow
    }

    public mutating func prompt(_ prompt: MLXArray) {
        guard includePromptInWindow else { return }
        append(tokens(of: prompt))
    }

    public func process(logits: MLXArray) -> MLXArray {
        let prefix = Array(history.suffix(ngramSize - 1))
        // Prefixe incomplet (debut de sequence) : rien a bloquer.
        guard prefix.count == ngramSize - 1 else { return logits }
        guard let banned = continuations[prefix], !banned.isEmpty else { return logits }

        // `putAlong` exige des indices de meme rang que les logits ([1, vocab]
        // en generation, [vocab] si appele a plat).
        let flat = MLXArray(banned.sorted())
        let indices = logits.ndim == 1 ? flat : expandedDimensions(flat, axis: 0)
        let negInf = MLX.full(indices.shape, values: MLXArray(-Float.infinity))
            .asType(logits.dtype)
        return putAlong(logits, indices, values: negInf, axis: -1)
    }

    public mutating func didSample(token: MLXArray) {
        append(tokens(of: token))
    }

    // MARK: - Interne

    private func tokens(of array: MLXArray) -> [Int32] {
        array.asType(.int32).asArray(Int32.self)
    }

    private mutating func append(_ newTokens: [Int32]) {
        for token in newTokens {
            history.append(token)
            guard history.count >= ngramSize else { continue }
            let prefix = Array(history[(history.count - ngramSize) ..< (history.count - 1)])
            continuations[prefix, default: []].insert(token)
        }
    }
}
