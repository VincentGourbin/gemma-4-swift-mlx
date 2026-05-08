// Port de mlx-vlm/generate.py:_speculative_walk
//
// Marche speculative greedy: accepte les tokens du drafter jusqu'a la premiere
// divergence avec le sample du target, puis prend la prediction du target a ce
// point. Garantit que la sortie finale est strictement equivalente a la
// generation greedy du target (sans speculation).

import Foundation

public enum SpeculativeWalk {

    /// Resultat d'un walk speculative.
    public struct Result {
        /// Nombre de drafts acceptes (0..n_draft).
        public let accepted: Int
        /// Tokens emis ce round: drafts acceptes + token de correction/bonus du target.
        /// Toujours de longueur `accepted + 1` (avant truncation par budget).
        public let newTokens: [Int32]
    }

    /// Effectue le walk greedy.
    ///
    /// - Parameters:
    ///   - drafts: tokens proposes par le drafter, longueur `K-1`.
    ///   - targets: tokens predits par le target sur les positions `[bonus | drafts]`,
    ///     longueur `K`. `targets[i]` est la prediction du target apres avoir vu
    ///     les positions `0..i` du verify-input.
    ///   - budget: nombre maximum de tokens a retourner (typiquement `max_tokens - emitted`).
    /// - Returns: `(accepted, new_tokens)`. `new_tokens` a longueur `min(accepted + 1, budget)`.
    public static func walk(
        drafts: [Int32],
        targets: [Int32],
        budget: Int
    ) -> Result {
        precondition(targets.count == drafts.count + 1,
                     "targets doit faire drafts.count + 1 (verify-input = [bonus | drafts])")
        precondition(budget >= 0)

        // Trouver le premier i ou drafts[i] != targets[i]; si aucun, accepted = drafts.count
        var accepted = drafts.count
        for i in 0 ..< drafts.count {
            if drafts[i] != targets[i] {
                accepted = i
                break
            }
        }

        // Tokens emis: drafts acceptes + correction/bonus du target a la position de divergence
        var emitted = Array(drafts.prefix(accepted))
        emitted.append(targets[accepted])

        // Truncate au budget
        if emitted.count > budget {
            emitted = Array(emitted.prefix(budget))
        }

        return Result(accepted: accepted, newTokens: emitted)
    }
}
