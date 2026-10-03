// Prefill par tranches commun aux wrappers Gemma 4 (K-13, P-01)

import MLX
import MLXLMCommon

/// Fait avancer le cache sur toutes les positions du prompt **sauf la derniere**,
/// par tranches, puis laisse la derniere position au `TokenIterator`.
///
/// Pourquoi : les wrappers rendaient le prompt entier au `TokenIterator`, qui
/// l'envoyait en une passe. Le head lie (vocabulaire de 262 144) tournait alors sur
/// toutes les positions (~2,1 Go de logits bf16 pour 4 000 jetons) avant d'en garder
/// une, et la taille de pas etait ignoree. Ici, les logits d'une tranche ne sont
/// jamais lus : MLX ne les calcule pas, ni les couches a KV partage (elles n'ecrivent
/// pas le cache). Seul le dernier jeton passe par le head.
///
/// La boucle est celle de mlx-swift-lm (`PrefillParameters.forEachChunk`, 3.32) :
/// annulation entre tranches, pool d'autorelease, progression par tranche, decoupage
/// `balanced` par defaut. Le terminal `(total, total)` est emis par le `TokenIterator`.
enum Gemma4ChunkedPrefill {

    /// - Parameters:
    ///   - count: nombre de positions du prompt
    ///   - prefill: pas (512 par defaut), decoupage et progression
    ///   - cache: caches mis a jour par `forward`
    ///   - forward: fait avancer le modele sur une plage de positions (sortie ignoree)
    static func run(
        count: Int, prefill: PrefillParameters, cache: [KVCache], forward: (Range<Int>) -> Void
    ) throws {
        guard count > 1 else { return }
        var prefill = prefill
        // `unchunked` rendrait tout le prompt au TokenIterator, qui le re-embarquerait sans
        // les medias : on le traite comme une seule tranche (toutes positions sauf la derniere).
        if case .unchunked = prefill.chunking {
            prefill.chunking = .remainder
            prefill.stepSize = count - 1
        }
        _ = try prefill.forEachChunk(total: count, reserving: 1) { range in
            forward(range)
            // Le CPU construit la tranche suivante pendant que le GPU evalue celle-ci.
            asyncEval(cache)
        }
        eval(cache)
    }
}
