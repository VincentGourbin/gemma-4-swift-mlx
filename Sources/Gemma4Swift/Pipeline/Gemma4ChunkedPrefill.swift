// Prefill par tranches commun aux wrappers Gemma 4 (K-13, P-01)

import MLX
import MLXLMCommon

/// Fait avancer le cache sur toutes les positions du prompt **sauf la derniere**,
/// par tranches de `step`, puis laisse la derniere position au `TokenIterator`.
///
/// Pourquoi : les wrappers rendaient le prompt entier au `TokenIterator`, qui
/// l'envoyait en une passe. Le head lie (vocabulaire de 262 144) tournait alors sur
/// toutes les positions (~2,1 Go de logits bf16 pour 4 000 jetons) avant d'en garder
/// une, et `prefillStepSize` etait ignore. Ici, les logits d'une tranche ne sont
/// jamais lus : MLX ne les calcule pas, ni les couches a KV partage (elles n'ecrivent
/// pas le cache). Seul le dernier jeton passe par le head.
enum Gemma4ChunkedPrefill {

    /// - Parameters:
    ///   - count: nombre de positions du prompt
    ///   - step: taille de tranche (`GenerateParameters.prefillStepSize`, 512 par defaut)
    ///   - cache: caches mis a jour par `forward`
    ///   - forward: fait avancer le modele sur une plage de positions (sortie ignoree)
    static func run(
        count: Int, step: Int, cache: [KVCache], forward: (Range<Int>) -> Void
    ) {
        let end = count - 1
        guard end > 0 else { return }
        let step = max(1, step)
        var start = 0
        while start < end {
            let stop = min(start + step, end)
            forward(start ..< stop)
            // Le CPU construit la tranche suivante pendant que le GPU evalue celle-ci.
            asyncEval(cache)
            start = stop
        }
        eval(cache)
    }
}
