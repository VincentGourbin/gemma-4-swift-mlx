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
/// Second axe, `includeThinkingInWindow` : quand le modele raisonne
/// (`enable_thinking`), les tokens du canal `<|channel>thought ... <channel|>`
/// sont des tokens generes ordinaires et alimentent donc la fenetre. Le modele
/// se voit alors interdire verbatim ce qu'il vient de poser dans son
/// raisonnement, et se rabat sur des paraphrases vagues. `false` les sort de
/// l'historique : ils sont generes normalement, simplement pas comptes comme
/// « deja ecrits ». La protection anti-boucle reste entiere sur le texte de
/// reponse, qui est ce qu'elle doit proteger.
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

    /// Si `true` (defaut), les tokens du canal de pensee alimentent
    /// l'historique comme les autres. Si `false`, ils en sont exclus.
    public let includeThinkingInWindow: Bool

    /// Historique pris en compte : `prompt + genere`, ou `genere` seul selon
    /// `includePromptInWindow`.
    private var history: [Int32] = []

    /// Prefixe de taille `n - 1` → tokens qui l'ont deja suivi.
    private var continuations: [[Int32]: Set<Int32>] = [:]

    /// Etat de l'automate de canal, utilise seulement quand
    /// `includeThinkingInWindow == false`.
    private enum ChannelState {
        /// Hors canal : les tokens comptent.
        case outside
        /// `<|channel>` vu, le token suivant nomme le canal.
        case awaitingName
        /// Dans `<|channel>thought` : les tokens ne comptent pas.
        case insideThought
    }

    private var channelState: ChannelState = .outside

    /// - Parameters:
    ///   - ngramSize: taille du n-gramme, >= 1.
    ///   - includePromptInWindow: inclure le prompt dans la fenetre
    ///     d'interdiction (defaut `true`, parite HF).
    ///   - includeThinkingInWindow: inclure les tokens du canal de pensee dans
    ///     la fenetre d'interdiction (defaut `true`, comportement historique).
    public init(
        ngramSize: Int,
        includePromptInWindow: Bool = true,
        includeThinkingInWindow: Bool = true
    ) {
        precondition(ngramSize >= 1, "ngramSize doit etre >= 1 (recu \(ngramSize))")
        self.ngramSize = ngramSize
        self.includePromptInWindow = includePromptInWindow
        self.includeThinkingInWindow = includeThinkingInWindow
    }

    public mutating func prompt(_ prompt: MLXArray) {
        guard includePromptInWindow else { return }
        append(tokens(of: prompt))
    }

    public func process(logits: MLXArray) -> MLXArray {
        // Dans le canal de pensee exclu, l'historique est gele : le prefixe
        // resterait fige sur les `n - 1` tokens d'avant l'ouverture du canal et
        // rebannirait leurs continuations a *chaque* pas du raisonnement, sur
        // des centaines de tokens sans rapport avec la position courante. Le
        // contrat est « genere normalement » : on ne bloque rien.
        guard channelState != .insideThought else { return logits }

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
            if !includeThinkingInWindow, advanceChannel(with: token) { continue }
            history.append(token)
            guard history.count >= ngramSize else { continue }
            let prefix = Array(history[(history.count - ngramSize) ..< (history.count - 1)])
            continuations[prefix, default: []].insert(token)
        }
    }

    /// Automate de canal. Retourne `true` si le token doit etre exclu de
    /// l'historique — soit parce qu'il est du balisage de canal, soit parce
    /// qu'il appartient au canal de pensee.
    ///
    /// `<|think|>` n'ouvre volontairement **pas** l'etat « dans le canal » :
    /// le chat template l'emet au sommet du tour system, donc dans le *prompt*,
    /// et sans `<channel|>` en face. L'y faire ouvrir un canal exclurait tout
    /// le prompt de la fenetre. Il est seulement retire comme balisage.
    ///
    /// Un canal jamais referme (generation coupee par `maxTokens` en plein
    /// raisonnement) laisse l'automate dans `.insideThought` : le reste n'est
    /// pas compte. Le mode degrade est « pas de blocage », jamais « blocage sur
    /// du raisonnement ».
    private mutating func advanceChannel(with token: Int32) -> Bool {
        switch channelState {
        case .outside:
            if token == Gemma4Processor.channelStartTokenId {
                channelState = .awaitingName
                return true
            }
            // Delimiteur de fermeture orphelin, ou marqueur de thinking du
            // prompt : du balisage, jamais du texte a proteger.
            return token == Gemma4Processor.channelEndTokenId
                || token == Gemma4Processor.thinkTokenId

        case .awaitingName:
            if token == Gemma4Processor.thoughtChannelNameTokenId {
                channelState = .insideThought
                return true
            }
            channelState = .outside
            // Le nom du canal de reponse est du balisage, il ne compte pas.
            // Un nom inattendu, en revanche — ou un `<|channel>` egare en
            // pleine reponse — est du contenu : le compter est le choix
            // conservateur, sinon on affaiblit la protection anti-boucle
            // exactement la ou le modele part en vrille.
            return token == Gemma4Processor.responseChannelNameTokenId

        case .insideThought:
            if token == Gemma4Processor.channelEndTokenId {
                channelState = .outside
            }
            return true
        }
    }
}
