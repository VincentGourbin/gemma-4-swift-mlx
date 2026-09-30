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
/// Perf (K-15) : avec `includeThinkingInWindow == true` (defaut), l'historique
/// reste sur le GPU et le masque des continuations interdites est calcule par
/// comparaison vectorisee des fenetres : aucune lecture CPU, le pipelining
/// `asyncEval` du `TokenIterator` est preserve. Avec `false`, l'automate de
/// canal a besoin de chaque token cote CPU : `didSample` force alors
/// l'evaluation du token a chaque etape (sync GPU→CPU), comme avant.
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

    /// Historique sur GPU (`[T]` int32), chemin sans synchronisation.
    private var deviceHistory: MLXArray?
    /// Longueur de `deviceHistory`, suivie cote CPU sans lire le GPU.
    private var deviceCount = 0

    /// Diagnostic : force l'historique CPU (ancien chemin, synchronisation par jeton)
    /// meme quand tous les tokens comptent — pour l'A/B de K-15.
    public let forceHostHistory: Bool

    /// Le chemin GPU n'a pas d'automate de canal : reserve au cas ou tous les
    /// tokens comptent.
    private var onDevice: Bool { includeThinkingInWindow && !forceHostHistory }

    /// - Parameters:
    ///   - ngramSize: taille du n-gramme, >= 1.
    ///   - includePromptInWindow: inclure le prompt dans la fenetre
    ///     d'interdiction (defaut `true`, parite HF).
    ///   - includeThinkingInWindow: inclure les tokens du canal de pensee dans
    ///     la fenetre d'interdiction (defaut `true`, comportement historique).
    public init(
        ngramSize: Int,
        includePromptInWindow: Bool = true,
        includeThinkingInWindow: Bool = true,
        forceHostHistory: Bool = false
    ) {
        precondition(ngramSize >= 1, "ngramSize doit etre >= 1 (recu \(ngramSize))")
        self.ngramSize = ngramSize
        self.includePromptInWindow = includePromptInWindow
        self.includeThinkingInWindow = includeThinkingInWindow
        self.forceHostHistory = forceHostHistory
    }

    public mutating func prompt(_ prompt: MLXArray) {
        guard includePromptInWindow else { return }
        if onDevice {
            appendOnDevice(prompt)
        } else {
            append(tokens(of: prompt))
        }
    }

    public func process(logits: MLXArray) -> MLXArray {
        if onDevice { return processOnDevice(logits: logits) }
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
        if onDevice {
            appendOnDevice(token)
        } else {
            append(tokens(of: token))
        }
    }

    // MARK: - Chemin GPU

    private mutating func appendOnDevice(_ tokens: MLXArray) {
        let flat = tokens.asType(.int32).reshaped(-1)
        deviceHistory = deviceHistory.map { concatenated([$0, flat]) } ?? flat
        deviceCount += flat.size
    }

    /// Meme regle que le chemin CPU : un token est interdit s'il a deja suivi
    /// les `n - 1` derniers tokens. Pour chacun des `W = T - n + 1` n-grammes de
    /// l'historique, on compare ses `n - 1` premiers tokens au prefixe courant ;
    /// les derniers tokens des n-grammes concordants sont interdits.
    private func processOnDevice(logits: MLXArray) -> MLXArray {
        let n = ngramSize
        let windows = deviceCount - n + 1
        guard let history = deviceHistory, windows > 0 else { return logits }

        var match = MLXArray.ones([windows], type: Bool.self)
        for j in 0 ..< (n - 1) {
            let current = history[deviceCount - (n - 1) + j]
            match = match .&& (history[j ..< (j + windows)] .== current)
        }
        let next = history[(n - 1) ..< (n - 1 + windows)]
        let vocab = logits.dim(-1)
        let hits = MLXArray.zeros([vocab], type: Int32.self).at[next].add(match.asType(.int32))
        let negInf = MLXArray(-Float.infinity).asType(logits.dtype)
        return MLX.where(hits .> 0, negInf, logits)
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
