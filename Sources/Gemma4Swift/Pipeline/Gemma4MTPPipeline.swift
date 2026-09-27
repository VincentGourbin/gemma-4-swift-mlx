// Pipeline MTP (Multi-Token Prediction / Speculative Decoding) pour Gemma 4 Assistant.
//
// Boucle d'un round:
//   1. setSharedKV depuis cache.state (apres prefill ou verify precedent)
//   2. drafter.draftBlock(lastBonus, lastHidden, blockSize) -> [B, blockSize-1] drafts
//   3. Target verify: forward sur [bonus | drafts] (blockSize tokens, en parallele)
//   4. Walk: accepter les drafts jusqu'a la 1ere divergence + correction du target
//   5. Yield les tokens emis
//   6. Si drafts rejetes: trim cache pour les positions invalides
//   7. Update bonus/hidden/sharedKV pour le round suivant
//
// Output: AsyncThrowingStream<String, Error> identique au contrat ChatSession.
// Le user voit les tokens en micro-rafales de 1 a blockSize tokens.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public actor Gemma4MTPPipeline {

    public let drafter: Gemma4AssistantDraftModel

    nonisolated private let target: ModelContainer

    public init(target: ModelContainer, drafter: Gemma4AssistantDraftModel) {
        self.target = target
        self.drafter = drafter
    }

    /// Genere une reponse en streaming MTP, equivalent a `ChatSession.streamResponse(to:)`.
    /// - Parameter sequentialVerify: si true, le verify est fait token-par-token au lieu
    ///   de en parallele. DIAGNOSTIC ONLY — perf catastrophique mais permet de tester si
    ///   la precision BF16 du parallel forward est la cause d'un faible taux d'acceptation.
    public func mtpStream(
        prompt: String,
        blockSize: Int = 4,
        maxTokens: Int = 256,
        useChatTemplate: Bool = true,
        sequentialVerify: Bool = false
    ) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try await self.runLoop(
                        prompt: prompt,
                        preTokenizedIds: nil,
                        blockSize: blockSize,
                        maxTokens: maxTokens,
                        useChatTemplate: useChatTemplate,
                        sequentialVerify: sequentialVerify,
                        continuation: continuation
                    )
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// Variante qui prend des tokens deja pre-tokenises (utile pour multi-turn chat
    /// ou le caller doit appliquer le chat template sur l'historique complet).
    public func mtpStreamFromTokens(
        tokenIds: [Int],
        blockSize: Int = 4,
        maxTokens: Int = 256,
        sequentialVerify: Bool = false
    ) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try await self.runLoop(
                        prompt: "",
                        preTokenizedIds: tokenIds,
                        blockSize: blockSize,
                        maxTokens: maxTokens,
                        useChatTemplate: false,
                        sequentialVerify: sequentialVerify,
                        continuation: continuation
                    )
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// Statistiques d'un run MTP — exposees apres la fin du stream.
    public struct Stats: Sendable {
        public var rounds: Int = 0
        public var totalDrafts: Int = 0
        public var acceptedDrafts: Int = 0
        public var emittedTokens: Int = 0

        public var acceptRate: Double {
            totalDrafts > 0 ? Double(acceptedDrafts) / Double(totalDrafts) : 0
        }
    }

    public private(set) var lastStats: Stats = Stats()

    private func runLoop(
        prompt: String,
        preTokenizedIds: [Int]?,
        blockSize: Int,
        maxTokens: Int,
        useChatTemplate: Bool,
        sequentialVerify: Bool,
        continuation: AsyncThrowingStream<String, Error>.Continuation
    ) async throws {
        precondition(blockSize >= 2, "blockSize doit etre >= 2 pour faire de la speculation")
        // K-9 : aucune inference pendant un entrainement (deadlock mlx-swift).
        try Gemma4ComputeGate.shared.beginInference()
        defer { Gemma4ComputeGate.shared.endInference() }

        nonisolated(unsafe) let drafterRef = drafter
        let bs = blockSize
        let maxTok = maxTokens
        let useTpl = useChatTemplate
        let userPrompt = prompt
        let seqVerify = sequentialVerify
        let preTokens = preTokenizedIds

        let stats = try await target.perform { context -> Stats in
            // Adapter: supporte Gemma4LLMModel (text-only) ET Gemma4MultimodalLLMModel
            // (image/video/audio via les pendingX deja set sur le model par le caller).
            let langModel: Gemma4LanguageModel
            let modelForward: (MLXArray, [any KVCache]) -> LanguageForwardOutput
            if let textOnly = context.model as? Gemma4LLMModel {
                langModel = textOnly.languageModel
                modelForward = { inputs, c in
                    textOnly.languageModel.forwardWithIntermediates(
                        inputs: inputs, cache: c.map { $0 as KVCache? }
                    )
                }
            } else if let multimodal = context.model as? Gemma4MultimodalLLMModel {
                langModel = multimodal.languageModel
                modelForward = { inputs, c in
                    multimodal.forwardWithIntermediates(inputs, cache: c)
                }
            } else {
                throw MTPError.unsupportedModel(String(describing: type(of: context.model)))
            }
            let textCfg = langModel.config

            var s = Stats()

            // 1) Tokenize (sauf si deja pre-tokenise par le caller, e.g. chat multi-turn)
            let promptIds: [Int]
            if let pre = preTokens {
                promptIds = pre
            } else if useTpl {
                let messages: [[String: String]] = [["role": "user", "content": userPrompt]]
                promptIds = try context.tokenizer.applyChatTemplate(messages: messages)
            } else {
                promptIds = try context.tokenizer.encode(text: userPrompt)
            }
            let inputArr = MLXArray(promptIds.map { Int32($0) }).reshaped(1, -1)

            // 2) Cache + prefill (multimodal: pendingX deja set sur le model par caller)
            // Caches glissants dimensionnes pour ne jamais tourner pendant le run : un
            // RotatingKVCache a 512 qui a tourne n'est plus trimmable, trimPromptCache ne
            // retirait alors rien, et les brouillons rejetes restaient dans le KV (sortie
            // fausse des que prompt + generation depassait la fenetre).
            let cache = langModel.makeCache(
                slidingCapacity: inputArr.dim(1) + maxTok + bs + 1)
            let prefillOut = modelForward(inputArr, cache)
            eval(prefillOut.logits, prefillOut.preNormHiddenStates)

            let promptLen = inputArr.dim(1)

            // 3) First bonus + last hidden
            var bonus = argMax(prefillOut.logits[0, promptLen - 1, 0...], axis: -1).item(Int32.self)
            var lastHidden = prefillOut.preNormHiddenStates[0..., (promptLen - 1) ..< promptLen, 0...]

            // Detokenisation incrementale : un caractere UTF-8 peut s'etaler sur
            // plusieurs tokens (byte-fallback) ; les decoder un par un le cassait.
            var detokenizer = Gemma4StreamingDetokenizer(tokenizer: context.tokenizer)

            // Yield le premier token (sauf si c'est deja un EOS)
            if isEOS(bonus, tokenizer: context.tokenizer) {
                return s
            }
            yieldToken(bonus, detokenizer: &detokenizer, continuation: continuation)
            s.emittedTokens = 1
            if s.emittedTokens >= maxTok {
                return s
            }

            // 4) Cache indices pour shared K/V (derniere couche concrete par type)
            let layerTypes = textCfg.resolvedLayerTypes
            let concreteTypes = Array(layerTypes.prefix(textCfg.firstKvSharedLayerIdx))
            guard let lastFullCacheIdx = concreteTypes.lastIndex(of: "full_attention"),
                  let lastSlidingCacheIdx = concreteTypes.lastIndex(of: "sliding_attention") else {
                throw MTPError.cacheLayoutInvalid
            }

            // 5) Bind drafter au target
            drafterRef.bind(target: langModel)

            // 6) Boucle MTP
            while s.emittedTokens < maxTok {
                // Le consommateur a lache le stream : onTermination a annule la Task.
                try Task.checkCancellation()
                // Lecture des K/V partages depuis le cache (etat valide jusqu'a cache.offset)
                let kvOffset = cache[lastFullCacheIdx].offset
                let sharedKV = extractSharedKV(
                    cache: cache,
                    fullIdx: lastFullCacheIdx,
                    slidingIdx: lastSlidingCacheIdx,
                    slidingWindow: textCfg.slidingWindow
                )

                drafterRef.setSharedKV(sharedKV, kvOffset: kvOffset)

                // Drafter genere bs-1 tokens
                let drafts = drafterRef.draftBlock(
                    lastBonus: bonus,
                    hidden: lastHidden,
                    blockSize: bs
                ) { logits in argMax(logits, axis: -1) }
                eval(drafts)

                // Materialiser les drafts en [Int32]
                var draftIds: [Int32] = []
                draftIds.reserveCapacity(drafts.dim(1))
                for i in 0 ..< drafts.dim(1) {
                    draftIds.append(drafts[0, i].item(Int32.self))
                }

                // Target verify: forward sur [bonus | drafts] de longueur bs
                let bonusArr = MLXArray([bonus])
                let draftsFlat = drafts.reshaped(-1)
                let verifyInput = concatenated([bonusArr, draftsFlat], axis: 0)
                    .asType(.int32)
                    .reshaped(1, bs)

                let verifyOut: LanguageForwardOutput
                if seqVerify {
                    // Sequential verify: bs forwards de L=1 chacun. Plus lent mais
                    // numeriquement bit-exact equivalent au sequential decode.
                    var allLogits: [MLXArray] = []
                    var allPreNormHidden: [MLXArray] = []
                    for i in 0 ..< bs {
                        let single = verifyInput[0..., i..<(i+1)]
                        let stepOut = modelForward(single, cache)
                        eval(stepOut.logits, stepOut.preNormHiddenStates)
                        allLogits.append(stepOut.logits)
                        allPreNormHidden.append(stepOut.preNormHiddenStates)
                    }
                    let logits = concatenated(allLogits, axis: 1)
                    let preNorm = concatenated(allPreNormHidden, axis: 1)
                    verifyOut = LanguageForwardOutput(
                        logits: logits,
                        hiddenStates: preNorm,  // unused
                        preNormHiddenStates: preNorm,
                        intermediates: []  // unused
                    )
                } else {
                    verifyOut = modelForward(verifyInput, cache)
                    eval(verifyOut.logits, verifyOut.preNormHiddenStates)
                }

                // Sample target sur chaque position [B, bs, vocab] -> [B, bs]
                let targetTokensArr = argMax(verifyOut.logits, axis: -1).reshaped(-1)
                eval(targetTokensArr)
                var targetIds: [Int32] = []
                targetIds.reserveCapacity(bs)
                for i in 0 ..< bs {
                    targetIds.append(targetTokensArr[i].item(Int32.self))
                }

                // Walk
                let budget = maxTok - s.emittedTokens
                let walkRes = SpeculativeWalk.walk(
                    drafts: draftIds, targets: targetIds, budget: budget
                )
                s.rounds += 1
                s.totalDrafts += draftIds.count
                s.acceptedDrafts += walkRes.accepted

                // Yield les tokens emis (stop avant EOS, ne yield pas l'EOS lui-meme)
                var sawEOS = false
                for tok in walkRes.newTokens {
                    if isEOS(tok, tokenizer: context.tokenizer) {
                        sawEOS = true
                        break
                    }
                    yieldToken(tok, detokenizer: &detokenizer, continuation: continuation)
                    s.emittedTokens += 1
                    if s.emittedTokens >= maxTok { break }
                }
                if sawEOS || s.emittedTokens >= maxTok { return s }

                // Rollback du cache: les positions de drafts rejetes (apres l'index accepted)
                // ont ete ecrites mais sont invalides. Trim de (bs - 1 - accepted).
                // Note: walkRes.accepted = nb drafts matches; on a ecrit bs slots dans cache,
                // garde les (accepted + 1) premiers (bonus + drafts acceptes), trim le reste.
                let toTrim = bs - 1 - walkRes.accepted
                if toTrim > 0 {
                    let trimmed = trimPromptCache(cache, numTokens: toTrim)
                    // Un retrait partiel laisserait des jetons faux dans le KV : erreur
                    // plutot que sortie silencieusement corrompue.
                    guard trimmed == toTrim else {
                        throw MTPError.cacheRollbackFailed(expected: toTrim, trimmed: trimmed)
                    }
                }

                // Update bonus + lastHidden pour le prochain round
                bonus = walkRes.newTokens.last ?? bonus
                let acceptedIdx = walkRes.accepted  // [0..bs-1]
                lastHidden = verifyOut.preNormHiddenStates[
                    0..., acceptedIdx ..< (acceptedIdx + 1), 0...
                ]
            }

            return s
        }

        await self.setStats(stats)
    }

    private func setStats(_ s: Stats) {
        self.lastStats = s
    }

    /// Extrait (full, sliding) K/V depuis le cache du target
    private nonisolated func extractSharedKV(
        cache: [any KVCache],
        fullIdx: Int,
        slidingIdx: Int,
        slidingWindow: Int
    ) -> SharedKVStates {
        let fullState = cache[fullIdx].state
        let slidingState = cache[slidingIdx].state
        precondition(fullState.count >= 2 && slidingState.count >= 2,
                     "Cache state inattendu (full=\(fullState.count), sliding=\(slidingState.count))")
        // Le cache glissant du run MTP ne tourne pas et garde tout l'historique : le
        // drafter ne voit que la fenetre, comme avec un cache glissant ordinaire.
        var slidingKeys = slidingState[0]
        var slidingValues = slidingState[1]
        let length = slidingKeys.dim(-2)
        if length > slidingWindow {
            slidingKeys = slidingKeys[.ellipsis, (length - slidingWindow)..., 0...]
            slidingValues = slidingValues[.ellipsis, (length - slidingWindow)..., 0...]
        }
        return [
            "full_attention": (keys: fullState[0], values: fullState[1]),
            "sliding_attention": (keys: slidingKeys, values: slidingValues),
        ]
    }

    private nonisolated func yieldToken(
        _ tokenId: Int32,
        detokenizer: inout Gemma4StreamingDetokenizer,
        continuation: AsyncThrowingStream<String, Error>.Continuation
    ) {
        if let piece = detokenizer.append(token: Int(tokenId)) {
            continuation.yield(piece)
        }
    }

    private nonisolated func isEOS(_ tokenId: Int32, tokenizer: any Tokenizer) -> Bool {
        // Gemma 4 utilise plusieurs tokens de fin: <eos>=1, <turn|>=106, <pad>=0
        // (cf. Gemma4Processor.eosTokenIds)
        if Gemma4Processor.eosTokenIds.contains(tokenId) {
            return true
        }
        if let eos = tokenizer.eosTokenId, Int(tokenId) == eos {
            return true
        }
        return false
    }

    public enum MTPError: LocalizedError {
        case unsupportedModel(String)
        case cacheLayoutInvalid
        case cacheRollbackFailed(expected: Int, trimmed: Int)

        public var errorDescription: String? {
            switch self {
            case .unsupportedModel(let t):
                return "MTP requires Gemma4LLMModel, got \(t)"
            case .cacheLayoutInvalid:
                return "Cannot find both full_attention and sliding_attention concrete layers — model layout incompatible"
            case .cacheRollbackFailed(let expected, let trimmed):
                return "MTP cache rollback removed \(trimmed) of \(expected) rejected draft positions — output would be corrupted"
            }
        }
    }
}
