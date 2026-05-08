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
    public func mtpStream(
        prompt: String,
        blockSize: Int = 4,
        maxTokens: Int = 256,
        useChatTemplate: Bool = true
    ) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    try await self.runLoop(
                        prompt: prompt,
                        blockSize: blockSize,
                        maxTokens: maxTokens,
                        useChatTemplate: useChatTemplate,
                        continuation: continuation
                    )
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
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
        blockSize: Int,
        maxTokens: Int,
        useChatTemplate: Bool,
        continuation: AsyncThrowingStream<String, Error>.Continuation
    ) async throws {
        precondition(blockSize >= 2, "blockSize doit etre >= 2 pour faire de la speculation")

        nonisolated(unsafe) let drafterRef = drafter
        let bs = blockSize
        let maxTok = maxTokens
        let useTpl = useChatTemplate
        let userPrompt = prompt

        let stats = try await target.perform { context -> Stats in
            guard let llm = context.model as? Gemma4LLMModel else {
                throw MTPError.unsupportedModel(String(describing: type(of: context.model)))
            }
            let langModel = llm.languageModel
            let textCfg = langModel.config

            var s = Stats()

            // 1) Tokenize
            let promptIds: [Int]
            if useTpl {
                let messages: [[String: String]] = [["role": "user", "content": userPrompt]]
                promptIds = try context.tokenizer.applyChatTemplate(messages: messages)
            } else {
                promptIds = try context.tokenizer.encode(text: userPrompt)
            }
            let inputArr = MLXArray(promptIds.map { Int32($0) }).reshaped(1, -1)

            // 2) Cache + prefill
            let cache = langModel.makeCache()
            let cacheArr = cache.map { $0 as KVCache? }
            let prefillOut = langModel.forwardWithIntermediates(
                inputs: inputArr, cache: cacheArr
            )
            eval(prefillOut.logits, prefillOut.hiddenStates)

            let promptLen = inputArr.dim(1)

            // 3) First bonus + last hidden
            var bonus = argMax(prefillOut.logits[0, promptLen - 1, 0...], axis: -1).item(Int32.self)
            var lastHidden = prefillOut.hiddenStates[0..., (promptLen - 1) ..< promptLen, 0...]

            // Yield le premier token (sauf si c'est deja un EOS)
            if isEOS(bonus, tokenizer: context.tokenizer) {
                return s
            }
            try yieldToken(bonus, tokenizer: context.tokenizer, continuation: continuation)
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
                // Sharedeading des K/V depuis le cache (etat valide jusqu'a cache.offset)
                let kvOffset = cache[lastFullCacheIdx].offset
                let sharedKV = extractSharedKV(
                    cache: cache,
                    fullIdx: lastFullCacheIdx,
                    slidingIdx: lastSlidingCacheIdx
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

                let verifyOut = langModel.forwardWithIntermediates(
                    inputs: verifyInput, cache: cacheArr
                )
                eval(verifyOut.logits, verifyOut.hiddenStates)

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
                    try yieldToken(tok, tokenizer: context.tokenizer, continuation: continuation)
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
                    trimPromptCache(cache, numTokens: toTrim)
                }

                // Update bonus + lastHidden pour le prochain round
                bonus = walkRes.newTokens.last ?? bonus
                let acceptedIdx = walkRes.accepted  // [0..bs-1]
                lastHidden = verifyOut.hiddenStates[
                    0..., acceptedIdx ..< (acceptedIdx + 1), 0...
                ]
            }

            return s
        }

        await MainActor.run { [stats] in
            // Side-effect: mettre a jour les stats accessibles
            // Note: lastStats est isolated par l'actor, donc setter via une methode async
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
        slidingIdx: Int
    ) -> SharedKVStates {
        let fullState = cache[fullIdx].state
        let slidingState = cache[slidingIdx].state
        precondition(fullState.count >= 2 && slidingState.count >= 2,
                     "Cache state inattendu (full=\(fullState.count), sliding=\(slidingState.count))")
        return [
            "full_attention": (keys: fullState[0], values: fullState[1]),
            "sliding_attention": (keys: slidingState[0], values: slidingState[1]),
        ]
    }

    private nonisolated func yieldToken(
        _ tokenId: Int32,
        tokenizer: any Tokenizer,
        continuation: AsyncThrowingStream<String, Error>.Continuation
    ) throws {
        let piece = tokenizer.decode(tokenIds: [Int(tokenId)])
        continuation.yield(piece)
    }

    private nonisolated func isEOS(_ tokenId: Int32, tokenizer: any Tokenizer) -> Bool {
        // Gemma 4 utilise plusieurs tokens de fin: <eos>=1, <end_of_turn>=106, <pad>=0
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

        public var errorDescription: String? {
            switch self {
            case .unsupportedModel(let t):
                return "MTP requires Gemma4LLMModel, got \(t)"
            case .cacheLayoutInvalid:
                return "Cannot find both full_attention and sliding_attention concrete layers — model layout incompatible"
            }
        }
    }
}
