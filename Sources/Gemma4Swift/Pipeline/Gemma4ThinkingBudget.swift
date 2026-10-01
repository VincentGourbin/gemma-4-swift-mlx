// Budget de pensee (K-42) : au-dela de N jetons dans le canal `<|channel>thought`, le
// prochain jeton est force a `<channel|>` et le modele passe a sa reponse.
//
// Mesure du 2026-10-01 (Claude Code sur 26B-A4B, pensee `adaptive`) : deux tours sur dix
// epuisaient les 8 192 jetons de sortie dans la pensee, sans repondre. Le plafond rend la
// main au modele pour la reponse au lieu de couper la generation.
//
// Tout l'etat vit sur le GPU (aucun `.item()` par jeton) : un etat de canal et un compteur
// mis a jour paresseusement dans `didSample`, un `which` dans `process`. Le chemin de
// generation garde son pipelining (`asyncEval`).

import Foundation
import MLX
import MLXLMCommon

public struct Gemma4ThinkingBudgetProcessor: LogitProcessor {
    /// Jetons de pensee autorises avant fermeture forcee du canal.
    public let budget: Int

    // 0 : hors canal ; 1 : `<|channel>` vu, nom attendu ; 2 : dans la pensee.
    private var state = MLXArray(Int32(0))
    private var count = MLXArray(Int32(0))

    public init(budget: Int) {
        precondition(budget >= 0, "budget de pensee negatif")
        self.budget = budget
    }

    public mutating func prompt(_ prompt: MLXArray) {
        state = MLXArray(Int32(0))
        count = MLXArray(Int32(0))
    }

    public func process(logits: MLXArray) -> MLXArray {
        let exhausted = logicalAnd(state .== Int32(2), count .>= Int32(budget))
        return which(exhausted, forcedRow.row(vocab: logits.dim(-1), dtype: logits.dtype), logits)
    }

    /// Ligne « seul `<channel|>` » construite une fois : la refaire a chaque jeton (un
    /// tableau de 262 144 indices cree sur le CPU puis copie) divisait le debit par 3.
    private let forcedRow = ForcedRow()

    private final class ForcedRow: @unchecked Sendable {
        private var cached: MLXArray?
        private let lock = NSLock()

        func row(vocab: Int, dtype: DType) -> MLXArray {
            lock.lock()
            defer { lock.unlock() }
            if let cached, cached.dim(-1) == vocab, cached.dtype == dtype { return cached }
            let isEnd = MLX.arange(vocab) .== Int(Gemma4Processor.channelEndTokenId)
            let row = which(isEnd, MLXArray(Float(0)), MLXArray(-Float.infinity)).asType(dtype)
            eval(row)
            cached = row
            return row
        }
    }

    public mutating func didSample(token: MLXArray) {
        let t = token.reshaped([]).asType(.int32)
        let outside = state .== Int32(0)
        let awaiting = state .== Int32(1)
        let inside = state .== Int32(2)
        let next = which(
            outside, which(t .== Gemma4Processor.channelStartTokenId, Int32(1), Int32(0)),
            which(
                awaiting, which(t .== Gemma4Processor.thoughtChannelNameTokenId, Int32(2), Int32(0)),
                which(t .== Gemma4Processor.channelEndTokenId, Int32(0), Int32(2))))
        // Compte les jetons de pensee : +1 a chaque jeton qui reste dans le canal.
        count = which(logicalAnd(inside, next .== Int32(2)), count + 1, which(next .== Int32(2), Int32(0), count))
        state = next
    }
}

/// Deux processeurs a la suite (ex. n-gramme ou penalites, puis budget de pensee).
public struct Gemma4ChainedLogitProcessor: LogitProcessor {
    public var first: LogitProcessor
    public var second: LogitProcessor

    public init(_ first: LogitProcessor, _ second: LogitProcessor) {
        self.first = first
        self.second = second
    }

    public mutating func prompt(_ prompt: MLXArray) {
        first.prompt(prompt)
        second.prompt(prompt)
    }

    public func process(logits: MLXArray) -> MLXArray {
        second.process(logits: first.process(logits: logits))
    }

    public mutating func didSample(token: MLXArray) {
        first.didSample(token: token)
        second.didSample(token: token)
    }
}
