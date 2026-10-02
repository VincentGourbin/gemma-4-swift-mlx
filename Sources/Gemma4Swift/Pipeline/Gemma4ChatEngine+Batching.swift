// Mode lot du moteur de conversation (K-41) : les requetes texte qui arrivent ensemble sont
// decodees dans le meme forward (`Gemma4BatchGeneration`). Desactive par defaut.
//
// Ce que le lot ne fait pas : reutilisation de prefixe de conversation (K-19), images,
// n-gramme, budget de pensee. Ces requetes passent par le chemin standard.

import Foundation
import MLX
import MLXLMCommon

/// Requete en attente d'un lot : sa sortie, son annulation et le signal de fin.
final class Gemma4PendingBatchRequest: @unchecked Sendable {
    let messages: [Gemma4ChatMessage]
    let tools: [[String: any Sendable]]
    let options: Gemma4ChatOptions
    let continuation: AsyncThrowingStream<Gemma4ChatEvent, Error>.Continuation
    private let lock = NSLock()
    private var cancelled = false
    private var finished = false
    private var waiter: CheckedContinuation<Void, Never>?

    init(messages: [Gemma4ChatMessage], tools: [[String: any Sendable]], options: Gemma4ChatOptions,
         continuation: AsyncThrowingStream<Gemma4ChatEvent, Error>.Continuation) {
        self.messages = messages
        self.tools = tools
        self.options = options
        self.continuation = continuation
    }

    var isCancelled: Bool { lock.withLock { cancelled } }
    func cancel() { lock.withLock { cancelled = true } }

    /// Fin de la requete (succes ou erreur) : debloque `waitUntilFinished`.
    func complete() {
        let resume = lock.withLock { () -> CheckedContinuation<Void, Never>? in
            finished = true
            defer { waiter = nil }
            return waiter
        }
        resume?.resume()
    }

    func wait() async {
        await withCheckedContinuation { c in
            let done = lock.withLock { () -> Bool in
                if finished { return true }
                waiter = c
                return false
            }
            if done { c.resume() }
        }
    }
}

extension Gemma4ChatEngine {

    /// Requete eligible au lot ?
    nonisolated static func batchable(messages: [Gemma4ChatMessage], options: Gemma4ChatOptions) -> Bool {
        messages.allSatisfy(\.images.isEmpty) && options.noRepeatNGramSize == nil
            && options.maxThinkingTokens == nil
    }

    /// Met la requete en file de lot et rend son `Gemma4ChatRun`.
    func enqueueBatched(
        messages: [Gemma4ChatMessage], tools: [[String: any Sendable]], options: Gemma4ChatOptions
    ) -> Gemma4ChatRun {
        let (events, continuation) = AsyncThrowingStream<Gemma4ChatEvent, Error>.makeStream()
        let pending = Gemma4PendingBatchRequest(
            messages: messages, tools: tools, options: options, continuation: continuation)
        let task = Task {
            await withTaskCancellationHandler {
                await pending.wait()
            } onCancel: {
                pending.cancel()
            }
        }
        continuation.onTermination = { _ in pending.cancel() }
        batchQueue.append(pending)
        if batchRunner == nil {
            batchRunner = Task { await self.drainBatches() }
        }
        return Gemma4ChatRun(events: events, task: task)
    }

    /// Lance les lots tant qu'il reste des requetes : fenetre d'attente, puis jusqu'a
    /// `maxBatch` requetes dans un meme forward.
    func drainBatches() async {
        while !batchQueue.isEmpty {
            if let window = batchWindow { try? await Task.sleep(for: window) }
            let take = min(batchMaxSize, batchQueue.count)
            let batch = Array(batchQueue.prefix(take))
            batchQueue.removeFirst(take)
            await runBatch(batch)
        }
        batchRunner = nil
    }

    private func runBatch(_ batch: [Gemma4PendingBatchRequest]) async {
        let live = batch.filter { !$0.isCancelled }
        for request in batch where request.isCancelled {
            request.continuation.finish()
            request.complete()
        }
        guard !live.isEmpty else { return }
        do { try Gemma4ComputeGate.shared.beginInference() } catch {
            for request in live {
                request.continuation.finish(throwing: error)
                request.complete()
            }
            return
        }
        defer { Gemma4ComputeGate.shared.endInference() }
        let profile = profile
        do {
            try await container.perform { context in
                try Self.runBatch(live, profile: profile, context: context)
            }
        } catch {
            for request in live {
                request.continuation.finish(throwing: error)
                request.complete()
            }
            return
        }
        if profile?.clearCacheAfterAnswer == true { Memory.clearCache() }
        for request in live {
            request.continuation.finish()
            request.complete()
        }
    }

    private static func runBatch(
        _ batch: [Gemma4PendingBatchRequest], profile: Gemma4ReferenceProfile?, context: ModelContext
    ) throws {
        let beacon = RuntimeBeacon.begin(task: "generate", model: RuntimeBeacon.modelName(context.configuration))
        defer { beacon?.end() }
        beacon?.update(phase: "batch-prefill", step: batch.count, totalSteps: batch.count)
        var rows: [Gemma4PendingBatchRequest] = []
        var requests: [Gemma4BatchGeneration.Request] = []
        var promptCounts: [Int] = []
        for request in batch {
            do {
                let ids = try promptIds(
                    messages: request.messages, tools: request.tools,
                    enableThinking: request.options.enableThinking, tokenizer: context.tokenizer)
                rows.append(request)
                promptCounts.append(ids.count)
                requests.append(.init(
                    ids: ids, maxTokens: request.options.maxTokens, temperature: request.options.temperature,
                    topP: request.options.topP, topK: request.options.topK))
            } catch {
                request.continuation.finish(throwing: error)
            }
        }
        guard !rows.isEmpty else { return }

        var decoders = rows.map { Gemma4TokenEventDecoder(tokenizer: context.tokenizer, tools: $0.tools) }
        var firstToken = Array(repeating: TimeInterval?.none, count: rows.count)
        Memory.peakMemory = 0
        let start = Date()
        let produced = try Gemma4BatchGeneration.run(model: context.model, requests: requests) { row, token in
            let request = rows[row]
            if request.isCancelled { return false }
            if firstToken[row] == nil { firstToken[row] = Date().timeIntervalSince(start) }
            if row == 0 { beacon?.update(phase: "batch-decode") }
            return decoders[row].consume(token, emit: { request.continuation.yield($0) })
        }
        let elapsed = Date().timeIntervalSince(start)
        for (row, request) in rows.enumerated() {
            let calls = decoders[row].finish(emit: { request.continuation.yield($0) })
            let finish: Gemma4ChatUsage.FinishReason
            if request.isCancelled {
                finish = .cancelled
            } else if calls > 0 || decoders[row].stoppedOnToolResponse {
                finish = .toolCalls
            } else if !produced[row].hitStopToken && produced[row].tokens >= request.options.maxTokens {
                finish = .length
            } else {
                finish = .stop
            }
            let decode = max(0, elapsed - (firstToken[row] ?? 0))
            request.continuation.yield(.done(Gemma4ChatUsage(
                promptTokens: promptCounts[row], cachedPromptTokens: 0, completionTokens: produced[row].tokens,
                prefillSeconds: firstToken[row] ?? 0,
                promptTokensPerSecond: (firstToken[row] ?? 0) > 0 ? Double(promptCounts[row]) / firstToken[row]! : 0,
                tokensPerSecond: decode > 0 ? Double(produced[row].tokens) / decode : 0,
                timeToFirstToken: firstToken[row], peakMemoryBytes: Memory.peakMemory,
                finishReason: finish)))
        }
    }
}
