// Moteur de conversation (K-36) : conversation OpenAI en entree, evenements types en
// sortie. Aucune dependance nouvelle : c'est la partie du serveur qui touche MLX, le
// paquet `Server/` n'en est qu'un adaptateur HTTP (audit-annexes-serveur.md § 2.3).
//
// Rendu par le `chat_template.jinja` du modele (outils, `enable_thinking`, tours
// `tool`), en ids directs + `strippingTemplateArtifacts` + expansion des images. Le
// canal de pensee est route d'apres les ids (`<|channel>` `thought` … `<channel|>`),
// les appels d'outils par le parseur amont `ToolCallFormat.gemma4`.

import CoreGraphics
import CryptoKit
import Foundation
import ImageIO
import MLX
@preconcurrency import MLXLMCommon

public struct Gemma4ToolCall: Sendable, Equatable {
    public let id: String
    public let name: String
    /// Arguments en objet JSON serialise.
    public let argumentsJSON: String

    public init(id: String, name: String, argumentsJSON: String) {
        self.id = id
        self.name = name
        self.argumentsJSON = argumentsJSON
    }
}

public struct Gemma4ChatMessage: Sendable {
    public enum Role: String, Sendable { case system, user, assistant, tool }

    public var role: Role
    public var content: String
    /// Images encodees (PNG, JPEG…), decodees en memoire. Tours `user` seulement.
    public var images: [Data]
    /// Appels d'outils d'un tour `assistant` passe.
    public var toolCalls: [Gemma4ToolCall]
    /// Tour `tool` : nom de l'outil et id de l'appel auxquels il repond.
    public var toolName: String?
    public var toolCallID: String?

    public init(
        role: Role, content: String, images: [Data] = [], toolCalls: [Gemma4ToolCall] = [],
        toolName: String? = nil, toolCallID: String? = nil
    ) {
        self.role = role
        self.content = content
        self.images = images
        self.toolCalls = toolCalls
        self.toolName = toolName
        self.toolCallID = toolCallID
    }
}

public struct Gemma4ChatOptions: Sendable, Equatable {
    public var maxTokens: Int
    public var temperature: Float
    public var topP: Float
    public var topK: Int
    /// Variable `enable_thinking` du gabarit : le modele raisonne dans le canal de pensee.
    public var enableThinking: Bool
    public var noRepeatNGramSize: Int?

    public init(
        maxTokens: Int = 1024, temperature: Float = 0.3, topP: Float = 0.95, topK: Int = 0,
        enableThinking: Bool = false, noRepeatNGramSize: Int? = nil
    ) {
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.enableThinking = enableThinking
        self.noRepeatNGramSize = noRepeatNGramSize
    }
}

public struct Gemma4ChatUsage: Sendable, Equatable {
    public enum FinishReason: String, Sendable { case stop, length, toolCalls = "tool_calls", cancelled }

    public let promptTokens: Int
    /// Jetons du prompt servis par l'instantane de conversation (K-19), non re-prefilles.
    public let cachedPromptTokens: Int
    public let completionTokens: Int
    public let prefillSeconds: Double
    public let promptTokensPerSecond: Double
    public let tokensPerSecond: Double
    public let timeToFirstToken: TimeInterval?
    public let peakMemoryBytes: Int
    public let finishReason: FinishReason

    public init(
        promptTokens: Int, cachedPromptTokens: Int = 0, completionTokens: Int, prefillSeconds: Double = 0,
        promptTokensPerSecond: Double = 0, tokensPerSecond: Double = 0,
        timeToFirstToken: TimeInterval? = nil, peakMemoryBytes: Int = 0, finishReason: FinishReason
    ) {
        self.promptTokens = promptTokens
        self.cachedPromptTokens = cachedPromptTokens
        self.completionTokens = completionTokens
        self.prefillSeconds = prefillSeconds
        self.promptTokensPerSecond = promptTokensPerSecond
        self.tokensPerSecond = tokensPerSecond
        self.timeToFirstToken = timeToFirstToken
        self.peakMemoryBytes = peakMemoryBytes
        self.finishReason = finishReason
    }
}

public enum Gemma4ChatEvent: Sendable {
    case reasoning(String)
    case text(String)
    case toolCall(Gemma4ToolCall)
    case done(Gemma4ChatUsage)
}

public enum Gemma4ChatEngineError: LocalizedError, Equatable {
    case emptyConversation
    case imagesNeedVision
    case imageOutsideUserTurn
    case invalidImage(Int)
    case markerInText(String)

    public var errorDescription: String? {
        switch self {
        case .emptyConversation: return "conversation vide"
        case .imagesNeedVision: return "images fournies mais le modele est charge sans vision"
        case .imageOutsideUserTurn: return "les images ne sont acceptees que dans un tour user"
        case .invalidImage(let index): return "image \(index) illisible"
        case .markerInText(let role):
            return "un tour \(role) contient un marqueur multimodal brut (<|image|>, <|audio|>…)"
        }
    }
}

/// Une generation en cours : ses evenements, son annulation, et sa fin **reelle** (le
/// calcul MLX termine et la porte K-9 rendue), distincte de la fin du flux cote client.
public struct Gemma4ChatRun: Sendable {
    public let events: AsyncThrowingStream<Gemma4ChatEvent, Error>
    let task: Task<Void, Never>

    public init(events: AsyncThrowingStream<Gemma4ChatEvent, Error>, task: Task<Void, Never>) {
        self.events = events
        self.task = task
    }

    public func cancel() { task.cancel() }
    public func waitUntilFinished() async { await task.value }
}

/// Ce dont un adaptateur (serveur HTTP) a besoin : testable avec un faux moteur.
public protocol Gemma4ChatBackend: Sendable {
    func start(
        messages: [Gemma4ChatMessage], tools: [[String: any Sendable]], options: Gemma4ChatOptions
    ) async -> Gemma4ChatRun
    /// Plafond de `max_tokens` (profil), s'il y en a un.
    var maxTokensCap: Int? { get async }
    /// Le modele accepte-t-il des images ?
    var acceptsImages: Bool { get async }
}

/// Instantane de conversation (K-19) : caches KV a la **fin du prompt** du tour
/// precedent (pas apres la generation : le gabarit retire la pensee des tours passes,
/// donc les jetons generes ne sont pas ceux du rendu suivant — piege 13), avec les ids
/// et les empreintes des images deja encodees.
final class ConversationStore: @unchecked Sendable {
    struct Snapshot {
        let ids: [Int]
        let caches: [any KVCache]
        let imageDigests: [Data]
        var bytes: Int { caches.reduce(0) { $0 + $1.state.reduce(0) { $0 + $1.nbytes } } }
    }

    private let lock = NSLock()
    /// Du plus ancien au plus recent (LRU). Plusieurs conversations (clients d'un serveur)
    /// alternent sans s'evincer tant que le budget tient (K-40).
    private var snapshots: [Snapshot] = []
    var capacity: Int
    var budgetBytes: Int

    init(capacity: Int = 8, budgetBytes: Int = 2 << 30) {
        self.capacity = capacity
        self.budgetBytes = budgetBytes
    }

    /// Le plus long instantane dont les ids sont un prefixe strict de `ids` et dont les
    /// images sont celles du debut de `digests` ; il redevient le plus recent.
    func bestPrefix(of ids: [Int], digests: [Data]) -> Snapshot? {
        lock.withLock {
            let candidates = snapshots.indices.filter {
                snapshots[$0].ids.count < ids.count && ids.starts(with: snapshots[$0].ids)
                    && digests.starts(with: snapshots[$0].imageDigests)
            }
            guard let best = candidates.max(by: { snapshots[$0].ids.count < snapshots[$1].ids.count }) else { return nil }
            let snapshot = snapshots.remove(at: best)
            snapshots.append(snapshot)
            return snapshot
        }
    }

    /// Ajoute un instantane ; celui qu'il prolonge (meme conversation) est remplace.
    func put(_ snapshot: Snapshot) {
        lock.withLock {
            snapshots.removeAll { snapshot.ids.starts(with: $0.ids) }
            snapshots.append(snapshot)
            var total = snapshots.reduce(0) { $0 + $1.bytes }
            while snapshots.count > max(1, capacity) || (total > budgetBytes && snapshots.count > 1) {
                total -= snapshots.removeFirst().bytes
            }
        }
    }

    func removeAll() { lock.withLock { snapshots.removeAll() } }
    var count: Int { lock.withLock { snapshots.count } }
}

public actor Gemma4ChatEngine: Gemma4ChatBackend {
    public let container: ModelContainer
    public let profile: Gemma4ReferenceProfile?
    /// Reutiliser le prefixe du tour precedent quand le nouveau prompt le prolonge
    /// strictement (K-19). Sans effet avec le n-gramme ni sur le 12B unified.
    public var reusesConversation = true
    let conversation = ConversationStore()

    public func setReusesConversation(_ value: Bool) {
        reusesConversation = value
        if !value { conversation.removeAll() }
    }

    /// Nombre de conversations gardees (LRU) et budget memoire de leurs caches.
    public func configureConversationCache(capacity: Int, budgetBytes: Int) {
        conversation.capacity = capacity
        conversation.budgetBytes = budgetBytes
    }

    /// Oublie les instantanes : le prochain tour re-prefille tout.
    public func resetConversation() { conversation.removeAll() }

    public init(container: ModelContainer, profile: Gemma4ReferenceProfile? = nil) {
        self.container = container
        self.profile = profile
    }

    /// Charge le modele (`Gemma4Registration.loadContainer`) et applique la politique
    /// memoire du profil.
    public static func load(
        from directory: URL, profile: Gemma4ReferenceProfile? = nil, audio: Bool = true
    ) async throws -> Gemma4ChatEngine {
        let container = try await Gemma4Registration.loadContainer(
            from: directory, multimodal: profile?.multimodal ?? true, audio: audio && (profile?.audio ?? true))
        if let profile {
            await container.perform {
                ($0.model as? Gemma4MultimodalLLMModel)?.releaseEncodersAfterPrefill = profile.releaseEncodersAfterPrefill
            }
        }
        profile?.applyGlobalPolicy()
        return Gemma4ChatEngine(container: container, profile: profile)
    }

    // MARK: - Rendu

    /// Ids du prompt : gabarit du modele (outils, `enable_thinking`), artefacts retires,
    /// chaque `<|image|>` developpe en `boi + image x N + eoi`. Pure : testable sans modele.
    public static func promptIds(
        messages: [Gemma4ChatMessage],
        tools: [[String: any Sendable]],
        enableThinking: Bool,
        tokenizer: any Tokenizer,
        imageSoftTokens: Int = Gemma4ImageProcessor.defaultMaxSoftTokens
    ) throws -> [Int] {
        guard !messages.isEmpty else { throw Gemma4ChatEngineError.emptyConversation }
        let markers = [
            Gemma4Processor.imageToken, "<|audio|>", "<|video|>",
        ]
        var rendered: [[String: any Sendable]] = []
        for message in messages {
            // Un marqueur tape dans le texte fausserait le compte du masked_scatter.
            if markers.contains(where: { message.content.contains($0) }) {
                throw Gemma4ChatEngineError.markerInText(message.role.rawValue)
            }
            if !message.images.isEmpty && message.role != .user {
                throw Gemma4ChatEngineError.imageOutsideUserTurn
            }
            var content = message.content
            if !message.images.isEmpty {
                content = Array(repeating: Gemma4Processor.imageToken, count: message.images.count)
                    .joined(separator: "\n") + "\n" + content
            }
            var entry: [String: any Sendable] = ["role": message.role.rawValue, "content": content]
            if !message.toolCalls.isEmpty {
                entry["tool_calls"] = message.toolCalls.map { call -> [String: any Sendable] in
                    let arguments = (try? JSONSerialization.jsonObject(
                        with: Data(call.argumentsJSON.utf8))) as? [String: any Sendable] ?? [:]
                    return ["id": call.id, "type": "function",
                            "function": ["name": call.name, "arguments": arguments] as [String: any Sendable]]
                }
            }
            if let name = message.toolName { entry["name"] = name }
            if let id = message.toolCallID { entry["tool_call_id"] = id }
            rendered.append(entry)
        }
        let ids = Gemma4Processor.strippingTemplateArtifacts(
            try tokenizer.applyChatTemplate(
                messages: rendered, tools: tools.isEmpty ? nil : tools,
                additionalContext: ["enable_thinking": enableThinking]))

        var expanded: [Int] = []
        expanded.reserveCapacity(ids.count)
        for id in ids {
            guard id == Int(Gemma4Processor.imageTokenId) else {
                expanded.append(id)
                continue
            }
            expanded.append(Int(Gemma4Processor.boiTokenId))
            expanded.append(contentsOf: repeatElement(Int(Gemma4Processor.imageTokenId), count: imageSoftTokens))
            expanded.append(Int(Gemma4Processor.eoiTokenId))
        }
        return expanded
    }

    // MARK: - Reponse

    public var maxTokensCap: Int? { nil }

    public var acceptsImages: Bool {
        get async { await container.perform { $0.model is Gemma4MultimodalLLMModel } }
    }

    /// Flux d'evenements ; arreter de l'iterer annule la generation.
    public func respond(
        to messages: [Gemma4ChatMessage],
        tools: [[String: any Sendable]] = [],
        options: Gemma4ChatOptions = .init()
    ) -> AsyncThrowingStream<Gemma4ChatEvent, Error> {
        start(messages: messages, tools: tools, options: options).events
    }

    /// Lance une generation et rend la main tout de suite : evenements, annulation et
    /// attente de la fin reelle (`Gemma4ChatRun`).
    public func start(
        messages: [Gemma4ChatMessage],
        tools: [[String: any Sendable]] = [],
        options: Gemma4ChatOptions = .init()
    ) -> Gemma4ChatRun {
        let container = container
        let profile = profile
        let store: ConversationStore? = reusesConversation ? conversation : nil
        let (events, continuation) = AsyncThrowingStream<Gemma4ChatEvent, Error>.makeStream()
        let task = Task {
            // K-9 : aucune inference pendant un entrainement (deadlock mlx-swift).
            do { try Gemma4ComputeGate.shared.beginInference() } catch {
                continuation.finish(throwing: error)
                return
            }
            defer { Gemma4ComputeGate.shared.endInference() }
            do {
                try await container.perform { context in
                    try await Self.run(
                        messages: messages, tools: tools, options: options, profile: profile,
                        store: store, context: context, continuation: continuation)
                }
                if profile?.clearCacheAfterAnswer == true { Memory.clearCache() }
                continuation.finish()
            } catch {
                if profile?.clearCacheAfterAnswer == true { Memory.clearCache() }
                continuation.finish(throwing: error)
            }
        }
        continuation.onTermination = { _ in task.cancel() }
        return Gemma4ChatRun(events: events, task: task)
    }

    private static func run(
        messages: [Gemma4ChatMessage],
        tools: [[String: any Sendable]],
        options: Gemma4ChatOptions,
        profile: Gemma4ReferenceProfile?,
        store: ConversationStore?,
        context: ModelContext,
        continuation: AsyncThrowingStream<Gemma4ChatEvent, Error>.Continuation
    ) async throws {
        let allImages = messages.flatMap(\.images)
        let ids = try promptIds(
            messages: messages, tools: tools, enableThinking: options.enableThinking,
            tokenizer: context.tokenizer)
        let digests = allImages.map { Data(SHA256.hash(data: $0)) }

        var parameters = GenerateParameters(
            maxTokens: options.maxTokens, temperature: options.temperature,
            topP: options.topP, topK: options.topK)
        profile?.apply(to: &parameters)

        // Reutilisation : extension stricte du prompt precedent, memes images en tete,
        // hors n-gramme (son historique ne verrait que le suffixe) et hors 12B unified
        // (masque bidirectionnel a l'offset 0, incompatible avec un cache non vide).
        let reusable = store != nil && options.noRepeatNGramSize == nil
            && (context.model is Gemma4LLMModel || context.model is Gemma4MultimodalLLMModel)
        var cache: [any KVCache]
        var suffix = ids
        var images = allImages
        var cached = 0
        if reusable, let snapshot = store?.bestPrefix(of: ids, digests: digests) {
            cache = snapshot.caches.map { $0.copy() }
            suffix = Array(ids[snapshot.ids.count...])
            images = Array(allImages[snapshot.imageDigests.count...])
            cached = snapshot.ids.count
        } else {
            cache = context.model.newCache(parameters: parameters)
        }

        if !images.isEmpty {
            guard let model = context.model as? Gemma4MultimodalLLMModel else {
                throw Gemma4ChatEngineError.imagesNeedVision
            }
            var pixels: [MLXArray] = []
            for (index, data) in images.enumerated() {
                guard let source = CGImageSourceCreateWithData(data as CFData, nil),
                      let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
                else { throw Gemma4ChatEngineError.invalidImage(index) }
                pixels.append(try Gemma4ImageProcessor.processImage(image))
            }
            let batch = concatenated(pixels, axis: 0)
            eval(batch)
            model.pendingPixelValues = batch
        }

        // Chrono avant l'iterateur : son init fait tout le prefill (TTFT et debit du prompt).
        Memory.peakMemory = 0
        let start = Date()
        let input = LMInput(tokens: MLXArray(suffix.map { Int32($0) }))
        let iterator: TokenIterator
        if let n = options.noRepeatNGramSize {
            iterator = try TokenIterator(
                input: input, model: context.model, cache: cache,
                processor: NoRepeatNGramLogitProcessor(ngramSize: n),
                sampler: parameters.sampler(), prefillStepSize: parameters.prefillStepSize,
                maxTokens: options.maxTokens)
        } else {
            iterator = try TokenIterator(input: input, model: context.model, cache: cache, parameters: parameters)
        }
        // L'iterateur a prefille tout le prompt : instantane « fin de prompt ».
        if reusable {
            store?.put(.init(ids: ids, caches: cache.map { $0.copy() }, imageDigests: digests))
        }

        let prefillEnd = Date()
        let (tokens, generation) = MLXLMCommon.generateTokenTask(
            promptTokenCount: suffix.count, modelConfiguration: context.configuration,
            tokenizer: context.tokenizer, iterator: iterator)

        var router = Gemma4ChannelRouter()
        var content = Gemma4StreamingDetokenizer(tokenizer: context.tokenizer)
        var reasoning = Gemma4StreamingDetokenizer(tokenizer: context.tokenizer)
        let toolParser = tools.isEmpty ? nil : ToolCallProcessor(format: .gemma4, tools: tools)
        var firstToken: TimeInterval?
        var completion = 0
        var info: GenerateCompletionInfo?
        var reasoningStarted = false
        var stoppedOnToolResponse = false

        for await event in tokens {
            if Task.isCancelled {
                generation.cancel()
                break
            }
            switch event {
            case .token(let id):
                completion += 1
                if firstToken == nil { firstToken = Date().timeIntervalSince(start) }
                // `<|tool_response>` : le modele attend la reponse de l'outil.
                if id == Gemma4ChannelRouter.toolResponseTokenId {
                    stoppedOnToolResponse = true
                    generation.cancel()
                    continue
                }
                switch router.route(Int32(id)) {
                case .markup:
                    continue
                case .reasoning:
                    guard var text = reasoning.append(token: id) else { continue }
                    if !reasoningStarted {
                        text = String(text.drop(while: \.isNewline))
                        reasoningStarted = !text.isEmpty
                    }
                    if !text.isEmpty { continuation.yield(.reasoning(text)) }
                case .content:
                    guard let text = content.append(token: id) else { continue }
                    let visible = toolParser.map { $0.processChunk(text) ?? "" } ?? text
                    if !visible.isEmpty { continuation.yield(.text(visible)) }
                }
            case .info(let completionInfo):
                info = completionInfo
            }
        }
        await generation.value

        if let toolParser, let tail = toolParser.processEOS(returnBufferedText: true), !tail.isEmpty {
            continuation.yield(.text(tail))
        }
        let calls = toolParser?.toolCalls ?? []
        for call in calls {
            let arguments = (try? JSONEncoder().encode(call.function.arguments))
                .map { String(decoding: $0, as: UTF8.self) } ?? "{}"
            let id = "call_" + UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(24)
            continuation.yield(.toolCall(Gemma4ToolCall(id: id, name: call.function.name, argumentsJSON: arguments)))
        }

        let finish: Gemma4ChatUsage.FinishReason
        if Task.isCancelled {
            finish = .cancelled
        } else if !calls.isEmpty || stoppedOnToolResponse {
            finish = .toolCalls
        } else if info?.stopReason == .length || completion >= options.maxTokens {
            finish = .length
        } else {
            finish = .stop
        }
        // `promptTime` de l'amont ne voit que l'amorce apres l'init de l'iterateur.
        let prefill = prefillEnd.timeIntervalSince(start) + (info?.promptTime ?? 0)
        let decode = info?.generateTime ?? 0
        continuation.yield(.done(Gemma4ChatUsage(
            promptTokens: ids.count, cachedPromptTokens: cached, completionTokens: completion,
            prefillSeconds: prefill,
            promptTokensPerSecond: prefill > 0 ? Double(suffix.count) / prefill : 0,
            tokensPerSecond: decode > 0 ? Double(completion) / decode : 0,
            timeToFirstToken: firstToken, peakMemoryBytes: Memory.peakMemory,
            finishReason: finish)))
    }
}

/// Automate du canal de pensee sur les ids (meme contrat que celui du n-gramme) :
/// `<|channel>` puis le nom du canal ouvrent ; `<channel|>` ferme.
struct Gemma4ChannelRouter {
    enum Route { case content, reasoning, markup }

    static let toolResponseTokenId = 50

    private enum State { case outside, awaitingName, insideThought }
    private var state: State = .outside

    mutating func route(_ token: Int32) -> Route {
        switch state {
        case .outside:
            if token == Gemma4Processor.channelStartTokenId {
                state = .awaitingName
                return .markup
            }
            return token == Gemma4Processor.channelEndTokenId ? .markup : .content
        case .awaitingName:
            if token == Gemma4Processor.thoughtChannelNameTokenId {
                state = .insideThought
                return .markup
            }
            state = .outside
            return token == Gemma4Processor.responseChannelNameTokenId ? .markup : .content
        case .insideThought:
            if token == Gemma4Processor.channelEndTokenId {
                state = .outside
                return .markup
            }
            return .reasoning
        }
    }
}
