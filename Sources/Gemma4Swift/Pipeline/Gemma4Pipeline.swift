// Phase 6: Pipeline haut niveau — API complete et simple

import CoreGraphics
import Foundation
import MLX
@preconcurrency import MLXLMCommon
@preconcurrency import MLXLLM

/// Pipeline Gemma 4 de haut niveau pour le chat multimodal.
/// Le chargement du modele est gere a l'exterieur (CLI via macros, app via loadModelContainer).
/// Ce pipeline gere la generation (texte, streaming, multi-turn).
@MainActor
@Observable
public final class Gemma4Pipeline: @unchecked Sendable {

    // MARK: - Types

    /// Modeles Gemma 4 disponibles
    public enum Model: String, CaseIterable, Sendable {
        // MLX Community (pre-quantises)
        case e2b4bit = "mlx-community/gemma-4-e2b-it-4bit"
        case e4b4bit = "mlx-community/gemma-4-e4b-it-4bit"

        // Google BF16 — Instruction-tuned (chat-ready)
        case e2bIT = "google/gemma-4-E2B-it"
        case e4bIT = "google/gemma-4-E4B-it"
        case a4bIT = "google/gemma-4-26B-A4B-it"
        case b31bIT = "google/gemma-4-31B-it"

        // Google BF16 — Base (pre-trained, pour fine-tuning)
        case e2b = "google/gemma-4-E2B"
        case e4b = "google/gemma-4-E4B"
        case a4b = "google/gemma-4-26B-A4B"
        case b31b = "google/gemma-4-31B"

        public var displayName: String {
            switch self {
            case .e2b4bit: return "Gemma 4 E2B (4-bit)"
            case .e4b4bit: return "Gemma 4 E4B (4-bit)"
            case .e2bIT: return "Gemma 4 E2B IT (BF16)"
            case .e4bIT: return "Gemma 4 E4B IT (BF16)"
            case .a4bIT: return "Gemma 4 26B-A4B IT (BF16)"
            case .b31bIT: return "Gemma 4 31B IT (BF16)"
            case .e2b: return "Gemma 4 E2B (BF16)"
            case .e4b: return "Gemma 4 E4B (BF16)"
            case .a4b: return "Gemma 4 26B-A4B (BF16)"
            case .b31b: return "Gemma 4 31B (BF16)"
            }
        }

        public var estimatedSizeGB: Float {
            switch self {
            case .e2b4bit: return 3.6
            case .e4b4bit: return 5.0
            case .e2bIT, .e2b: return 10.0
            case .e4bIT, .e4b: return 19.0
            case .a4bIT, .a4b: return 52.0
            case .b31bIT, .b31b: return 63.0
            }
        }

        /// Nombre de parametres total (affichage)
        public var parameterCount: String {
            switch self {
            case .e2b4bit, .e2bIT, .e2b: return "5.1B"
            case .e4b4bit, .e4bIT, .e4b: return "9.6B"
            case .a4bIT, .a4b: return "25.8B"
            case .b31bIT, .b31b: return "31.3B"
            }
        }

        /// Parametres effectifs par token (pour MoE, seuls les experts actifs comptent)
        public var effectiveParameters: String {
            switch self {
            case .e2b4bit, .e2bIT, .e2b: return "2.3B"
            case .e4b4bit, .e4bIT, .e4b: return "4.5B"
            case .a4bIT, .a4b: return "3.8B"
            case .b31bIT, .b31b: return "31.3B"
            }
        }

        public var isMoE: Bool {
            switch self {
            case .a4bIT, .a4b: return true
            default: return false
            }
        }

        public var isInstructionTuned: Bool {
            switch self {
            case .e2b4bit, .e4b4bit, .e2bIT, .e4bIT, .a4bIT, .b31bIT: return true
            case .e2b, .e4b, .a4b, .b31b: return false
            }
        }

        public var isBF16: Bool {
            switch self {
            case .e2b4bit, .e4b4bit: return false
            default: return true
            }
        }

        /// Modalites supportees par le modele
        public struct Capabilities: OptionSet, Sendable {
            public let rawValue: Int
            public init(rawValue: Int) { self.rawValue = rawValue }

            public static let text = Capabilities(rawValue: 1 << 0)
            public static let image = Capabilities(rawValue: 1 << 1)
            public static let audio = Capabilities(rawValue: 1 << 2)
            public static let video = Capabilities(rawValue: 1 << 3)

            /// Toutes les modalites visuelles (image + video)
            public static let vision: Capabilities = [.image, .video]
            /// Modeles E2B/E4B : text + image + audio + video
            public static let anyToAny: Capabilities = [.text, .image, .audio, .video]
            /// Modeles 26B/31B : text + image + video (pas d'audio)
            public static let imageTextToText: Capabilities = [.text, .image, .video]
        }

        public var capabilities: Capabilities {
            switch self {
            // E2B et E4B supportent toutes les modalites (any-to-any)
            case .e2b4bit, .e4b4bit, .e2bIT, .e2b, .e4bIT, .e4b:
                return .anyToAny
            // 26B-A4B et 31B : text + image + video (pas d'audio)
            case .a4bIT, .a4b, .b31bIT, .b31b:
                return .imageTextToText
            }
        }

        public var supportsAudio: Bool { capabilities.contains(.audio) }
        public var supportsImage: Bool { capabilities.contains(.image) }
        public var supportsVideo: Bool { capabilities.contains(.video) }

        /// RAM minimale recommandee (Go) pour charger le modele
        public var recommendedRAMGB: Int {
            switch self {
            case .e2b4bit: return 8
            case .e4b4bit: return 12
            case .e2bIT, .e2b: return 16
            case .e4bIT, .e4b: return 36
            case .a4bIT, .a4b: return 96
            case .b31bIT, .b31b: return 128
            }
        }

        /// RAM systeme en Go
        public static var systemRAMGB: Int {
            Int(ProcessInfo.processInfo.physicalMemory / (1024 * 1024 * 1024))
        }

        /// Modeles recommandes pour la RAM disponible (IT uniquement)
        public static func recommended(forRAMGB ram: Int) -> [Model] {
            allCases
                .filter { $0.isInstructionTuned && $0.recommendedRAMGB <= ram }
                .sorted { $0.estimatedSizeGB < $1.estimatedSizeGB }
        }
    }

    /// Etat du pipeline
    public enum State: Sendable {
        case unloaded
        case ready
        case processing
        case error(String)
    }

    /// Stats de generation
    public struct GenerationStats: Sendable {
        public let tokensGenerated: Int
        public let tokensPerSecond: Double
        public let totalTime: TimeInterval
        public let peakMemoryMB: Int
    }

    // MARK: - Proprietes

    public private(set) var state: State = .unloaded
    public private(set) var lastStats: GenerationStats?

    public var isReady: Bool {
        if case .ready = state { return true }
        return false
    }

    private var container: ModelContainer?
    nonisolated(unsafe) private var currentSession: ChatSession?

    // MARK: - Chargement

    /// Initialise le pipeline avec un ModelContainer deja charge
    public func setContainer(_ container: ModelContainer) {
        self.container = container
        state = .ready
    }

    /// Decharge le modele
    public func unload() {
        container = nil
        currentSession = nil
        state = .unloaded
        lastStats = nil
        MLX.GPU.clearCache()
    }

    // MARK: - Generation texte

    /// Genere une reponse complete (non-streaming)
    public func chat(
        prompt: String,
        systemPrompt: String? = nil,
        temperature: Float = 0.3,
        maxTokens: Int = 1024
    ) async throws -> String {
        guard let container = container else {
            throw Gemma4PipelineError.modelNotLoaded
        }

        let params = GenerateParameters(maxTokens: maxTokens, temperature: temperature, topP: 0.95)
        let session = ChatSession(
            container,
            instructions: systemPrompt ?? "Tu es un assistant utile.",
            generateParameters: params
        )
        currentSession = session

        state = .processing
        defer { state = .ready }

        let startTime = Date()
        let response = try await session.respond(to: prompt)
        let elapsed = Date().timeIntervalSince(startTime)

        lastStats = GenerationStats(
            tokensGenerated: max(1, response.count / 4),
            tokensPerSecond: Double(response.count / 4) / max(0.01, elapsed),
            totalTime: elapsed,
            peakMemoryMB: Int(MLX.GPU.peakMemory / (1024 * 1024))
        )

        return response
    }

    /// Genere en streaming (token par token)
    public func chatStream(
        prompt: String,
        systemPrompt: String? = nil,
        temperature: Float = 0.3,
        maxTokens: Int = 1024
    ) throws -> AsyncThrowingStream<String, Error> {
        guard let container = container else {
            throw Gemma4PipelineError.modelNotLoaded
        }

        let params = GenerateParameters(maxTokens: maxTokens, temperature: temperature, topP: 0.95)
        let session = ChatSession(
            container,
            instructions: systemPrompt ?? "Tu es un assistant utile.",
            generateParameters: params
        )
        currentSession = session

        state = .processing
        let stream = session.streamResponse(to: prompt)

        return AsyncThrowingStream { continuation in
            Task { [weak self] in
                do {
                    for try await token in stream {
                        continuation.yield(token)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
                await MainActor.run {
                    self?.state = .ready
                }
            }
        }
    }

    /// Continue la conversation dans la session courante (multi-turn)
    public func continueChat(
        prompt: String
    ) async throws -> String {
        guard let session = currentSession else {
            throw Gemma4PipelineError.modelNotLoaded
        }
        state = .processing
        defer { state = .ready }
        return try await session.respond(to: prompt)
    }

    /// Continue la conversation en streaming (multi-turn)
    public func continueChatStream(
        prompt: String
    ) throws -> AsyncThrowingStream<String, Error> {
        guard let session = currentSession else {
            throw Gemma4PipelineError.modelNotLoaded
        }
        state = .processing
        let stream = session.streamResponse(to: prompt)

        return AsyncThrowingStream { continuation in
            Task { [weak self] in
                do {
                    for try await token in stream {
                        continuation.yield(token)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
                await MainActor.run {
                    self?.state = .ready
                }
            }
        }
    }

    // MARK: - GPU Stats

    public var activeMemoryMB: Int {
        Int(MLX.GPU.activeMemory / (1024 * 1024))
    }

    public var peakMemoryMB: Int {
        Int(MLX.GPU.peakMemory / (1024 * 1024))
    }

    public func clearGPUCache() {
        MLX.GPU.clearCache()
    }
}

// MARK: - Erreurs

public enum Gemma4PipelineError: LocalizedError {
    case modelNotLoaded
    case invalidInput(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotLoaded: return "Modele non charge"
        case .invalidInput(let msg): return "Entree invalide: \(msg)"
        }
    }
}
