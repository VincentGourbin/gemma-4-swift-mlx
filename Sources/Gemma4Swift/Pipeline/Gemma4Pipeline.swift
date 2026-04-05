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
        case e2b4bit = "mlx-community/gemma-4-e2b-it-4bit"
        case e4b4bit = "mlx-community/gemma-4-e4b-it-4bit"

        public var displayName: String {
            switch self {
            case .e2b4bit: return "Gemma 4 E2B (4-bit)"
            case .e4b4bit: return "Gemma 4 E4B (4-bit)"
            }
        }

        public var estimatedSizeGB: Float {
            switch self {
            case .e2b4bit: return 3.6
            case .e4b4bit: return 5.0
            }
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
