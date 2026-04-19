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

    /// Modeles Gemma 4 disponibles (MLX Community pre-quantises + Google BF16)
    public enum Model: String, CaseIterable, Sendable {
        // E2B — 2.3B effectifs, text+vision+audio+video
        case e2b4bit = "mlx-community/gemma-4-e2b-it-4bit"
        case e2b8bit = "mlx-community/gemma-4-e2b-it-8bit"
        case e2b6bit = "mlx-community/gemma-4-e2b-it-6bit"
        case e2bBf16 = "mlx-community/gemma-4-e2b-it-bf16"

        // E4B — 4.5B effectifs, text+vision+audio+video
        case e4b4bit = "mlx-community/gemma-4-e4b-it-4bit"
        case e4b8bit = "mlx-community/gemma-4-e4b-it-8bit"
        case e4b6bit = "mlx-community/gemma-4-e4b-it-6bit"
        case e4bBf16 = "mlx-community/gemma-4-e4b-it-bf16"

        // 31B — 31B dense, text+vision (pas d'audio), K=V attention
        case b31b4bit = "mlx-community/gemma-4-31b-it-4bit"
        case b31b8bit = "mlx-community/gemma-4-31b-it-8bit"
        case b31b6bit = "mlx-community/gemma-4-31b-it-6bit"
        case b31bBf16 = "mlx-community/gemma-4-31b-it-bf16"

        // 26B-A4B — MoE 128 experts top-8, 3.8B actifs, text+vision, K=V
        case a4b4bit = "mlx-community/gemma-4-26b-a4b-it-4bit"
        case a4b8bit = "mlx-community/gemma-4-26b-a4b-it-8bit"
        case a4b6bit = "mlx-community/gemma-4-26b-a4b-it-6bit"
        case a4bBf16 = "mlx-community/gemma-4-26b-a4b-it-bf16"

        /// Famille du modele
        public enum Family: String, Sendable {
            case e2b, e4b, b31b, a4b
        }

        public var family: Family {
            switch self {
            case .e2b4bit, .e2b8bit, .e2b6bit, .e2bBf16: return .e2b
            case .e4b4bit, .e4b8bit, .e4b6bit, .e4bBf16: return .e4b
            case .b31b4bit, .b31b8bit, .b31b6bit, .b31bBf16: return .b31b
            case .a4b4bit, .a4b8bit, .a4b6bit, .a4bBf16: return .a4b
            }
        }

        public var displayName: String {
            let quant: String
            switch self {
            case .e2b4bit, .e4b4bit, .b31b4bit, .a4b4bit: quant = "4-bit"
            case .e2b8bit, .e4b8bit, .b31b8bit, .a4b8bit: quant = "8-bit"
            case .e2b6bit, .e4b6bit, .b31b6bit, .a4b6bit: quant = "6-bit"
            case .e2bBf16, .e4bBf16, .b31bBf16, .a4bBf16: quant = "BF16"
            }
            switch family {
            case .e2b: return "Gemma 4 E2B (\(quant))"
            case .e4b: return "Gemma 4 E4B (\(quant))"
            case .b31b: return "Gemma 4 31B (\(quant))"
            case .a4b: return "Gemma 4 26B-A4B (\(quant))"
            }
        }

        public var estimatedSizeGB: Float {
            switch self {
            case .e2b4bit: return 3.6
            case .e2b6bit: return 4.2
            case .e2b8bit: return 5.2
            case .e2bBf16: return 10.0
            case .e4b4bit: return 5.0
            case .e4b6bit: return 6.5
            case .e4b8bit: return 8.0
            case .e4bBf16: return 19.0
            case .b31b4bit: return 17.0
            case .b31b6bit: return 25.0
            case .b31b8bit: return 33.0
            case .b31bBf16: return 63.0
            case .a4b4bit: return 14.0
            case .a4b6bit: return 21.0
            case .a4b8bit: return 27.0
            case .a4bBf16: return 52.0
            }
        }

        /// Nombre de parametres total
        public var parameterCount: String {
            switch family {
            case .e2b: return "5.1B"
            case .e4b: return "9.6B"
            case .b31b: return "31.3B"
            case .a4b: return "25.8B"
            }
        }

        /// Parametres effectifs par token (MoE: seuls les experts actifs)
        public var effectiveParameters: String {
            switch family {
            case .e2b: return "2.3B"
            case .e4b: return "4.5B"
            case .b31b: return "31.3B"
            case .a4b: return "3.8B"
            }
        }

        public var isMoE: Bool { family == .a4b }
        public var isInstructionTuned: Bool { true } // Tous les modeles listes sont -it

        public var quantization: String {
            switch self {
            case .e2b4bit, .e4b4bit, .b31b4bit, .a4b4bit: return "4-bit"
            case .e2b8bit, .e4b8bit, .b31b8bit, .a4b8bit: return "8-bit"
            case .e2b6bit, .e4b6bit, .b31b6bit, .a4b6bit: return "6-bit"
            case .e2bBf16, .e4bBf16, .b31bBf16, .a4bBf16: return "bf16"
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
            switch family {
            // E2B et E4B supportent toutes les modalites (any-to-any)
            case .e2b, .e4b:
                return .anyToAny
            // 26B-A4B et 31B : text + image + video (pas d'audio)
            case .a4b, .b31b:
                return .imageTextToText
            }
        }

        public var supportsAudio: Bool { capabilities.contains(.audio) }
        public var supportsImage: Bool { capabilities.contains(.image) }
        public var supportsVideo: Bool { capabilities.contains(.video) }

        /// RAM minimale recommandee (Go) pour charger le modele
        public var recommendedRAMGB: Int {
            Int(ceil(estimatedSizeGB * 1.3)) // ~30% overhead pour KV cache + OS
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

    // MARK: - Proprietes

    public private(set) var state: State = .unloaded

    public init() {}

    public var isReady: Bool {
        if case .ready = state { return true }
        return false
    }

    private var container: ModelContainer?
    nonisolated(unsafe) private var currentSession: ChatSession?

    // MARK: - Chargement

    /// Charge un modele Gemma 4, avec telechargement optionnel.
    /// Gere automatiquement l'enregistrement du type de modele et le chargement du tokenizer.
    /// - Parameters:
    ///   - model: le modele a charger (enum Model)
    ///   - multimodal: si true, charge le modele multimodal complet (vision+audio+video). Defaut: true.
    ///   - downloadIfNeeded: si true, telecharge le modele s'il n'est pas en cache. Defaut: false.
    ///   - hfToken: token HuggingFace optionnel (pour modeles prives)
    ///   - progress: callback de progression du telechargement
    /// - Throws: Gemma4PipelineError.modelNotDownloaded si le modele n'est pas telecharge et downloadIfNeeded est false
    public func load(
        _ model: Model,
        multimodal: Bool = true,
        downloadIfNeeded: Bool = false,
        hfToken: String? = nil,
        progress: (@Sendable (Gemma4ModelDownloader.Progress) -> Void)? = nil
    ) async throws {
        // Telecharger si necessaire
        if downloadIfNeeded && !Gemma4ModelCache.isDownloaded(model) {
            let _ = try await Gemma4ModelDownloader.download(model, token: hfToken, progress: progress)
        }

        guard let localPath = Gemma4ModelCache.localPath(for: model) else {
            throw Gemma4PipelineError.modelNotDownloaded(model.rawValue)
        }
        try await load(from: localPath, multimodal: multimodal)
    }

    /// Charge un modele Gemma 4 depuis un chemin local arbitraire.
    /// - Parameters:
    ///   - path: URL du repertoire contenant config.json + safetensors + tokenizer.json
    ///   - multimodal: si true, charge le modele multimodal complet. Defaut: true.
    public func load(from path: URL, multimodal: Bool = true) async throws {
        state = .unloaded
        await Gemma4Registration.register(multimodal: multimodal)
        let loaded = try await loadModelContainer(from: path, using: Gemma4TokenizerLoader())
        setContainer(loaded)
    }

    /// Initialise le pipeline avec un ModelContainer deja charge
    public func setContainer(_ container: ModelContainer) {
        self.container = container
        state = .ready
    }

    // MARK: - LoRA Adapters

    /// Charge un adapter LoRA et l'applique au modele
    public func loadAdapter(from directory: URL) async throws {
        guard let container else { throw Gemma4PipelineError.modelNotLoaded }
        try await Gemma4LoRAInference.loadAdapter(into: container, from: directory)
    }

    /// Fuse un adapter LoRA dans les poids du modele (permanent, meilleures perfs d'inference)
    public func fuseAdapter(from directory: URL) async throws {
        guard let container else { throw Gemma4PipelineError.modelNotLoaded }
        try await Gemma4LoRAInference.fuseAdapter(into: container, from: directory)
    }

    /// Decharge le modele
    public func unload() {
        container = nil
        currentSession = nil
        state = .unloaded
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

        return try await session.respond(to: prompt)
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

}

// MARK: - Erreurs

public enum Gemma4PipelineError: LocalizedError {
    case modelNotLoaded
    case modelNotDownloaded(String)
    case invalidInput(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotLoaded: return "Modele non charge. Appelez load() d'abord."
        case .modelNotDownloaded(let id): return "Modele '\(id)' non telecharge. Utilisez gemma4-cli download."
        case .invalidInput(let msg): return "Entree invalide: \(msg)"
        }
    }
}
