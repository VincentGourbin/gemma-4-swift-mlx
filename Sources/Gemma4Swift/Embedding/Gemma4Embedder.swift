// Embeddings EmbeddingGemma 2 (texte, image, audio, video, entrelaces) dans un espace de 768 dims.
//
// Le modele vient de MLXLMCommon (`EmbeddingGemma2`, PR mlx-swift-lm #684) : encodeur
// bidirectionnel, tours vision/audio Gemma 4, mean pooling, projection 512 -> 768, L2.
// Le pretraitement (pixels, log-mel, frames) est celui du paquet : MLXVLM n'est pas lie
// (collision de registres, cf. CLAUDE.md « Dependency pinning »).

import AVFoundation
import CoreGraphics
import Foundation
import MLX
import MLXLMCommon

/// Embeddings EmbeddingGemma 2. Charger une fois, reutiliser pour tous les appels.
///
/// ```swift
/// let embedder = try await Gemma4Embedder.load(from: directory)
/// let query = try await embedder.embed([.text("Eteins la lumiere")], task: .classification)
/// let shot = try await embedder.embed([.image(url)], task: .classification, dimensions: 256)
/// ```
public actor Gemma4Embedder {

    /// Encodeurs de modalite optionnels (le texte est toujours charge).
    public enum Encoder: Hashable, Sendable {
        case vision
        case audio
    }

    /// Un morceau d'entree ; les morceaux sont concatenes dans l'ordre en une seule sequence.
    public enum Part: Sendable {
        case text(String)
        case image(URL)
        case cgImage(CGImage)
        case audio(URL)
        case video(URL)
    }

    /// Prefixe d'instruction de la model card. Les medias seuls n'en recoivent pas.
    public enum Task: Sendable, Equatable {
        case none
        case classification
        case clustering
        case sentenceSimilarity
        case searchQuery
        case questionAnswering
        case factChecking
        case codeRetrieval
        /// Cote corpus d'une tache asymetrique ; `title: none` sans titre.
        case document(title: String?)

        func prefix() -> String {
            switch self {
            case .none: return ""
            case .classification: return "task: classification | query: "
            case .clustering: return "task: clustering | query: "
            case .sentenceSimilarity: return "task: sentence similarity | query: "
            case .searchQuery: return "task: search result | query: "
            case .questionAnswering: return "task: question answering | query: "
            case .factChecking: return "task: fact checking | query: "
            case .codeRetrieval: return "task: code retrieval | query: "
            case .document(let title):
                let line = title?.split(whereSeparator: \.isWhitespace).joined(separator: " ")
                return "title: \(line.flatMap { $0.isEmpty ? nil : $0 } ?? "none") | text: "
            }
        }
    }

    public enum EmbedderError: Error, LocalizedError, Equatable {
        case emptyInput
        case unsupportedModality(String)
        case contextExceeded(tokens: Int)
        case invalidDimension(Int)
        case invalidEmbedding
        case placeholderInText

        public var errorDescription: String? {
            switch self {
            case .emptyInput: return "Entree vide."
            case .unsupportedModality(let m): return "Modalite \(m) non chargee ou absente du checkpoint."
            case .contextExceeded(let n): return "\(n) tokens : depasse le contexte de 8192."
            case .invalidDimension(let d): return "Dimension \(d) non supportee (128, 256, 512 ou 768)."
            case .invalidEmbedding: return "Embedding non fini ou nul."
            case .placeholderInText: return "Le texte contient un jeton de media (<|image|>, <|audio|>, <|video|>)."
            }
        }
    }

    /// Budgets de la reference (`processor_config.json` d'EmbeddingGemma 2).
    public static let imageSoftTokens = 280
    public static let videoSoftTokensPerFrame = 140
    public static let videoFramesPerSecond = 1.0
    public static let videoMaxFrames = 32
    public static let contextLength = EmbeddingGemma2Configuration.contextLength
    /// Dimensions Matryoshka entrainees, en plus de la dimension native.
    public static let matryoshkaDimensions = [128, 256, 512]

    public nonisolated let embeddingDimension: Int
    public nonisolated let encoders: Set<Encoder>

    /// Jetons speciaux et encodeurs presents, lus dans `config.json` (les sous-configurations
    /// de `EmbeddingGemma2Configuration` sont `package` en amont).
    struct SpecialTokens: Decodable {
        let imageTokenId: Int
        let videoTokenId: Int
        let audioTokenId: Int
        let boiTokenId: Int
        let eoiTokenId: Int
        let boaTokenId: Int
        let eoaTokenId: Int
        let hasVision: Bool
        let hasAudio: Bool

        enum CodingKeys: String, CodingKey {
            case imageTokenId = "image_token_id"
            case videoTokenId = "video_token_id"
            case audioTokenId = "audio_token_id"
            case boiTokenId = "boi_token_id"
            case eoiTokenId = "eoi_token_id"
            case boaTokenId = "boa_token_id"
            // Nom du checkpoint Google (pas `eoa_token_id`).
            case eoaTokenIndex = "eoa_token_index"
            case visionConfig = "vision_config"
            case audioConfig = "audio_config"
        }

        init(from decoder: any Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            imageTokenId = try c.decodeIfPresent(Int.self, forKey: .imageTokenId) ?? 258_880
            videoTokenId = try c.decodeIfPresent(Int.self, forKey: .videoTokenId) ?? 258_884
            audioTokenId = try c.decodeIfPresent(Int.self, forKey: .audioTokenId) ?? 258_881
            boiTokenId = try c.decodeIfPresent(Int.self, forKey: .boiTokenId) ?? 255_999
            eoiTokenId = try c.decodeIfPresent(Int.self, forKey: .eoiTokenId) ?? 258_882
            boaTokenId = try c.decodeIfPresent(Int.self, forKey: .boaTokenId) ?? 256_000
            eoaTokenId = try c.decodeIfPresent(Int.self, forKey: .eoaTokenIndex) ?? 258_883
            hasVision = (try? c.decodeNil(forKey: .visionConfig)) == false
            hasAudio = (try? c.decodeNil(forKey: .audioConfig)) == false
        }

        var media: Set<Int> { [imageTokenId, videoTokenId, audioTokenId] }
    }

    private let model: EmbeddingGemma2
    private let tokenizer: any MLXLMCommon.Tokenizer
    private let special: SpecialTokens
    private let bos: Int
    private let eos: Int

    private init(
        model: EmbeddingGemma2, tokenizer: any MLXLMCommon.Tokenizer, special: SpecialTokens,
        encoders: Set<Encoder>, bos: Int, eos: Int
    ) {
        self.model = model
        self.tokenizer = tokenizer
        self.special = special
        self.bos = bos
        self.eos = eos
        self.embeddingDimension = model.config.embeddingDim
        self.encoders = encoders
    }

    /// Charge un checkpoint EmbeddingGemma 2 (`google/embeddinggemma-2` ou une conversion
    /// mlx-community, quantifiee ou non).
    /// - Parameter encoders: encodeurs a garder ; les autres ne sont ni lus ni alloues
    ///   (texte seul : ~270M parametres). Les encodeurs gardes se chargent au premier usage.
    public static func load(
        from directory: URL, encoders: Set<Encoder> = [.vision, .audio]
    ) async throws -> Gemma4Embedder {
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let special = try JSONDecoder().decode(SpecialTokens.self, from: data)
        var kept = encoders
        if !special.hasVision { kept.remove(.vision) }
        if !special.hasAudio { kept.remove(.audio) }
        let config = try JSONDecoder().decode(EmbeddingGemma2Configuration.self, from: data)
            .selectingEncoders(vision: kept.contains(.vision), audio: kept.contains(.audio))
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        let model = EmbeddingGemma2(config)
        try await loadWeights(
            modelDirectory: directory, model: model,
            perLayerQuantization: base.perLayerQuantization)
        let tokenizer = try await Gemma4TokenizerLoader().load(from: directory)
        guard let bos = tokenizer.convertTokenToId("<bos>"),
              let eos = tokenizer.convertTokenToId("<eos>")
        else { throw EmbedderError.invalidEmbedding }
        return Gemma4Embedder(
            model: model, tokenizer: tokenizer, special: special, encoders: kept, bos: bos, eos: eos)
    }

    /// Embedding d'une entree, vecteur unitaire float32.
    /// - Parameters:
    ///   - task: prefixe ajoute au premier texte ; ignore si l'entree n'a pas de texte
    ///     ou si le texte commence deja par `task:`/`title:`.
    ///   - dimensions: troncature Matryoshka (128, 256, 512), renormalisee ; `nil` = native.
    public func embed(_ parts: [Part], task: Task, dimensions: Int? = nil) async throws -> [Float] {
        let dimension = dimensions ?? embeddingDimension
        guard dimension == embeddingDimension
                || (Self.matryoshkaDimensions.contains(dimension) && dimension < embeddingDimension)
        else { throw EmbedderError.invalidDimension(dimension) }
        guard !parts.isEmpty else { throw EmbedderError.emptyInput }

        // K-9 : aucune inference pendant un entrainement (Gemma4ComputeGate).
        try Gemma4ComputeGate.shared.beginInference()
        defer { Gemma4ComputeGate.shared.endInference() }

        let (ids, softTokens) = try await sequence(for: Self.prompted(parts, task: task))
        let vector = model.embed(
            inputIds: MLXArray(ids.map(Int32.init), [1, ids.count]), attentionMask: nil,
            softTokens: softTokens)
        try checkedEval(vector)
        return try Self.truncated(vector.asArray(Float.self), to: dimension)
    }

    /// Embeddings de plusieurs entrees, traitees une a une (memoire bornee).
    public func embed(_ inputs: [[Part]], task: Task, dimensions: Int? = nil) async throws -> [[Float]] {
        var vectors: [[Float]] = []
        vectors.reserveCapacity(inputs.count)
        for input in inputs {
            try _Concurrency.Task.checkCancellation()
            vectors.append(try await embed(input, task: task, dimensions: dimensions))
        }
        return vectors
    }

    // MARK: - Sequence

    /// Le prefixe rejoint le premier texte, comme le gabarit de la reference le rend.
    static func prompted(_ parts: [Part], task: Task) -> [Part] {
        let prefix = task.prefix()
        guard !prefix.isEmpty, parts.contains(where: { if case .text = $0 { true } else { false } })
        else { return parts }
        if case .text(let first) = parts[0] {
            if first.hasPrefix("task:") || first.hasPrefix("title:") { return parts }
            return [.text(prefix + first)] + parts.dropFirst()
        }
        return [.text(prefix)] + parts
    }

    /// `<bos>`, les segments dans l'ordre, `<eos>`. Une image (et chaque frame video)
    /// devient `<boi> <placeholder>×n <eoi>`, un audio `<boa> <audio>×n <eoa>`.
    private func sequence(for parts: [Part]) async throws -> ([Int], MLXArray?) {
        var ids = [bos]
        var features: [MLXArray] = []
        var text = ""
        func flushText() throws {
            guard !text.isEmpty else { return }
            let tokens = tokenizer.encode(text: text, addSpecialTokens: false)
            if tokens.contains(where: special.media.contains) { throw EmbedderError.placeholderInText }
            ids += tokens
            text = ""
        }
        for part in parts {
            if case .text(let value) = part {
                text += value
                continue
            }
            try flushText()
            let (block, soft) = try await encode(part)
            // Un media a la fois : les activations d'un seul encodeur vivent en meme temps.
            if let soft {
                try checkedEval(soft)
                features.append(soft.reshaped(-1, soft.dim(-1)))
            }
            ids += block
        }
        try flushText()
        ids.append(eos)
        guard ids.count > 2 else { throw EmbedderError.emptyInput }
        guard ids.count <= Self.contextLength else { throw EmbedderError.contextExceeded(tokens: ids.count) }
        return (ids, features.isEmpty ? nil : concatenated(features, axis: 0))
    }

    private func encode(_ part: Part) async throws -> ([Int], MLXArray?) {
        func marked(_ begin: Int, _ token: Int, _ count: Int, _ end: Int) -> [Int] {
            [begin] + repeatElement(token, count: count) + [end]
        }
        switch part {
        case .text:
            preconditionFailure("le texte n'a pas de soft tokens")
        case .image(let url):
            guard encoders.contains(.vision) else { throw EmbedderError.unsupportedModality("image") }
            let pixels = try Gemma4ImageProcessor.processImage(
                url: url, maxSoftTokens: Self.imageSoftTokens, resampling: .pillowBicubic)
            return try imageBlock(pixels)
        case .cgImage(let image):
            guard encoders.contains(.vision) else { throw EmbedderError.unsupportedModality("image") }
            let pixels = try Gemma4ImageProcessor.processImage(
                image, maxSoftTokens: Self.imageSoftTokens, resampling: .pillowBicubic)
            return try imageBlock(pixels)
        case .video(let url):
            guard encoders.contains(.vision) else { throw EmbedderError.unsupportedModality("video") }
            let frames = try await Self.videoFrames(url: url)
            guard let soft = try model.imageFeatures(frames) else {
                throw EmbedderError.unsupportedModality("video")
            }
            let frame = marked(special.boiTokenId, special.videoTokenId, soft.dim(1), special.eoiTokenId)
            return (Array(repeatElement(frame, count: soft.dim(0)).joined()), soft)
        case .audio(let url):
            guard encoders.contains(.audio) else { throw EmbedderError.unsupportedModality("audio") }
            let mel = try await Gemma4AudioProcessor.processAudio(url: url).features
            guard mel.dim(1) > 0 else {
                return (marked(special.boaTokenId, special.audioTokenId, 0, special.eoaTokenId), nil)
            }
            guard let soft = try model.audioFeatures(mel.squeezed(axis: 0)) else {
                throw EmbedderError.unsupportedModality("audio")
            }
            return (marked(special.boaTokenId, special.audioTokenId, soft.dim(1), special.eoaTokenId), soft)
        }

        func imageBlock(_ pixels: MLXArray) throws -> ([Int], MLXArray?) {
            guard let soft = try model.imageFeatures(pixels) else {
                throw EmbedderError.unsupportedModality("image")
            }
            return (marked(special.boiTokenId, special.imageTokenId, soft.dim(1), special.eoiTokenId), soft)
        }
    }

    /// Frames a `t = i / fps` (reference `sample_frames`), plafonnees a `videoMaxFrames`
    /// par sous-echantillonnage uniforme, au budget de 140 soft tokens chacune.
    static func videoFrames(url: URL) async throws -> MLXArray {
        let asset = AVURLAsset(url: url)
        let duration = CMTimeGetSeconds(try await asset.load(.duration))
        guard duration > 0 else { throw VideoProcessingError.invalidVideo("Duree video invalide") }
        let times = frameTimes(duration: duration)
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero
        var frames: [MLXArray] = []
        for t in times {
            let (image, _) = try await generator.image(at: CMTime(seconds: t, preferredTimescale: 600))
            frames.append(try Gemma4ImageProcessor.processImage(
                image, maxSoftTokens: videoSoftTokensPerFrame, resampling: .pillowBicubic))
        }
        guard !frames.isEmpty else { throw VideoProcessingError.noFramesExtracted }
        return concatenated(frames, axis: 0)
    }

    static func frameTimes(duration: Double) -> [Double] {
        let count = max(1, Int((duration * videoFramesPerSecond).rounded(.down)))
        let all = (0 ..< count).map { Double($0) / videoFramesPerSecond }
        guard all.count > videoMaxFrames else { return all }
        // overflow_strategy "uniform" : np.linspace(0, n - 1, max_frames, dtype=int).
        return (0 ..< videoMaxFrames).map { all[$0 * (all.count - 1) / (videoMaxFrames - 1)] }
    }

    /// Troncature Matryoshka puis renormalisation L2 (en double).
    static func truncated(_ values: [Float], to dimension: Int) throws -> [Float] {
        guard values.allSatisfy(\.isFinite) else { throw EmbedderError.invalidEmbedding }
        let head = values.prefix(dimension)
        let norm = head.reduce(0.0) { $0 + Double($1) * Double($1) }.squareRoot()
        guard norm.isFinite, norm > 0 else { throw EmbedderError.invalidEmbedding }
        return head.map { Float(Double($0) / norm) }
    }
}
