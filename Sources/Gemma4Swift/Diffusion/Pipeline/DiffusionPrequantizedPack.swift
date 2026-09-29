// Pack pre-quantifie DiffusionGemma (K-D12).
//
// La quantification a la volee part du bf16 (~49 Go lus, pic de chargement ~51 Go meme
// couche par couche). Un pack exporte une fois se charge directement quantifie : c'est le
// seul moyen de passer sous ce pic (Mac 32-48 Go).
//
// Format `gemma4-diffusion-prequantized-v1` :
//   - `prequantized.json` : manifeste (format, quantification, carte module -> bits,
//     fichiers + SHA-256) ;
//   - `model-XXXXX-of-YYYYY.safetensors` : cles sanitisees du module Swift
//     (`encoder.language_model.…`, `decoder.…`, `encoder.vision_tower.…`). Les modules du
//     decodeur partages avec l'encodeur (contrat du sanitizer, `sharedDecoderLeafPaths`)
//     ne sont ecrits qu'une fois, cote encodeur ;
//   - les autres fichiers du checkpoint (config, tokenizer, gabarit) copies tels quels.

import CryptoKit
import Foundation
import MLX
import MLXLMCommon
import MLXNN

public enum DiffusionPrequantizedPack {

    public static let format = "gemma4-diffusion-prequantized-v1"
    public static let manifestName = "prequantized.json"

    public enum PackError: Error, LocalizedError {
        case unsupportedFormat(String)
        case missingFile(String)
        case checksumMismatch(String)
        case missingWeights([String])
        case quantizationMismatch(pack: String, profile: String)

        public var errorDescription: String? {
            switch self {
            case .unsupportedFormat(let f): return "format de pack non supporte : \(f)"
            case .missingFile(let f): return "fichier du pack manquant : \(f)"
            case .checksumMismatch(let f): return "SHA-256 different pour \(f) (pack corrompu ou incomplet)"
            case .missingWeights(let keys):
                return "\(keys.count) poids absents du pack (ex. \(keys.prefix(3).joined(separator: ", ")))"
            case .quantizationMismatch(let pack, let profile):
                return "le pack est quantifie en \(pack), le profil demande \(profile)"
            }
        }
    }

    public struct ModuleQuantization: Codable, Sendable, Equatable {
        public let bits: Int
        public let groupSize: Int
        public let mode: String

        enum CodingKeys: String, CodingKey {
            case bits, mode
            case groupSize = "group_size"
        }
    }

    public struct Manifest: Codable, Sendable {
        public let format: String
        /// Signature de la quantification (`DiffusionReferenceProfile.Quantization.signature`).
        public let quantization: String
        public let source: String
        public let created: String
        /// Chemin du module feuille -> quantification.
        public let modules: [String: ModuleQuantization]
        /// Fichier de poids -> SHA-256 (hex).
        public let files: [String: String]
    }

    /// Vrai si `directory` contient un pack pre-quantifie.
    public static func isPack(_ directory: URL) -> Bool {
        FileManager.default.fileExists(atPath: directory.appendingPathComponent(manifestName).path)
    }

    public static func readManifest(_ directory: URL) throws -> Manifest {
        let manifest = try JSONDecoder().decode(
            Manifest.self, from: Data(contentsOf: directory.appendingPathComponent(manifestName)))
        guard manifest.format == format else { throw PackError.unsupportedFormat(manifest.format) }
        return manifest
    }

    // MARK: - Export

    /// Ecrit le modele (deja quantifie par un profil) dans `destination`.
    /// - Parameters:
    ///   - model : modele quantifie (`DiffusionGemmaRegistration.load(from:profile:)`).
    ///   - quantization : signature a enregistrer.
    ///   - sourceDirectory : checkpoint bf16 d'origine (fichiers annexes copies).
    ///   - shardBytes : taille cible d'un fichier de poids.
    @discardableResult
    public static func export(
        model: DiffusionGemmaForBlockDiffusion,
        quantization: String,
        sourceDirectory: URL,
        to destination: URL,
        shardBytes: Int = 5 << 30
    ) throws -> Manifest {
        let fm = FileManager.default
        try fm.createDirectory(at: destination, withIntermediateDirectories: true)

        // Fichiers annexes (config, tokenizer, gabarit...) : tout sauf les poids.
        for item in try fm.contentsOfDirectory(at: sourceDirectory, includingPropertiesForKeys: nil) {
            let name = item.lastPathComponent
            guard !name.hasSuffix(".safetensors"), !name.hasSuffix(".safetensors.index.json"),
                  !name.hasPrefix("."), name != manifestName else { continue }
            var isDir: ObjCBool = false
            guard fm.fileExists(atPath: item.path, isDirectory: &isDir), !isDir.boolValue else { continue }
            let target = destination.appendingPathComponent(name)
            if fm.fileExists(atPath: target.path) { try fm.removeItem(at: target) }
            try fm.copyItem(at: item, to: target)
        }

        // Carte de quantification par module feuille.
        var modules: [String: ModuleQuantization] = [:]
        for (path, module) in model.leafModules().flattened() {
            if let q = quantizationOf(module) { modules[path] = q }
        }

        // Poids : une seule copie des modules partages (cote encodeur).
        let sharedDecoderKeys = sharedDecoderParameterKeys(model)
        // Tableaux vides (blocs desactives) : safetensors ne sait pas les ecrire, et ils
        // n'ont rien a restaurer.
        let weights = model.parameters().flattened().filter { !sharedDecoderKeys.contains($0.0) && $0.1.size > 0 }

        // Shards de ~shardBytes, dans l'ordre des cles (deterministe).
        var shards: [[(String, MLXArray)]] = [[]]
        var current = 0
        for (key, array) in weights.sorted(by: { $0.0 < $1.0 }) {
            if current > 0 && current + array.nbytes > shardBytes {
                shards.append([])
                current = 0
            }
            shards[shards.count - 1].append((key, array))
            current += array.nbytes
        }
        var files: [String: String] = [:]
        for (i, shard) in shards.enumerated() {
            let name = String(format: "model-%05d-of-%05d.safetensors", i + 1, shards.count)
            let url = destination.appendingPathComponent(name)
            try MLX.save(
                arrays: Dictionary(shard, uniquingKeysWith: { a, _ in a }),
                metadata: ["format": "mlx"], url: url)
            files[name] = try sha256(of: url)
        }

        let manifest = Manifest(
            format: format, quantization: quantization,
            source: DiffusionReferenceProfile.checkpointID,
            created: ISO8601DateFormatter().string(from: Date()),
            modules: modules, files: files)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(manifest).write(to: destination.appendingPathComponent(manifestName))
        return manifest
    }

    // MARK: - Chargement

    /// Charge un pack : structure quantifiee d'apres le manifeste, modules partages
    /// reconstitues, puis poids. Ne lit jamais le bf16.
    /// - Parameter verifyChecksums: recalcule les SHA-256 (lit tout le pack une fois de plus).
    public static func load(
        from directory: URL,
        includeVision: Bool,
        verifyChecksums: Bool = false
    ) throws -> (model: DiffusionGemmaForBlockDiffusion, config: DiffusionGemmaConfig, manifest: Manifest) {
        let manifest = try readManifest(directory)
        for (name, digest) in manifest.files.sorted(by: { $0.key < $1.key }) {
            let url = directory.appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: url.path) else { throw PackError.missingFile(name) }
            if verifyChecksums, try sha256(of: url) != digest { throw PackError.checksumMismatch(name) }
        }

        let config = try DiffusionGemmaLoader.loadConfig(from: directory)
        let model = DiffusionGemmaForBlockDiffusion(config)
        // Contrat structurel (types, formes, dtypes) : se calcule sur le modele neuf.
        let shared = DiffusionOnTheFlyQuantization.sharedDecoderLeafPaths(model)

        // Structure quantifiee : les poids aleatoires paresseux ne sont jamais evalues.
        MLXNN.quantize(model: model, filter: { path, m in
            guard let q = manifest.modules[path], m is Quantizable, !(m is Quantized) else { return nil }
            return (groupSize: q.groupSize, bits: q.bits, mode: QuantizationMode(rawValue: q.mode) ?? .affine)
        })
        DiffusionOnTheFlyQuantization.shareEncoderModules(model, paths: shared)

        var weights: [String: MLXArray] = [:]
        for name in manifest.files.keys.sorted() {
            let (arrays, _) = try loadArraysAndMetadata(url: directory.appendingPathComponent(name))
            for (key, value) in arrays {
                if !includeVision && isVisionKey(key) { continue }
                weights[key] = value
            }
        }

        // Tout parametre du modele doit venir du pack : directement, ou via un module
        // partage avec l'encodeur (les deux cles pointent alors le meme module).
        let sharedKeys = sharedDecoderParameterKeys(model)
        let missing = model.parameters().flattened().filter { key, array in
            weights[key] == nil && array.size > 0 && !sharedKeys.contains(key) && (includeVision || !isVisionKey(key))
        }.map(\.0)
        guard missing.isEmpty else { throw PackError.missingWeights(missing.sorted()) }

        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.noUnusedKeys, .shapeMismatch])
        eval(model)
        return (model, config, manifest)
    }

    // MARK: - Outils

    static func isVisionKey(_ key: String) -> Bool {
        key.hasPrefix("encoder.vision_tower.") || key.hasPrefix("encoder.embed_vision.")
    }

    /// Cles `decoder.…` portees par un module partage avec l'encodeur.
    static func sharedDecoderParameterKeys(_ model: DiffusionGemmaForBlockDiffusion) -> Set<String> {
        let encoderLeaves = Dictionary(
            model.encoder.languageModel.leafModules().flattened(), uniquingKeysWith: { a, _ in a })
        var keys = Set<String>()
        for (path, module) in model.decoder.leafModules().flattened() {
            guard let source = encoderLeaves[path], source === module else { continue }
            for (param, _) in module.parameters().flattened() {
                keys.insert("decoder.\(path).\(param)")
            }
        }
        return keys
    }

    static func quantizationOf(_ module: Module) -> ModuleQuantization? {
        switch module {
        case let q as QuantizedLinear: return .init(bits: q.bits, groupSize: q.groupSize, mode: q.mode.rawValue)
        case let q as QuantizedEmbedding: return .init(bits: q.bits, groupSize: q.groupSize, mode: q.mode.rawValue)
        case let q as QuantizedSwitchLinear: return .init(bits: q.bits, groupSize: q.groupSize, mode: q.mode.rawValue)
        default: return nil
        }
    }

    static func sha256(of url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while let chunk = try handle.read(upToCount: 64 << 20), !chunk.isEmpty {
            hasher.update(data: chunk)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
