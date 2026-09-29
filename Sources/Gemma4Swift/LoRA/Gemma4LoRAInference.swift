// Chargement et gestion des adapters LoRA pour l'inference

import Foundation
import MLX
import MLXLMCommon

/// Utilitaires pour charger/fusionner/retirer des adapters LoRA sur un modele Gemma 4
public enum Gemma4LoRAInference {

    /// Charge un adapter LoRA depuis un repertoire et l'applique au modele
    ///
    /// Le repertoire doit contenir:
    /// - `adapter_config.json` — configuration LoRA
    /// - `adapters.safetensors` — poids de l'adapter
    ///
    /// - Parameters:
    ///   - container: le ModelContainer contenant le modele de base
    ///   - directory: repertoire contenant les fichiers de l'adapter
    public static func loadAdapter(
        into container: ModelContainer,
        from directory: URL
    ) async throws {
        let adapter = try LoRAContainer.from(directory: directory)
        try await container.perform { context in
            try adapter.load(into: context.model)
        }
    }

    /// Fusionne definitivement un adapter LoRA dans les poids du modele
    ///
    /// Apres fusion, l'adapter n'est plus necessaire et le modele peut etre
    /// utilise normalement avec des performances d'inference identiques au modele de base.
    ///
    /// - Parameters:
    ///   - container: le ModelContainer contenant le modele
    ///   - directory: repertoire contenant les fichiers de l'adapter
    public static func fuseAdapter(
        into container: ModelContainer,
        from directory: URL
    ) async throws {
        let adapter = try LoRAContainer.from(directory: directory)
        try await container.perform { context in
            try adapter.fuse(with: context.model)
        }
    }

    /// Fusionne un adaptateur dans le modele de base et ecrit un modele complet (K-35).
    ///
    /// Le resultat part de **tous** les tenseurs du checkpoint d'origine (tours vision et
    /// audio comprises, que le chemin texte ne charge pas) et n'y remplace que ceux du
    /// modele de langue, sous leurs cles d'origine (`language_model.model.…`, identiques
    /// a la hierarchie Swift). Ecrit en shards de ~5 Go avec `model.safetensors.index.json`,
    /// et copie tous les fichiers annexes (config, tokenizer, `chat_template.jinja`,
    /// `processor_config.json`…). Aucune erreur n'est avalee.
    /// - Returns: noms des fichiers de poids ecrits.
    @discardableResult
    public static func fuseAndSave(
        baseDirectory: URL, adapterDirectory: URL, output: URL, shardBytes: Int = 5 << 30
    ) async throws -> [String] {
        let fm = FileManager.default
        let weightFiles = try fm.contentsOfDirectory(at: baseDirectory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !weightFiles.isEmpty else { throw Gemma4LoRAError.trainingFailed("aucun safetensors dans \(baseDirectory.path)") }

        // Couches ciblees par l'adaptateur : leurs poids doivent exister dans le checkpoint.
        let adapterFile = adapterDirectory.appendingPathComponent("adapters.safetensors")
        let targets = Set(try loadArrays(url: adapterFile).keys.compactMap { key -> String? in
            guard key.hasSuffix(".lora_a") else { return nil }
            return String(key.dropLast(".lora_a".count))
        })

        let container = try await Gemma4Registration.loadContainer(from: baseDirectory, multimodal: false)
        try await fuseAdapter(into: container, from: adapterDirectory)
        try fm.createDirectory(at: output, withIntermediateDirectories: true)

        // Tout le travail sur les tenseurs reste dans `perform` (MLXArray n'est pas Sendable).
        let names = try await container.perform { context -> [String] in
            var tensors: [String: MLXArray] = [:]
            for file in weightFiles {
                for (key, value) in try loadArrays(url: file) { tensors[key] = value }
            }
            let missing = targets.filter { tensors["\($0).weight"] == nil }.sorted()
            guard missing.isEmpty else {
                throw Gemma4LoRAError.trainingFailed(
                    "couches de l'adaptateur absentes du checkpoint : \(missing.prefix(3).joined(separator: ", "))")
            }
            let fused = Dictionary(context.model.parameters().flattened(), uniquingKeysWith: { a, _ in a })
            let untouched = targets.filter { fused["\($0).weight"] == nil }
            guard untouched.isEmpty else {
                throw Gemma4LoRAError.trainingFailed("poids fusionnes introuvables pour \(untouched.sorted().prefix(3))")
            }
            for (key, value) in fused where tensors[key] != nil { tensors[key] = value }

            var shards: [[(String, MLXArray)]] = [[]]
            var current = 0
            for (key, array) in tensors.sorted(by: { $0.key < $1.key }) {
                if current > 0 && current + array.nbytes > shardBytes {
                    shards.append([])
                    current = 0
                }
                shards[shards.count - 1].append((key, array))
                current += array.nbytes
            }
            var weightMap: [String: String] = [:]
            var names: [String] = []
            for (i, shard) in shards.enumerated() {
                let name = String(format: "model-%05d-of-%05d.safetensors", i + 1, shards.count)
                try MLX.save(arrays: Dictionary(shard, uniquingKeysWith: { a, _ in a }),
                             metadata: ["format": "mlx"], url: output.appendingPathComponent(name))
                for (key, _) in shard { weightMap[key] = name }
                names.append(name)
            }
            let total = tensors.values.reduce(0) { $0 + $1.nbytes }
            let index: [String: Any] = ["metadata": ["total_size": total], "weight_map": weightMap]
            try JSONSerialization.data(withJSONObject: index, options: [.prettyPrinted, .sortedKeys])
                .write(to: output.appendingPathComponent("model.safetensors.index.json"))
            return names
        }

        for item in try fm.contentsOfDirectory(at: baseDirectory, includingPropertiesForKeys: nil) {
            let name = item.lastPathComponent
            guard !name.hasSuffix(".safetensors"), name != "model.safetensors.index.json", !name.hasPrefix(".") else { continue }
            var isDir: ObjCBool = false
            guard fm.fileExists(atPath: item.path, isDirectory: &isDir), !isDir.boolValue else { continue }
            let target = output.appendingPathComponent(name)
            if fm.fileExists(atPath: target.path) { try fm.removeItem(at: target) }
            try fm.copyItem(at: item, to: target)
        }
        return names
    }

    /// Retire un adapter LoRA et restaure les poids de base du modele
    ///
    /// - Parameters:
    ///   - container: le ModelContainer contenant le modele avec adapter
    ///   - directory: repertoire contenant la config de l'adapter (pour connaitre les layers)
    public static func unloadAdapter(
        from container: ModelContainer,
        directory: URL
    ) async throws {
        let adapter = try LoRAContainer.from(directory: directory)
        await container.perform { context in
            adapter.unload(from: context.model)
        }
    }
}

public enum Gemma4LoRAError: LocalizedError {
    case incompatibleModel
    case trainingFailed(String)
    case adapterNotFound(URL)

    public var errorDescription: String? {
        switch self {
        case .incompatibleModel:
            return "Le modele n'est pas compatible avec LoRA (doit conformer a LanguageModel + LoRAModel)"
        case .trainingFailed(let reason):
            return "Echec de l'entrainement: \(reason)"
        case .adapterNotFound(let url):
            return "Adapter introuvable a \(url.path())"
        }
    }
}
