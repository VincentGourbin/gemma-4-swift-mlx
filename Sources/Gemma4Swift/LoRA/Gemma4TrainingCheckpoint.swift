// Checkpoints d'entrainement surs et reprise (K-25, constat A-02).
//
// Avant : `adapter_config.json` n'etait ecrit qu'a la fin (un run interrompu laissait un
// adaptateur inchargeable), la sauvegarde ecrasait en place le seul exemplaire, et rien ne
// permettait de reprendre. Maintenant : config ecrite au demarrage, sauvegardes atomiques
// (fichier temporaire puis renommage), numerotees + `latest`, avec l'etat de l'optimiseur,
// le pas et la graine ; la reprise rejoue le melange sans calcul et continue au pas suivant.

import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// Adam / AdamW au calcul identique a MLXOptimizers (sans correction de biais, decroissance
/// `parameter * (1 - lr * wd)` avant le pas), mais dont l'etat se sauvegarde et se
/// restaure : celui de MLXOptimizers (`stateStorage`) est interne au module.
public final class Gemma4ResumableAdam: Optimizer {
    public var learningRate: Float
    public let betas: (Float, Float)
    public let eps: Float
    public let weightDecay: Float
    private var moments: [String: (MLXArray, MLXArray)] = [:]

    public init(learningRate: Float, betas: (Float, Float) = (0.9, 0.999), eps: Float = 1e-8, weightDecay: Float = 0) {
        self.learningRate = learningRate
        self.betas = betas
        self.eps = eps
        self.weightDecay = weightDecay
    }

    public func update(model: Module, gradients: ModuleParameters) {
        let (b1, b2) = betas
        let parameters = Dictionary(model.parameters().flattened(), uniquingKeysWith: { a, _ in a })
        var updated: [(String, MLXArray)] = []
        for (key, gradient) in gradients.flattened() {
            guard var parameter = parameters[key] else { continue }
            if weightDecay != 0 { parameter = parameter * (1 - learningRate * weightDecay) }
            let (m0, v0) = moments[key] ?? (MLXArray.zeros(like: gradient), MLXArray.zeros(like: gradient))
            let m = b1 * m0 + (1 - b1) * gradient
            let v = b2 * v0 + (1 - b2) * square(gradient)
            moments[key] = (m, v)
            updated.append((key, parameter - learningRate * m / (sqrt(v) + eps)))
        }
        model.update(parameters: ModuleParameters.unflattened(updated))
    }

    public func innerState() -> [MLXArray] {
        moments.values.flatMap { [$0.0, $0.1] }
    }

    func stateArrays() -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (key, (m, v)) in moments {
            out["m." + key] = m
            out["v." + key] = v
        }
        return out
    }

    func restore(_ arrays: [String: MLXArray]) {
        moments = [:]
        for (key, m) in arrays where key.hasPrefix("m.") {
            let name = String(key.dropFirst(2))
            if let v = arrays["v." + name] { moments[name] = (m, v) }
        }
    }
}

/// Etat d'un run, a cote des poids.
public struct Gemma4TrainingState: Codable, Sendable, Equatable {
    /// Dernier pas termine (numerotation 1…N).
    public let iteration: Int
    public let seed: UInt64

    public init(iteration: Int, seed: UInt64) {
        self.iteration = iteration
        self.seed = seed
    }
}

public enum Gemma4TrainingCheckpoint {
    public static let latestWeights = "adapters.safetensors"
    public static let optimizerFile = "optimizer.safetensors"
    public static let stateFile = "training_state.json"

    /// Ecrit un checkpoint complet : poids numerotes + `latest`, etat de l'optimiseur, pas.
    /// Chaque fichier passe par un temporaire renomme : une coupure ne laisse jamais un
    /// fichier a moitie ecrit a la place du precedent.
    public static func write(
        weights: [String: MLXArray], optimizer: (any Optimizer)?, state: Gemma4TrainingState,
        directory: URL, weightsName: String = latestWeights
    ) throws {
        eval(Array(weights.values))
        let numbered = String(format: "%07d_", state.iteration) + weightsName
        try atomicSave(weights, to: directory.appendingPathComponent(numbered))
        try atomicSave(weights, to: directory.appendingPathComponent(weightsName))
        if let adam = optimizer as? Gemma4ResumableAdam {
            let arrays = adam.stateArrays()
            eval(Array(arrays.values))
            try atomicSave(arrays, to: directory.appendingPathComponent(optimizerFile))
        }
        try atomicWrite(try JSONEncoder().encode(state), to: directory.appendingPathComponent(stateFile))
    }

    /// Etat du dernier checkpoint du dossier, s'il y en a un.
    public static func readState(in directory: URL) -> Gemma4TrainingState? {
        guard let data = try? Data(contentsOf: directory.appendingPathComponent(stateFile)) else { return nil }
        return try? JSONDecoder().decode(Gemma4TrainingState.self, from: data)
    }

    /// Recharge poids (`latest`) et etat de l'optimiseur dans un modele deja prepare
    /// (couches LoRA en place).
    public static func restore(
        into model: Module, optimizer: (any Optimizer)?, directory: URL, weightsName: String = latestWeights
    ) throws {
        let weights = try loadArrays(url: directory.appendingPathComponent(weightsName))
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.noUnusedKeys, .shapeMismatch])
        if let adam = optimizer as? Gemma4ResumableAdam {
            let url = directory.appendingPathComponent(optimizerFile)
            if FileManager.default.fileExists(atPath: url.path) { adam.restore(try loadArrays(url: url)) }
        }
        eval(model)
    }

    static func atomicSave(_ arrays: [String: MLXArray], to url: URL) throws {
        let temporary = url.deletingLastPathComponent()
            .appendingPathComponent(".tmp-\(UUID().uuidString).safetensors")
        try MLX.save(arrays: arrays, url: temporary)
        try replace(url, with: temporary)
    }

    static func atomicWrite(_ data: Data, to url: URL) throws {
        let temporary = url.deletingLastPathComponent().appendingPathComponent(".tmp-\(UUID().uuidString)")
        try data.write(to: temporary)
        try replace(url, with: temporary)
    }

    private static func replace(_ url: URL, with temporary: URL) throws {
        if FileManager.default.fileExists(atPath: url.path) {
            _ = try FileManager.default.replaceItemAt(url, withItemAt: temporary)
        } else {
            try FileManager.default.moveItem(at: temporary, to: url)
        }
    }
}
