// Smoke test pour valider le chargement des poids du drafter Assistant (MTP).
//
// Telecharge le checkpoint depuis HuggingFace, decode le config.json, instancie
// Gemma4AssistantDraftModel, sanitize les poids et fait un update(parameters:verify:[.all])
// qui echoue si une cle est manquante, surplus ou de mauvaise shape.
//
// Usage:
//   gemma4-cli mtp-smoke                                     # default: google/gemma-4-E2B-it-assistant
//   gemma4-cli mtp-smoke --repo google/gemma-4-E4B-it-assistant
//   gemma4-cli mtp-smoke --no-forward                        # skip le forward synthetique

import ArgumentParser
import Foundation
import MLX
import MLXNN
import Gemma4Swift

struct MtpSmoke: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "mtp-smoke",
        abstract: "Valide le chargement des poids du drafter Assistant (Gemma 4 MTP)"
    )

    @Option(name: .long, help: "Repo HuggingFace du drafter Assistant")
    var repo: String = "google/gemma-4-E2B-it-assistant"

    @Option(name: .long, help: "Token HuggingFace (pour modeles prives)")
    var hfToken: String?

    @Flag(name: .long, help: "Skip le forward synthetique apres chargement")
    var noForward: Bool = false

    @Flag(name: .long, help: "Forcer le re-telechargement")
    var force: Bool = false

    func run() async throws {
        print("[mtp-smoke] target: \(repo)")
        print("[mtp-smoke] step 1/4: download")

        let modelDir = try await Gemma4ModelDownloader.download(
            modelId: repo,
            token: resolveHFToken(hfToken),
            force: force
        ) { progress in
            print("\r  \(Int(progress.fraction * 100))% — \(progress.currentFile)              ", terminator: "")
            fflush(stdout)
        }
        print()
        print("  -> \(modelDir.path)")

        // Step 2: decode config
        print("[mtp-smoke] step 2/4: decode config.json")
        let configURL = modelDir.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            print("  ERROR: config.json absent dans \(modelDir.path)")
            throw ExitCode.failure
        }
        let configData = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(Gemma4AssistantConfig.self, from: configData)
        print("  model_type=\(config.modelType)")
        print("  num_layers=\(config.textConfig.numHiddenLayers), hidden=\(config.textConfig.hiddenSize)")
        print("  vocab=\(config.textConfig.vocabSize), backbone=\(config.backboneHiddenSize)")
        print("  num_centroids=\(config.numCentroids), top_k=\(config.centroidIntermediateTopK)")
        print("  use_ordered_embeddings=\(config.useOrderedEmbeddings), tie_word_embeddings=\(config.tieWordEmbeddings)")

        // Step 3: load + sanitize weights
        print("[mtp-smoke] step 3/4: load + sanitize safetensors")
        let safetensorsURLs = try findSafetensors(in: modelDir)
        guard !safetensorsURLs.isEmpty else {
            print("  ERROR: aucun fichier .safetensors dans \(modelDir.path)")
            throw ExitCode.failure
        }
        print("  found \(safetensorsURLs.count) safetensors file(s)")

        var rawWeights: [String: MLXArray] = [:]
        for url in safetensorsURLs {
            let arrays = try MLX.loadArrays(url: url)
            for (k, v) in arrays {
                rawWeights[k] = v
            }
        }
        print("  loaded \(rawWeights.count) raw tensors")

        let sanitized = Gemma4AssistantWeightSanitizer.sanitize(
            weights: rawWeights,
            tieWordEmbeddings: config.tieWordEmbeddings
        )
        print("  sanitized to \(sanitized.count) tensors")

        // Step 4: instantiate + update(verify: .all)
        print("[mtp-smoke] step 4/4: instantiate + update(verify: .all)")
        let drafter = Gemma4AssistantDraftModel(config)

        // Liste les parametres du module avant update — pour diagnostic en cas d'echec
        let modelParams = drafter.parameters().flattened()
        let modelKeys = Set(modelParams.map { $0.0 })
        let weightKeys = Set(sanitized.keys)

        let missingFromWeights = modelKeys.subtracting(weightKeys)
        let extraInWeights = weightKeys.subtracting(modelKeys)

        if !missingFromWeights.isEmpty {
            print("  WARNING: \(missingFromWeights.count) parametres du module sans cle correspondante dans les poids:")
            for k in missingFromWeights.sorted().prefix(20) {
                print("    - \(k)")
            }
            if missingFromWeights.count > 20 {
                print("    ... et \(missingFromWeights.count - 20) de plus")
            }
        }
        if !extraInWeights.isEmpty {
            print("  WARNING: \(extraInWeights.count) cles dans les poids sans parametre correspondant dans le module:")
            for k in extraInWeights.sorted().prefix(20) {
                print("    - \(k)")
            }
            if extraInWeights.count > 20 {
                print("    ... et \(extraInWeights.count - 20) de plus")
            }
        }

        let parameters = ModuleParameters.unflattened(sanitized)
        do {
            try drafter.update(parameters: parameters, verify: .all)
            print("  OK: update(verify: .all) sans erreur")
        } catch {
            print("  ERROR update: \(error)")
            throw ExitCode.failure
        }

        if !noForward {
            print("[mtp-smoke] bonus: forward synthetique pour verifier les shapes a posteriori")
            try synteticForward(drafter: drafter, config: config)
        }

        print("\n[mtp-smoke] SUCCES — \(repo) charge proprement.")
    }

    private func findSafetensors(in dir: URL) throws -> [URL] {
        let contents = try FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
        return contents
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    private func synteticForward(drafter: Gemma4AssistantDraftModel, config: Gemma4AssistantConfig) throws {
        let B = 1
        let L = 1
        let kvLen = 8
        let numKVHeads = config.textConfig.numKeyValueHeads
        let slidingHeadDim = config.textConfig.headDim
        let fullHeadDim = config.textConfig.globalHeadDim

        let sharedKV: SharedKVStates = [
            "sliding_attention": (
                keys: MLXRandom.normal([B, numKVHeads, kvLen, slidingHeadDim]),
                values: MLXRandom.normal([B, numKVHeads, kvLen, slidingHeadDim])
            ),
            "full_attention": (
                keys: MLXRandom.normal([B, numKVHeads, kvLen, fullHeadDim]),
                values: MLXRandom.normal([B, numKVHeads, kvLen, fullHeadDim])
            ),
        ]

        let inputs = MLXRandom.normal([B, L, 2 * config.backboneHiddenSize])
        let (lastHidden, logits) = drafter(
            inputsEmbeds: inputs,
            sharedKVStates: sharedKV,
            position: kvLen
        )
        eval(lastHidden, logits)

        let expectedHidden = [B, L, config.backboneHiddenSize]
        let expectedLogits = [B, L, config.textConfig.vocabSize]
        guard lastHidden.shape == expectedHidden, logits.shape == expectedLogits else {
            print("  shape mismatch: lastHidden=\(lastHidden.shape) (expected \(expectedHidden)), logits=\(logits.shape) (expected \(expectedLogits))")
            throw ExitCode.failure
        }

        let hasNaN = isNaN(logits).any().item(Bool.self) || isNaN(lastHidden).any().item(Bool.self)
        guard !hasNaN else {
            print("  ERROR: NaN dans logits ou lastHidden")
            throw ExitCode.failure
        }

        // Top-1 token sur les logits — juste pour avoir un signe de vie
        let top1 = argMax(logits, axis: -1).reshaped(-1).item(Int32.self)
        print("  shapes OK, no NaN, top1 token (sur input random) = \(top1)")
    }
}

import MLXRandom
