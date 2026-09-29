import Testing
import Foundation
import MLX
import MLXNN
import MLXLMCommon
import MLXOptimizers
@testable import Gemma4Swift

/// K-26 / K-27 sur un petit modele (CPU, fp32) : reproductibilite, ecretage reel,
/// troncature comptee, full refuse sur un modele quantifie, arret sur annulation.
@Suite("Boucle d'entrainement LoRA", .serialized)
struct LoRATrainingLoopTests {

    static let config = """
    {
        "model_type": "gemma4_text", "hidden_size": 64, "num_hidden_layers": 2,
        "intermediate_size": 128, "num_attention_heads": 2, "head_dim": 32,
        "global_head_dim": 32, "rms_norm_eps": 1e-6, "vocab_size": 128,
        "num_key_value_heads": 1, "num_kv_shared_layers": 0, "sliding_window": 16,
        "sliding_window_pattern": 2, "max_position_embeddings": 256,
        "attention_bias": false, "use_double_wide_mlp": false, "enable_moe_block": false,
        "tie_word_embeddings": true,
        "rope_parameters": {
            "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
            "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}
        },
        "layer_types": ["sliding_attention", "full_attention"]
    }
    """

    private func loraModel() throws -> Gemma4LLMModel {
        let config = try JSONDecoder().decode(Gemma4TextConfig.self, from: Data(Self.config.utf8))
        MLXRandom.seed(1)
        let model = Gemma4LLMModel(config: config)
        MLXRandom.seed(2)
        _ = try LoRAContainer.from(model: model, configuration: Gemma4LoRADefaults.configuration(numLayers: 2))
        return model
    }

    private func samples(_ n: Int, length: Int = 12) -> [TrainingBatchIterator.TokenizedSample] {
        (0 ..< n).map { i in
            .init(tokens: (0 ..< length).map { ($0 * 7 + i * 13) % 128 }, promptOffset: 0)
        }
    }

    private func losses(seed: UInt64, clip: Float = 0, iterations: Int = 20) throws -> ([Float], Gemma4LLMModel) {
        let model = try loraModel()
        var out: [Float] = []
        try trainLoRA(
            model: model, trainSamples: samples(8), validSamples: samples(2),
            optimizer: Adam(learningRate: 1e-2), iterations: iterations,
            stepsPerReport: 1, stepsPerEval: 1_000, seed: seed, gradClipMaxNorm: clip
        ) { progress in
            if case .train(_, let loss, _, _) = progress { out.append(loss) }
            return .more
        }
        return (out, model)
    }

    @Test("meme graine : pertes identiques bit a bit ; autre graine : melange different")
    func testReproducible() throws {
        try Device.withDefaultDevice(.cpu) {
            let (a, _) = try losses(seed: 7)
            let (b, _) = try losses(seed: 7)
            let (c, _) = try losses(seed: 8)
            #expect(a.count == 20)
            #expect(a == b)
            #expect(a != c)
        }
    }

    @Test("ecretage applique : norme minuscule = parametres LoRA presque immobiles")
    func testClipApplied() throws {
        try Device.withDefaultDevice(.cpu) {
            func drift(_ clip: Float) throws -> Float {
                let before = try loraModel().trainableParameters().flattened()
                let (_, model) = try losses(seed: 3, clip: clip, iterations: 3)
                let after = Dictionary(model.trainableParameters().flattened(), uniquingKeysWith: { a, _ in a })
                return before.reduce(Float(0)) { total, pair in
                    total + (after[pair.0].map { abs($0 - pair.1).sum().item(Float.self) } ?? 0)
                }
            }
            // Adam normalise le pas : l'ecretage se voit sur les premiers pas via les moments.
            let free = try drift(0)
            let clipped = try drift(1e-9)
            #expect(free > 0)
            #expect(clipped < free, "clip \(clipped) contre libre \(free)")
        }
    }

    @Test("troncature comptee")
    func testTruncate() {
        let (out, count) = Gemma4LoRATrain.truncate([[1, 2, 3], Array(0 ..< 10), [4]], maxLength: 4)
        #expect(count == 1)
        #expect(out.map(\.count) == [3, 4, 1])
        #expect(Gemma4LoRATrain.truncate([[1, 2, 3]], maxLength: nil).truncated == 0)
    }

    @Test("full refuse sur un modele quantifie")
    func testFullOnQuantized() throws {
        let config = try JSONDecoder().decode(Gemma4TextConfig.self, from: Data(Self.config.utf8))
        let model = Gemma4LLMModel(config: config)
        try Gemma4LoRATrain.checkFullFineTune(model)
        MLXNN.quantize(model: model, groupSize: 32, bits: 4)
        #expect(throws: Gemma4LoRATrain.TrainingSetupError.fullFineTuneOnQuantizedModel) {
            try Gemma4LoRATrain.checkFullFineTune(model)
        }
    }

    private func run(
        _ model: Gemma4LLMModel, optimizer: any Optimizer, iterations: Int, start: Int = 0, directory: URL? = nil
    ) throws -> [Float] {
        var out: [Float] = []
        try trainLoRA(
            model: model, trainSamples: samples(8), validSamples: samples(2),
            optimizer: optimizer, iterations: iterations, stepsPerReport: 1, stepsPerEval: 1_000,
            saveEvery: 5, weightsURL: directory?.appendingPathComponent("adapters.safetensors"),
            seed: 5, startIteration: start, checkpointDirectory: directory
        ) { progress in
            if case .train(_, let loss, _, _) = progress { out.append(loss) }
            return .more
        }
        return out
    }

    @Test("Gemma4ResumableAdam = Adam de MLXOptimizers, bit a bit (et AdamW)")
    func testResumableAdamMatchesMLX() throws {
        try Device.withDefaultDevice(.cpu) {
            let mlxAdam = try run(loraModel(), optimizer: Adam(learningRate: 1e-2), iterations: 12)
            let ours = try run(loraModel(), optimizer: Gemma4ResumableAdam(learningRate: 1e-2), iterations: 12)
            #expect(mlxAdam == ours)
            let mlxAdamW = try run(loraModel(), optimizer: AdamW(learningRate: 1e-2, weightDecay: 0.01), iterations: 12)
            let oursW = try run(loraModel(), optimizer: Gemma4ResumableAdam(learningRate: 1e-2, weightDecay: 0.01), iterations: 12)
            #expect(mlxAdamW == oursW)
        }
    }

    @Test("reprise 10 -> 20 = run continu de 20 pas ; checkpoints numerotes + etat")
    func testResume() throws {
        try Device.withDefaultDevice(.cpu) {
            let dir = FileManager.default.temporaryDirectory.appendingPathComponent("gemma4-resume-\(UUID().uuidString)")
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: dir) }

            let continuous = try run(loraModel(), optimizer: Gemma4ResumableAdam(learningRate: 1e-2), iterations: 20)

            let first = try run(loraModel(), optimizer: Gemma4ResumableAdam(learningRate: 1e-2), iterations: 10, directory: dir)
            let state = try #require(Gemma4TrainingCheckpoint.readState(in: dir))
            #expect(state == Gemma4TrainingState(iteration: 10, seed: 5))
            for file in ["0000005_adapters.safetensors", "0000010_adapters.safetensors", "adapters.safetensors",
                         "optimizer.safetensors", "training_state.json"] {
                #expect(FileManager.default.fileExists(atPath: dir.appendingPathComponent(file).path), "\(file)")
            }
            let leftovers = try FileManager.default.contentsOfDirectory(atPath: dir.path).filter { $0.hasPrefix(".tmp-") }
            #expect(leftovers.isEmpty, "temporaires restants : \(leftovers)")

            let model = try loraModel()
            let optimizer = Gemma4ResumableAdam(learningRate: 1e-2)
            try Gemma4TrainingCheckpoint.restore(into: model, optimizer: optimizer, directory: dir)
            let second = try run(model, optimizer: optimizer, iterations: 20, start: state.iteration, directory: dir)

            #expect(first + second == continuous, "reprise \(second.prefix(3)) contre continu \(continuous[10 ..< 13])")
        }
    }
}

