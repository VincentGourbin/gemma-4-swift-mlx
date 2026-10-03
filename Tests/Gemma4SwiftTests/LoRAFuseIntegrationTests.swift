import Testing
import Foundation
import MLX
import MLXLMCommon
@testable import Gemma4Swift

/// Porte K-35 : le modele fusionne (`fuseAndSave`), recharge, se comporte comme la base +
/// adaptateur, et garde tours et fichiers annexes.
///
/// La fusion arrondit `W + scale * B A` en bf16 alors que le chemin non fusionne calcule
/// `x W + scale * x A B` : les deux different d'un arrondi, et une generation gloutonne
/// finit par changer d'argmax (21e a 27e jeton selon l'adaptateur et la version de MLX).
/// D'ou : 16 premiers jetons identiques, et au premier pas un ecart fusionne / adaptateur
/// petit devant l'ecart base seule / adaptateur (l'adaptateur est bien applique).
///   GEMMA4_LORA_BASE_PATH=/Volumes/Lexar/models/mlx-community/gemma-4-e2b-it-bf16 \
///   GEMMA4_LORA_ADAPTER_PATH=".../director-v3" GEMMA4_LORA_FUSE_OUTPUT=/Volumes/Lexar/models/local/fuse-test \
///     Scripts/run-tests.sh -only-testing:Gemma4SwiftTests/LoRAFuseIntegrationTests
private let env = ProcessInfo.processInfo.environment
private let basePath = env["GEMMA4_LORA_BASE_PATH"]
private let adapterPath = env["GEMMA4_LORA_ADAPTER_PATH"]
private let outputPath = env["GEMMA4_LORA_FUSE_OUTPUT"]

@Suite("lora fuse (integration)", .serialized,
       .enabled(if: basePath != nil && adapterPath != nil && outputPath != nil))
struct LoRAFuseIntegrationTests {

    private func greedy(_ container: ModelContainer) async throws -> [Int] {
        try await container.perform { context in
            let ids = try Gemma4Processor.textChatIds(
                userPrompt: "Write a short scene brief for a trailer about a lighthouse keeper.",
                tokenizer: context.tokenizer)
            var iterator = try TokenIterator(
                input: LMInput(tokens: MLXArray(ids.map { Int32($0) })), model: context.model,
                parameters: GenerateParameters(maxTokens: 32, temperature: 0))
            var out: [Int] = []
            while out.count < 32, let t = iterator.next() { out.append(t) }
            return out
        }
    }

    private func firstLogits(_ container: ModelContainer) async throws -> [Float] {
        try await container.perform { context in
            let ids = try Gemma4Processor.textChatIds(
                userPrompt: "Write a short scene brief for a trailer about a lighthouse keeper.",
                tokenizer: context.tokenizer)
            let logits = context.model(
                MLXArray(ids.map { Int32($0) }).reshaped(1, -1),
                cache: try context.model.newCache(parameters: nil))[0, -1].asType(.float32)
            return logits.asArray(Float.self)
        }
    }

    private func relativeGap(_ a: [Float], _ b: [Float]) -> Float {
        let diff = zip(a, b).reduce(Float(0)) { $0 + ($1.0 - $1.1) * ($1.0 - $1.1) }
        return (diff / b.reduce(Float(0)) { $0 + $1 * $1 }).squareRoot()
    }

    @Test("fusionne ~ base + adaptateur (16 jetons greedy, logits), tours et gabarit conserves")
    func testFuseRoundTrip() async throws {
        let base = URL(fileURLWithPath: basePath!)
        let adapter = URL(fileURLWithPath: adapterPath!)
        let output = URL(fileURLWithPath: outputPath!)
        defer { try? FileManager.default.removeItem(at: output) }

        let reference = try await Gemma4Registration.loadContainer(from: base, multimodal: false)
        let baseLogits = try await firstLogits(reference)
        try await Gemma4LoRAInference.loadAdapter(into: reference, from: adapter)
        let expected = try await greedy(reference)
        let adaptedLogits = try await firstLogits(reference)

        try await Gemma4LoRAInference.fuseAndSave(baseDirectory: base, adapterDirectory: adapter, output: output)
        for file in ["chat_template.jinja", "processor_config.json", "config.json", "model.safetensors.index.json"] {
            #expect(FileManager.default.fileExists(atPath: output.appendingPathComponent(file).path), "\(file)")
        }
        #expect(Gemma4ModelCache.hasModelFiles(at: output))

        let fused = try await Gemma4Registration.loadContainer(from: output, multimodal: false)
        let got = try await greedy(fused)
        let divergence = Array(zip(got, expected)).firstIndex { $0 != $1 } ?? min(got.count, expected.count)
        #expect(divergence >= 16, "fusionne \(got.prefix(20)) contre base+adaptateur \(expected.prefix(20))")
        let fusedGap = relativeGap(try await firstLogits(fused), adaptedLogits)
        let adapterEffect = relativeGap(baseLogits, adaptedLogits)
        print("K35 divergence au jeton \(divergence) ; logits 1er pas : fusionne \(fusedGap), base seule \(adapterEffect)")
        #expect(adapterEffect > 0.05)
        #expect(fusedGap < adapterEffect / 10)

        // Les tours (vision, audio) sont toujours la : le multimodal se recharge.
        _ = try await Gemma4Registration.loadContainer(from: output, multimodal: true)
    }
}
