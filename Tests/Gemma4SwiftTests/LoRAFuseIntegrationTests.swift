import Testing
import Foundation
import MLX
import MLXLMCommon
@testable import Gemma4Swift

/// Porte K-35 : le modele fusionne (`fuseAndSave`), recharge, genere les memes 32 jetons
/// greedy que la base + adaptateur, et garde tours et fichiers annexes.
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

    @Test("fusionne = base + adaptateur (32 jetons greedy), tours et gabarit conserves")
    func testFuseRoundTrip() async throws {
        let base = URL(fileURLWithPath: basePath!)
        let adapter = URL(fileURLWithPath: adapterPath!)
        let output = URL(fileURLWithPath: outputPath!)
        defer { try? FileManager.default.removeItem(at: output) }

        let reference = try await Gemma4Registration.loadContainer(from: base, multimodal: false)
        try await Gemma4LoRAInference.loadAdapter(into: reference, from: adapter)
        let expected = try await greedy(reference)

        try await Gemma4LoRAInference.fuseAndSave(baseDirectory: base, adapterDirectory: adapter, output: output)
        for file in ["chat_template.jinja", "processor_config.json", "config.json", "model.safetensors.index.json"] {
            #expect(FileManager.default.fileExists(atPath: output.appendingPathComponent(file).path), "\(file)")
        }
        #expect(Gemma4ModelCache.hasModelFiles(at: output))

        let fused = try await Gemma4Registration.loadContainer(from: output, multimodal: false)
        let got = try await greedy(fused)
        #expect(got == expected, "fusionne \(got.prefix(8))… contre base+adaptateur \(expected.prefix(8))…")

        // Les tours (vision, audio) sont toujours la : le multimodal se recharge.
        _ = try await Gemma4Registration.loadContainer(from: output, multimodal: true)
    }
}
