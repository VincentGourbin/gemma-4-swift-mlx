// Tests pour le fine-tuning LoRA

import XCTest
@testable import Gemma4Swift
import MLX
import MLXLMCommon

final class LoRATests: XCTestCase {

    // MARK: - Chat Template Tests

    func testApplyGemma4ChatTemplateSimple() {
        let messages: [ChatMessage] = [
            ChatMessage(role: "user", content: "Hello"),
            ChatMessage(role: "assistant", content: "Hi there!"),
        ]

        let result = applyGemma4ChatTemplate(messages: messages)

        XCTAssertEqual(result, """
        <start_of_turn>user
        Hello<end_of_turn>
        <start_of_turn>model
        Hi there!<end_of_turn>
        """)
    }

    func testApplyGemma4ChatTemplateWithSystem() {
        let messages: [ChatMessage] = [
            ChatMessage(role: "system", content: "Tu es un expert Swift."),
            ChatMessage(role: "user", content: "Comment faire un struct?"),
            ChatMessage(role: "assistant", content: "struct Foo { }"),
        ]

        let result = applyGemma4ChatTemplate(messages: messages)

        XCTAssertTrue(result.contains("<start_of_turn>system\nTu es un expert Swift.<end_of_turn>"))
        XCTAssertTrue(result.contains("<start_of_turn>user\nComment faire un struct?<end_of_turn>"))
        XCTAssertTrue(result.contains("<start_of_turn>model\nstruct Foo { }<end_of_turn>"))
    }

    func testApplyGemma4ChatTemplateModelRole() {
        // "model" et "assistant" doivent tous les deux mapper vers "model"
        let messages1 = [ChatMessage(role: "assistant", content: "test")]
        let messages2 = [ChatMessage(role: "model", content: "test")]

        let result1 = applyGemma4ChatTemplate(messages: messages1)
        let result2 = applyGemma4ChatTemplate(messages: messages2)

        XCTAssertEqual(result1, result2)
        XCTAssertTrue(result1.contains("<start_of_turn>model"))
    }

    func testApplyGemma4ChatTemplateMultiTurn() {
        let messages: [ChatMessage] = [
            ChatMessage(role: "user", content: "Bonjour"),
            ChatMessage(role: "assistant", content: "Bonjour!"),
            ChatMessage(role: "user", content: "Comment ca va?"),
            ChatMessage(role: "assistant", content: "Bien merci!"),
        ]

        let result = applyGemma4ChatTemplate(messages: messages)
        let turns = result.components(separatedBy: "<start_of_turn>").filter { !$0.isEmpty }

        XCTAssertEqual(turns.count, 4)
    }

    // MARK: - Data Loading Tests

    func testLoadGemma4TrainingDataTextFormat() throws {
        let tmpDir = FileManager.default.temporaryDirectory
            .appending(component: "lora_test_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        let trainContent = """
        {"text": "Ligne un."}
        {"text": "Ligne deux."}
        {"text": "Ligne trois."}
        """
        try trainContent.write(to: tmpDir.appending(component: "train.jsonl"), atomically: true, encoding: .utf8)

        let data = try loadGemma4TrainingData(directory: tmpDir, name: "train")
        XCTAssertEqual(data.count, 3)
        XCTAssertEqual(data[0], "Ligne un.")
        XCTAssertEqual(data[1], "Ligne deux.")
    }

    func testLoadGemma4TrainingDataChatFormat() throws {
        let tmpDir = FileManager.default.temporaryDirectory
            .appending(component: "lora_test_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        let trainContent = """
        {"messages": [{"role": "user", "content": "Bonjour"}, {"role": "assistant", "content": "Salut!"}]}
        """
        try trainContent.write(to: tmpDir.appending(component: "train.jsonl"), atomically: true, encoding: .utf8)

        let data = try loadGemma4TrainingData(directory: tmpDir, name: "train")
        XCTAssertEqual(data.count, 1)
        XCTAssertTrue(data[0].contains("<start_of_turn>user"))
        XCTAssertTrue(data[0].contains("Bonjour"))
        XCTAssertTrue(data[0].contains("<start_of_turn>model"))
        XCTAssertTrue(data[0].contains("Salut!"))
    }

    func testLoadGemma4TrainingDataMixedFormat() throws {
        let tmpDir = FileManager.default.temporaryDirectory
            .appending(component: "lora_test_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        let trainContent = """
        {"text": "Ligne texte."}
        {"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "R"}]}
        """
        try trainContent.write(to: tmpDir.appending(component: "train.jsonl"), atomically: true, encoding: .utf8)

        let data = try loadGemma4TrainingData(directory: tmpDir, name: "train")
        XCTAssertEqual(data.count, 2)
        XCTAssertEqual(data[0], "Ligne texte.")
        XCTAssertTrue(data[1].contains("<start_of_turn>"))
    }

    func testLoadGemma4TrainingDataTxtFormat() throws {
        let tmpDir = FileManager.default.temporaryDirectory
            .appending(component: "lora_test_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        let content = "Premiere ligne\nDeuxieme ligne\nTroisieme ligne"
        try content.write(to: tmpDir.appending(component: "train.txt"), atomically: true, encoding: .utf8)

        let data = try loadGemma4TrainingData(directory: tmpDir, name: "train")
        XCTAssertEqual(data.count, 3)
    }

    func testLoadGemma4TrainingDataFileNotFound() {
        let tmpDir = FileManager.default.temporaryDirectory
            .appending(component: "lora_test_nonexistent_\(UUID().uuidString)")

        XCTAssertThrowsError(try loadGemma4TrainingData(directory: tmpDir, name: "train")) { error in
            XCTAssertTrue(error is Gemma4LoRADataError)
        }
    }

    // MARK: - Config Tests

    func testLoRAConfigDefaultsE2B() {
        let config = Gemma4LoRADefaults.configuration(for: .e2b)
        XCTAssertEqual(config.numLayers, 8)
        XCTAssertEqual(config.loraParameters.rank, 8)
        XCTAssertEqual(config.loraParameters.scale, 20.0)
        XCTAssertEqual(config.fineTuneType, .lora)
    }

    func testLoRAConfigDefaultsE4B() {
        let config = Gemma4LoRADefaults.configuration(for: .e4b)
        XCTAssertEqual(config.numLayers, 12)
    }

    func testLoRAConfigDefaults31B() {
        let config = Gemma4LoRADefaults.configuration(for: .dense31b)
        XCTAssertEqual(config.numLayers, 16)
    }

    func testLoRAConfigDefaultsA4B() {
        let config = Gemma4LoRADefaults.configuration(for: .a4b)
        XCTAssertEqual(config.numLayers, 10)
    }

    func testLoRAConfigCustom() {
        let config = Gemma4LoRADefaults.configuration(
            for: .e2b,
            rank: 16,
            scale: 32.0,
            numLayers: 4
        )
        XCTAssertEqual(config.numLayers, 4)
        XCTAssertEqual(config.loraParameters.rank, 16)
        XCTAssertEqual(config.loraParameters.scale, 32.0)
    }

    func testModelFamilyDetection() {
        XCTAssertEqual(
            Gemma4LoRADefaults.ModelFamily.from(modelId: "mlx-community/gemma-4-e2b-it-4bit"),
            .e2b
        )
        XCTAssertEqual(
            Gemma4LoRADefaults.ModelFamily.from(modelId: "mlx-community/gemma-4-e4b-it-8bit"),
            .e4b
        )
        XCTAssertEqual(
            Gemma4LoRADefaults.ModelFamily.from(modelId: "mlx-community/gemma-4-31b-it-4bit"),
            .dense31b
        )
        XCTAssertEqual(
            Gemma4LoRADefaults.ModelFamily.from(modelId: "mlx-community/gemma-4-26b-a4b-it-4bit"),
            .a4b
        )
    }

    // MARK: - Training Batch Iterator Tests

    func testBatchIteratorSingleSample() {
        let samples = [
            TrainingBatchIterator.TokenizedSample(tokens: [2, 105, 882, 107, 100, 105, 4368, 107, 200, 106], promptOffset: 8)
        ]
        var iter = TrainingBatchIterator(samples: samples, batchSize: 1, train: false)

        let (batch, lengths) = iter.next()!
        XCTAssertEqual(batch.shape, [1, 10])
        XCTAssertEqual(lengths[0, 0].item(Int32.self), 8)   // prompt offset
        XCTAssertEqual(lengths[0, 1].item(Int32.self), 10)  // total length
    }

    func testBatchIteratorPadding() {
        // Two samples of different lengths — should pad to max length
        // Iterator sorts by length, so shorter sample comes first (index 0)
        let samples = [
            TrainingBatchIterator.TokenizedSample(tokens: [1, 2, 3, 4, 5], promptOffset: 0),
            TrainingBatchIterator.TokenizedSample(tokens: [1, 2, 3], promptOffset: 0),
        ]
        var iter = TrainingBatchIterator(samples: samples, batchSize: 2, train: false)

        let (batch, _) = iter.next()!
        XCTAssertEqual(batch.shape, [2, 5])  // padded to max length
        // Shorter sample (index 0 after sorting) should have zeros at the end
        XCTAssertEqual(batch[0, 3].item(Int32.self), 0)
        XCTAssertEqual(batch[0, 4].item(Int32.self), 0)
    }

    func testBatchIteratorSortsByLength() {
        // Samples of different lengths
        let samples = [
            TrainingBatchIterator.TokenizedSample(tokens: [1, 2, 3, 4, 5, 6, 7, 8], promptOffset: 0),
            TrainingBatchIterator.TokenizedSample(tokens: [1, 2], promptOffset: 0),
            TrainingBatchIterator.TokenizedSample(tokens: [1, 2, 3, 4], promptOffset: 0),
        ]
        // With batch_size=1 and train=false, batches come in sorted order
        var iter = TrainingBatchIterator(samples: samples, batchSize: 1, train: false)

        let (b1, _) = iter.next()!
        let (b2, _) = iter.next()!
        let (b3, _) = iter.next()!
        XCTAssertEqual(b1.shape[1], 2)  // shortest first
        XCTAssertEqual(b2.shape[1], 4)
        XCTAssertEqual(b3.shape[1], 8)  // longest last
    }

    func testBatchIteratorPromptOffset() {
        // Simulate a chat sample with known token pattern
        // <|turn> = 105, model = 4368, \n = 107
        let tokens = [2, 105, 882, 107, 42, 43, 105, 4368, 107, 200, 201, 106]
        let sample = TrainingBatchIterator.TokenizedSample(tokens: tokens, promptOffset: 9)

        var iter = TrainingBatchIterator(samples: [sample], batchSize: 1, train: false)
        let (_, lengths) = iter.next()!

        XCTAssertEqual(lengths[0, 0].item(Int32.self), 9)   // offset after <|turn>model\n
        XCTAssertEqual(lengths[0, 1].item(Int32.self), 12)  // total length
    }

    func testBatchIteratorNoMaskingOffset() {
        let sample = TrainingBatchIterator.TokenizedSample(tokens: [1, 2, 3, 4], promptOffset: 0)
        var iter = TrainingBatchIterator(samples: [sample], batchSize: 1, train: false)
        let (_, lengths) = iter.next()!

        XCTAssertEqual(lengths[0, 0].item(Int32.self), 0)  // no masking = offset 0
    }

    // MARK: - KV Sharing Config Tests

    func testKVSharingConfigResolution() throws {
        // E2B-like: 35 layers, KV sharing from layer 15
        let json = """
        {
            "model_type": "gemma4_text",
            "hidden_size": 1536,
            "num_hidden_layers": 35,
            "intermediate_size": 12288,
            "num_attention_heads": 8,
            "head_dim": 256,
            "vocab_size": 262144,
            "num_key_value_heads": 1,
            "num_kv_shared_layers": 20,
            "sliding_window_pattern": 5
        }
        """
        let config = try JSONDecoder().decode(Gemma4TextConfig.self, from: json.data(using: .utf8)!)

        // firstKvSharedLayerIdx = 35 - 20 = 15
        XCTAssertEqual(config.firstKvSharedLayerIdx, 15)

        // resolvedLayerTypes: pattern [S,S,S,S,F] repeated
        let types = config.resolvedLayerTypes
        XCTAssertEqual(types.count, 35)
        XCTAssertEqual(types[4], "full_attention")
        XCTAssertEqual(types[9], "full_attention")
        XCTAssertEqual(types[14], "full_attention")  // last non-shared full attention
        XCTAssertEqual(types[3], "sliding_attention")
        XCTAssertEqual(types[13], "sliding_attention") // last non-shared sliding attention

        // Verify the mapping logic: shared layers should map to
        // the last non-shared layer of the same type
        let concreteLayers = Array(types[..<15])
        let sharedFullIdx = concreteLayers.lastIndex(of: "full_attention")!
        let sharedSlidingIdx = concreteLayers.lastIndex(of: "sliding_attention")!

        XCTAssertEqual(sharedFullIdx, 14)   // layer 14 is the last full_attention before 15
        XCTAssertEqual(sharedSlidingIdx, 13) // layer 13 is the last sliding before 15

        // All LoRA-adapted layers (19-34 with numLayers=16) are shared
        for i in 19 ..< 35 {
            XCTAssertGreaterThanOrEqual(i, config.firstKvSharedLayerIdx,
                "LoRA layer \(i) should be in the shared range")
        }
    }

    func testTrainingConfigDefaults() {
        let config = Gemma4LoRATrain.TrainingConfig()
        XCTAssertEqual(config.fineTuneType, .lora)
        XCTAssertEqual(config.loraRank, 8)
        XCTAssertEqual(config.loraScale, 20.0)
        XCTAssertEqual(config.learningRate, 1e-5)
        XCTAssertEqual(config.batchSize, 1)
        XCTAssertFalse(config.maskPrompt)
    }

    func testDoRAConfig() {
        let config = Gemma4LoRADefaults.configuration(for: .e2b, useDora: true)
        XCTAssertEqual(config.fineTuneType, .dora)
    }
}
