// Tests pour le fine-tuning LoRA

import XCTest
@testable import Gemma4Swift
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
        XCTAssertEqual(config.numLayers, 8)
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
}
