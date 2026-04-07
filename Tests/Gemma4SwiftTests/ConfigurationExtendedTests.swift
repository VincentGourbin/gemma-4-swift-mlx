import XCTest
import Foundation
@testable import Gemma4Swift

final class ConfigurationExtendedTests: XCTestCase {

    // MARK: - Gemma4VisionConfig defaults

    func testVisionConfigDefaults() {
        let config = Gemma4VisionConfig.defaultConfig
        XCTAssertEqual(config.hiddenSize, 768)
        XCTAssertEqual(config.numHiddenLayers, 16)
        XCTAssertEqual(config.patchSize, 16)
        XCTAssertEqual(config.poolingKernelSize, 3)
        XCTAssertEqual(config.defaultOutputLength, 280)
    }

    func testVisionConfigMaxPatches() {
        // maxPatches = defaultOutputLength * poolingKernelSize * poolingKernelSize
        // = 280 * 3 * 3 = 2520
        let config = Gemma4VisionConfig.defaultConfig
        XCTAssertEqual(config.maxPatches, 2520)
    }

    func testVisionConfigRopeTheta() {
        // No ropeParameters in defaultConfig → falls back to 100.0
        let config = Gemma4VisionConfig.defaultConfig
        XCTAssertEqual(config.ropeTheta, 100.0, accuracy: 1e-6)
    }

    // MARK: - Gemma4AudioConfig defaults

    func testAudioConfigDefaults() {
        let config = Gemma4AudioConfig.defaultConfig
        XCTAssertEqual(config.hiddenSize, 1024)
        XCTAssertEqual(config.numHiddenLayers, 12)
        XCTAssertEqual(config.attentionChunkSize, 12)
        XCTAssertEqual(config.attentionContextLeft, 13)
        XCTAssertEqual(config.attentionContextRight, 0)
        XCTAssertEqual(config.outputProjDims, 1536)
        XCTAssertEqual(config.convKernelSize, 5)
        XCTAssertEqual(config.residualWeight, 0.5, accuracy: 1e-6)
        XCTAssertEqual(config.subsamplingConvChannels, [128, 32])
    }

    // MARK: - JSON decoding

    func testAudioConfigDecoding() throws {
        let json = """
        {
            "hidden_size": 1024,
            "num_hidden_layers": 12,
            "num_attention_heads": 8,
            "hidden_act": "silu",
            "subsampling_conv_channels": [128, 32],
            "conv_kernel_size": 5,
            "residual_weight": 0.5,
            "attention_chunk_size": 12,
            "attention_context_left": 13,
            "attention_context_right": 0,
            "attention_logit_cap": 50.0,
            "attention_invalid_logits_value": -1e9,
            "use_clipped_linears": true,
            "rms_norm_eps": 1e-6,
            "gradient_clipping": 1e10,
            "output_proj_dims": 1536
        }
        """
        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(Gemma4AudioConfig.self, from: data)

        XCTAssertEqual(config.hiddenSize, 1024)
        XCTAssertEqual(config.numHiddenLayers, 12)
        XCTAssertEqual(config.numAttentionHeads, 8)
        XCTAssertEqual(config.hiddenAct, "silu")
        XCTAssertEqual(config.subsamplingConvChannels, [128, 32])
        XCTAssertEqual(config.convKernelSize, 5)
        XCTAssertEqual(config.residualWeight, 0.5, accuracy: 1e-6)
        XCTAssertEqual(config.attentionChunkSize, 12)
        XCTAssertEqual(config.attentionContextLeft, 13)
        XCTAssertEqual(config.attentionContextRight, 0)
        XCTAssertEqual(config.attentionLogitCap, 50.0, accuracy: 1e-6)
        XCTAssertTrue(config.useClippedLinears)
        XCTAssertEqual(config.outputProjDims, 1536)
    }

    func testVisionConfigDecoding() throws {
        let json = """
        {
            "model_type": "gemma4_vision",
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_hidden_layers": 16,
            "num_attention_heads": 12,
            "num_key_value_heads": 12,
            "head_dim": 64,
            "global_head_dim": 64,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 131072,
            "patch_size": 16,
            "pooling_kernel_size": 3,
            "position_embedding_size": 10240,
            "default_output_length": 280,
            "use_clipped_linears": true,
            "standardize": false
        }
        """
        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(Gemma4VisionConfig.self, from: data)

        XCTAssertEqual(config.modelType, "gemma4_vision")
        XCTAssertEqual(config.hiddenSize, 768)
        XCTAssertEqual(config.intermediateSize, 3072)
        XCTAssertEqual(config.numHiddenLayers, 16)
        XCTAssertEqual(config.numAttentionHeads, 12)
        XCTAssertEqual(config.headDim, 64)
        XCTAssertEqual(config.globalHeadDim, 64)
        XCTAssertEqual(config.patchSize, 16)
        XCTAssertEqual(config.poolingKernelSize, 3)
        XCTAssertEqual(config.positionEmbeddingSize, 10240)
        XCTAssertEqual(config.defaultOutputLength, 280)
        XCTAssertTrue(config.useClippedLinears)
        XCTAssertFalse(config.standardize)
        XCTAssertNil(config.ropeParameters)
        // Computed properties
        XCTAssertEqual(config.maxPatches, 2520)
        XCTAssertEqual(config.ropeTheta, 100.0, accuracy: 1e-6)
    }
}
