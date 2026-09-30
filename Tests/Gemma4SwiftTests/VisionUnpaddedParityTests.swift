import Testing
import Foundation
import MLX
import MLXNN
@testable import Gemma4Swift

/// K-18 : l'encodeur vision n'encode plus que les patches reels (plus de padding a
/// `maxPatches` ni de masque L x L). Sortie identique aux arrondis pres a l'ancien chemin.
@Suite("Vision sans padding", .serialized)
struct VisionUnpaddedParityTests {

    private func model() throws -> VisionModel {
        let json = """
            {"model_type": "gemma4_vision", "hidden_size": 32, "intermediate_size": 64,
             "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": 16,
             "global_head_dim": 16, "rms_norm_eps": 1e-6, "max_position_embeddings": 64, "patch_size": 16,
             "pooling_kernel_size": 2, "position_embedding_size": 64, "default_output_length": 4,
             "use_clipped_linears": false, "standardize": false}
            """
        let config = try JSONDecoder().decode(Gemma4VisionConfig.self, from: Data(json.utf8))
        MLXRandom.seed(11)
        let model = VisionModel(config)
        // Table de position aleatoire (init a 1 : toutes les positions pareilles).
        model.patchEmbedder.update(parameters: ModuleParameters.unflattened(
            ["position_embedding_table": MLXRandom.normal([2, 64, 32]) * 0.1]))
        return model
    }

    private func maxRelError(_ a: MLXArray, _ b: MLXArray) -> Float {
        let diff = abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
        let scale = abs(b.asType(.float32)).max().item(Float.self)
        return diff / max(scale, 1e-6)
    }

    @Test("image partielle (12 patches sur 16) et image a nombre de patches = sortie (4)",
          arguments: [(64, 48), (32, 32)])
    func testParity(size: (Int, Int)) throws {
        // Sur GPU : l'ancien chemin paddé n'est valide qu'avec le SDPA fusionne. Sur CPU
        // (repli non fusionne), ses lignes entierement masquees donnent des NaN qui se
        // propagent — le nouveau chemin, sans ligne masquee, n'a pas ce defaut.
        do {
            let vision = try model()
            let pixels = MLXRandom.uniform(low: 0, high: 1, [1, 3, size.0, size.1])
            vision.padToMaxPatches = true
            let padded = vision(pixels)
            vision.padToMaxPatches = false
            let unpadded = vision(pixels)
            eval(padded, unpadded)
            let paddedNaN = any(isNaN(padded)).item(Bool.self)
            let unpaddedNaN = any(isNaN(unpadded)).item(Bool.self)
            #expect(!paddedNaN && !unpaddedNaN)
            #expect(unpadded.shape == padded.shape)
            let error = maxRelError(unpadded, padded)
            #expect(error < 1e-4, "ecart relatif \(error)")
        }
    }

    @Test("CPU : le chemin sans padding ne produit pas de NaN")
    func testNoNaNOnCPU() throws {
        try Device.withDefaultDevice(.cpu) {
            let out = try model()(MLXRandom.uniform(low: 0, high: 1, [1, 3, 64, 48]))
            #expect(!any(isNaN(out)).item(Bool.self))
        }
    }

    @Test("bf16 en entree : la sortie reste en bf16 (plus de promotion fp32)")
    func testDTypeKept() throws {
        try Device.withDefaultDevice(.cpu) {
            let vision = try model()
            vision.update(parameters: ModuleParameters.unflattened(
                vision.parameters().flattened().map { ($0.0, $0.1.asType(.bfloat16)) }))
            let out = vision(MLXRandom.uniform(low: 0, high: 1, [1, 3, 64, 48]))
            #expect(out.dtype == .bfloat16)
        }
    }

    @Test("embedding de position par lecture de table = one-hot x matmul (port Python)")
    func testPositionEmbeddingGather() throws {
        try Device.withDefaultDevice(.cpu) {
            let vision = try model()
            let embedder = vision.patchEmbedder
            let positions = MLXArray([Int32(0), 0, 3, 1, 7, 5, -1, -1]).reshaped(1, 4, 2)
            let padding = MLXArray([false, false, false, true]).reshaped(1, 4)
            let got = embedder.positionEmbeddings(patchPositions: positions, paddingPositions: padding)
            // Reference : one-hot x matmul, somme des axes x et y, padding a zero.
            let oh = oneHot(positions, numClasses: 64).transposed(0, 2, 1, 3).asType(embedder.positionEmbeddingTable.dtype)
            var expected = matmul(oh, embedder.positionEmbeddingTable).sum(axis: 1)
            expected = MLX.where(expandedDimensions(padding, axis: -1), MLXArray(Float(0)), expected)
            #expect(allClose(got, expected, atol: 1e-6).item(Bool.self))
        }
    }
}

