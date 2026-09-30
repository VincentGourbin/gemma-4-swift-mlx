import Testing
import Foundation
import MLX
import MLXNN
@testable import Gemma4Swift

/// Diagnostic K-D11 : la campagne du 2026-09-28 mesure un pic de 76 Go en 8 bits et
/// 64 Go en 4 bits, contre 50,5 Go en bf16. Ce test charge le vrai checkpoint,
/// quantifie comme le profil, puis compare la memoire MLX active a la taille des
/// parametres atteignables (tableaux uniques). Active par :
///   GEMMA4_DIFFUSION_MODEL_PATH=/Volumes/Lexar/models/google/diffusiongemma-26B-A4B-it \
///     Scripts/run-tests.sh -only-testing:Gemma4SwiftTests/DiffusionQuantizationMemoryDiagnosticTests
private let diffusionModelPath = ProcessInfo.processInfo.environment["GEMMA4_DIFFUSION_MODEL_PATH"]

@Suite("Diagnostic memoire de la quantification diffusion", .serialized,
       .enabled(if: diffusionModelPath != nil))
struct DiffusionQuantizationMemoryDiagnosticTests {

    private func mb(_ bytes: Int) -> Int { bytes / 1_048_576 }

    private func report(_ label: String, _ model: DiffusionGemmaForBlockDiffusion) {
        eval(model)
        let flat = model.parameters().flattened()
        var unique: [ObjectIdentifier: (String, MLXArray)] = [:]
        for (key, array) in flat { unique[ObjectIdentifier(array)] = unique[ObjectIdentifier(array)] ?? (key, array) }
        var byDtype: [String: Int] = [:]
        var byBranch: [String: Int] = [:]
        for (_, (key, array)) in unique {
            byDtype["\(array.dtype)", default: 0] += array.nbytes
            let branch = key.hasPrefix("decoder.") ? "decoder"
                : key.contains("vision") ? "vision"
                : key.hasPrefix("encoder.") ? "encoder" : "autre"
            byBranch[branch, default: 0] += array.nbytes
        }
        let encoder = Dictionary(model.encoder.languageModel.parameters().flattened(), uniquingKeysWith: { a, _ in a })
        let shared = model.decoder.parameters().flattened().filter { key, array in
            encoder[key].map { ObjectIdentifier($0) == ObjectIdentifier(array) } ?? false
        }.count
        let uniqueBytes = unique.values.reduce(0) { $0 + $1.1.nbytes }
        print("""
        DIAG \(label): actif MLX \(mb(Memory.activeMemory)) Mo, cache \(mb(Memory.cacheMemory)) Mo ; \
        parametres \(flat.count) (\(unique.count) uniques) = \(mb(uniqueBytes)) Mo ; \
        par dtype \(byDtype.mapValues(mb)) ; par branche \(byBranch.mapValues(mb)) ; \
        cles decodeur partagees avec l'encodeur : \(shared)
        """)
    }

    @Test("8 bits : memoire active contre parametres atteignables")
    func testEightBits() throws {
        let url = URL(fileURLWithPath: diffusionModelPath!)
        let (model, _) = try DiffusionGemmaLoader.load(from: url, includeVision: true)
        Memory.clearCache()
        report("bf16 charge", model)
        DiffusionOnTheFlyQuantization.apply(
            to: model, bits: 8, groupSize: 64,
            excludedPathPrefixes: DiffusionOnTheFlyQuantization.multimodalEncoderPrefixes)
        Memory.clearCache()
        report("apres quantification 8 bits", model)
    }

    @Test("8 bits sans reliaison encodeur/decodeur : ou vont les bf16 ?")
    func testEightBitsWithoutRetie() throws {
        let url = URL(fileURLWithPath: diffusionModelPath!)
        let (model, _) = try DiffusionGemmaLoader.load(from: url, includeVision: true)
        Memory.clearCache()
        report("bf16 charge (sans reliaison)", model)
        MLXNN.quantize(model: model, filter: { path, m in
            if path.contains("vision") { return nil }
            guard m is Quantizable, !(m is Quantized) else { return nil }
            if let lin = m as? Linear, (lin.weight.shape.last ?? 0) % 64 != 0 { return nil }
            if let emb = m as? Embedding, (emb.weight.shape.last ?? 0) % 64 != 0 { return nil }
            return (groupSize: 64, bits: 8, mode: .affine)
        })
        eval(model)
        Memory.clearCache()
        report("8 bits sans reliaison", model)
    }

    @Test("8 bits, encodeur seul quantifie puis modules partages avec le decodeur")
    func testEightBitsSharedModules() throws {
        let url = URL(fileURLWithPath: diffusionModelPath!)
        let (model, _) = try DiffusionGemmaLoader.load(from: url, includeVision: true)
        Memory.clearCache()
        report("bf16 charge (modules partages)", model)
        let encoder = model.encoder.languageModel
        MLXNN.quantize(model: encoder, filter: { path, m in
            guard m is Quantizable, !(m is Quantized) else { return nil }
            if let lin = m as? Linear, (lin.weight.shape.last ?? 0) % 64 != 0 { return nil }
            if let emb = m as? Embedding, (emb.weight.shape.last ?? 0) % 64 != 0 { return nil }
            return (groupSize: 64, bits: 8, mode: .affine)
        })
        eval(encoder)
        let encoderLeaves = Dictionary(encoder.leafModules().flattened(), uniquingKeysWith: { a, _ in a })
        var replacements: [(String, Module)] = []
        for (path, _) in model.decoder.leafModules().flattened() where !path.hasPrefix("self_conditioning") {
            if let source = encoderLeaves[path], source is Quantized { replacements.append((path, source)) }
        }
        print("DIAG modules du decodeur remplaces : \(replacements.count)")
        model.decoder.update(modules: ModuleChildren.unflattened(replacements))
        eval(model)
        Memory.clearCache()
        report("8 bits modules partages", model)
    }
}
