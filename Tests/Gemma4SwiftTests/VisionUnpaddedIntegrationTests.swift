import Testing
import Foundation
import MLX
import MLXLMCommon
@testable import Gemma4Swift

/// K-18 sur vrai modele : ecart des features vision paddees / non paddees, et descriptions
/// greedy des deux chemins sur l'image de reference.
private let modelPath = ProcessInfo.processInfo.environment["GEMMA4_INTEGRATION_MODEL_PATH"]

@Suite("Vision sans padding (integration)", .serialized, .enabled(if: modelPath != nil))
struct VisionUnpaddedIntegrationTests {

    @Test("features et descriptions sur UI.png")
    func testRealImage() async throws {
        let container = try await Gemma4Registration.loadContainer(
            from: URL(fileURLWithPath: modelPath!), multimodal: true)
        let image = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appendingPathComponent("../../docs/examples/vision-image-description/UI.png").standardized
        let report = try await container.perform { context -> String in
            let model = try #require(context.model as? Gemma4MultimodalLLMModel)
            let pixels = try Gemma4ImageProcessor.processImage(url: image)
            model.visionTower.padToMaxPatches = true
            let padded = model.visionTower(pixels)
            model.visionTower.padToMaxPatches = false
            let unpadded = model.visionTower(pixels)
            eval(padded, unpadded)
            let diff = abs(padded.asType(.float32) - unpadded.asType(.float32))
            let rel = (diff.max() / abs(padded.asType(.float32)).max()).item(Float.self)
            let meanRel = (diff.mean() / abs(padded.asType(.float32)).mean()).item(Float.self)

            var texts: [String] = []
            for padding in [true, false] {
                model.visionTower.padToMaxPatches = padding
                let ids = try Gemma4Processor.multimodalChatIds(
                    userPrompt: "Describe this image in detail.", tokenizer: context.tokenizer)
                model.pendingPixelValues = pixels
                var iterator = try TokenIterator(
                    input: LMInput(tokens: MLXArray(ids.map { Int32($0) })), model: context.model,
                    parameters: GenerateParameters(maxTokens: 64, temperature: 0))
                var out: [Int] = []
                while out.count < 64, let t = iterator.next() { out.append(t) }
                texts.append(context.tokenizer.decode(tokenIds: out))
            }
            model.visionTower.padToMaxPatches = false
            let common = zip(texts[0], texts[1]).prefix { $0 == $1 }.count
            return "pixels \(pixels.shape) dtype \(padded.dtype) ; ecart max rel \(rel), moyen rel \(meanRel) ; "
                + "prefixe commun \(common) car.\nPADDED  : \(texts[0])\nUNPADDED: \(texts[1])"
        }
        print("DIAG K-18 " + report)
    }

    @Test("video : temps de l'encodeur par frame, paddé contre non paddé (A/B/B/A)")
    func testVideoEncoderTime() async throws {
        let container = try await Gemma4Registration.loadContainer(
            from: URL(fileURLWithPath: modelPath!), multimodal: true)
        let frames = try await Gemma4VideoProcessor.processVideo(
            url: URL(fileURLWithPath: "/Volumes/Lexar/models/h3-outputs/fox-960x544.mp4"), maxFrames: 8)
        nonisolated(unsafe) let pixels = frames.pixelValues
        eval(pixels)
        let report = try await container.perform { context -> String in
            let model = try #require(context.model as? Gemma4MultimodalLLMModel)
            func run(_ padded: Bool) -> (Double, MLXArray) {
                model.visionTower.padToMaxPatches = padded
                let start = Date()
                var outs: [MLXArray] = []
                for i in 0 ..< pixels.dim(0) {
                    let f = model.visionTower(pixels[i ..< (i + 1)])
                    eval(f)
                    outs.append(f)
                }
                return (Date().timeIntervalSince(start) * 1000 / Double(pixels.dim(0)), concatenated(outs, axis: 0))
            }
            _ = run(true); _ = run(false)  // echauffement
            var times: [Bool: [Double]] = [true: [], false: []]
            var last: [Bool: MLXArray] = [:]
            for padded in [true, false, false, true] {
                let (ms, out) = run(padded)
                times[padded]!.append(ms)
                last[padded] = out
            }
            model.visionTower.padToMaxPatches = false
            let n = frames.softTokensPerFrame
            let a = last[true]![0..., 0 ..< n].asType(.float32)
            let b = last[false]![0..., 0 ..< n].asType(.float32)
            let rel = (abs(a - b).max() / abs(a).max()).item(Float.self)

            // Reference fp32 (poids de la tour convertis, chemin sans padding = calcul exact).
            let saved = model.visionTower.parameters()
            model.visionTower.update(parameters: saved.mapValues { $0.asType(.float32) })
            var refs: [MLXArray] = []
            for i in 0 ..< pixels.dim(0) {
                let f = model.visionTower(pixels[i ..< (i + 1)].asType(.float32))
                eval(f)
                refs.append(f)
            }
            model.visionTower.update(parameters: saved)
            let ref = concatenated(refs, axis: 0)[0..., 0 ..< n]
            func err(_ x: MLXArray) -> Float { (abs(x - ref).max() / abs(ref).max()).item(Float.self) }
            return "frames \(pixels.shape), \(n) jetons/frame ; ms/frame padded \(times[true]!) unpadded \(times[false]!) ; "
                + "ecart max rel padded/unpadded \(rel) ; contre fp32 : padded \(err(a)), unpadded \(err(b))"
        }
        print("DIAG K-18 video " + report)
    }
}

