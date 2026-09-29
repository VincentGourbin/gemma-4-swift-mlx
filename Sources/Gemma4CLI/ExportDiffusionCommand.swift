// export-diffusion : ecrit un pack pre-quantifie DiffusionGemma (K-D12) depuis le bf16
// officiel et un profil de reference. Le pack se charge ensuite directement quantifie
// (`--model-path <pack>` partout ou un checkpoint est attendu), sans le pic bf16.

import ArgumentParser
import Foundation
import Gemma4Swift
import MLX

struct ExportDiffusion: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "export-diffusion",
        abstract: "Exporte un pack pre-quantifie DiffusionGemma pour un profil a4bdiff/*"
    )

    @Option(name: .long, help: "Dossier du checkpoint bf16 officiel (google/diffusiongemma-26B-A4B-it)")
    var modelPath: String

    @Option(name: .long, help: "Profil de reference (a4bdiff/8bit-fast, a4bdiff/4bit-fast, …)")
    var reference: String

    @Option(name: .long, help: "Dossier de destination du pack")
    var out: String

    @Flag(name: .long, help: "Relire le pack et verifier les SHA-256 apres ecriture")
    var verify = false

    func run() async throws {
        guard let profile = DiffusionReferenceProfile.named(reference) else {
            throw ValidationError("profil inconnu : \(reference)")
        }
        guard profile.quantization != .none else {
            throw ValidationError("\(profile.qualifiedID) n'est pas quantifie : rien a exporter")
        }
        let source = URL(fileURLWithPath: modelPath)
        guard !DiffusionPrequantizedPack.isPack(source) else {
            throw ValidationError("--model-path doit etre le checkpoint bf16, pas un pack")
        }
        let destination = URL(fileURLWithPath: out)

        let start = Date()
        let container = try await DiffusionGemmaRegistration.load(from: source, profile: profile)
        print("quantifie en \(Int(Date().timeIntervalSince(start))) s (\(profile.quantization.signature))")

        let writeStart = Date()
        let manifest = try DiffusionPrequantizedPack.export(
            model: container.model, quantization: profile.quantization.signature,
            sourceDirectory: source, to: destination)
        let bytes = manifest.files.keys.reduce(0) { total, name in
            let size = (try? FileManager.default.attributesOfItem(
                atPath: destination.appendingPathComponent(name).path)[.size] as? Int) ?? 0
            return total + (size ?? 0)
        }
        print("pack ecrit en \(Int(Date().timeIntervalSince(writeStart))) s : \(manifest.files.count) fichier(s), "
            + String(format: "%.1f Go", Double(bytes) / 1e9) + ", \(manifest.modules.count) modules quantifies")
        print("-> \(destination.path)")

        if verify {
            _ = try DiffusionPrequantizedPack.load(from: destination, includeVision: false, verifyChecksums: true)
            print("verification SHA-256 : OK")
        }
    }
}
