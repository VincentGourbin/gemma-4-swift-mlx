// gemma4-cli references — liste des profils de reference (K-20)

import ArgumentParser
import Foundation
import Gemma4Swift

struct References: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Liste les profils de reference <bits>bit-<fast|lean> et leurs reglages"
    )

    @Option(name: .long, help: "Famille : e2b, e4b, b12b, a4b, b31b")
    var family: String?

    func run() throws {
        let families: [Gemma4Pipeline.Model.Family]
        if let family {
            guard let parsed = Gemma4Pipeline.Model.Family(rawValue: family) else {
                throw ValidationError("famille inconnue : \(family)")
            }
            families = [parsed]
        } else {
            families = Gemma4Pipeline.Model.Family.allCases.filter { $0 != .a4bDiff }
        }
        let available = Gemma4ReferenceProfile.availableMemoryMB()
        print("Memoire disponible estimee : \(available) Mo (GEMMA4_AVAILABLE_MB pour simuler une autre machine)\n")
        for family in families {
            let recommended = Gemma4ReferenceProfile.recommended(for: family, availableMB: available)
            for profile in Gemma4ReferenceProfile.profiles(for: family) {
                let mark = profile == recommended ? " ← conseille ici" : ""
                let kv = profile.kvBits.map { "\($0) bits" } ?? "bf16"
                let cache = profile.cacheLimitMB.map { "\($0) Mo" } ?? "MLX"
                let limit = profile.memoryLimitMB.map { "\($0) Mo" } ?? "-"
                print("\(profile.qualifiedID)\(mark)")
                print("  poids   : \(profile.model.rawValue) (~\(Int(profile.model.estimatedSizeGB)) Go)")
                print("  reglages: KV \(kv), prefill \(profile.prefillStepSize), cache \(cache), seuil \(limit), "
                    + "vidage apres reponse \(profile.clearCacheAfterAnswer ? "oui" : "non")")
                print("  \(profile.summary)")
            }
            print()
        }
        print("Utiliser : gemma4-cli bench --model-path <poids> --reference <famille/id>")
    }
}
