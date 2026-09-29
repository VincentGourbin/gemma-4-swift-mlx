// gemma4-cli bench — instrument de mesure du chemin de generation de la bibliotheque (K-11, P-07)

import ArgumentParser
import CoreGraphics
import CryptoKit
import Darwin
import Foundation
import Gemma4Swift
import MLX
import MLXLMCommon

/// Mesure le chemin que les consommateurs executent : chargement par
/// `Gemma4Registration.loadContainer`, puis `TokenIterator` de mlx-swift-lm (celui
/// qu'utilise `Gemma4Pipeline`, avec son pipelining `asyncEval`). L'ancien
/// `profile` decodait dans une boucle maison ou `asyncEval` etait du code mort : ses
/// chiffres ne mesuraient pas la bibliotheque.
///
/// Une ligne JSON par point (stdout, et `--out` en ajout). Protocole : binaire
/// Release, machine au repos, `--cooldown`, A/B/B/A pour comparer deux variantes,
/// un ecart < 5 % est du bruit (voir le skill mlx-swift-audit, references/measurement.md).
struct Bench: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Mesure prefill, TTFT, decodage par jeton et memoire (une ligne JSON par point)"
    )

    @Option(name: .long, help: "Chemin local du modele")
    var modelPath: String

    @Option(name: .long, help: "Tailles de prompt en jetons, separees par des virgules")
    var promptTokens: String = "128,1024,4096"

    @Option(name: .long, help: "Jetons generes par point (nombre fixe, EOS ignore)")
    var maxTokens: Int = 128

    @Option(name: .long, help: "Fichier texte source des prompts (defaut : texte integre)")
    var promptFile: String?

    @Option(name: .long, help: "Image : mesure le chemin multimodal (prompt court fixe)")
    var image: String?

    @Option(name: .long, help: "Tranche de prefill (GenerateParameters.prefillStepSize)")
    var prefillStep: Int?

    @Option(name: .long, help: "Profil de reference a appliquer (ex. e2b/4bit-lean ; voir `references`)")
    var reference: String?

    @Option(name: .long, help: "Secondes de repos avant chaque point")
    var cooldown: Int = 0

    @Option(name: .long, help: "Nombre de passes par point")
    var repeats: Int = 1

    @Option(name: .long, help: "Fichier JSONL ou ajouter les lignes")
    var out: String?

    @Option(name: .long, help: "Etiquette libre (variante A/B, commentaire)")
    var label: String = ""

    @Option(name: .long, help: "Temperature d'echantillonnage (defaut 0 : greedy)")
    var temperature: Float = 0

    @Option(name: .long, help: "top-p (avec --temperature > 0)")
    var topP: Float = 1

    @Option(name: .long, help: "top-k (avec --temperature > 0 ; 0 = desactive)")
    var topK: Int = 0

    @Option(name: .long, help: "Blocage des n-grammes repetes (NoRepeatNGramLogitProcessor), taille n")
    var noRepeatNgram: Int?

    @Option(name: .long, help: "Chemin du n-gramme : device (historique GPU, defaut) ou host (historique CPU, sync par jeton)")
    var ngramPath: String = "device"

    @Flag(name: .long, help: "Diagnostic K-18 : encodeur vision paddé a maxPatches (ancien chemin)")
    var visionPadded = false

    @Flag(name: .long, help: "Avec --image : charger le modele sans tour audio (K-16)")
    var noAudio = false

    @Flag(name: .long, help: "Ne pas faire la passe d'echauffement (compilation Metal) non chronometree")
    var noWarmup = false

    func run() async throws {
        let sizes = promptTokens.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        guard !sizes.isEmpty || image != nil else {
            throw ValidationError("--prompt-tokens vide")
        }
        let modelURL = URL(fileURLWithPath: modelPath)
        let multimodal = image != nil

        let loadStart = Date()
        let container = try await Gemma4Registration.loadContainer(
            from: modelURL, using: LocalTokenizerLoader(), multimodal: multimodal, audio: !noAudio)
        await container.perform { context in eval(context.model) }
        let loadSeconds = Date().timeIntervalSince(loadStart)
        let loadActiveMB = Memory.activeMemory / 1_048_576

        let profile = try reference.map { id in
            guard let profile = Gemma4ReferenceProfile.named(id) else {
                throw ValidationError("profil inconnu : \(id) (voir `gemma4-cli references`)")
            }
            return profile
        }
        profile?.applyGlobalPolicy()

        let pixels: MLXArray? = try image.map { path in
            let pixels = try Gemma4ImageProcessor.processImage(url: URL(fileURLWithPath: path))
            eval(pixels)
            return pixels
        }
        let filler = try promptFile.map { try String(contentsOfFile: $0, encoding: .utf8) } ?? Self.defaultFiller
        var context = BenchContext.collect(modelURL: modelURL, loadSeconds: loadSeconds)
        context.fields["load_active_mlx_mb"] = loadActiveMB
        if multimodal { context.fields["audio_loaded"] = !noAudio }
        if let profile {
            context.fields["profile"] = profile.qualifiedID
            // Les poids mesures doivent etre ceux du profil, sinon la ligne ne le
            // represente pas (ex. un 6 bits mesure sous un profil 8 bits).
            let expected = profile.model.rawValue.split(separator: "/").last.map(String.init) ?? ""
            let matches = modelURL.lastPathComponent == expected
            context.fields["profile_weights_match"] = matches
            if !matches {
                FileHandle.standardError.write(Data(
                    "⚠ poids \(modelURL.lastPathComponent) ≠ poids du profil \(expected) : ligne marquee profile_weights_match=false\n".utf8))
            }
        }

        if !noWarmup {
            _ = try await measure(
                container: container, promptSize: 32, pixels: pixels, filler: filler, maxTokens: 4, profile: profile)
        }

        let points: [Int] = multimodal ? [0] : sizes
        for size in points {
            for pass in 1 ... max(1, repeats) {
                if cooldown > 0 { try await Task.sleep(for: .seconds(cooldown)) }
                var line = try await measure(
                    container: container, promptSize: size, pixels: pixels, filler: filler,
                    maxTokens: maxTokens, profile: profile)
                line.merge(context.fields) { current, _ in current }
                line["pass"] = pass
                line["label"] = label
                line["cooldown_s"] = cooldown
                try emit(line)
            }
        }
    }

    /// Un point de mesure. `promptSize` = nombre exact de jetons du prompt (mode texte,
    /// sans gabarit de chat) ; ignore en mode image (prompt court fixe + 280 jetons image).
    private func measure(
        container: ModelContainer, promptSize: Int, pixels: MLXArray?, filler: String, maxTokens: Int,
        profile: Gemma4ReferenceProfile?
    ) async throws -> [String: Any] {
        nonisolated(unsafe) let pixelsCapture = pixels
        let modelPath = self.modelPath
        let prefillStep = self.prefillStep
        let ngram = self.noRepeatNgram
        let visionPadded = self.visionPadded
        let temperature = self.temperature
        let topP = self.topP
        let topK = self.topK
        let ngramOnHost = self.ngramPath == "host"
        let line = try await container.perform { context -> BenchLine in
            let ids: [Int]
            if let pixelsCapture {
                ids = try Gemma4Processor.multimodalChatIds(
                    userPrompt: "Describe this image in detail.", tokenizer: context.tokenizer)
                guard let model = context.model as? Gemma4MultimodalLLMModel else {
                    throw ValidationError("--image exige un modele multimodal E2B/E4B")
                }
                model.pendingPixelValues = pixelsCapture
                model.visionTower.padToMaxPatches = visionPadded
            } else {
                ids = try Self.promptIds(size: promptSize, filler: filler, tokenizer: context.tokenizer)
            }

            var parameters = GenerateParameters(maxTokens: maxTokens, temperature: temperature, topP: topP, topK: topK)
            profile?.apply(to: &parameters)
            // Une option explicite l'emporte sur le profil (balayage d'une variable).
            if let prefillStep { parameters.prefillStepSize = prefillStep }

            Memory.clearCache()
            Memory.peakMemory = 0
            let start = Date()
            let input = LMInput(tokens: MLXArray(ids.map { Int32($0) }))
            var iterator: TokenIterator
            if let ngram {
                // Chemin host = ancien comportement (synchronisation par jeton) : A/B de K-15.
                iterator = try TokenIterator(
                    input: input, model: context.model, cache: nil,
                    processor: NoRepeatNGramLogitProcessor(ngramSize: ngram, includeThinkingInWindow: !ngramOnHost),
                    sampler: parameters.sampler(), prefillStepSize: parameters.prefillStepSize,
                    maxTokens: maxTokens)
            } else {
                iterator = try TokenIterator(input: input, model: context.model, parameters: parameters)
            }
            let prefillSeconds = iterator.promptPrefillTime

            var stamps: [TimeInterval] = []
            var generated: [Int] = []
            stamps.reserveCapacity(maxTokens)
            while stamps.count < maxTokens, let token = iterator.next() {
                stamps.append(Date().timeIntervalSince(start))
                generated.append(token)
            }
            let snapshot = Memory.snapshot()

            let intervals = zip(stamps.dropFirst(), stamps).map { ($0 - $1) * 1000 }.sorted()
            let decodeSeconds = (stamps.last ?? 0) - (stamps.first ?? 0)
            let decodeTokS = intervals.isEmpty ? 0 : Double(intervals.count) / decodeSeconds
            let footprint = BenchContext.physFootprint()

            var line: [String: Any] = [
                "mode": pixelsCapture == nil ? "text" : "image",
                "prompt_tokens": ids.count,
                "generated_tokens": stamps.count,
                "prefill_ms": Self.round(prefillSeconds * 1000),
                "prefill_tok_s": Self.round(Double(ids.count) / max(prefillSeconds, 1e-9)),
                "ttft_ms": Self.round((stamps.first ?? 0) * 1000),
                // Moyenne sur la duree totale (sensible aux pas lents isoles) et debit
                // median (robuste) : la validation A/A et les comparaisons A/B utilisent le second.
                "decode_tok_s": Self.round(decodeTokS),
                "decode_tok_s_median": Self.round(
                    intervals.isEmpty ? 0 : 1000 / max(Self.percentile(intervals, 0.5), 1e-9)),
                "step_ms_median": Self.round(Self.percentile(intervals, 0.5)),
                "step_ms_p90": Self.round(Self.percentile(intervals, 0.9)),
                "peak_mlx_mb": snapshot.peakMemory / 1_048_576,
                "active_mlx_mb": snapshot.activeMemory / 1_048_576,
                "cache_mlx_mb": snapshot.cacheMemory / 1_048_576,
                "phys_footprint_mb": footprint.current,
                "phys_footprint_peak_mb": footprint.peak,
                "prefill_step": parameters.prefillStepSize,
                "kv_bits": parameters.kvBits ?? 16,
                "temperature": Double(temperature), "top_p": Double(topP), "top_k": topK,
                // Empreinte des jetons generes : parite de sortie entre variantes A/B.
                "output_sha": Self.digest(generated),
            ]
            if pixelsCapture != nil { line["vision_padded"] = visionPadded }
            if let ngram {
                line["ngram"] = ngram
                line["ngram_path"] = ngramOnHost ? "host" : "device"
            }
            // Taille des poids x tok/s : indicateur, pas une mesure. Surestime sur E2B/E4B,
            // dont les tables d'embeddings par couche ne sont lues que sur quelques lignes
            // par jeton (on observe alors plus que le plafond de la puce).
            if let weights = BenchContext.weightsGB(modelURL: URL(fileURLWithPath: modelPath)) {
                line["weights_bw_gbps"] = Self.round(weights * decodeTokS)
            }
            return BenchLine(fields: line)
        }
        return line.fields
    }

    /// SHA-256 court (12 hex) d'une suite de jetons.
    static func digest(_ tokens: [Int]) -> String {
        let bytes = tokens.flatMap { withUnsafeBytes(of: Int32($0).littleEndian, Array.init) }
        return SHA256.hash(data: bytes).prefix(6).map { String(format: "%02x", $0) }.joined()
    }

    private func emit(_ line: [String: Any]) throws {
        try Self.writeLine(line, to: out)
    }

    /// Une ligne JSON sur stdout, et en ajout dans `out` si fourni (partage avec bench-diffusion).
    static func writeLine(_ line: [String: Any], to out: String?) throws {
        let data = try JSONSerialization.data(withJSONObject: line, options: [.sortedKeys, .withoutEscapingSlashes])
        let text = String(decoding: data, as: UTF8.self)
        print(text)
        if let out {
            let url = URL(fileURLWithPath: out)
            if !FileManager.default.fileExists(atPath: out) {
                FileManager.default.createFile(atPath: out, contents: nil)
            }
            let handle = try FileHandle(forWritingTo: url)
            defer { try? handle.close() }
            try handle.seekToEnd()
            try handle.write(contentsOf: Data((text + "\n").utf8))
        }
    }

    /// Exactement `size` jetons : BOS puis le texte source repete, tronque.
    static func promptIds(size: Int, filler: String, tokenizer: any Tokenizer) throws -> [Int] {
        var ids = tokenizer.encode(text: filler, addSpecialTokens: true)
        guard ids.count > 1 else { throw ValidationError("texte source vide") }
        let body = Array(ids.dropFirst())
        while ids.count < size { ids += body }
        return Array(ids.prefix(size))
    }

    static func percentile(_ sorted: [Double], _ q: Double) -> Double {
        guard !sorted.isEmpty else { return 0 }
        return sorted[min(sorted.count - 1, Int((Double(sorted.count - 1) * q).rounded()))]
    }

    /// Deux decimales, ecrites telles quelles dans le JSON (un Double y garderait
    /// ses queues d'arrondi binaire : 92.200000000000003).
    static func round(_ value: Double) -> Decimal {
        Decimal(string: String(format: "%.2f", value)) ?? 0
    }

    static let defaultFiller = """
    The history of the bicycle spans more than two centuries. Karl Drais built his running machine in \
    1817, a wooden frame with two wheels that the rider pushed along with the feet. Pedals attached to \
    the front wheel appeared in the 1860s, and the high-wheeled penny-farthing followed, fast but \
    dangerous. The safety bicycle of the 1880s, with a chain drive and two wheels of equal size, gave \
    the machine its modern shape, and the pneumatic tyre made it comfortable. Bicycles changed how \
    people worked, courted and travelled, and they remain one of the most efficient ways to move a \
    human being over land. Today frames are made of steel, aluminium, titanium or carbon fibre, and \
    electric assistance has opened cycling to many more riders.
    """
}

/// Ligne de mesure rendue par `container.perform` (valeurs JSON : nombres et chaines).
struct BenchLine: @unchecked Sendable {
    let fields: [String: Any]
}

/// Contexte commun a toutes les lignes d'un run : machine, commit, dependances.
struct BenchContext {
    var fields: [String: Any]

    static func collect(modelURL: URL, loadSeconds: Double) -> BenchContext {
        var fields: [String: Any] = [
            "date": ISO8601DateFormatter().string(from: Date()),
            "model": modelURL.lastPathComponent,
            "load_s": Bench.round(loadSeconds),
            "host": sysctlString("hw.model") ?? "unknown",
            "ram_gb": Int(ProcessInfo.processInfo.physicalMemory / 1_073_741_824),
            "os": ProcessInfo.processInfo.operatingSystemVersionString,
        ]
        if let weights = weightsGB(modelURL: modelURL) { fields["weights_gb"] = Bench.round(weights) }
        // Revision du depot qui contient le binaire (et non du repertoire courant :
        // comparer `main` a une branche melangeait les deux).
        let binaryDir = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath()
            .deletingLastPathComponent().path
        if let sha = shell("git", "-C", binaryDir, "rev-parse", "--short", "HEAD") {
            let dirty = shell("git", "-C", binaryDir, "status", "--porcelain", "--untracked-files=no")
                .map { !$0.isEmpty } ?? false
            fields["commit"] = dirty ? sha + "+modifs" : sha
        }
        for (key, value) in dependencyRevisions() { fields[key] = value }
        #if DEBUG
        fields["build"] = "debug"
        #else
        fields["build"] = "release"
        #endif
        return BenchContext(fields: fields)
    }

    /// Taille des poids sur disque (liens suivis), en Go.
    static func weightsGB(modelURL: URL) -> Double? {
        let fm = FileManager.default
        guard let names = try? fm.contentsOfDirectory(atPath: modelURL.path) else { return nil }
        let bytes = names.filter { $0.hasSuffix(".safetensors") }.reduce(Int64(0)) { total, name in
            let path = modelURL.appendingPathComponent(name).resolvingSymlinksInPath().path
            return total + ((try? fm.attributesOfItem(atPath: path)[.size] as? Int64) ?? 0)
        }
        return bytes > 0 ? Double(bytes) / 1_073_741_824 : nil
    }

    /// Empreinte physique du process (ce que juge jetsam et le Moniteur d'activite),
    /// courante et pic depuis le lancement. `ps`/RSS sous-compte la memoire Metal.
    static func physFootprint() -> (current: Int, peak: Int) {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return (0, 0) }
        return (Int(info.phys_footprint) / 1_048_576, Int(info.ledger_phys_footprint_peak) / 1_048_576)
    }

    /// Versions resolues de mlx-swift et mlx-swift-lm, lues dans Package.resolved (repertoire courant).
    static func dependencyRevisions() -> [String: String] {
        guard let data = FileManager.default.contents(atPath: "Package.resolved"),
              let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let pins = root["pins"] as? [[String: Any]] else { return [:] }
        var result: [String: String] = [:]
        for pin in pins {
            guard let identity = pin["identity"] as? String, identity.hasPrefix("mlx-swift"),
                  let state = pin["state"] as? [String: Any] else { continue }
            let version = state["version"] as? String
            let revision = (state["revision"] as? String).map { String($0.prefix(8)) }
            result["dep_" + identity.replacingOccurrences(of: "-", with: "_")] = version ?? revision ?? "?"
        }
        return result
    }

    static func sysctlString(_ name: String) -> String? {
        var size = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return nil }
        var buffer = [CChar](repeating: 0, count: size)
        guard sysctlbyname(name, &buffer, &size, nil, 0) == 0 else { return nil }
        return String(decoding: buffer.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }

    static func shell(_ args: String...) -> String? {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = args
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = FileHandle.nullDevice
        guard (try? process.run()) != nil else { return nil }
        process.waitUntilExit()
        guard process.terminationStatus == 0 else { return nil }
        return String(decoding: pipe.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
