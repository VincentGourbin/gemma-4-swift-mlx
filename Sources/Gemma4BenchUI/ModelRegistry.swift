// Registre global partage entre les 3 onglets (Bench, Agent, Akinator VQA).
// Maintient l'etat des 3 modeles susceptibles d'etre charges :
//   - DiffusionGemma 26B-A4B bf16   (~50 Go) — utilise par Bench/Agent/Akinator
//   - Gemma 4 26B-A4B AR bf16/4-bit (~50/14 Go) — utilise par Bench (panel AR)
//   - Gemma 4 E4B 4-bit             (~5 Go)  — drafter optionnel, utilise par Bench
//
// Chaque onglet lit `isXLoaded` (Published) pour conditionner ses boutons,
// et appelle `loadX()` / `unloadX()` au lieu d'avoir ses propres pipelines.
// Un seul "load Diffusion" pour les 3 onglets : on ne paie qu'une fois les 50 Go.

import Foundation
import Gemma4Swift
import MLX
import SwiftUI
import Tokenizers

@MainActor
final class ModelRegistry: ObservableObject {

    enum ModelKind: String, CaseIterable, Identifiable {
        case e4b = "E4B 4-bit"
        case arBf16 = "AR 26B-A4B bf16"
        case diffusion = "DiffusionGemma 26B-A4B bf16"
        var id: String { rawValue }
    }

    // MARK: - Etat de chargement
    /// Identifie le modele en train d'etre charge/decharge (nil = idle).
    @Published private(set) var busy: ModelKind? = nil
    @Published private(set) var busyDetail: String = ""
    @Published var lastError: String? = nil

    // MARK: - Storage modeles
    // E4B (multimodal, 4-bit)
    @Published private(set) var e4bPipeline: Gemma4Pipeline?
    // AR 26B-A4B (multimodal, bf16)
    @Published private(set) var arPipeline: Gemma4Pipeline?
    // Diffusion (4 instances liees)
    @Published private(set) var diffModel: DiffusionGemmaForBlockDiffusion?
    @Published private(set) var diffConfig: DiffusionGemmaConfig?
    @Published private(set) var diffGenConfig: DiffusionGenerationConfig?
    @Published private(set) var diffTokenizer: Tokenizer?

    // MARK: - Computed flags
    var isE4BLoaded: Bool { e4bPipeline != nil }
    var isARLoaded: Bool { arPipeline != nil }
    var isDiffusionLoaded: Bool { diffModel != nil }

    func isLoaded(_ kind: ModelKind) -> Bool {
        switch kind {
        case .e4b: return isE4BLoaded
        case .arBf16: return isARLoaded
        case .diffusion: return isDiffusionLoaded
        }
    }

    /// True quand on est en train de charger/decharger N'IMPORTE QUEL modele.
    var isAnyBusy: Bool { busy != nil }

    // MARK: - Paths
    let e4bModelID = "mlx-community/gemma-4-e4b-it-4bit"
    let arModelID = "mlx-community/gemma-4-26b-a4b-it-bf16"
    let diffModelID = "google/diffusiongemma-26B-A4B-it"

    private func path(for repo: String) -> URL {
        var p = Gemma4ModelCache.modelsDirectory
        for part in repo.split(separator: "/") {
            p = p.appendingPathComponent(String(part))
        }
        return p
    }

    // MARK: - Chargement E4B
    @discardableResult
    func loadE4B() async -> Bool {
        guard !isE4BLoaded else { return true }
        guard busy == nil else { lastError = "Un autre chargement est en cours"; return false }
        busy = .e4b
        busyDetail = "\(e4bModelID)…"
        lastError = nil
        defer { busy = nil; busyDetail = "" }
        do {
            await Gemma4Registration.register(multimodal: true)
            let pipe = Gemma4Pipeline()
            try await pipe.load(from: path(for: e4bModelID), multimodal: true)
            e4bPipeline = pipe
            return true
        } catch {
            lastError = "E4B : \(error.localizedDescription)"
            return false
        }
    }

    func unloadE4B() {
        guard busy == nil else { return }
        e4bPipeline?.unload()
        e4bPipeline = nil
        MLX.GPU.clearCache()
    }

    // MARK: - Chargement AR 26B-A4B bf16
    @discardableResult
    func loadAR() async -> Bool {
        guard !isARLoaded else { return true }
        guard busy == nil else { lastError = "Un autre chargement est en cours"; return false }
        busy = .arBf16
        busyDetail = "\(arModelID) (~50 Go)…"
        lastError = nil
        defer { busy = nil; busyDetail = "" }
        do {
            await Gemma4Registration.register(multimodal: true)
            let pipe = Gemma4Pipeline()
            try await pipe.load(from: path(for: arModelID), multimodal: true)
            arPipeline = pipe
            return true
        } catch {
            lastError = "AR : \(error.localizedDescription)"
            return false
        }
    }

    func unloadAR() {
        guard busy == nil else { return }
        arPipeline?.unload()
        arPipeline = nil
        MLX.GPU.clearCache()
    }

    // MARK: - Chargement DiffusionGemma
    /// Charge DiffusionGemma. Si `mixedPrecision` est fourni, applique la
    /// quantization mixed-precision Q-DiT/ViDiT-Q apres le load.
    @discardableResult
    func loadDiffusion(mixedPrecision: DiffusionOnTheFlyQuantization.MixedPrecisionConfig? = nil) async -> Bool {
        guard !isDiffusionLoaded else { return true }
        guard busy == nil else { lastError = "Un autre chargement est en cours"; return false }
        busy = .diffusion
        busyDetail = "\(diffModelID) (~50 Go)…"
        lastError = nil
        defer { busy = nil; busyDetail = "" }
        do {
            let (model, config) = try DiffusionGemmaLoader.load(
                from: path(for: diffModelID), includeVision: true
            )
            if let mpConfig = mixedPrecision {
                busyDetail = "Quantization mixed precision…"
                let stats = DiffusionOnTheFlyQuantization.applyMixedPrecision(to: model, config: mpConfig)
                print("[Registry] Diffusion quant: \(stats.quantizedHigh) high-bit, \(stats.quantizedLow) low-bit")
            }
            diffModel = model
            diffConfig = config

            let url = path(for: diffModelID).appendingPathComponent("generation_config.json")
            if FileManager.default.fileExists(atPath: url.path),
               let data = try? Data(contentsOf: url),
               let parsed = try? JSONDecoder().decode(DiffusionGenerationConfig.self, from: data)
            {
                diffGenConfig = parsed
            } else {
                diffGenConfig = DiffusionGenerationConfig()
            }
            diffTokenizer = try await AutoTokenizer.from(modelFolder: path(for: diffModelID))
            return true
        } catch {
            lastError = "Diffusion : \(error.localizedDescription)"
            return false
        }
    }

    func unloadDiffusion() {
        guard busy == nil else { return }
        diffModel = nil
        diffConfig = nil
        diffGenConfig = nil
        diffTokenizer = nil
        MLX.GPU.clearCache()
    }

    /// Decharge TOUT (utile avant de basculer entre modeles lourds).
    func unloadAll() {
        guard busy == nil else { return }
        e4bPipeline?.unload(); e4bPipeline = nil
        arPipeline?.unload(); arPipeline = nil
        diffModel = nil
        diffConfig = nil
        diffGenConfig = nil
        diffTokenizer = nil
        MLX.GPU.clearCache()
    }

    // MARK: - Estimation RAM
    var totalLoadedRAMGo: Double {
        var ram: Double = 0
        if isE4BLoaded { ram += 5 }
        if isARLoaded { ram += 50 }
        if isDiffusionLoaded { ram += 50 }
        return ram
    }
}
