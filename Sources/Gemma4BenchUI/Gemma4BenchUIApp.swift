// Gemma4BenchUI — Mini GUI pour benchmarker côte à côte :
//   - LLM AR classique (Gemma 4 26B-A4B bf16) — streaming token-par-token
//   - DiffusionGemma 26B-A4B bf16 — denoising step-par-step
//
// Chargement séquentiel (un modèle à la fois) car ~96 Go en RAM sinon.

import SwiftUI

@main
struct Gemma4BenchUIApp: App {
    var body: some Scene {
        WindowGroup("Gemma 4 — AR vs Diffusion") {
            ContentView()
                .frame(minWidth: 1100, minHeight: 700)
                .preferredColorScheme(.dark)
        }
        .windowStyle(.hiddenTitleBar)
    }
}
