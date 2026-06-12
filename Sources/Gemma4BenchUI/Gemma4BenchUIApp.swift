// Gemma4BenchUI — Mini GUI pour benchmarker côte à côte :
//   - LLM AR classique (Gemma 4 26B-A4B bf16) — streaming token-par-token
//   - DiffusionGemma 26B-A4B bf16 — denoising step-par-step
//
// Chargement séquentiel (un modèle à la fois) car ~96 Go en RAM sinon.

import AppKit
import SwiftUI

// Un exécutable SwiftPM n'est pas un .app bundle, donc macOS ne lui donne
// pas d'activation policy par défaut → la fenêtre est créée mais reste cachée.
// Force le mode "regular app" + activate au démarrage.
final class BenchAppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
        // Force la première fenêtre au premier plan
        for window in NSApp.windows {
            window.makeKeyAndOrderFront(nil)
        }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }
}

@main
struct Gemma4BenchUIApp: App {
    @NSApplicationDelegateAdaptor(BenchAppDelegate.self) var appDelegate

    var body: some Scene {
        WindowGroup("Gemma 4 — AR vs Diffusion") {
            ContentView()
                .frame(minWidth: 1100, minHeight: 700)
                .preferredColorScheme(.dark)
        }
        .windowStyle(.hiddenTitleBar)
    }
}
