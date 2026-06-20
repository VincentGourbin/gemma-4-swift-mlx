// Controles UI partages entre les 3 onglets pour gerer un modele du registry.
//   ModelStatusChip       : pastille verte/cyan/grise + label
//   ModelControlButtons   : Charger + Decharger, conditionnes a l'etat reel

import SwiftUI

struct ModelStatusChip: View {
    @EnvironmentObject private var registry: ModelRegistry
    let kind: ModelRegistry.ModelKind

    var body: some View {
        let loaded = registry.isLoaded(kind)
        let busyForThis = registry.busy == kind
        let color: Color = busyForThis ? .cyan : (loaded ? .green : .gray)
        let label: String = {
            if busyForThis {
                return "Chargement \(kind.rawValue)…"
            }
            return loaded ? "\(kind.rawValue) chargé" : "\(kind.rawValue) déchargé"
        }()
        return HStack(spacing: 4) {
            Circle().fill(color).frame(width: 7, height: 7)
            Text(label)
                .font(.system(size: 10, design: .monospaced))
                .foregroundStyle(.white.opacity(0.7))
                .lineLimit(1)
        }
    }
}

struct ModelControlButtons: View {
    @EnvironmentObject private var registry: ModelRegistry
    let kind: ModelRegistry.ModelKind

    var body: some View {
        let loaded = registry.isLoaded(kind)
        let busy = registry.isAnyBusy
        HStack(spacing: 6) {
            Button {
                Task {
                    switch kind {
                    case .e4b: await registry.loadE4B()
                    case .arBf16: await registry.loadAR()
                    case .diffusion: await registry.loadDiffusion()
                    }
                }
            } label: {
                Label("Charger", systemImage: "arrow.down.circle.fill").font(.system(size: 10))
            }
            .buttonStyle(GlowButtonStyle(color: .blue))
            .disabled(loaded || busy)

            Button {
                switch kind {
                case .e4b: registry.unloadE4B()
                case .arBf16: registry.unloadAR()
                case .diffusion: registry.unloadDiffusion()
                }
            } label: {
                Label("Décharger", systemImage: "xmark.circle").font(.system(size: 10))
            }
            .buttonStyle(GlowButtonStyle(color: .red))
            .disabled(!loaded || busy)
        }
    }
}

/// Affichage global (utilise dans BenchTab) montrant l'etat des 3 modeles + RAM totale.
struct ModelRegistrySummary: View {
    @EnvironmentObject private var registry: ModelRegistry

    var body: some View {
        HStack(spacing: 12) {
            ForEach(ModelRegistry.ModelKind.allCases) { k in
                HStack(spacing: 4) {
                    Circle().fill(registry.isLoaded(k) ? .green : .gray.opacity(0.5))
                        .frame(width: 6, height: 6)
                    Text(k.rawValue).font(.system(size: 9, design: .monospaced))
                        .foregroundStyle(.white.opacity(0.6))
                }
            }
            if registry.totalLoadedRAMGo > 0 {
                Text(String(format: "≈ %.0f Go", registry.totalLoadedRAMGo))
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundStyle(.purple.opacity(0.85))
            }
        }
    }
}
