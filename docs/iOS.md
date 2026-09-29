# Gemma 4 sur iPhone / iPad

Profil de référence **`e2b/4bit-tiny`** : Gemma 4 E2B 4 bits sous **4 Go d'empreinte**, en texte comme
en image (lot I de l'audit du 2026-09-27, sur le modèle de YuE2 `4bit-lean`).

## Prérequis

- iOS 17+, appareil Apple Silicon (A17 Pro / M1 ou plus récent recommandé).
- Capability **Increased Memory Limit** (`com.apple.developer.kernel.increased-memory-limit`) : sans
  elle, la limite jetsam d'une app tombe sous les 3,9 Go du cas image.
- Le pack `mlx-community/gemma-4-e2b-it-4bit` (≈ 3,6 Go) dans un dossier de l'app, par exemple
  `Caches/models/mlx-community/gemma-4-e2b-it-4bit/` avec `Gemma4ModelCache.customModelsDirectory`.
- La bibliothèque compile pour iOS (vérifié en CI, `generic/platform=iOS`) depuis `swift-mlx-profiler`
  1.5.1, qui réserve à macOS son code à base de `Process` (Metal System Trace via xctrace, capture des
  réglages système) ; le reste du profiler fonctionne sur iOS.

## Utilisation

```swift
let profile = Gemma4ReferenceProfile.named("e2b/4bit-tiny")!
// ou : Gemma4ReferenceProfile.recommended(for: .e2b) — choisit `tiny` sous 6 Go disponibles

let pipeline = Gemma4Pipeline()
try await pipeline.load(profile: profile)

// ou le moteur de conversation (outils, pensée, images, réutilisation de conversation) :
let engine = try await Gemma4ChatEngine.load(from: modelDirectory, profile: profile)
```

Ce que fait le profil :

| Réglage | Valeur | Pourquoi |
|---|---|---|
| Tour audio | non chargée | 0,61 Go en bf16, même dans le pack 4 bits. `withAudioVariant()` la remet (+0,6 Go) ; un audio envoyé sans elle est refusé (`audioTowerUnavailable`), jamais ignoré |
| Tours vision | libérées après chaque préfill avec image, rechargées à la demande | −1,24 Go en régime pour +80 ms de TTFT à l'image suivante |
| Cache MLX | 256 Mo, vidé après chaque réponse | le cache de 1 Go du profil `lean` fait passer l'image à 4,17 Go |
| Seuil mémoire | `max(4 Go, disponible − 2 Go)` | disponible = `os_proc_available_memory()` |
| Tranche de préfill | 256 | |

## Mesures

Sur Mac (M3 Max), mémoire disponible simulée à 6 Go (`GEMMA4_AVAILABLE_MB=6000`), empreinte physique
maximale du processus (`phys_footprint`), 2 passes, sorties identiques au profil `fast`
(`benchmarks/residency-lot-i-20260929.jsonl`) :

| Charge | Empreinte max | Décodage | TTFT |
|---|---|---|---|
| texte, prompt 128 | 2,92 Go | 133 tok/s | 77-88 ms |
| texte, prompt 1 024 | 3,11 Go | 127 tok/s | 211-216 ms |
| texte, prompt 4 096 | 3,28 Go | 123 tok/s | 713-720 ms |
| image (1 image, 280 jetons) | **3,90 Go** | 130 tok/s | 315-319 ms |

Pour comparaison, `e2b/4bit-fast` : texte 3,39 Go, image 5,06 Go.

**Non vérifié** : aucune mesure sur un iPhone réel à ce jour. Les débits d'un iPhone seront plus bas que
ceux d'un M3 Max ; l'empreinte, elle, dépend peu de la puce.

## Limites connues

- Audio et vidéo : non couverts par la porte des 4 Go (audio désactivé dans le profil ; la vidéo,
  plusieurs frames, n'a pas été mesurée sous ce profil).
- Plusieurs images par requête : chaque image ajoute ses 280 jetons et son passage dans la tour vision.
- Pas d'entraînement LoRA sur iPhone (pic de 40 Go sur E2B bf16).
