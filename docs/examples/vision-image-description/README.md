# Vision: Image Description & Understanding

Multimodal image understanding across all Gemma 4 model families (4-bit quantization). Tests include single image description, UI/OCR comprehension, and multi-image reasoning.

## Hardware

| Component | Specification |
|-----------|--------------|
| **Machine** | Mac Studio |
| **Chip** | Apple M3 Max |
| **Memory** | 96 GB Unified Memory |
| **OS** | macOS 26.4 |
| **MLX** | mlx-swift 0.30.6+ |

## Test Images

### input_sample.jpg — Vehicle Identification

![input_sample.jpg](input_sample.jpg)

A photograph of a classic red car parked in a natural setting.

### UI.png — Interface Comprehension + OCR

![UI.png](UI.png)

A macOS application screenshot (FluxForge Studio) with French-language UI elements.

---

## Test 1: Vehicle Description

**Prompt:** `Describe this image in detail. What type of vehicle is this? What color? What is the setting?`

**Command:**
```bash
gemma4-cli describe --model-path ~/Library/Caches/models/mlx-community/gemma-4-{model}-it-4bit \
  --image input_sample.jpg \
  --prompt "Describe this image in detail. What type of vehicle is this? What color? What is the setting?" \
  --max-tokens 500 --temperature 0.3
```

### Performance

| Model | Tokens | Speed | GPU Peak | Vehicle ID |
|-------|:------:|:-----:|:--------:|------------|
| **E2B 4-bit** | 307 | 74.2 tok/s | 4.4 Go | "classic, small, open-top car, vintage" |
| **E4B 4-bit** | 325 | 44.9 tok/s | 5.9 Go | "FIAT 600 or similar era microcar" |
| **26B-A4B 4-bit** | 337 | 24.8 tok/s | 15.9 Go | "Citroën 2CV (Deux Chevaux)" |
| **26B-A4B bf16** | 272 | **1.4 tok/s** ⚠ | 55.6 Go | "Citroën 2CV (vintage economy car)" |
| **31B 4-bit** | 145 | 7.4 tok/s | 19.0 Go | "red Citroën 2CV" |
| **DiffusionGemma 26B-A4B bf16** | 256 | 22.9 tok/s | 50.8 Go | "Citroën 2CV (classic French)" |

### Key Observations

- **E2B** identifies the era and style but not the exact model
- **E4B** guesses "FIAT 600" — close but wrong manufacturer
- **26B-A4B** correctly identifies "Citroën 2CV" with the French nickname "Deux Chevaux"
- **31B** also identifies "Citroën 2CV" and provides the most concise, accurate description with specific details (black soft-top roof, chrome hubcaps, gravel path)
- **DiffusionGemma 26B-A4B** correctly identifies the Citroën 2CV via block-AR
  diffusion (256-token canvas, 9-15 denoising steps per canvas via adaptive
  stopping). Quality on par with 26B-A4B AR, generation paradigm is fundamentally
  different (parallel denoising vs token-by-token). See `a4b-diff_2cv.txt`.

### 31B Output (Best Quality)

> A side-profile, full-shot photograph shows a red Citroën 2CV parked on a gravel and grass surface in front of a dense green hedge. The car is a small, two-door sedan with a rounded body and a black soft-top roof. The car is painted a bright red, and the wheels are silver with chrome hubcaps. The car is parked on a light-colored gravel path, with patches of green grass and small weeds growing around it. The background is a thick wall of green foliage, with various types of leaves and branches. The lighting is bright and even, suggesting an overcast day. The car is the central focus of the image, and the composition is simple and clean.

---

## Test 2: UI Description + OCR

**Prompt:** `Describe this user interface screenshot in detail. Read all visible text including French text. Describe the layout and application purpose.`

### Performance

| Model | Tokens | Speed | GPU Peak |
|-------|:------:|:-----:|:--------:|
| **E2B 4-bit** | 500 | 77.2 tok/s | 4.4 Go |
| **E4B 4-bit** | 500 | 46.1 tok/s | 5.9 Go |
| **26B-A4B 4-bit** | 500 | 24.7 tok/s | 15.9 Go |
| **26B-A4B bf16** | 500 | **0.9 tok/s** ⚠ | 55.6 Go |
| **31B 4-bit** | 500 | 8.3 tok/s | 19.5 Go |
| **DiffusionGemma 26B-A4B bf16** | 512 | 26.7 tok/s | 50.9 Go |

### Text Recognition (French UI Elements)

All 4 AR models + DiffusionGemma correctly read the following French text:

| UI Element | Text | All Models |
|------------|------|:---:|
| Main title | "Forge ton idée" | yes |
| Menu item | "Génération d'image" | yes |
| Menu item | "Générer une vidéo" | yes |
| Queue badge | "File d'attente" + badge "1" | yes |
| Tool | "Supprimer l'arrière-..." | yes |
| Tool | "Upscale" | yes |
| Tool | "Créer un LoRA" | yes |
| Library | "Bibliothèque" | yes |
| Library | "Adaptateurs LoRA" | yes |
| Settings | "Modèles", "Préférences", "Cache" | yes |
| Help | "Documentation" | yes |
| App name | "FluxForge Studio" (central area) | 26B, 31B |
| Status | "Débruitage... 36%" | 26B, 31B |
| Button | "Nouvelle idée" | 26B, 31B, **DiffusionGemma** |
| Status | "Auto-sauvé" / "Auto-sauveé" | 26B, **DiffusionGemma** |
| Section | "Historique 2" | **DiffusionGemma only** |
| Misc | "Site web & Support" | **DiffusionGemma only** |

### Key Observations

- All models read the main menu items correctly (French text)
- **26B-A4B** and **31B** additionally identify smaller UI elements (progress bar text, button labels, app name in the central area)
- **E4B** identifies the application as "an AI image generation tool"
- **26B-A4B** provides the most complete inventory of visible text for the AR family
- **DiffusionGemma 26B-A4B** is the **most thorough** : reads "Historique 2" and
  "Site web & Support" that no AR model mentions, and ventures into the "Forge
  ton idée" workspace description with "Nouvelle idée" back button and
  "Auto-sauveé" badge. See `a4b-diff_ui.txt`.

---

## Test 3: Multi-Image Reasoning

**Prompt:** `Find 3 things these two images have in common. Be creative!`

**Command:**
```bash
gemma4-cli describe --model-path ~/Library/Caches/models/mlx-community/gemma-4-26b-a4b-it-4bit \
  --image input_sample.jpg --image UI.png \
  --prompt "Find 3 things these two images have in common. Be creative!" \
  --max-tokens 400 --temperature 0.5
```

### Performance

| Model | Tokens | Speed | GPU Peak |
|-------|:------:|:-----:|:--------:|
| **E2B 4-bit** | 246 | 63.5 tok/s | 5.2 Go |
| **E4B 4-bit** | 400 | 42.0 tok/s | 6.7 Go |
| **26B-A4B 4-bit** | 242 | 20.2 tok/s | 16.3 Go |
| **26B-A4B bf16** | 254 | **1.4 tok/s** ⚠ | 55.9 Go |
| **31B 4-bit** | 233 | 6.7 tok/s | 19.9 Go |
| **DiffusionGemma 26B-A4B** | — | — | — | not supported (Phase 4 simplification, see `a4b-diff_multi.txt`) |

### 26B-A4B Output (Most Creative)

> 1. **The "Red" Connection:** The first image is dominated by a vibrant, saturated red car, while the second image features a subtle, dark reddish-purple hue in its background and UI elements. They both use a strong, singular color palette to define their visual identity.
>
> 2. **A Journey in Progress:** The first image shows a vehicle designed for travel, representing a physical journey. The second image shows a loading bar ("Débruitage... 36%"), representing a digital journey. Both images capture a moment of transition or movement toward a destination.
>
> 3. **The Concept of "Creation":** The first image features a classic, iconic design (the Citroën 2CV) that is a masterpiece of industrial creation. The second image is a screenshot of a creative tool (an AI generation interface), showing the process of digital creation in real-time. One is a finished product of human engineering; the other is the birth of a new digital idea.

### Key Observations

- All models successfully process both images simultaneously and find creative connections
- **26B-A4B** reads "Débruitage... 36%" from the UI to build the "journey in progress" metaphor
- The multi-image test validates that the vision encoder processes each image independently and the language model can reason across both

---

## Summary

| Capability | E2B | E4B | 26B-A4B | 31B |
|------------|:---:|:---:|:-------:|:---:|
| Vehicle description | Generic | Approximate ID | **Exact ID** | **Exact ID** |
| OCR (French text) | Main menus | Main menus | **All text** | Most text |
| Multi-image reasoning | Basic | Good | **Creative** | Good |
| Speed | **74 tok/s** | 45 tok/s | 25 tok/s | 8 tok/s |
| GPU Memory | **4.4 Go** | 5.9 Go | 15.9 Go | 19.5 Go |

**Recommendation:** 26B-A4B offers the best quality/resource trade-off for vision tasks. E2B is 3x faster but less accurate for identification. 31B has the highest accuracy but is 3x slower than 26B-A4B.

---

## bf16 vs 4-bit on the same 26B-A4B backbone

Same `mlx-community/gemma-4-26b-a4b-it-{4bit, bf16}` checkpoints, same CLI, same prompts, same M3 Max 96 GB.

| Test | 4-bit | bf16 | Slowdown |
|---|---|---|---|
| Vehicle (1 image, ~270 tokens) | 24.8 tok/s | 1.4 tok/s | **17.7x** |
| UI/OCR (1 image, 500 tokens) | 24.7 tok/s | 0.9 tok/s | **27.4x** |
| Multi-image (2 images, ~250 tokens) | 20.2 tok/s | 1.4 tok/s | **14.4x** |
| **Geometric mean** | — | — | **~19x** |

### Why is the slowdown so large?

Naive bandwidth ratio = bf16 weight size / 4-bit weight size = 4x. Observed = ~19x. The extra ~5x comes from:

1. **MLX fused quantized kernels.** The 4-bit path uses matmul + dequant fused kernels that are heavily tuned on Apple Silicon. The bf16 path is pure-bandwidth-bound dense matmul — no fusion opportunity.
2. **Wired memory saturation.** bf16 peaks at 55.9 Go GPU (4-bit peaks at 15.9 Go). On a 96 Go Unified Memory, bf16 is right against the wired-memory budget — every forward triggers cache thrashing between the metal heap and the VM.
3. **LM head cost.** Final logits projection (262k vocab × 4096 hidden, ~2 Go in bf16) is read at every generated token. In 4-bit it's ~512 Mo — 4x less bandwidth per token, on top of the matmul kernel difference.

### Quality difference: marginal

Compare the bf16 Citroen 2CV description (`26b-a4b-bf16_2cv.txt`) with the 4-bit one (`26b-a4b_2cv.txt`):

- Both identify the **Citroen 2CV** correctly with the same confidence.
- Both list the same fine details (chrome trim, stickers on rear window, bushes background, overcast lighting).
- bf16 mentions the **radio antenna**, 4-bit mentions the **sloping rear roofline** — different small details, not better or worse.
- On the multi-image creative reasoning test, bf16 produces "Nostalgia vs Future" as the 3rd theme; 4-bit produces "Concept of Creation". Both equally valid, neither obviously better.

**Conclusion:** bf16 brings no measurable quality advantage on standard vision/OCR/multi-image tasks for 26B-A4B, at a 17-27x cost. **Always prefer 4-bit for vision use cases.** The bf16 checkpoint is only useful as a baseline for quantization studies, or for the diffusion variant where bf16 is mandatory (no quantized release available yet).

### Are the GUI numbers a bug?

The GUI bench app loads `mlx-community/gemma-4-26b-a4b-it-bf16` and shows ~1.4 tok/s on Vehicle-style prompts. **This matches the CLI baseline exactly** (1.4 tok/s on the same test). No bug in the multimodal call path — just bf16 being intrinsically slow. Loading the 4-bit checkpoint in the GUI would yield ~25 tok/s.

---

*Full model outputs are available in this directory as `{model}_{test}.txt` files. The bf16 results are in `26b-a4b-bf16_{2cv,ui,multi}.txt`.*
