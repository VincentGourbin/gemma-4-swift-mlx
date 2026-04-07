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
| **31B 4-bit** | 145 | 7.4 tok/s | 19.0 Go | "red Citroën 2CV" |

### Key Observations

- **E2B** identifies the era and style but not the exact model
- **E4B** guesses "FIAT 600" — close but wrong manufacturer
- **26B-A4B** correctly identifies "Citroën 2CV" with the French nickname "Deux Chevaux"
- **31B** also identifies "Citroën 2CV" and provides the most concise, accurate description with specific details (black soft-top roof, chrome hubcaps, gravel path)

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
| **31B 4-bit** | 500 | 8.3 tok/s | 19.5 Go |

### Text Recognition (French UI Elements)

All 4 models correctly read the following French text:

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
| Button | "Nouvelle idée" | 26B, 31B |
| Status | "Auto-sauvé" | 26B |

### Key Observations

- All models read the main menu items correctly (French text)
- **26B-A4B** and **31B** additionally identify smaller UI elements (progress bar text, button labels, app name in the central area)
- **E4B** identifies the application as "an AI image generation tool"
- **26B-A4B** provides the most complete inventory of visible text

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
| **31B 4-bit** | 233 | 6.7 tok/s | 19.9 Go |

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

*Full model outputs are available in this directory as `{model}_{test}.txt` files.*
