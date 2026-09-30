# Gemma 4 Swift MLX

Native Gemma 4 multimodal inference for Apple Silicon via [MLX Swift](https://github.com/ml-explore/mlx-swift).

## Status

| Feature | Status | Details |
|---------|--------|---------|
| Text generation | ✅ **Working** | 5 families (E2B, E4B, 12B, 26B-A4B, 31B), 4/8/16-bit packs. 4.5–134 tok/s decode (M3 Max, measured profiles) |
| Vision (image understanding) | ✅ **Working** | Single + multi-image. E2B, E4B, 26B-A4B, 31B validated |
| Video (frame-by-frame) | ✅ **Working** | ~1fps, 70 tokens/frame, MM:SS timestamps. All 4 families validated |
| Audio (speech understanding) | ✅ **Working** | Conformer encoder, 30s max, ASR/comprehension. E2B + E4B validated |
| Thinking mode filter | ✅ **Working** | Filters `<\|channel>thought` blocks. Structured response separation |
| LoRA/DoRA fine-tuning | ✅ **Working** | LoRA, DoRA, full SFT. Response masking, chat template. 97% accuracy on classifier benchmark |
| Multimodal LoRA | ✅ **Working** | Audio + vision fine-tuning. 50% accuracy on 20-species bird call classification, LaTeX OCR verified |
| Speculative decoding (MTP) | ✅ **Working** | Gemma 4 Assistant drafter. Matches greedy decoding except on near-ties of logits (not bit-exact in general). Fine-tuneable: ×2.5 acceptance on domain-specific dataset |
| KV cache quantization | ✅ **Working** | mlx-swift-lm native `QuantizedKVCache` via `kvBits`, all families (8-bit KV in the `lean` profiles of 26B-A4B and 31B). TurboQuant kept, explicit only. See [KV Cache Quantization](#kv-cache-quantization) |
| Multi-turn chat | ✅ **Working** | Via ChatSession streaming, or `Gemma4ChatEngine` (tools, thinking channel, images, prefix reuse: TTFT −78 % on the next turn) |
| OpenAI-compatible server | ✅ **Working** | `Server/` package, `gemma4-server` (SSE, tools, images, API key, per-client conversation cache). See [Inference Server](#inference-server-openai-compatible) |
| Reference profiles | ✅ **Measured** | `<bits>bit-<fast\|lean>` for inference (30 profiles), `lora-<bits>bit-<fast\|lean>` for training (11 profiles), `a4bdiff/*` for diffusion. `gemma4-cli references`, `lora profiles` |
| iPhone / iPad | 🧪 **Profile ready** | `e2b/4bit-tiny`: under 4 GB footprint (text 3.1 GB, image 3.9 GB), measured on Mac only. See [docs/iOS.md](docs/iOS.md) |
| Profiling toolkit | ✅ **Working** | Chrome Trace export, SQLite benchmarks, context sweep |
| Model download | ✅ **Working** | Direct HTTPS from HuggingFace (no HF SDK dependency) |
| **DiffusionGemma 26B-A4B** | ✅ **Ported** | Block-AR text diffusion + vision. **80.8% OCRBench**, **79% ScreenSpot v1**, 95% BFCL. Pre-quantized 8-bit and 4-bit (mixed) packs on Hugging Face: `gemma4-cli download diff-8bit` / `diff-4bit`. Voir [docs/DIFFUSIONGEMMA.md](docs/DIFFUSIONGEMMA.md) |
| `gemma4-bench-ui` GUI | ✅ **Working** | 4 onglets (Bench AR vs Diffusion, Web agent step-by-step, Akinator VQA, iOS Sim agent) |

## Requirements

- macOS 15+ (Sequoia) — `Package.swift` : `.macOS(.v15)`
- Apple Silicon (M1/M2/M3/M4)
- Xcode 26 or later with Swift 6.3+ (`mlx-swift` 0.31.6 declares swift-tools-version 6.3; CI builds with
  Xcode 26.6, development uses Xcode 27)

## Quick Start

### Build

```bash
git clone https://github.com/VincentGourbin/gemma-4-swift-mlx
cd gemma-4-swift-mlx
xcodebuild -scheme gemma4-cli -configuration Release \
  -destination "platform=macOS" -derivedDataPath .build/xcode \
  -skipMacroValidation build
```

> **Note:** Use `xcodebuild` (not `swift build`) — Metal shader support required by MLX.

### Choose where models live

```bash
export GEMMA4_MODELS_DIR=/Volumes/YourSSD/models   # default: ~/Library/Caches/models
```

Order of precedence: `Gemma4ModelCache.customModelsDirectory` (set by an app), then
`$GEMMA4_MODELS_DIR`, then `~/Library/Caches/models`. Models already present in the default
location are still found. If the directory sits on an external volume that is not mounted,
downloads fail with `Gemma4DownloadError.volumeNotMounted` instead of writing to a fake
`/Volumes/…` folder on the internal disk. `gemma4-cli download --models-dir <dir>` overrides
it for one command. Link weight **files**, never the model folder itself: the loader does not
walk a model folder that is a symbolic link.

### Download a model

```bash
# List available models
gemma4-cli models

# Download (e.g., E2B 4-bit, ~3.6 GB)
gemma4-cli download e2b-4bit

# Shortcuts: e2b-4bit, e4b-4bit, a4b-4bit, 31b-4bit, e2b-8bit, e2b-bf16, ...
```

### Text generation

```bash
gemma4-cli generate --model-path ~/Library/Caches/models/mlx-community/gemma-4-e2b-it-4bit \
  "Explain machine learning in 3 sentences" --max-tokens 200
```

### Image description

```bash
# Single image
gemma4-cli describe --model-path ~/Library/Caches/models/mlx-community/gemma-4-26b-a4b-it-4bit \
  --image photo.jpg --prompt "Describe this image in detail."

# Multi-image comparison
gemma4-cli describe --model-path ~/Library/Caches/models/mlx-community/gemma-4-26b-a4b-it-4bit \
  --image photo1.jpg --image photo2.png \
  --prompt "What do these images have in common?"
```

### Video description

```bash
gemma4-cli describe --model-path ~/Library/Caches/models/mlx-community/gemma-4-e2b-it-4bit \
  --video clip.mp4 \
  --prompt "Describe this video in detail."
```

> Video is processed as ~1fps frames (max 32 frames, 60s) with 70 soft tokens per frame and MM:SS timestamps. Uses `video_token` (ID 258884), aligned with Google's reference implementation. See [docs/examples/video-description/](docs/examples/video-description/) for benchmarks.

### Audio transcription

```bash
gemma4-cli describe --model-path ~/Library/Caches/models/mlx-community/gemma-4-e2b-it-4bit \
  --audio speech.mp3 \
  --prompt "Transcribe the following speech segment in English into English text."
```

> Audio supports up to 30 seconds, processed via Conformer encoder (750 tokens max). Only E2B and E4B models have an audio tower. Supports ASR (transcription) and comprehension tasks.

### LoRA fine-tuning

```bash
# Train a LoRA adapter
gemma4-cli lora train \
  --model-path ~/Library/Caches/models/mlx-community/gemma-4-e2b-it-bf16 \
  --data /path/to/dataset \
  --output ./my-adapter \
  --reference lora-16bit-fast --mask-prompt --iterations 1300   # measured profile, see below

# Generate with adapter
gemma4-cli lora generate \
  --model-path ~/Library/Caches/models/mlx-community/gemma-4-e2b-it-bf16 \
  --adapter-path ./my-adapter \
  "your prompt here"

# Fuse adapter into model weights (permanent)
gemma4-cli lora fuse \
  --model-path ~/Library/Caches/models/mlx-community/gemma-4-e2b-it-bf16 \
  --adapter-path ./my-adapter \
  --output ./fused-model
```

See [LoRA Fine-Tuning Guide](#lora-fine-tuning) for details.

### Interactive chat

```bash
gemma4-cli chat --model-path ~/Library/Caches/models/mlx-community/gemma-4-e2b-it-4bit
```

### Profiling

```bash
# Single profiled run with Chrome Trace export
gemma4-cli profile run --model-path ~/Library/Caches/models/mlx-community/gemma-4-e2b-it-4bit \
  --max-tokens 100 --prompt "Hello"

# Context size sweep with SQLite output
gemma4-cli profile sweep --model-path ~/Library/Caches/models/mlx-community/gemma-4-e2b-it-4bit \
  --context-sizes 500,2000,8000 --kv-bits-list 0,4 --output results.sqlite
```

### GUI bench app (`gemma4-bench-ui`)

A small SwiftUI app for interactive testing of the multimodal pipelines —
4 tabs : AR vs Diffusion bench, web agent step-by-step, Akinator VQA,
iOS Simulator agent.

```bash
# Build (Release for the best inference speed)
xcodebuild -scheme gemma4-bench-ui -configuration Release \
  -destination "platform=macOS" -derivedDataPath .build/xcode \
  -skipMacroValidation build

# Run
.build/xcode/Build/Products/Release/gemma4-bench-ui
```

> Or open the package in Xcode and run the `gemma4-bench-ui` scheme
> directly. First launch downloads the selected model from HuggingFace
> (cached under `~/Library/Caches/models/`).

**Tabs :**

- **Bench AR vs Diffusion** — side-by-side comparison of an autoregressive
  Gemma 4 family (E2B/E4B/26B-A4B/31B, 4-bit/bf16) vs DiffusionGemma
  26B-A4B on the same prompt. Shows tok/s, GPU peak, latency.
- **Web agent (step-by-step)** — pilot a `WKWebView` with DiffusionGemma
  or E4B as the visual planner. Action vocabulary: `click` / `scroll` /
  `type` / `click_and_type` / `done`. Each run is logged to
  `/tmp/web-agent-runs/run-<ts>/` (prompts, raw outputs, screenshots).
  See [docs/DIFFUSIONGEMMA.md](docs/DIFFUSIONGEMMA.md#use-cases-web-navigation)
  for the Wikipedia "Montcuq" demo.
- **VQA Akinator** — vision-language guessing game. Model asks
  yes/no questions about an image you choose, narrows down to a guess.
- **iOS Simulator agent** — visual agent that drives the iOS Simulator
  via `xcrun simctl`. Limitations of small-model tap precision documented
  in [issue #30](https://github.com/VincentGourbin/gemma-4-swift-mlx/issues/30).

For the Toolathlon tool-use benchmark POC (CLI + OpenAI-compatible proxy
+ MCP harness on `find-alita-paper`), see
[docs/examples/toolathlon-bench/](docs/examples/toolathlon-bench/).

## Supported Models

### Model Families

| Family | Total Params | Active Params | MoE | Audio | Key Features |
|--------|:---:|:---:|:---:|:---:|---|
| **E2B** | 5.1B | 2.3B | No | Yes | Fastest. Text + Vision + Audio + Video |
| **E4B** | 9.6B | 4.5B | No | Yes | Best quality/size ratio. Full multimodal |
| **12B** | ~12B | ~12B | No | — | Unified multimodal checkpoint. **8-bit recommended** (4-bit loses 20 MMLU points) |
| **31B** | 31.3B | 31.3B | No | No | Highest quality. Text + Vision + Video |
| **26B-A4B** | 25.8B | 3.8B | Yes (128 experts, top-8) | No | MoE efficiency. Text + Vision + Video |

### Available Quantizations

| Model | 4-bit | 6-bit | 8-bit | BF16 | HuggingFace ID pattern |
|-------|:---:|:---:|:---:|:---:|---|
| **E2B** | ~3.6 GB | ~4.2 GB | ~5.2 GB | ~10 GB | `mlx-community/gemma-4-e2b-it-{quant}` |
| **E4B** | ~5 GB | ~6.5 GB | ~8 GB | ~19 GB | `mlx-community/gemma-4-e4b-it-{quant}` |
| **12B** | ~7 GB | ~10 GB | ~13 GB | ~24 GB | `mlx-community/gemma-4-12B-it-{quant}` |
| **31B** | ~17 GB | ~25 GB | ~33 GB | ~63 GB | `mlx-community/gemma-4-31b-it-{quant}` |
| **26B-A4B** | ~14 GB | ~21 GB | ~27 GB | ~52 GB | `mlx-community/gemma-4-26b-a4b-it-{quant}` |

> Additional formats available: `mxfp4`, `mxfp8`, `nvfp4`, `5-bit`. See [mlx-community on HuggingFace](https://huggingface.co/mlx-community?search=gemma-4).

## Performance (Apple M3 Max, 96 GB)

### Text Generation (decode tok/s)

Median decode speed of the `fast` reference profiles, 128-token prompt, measured with
`gemma4-cli bench` on the library's own generation path (`TokenIterator`, release build). Prefill,
TTFT and memory at 128 / 1k / 4k tokens, `lean` profiles and images: [docs/References.md](docs/References.md).

| Model | 4-bit | 8-bit | BF16 |
|-------|:-----:|:-----:|:----:|
| **E2B** | **134.5** | 87.5 | 53.6 |
| **E4B** | **78.0** | 47.8 | 27.6 |
| **12B** | 36.2 | **20.9** | 11.6 |
| **26B-A4B** | **81.3** | 50.9 | 32.6 |
| **31B** | **15.1** | 8.2 | 4.6 |

> Older figures (6-bit column, video and audio tables below) came from the `profile` command,
> whose decode loop missed the library's `asyncEval` pipelining; they are kept for the quality
> observations, not as speed references. Full raw results: [benchmarks/](benchmarks/results/).

### Vision Quality (vehicle identification across quantizations)

| Model | 4-bit | 6-bit | 8-bit | BF16 |
|-------|:-----:|:-----:|:-----:|:----:|
| **E2B** | "classic car" | "VW Beetle" | "Fiat 600" | "VW Beetle" |
| **E4B** | "classic Fiat" | "VW Beetle" | "Citroën 2CV" | "VW Beetle" |
| **26B-A4B** | **"Citroën 2CV"** | **"Citroën 2CV"** | **"Citroën 2CV"** | **"Citroën 2CV"** |
| **31B** | **"Citroën 2CV"** | **"Citroën 2CV"** | **"Citroën 2CV"** | **"Citroën 2CV"** |

> **Key finding:** Quality depends on architecture more than quantization. 26B-A4B/31B identify the vehicle correctly at all quantizations, and 4-bit is the sweet spot for E2B, E4B, 26B-A4B and 31B. **Exception: 12B**, where 4-bit drops MMLU from 57 % to 37 % (100 questions): use 8-bit. See [docs/examples/](docs/examples/) and [benchmarks/](benchmarks/results/) for full results.

### Video (4-bit models, 9 frames ~1fps, 70 tokens/frame)

| Model | Speed | GPU Peak | Temporal Reasoning |
|-------|:-----:|:--------:|:---:|
| **E2B** | 50.8 tok/s | 5.6 GB | Basic (per-frame) |
| **E4B** | 36.5 tok/s | 7.0 GB | Good (time ranges) |
| **26B-A4B** | 18.8 tok/s | 17.0 GB | Excellent (motion/depth) |
| **31B** | 5.8 tok/s | 20.8 GB | Best (concise, natural stop) |

### Audio (4-bit models, 30s speech, 750 tokens)

| Model | Transcription | Comprehension | GPU Peak |
|-------|:---:|:---:|:--------:|
| **E2B** | ✅ Accurate | ✅ Context understood | 5.6 GB |
| **E4B** | ✅ Accurate | ✅ Structured analysis | 7.1 GB |
| **26B-A4B** | — | No audio tower | — |
| **31B** | — | No audio tower | — |

## LoRA Fine-Tuning

Train LoRA, DoRA, or full SFT adapters entirely on-device. Compatible with [mlx-lm](https://github.com/ml-explore/mlx-swift-lm) Python adapters — train in one, infer in the other.

> **Do not run training concurrently with inference.** `mlx-swift` takes its
> per-compiled-function lock and its global `evalLock` in opposite orders in
> `CompiledFunction.call` versus `vjp`/`jvp`, so a gradient step on one thread and
> a forward pass on another can deadlock the process (both locks are global —
> `Gemma4LoRATrain.train` is callable from any task, `Gemma4Pipeline` is
> `@MainActor`). Serialize the two until this is fixed upstream.

### Supported modes

| Mode | Description | Memory (E2B bf16, short sequences) |
|------|-------------|:---:|
| **LoRA** | Low-rank adaptation (default) | ~11 GB |
| **DoRA** | Weight-Decomposed LoRA | ~11 GB |
| **Full SFT** | All weights trainable | ~20 GB |

Memory grows with sequence length: see [Training profiles](#training-profiles) for measured peaks on
examples up to 3,320 tokens.

### Dataset format

JSONL with chat messages (same format as mlx-lm):

```jsonl
{"messages": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "Hi!"}]}
{"messages": [{"role": "user", "content": "bye"}, {"role": "assistant", "content": "Goodbye!"}]}
```

Organize as:
```
my-dataset/
├── train.jsonl
├── valid.jsonl
└── test.jsonl    # optional
```

### CLI commands

```bash
# Train
gemma4-cli lora train \
  --model-path <model> \
  --data <dataset-dir> \
  --output ./adapters \
  --reference lora-16bit-fast \  # Measured profile (see below); overrides the knobs it sets
  --mask-prompt \              # Loss only on response tokens (recommended)
  --num-layers 16 \            # Number of layers to adapt
  --iterations 1300 \          # Training steps
  --learning-rate 1e-4 \       # Adam LR
  --rank 8 \                   # LoRA rank (default: 8)
  --scale 20.0 \               # LoRA alpha/scaling (default: 20.0)
  --fine-tune-type lora        # lora | dora | full

# Evaluate
gemma4-cli lora eval \
  --model-path <model> \
  --adapter-path ./adapters \
  --data <dataset-dir>

# Generate
gemma4-cli lora generate \
  --model-path <model> \
  --adapter-path ./adapters \
  --temperature 0.3 \
  "your prompt"

# Fuse (merge adapter into model permanently)
gemma4-cli lora fuse \
  --model-path <model> \
  --adapter-path ./adapters \
  --output ./fused-model
```

### Training profiles

`--reference <profile>` sets rank, scale, layers, learning rate, batch, gradient checkpointing,
MLX cache limit and validation batches in one go (`gemma4-cli lora profiles` lists them).
All use rank 8, scale 20, lr 1e-4, batch 1, loss and head on the response only. `lean` adds
per-layer gradient checkpointing (same losses, −33 to −44 % peak, −20 to −23 % speed).

Measured on the Fluxforge "director" dataset (898 chat examples up to 3,320 tokens, 50 steps,
M3 Max 96 GB); throughput counts trained (response) tokens:

| Profile | Base weights | Peak MLX | Footprint | Throughput | Val loss (step 1 → 50) |
|---|---|---|---|---|---|
| `e2b/lora-16bit-fast` | E2B bf16 | 37.0 GB | 13.9 GB | 218 tok/s | 1.847 → 1.276 |
| `e2b/lora-16bit-lean` | E2B bf16 | 20.7 GB | 12.8 GB | 168 tok/s | 1.847 → 1.276 |
| `e2b/lora-4bit-lean` | E2B 4-bit | 14.5 GB | 6.7 GB | 132 tok/s | 1.893 → 1.339 |
| `e4b/lora-16bit-fast` ✅ E7 29/30 | E4B bf16 | 35.7 GB | 18.7 GB | 144 tok/s | 1.411 → 1.157 |
| `e4b/lora-16bit-lean` | E4B bf16 | 24.1 GB | 18.3 GB | 115 tok/s | 1.411 → 1.157 |
| `e4b/lora-8bit-lean` | E4B 8-bit | 16.8 GB | 11.5 GB | 85 tok/s | 1.412 → 1.160 |
| `b12b/lora-16bit-fast` | 12B bf16 | 37.9 GB | 27.5 GB | 35 tok/s | 1.424 → 1.131 |
| `b12b/lora-8bit-lean` | 12B 8-bit | 27.9 GB | 16.4 GB | 30 tok/s | 1.401 → 1.100 |
| `a4b/lora-16bit-fast` | 26B-A4B bf16 | 57.6 GB | 52.3 GB | 67 tok/s | 1.714 → 1.084 |
| `a4b/lora-4bit-lean` | 26B-A4B 4-bit | 20.4 GB | 17.6 GB | 68 tok/s | 1.867 → 1.140 |
| `b31b/lora-8bit-fast` | 31B 8-bit | 54.8 GB | 35.9 GB | 14 tok/s | 1.548 → 1.289 |

- Quality gate (E7, 30 held-out briefs validated by `director-tool`): E4B `lora-16bit-fast`, one full
  epoch → **29/30** (E4B base alone 19/30). E2B bf16 with the same defaults: 30/30.
- 12B, 26B-A4B and 31B keep gradient checkpointing even in `fast` (activations would not fit in 96 GB).
- MoE (26B-A4B): experts (`SwitchLinear`) are not adapted; attention and dense MLP are.
- `b31b/lora-4bit-lean` is not published: it diverges at lr 1e-4.
- Always use `--mask-prompt` for chat-format data. Long examples are not truncated by default
  (`--max-seq-length` to bound memory; truncation cut the end of director answers: 30/30 → 27/30).
- Reproducible runs (`--seed`), safe checkpoints and exact resume (`--resume`), JSONL metrics (`--metrics-out`).
- Adapters trained in Swift work in Python mlx-lm and vice versa

### Library API

```swift
import Gemma4Swift

// Load model + adapter for inference
let container = try await Gemma4Registration.loadContainer(from: modelURL)
try await Gemma4LoRAInference.loadAdapter(into: container, from: adapterURL)

// Or fuse permanently
try await Gemma4LoRAInference.fuseAdapter(into: container, from: adapterURL)

// Training
var config = Gemma4LoRATrain.TrainingConfig(
    modelFamily: .e2b,
    iterations: 1300,
    outputDirectory: outputURL,
    maskPrompt: true
)
Gemma4TrainingProfile.named("e2b/lora-16bit-fast")?.apply(to: &config)  // measured settings
try await Gemma4LoRATrain.train(
    container: container,
    trainData: trainTokens,    // [[Int]] — pre-tokenized sequences
    validData: validTokens,
    config: config
) { progress in
    print(progress)
    return .more
}
```

## Multimodal LoRA Fine-Tuning

Train LoRA adapters on audio or image inputs. The model learns to generate structured responses from multimodal content.

### Example: Bird Call Identification

Train a model to identify bird species from 5-second audio recordings. Dataset: [tglcourse/5s_birdcall_samples_top20](https://huggingface.co/datasets/tglcourse/5s_birdcall_samples_top20) (20 species, ~9600 recordings).

**Dataset format** — JSONL with `audio` or `image` field pointing to media files:

```jsonl
{"messages": [{"role": "user", "content": "identify"}, {"role": "assistant", "content": "{\"common_name\": \"Mallard\", \"scientific_name\": \"Anas platyrhynchos\", \"call_type\": \"song\"}"}], "audio": "audio/mallard_001.wav"}
{"messages": [{"role": "user", "content": "identify"}, {"role": "assistant", "content": "{\"common_name\": \"Common Raven\", \"scientific_name\": \"Corvus corax\", \"call_type\": \"call\"}"}], "audio": "audio/raven_042.wav"}
```

**Train:**

```bash
gemma4-cli lora train \
  --model-path ~/Library/Caches/models/mlx-community/gemma-4-e2b-it-bf16 \
  --data ./birdcall-dataset \
  --output ./birdcall-adapter \
  --multimodal \
  --mask-prompt \
  --num-layers 16 \
  --rank 16 \
  --learning-rate 5e-5 \
  --iterations 8636
```

**Benchmark:**

```bash
gemma4-cli lora bench-multimodal \
  --model-path ~/Library/Caches/models/mlx-community/gemma-4-e2b-it-bf16 \
  --adapter-path ./birdcall-adapter \
  --data ./birdcall-dataset \
  --max-tokens 100
```

**Results (E2B bf16, rank 16, 1 epoch on 8636 samples):**

| Metric | Value |
|--------|-------|
| Training loss | 0.036 |
| Validation loss | 0.038 |
| Species accuracy (20 classes) | 50% |
| GPU peak memory | 23 GB |

**Key insights:**
- Use **rich JSON responses** (50+ tokens) rather than short labels — more gradient signal for the frozen audio encoder
- Model produces valid, internally consistent JSON with species name, scientific name, and call description
- Use `--multimodal` flag to load the full multimodal model (vision + audio encoders)
- bf16 model recommended: the base stays in bf16 and only the LoRA parameters and the loss run in float32
  (peak 24 → 14 GB, +24 % throughput, same validation loss). `--fp32-model` restores the old full-float32 path,
  which the peak figures in these two examples were measured with.

### Example: LaTeX OCR

Train a model to convert images of mathematical equations into LaTeX code. Dataset: [unsloth/LaTeX_OCR](https://huggingface.co/datasets/unsloth/LaTeX_OCR) (68K images, we use a 500-sample subset).

**Dataset format:**

```jsonl
{"messages": [{"role": "user", "content": "Convert this mathematical expression to LaTeX."}, {"role": "assistant", "content": "\\sum _ { n = 1 } ^ { \\infty } \\frac { 1 } { n ^ { z } }"}], "image": "images/img_000192.png"}
```

**Train:**

```bash
gemma4-cli lora train \
  --model-path ~/Library/Caches/models/mlx-community/gemma-4-e2b-it-bf16 \
  --data ./latex-ocr-dataset \
  --output ./latex-adapter \
  --multimodal \
  --mask-prompt \
  --num-layers 16 \
  --rank 16 \
  --learning-rate 5e-5 \
  --iterations 500
```

**Results (E2B bf16, rank 16, 500 iterations on 500 images):**

| Metric | Value |
|--------|-------|
| Training loss | 0.41 |
| Validation loss | 0.36 |
| GPU peak memory | 24 GB |

The model generates contextually correct LaTeX for each input image — different equations produce different, appropriate LaTeX output. Sample:

| Input image content | Model output |
|--------------------|----|
| `\partial_+ \partial_- \Omega = 0` | `\partial _ { + } \partial _ { - } \Omega = 0` |
| `O_\Sigma = -\nabla^2_\Sigma + m^2` | `\mathcal { O _ { \Sigma } } = - \nabla _ { \Sigma } ^ { 2 } + m ^ { 2 }` |
| `z = \omega\tau / 2` | `z = \frac { \omega T } { 2 }` |

### Library API

```swift
import Gemma4Swift

let pipeline = Gemma4Pipeline()
try await pipeline.load(.e2b4bit, downloadIfNeeded: true)

// Load a multimodal LoRA adapter
try await pipeline.loadAdapter(from: adapterDirectoryURL)

// Or fuse it permanently for better inference speed
try await pipeline.fuseAdapter(from: adapterDirectoryURL)
```

### Supported modalities

| Modality | Training | Inference | Notes |
|----------|:--------:|:---------:|-------|
| Audio | ✅ | ✅ | Conformer encoder, 5-30s clips |
| Vision | ✅ | ✅ | SigLIP encoder, any image size |
| Video | - | ✅ | Inference only (training not yet implemented) |

## Speculative Decoding (MTP)

Accelerate text generation with Google's `gemma-4-{E2B,E4B}-it-assistant` drafter models via Multi-Token Prediction. The drafter proposes K-1 tokens per round; the target verifies all in one parallel forward and accepts only those matching its own argmax. Output matches standard greedy generation, except where two logits are nearly tied: the batched verify pass can flip such an argmax, so it is not bit-exact in general.

### Quick start

```bash
# Inference with pretrained drafter (~35% acceptance on generic prompts)
gemma4-cli generate \
  --model-path ~/Library/Caches/models/mlx-community/gemma-4-e2b-it-bf16 \
  --draft-model google/gemma-4-E2B-it-assistant \
  --temperature 0 \
  "Your prompt"

# Chat mode (multi-turn)
gemma4-cli chat \
  --model-path ~/Library/Caches/models/mlx-community/gemma-4-e2b-it-bf16 \
  --draft-model google/gemma-4-E2B-it-assistant
```

### Fine-tuning the drafter for your domain

The pretrained drafter is generic — to actually win throughput, fine-tune it on the kind of text your target produces. Self-distillation against `argmax(target_logits)`:

```bash
# Train (11 min for 2000 iter on 4.3k samples, batch=4)
gemma4-cli mtp-train \
  --target ~/Library/Caches/models/mlx-community/gemma-4-e2b-it-bf16 \
  --drafter google/gemma-4-E2B-it-assistant \
  --data my_corpus/train.jsonl \
  --valid-data my_corpus/valid.jsonl \
  --output ./my-drafter \
  --iterations 2000 --batch-size 4 \
  --steps-per-valid 200

# Inference with fine-tuned drafter + optional LoRA on target
gemma4-cli mtp-generate \
  --target mlx-community/gemma-4-e2b-it-bf16 \
  --drafter google/gemma-4-E2B-it-assistant \
  --drafter-path ./my-drafter/drafter.best.safetensors \
  --full-lm-head \
  --adapter-path ./my-target-lora \
  --prompt "Your domain-specific query"
```

Dataset format: same JSONL conventions as LoRA training (`{"text": "..."}` or `{"messages": [{"role": ..., "content": ...}]}`).

### Validation production (toolsforge French→SQL dataset)

| Metric | Pretrained drafter | Fine-tuned drafter | Δ |
|---|---|---|---|
| Acceptance moyenne | 8.8% | 22.4% | **×2.5** |
| Temps de génération | baseline | -12% | **-12%** |

Greedy equivalence preserved on this dataset. See PR #25 for full bench.

### Validation tools

```bash
gemma4-cli mtp-smoke <repo>              # validate drafter weights load cleanly
gemma4-cli mtp-forward                   # 1-round drafter parity test
gemma4-cli mtp-generate --compare        # compare against standard greedy generation
gemma4-cli mtp-diag-verify               # sequential vs parallel hidden diff (advanced)
```

## iPhone / iPad

Profile `e2b/4bit-tiny` keeps Gemma 4 E2B 4-bit under 4 GB of footprint (text 3.1 GB, one image 3.9 GB, measured on Mac with 6 GB available simulated): no audio tower, vision towers released after prefill, 256 MB MLX cache. See [docs/iOS.md](docs/iOS.md).

## Inference Server (OpenAI-compatible)

`gemma4-server` lives in the nested package `Server/`, so apps that depend on the library never
resolve Hummingbird or swift-nio (`Scripts/check-server-isolation.sh` checks it in CI).

```bash
cd Server && xcodebuild -scheme gemma4-server -configuration Release -destination "platform=macOS" \
  -derivedDataPath ../.build/xcode-server -skipMacroValidation build
../.build/xcode-server/Build/Products/Release/gemma4-server \
  --model-path /Volumes/Lexar/models/mlx-community/gemma-4-e2b-it-4bit --reference e2b/4bit-fast --port 8080
```

- `POST /v1/chat/completions` — JSON or SSE (`stream: true`); `tools` / `tool_calls` and `role: "tool"`
  turns; `reasoning_content` with `chat_template_kwargs: {"enable_thinking": true}`; images as
  `image_url` **`data:` base64 URLs only**. `GET /v1/models`, `GET /healthz`, `GET /metrics` (counters only).
- Listens on `127.0.0.1` by default; any other `--host` requires `--api-key` (or `GEMMA4_SERVER_API_KEY`),
  checked in constant time. Limits: 32 MiB body, 4 media, 20 Mpx per image, `max_tokens` cap, queue of 16
  (then HTTP 429). `input_audio` is rejected in v1. `--no-audio` skips the audio tower (−0.6 GB on E2B).
- One generation at a time: the queue is released only when the computation has really stopped. A
  client that disconnects cancels its generation; it is detected at the next failed write, so the next
  request's time to first token is about +18 ms (≈ 2 decode steps on E2B 4-bit) above an idle server.
- Conversation reuse: each turn keeps a snapshot of the KV caches at the end of its prompt; a request
  whose prompt strictly extends one of them only prefills the new suffix (LRU of 8 conversations,
  2 GB budget: `--conversation-cache-count`, `--conversation-cache-gb`, `0` disables). Reported as
  `usage.prompt_tokens_details.cached_tokens`. E2B 4-bit, ~2 350-token system prompt: 98 % of the prompt
  served from cache from turn 2, time to first token 460 → 100 ms; two interleaved clients both keep theirs.
- The engine is `Gemma4ChatEngine` in the library (no extra dependency): usable directly from an app.

## Library Integration

```swift
import Gemma4Swift

// Load a model (handles registration + tokenizer automatically)
let pipeline = Gemma4Pipeline()
try await pipeline.load(.e2b4bit)

// Or download + load in one call (no MLXLMCommon import needed)
try await pipeline.load(.e2b4bit, downloadIfNeeded: true) { progress in
    print("Downloading: \(Int(progress.fraction * 100))% — \(progress.currentFile)")
}

// Or from a custom local path
try await pipeline.load(from: URL(fileURLWithPath: "/path/to/model"))

// Chat
let response = try await pipeline.chat(prompt: "Hello!")

// Streaming
let stream = try pipeline.chatStream(prompt: "Write a poem")
for try await token in stream {
    print(token, terminator: "")
}

// Multi-turn
let followUp = try await pipeline.continueChat(prompt: "Make it shorter")
```

### Image preprocessing off the main thread

`Gemma4ImageProcessor.processImage` and `Gemma4UnifiedImageProcessor.processImage`
are synchronous: decoding (`NSImage` on macOS, `UIImage` on iOS) and resizing
(CoreGraphics) run **on the calling thread**. Called from a `@MainActor` view model
they block the main thread — measured at ~10 ms for a 1920×1280 JPEG on an M3 Max,
most of it the decode that `NSImage` defers until the first `draw`.

Both types expose an `async` overload that runs that work on a detached task:

```swift
// From a @MainActor context
let pixels = try await Gemma4ImageProcessor.processImage(
    url: imageURL,
    priority: .userInitiated)          // required — no default

let processed = try await Gemma4UnifiedImageProcessor.processImage(
    url: imageURL,
    config: visionConfig,
    priority: .userInitiated)
```

`priority` has **no default value** on purpose: it is what distinguishes the async
overload from the synchronous one of the same name, so existing synchronous call
sites keep resolving to the synchronous version and keep compiling unchanged.

Two things to know before using it:

- **The result is evaluated before it crosses the task boundary.** `MLXArray` is not
  thread safe — mlx-swift's own documentation states that it is *"not safe to create
  `c` in one thread and consume/evaluate it in another"* — so returning a lazy graph
  built on the detached task would be unsound. The array you get back is already
  materialized: the caller pays no compute, but the GPU cost is paid inside the call
  rather than at first use.
- **`Task.detached` does not inherit task-locals.** `withError`,
  `Device.withDefaultDevice`, `Stream.withNewDefaultStream` and the `MLXRandom` state
  installed by the caller do not cross into it: preprocessing runs on the global
  default device, and an MLX error there reaches the global handler (or `fatalError`)
  instead of your scoped one. If you depend on any of those, call the synchronous
  overload from a task you control.

Note the async overload does **not**, on its own, silence the Xcode *"User-initiated
thread waiting on a lower QoS thread"* diagnostic: the Thread Performance Checker
fires on the QoS pairing, not on whether the waiter is the main thread, and
`Task.value` escalates the detached task to the awaiting task's priority anyway. What
it fixes is the main thread being blocked, which is the part that is actually felt.

The video and audio processors (`Gemma4VideoProcessor`, `Gemma4UnifiedVideoProcessor`,
`Gemma4AudioProcessor`, `Gemma4UnifiedAudioProcessor`) are `async` statics that today
run off the main actor under SE-0338 semantics. That is a property of the current
language mode, not an annotation: enabling `NonisolatedNonsendingByDefault`
(Approachable Concurrency) would run them on the caller's actor, and
`Gemma4VideoProcessor.processVideo` calls the **synchronous** `processImage` up to 32
times in a loop. Only the image overloads above survive that flip, because they go
through `Task.detached`.

### System role

`chatStream` and `chatStreamMultimodal` both take an optional `systemPrompt`. It is
rendered as a **separate system turn** by the model's `chat_template.jinja` — Gemma 4
emits `<|turn>system … <turn|>` ahead of the user turn rather than folding the
instructions into it, which is what the HF reference implementations produce:

```swift
let stream = try pipeline.chatStreamMultimodal(
    prompt: "user prompt: a 2CV on a coastal road",
    pixelValues: pixels,
    systemPrompt: enhancerSystemPrompt,   // its own turn, not concatenated
    temperature: 0.0,
    maxTokens: 600)
```

Concatenating the instructions into the user turn instead produces a structurally
different render, and the model follows them less closely.

`nil` (the default) emits no system turn. The image expansion (`boi + image_token ×
280 + eoi`) stays confined to the user turn; an image marker inside `systemPrompt` is
rejected with `invalidInput`, since `maskedScatter` would otherwise be handed more
positions than there are image embeddings.

The ids are token-for-token identical to the HF render of the same
`chat_template.jinja`. Two stray newlines that swift-jinja emits are repaired on this
path: a `\n` between `<bos>` and the first `<|turn>`, and `\n\n` instead of `\n` between
the system and user turns. Both come from one cause — the template's
`{#- Pre-scan … -#}` comment tag, whose leading `-` should swallow the preceding
whitespace; jinja2 honours that whitespace control on comments, swift-jinja does not.
Because of this repair, ids on this path differ from pre-1.3.0 output by one `\n` even
when `systemPrompt` is `nil`.

The text path (`chatStream`, `chat`) goes through `ChatSession` and still carries the
first artifact; a general fix belongs upstream in swift-jinja. Note also that those
two entry points are **not** symmetric with the multimodal one: they fall back to a
default `"Tu es un assistant utile."` system turn when `systemPrompt` is `nil`, so
there is currently no way to ask the text path for no system turn at all.

### Chat template variables (thinking mode)

`chatStream` and `chatStreamMultimodal` take `templateVariables`, passed straight to
the chat template as `additionalContext`. The Gemma 4 template reads
`enable_thinking`, which makes the model emit its reasoning in a
`<|channel>thought … <channel|>` block before the answer:

```swift
let stream = try pipeline.chatStreamMultimodal(
    prompt: "How many shapes do you see?",
    pixelValues: pixels,
    maxTokens: 400,
    templateVariables: ["enable_thinking": true])
```

The template creates the system turn itself when there is none, so
`enable_thinking` works with or without a `systemPrompt`; ids are token-for-token
identical to the HF render in both cases. `nil` (the default) leaves the render
untouched.

**The stream is raw.** Reasoning and answer arrive as one text stream, with the
`<|channel>` / `<channel|>` delimiters intact — verified on a real model, the
streaming detokenizer does not swallow them. Filtering is the caller's job;
`Gemma4TokenFilter` does it (`.disabled` strips the thought, `.structured` returns
both parts separately) but works on token ids, so it fits a manual generation loop
rather than this stream.

Two things worth knowing before enabling it:

- Reasoning is verbose — ~350 tokens for a one-sentence answer in our test. Budget
  `maxTokens` accordingly, or the generation is cut off mid-thought and never
  reaches the answer.
- Reasoning tokens are ordinary generated tokens: with `noRepeatNGramSize` set they
  feed the ban window like any other, so a phrase used in the thought cannot be
  reused verbatim in the answer. `noRepeatNGramIncludesThinking: false` takes them
  out of the window — see [Ban window: the thinking channel](#ban-window-the-thinking-channel).

On the text path, `templateVariables` bypasses `ChatSession` exactly like
`noRepeatNGramSize` does, so `continueChat` is unavailable afterwards.

### Blocking repeated n-grams

`chatStream` and `chatStreamMultimodal` accept `noRepeatNGramSize`, the equivalent
of HF transformers' `no_repeat_ngram_size`: at each step, any token that would
complete an n-gram already present in `prompt + generated` gets a `-inf` logit.
This matters for long greedy captions (e.g. the LTX-2.5 prompt enhancer, which
uses `no_repeat_ngram_size = 5`), where decoding otherwise drifts into
repetitions.

```swift
let stream = try pipeline.chatStream(
    prompt: "user prompt: a cat on a red carpet",
    systemPrompt: enhancerSystemPrompt,
    temperature: 0.0,        // greedy
    maxTokens: 600,
    noRepeatNGramSize: 5)
```

`nil` (the default) keeps the previous behavior. When set, the text path bypasses
`ChatSession` — which cannot carry a custom `LogitProcessor` — so the turn is not
recorded in the session and `continueChat` is unavailable afterwards.

#### Ban window: prompt included or not

`noRepeatNGramIncludesPrompt` (default `true`, HF parity) selects what feeds the
ban window:

| Value | Window | Effect |
|---|---|---|
| `true` (default) | `prompt + generated` | HF parity. A prompt passage cannot be quoted verbatim once it is `n` tokens long. |
| `false` | generated only | Generation loops are still killed; the prompt stays quotable verbatim. |

Use `false` when the answer must reproduce part of the prompt exactly —
timelines, timestamps, identifiers. In HF mode, such a passage bans its own
faithful quotation (`"From 00:08.000 to 00:14.000"` repeated verbatim is a
repeated 5-gram), and greedy decoding routes around it with degraded spellings
(`"From the 0008.0"`). Killing loops is the actual purpose of the mechanism, and
that still works from the generated history alone.

```swift
let stream = try pipeline.chatStream(
    prompt: "user prompt: <caption with an explicit timeline>",
    systemPrompt: enhancerSystemPrompt,
    temperature: 0.0,
    maxTokens: 600,
    noRepeatNGramSize: 5,
    noRepeatNGramIncludesPrompt: false)   // prompt stays quotable
```

#### Ban window: the thinking channel

`noRepeatNGramIncludesThinking` (default `true`, previous behavior) selects
whether the `<|channel>thought … <channel|>` block feeds the ban window.

The two features fight each other when combined. Measured on the LTX-2.5 prompt
enhancer (E2B, same prompt and image, greedy):

| Configuration | Result |
|---|---|
| thinking off, n-gram 5 | 6 timestamps, 3 different formats, timeline contradicts the prompt |
| thinking on, n-gram off | 6 timestamps, one format, timeline consistent |
| thinking on, n-gram 5 | **zero** timestamps — vague prose instead |

The third row is the mechanism eating itself: the model reasons *with* the
timestamps, which puts them in the ban window, so it cannot restate them in the
answer and falls back to "at the start of the sequence". Set
`noRepeatNGramIncludesThinking: false` to keep reasoning and n-gram blocking
together — the thought is generated normally, just not counted as already
written, and the loop protection still covers the answer, which is what it is
for.

```swift
let stream = try pipeline.chatStreamMultimodal(
    prompt: "user prompt: <caption with an explicit timeline>",
    pixelValues: pixels,
    temperature: 0.0,
    maxTokens: 1200,
    noRepeatNGramSize: 5,
    noRepeatNGramIncludesPrompt: false,
    noRepeatNGramIncludesThinking: false,      // reasoning stays out of the window
    templateVariables: ["enable_thinking": true])
```

Detection is a three-state automaton over token ids, driven by `<|channel>`
(100), the channel name (`thought` = 45518, `response` = 6275) and `<channel|>`
(101) — a `response` channel keeps feeding the window, only `thought` is exempt.
Three details worth knowing:

- **No ban applies while inside the thought.** The history is frozen there, so
  a ban would come from a stale prefix — the last `n-1` tokens from before the
  channel opened — and would be re-applied at every step of the reasoning.
  Blocking is suspended for the duration and resumes at `<channel|>`.

- **`<|think|>` does not open the channel.** The chat template emits it at the
  top of the system turn — in the *prompt*, with no `<channel|>` facing it.
  Opening a thought state on it would drop the whole prompt out of the window.
  It is only removed as markup.
- **An unclosed channel stays open.** If `maxTokens` cuts the generation
  mid-thought, the automaton stays inside and nothing further is counted. The
  degraded mode is "no blocking", never "blocking on reasoning".

The prompt is filtered by the same automaton, so a thought block rendered from a
previous turn (`reasoning` / `reasoning_content`) is exempt too, consistently
with the generated one.

`NoRepeatNGramLogitProcessor(ngramSize:includePromptInWindow:includeThinkingInWindow:)`
exposes both switches if you build the `TokenIterator` yourself.

> No need to import `MLXLMCommon` — `Gemma4Pipeline.load()` handles registration, tokenizer loading, and model container setup internally.

### Thinking Mode Filter

Gemma 4 models may spontaneously generate thinking blocks (`<|channel>thought...`). Use `Gemma4TokenFilter` to control this:

```swift
import Gemma4Swift

// Filter thinking (default) — only response content is emitted
let filter = Gemma4TokenFilter(mode: .disabled)

// Pass-through — raw tokens including thinking blocks
let filter = Gemma4TokenFilter(mode: .enabled)

// Structured — separate thinking and response
let filter = Gemma4TokenFilter(mode: .structured)

// In your generation loop:
let output = filter.process(tokenId: tokenId, text: decodedText)
if !output.isEmpty { print(output, terminator: "") }

// After generation (structured mode):
let result = filter.structuredResponse()
print("Thinking: \(result.thinking ?? "none")")
print("Response: \(result.response)")
```

## Architecture

```
Gemma4Swift/
├── Configuration/       # Model configs (text, vision, audio)
├── TextModel/           # Decoder layers, attention, MLP, MoE, per-layer inputs
├── RoPE/                # ProportionalRoPE with partial rotation
├── VisionEncoder/       # SigLIP: patch embed, 2D RoPE, pooler, head_dim padding
├── AudioEncoder/        # Conformer: SubSampleConv, chunked attention, rel positions
├── VideoProcessor/      # AVAsset frame extraction, ~1fps, aspect-ratio resize
├── Multimodal/          # Embedding fusion via masked_scatter
├── LoRA/                # LoRA/DoRA/Full SFT training, adapter load/fuse/unload
├── TurboQuant/          # MSE codec, Metal kernels, chunked prefill attention
├── Pipeline/            # High-level API, processors, token filter, registration
├── Norms/               # RMSNormNoScale, RMSNormZeroShift
└── Utils/               # Weight sanitizer, profiling toolkit
```

### KV Cache Quantization

`kvBits` uses **mlx-swift-lm's native `QuantizedKVCache`** for every family (the KV-shared layers of
E2B/E4B read their source layers correctly). The project also includes a custom TurboQuant implementation
(rotation + Beta-optimal codebook) in `TurboQuant/`, reachable only explicitly through
`languageModel.makeCache(kvBits:)`:

```swift
// Native KV cache quantization — no custom code needed
let params = GenerateParameters(kvBits: 4, kvGroupSize: 64, quantizedKVStart: 5000)
```

**Why migrate:** TurboQuant's theoretical 3.85x compression is real, but runtime intermediate tensor materialization erases memory gains. The disabled Fast Hadamard Transform (O(D^2) dense rotation instead of O(D log D)) adds significant compute overhead. Meanwhile, mlx-swift-lm ships battle-tested 4/8-bit KV quantization with Metal-accelerated `quantizedMM()`, automatic attention routing, and zero maintenance.

The `TurboQuant/` directory is retained for reference but is not used by the default inference pipeline.

### Key design decisions

- **Gemma 4 ≠ Gemma 3n**: No AltUp, no Laurel blocks, no activation sparsity. Simpler decoder with `global_head_dim`, `partial_rotary_factor`, `use_double_wide_mlp`, and optional K=V attention.
- **Private model factory**: `Gemma4Registration.loadContainer(from:)` loads through its own `LLMModelFactory` that only knows the Gemma 4 types, so neither upstream's own Gemma 4 nor `MLXVLM` (VLM-first trampoline) can answer instead. `register()` remains for code that loads through `LLMModelFactory.shared`.
- **Multimodal via masked_scatter**: Image/audio/video embeddings replace special token positions in the text embedding sequence.
- **Video aligned with Google reference**: `video_token` (258884), 70 soft tokens/frame, MM:SS timestamps, ~1fps sampling.
- **Audio aligned with Google reference**: Conformer with relative position embeddings, causal chunked attention, mel spectrogram matching `feature_extraction_gemma4.py`.
- **Thinking mode handling**: `Gemma4TokenFilter` detects `<|channel>thought`/`<|channel>response` blocks and filters or separates them at the API level.
- **ProportionalRoPE**: Only 25% of head dimensions get rotary encoding for full-attention layers.
- **Vision head_dim padding**: Pads non-standard head dimensions (72 → 80) for fused SDPA to avoid NaN from all-masked padding rows.
- **No HuggingFace SDK dependency**: Direct HTTPS downloads + local tokenizer loading.

## Acknowledgments

- [Google Gemma 4](https://ai.google.dev/gemma) — Original model architecture
- [mlx-swift](https://github.com/ml-explore/mlx-swift) — Apple MLX framework for Swift
- [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) — LLM infrastructure
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) — Python reference implementation
- [swift-transformers](https://github.com/huggingface/swift-transformers) — Tokenizer support

## License

MIT License — See [LICENSE](LICENSE) file.
