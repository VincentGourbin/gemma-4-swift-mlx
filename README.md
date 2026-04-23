# Gemma 4 Swift MLX

Native Gemma 4 multimodal inference for Apple Silicon via [MLX Swift](https://github.com/ml-explore/mlx-swift).

## Status

| Feature | Status | Details |
|---------|--------|---------|
| Text generation | ✅ **Working** | All 4 families, 4 quantizations each. 8-103 tok/s |
| Vision (image understanding) | ✅ **Working** | Single + multi-image. All 4 families validated |
| Video (frame-by-frame) | ✅ **Working** | ~1fps, 70 tokens/frame, MM:SS timestamps. All 4 families validated |
| Audio (speech understanding) | ✅ **Working** | Conformer encoder, 30s max, ASR/comprehension. E2B + E4B validated |
| Thinking mode filter | ✅ **Working** | Filters `<\|channel>thought` blocks. Structured response separation |
| LoRA/DoRA fine-tuning | ✅ **Working** | LoRA, DoRA, full SFT. Response masking, chat template. 97% accuracy on classifier benchmark |
| Multimodal LoRA | ✅ **Working** | Audio + vision fine-tuning. 50% accuracy on 20-species bird call classification, LaTeX OCR verified |
| KV cache quantization | 🔄 **Migrating** | TurboQuant (custom) to be replaced by mlx-swift-lm native `QuantizedKVCache`. See [optimization report](#kv-cache-quantization) |
| Multi-turn chat | ✅ **Working** | Via ChatSession streaming |
| Profiling toolkit | ✅ **Working** | Chrome Trace export, SQLite benchmarks, context sweep |
| Model download | ✅ **Working** | Direct HTTPS from HuggingFace (no HF SDK dependency) |

## Requirements

- macOS 14+ (Sonoma)
- Apple Silicon (M1/M2/M3/M4)
- Swift 6.0+
- Xcode 16+

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
  --mask-prompt --num-layers 16 --iterations 1300 --learning-rate 1e-4

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

## Supported Models

### Model Families

| Family | Total Params | Active Params | MoE | Audio | Key Features |
|--------|:---:|:---:|:---:|:---:|---|
| **E2B** | 5.1B | 2.3B | No | Yes | Fastest. Text + Vision + Audio + Video |
| **E4B** | 9.6B | 4.5B | No | Yes | Best quality/size ratio. Full multimodal |
| **31B** | 31.3B | 31.3B | No | No | Highest quality. Text + Vision + Video |
| **26B-A4B** | 25.8B | 3.8B | Yes (128 experts, top-8) | No | MoE efficiency. Text + Vision + Video |

### Available Quantizations

| Model | 4-bit | 6-bit | 8-bit | BF16 | HuggingFace ID pattern |
|-------|:---:|:---:|:---:|:---:|---|
| **E2B** | ~3.6 GB | ~4.2 GB | ~5.2 GB | ~10 GB | `mlx-community/gemma-4-e2b-it-{quant}` |
| **E4B** | ~5 GB | ~6.5 GB | ~8 GB | ~19 GB | `mlx-community/gemma-4-e4b-it-{quant}` |
| **31B** | ~17 GB | ~25 GB | ~33 GB | ~63 GB | `mlx-community/gemma-4-31b-it-{quant}` |
| **26B-A4B** | ~14 GB | ~21 GB | ~27 GB | ~52 GB | `mlx-community/gemma-4-26b-a4b-it-{quant}` |

> Additional formats available: `mxfp4`, `mxfp8`, `nvfp4`, `5-bit`. See [mlx-community on HuggingFace](https://huggingface.co/mlx-community?search=gemma-4).

## Performance (Apple M3 Max, 96 GB)

All 16 model variants (4 families × 4 quantizations) benchmarked. Full results in [benchmarks/](benchmarks/results/).

### Text Generation (tok/s)

| Model | 4-bit | 6-bit | 8-bit | BF16 |
|-------|:-----:|:-----:|:-----:|:----:|
| **E2B** | **97** | 62 | 72 | 42 |
| **E4B** | **61** | 48 | 42 | 25 |
| **26B-A4B** | **56** | 43 | 41 | 10 |
| **31B** | **12** | 9 | 7 | 3 |

### Vision Quality (vehicle identification across quantizations)

| Model | 4-bit | 6-bit | 8-bit | BF16 |
|-------|:-----:|:-----:|:-----:|:----:|
| **E2B** | "classic car" | "VW Beetle" | "Fiat 600" | "VW Beetle" |
| **E4B** | "classic Fiat" | "VW Beetle" | "Citroën 2CV" | "VW Beetle" |
| **26B-A4B** | **"Citroën 2CV"** | **"Citroën 2CV"** | **"Citroën 2CV"** | **"Citroën 2CV"** |
| **31B** | **"Citroën 2CV"** | **"Citroën 2CV"** | **"Citroën 2CV"** | **"Citroën 2CV"** |

> **Key finding:** Quality depends on architecture, not quantization. 26B-A4B/31B identify the vehicle correctly at all quantizations. 4-bit is the sweet spot — fastest with no quality loss. BF16 is 2-24x slower with no gain. See [docs/examples/](docs/examples/) and [benchmarks/](benchmarks/results/) for full results.

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

### Supported modes

| Mode | Description | Memory (E2B bf16) |
|------|-------------|:---:|
| **LoRA** | Low-rank adaptation (default) | ~11 GB |
| **DoRA** | Weight-Decomposed LoRA | ~11 GB |
| **Full SFT** | All weights trainable | ~20 GB |

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

### Recommended hyperparameters

| Model | Layers | LR | Iterations | Notes |
|-------|:------:|:--:|:----------:|-------|
| E2B (bf16) | 16 | 1e-4 | 1 epoch | Best for fine-tuning |
| E4B (bf16) | 12 | 1e-4 | 1 epoch | Higher quality base |
| E2B (4-bit) | 8 | 1e-5 | 1 epoch | Works but noisier gradients |

- Always use `--mask-prompt` for chat-format data
- Use bf16 models for training (not quantized)
- Adapters trained in Swift work in Python mlx-lm and vice versa

### Library API

```swift
import Gemma4Swift

// Load model + adapter for inference
let container = try await loadLocalModel(path: modelPath)
try await Gemma4LoRAInference.loadAdapter(into: container, from: adapterURL)

// Or fuse permanently
try await Gemma4LoRAInference.fuseAdapter(into: container, from: adapterURL)

// Training
let config = Gemma4LoRATrain.TrainingConfig(
    loraRank: 8,
    loraScale: 20.0,
    numLayers: 16,
    learningRate: 1e-4,
    iterations: 1300,
    maskPrompt: true,
    outputDirectory: outputURL
)
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
- bf16 model required (not quantized) — model is converted to float32 internally for training stability

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

The project includes a custom TurboQuant implementation (rotation + Beta-optimal codebook) in `TurboQuant/`. After thorough audit, we recommend migrating to **mlx-swift-lm's native `QuantizedKVCache`** instead:

```swift
// Native KV cache quantization — no custom code needed
let params = GenerateParameters(kvBits: 4, kvGroupSize: 64, quantizedKVStart: 5000)
```

**Why migrate:** TurboQuant's theoretical 3.85x compression is real, but runtime intermediate tensor materialization erases memory gains. The disabled Fast Hadamard Transform (O(D^2) dense rotation instead of O(D log D)) adds significant compute overhead. Meanwhile, mlx-swift-lm ships battle-tested 4/8-bit KV quantization with Metal-accelerated `quantizedMM()`, automatic attention routing, and zero maintenance.

The `TurboQuant/` directory is retained for reference but is not used in the default inference pipeline.

### Key design decisions

- **Gemma 4 ≠ Gemma 3n**: No AltUp, no Laurel blocks, no activation sparsity. Simpler decoder with `global_head_dim`, `partial_rotary_factor`, `use_double_wide_mlp`, and optional K=V attention.
- **Registration-based**: Registers `"gemma4"` and `"gemma4_text"` model types into mlx-swift-lm's `LLMTypeRegistry`.
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
