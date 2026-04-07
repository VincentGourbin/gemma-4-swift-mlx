# Gemma 4 Swift MLX

Native Gemma 4 multimodal inference for Apple Silicon via [MLX Swift](https://github.com/ml-explore/mlx-swift).

## Status

| Feature | Status | Details |
|---------|--------|---------|
| Text generation | ✅ **Working** | All 4 families, 4 quantizations each. 8-103 tok/s |
| Vision (image understanding) | ✅ **Working** | Single + multi-image. All 4 families validated |
| Video (frame-by-frame) | 🚧 In progress | Architecture complete, frame extraction works |
| Audio (speech understanding) | 🚧 In progress | Conformer encoder implemented, integration pending |
| TurboQuant KV cache | 🧪 Experimental | Chunked prefill + fused Metal kernel. Modest GPU savings at <32K context. [Details](docs/examples/turboquant_paper.txt) |
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
| **31B** | 31.3B | 31.3B | No | No | Highest quality. Text + Vision. K=V attention |
| **26B-A4B** | 25.8B | 3.8B | Yes (128 experts, top-8) | No | MoE efficiency. Text + Vision. K=V attention |

### Available Quantizations

| Model | 4-bit | 6-bit | 8-bit | BF16 | HuggingFace ID pattern |
|-------|:---:|:---:|:---:|:---:|---|
| **E2B** | ~3.6 GB | ~4.2 GB | ~5.2 GB | ~10 GB | `mlx-community/gemma-4-e2b-it-{quant}` |
| **E4B** | ~5 GB | ~6.5 GB | ~8 GB | ~19 GB | `mlx-community/gemma-4-e4b-it-{quant}` |
| **31B** | ~17 GB | ~25 GB | ~33 GB | ~63 GB | `mlx-community/gemma-4-31b-it-{quant}` |
| **26B-A4B** | ~14 GB | ~21 GB | ~27 GB | ~52 GB | `mlx-community/gemma-4-26b-a4b-it-{quant}` |

> Additional formats available: `mxfp4`, `mxfp8`, `nvfp4`, `5-bit`. See [mlx-community on HuggingFace](https://huggingface.co/mlx-community?search=gemma-4).

## Performance (Apple M3 Max, 96 GB)

### Text Generation (4-bit models, 500 tokens)

| Model | Throughput | TTFT (16K context) | GPU Peak |
|-------|:---------:|:------------------:|:--------:|
| **E2B** | 103 tok/s | 85 ms | 2.5 GB |
| **E4B** | 47 tok/s | 250 ms | 4.3 GB |
| **26B-A4B** | 29 tok/s | 1.4 s | 13.8 GB |
| **31B** | 11 tok/s | 3.5 s | 16.8 GB |

### Vision (4-bit models, image description)

| Model | Speed | GPU Peak | Vehicle ID Accuracy |
|-------|:-----:|:--------:|:---:|
| **E2B** | 74 tok/s | 4.4 GB | Generic ("classic car") |
| **E4B** | 45 tok/s | 5.9 GB | Approximate ("FIAT 600") |
| **26B-A4B** | 25 tok/s | 15.9 GB | **Exact ("Citroën 2CV")** |
| **31B** | 8 tok/s | 19.5 GB | **Exact ("Citroën 2CV")** |

> See [docs/examples/vision-image-description/](docs/examples/vision-image-description/) for full benchmarks with OCR and multi-image tests.

## Library Integration

```swift
import Gemma4Swift
import MLXLMCommon

// Register Gemma 4 model type
await Gemma4Registration.register()

// Load model from local path
let container = try await loadModelContainer(
    from: URL(fileURLWithPath: modelPath),
    using: LocalTokenizerLoader()
)

// Chat
let session = ChatSession(container, instructions: "You are helpful.")
let response = try await session.respond(to: "Hello!")

// Streaming
let stream = session.streamResponse(to: "Write a poem")
for try await token in stream {
    print(token, terminator: "")
}
```

## Architecture

```
Gemma4Swift/
├── Configuration/       # Model configs (text, vision, audio)
├── TextModel/           # Decoder layers, attention, MLP, MoE, per-layer inputs
├── RoPE/                # ProportionalRoPE with partial rotation
├── VisionEncoder/       # SigLIP: patch embed, 2D RoPE, pooler, head_dim padding
├── AudioEncoder/        # Conformer: SubSampleConv, chunked attention
├── VideoProcessor/      # AVAsset frame extraction
├── Multimodal/          # Embedding fusion via masked_scatter
├── TurboQuant/          # MSE codec, Metal kernels, chunked prefill attention
├── Pipeline/            # High-level API, processors, registration, model cache
├── Norms/               # RMSNormNoScale, RMSNormZeroShift
└── Utils/               # Weight sanitizer, profiling toolkit
```

### Key design decisions

- **Gemma 4 ≠ Gemma 3n**: No AltUp, no Laurel blocks, no activation sparsity. Simpler decoder with `global_head_dim`, `partial_rotary_factor`, `use_double_wide_mlp`, and optional K=V attention.
- **Registration-based**: Registers `"gemma4"` and `"gemma4_text"` model types into mlx-swift-lm's `LLMTypeRegistry`.
- **Multimodal via masked_scatter**: Image/audio/video embeddings replace special token positions in the text embedding sequence.
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
