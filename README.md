# Gemma 4 Swift MLX

Native Gemma 4 multimodal inference for Apple Silicon via [MLX Swift](https://github.com/ml-explore/mlx-swift).

> **Work in Progress** — This project is under active development. Text and vision inference are functional; audio and video pipelines are architecturally complete and being refined.

## Features

| Feature | Status |
|---------|--------|
| Text generation (E2B/E4B 4-bit) | ✅ Working (45-72 tok/s) |
| Vision (image understanding) | ✅ Working |
| Audio (speech understanding) | 🚧 Architecture complete, upstream mlx-vlm parity |
| Video (frame-by-frame) | 🚧 Architecture complete |
| TurboQuant KV cache | 🚧 Experimental |
| Multi-turn chat | ✅ Working |
| Streaming | ✅ Working |
| HuggingFace Hub download | ✅ Working |

## Requirements

- macOS 14+ (Sonoma)
- Apple Silicon (M1/M2/M3/M4)
- Swift 6.0+
- Xcode 16+

## Installation

### Swift Package Manager

```swift
dependencies: [
    .package(url: "https://github.com/VincentGourbin/gemma-4-swift-mlx", branch: "main"),
],
targets: [
    .target(
        name: "YourTarget",
        dependencies: [
            .product(name: "Gemma4Swift", package: "gemma-4-swift-mlx"),
        ]
    ),
]
```

### Build CLI from source

```bash
git clone https://github.com/VincentGourbin/gemma-4-swift-mlx
cd gemma-4-swift-mlx
xcodebuild -scheme gemma4-cli -configuration Release \
  -destination "platform=macOS" -derivedDataPath .build/xcode \
  -skipMacroValidation build
```

The binary will be at `.build/xcode/Build/Products/Release/gemma4-cli`.

> **Note:** Use `xcodebuild` (not `swift build`) to compile Metal shader support required by MLX.

## Usage

### CLI

```bash
# Text generation
gemma4-cli generate "Explain machine learning in 3 sentences" --max-tokens 200

# Image description
gemma4-cli describe --image photo.png --prompt "Describe this image"

# Audio understanding
gemma4-cli describe --audio speech.mp3 --prompt "What is being said?"

# Video understanding
gemma4-cli describe --video clip.mp4 --prompt "What happens in this video?"

# Interactive chat
gemma4-cli chat --model mlx-community/gemma-4-e2b-it-4bit
```

### Library integration

```swift
import Gemma4Swift
import MLXLMCommon

// Register Gemma 4 model type
await Gemma4Registration.register()

// Load model (downloads from HuggingFace if needed)
let container = try await loadModelContainer(
    from: downloader,
    using: tokenizerLoader,
    id: "mlx-community/gemma-4-e2b-it-4bit"
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

## Architecture

```
Gemma4Swift/
├── Configuration/       # Model configs (text, vision, audio)
├── TextModel/           # Decoder layers, attention, MLP, per-layer inputs
├── RoPE/                # ProportionalRoPE with partial rotation
├── VisionEncoder/       # SigLIP: patch embed, 2D RoPE, pooler
├── AudioEncoder/        # Conformer: SubSampleConv, chunked attention
├── VideoProcessor/      # AVAsset frame extraction
├── Multimodal/          # Embedding fusion via masked_scatter
├── TurboQuant/          # MSE codec KV cache compression
├── Pipeline/            # High-level API, processors, registration
├── Norms/               # RMSNormNoScale, RMSNormZeroShift
└── Utils/               # Weight sanitizer, memory management
```

### Key design decisions

- **Gemma 4 ≠ Gemma 3n**: No AltUp, no Laurel blocks, no activation sparsity. Simpler decoder with `global_head_dim`, `partial_rotary_factor`, `use_double_wide_mlp`, and optional K=V attention.
- **Registration-based**: Registers `"gemma4"` and `"gemma4_text"` model types into mlx-swift-lm's `LLMTypeRegistry`, enabling seamless use with `ChatSession` and `ModelContainer`.
- **Multimodal via masked_scatter**: Image/audio/video embeddings replace special token positions in the text embedding sequence.
- **ProportionalRoPE**: Only 25% of head dimensions get rotary encoding for full-attention layers.

## Performance

Tested on Apple Silicon with `mlx-community/gemma-4-e2b-it-4bit`:

| Metric | Value |
|--------|-------|
| Model load time | ~1.5s (from cache) |
| GPU memory (text) | ~2.5 GB |
| GPU memory (multimodal) | ~3.4 GB |
| Text generation speed | 45-72 tok/s |
| Vision inference | ~4 tok/s (incl. encoder) |

## Acknowledgments

- [Google Gemma 4](https://ai.google.dev/gemma) — Original model architecture
- [mlx-swift](https://github.com/ml-explore/mlx-swift) — Apple MLX framework for Swift
- [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) — LLM infrastructure
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) — Python reference implementation
- [swift-transformers](https://github.com/huggingface/swift-transformers) — Tokenizer support

## License

MIT License — See [LICENSE](LICENSE) file.
