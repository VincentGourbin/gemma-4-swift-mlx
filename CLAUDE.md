# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

**Must use `xcodebuild`** (not `swift build`) — Metal shader support required by MLX:

```bash
# Build CLI (Release)
xcodebuild -scheme gemma4-cli -configuration Release \
  -destination "platform=macOS" -derivedDataPath .build/xcode \
  -skipMacroValidation build

# Binary location
.build/xcode/Build/Products/Release/gemma4-cli

# Run tests
xcodebuild -scheme Gemma4Swift -destination "platform=macOS" \
  -derivedDataPath .build/xcode -skipMacroValidation test
```

## Architecture

Swift 6.0 / macOS 14+ / Apple Silicon only. Two products: `Gemma4Swift` library and `gemma4-cli` executable.

### Multimodal Pipeline

The model fuses text, vision, and audio through **masked_scatter** — special tokens in the text embedding sequence are replaced with projected modality embeddings:

1. **VisionEncoder** (SigLIP): image → patches → 2D RoPE transformer → pooler → 280 soft tokens per image
2. **AudioEncoder** (Conformer): PCM → mel-spectrogram → SubSampleConv → ConformerBlocks → variable-length tokens
3. **MultimodalEmbedder**: projects modality tokens into text embedding space
4. **Gemma4Model**: calls `maskedScatter()` to splice modality embeddings at `[boi]`/`[boa]` token positions

### Text Model Internals

The decoder has two layer types with different configurations:
- **Full attention layers**: use `global_head_dim` (512), ProportionalRoPE (25% partial rotation)
- **Sliding window layers**: use `head_dim` (256), standard RoPE
- **KV sharing**: layers 15+ reuse KV cache from earlier layers
- **Per-layer input gating**: each layer receives additional embeddings via `per_layer_input_gate` (for E2B/E4B models with `hidden_size_per_layer_input`)
- **Double-wide MLP**: KV-shared layers use 2x intermediate size

### Registration System

`Gemma4Registration.register()` registers `"gemma4"` and `"gemma4_text"` model types with mlx-swift-lm's `LLMTypeRegistry`. This enables loading via `loadModelContainer()` and usage with `ChatSession`/`ModelContainer`. Text-only vs multimodal is controlled by `register(multimodal:)`.

### Weight Loading

`WeightSanitizer` remaps PyTorch checkpoint keys to the Swift module hierarchy:
- Strips `"model."` prefix, remaps `"language_model.X"` → `"language_model.model.X"`
- Skips rotary_emb and unused clipping params
- Splits MoE `gate_up_proj` into separate gate/up projections

## Key Conventions

- All neural network types subclass `Module` from MLXNN with `@ModuleInfo` property wrappers for parameter tracking
- `@unchecked Sendable` is used where needed for MLXArray (Swift 6 strict concurrency)
- RoPE selection uses factory pattern: `RoPEFactory.create()` picks standard vs proportional based on layer type
- Special token IDs are constants in `Gemma4Processor` (e.g., `imageTokenId = 258880`)

## Supported Models

- `mlx-community/gemma-4-e2b-it-4bit` (~3.6 GB, 2.3B effective params)
- `mlx-community/gemma-4-e4b-it-4bit` (~5 GB, 4B effective params)
