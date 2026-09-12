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

# Run tests — always through this wrapper, never bare `xcodebuild test`
Scripts/run-tests.sh
Scripts/run-tests.sh -only-testing:Gemma4SwiftTests/WeightSanitizerTests
```

### Why the test wrapper

A bare `xcodebuild ... test` **hangs forever** (0% CPU after ~250 tests). It is a
lock-ordering deadlock in mlx-swift, not a slow test: `CompiledFunction.call`
takes the per-function `NSLock` then the global `evalLock`
(`Transforms+Compile.swift:39` and `:89`), while `vjp`/`jvp` — every
`value_and_grad` — take the global `evalLock` first and then re-enter compiled
functions during tracing (`Transforms.swift:31`, `:68`). One thread in a
gradient (`DrafterTrainingTests`, `LoRATests`) plus one thread in a forward
going through `geluApproximate` (a `compile`d function) is enough. `evalLock` is
recursive, so single-threaded use never deadlocks — it takes two threads.

**This is not a test-only hazard.** Both locks are process-global, and the
library exposes both sides: `Gemma4LoRATrain.train` is a nonisolated public
static (runnable from any task) while `Gemma4Pipeline` is `@MainActor`. An app
that fine-tunes on a background task while streaming inference can hit the same
ABBA and wedge. Until upstream is fixed, do not run gradients concurrently with
inference — serialize the two.

`Scripts/run-tests.sh` sets `SWT_EXPERIMENTAL_MAXIMUM_PARALLELIZATION_WIDTH=1`
(via the `TEST_RUNNER_` prefix, the only env vars xcodebuild forwards to the test
process). The whole suite then passes in ~1.2s. Drop the wrapper once the
deadlock is fixed upstream.

Integration tests that need a local model are gated on an env var, forwarded by
the wrapper:

```bash
GEMMA4_INTEGRATION_MODEL_PATH=~/Library/Caches/models/mlx-community/gemma-4-e4b-it-4bit \
  Scripts/run-tests.sh -only-testing:Gemma4SwiftTests/NoRepeatNGramIntegrationTests
```

## Dependency pinning

`mlx-swift` and `mlx-swift-lm` are bounded with `.upToNextMinor` — both have broken
APIs on minor bumps. Never move them to a `branch:` requirement: SwiftPM refuses a
branch dependency transitively under a semver-tagged package, which makes this repo
unconsumable by any downstream project that depends on a tag.

Since mlx-swift-lm 3.x, upstream ships its **own** Gemma 4 (`"gemma4"`, `"gemma4_text"`,
`"gemma4_unified"` in both `LLMTypeRegistry` and `VLMTypeRegistry`).
`Gemma4Registration.register()` deliberately overwrites the `LLMTypeRegistry` entries —
`ModelTypeRegistry.registerModelType` is last-write-wins — so it must be called before
any load, otherwise upstream's text-only implementation silently answers instead.

**It does not, and cannot, overwrite the `VLMTypeRegistry` entries**: this package does
not link `MLXVLM`. That matters because the free function
`MLXLMCommon.loadModelContainer(...)` goes through `ModelFactoryRegistry.shared`, whose
trampoline order is fixed **VLM first, then LLM**, keeping the first factory that
succeeds (`ModelFactory.swift`, `load(loader:)`). So in any process that also links
`MLXVLM` — directly or through another package — the free function returns
`MLXVLM.Gemma4`, not ours, whatever `multimodal:` was asked for. The failure surfaces
downstream as `unsupportedModelFamily` from `chatStreamMultimodal`'s
`as? Gemma4MultimodalLLMModel`, and it is load-order dependent (it only bites once
`MLXVLM`'s trampoline class is realized), so it can pass once and fail on the next call.

**Always load via `Gemma4Registration.loadContainer(from:using:multimodal:)`**, which
registers and then calls `LLMModelFactory.shared.loadContainer` directly, bypassing
`ModelFactoryRegistry`. Never call the free `loadModelContainer` for a Gemma 4 model.

If `swift package resolve` fails with `bad object refs/remotes/origin/<branch>`, a cached
SwiftPM checkout holds a ref to an upstream branch that was deleted. Drop the stale line
from `.build/*/checkouts/mlx-swift-lm/.git/packed-refs` (or delete the checkout) and
re-resolve.

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

`Gemma4Registration.register()` registers `"gemma4"` and `"gemma4_text"` model types with mlx-swift-lm's `LLMTypeRegistry`. Text-only vs multimodal is controlled by `register(multimodal:)`.

`Gemma4Registration.loadContainer(from:using:multimodal:)` is the entry point to use: it registers, then loads through `LLMModelFactory.shared` so `ModelFactoryRegistry`'s VLM-first ordering can't hand back upstream's `MLXVLM.Gemma4` (see "Dependency pinning"). The resulting `ModelContainer` works with `ChatSession` as usual.

### Weight Loading

`WeightSanitizer` remaps PyTorch checkpoint keys to the Swift module hierarchy:
- Strips `"model."` prefix, remaps `"language_model.X"` → `"language_model.model.X"`
- Skips rotary_emb and unused clipping params
- Splits MoE `gate_up_proj` into separate gate/up projections

### Speculative Decoding (MTP)

Supports `gemma-4-{E2B,E4B}-it-assistant` drafter models for multi-token prediction. The drafter is a 4-layer mini-transformer (hidden 256) where **all layers are kv-shared-only** — they consume the target's K/V cache via `bind(target:)` + `setSharedKV(...)` instead of computing their own.

Key files:
- `Speculative/Gemma4AssistantDraftModel.swift` — drafter with `draftBlock` (autoregressive K-step) and `trainForward` (parallel multi-position, used during fine-tuning)
- `Speculative/MaskedEmbedder.swift` — sparse softmax LM head: scores 2048 token clusters, picks top-K=32, computes dense logits only on the ~4096 tokens of those clusters
- `Speculative/SpeculativeWalk.swift` — pure-logic greedy walk (accept until first divergence)
- `Speculative/Gemma4DrafterTraining.swift` — self-distillation training (target frozen, drafter trainable, CE loss against `argmax(target_logits)`)
- `Pipeline/Gemma4MTPPipeline.swift` — actor with `mtpStream(...)` end-to-end (prefill → draft → verify → walk → rollback)

Quirks:
- The drafter checkpoint OMITS `k_proj`, `v_proj`, `k_norm` weights — they're optional in `Gemma4Attention` gated by `kvSharedOnly: Bool` (default `false`, back-compat).
- `Gemma4TextModel.forwardCollectingIntermediates` exposes the LAST decoder layer's output **BEFORE** the final RMSNorm via `preNormHidden` — the drafter's `pre_projection` was trained against pre-norm.
- For kv-shared inference path with cache, queries get RoPE at `cache.offset - L` (pre-write offset), not `cache.offset` (post-write).
- During training, `MaskedEmbedder` is bypassed via `useFullLMHead = true` because `putAlong` (scatter) has no VJP in MLX.

## Key Conventions

- All neural network types subclass `Module` from MLXNN with `@ModuleInfo` property wrappers for parameter tracking
- `@unchecked Sendable` is used where needed for MLXArray (Swift 6 strict concurrency)
- RoPE selection uses factory pattern: `RoPEFactory.create()` picks standard vs proportional based on layer type
- Special token IDs are constants in `Gemma4Processor` (e.g., `imageTokenId = 258880`)

## Supported Models

- `mlx-community/gemma-4-e2b-it-4bit` (~3.6 GB, 2.3B effective params)
- `mlx-community/gemma-4-e4b-it-4bit` (~5 GB, 4B effective params)
