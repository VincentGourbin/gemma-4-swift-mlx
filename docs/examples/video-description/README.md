# Video: Frame-by-Frame Description

Video understanding across all 4 Gemma 4 model families (4-bit quantization). The video pipeline is aligned with the reference Python implementation (HuggingFace transformers):

- **~1 fps** uniform sampling (max 32 frames, max 60s)
- **70 soft tokens per frame** (vs 280 for images) via `video_token` (ID 258884)
- **MM:SS timestamps** injected before each frame in the prompt
- Aspect-ratio preserving resize per frame
- Each frame processed through the same SigLIP vision encoder as images

## Hardware

| Component | Specification |
|-----------|--------------|
| **Machine** | Mac Studio |
| **Chip** | Apple M3 Max |
| **Memory** | 96 GB Unified Memory |
| **OS** | macOS 26.4 |
| **MLX** | mlx-swift 0.30.6+ |

## Test Video

**step_3797B1A8.mp4** — FluxForge Studio logo animation sequence

| Property | Value |
|----------|-------|
| Resolution | 1024x1024 |
| Codec | HEVC (H.265) |
| FPS | 24 |
| Duration | ~10s |
| Size | 8.8 MB |

A branding/logo reveal video showing "The FluxForge Studio" text with progressive appearance of black geometric icons and technical shapes building up around the logo.

---

## Results

**Prompt:** `Describe this video in detail. What is happening frame by frame?`

**Command:**
```bash
gemma4-cli describe --model-path ~/Library/Caches/models/mlx-community/gemma-4-{model}-it-4bit \
  --video step_3797B1A8.mp4 \
  --prompt "Describe this video in detail. What is happening frame by frame?" \
  --max-tokens 500 --temperature 0.3
```

### Benchmarks

| Model | Params | Frames | Tokens/frame | Input tokens | Output tokens | Speed | GPU peak |
|-------|--------|--------|-------------|-------------|---------------|-------|----------|
| **E2B-4bit** | 2.3B eff. | 9 | 70 | 734 | 392 | **50.8 t/s** | 5.6 GB |
| **E4B-4bit** | 4.5B eff. | 9 | 70 | 734 | 500 | **36.5 t/s** | 7.0 GB |
| **26B-A4B-4bit** | 3.8B eff. (MoE) | 9 | 70 | 738 | 500 | **18.8 t/s** | 17.0 GB |
| **31B-4bit** | 31.3B | 9 | 70 | 738 | 263 (natural stop) | **5.8 t/s** | 20.8 GB |

### Quality Comparison

| Model | Timestamps used | Temporal reasoning | Detail level | Natural stop |
|-------|:-:|:-:|:-:|:-:|
| **E2B** | Yes | Basic (per-frame) | Good | No (hit limit) |
| **E4B** | Yes | Good (time ranges) | Very good | No (hit limit) |
| **26B-A4B** | Yes | Excellent (motion/depth) | Excellent | No (hit limit) |
| **31B** | Yes | Excellent (motion/depth) | Best (concise) | Yes (263 tokens) |

### Key Observations

1. **All models correctly use timestamps** in their responses, thanks to the MM:SS annotations in the prompt
2. **26B-A4B and 31B** detect motion and depth effects (zoom, tunnel, camera movement) that smaller models miss
3. **31B stops naturally at 263 tokens** — a sign of better comprehension (says what's needed, then stops)
4. **Token efficiency**: 70 tokens/frame means 630 total video tokens vs 2240 with the old 280/frame approach (3.5x reduction), with no quality loss
5. **GPU memory** is significantly reduced compared to treating frames as full-resolution images

### Individual Outputs

- [E2B output](e2b.txt)
- [E4B output](e4b.txt)
- [26B-A4B output](26b-a4b.txt)
- [31B output](31b.txt)
