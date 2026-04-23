# Bird Call Identification — Multimodal LoRA Example

Identifies bird species from 5-second audio recordings using a LoRA-adapted Gemma 4 E2B model.

## Results

- **20 species**, 8636 training samples, 959 validation samples
- **50% accuracy** on validation set (species-level, greedy decoding)
- Training loss: 0.036, validation loss: 0.038
- Model generates valid JSON with common name, scientific name, and call description

## Reproduce

### 1. Prepare the dataset

```bash
pip install pyarrow huggingface_hub
python examples/birdcall-adapter/prepare_dataset.py
```

This downloads audio from [tglcourse/5s_birdcall_samples_top20](https://huggingface.co/datasets/tglcourse/5s_birdcall_samples_top20) and creates JSONL files at `/tmp/birdcall-full-rich/`.

### 2. Train

```bash
gemma4-cli lora train \
  --model-path ~/Library/Caches/models/mlx-community/gemma-4-e2b-it-bf16 \
  --data /tmp/birdcall-full-rich \
  --output ./birdcall-adapter \
  --multimodal \
  --mask-prompt \
  --num-layers 16 \
  --rank 16 \
  --learning-rate 5e-5 \
  --iterations 8636
```

Training takes ~1.5 hours on M-series Mac with 32GB+ RAM. GPU peak: ~23 GB.

### 3. Benchmark

```bash
gemma4-cli lora bench-multimodal \
  --model-path ~/Library/Caches/models/mlx-community/gemma-4-e2b-it-bf16 \
  --adapter-path ./birdcall-adapter \
  --data /tmp/birdcall-full-rich \
  --max-tokens 100
```

### Sample output

Input: 5-second audio of a Common Raven call

```json
{
  "common_name": "Common Raven",
  "scientific_name": "Corvus corax",
  "call_type": "call"
}
```

## Dataset format

Each line in `train.jsonl` / `valid.jsonl`:

```json
{
  "messages": [
    {"role": "user", "content": "identify"},
    {"role": "assistant", "content": "{\"common_name\": \"Mallard\", \"scientific_name\": \"Anas platyrhynchos\", \"call_type\": \"song\"}"}
  ],
  "audio": "audio/mallar3_00229.wav"
}
```

## Species list (20)

American Robin, Barn Swallow, Bewick's Wren, Black-capped Flycatcher, Carolina Wren, Common Raven, Cuban Thrush, European Starling, Great Bowerbird Wren, House Sparrow, House Wren, Mallard, Northern Cardinal, Red Crossbill, Red-winged Blackbird, Ruby Pepper Shrike, Rufous-crowned Sparrow, Song Sparrow, Spotted Towhee, Swainson's Thrush
