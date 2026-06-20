# OCR Bench — DiffusionGemma vs Gemma 4 AR (mini OCRBench)

Comparaison de la qualité OCR de **DiffusionGemma 26B-A4B bf16** (block-AR diffusion) contre les modèles Gemma 4 autoregressifs sur un mini-bench dérivé de [OCRBench](https://huggingface.co/datasets/echo840/OCRBench) (Yuliang Liu et al., Science China Information Sciences 2024).

## Setup

| Component | Specification |
|-----------|--------------|
| **Hardware** | Mac Studio M3 Max 96 GB |
| **Source dataset** | [echo840/OCRBench](https://huggingface.co/datasets/echo840/OCRBench) (1000 images, 25 sub-datasets) |
| **Sample** | 30 images stratifiées (3 par catégorie × 10 catégories), `random_state=42` |
| **Métrique** | Substring match case-insensitive de la ground-truth dans l'output |

## Modèles comparés

| Modèle | Architecture | Quantization | RAM peak | Speed (avg) |
|---|---|---|---|---|
| **DiffusionGemma 26B-A4B** | Block-AR diffusion (canvas 256, 9-15 denoising steps/canvas) | bf16 | 50.9 GB | ~30 tok/s |
| **Gemma 4 26B-A4B** | Autoregressive | 4-bit | 16 GB | ~25 tok/s |
| **Gemma 4 E4B** | Autoregressive | 4-bit | 6 GB | ~45 tok/s |

## Résultats

### Accuracy globale

| Modèle | Score |
|---|---|
| **Gemma 4 26B-A4B 4-bit** | **86.7% (26/30)** ⭐ |
| **DiffusionGemma 26B-A4B bf16** | **83.3% (25/30)** |
| **Gemma 4 E4B 4-bit** | **80.0% (24/30)** |

### Par catégorie OCRBench

| Catégorie | DiffusionGemma bf16 | 26B-A4B 4-bit | E4B 4-bit |
|---|:---:|:---:|:---:|
| Artistic Text Recognition | 3/3 (100%) | 3/3 (100%) | 3/3 (100%) |
| Doc-oriented VQA | 3/3 (100%) | 3/3 (100%) | 3/3 (100%) |
| Irregular Text Recognition | **3/3 (100%)** ⭐ | **3/3 (100%)** ⭐ | 2/3 (67%) |
| Key Information Extraction | 3/3 (100%) | 3/3 (100%) | 3/3 (100%) |
| Non-Semantic Text Recognition | 3/3 (100%) | 3/3 (100%) | 3/3 (100%) |
| Regular Text Recognition | 3/3 (100%) | 3/3 (100%) | 3/3 (100%) |
| Scene Text-centric VQA | 3/3 (100%) | 3/3 (100%) | 3/3 (100%) |
| **Handwriting Recognition** | **2/3 (67%)** ⭐ | **2/3 (67%)** ⭐ | 1/3 (33%) |
| Digit String Recognition | 1/3 (33%) | **2/3 (67%)** ⭐ | 2/3 (67%) |
| Handwritten Math Recognition | 1/3 (33%) | 1/3 (33%) | 1/3 (33%) |

## Observations

### DiffusionGemma vs 26B-A4B (même architecture base)

- **Quasi parité** : 25 vs 26 sur 30 (différence d'1 image)
- DiffusionGemma rate "8000" lu comme "808" (Digit String) où le 26B-A4B AR réussit
- Sur **8 catégories sur 10**, performance identique
- La conjecture initiale d'une dégradation due au paradigme block-AR diffusion vs AR n'est **pas confirmée** — la qualité de lecture est très proche

### DiffusionGemma vs E4B (cible légèrement inférieure)

- **+3.3%** : DiffusionGemma fait mieux
- Avantage marqué sur **Irregular Text** (texte courbe/orienté) : 100% vs 67%
- Avantage marqué sur **Handwriting** : 67% vs 33%
- Le block-AR diffusion **regarde le canvas entier en parallèle**, ce qui aide pour les textes spatialement complexes

### Faiblesses universelles

3 catégories où aucun modèle ne dépasse 67% :

1. **Handwritten Math** (33% pour tous) — formules LaTeX complexes avec fractions imbriquées et variables Unicode
2. **Digit Strings** (variable) — confusion entre chiffres similaires ("6262"→"2662", "8000"→"808")
3. **Handwriting** (33-67%) — écriture cursive avec dégradation lecture

Ces faiblesses sont cohérentes avec celles rapportées dans le paper OCRBench original sur les modèles vision-LLM open-source.

## Reproduction

```bash
# 1. Télécharger le sample (parquet → 30 images)
cd /tmp/ocrbench
curl -L "https://huggingface.co/datasets/echo840/OCRBench/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet" -o ocrbench.parquet

# 2. Extraire le sample stratifié
python3 <<'EOF'
import pyarrow.parquet as pq
import json, io, os
from PIL import Image

df = pq.read_table('/tmp/ocrbench/ocrbench.parquet').to_pandas()
sampled = []
for qt, sub in df.groupby('question_type'):
    sampled.extend(sub.sample(min(len(sub), 3), random_state=42).index.tolist())
sample_df = df.loc[sampled].reset_index(drop=True)

os.makedirs('/tmp/ocrbench/sample', exist_ok=True)
meta = []
for i, row in sample_df.iterrows():
    img = Image.open(io.BytesIO(row['image']['bytes'])).convert('RGB')
    if min(img.size) < 64:
        scale = 64 / min(img.size)
        img = img.resize((int(img.size[0]*scale), int(img.size[1]*scale)))
    out = f'/tmp/ocrbench/sample/img_{i:03d}.png'
    img.save(out)
    meta.append({'idx': i, 'image': out, 'dataset': row['dataset'],
                 'question_type': row['question_type'], 'question': row['question'],
                 'answer': list(row['answer'])})
json.dump(meta, open('/tmp/ocrbench/sample/meta.json', 'w'), indent=2)
EOF

# 3. Bench DiffusionGemma (copier run_bench.sh + score.py depuis docs/examples/ocr-bench/)
./run_bench.sh diff
python3 score.py diff

# 4. Comparer avec autres
./run_bench.sh e4b
./run_bench.sh a4b
python3 score.py e4b
python3 score.py a4b
```

## Fichiers

- `meta.json` : 30 images stratifiées avec ground truth answers
- `results/diff/*.txt` : sorties brutes DiffusionGemma
- `results/e4b/*.txt` : sorties brutes Gemma 4 E4B 4-bit
- `results/a4b/*.txt` : sorties brutes Gemma 4 26B-A4B 4-bit
- `score.py` : script de scoring (substring match case-insensitive)

## Notes méthodologiques

- **Échantillon réduit (30 images)** vs benchmark complet (1000 images). Les résultats sont indicatifs mais cohérents avec la hiérarchie attendue (taille modèle, archi).
- **Substring matching** est tolérant : "playin'" est marqué correct si la réponse contient "playin" même avec extras autour. Pour un scoring strict, il faudrait edit distance + normalisation Unicode.
- **DiffusionGemma utilise bf16** (48 GB) vs **AR en 4-bit** (15 GB). La quantization 4-bit du 26B-A4B AR peut introduire un biais qui équilibre artificiellement (cf. issue [#27](https://github.com/VincentGourbin/gemma-4-swift-mlx/issues/27)).
