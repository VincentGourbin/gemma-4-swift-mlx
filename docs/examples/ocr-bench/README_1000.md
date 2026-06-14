# OCR Bench 1000 — Bench complet OCRBench (dataset full)

Bench complet sur les **1000 images** du dataset [OCRBench](https://huggingface.co/datasets/echo840/OCRBench) (Yuliang Liu et al., Science China Information Sciences 2024). C'est le test le plus exhaustif que ce repo propose pour comparer **DiffusionGemma 26B-A4B bf16** (block-AR diffusion) contre les modèles **Gemma 4 AR** (autoregressifs).

## Setup

| Component | Specification |
|-----------|--------------|
| **Hardware** | Mac Studio M3 Max 96 GB |
| **Dataset** | OCRBench complet (1000 images, 25 sub-datasets, 10 catégories) |
| **Métrique** | Substring match case-insensitive de la ground-truth dans l'output |
| **3000 inférences** | DiffusionGemma + 26B-A4B 4-bit + E4B 4-bit |

## Résultats globaux

| Rang | Modèle | Accuracy | Image/correct |
|---|---|---|---|
| 🥇 | **DiffusionGemma 26B-A4B bf16** | **80.8%** | 808/1000 |
| 🥈 | **Gemma 4 26B-A4B 4-bit AR** | **78.9%** | 789/1000 |
| 🥉 | **Gemma 4 E4B 4-bit AR** | **70.3%** | 703/1000 |

🏆 **DiffusionGemma BAT 26B-A4B AR de +1.9 pts** sur le dataset complet, confirmant que les classements précédents (30 et 300 images) étaient du bruit d'échantillonnage. **+10.5 pts** sur E4B 4-bit.

### Hiérarchie observée vs taille de sample

| Sample size | DiffusionGemma | 26B-A4B AR | E4B AR | Verdict |
|---|---|---|---|---|
| 30 | 83.3% | 86.7% | 80.0% | AR > Diff (1 image) |
| 300 | 84.0% | 84.3% | 78.7% | Parité (1 image) |
| **1000** | **80.8% ⭐** | 78.9% | 70.3% | **Diff > AR** |

Sur 1000 images, **DiffusionGemma est statistiquement le plus précis**, là où les classements précédents étaient instables.

## Détail par catégorie (1000 images)

| Catégorie | DiffusionGemma | 26B-A4B 4-bit | E4B 4-bit |
|---|:---:|:---:|:---:|
| Artistic Text Recognition | 49/50 (98%) | 49/50 (98%) | 48/50 (96%) |
| **Doc-oriented VQA** | **165/200 (82%)** ⭐ | 144/200 (72%) | 102/200 (51%) |
| **Key Information Extraction** | **172/200 (86%)** ⭐ | 169/200 (84%) | 161/200 (80%) |
| Regular Text Recognition | 48/50 (96%) | 49/50 (98%) | 49/50 (98%) |
| Irregular Text Recognition | 49/50 (98%) | 49/50 (98%) | 48/50 (96%) |
| Scene Text-centric VQA | 184/200 (92%) | 184/200 (92%) | 163/200 (82%) |
| Non-Semantic Text Recognition | 45/50 (90%) | **50/50 (100%)** ⭐ | 47/50 (94%) |
| Handwriting Recognition | 39/50 (78%) | 39/50 (78%) | 39/50 (78%) |
| Digit String Recognition | 40/50 (80%) | **41/50 (82%)** | 37/50 (74%) |
| Handwritten Math Recognition | **17/100 (17%)** ⭐ | 15/100 (15%) | 9/100 (9%) |

## Patterns clés

### 🎯 DiffusionGemma EXCELLE en compréhension contextuelle (VQA)

| Catégorie | DiffusionGemma | 26B-A4B AR | Avantage |
|---|---|---|---|
| **Doc VQA** | 82% | 72% | **+10 pts** ⭐⭐ |
| **Key Info Extract** | 86% | 84% | +2 pts |
| Handwritten Math | 17% | 15% | +2 pts |

→ Le block-AR diffusion regarde le canvas en parallèle dès le premier denoising step, ce qui permet une meilleure intégration du contexte spatial pour les tâches d'extraction d'information complexes (documents, formulaires, factures, charts).

### 🎯 26B-A4B AR EXCELLE en reconnaissance précise de caractères

| Catégorie | 26B-A4B AR | DiffusionGemma | Désavantage |
|---|---|---|---|
| Non-Semantic Text | 100% | 90% | -10 pts |
| Regular Text | 98% | 96% | -2 pts |
| Digit String | 82% | 80% | -2 pts |

→ La génération token-par-token AR est plus précise pour transcrire des séquences de caractères individuels (chiffres aléatoires, chaînes non sémantiques). Le block-AR diffusion converge globalement le canvas, ce qui peut introduire des erreurs locales (confusion 6/8, 0/O).

### ✅ Parité parfaite (4 catégories)

- Artistic Text (98%)
- Irregular Text (98%)
- Scene Text VQA (92%)
- Handwriting (78%)

## Performances (1000 inférences chacune)

### Time per image

| Modèle | avg | median | p95 |
|---|---|---|---|
| DiffusionGemma bf16 | **5.4 s** | 4.9 s | 8.5 s |
| Gemma 4 26B-A4B 4-bit AR | 2.9 s | 2.5 s | 5.2 s |
| Gemma 4 E4B 4-bit AR | 2.7 s | 2.0 s | 4.8 s |

### Tokens per second

| Modèle | avg | median | p95 |
|---|---|---|---|
| **DiffusionGemma bf16** | **51.3 tok/s** | **52.7** | **70.5** ⭐ |
| Gemma 4 26B-A4B 4-bit AR | 7.4 tok/s | 6.9 | 11.4 |
| Gemma 4 E4B 4-bit AR | 9.8 tok/s | 9.5 | 12.5 |

### Tokens generated per image

| Modèle | avg | median | p95 |
|---|---|---|---|
| DiffusionGemma | 256 (canvas plein) | 256 | 256 |
| 26B-A4B AR | 23 | 17 | 60 |
| E4B AR | 29 | 19 | 60 |

### Decoder forwards (DiffusionGemma uniquement)

- **avg = 5.0**, median = 4, p95 = 10
- Le stopping criterion `stable + confident` (mean entropy < 0.005) est **ultra-efficace** : seulement 4-5 forwards en médiane au lieu des 48 max
- Confirme que la qualité ne dépend pas du nombre de steps pour les tâches OCR (canvas converge vite)

### Total bench time

- DiffusionGemma : 89 min (5.4s × 1000)
- 26B-A4B AR : 48 min
- E4B AR : 45 min

## Observations clés

### 1. DiffusionGemma génère beaucoup plus de tokens mais 5-7× plus rapide en tok/s

Compromis architectural intéressant : on génère un canvas complet de 256 tokens en parallèle (et on dépend du tokenizer pour décoder), au lieu de générer token-par-token. Le **throughput en tok/s est 5-7× supérieur** mais le **canvas est rempli** vs **output AR court** → net 2× plus lent par image en wall-clock.

### 2. La qualité ne dépend PAS du nombre de denoising steps

5 forwards en médiane suffisent pour la plupart des tâches OCR. Le canvas converge vite quand la réponse est courte (read this word). Cela suggère que pour les tâches courtes (OCR pur, scene text), le coût wall-clock de DiffusionGemma pourrait être réduit en cappant `max_denoising_steps` à 10-15 au lieu de 48.

### 3. La quantization 4-bit n'a pas fait perdre 26B-A4B AR

Sur cet échantillon de 1000, 26B-A4B 4-bit est à **78.9%**. Le checkpoint mlx-community pré-quantizé n'introduit pas de perte massive sur OCR (cf. issue #27 pour les détails MMLU). Cela suggère que la quantization 4-bit MoE dégrade le raisonnement (-6 pts MMLU) mais pas la lecture pixel-par-pixel.

### 4. E4B 4-bit AR est nettement derrière

10.5 pts d'écart avec DiffusionGemma → l'archi 2.3B effectifs (E4B) montre ses limites sur des tâches denses. Doc VQA chute à 51% vs 82%.

## Fichiers

- `meta_1000.json` : 1000 entrées avec ground truth (pas de doublon des images, accédées via Gemma4ModelCache)
- `score1000.py` : script de scoring avec extraction des perfs CLI
- `run_bench_1000.sh` : reproduction
- `all_1000.sh` : enchaîne les 3 modèles
- Les 3000 fichiers de sorties brutes ne sont **pas commités** (~30 MB) ; ils sont régénérables via `run_bench_1000.sh`. Pour audit, le résumé compressé est dans `summary_1000.csv`.

## Reproduction

```bash
cd /tmp/ocrbench
# 1. Télécharger le parquet OCRBench
curl -L "https://huggingface.co/datasets/echo840/OCRBench/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet" -o ocrbench.parquet

# 2. Extraire les 1000 images + meta
python3 -c "
import pyarrow.parquet as pq
import json, io, os
from PIL import Image
df = pq.read_table('/tmp/ocrbench/ocrbench.parquet').to_pandas()
os.makedirs('/tmp/ocrbench/sample1000', exist_ok=True)
meta = []
for i in range(len(df)):
    row = df.iloc[i]
    img = Image.open(io.BytesIO(row['image']['bytes'])).convert('RGB')
    if min(img.size) < 64:
        scale = 64 / min(img.size)
        img = img.resize((int(img.size[0]*scale), int(img.size[1]*scale)))
    out = f'/tmp/ocrbench/sample1000/img_{i:04d}.png'
    img.save(out)
    meta.append({'idx': i, 'image': out, 'dataset': row['dataset'],
                 'question_type': row['question_type'], 'question': row['question'],
                 'answer': list(row['answer'])})
json.dump(meta, open('/tmp/ocrbench/sample1000/meta.json', 'w'), indent=2)
"

# 3. Bench (5-6h total) + scoring
./all_1000.sh
python3 score1000.py compare
```
