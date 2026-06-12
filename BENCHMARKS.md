# Gemma 4 — Benchmarks collaboratifs

Ce fichier rassemble tous les benchmarks reproductibles pour la famille Gemma 4
(E2B, E4B, 12B Unified, 26B-A4B, 31B) à travers les différents frameworks
disponibles. Il est conçu pour être **collaboratif** : chaque ligne de résultat
référence le framework, la version, le matériel et la commande exacte utilisée.

L'objectif est triple :
1. Documenter la baseline de notre framework (Gemma 4 Swift MLX)
2. Permettre la comparaison avec d'autres ports (mlx-vlm Python, futurs ports CUDA, etc.)
3. Suivre l'évolution des perfs au fil des versions et des optimisations

---

## Comment lire ce fichier

Chaque catégorie de benchmark a :
- **Standard setup** : prompt, paramètres exacts, hardware de référence
- **Commande Swift** : ligne CLI pour reproduire avec ce repo
- **Commande Python** (quand applicable) : référence mlx-vlm
- **Tableau de résultats** : framework × version × hardware × score × contributeur

Le hardware de référence est **Apple M4 Max 96 GB RAM** (les contributeurs avec
d'autres machines sont encouragés à ajouter leurs colonnes).

---

## Comment contribuer un résultat

1. Reproduis le bench exact (commande dans la section concernée)
2. Ouvre une PR avec :
   - Une ligne de tableau ajoutée dans la bonne section
   - Format de la ligne : `| framework version | hardware | métrique | score | contributeur | date | commit |`
   - Le contributeur est ton handle GitHub
   - Le commit est le SHA court de ton repo (ou "N/A" pour un framework externe stable)
3. Si tu portes sur un nouveau framework (CUDA, ROCm, etc.), ajoute une section
   au-dessus du tableau qui décrit ton setup

Pour un nouveau benchmark (autre tâche, autre dataset), ouvre une PR qui :
- Ajoute une section sous "Benchmarks"
- Définit le standard setup
- Ajoute les commandes de reproduction
- Soumet au moins UN résultat de référence

---

## Frameworks référencés

| ID | Framework | Repo |
|---|---|---|
| `swift-mlx` | Gemma 4 Swift MLX (ce repo) | https://github.com/VincentGourbin/gemma-4-swift-mlx |
| `mlx-vlm-py` | mlx-vlm Python | https://github.com/Blaizzy/mlx-vlm |
| `mlx-lm-py` | mlx-lm Python (text-only) | https://github.com/ml-explore/mlx-lm |
| `transformers-cuda` | HuggingFace transformers + CUDA | https://github.com/huggingface/transformers |

---

## Modèles standard utilisés

| Alias | HF ID | Taille | Quantization |
|---|---|---|---|
| `E2B-bf16` | `mlx-community/gemma-4-e2b-it-bf16` | ~9 GB | bf16 |
| `12B-bf16` | `mlx-community/gemma-4-12B-it-bf16` | ~22 GB | bf16 |
| `12B-4bit` | `mlx-community/gemma-4-12B-it-4bit` | ~10 GB | mix 4-bit attn + 8-bit MLP |
| `31B-4bit` | `mlx-community/gemma-4-31b-it-4bit` | ~17 GB | 4-bit affine g=64 |

Les variantes 6-bit / 8-bit / bf16 sont disponibles symétriquement chez mlx-community.

---

# Benchmarks

## 1. Inference throughput (text-only)

### Standard setup

Prompt : `"Explain the theory of relativity, including special and general relativity, in detail with mathematical formulations."` (45 tokens)
Génération : 100 tokens, temperature 0.0, greedy
KV cache : bf16 (sauf si KV TurboQuant spécifié)

### Commandes

**swift-mlx** :
```bash
./.build/xcode/Build/Products/Release/gemma4-cli profile run \
  --model-path <MODEL_PATH> \
  --prompt "Explain the theory of relativity, including special and general relativity, in detail with mathematical formulations." \
  --max-tokens 100 --temperature 0.0 --no-chrome-trace \
  [--quantize-bits N --quantize-mode {affine|mxfp4} --quantize-group-size G]
```

**mlx-vlm-py** : utiliser `mlx_vlm.generate` avec les mêmes paramètres (voir `BENCHMARKS_python_scripts/` pour le script de référence).

### Résultats — 12B (M4 Max)

| Config | t/s | RAM MLX | framework version | hardware | contributor | date | commit |
|---|---|---|---|---|---|---|---|
| `12B-bf16` natif | 11.3 | 22.8 GB | swift-mlx@e2c5a99 | M4 Max 96 GB | @VincentGourbin | 2026-06-06 | e2c5a99 |
| `12B-bf16` + OTF 8-bit affine g=64 | 19.7 | 12.2 GB | swift-mlx@e2c5a99 | M4 Max 96 GB | @VincentGourbin | 2026-06-06 | e2c5a99 |
| `12B-bf16` + OTF 6-bit affine g=64 | 24.3 | 9.4 GB | swift-mlx@e2c5a99 | M4 Max 96 GB | @VincentGourbin | 2026-06-06 | e2c5a99 |
| `12B-bf16` + OTF 4-bit affine g=64 | 33.4 | 6.5 GB | swift-mlx@e2c5a99 | M4 Max 96 GB | @VincentGourbin | 2026-06-06 | e2c5a99 |
| `12B-bf16` + OTF 4-bit affine g=128 | 33.0 | 6.2 GB | swift-mlx@e2c5a99 | M4 Max 96 GB | @VincentGourbin | 2026-06-06 | e2c5a99 |
| **`12B-bf16` + OTF 4-bit mxfp4 g=32** | **34.7** | **6.2 GB** | swift-mlx@e2c5a99 | M4 Max 96 GB | @VincentGourbin | 2026-06-06 | e2c5a99 |
| `12B-4bit` pre-quant (mix 4/8) | 22.5 | 10.6 GB | swift-mlx@e2c5a99 | M4 Max 96 GB | @VincentGourbin | 2026-06-06 | e2c5a99 |
| `12B-4bit` pre-quant | 21.5 | 10.6 GB | mlx-vlm-py@0.6.2 | M4 Max 96 GB | @VincentGourbin | 2026-06-06 | N/A |

### Résultats — E2B (M4 Max)

| Config | t/s | RAM MLX | framework version | hardware | contributor | date | commit |
|---|---|---|---|---|---|---|---|
| `E2B-bf16` natif | 46.5 | 8.9 GB | swift-mlx@e2c5a99 | M4 Max 96 GB | @VincentGourbin | 2026-06-06 | e2c5a99 |
| `E2B-bf16` + OTF 8-bit | 74.6 | 4.7 GB | swift-mlx@e2c5a99 | M4 Max 96 GB | @VincentGourbin | 2026-06-06 | e2c5a99 |
| `E2B-bf16` + OTF 6-bit | 86.2 | 3.6 GB | swift-mlx@e2c5a99 | M4 Max 96 GB | @VincentGourbin | 2026-06-06 | e2c5a99 |
| `E2B-bf16` + OTF 4-bit affine | 105.3 | 2.5 GB | swift-mlx@e2c5a99 | M4 Max 96 GB | @VincentGourbin | 2026-06-06 | e2c5a99 |
| **`E2B-bf16` + OTF 4-bit mxfp4** | **108** | **2.4 GB** | swift-mlx@e2c5a99 | M4 Max 96 GB | @VincentGourbin | 2026-06-06 | e2c5a99 |

---

## 2. Prefill scaling (decode rate vs context length)

### Standard setup

Modèle : `12B-bf16` + OTF 4-bit mxfp4 g=32
Génération : 30 tokens en greedy après le prompt
Mesure : moyenne `ms/token` sur la phase de génération (post-prefill)

### Résultats — 12B (M4 Max)

| Prompt size | ms/token | t/s | RAM | framework version | hardware | contributor |
|---|---|---|---|---|---|---|
| 28 tokens | 30.5 | 32.8 | 6.5 GB | swift-mlx@e2c5a99 | M4 Max 96 GB | @VincentGourbin |
| 127 tokens | 30.4 | 32.9 | 6.6 GB | swift-mlx@e2c5a99 | M4 Max 96 GB | @VincentGourbin |
| 1226 tokens | 39.2 | 25.5 | 7.4 GB | swift-mlx@e2c5a99 | M4 Max 96 GB | @VincentGourbin |

Observation : decode reste stable jusqu'à ~150 tokens de contexte, puis chute
progressive au-delà (KV cache à parcourir grossit).

---

## 3. TurboQuant KV cache validation

### Standard setup

Modèle : `12B-bf16` ou `31B-4bit`
Test : sweep prompt size 256 → 8192 tokens, generation 30 tokens, temp 0.0
Configs comparées : KV bf16 vs KV TQ4 (4-bit affine TurboQuant)

### Commande Swift

```bash
./.build/xcode/Build/Products/Release/gemma4-cli profile run \
  --model-path <MODEL_PATH> --prompt "<LONG_PROMPT>" \
  --max-tokens 30 --temperature 0.0 --no-chrome-trace \
  --kv-bits 4   # active TurboQuant sur full_attention layers
```

### Comportement attendu du check viability

| Modèle | KV heads sur full_attn | Verdict check | Raison |
|---|---|---|---|
| `E2B` | 1 (MQA) | Désactivé silencieusement | gain 64 MB < 250 MB overhead |
| `12B Unified` | 1 (MQA full attn) | Désactivé silencieusement | gain 171 MB < 250 MB overhead |
| `31B` | 4 (GQA full attn) | Activé | gain 950 MB > 250 MB overhead |
| `26B-A4B` | 4 (GQA full attn) | Activé (à valider) | TBD |

Le check est exposé en static : `Gemma4LanguageModel.turboQuantViability(config:bits:)`.

### Résultats — 31B (TQ activé)

| Prompt | KV bf16 ms/t | KV TQ4 ms/t | Δ time | RAM bf16 | RAM TQ4 | Δ RAM | framework | hardware |
|---|---|---|---|---|---|---|---|---|
| 421 | 75.3 | 85.3 | +13% | 17.3 GB | 17.2 GB | -24 MB | swift-mlx@e2c5a99 | M4 Max 96 GB |
| 1573 | 78.5 | 88.6 | +13% | 18.6 GB | 18.5 GB | -107 MB | swift-mlx@e2c5a99 | M4 Max 96 GB |
| 6181 | 101.6 | 132.4 | +30% | 24.9 GB | 24.5 GB | -375 MB | swift-mlx@e2c5a99 | M4 Max 96 GB |
| 12325 | 162.0 | 164.8 | +2% | 33.2 GB | 32.5 GB | **-735 MB** | swift-mlx@e2c5a99 | M4 Max 96 GB |

Observation : TQ a un coût perf court contexte (+13-30%) qui s'amortit à long
contexte. Le gain RAM scale linéairement (~30 MB par K tokens). Pertinent pour
machines RAM-limitées sur contextes ≥ 8K.

### Résultats — 12B (TQ auto-désactivé)

| Prompt | bf16-kv | TQ4-kv (auto-désactivé) | Agreement | framework |
|---|---|---|---|---|
| 256 → 8192 (4 ctx) | identique | identique | 100% | swift-mlx@e2c5a99 |

Le check refuse silencieusement → `--kv-bits 4` fallback vers KV bf16 standard.

---

## 4. MMLU plain 5-shot (logit-based)

### Standard setup

Dataset : `cais/mmlu`, 10 sujets stratifiés × 10 questions test = 100 questions
5-shot examples : dev split de `cais/mmlu` (5 par sujet)
Méthodologie : argmax sur logits des tokens " A", " B", " C", " D" après "Answer:"

### Commande Swift

```bash
# Fetch dataset
python3 /tmp/benchwork/fetch_mmlu.py  # produit /tmp/mmlu_5shot.json

./.build/xcode/Build/Products/Release/gemma4-cli eval-mmlu \
  --model-path <MODEL_PATH> --dataset /tmp/mmlu_5shot.json \
  [--kv-bits 4] [--quantize-bits N --quantize-mode {affine|mxfp4}]
```

### Commande Python (référence)

```bash
PYTHONPATH=~/Library/Python/3.12/lib/python/site-packages \
  python3 /tmp/benchwork/mmlu_python.py --model-path <MODEL_PATH>
```

### Résultats

| Modèle | swift-mlx | mlx-vlm-py 0.6.2 | Δ Swift-Python | hardware | contributor |
|---|---|---|---|---|---|
| `E2B-bf16` | 47.0% | crash (KV-shared bug) | — | M4 Max 96 GB | @VincentGourbin |
| `E4B-4bit` | 55.0% | crash (KV-shared bug) | — | M4 Max 96 GB | @VincentGourbin |
| `12B-bf16` | 57.0% | 61.0% | -4 pts | M4 Max 96 GB | @VincentGourbin |
| `26B-A4B-4bit` (MoE) | 57.0% | **63.0%** | **-6 pts** ⚠ | M4 Max 96 GB | @VincentGourbin |
| **`31B-4bit`** | **66.0%** | **66.0%** | **0 (exact match)** | M4 Max 96 GB | @VincentGourbin |
| `12B-bf16` + OTF 8-bit | 58.0% | — | — | M4 Max 96 GB | @VincentGourbin |
| `12B-bf16` + OTF 6-bit | 50.0% | — | — | M4 Max 96 GB | @VincentGourbin |
| `12B-bf16` + OTF 4-bit affine | 32.0% | — | — | M4 Max 96 GB | @VincentGourbin |
| `12B-bf16` + OTF 4-bit mxfp4 | 34.0% | — | — | M4 Max 96 GB | @VincentGourbin |
| `12B-4bit` pre-quant (mix 4/8) | 37.0% | — | — | M4 Max 96 GB | @VincentGourbin |

**Notes** :
- E2B et E4B Python crash : bug `mlx-vlm 0.6.2` sur le KV-sharing (parameters not in model). Notre Swift port gère via `WeightSanitizer` qui strip les K/V projections des couches partagées.
- **26B-A4B (MoE) Swift -6 pts vs Python en 4-bit MAIS +2 pts en bf16** : la divergence est **isolée au path quantisé du SwitchGLU MLX-swift-lm**, pas dans notre Router ni dans notre archi MoE.

### Sweep isolation quant vs Router/MoE Swift (26B-A4B)

| Config | Swift plain | Python plain | Δ Swift-Py | Swift Pro | Python Pro | Δ Swift-Py |
|---|---|---|---|---|---|---|
| 26B-A4B-4bit | 57.0% | **63.0%** | **-6** ⚠ | 44.8% | 47.6% | -2.8 |
| **26B-A4B-bf16** | **59.0%** | 57.0% | **+2** ✓ | **51.4%** | 48.6% | **+2.8** ✓ |

**Verdict isolation** : sur bf16 non-quantisé, Swift donne un résultat équivalent ou meilleur que Python (+2 à +2.8 pts). Le bug -6 pts en 4-bit vient donc du dispatch quantisé de `SwitchGLU` dans `mlx-swift-lm` (probablement `quantizedSwitchLinear` ou `gather_mm` quantisé moins précis qu'en Python).

**Suivi** : voir [issue #27](https://github.com/VincentGourbin/gemma-4-swift-mlx/issues/27) pour le plan d'investigation détaillé et les hypothèses.

Observation : sur 12B Unified, la quantification 4-bit (n'importe quel mode)
**dégrade significativement** la qualité MMLU (-20 à -25 pts). Le 8-bit
préserve la qualité, le 6-bit perd 7 pts (acceptable). Le 31B est beaucoup plus
robuste à la quant 4-bit grâce à sa GQA(4) et ses 60 layers.

---

## 5. MMLU Pro 5-shot (logit-based, non-CoT)

### Standard setup

Dataset : `TIGER-Lab/MMLU-Pro`, 14 catégories × 15 questions = 210 questions
5-shot examples : split `validation` (5 par catégorie)
Méthodologie : argmax sur logits des tokens " A".." J" après "Answer:"
**Note** : Sans CoT → score absolu plus bas que les chiffres officiels Gemma 4 (77.2% / 85.2% sont avec CoT)

### Commande Swift

```bash
python3 /tmp/benchwork/fetch_mmlu_pro.py  # produit /tmp/mmlu_pro_5shot.json

./.build/xcode/Build/Products/Release/gemma4-cli eval-mmlu \
  --model-path <MODEL_PATH> --dataset /tmp/mmlu_pro_5shot.json
```

### Résultats

| Modèle | swift-mlx | mlx-vlm-py 0.6.2 | Δ Swift-Python | Google ref (CoT + chat template) | Δ vs Google | hardware | contributor |
|---|---|---|---|---|---|---|---|
| `E2B-bf16` | 24.8% | crash | — | 60.0% | -35 pts | M4 Max 96 GB | @VincentGourbin |
| `E4B-4bit` | 32.9% | crash | — | 69.4% | -36 pts | M4 Max 96 GB | @VincentGourbin |
| `12B-bf16` | 40.0% | 38.1% | +1.9 pts | 77.2% | -37 pts | M4 Max 96 GB | @VincentGourbin |
| `26B-A4B-4bit` (MoE) | 44.8% | **47.6%** | **-2.8 pts** | 82.6% | -38 pts | M4 Max 96 GB | @VincentGourbin |
| `31B-4bit` | 52.9% | 53.3% | -0.4 pts | 85.2% | -32 pts | M4 Max 96 GB | @VincentGourbin |

**Observations** :
- Δ Swift ↔ Python ≤ 3 pts sur tous les modèles testables → **portage validé**.
- Écart constant ~35 pts vs Google → c'est le **gap méthodologique** (raw text non-CoT vs chat template + CoT). Cohérent à travers les 5 modèles.
- 26B-A4B Swift -2.8 vs Python : à investiguer côté MoE.

---

## 6. MMLU Pro 5-shot CoT (Chain-of-Thought)

### Standard setup

Dataset : `TIGER-Lab/MMLU-Pro` avec `cot_content` (raisonnement pré-rédigé) dans les 5-shot
Génération : greedy, max_tokens 512, parse "The answer is (X)" / "(X)"
**Note** : Pour matcher les 77.2% / 85.2% officiels, il faut probablement utiliser la chat template Gemma 4 et un prompt engineering plus poussé. Notre format raw text donne des scores plus bas mais permet la comparaison Swift ↔ Python.

### Commande Swift

```bash
python3 /tmp/benchwork/fetch_mmlu_pro_cot.py  # produit /tmp/mmlu_pro_cot.json

./.build/xcode/Build/Products/Release/gemma4-cli eval-mmlu \
  --model-path <MODEL_PATH> --dataset /tmp/mmlu_pro_cot.json \
  --cot [--cot-max-tokens 512]
```

### Résultats

| Modèle | swift-mlx | mlx-vlm-py 0.6.2 | Δ Swift-Python | Référence Gemma 4 (CoT + chat template) | contributor | commit |
|---|---|---|---|---|---|---|
| `12B-bf16` v1 (avant fix perf) | 23.3% (7h08) | 19.5% (2h47) | +3.8 pts (dans le bruit) | 77.2% | @VincentGourbin | 08b83b6 |
| **`12B-bf16` v2 (après fast-path)** | **23.3% (3h06)** | 19.5% (2h47) | +3.8 pts (dans le bruit) | 77.2% | @VincentGourbin | 5403548 |
| `31B-4bit` | TBD | TBD | — | 85.2% | @VincentGourbin | — |

**Score identique avant/après fix** → déterminisme greedy validé.

**Note sur l'écart aux chiffres officiels** (~55 pts) : le format raw text 5-shot
CoT sous-utilise un modèle instruction-tuned. Pour matcher 77.2% officiel, il
faudrait :
- Chat template Gemma 4 (`<bos><start_of_turn>user...<end_of_turn>`)
- System prompt potentiellement spécifique
- Prompt engineering pour la structure CoT

**Note sur la perf CoT — évolution** :
- v1 (commit 08b83b6) : Swift 7h08, Python 2h47 → Swift **2.5× plus lent**
- v2 (commit 5403548) : Swift 3h06, Python 2h47 → Swift **1.11× plus lent** (gain 2.3×)

Le fix v2 introduit un fast-path `forwardWithoutIntermediates` dans
`Gemma4TextModel.callAsFunction` qui bypass la collecte d'intermediates K/V
pour les modèles SANS KV-sharing (12B Unified, 31B). Avant le fix, on retenait
~150 MB de K/V refs inutilisées pendant chaque forward, ce qui ajoutait une
pression mémoire significative pendant le prefill long (1500+ tokens du 5-shot
prompt).

Pour E2B/E4B (avec KV-sharing), le path complet `forwardCollectingIntermediates`
reste utilisé car nécessaire à l'algorithme de partage K/V entre couches.

---

## 7. Visualisation multimodale (qualitative)

### Standard setup

Test : description d'image avec prompt "Describe what's in the image in 2 sentences."
Image de référence : voir `tests/fixtures/runner.jpg` (à ajouter)
Modèle : `12B-bf16` + OTF 4-bit mxfp4

### Commande Swift

```bash
./.build/xcode/Build/Products/Release/gemma4-cli describe \
  --model-path <MODEL_PATH> --image <IMAGE_PATH> \
  --prompt "Describe what's in the image in 2 sentences." \
  --max-tokens 100 --temperature 0.0 \
  --quantize-bits 4 --quantize-mode mxfp4
```

### Qualité (subjective, à valider sur dataset MM-Vet ou similaire)

Sur l'image de coureurs au marathon, le 12B Unified lit correctement :
- Textes sur les vêtements : "F&M TRACK CLUB" ✓
- Textes sur banderoles : "START", "SPAR" ✓
- Textes sur dossards : "CAELIN", "WELMA" ✓

→ La bidirectional attention sur tokens vision (commit 79341c2) est critique
pour la lecture d'OCR. Sans elle, le modèle hallucine ("ASICS TRAXXON" inventé,
"RUNNING FOR LIFE" inventé).

---

## Références externes

### Chiffres officiels Gemma 4 (depuis HuggingFace model cards)

| Modèle | MMLU Pro (CoT) | GPQA Diamond | AIME 2026 | LiveCodeBench v6 | MMMLU |
|---|---|---|---|---|---|
| `E2B-it` | 60.0% | 43.4% | 37.5% | 44.0% | 67.4% |
| `E4B-it` | 69.4% | 58.6% | 42.5% | 52.0% | 76.6% |
| `12B-it` | 77.2% | 78.8% | 77.5% | 72.0% | 83.4% |
| `26B-A4B-it` (MoE) | 82.6% | 82.3% | 88.3% | 77.1% | 86.3% |
| `31B-it` | 85.2% | 84.3% | 89.2% | 80.0% | 88.4% |

Sources :
- https://huggingface.co/google/gemma-4-E2B-it
- https://huggingface.co/google/gemma-4-E4B-it
- https://huggingface.co/google/gemma-4-12B-it
- https://huggingface.co/google/gemma-4-26b-a4b-it
- https://huggingface.co/google/gemma-4-31B-it

**Note méthodologique** : Google reporte uniquement MMLU Pro avec CoT + chat template. Notre eval-mmlu utilise un format raw text 5-shot logit-based — donc Δ ~35 pts constant attendu vs Google. La comparaison Swift ↔ Python valide le portage à epsilon près ; la comparaison vs Google valide le tier du modèle (E2B < E4B < 12B < 26B-MoE < 31B est préservé).

### Hardware de référence M4 Max

- Apple M4 Max
- 96 GB RAM unified
- 546 GB/s memory bandwidth nominale
- macOS 14+
- Swift 6.0
- mlx-swift 0.31.4
- mlx-vlm Python 0.6.2

---

## Historique des optimisations

| Date | Commit | Optimisation | Impact mesuré |
|---|---|---|---|
| 2026-06-06 | 79341c2 | Port Gemma 4 12B Unified + bidirectional attention | qualité OCR multimodal: hallucinations → lectures exactes |
| 2026-06-06 | 79341c2 | ProportionalRoPE simplifiée (inf-padded freqs) | +2-3% throughput 12B |
| 2026-06-06 | 79341c2 | On-the-fly quantization (`--quantize-bits N`) | -3.5× RAM, +3× throughput vs bf16 |
| 2026-06-06 | 5fde514 | Bump mlx-swift 0.31.3 → 0.31.4 | mxfp4 fonctionnel, +6% vitesse 12B 4-bit |
| 2026-06-07 | e2c5a99 | Fix TurboQuant viability check pour MQA | TQ désactivé proprement sur 12B/E2B, évite -19% latence accidentelle |
| 2026-06-10 | 5403548 | Fast-path callAsFunction sans intermediates (12B/31B) | MMLU Pro CoT 12B-bf16 7h08 → 3h06 (**2.3×**), throughput pur inchangé |

---

## Annexe : scripts de fetch des datasets

Tous les scripts vivent dans `BENCHMARKS_python_scripts/` (à ajouter au repo).

| Script | Sortie | Source HF |
|---|---|---|
| `fetch_mmlu.py` | `/tmp/mmlu_5shot.json` | `cais/mmlu` (10 subjects × 10 q) |
| `fetch_mmlu_pro.py` | `/tmp/mmlu_pro_5shot.json` | `TIGER-Lab/MMLU-Pro` (14 cats × 15 q) |
| `fetch_mmlu_pro_cot.py` | `/tmp/mmlu_pro_cot.json` | `TIGER-Lab/MMLU-Pro` avec `cot_content` |
| `mmlu_python.py` | run éval Python mlx-vlm | logit-based |
| `mmlu_pro_cot_py.py` | run éval Python CoT | génération + parse |
