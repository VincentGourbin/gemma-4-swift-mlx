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
- macOS 15+
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

## Profils de référence — campagne du 2026-09-27

Lignes brutes (une par mesure, jamais modifiées) : [`benchmarks/campaign-20260927-2256.jsonl`](benchmarks/campaign-20260927-2256.jsonl) (210 lignes : 6 A/A + 204 profils). Outil : `gemma4-cli bench` via `Scripts/bench-campaign.sh`.

Machine Mac15,10 (96 Go), Version 27.0 (Build 26A428), build release, commit e5a30020, mlx-swift 0.31.6, mlx-swift-lm 3.31.4, 2026-09-27. Protocole : docs/Benchmarks.md (cooldown 120 s, 2 passes, moyenne des passes ; pic = max).

| Profil | Préfill 128 / 1k / 4k (tok/s) | Décodage médian 128 / 1k / 4k (tok/s) | TTFT 4k (ms) | Pic MLX 4k (Mo) | Empreinte 4k (Mo) | Image : préfill / décodage / pic |
|---|---|---|---|---|---|---|
| `e2b/4bit-fast` | 1928 / 5137 / 6088 | 134.5 / 128.0 / 123.6 | 682 | 3086 | 3789 | 1188 / 130.3 / 4271 |
| `e2b/4bit-lean` | 2071 / 4907 / 5979 | 134.5 / 127.7 / 123.5 | 695 | 2788 | 3540 | 1189 / 129.9 / 4271 |
| `e2b/8bit-fast` | 1533 / 4952 / 5909 | 87.5 / 84.8 / 81.9 | 710 | 5288 | 6008 | 1180 / 85.7 / 6480 |
| `e2b/8bit-lean` | 1614 / 4737 / 5840 | 87.8 / 84.9 / 82.7 | 718 | 4996 | 5749 | 1187 / 85.9 / 6480 |
| `e2b/16bit-fast` | 928 / 3989 / 5632 | 53.6 / 52.2 / 51.3 | 751 | 9295 | 9889 | 925 / 52.8 / 10621 |
| `e2b/16bit-lean` | 872 / 3757 / 5530 | 53.5 / 52.2 / 51.4 | 765 | 9186 | 9930 | 924 / 52.9 / 10621 |
| `e4b/4bit-fast` | 949 / 1620 / 1684 | 78.0 / 73.0 / 72.0 | 2447 | 4785 | 5740 | 767 / 77.2 / 5796 |
| `e4b/4bit-lean` | 949 / 1594 / 1667 | 77.2 / 74.0 / 70.7 | 2472 | 4604 | 5406 | 740 / 75.7 / 5796 |
| `e4b/8bit-fast` | 804 / 1534 / 1660 | 47.8 / 46.4 / 45.3 | 2494 | 8269 | 9218 | 725 / 47.2 / 9357 |
| `e4b/8bit-lean` | 822 / 1530 / 1640 | 47.8 / 46.1 / 44.9 | 2523 | 8096 | 8980 | 717 / 47.3 / 9357 |
| `e4b/16bit-fast` | 515 / 1420 / 1683 | 27.6 / 27.2 / 26.9 | 2477 | 14755 | 15521 | 606 / 27.4 / 16032 |
| `e4b/16bit-lean` | 541 / 1416 / 1650 | 27.6 / 27.2 / 26.8 | 2525 | 14585 | 15536 | 585 / 27.4 / 16032 |
| `b12b/4bit-fast` | 307 / 367 / 361 | 36.2 / 34.5 / 34.1 | 11376 | 7633 | 9217 | — |
| `b12b/4bit-lean` | 306 / 362 / 357 | 36.1 / 34.8 / 34.3 | 11510 | 7360 | 8069 | — |
| `b12b/8bit-fast` | 283 / 363 / 358 | 20.9 / 20.3 / 19.9 | 11490 | 13196 | 14763 | — |
| `b12b/8bit-lean` | 284 / 356 / 353 | 21.0 / 20.3 / 20.0 | 11654 | 12893 | 13791 | — |
| `b12b/16bit-fast` | 255 / 370 / 376 | 11.6 / 11.4 / 11.3 | 10991 | 23639 | 25083 | — |
| `b12b/16bit-lean` | 256 / 363 / 362 | 11.6 / 11.4 / 11.3 | 11409 | 23391 | 25244 | — |
| `a4b/4bit-fast` | 416 / 930 / 965 | 81.3 / 77.1 / 74.4 | 4264 | 14439 | 15960 | — |
| `a4b/4bit-lean` | 570 / 851 / 850 | 81.5 / 76.1 / 73.2 | 4842 | 14237 | 15136 | — |
| `a4b/8bit-fast` | 375 / 924 / 927 | 50.9 / 48.9 / 47.9 | 4445 | 26465 | 27991 | — |
| `a4b/8bit-lean` | 421 / 789 / 796 | 50.9 / 48.9 / 47.5 | 5172 | 26185 | 27169 | — |
| `a4b/16bit-fast` | 161 / 714 / 808 | 32.6 / 31.8 / 31.3 | 5120 | 48938 | 50509 | — |
| `a4b/16bit-lean` | 167 / 525 / 630 | 32.6 / 31.6 / 31.0 | 6546 | 48758 | 50503 | — |
| `b31b/4bit-fast` | 123 / 138 / 132 | 15.1 / 14.3 / 13.9 | 31094 | 18607 | 21951 | — |
| `b31b/4bit-lean` | 123 / 137 / 129 | 14.9 / 14.2 / 13.6 | 31809 | 18249 | 18787 | — |
| `b31b/8bit-fast` | 115 / 134 / 124 | 8.2 / 8.1 / 7.7 | 33049 | 33172 | 36663 | — |
| `b31b/8bit-lean` | 114 / 133 / 125 | 8.2 / 8.0 / 7.7 | 32941 | 32832 | 33413 | — |
| `b31b/16bit-fast` | 108 / 135 / 141 | 4.6 / 4.5 / 4.5 | 29305 | 60512 | 64095 | — |
| `b31b/16bit-lean` | 109 / 129 / 134 | 4.6 / 4.5 / 4.4 | 30718 | 60144 | 63949 | — |

**Lecture** (une variable = le profil ; chiffres au repos, validation A/A de l'instrument : 2,9 % au pire) :
- `lean` coûte ≤ 2 % de préfill et rien en décodage sur E2B, E4B, 12B et 31B, pour 3 à 12 % de mémoire en moins (pic MLX, empreinte).
- **`a4b/*-lean` est mal réglé** : préfill 4k −12 % (4 bits) à −22 % (bf16) pour 1 à 5 % de mémoire en moins. La tranche de 256 pénalise ce MoE : à revoir (tranche 512).
- **Préfill du 12B (~360 tok/s) et du 31B (~130 tok/s) quasi indépendant des bits** : le calcul n'est pas borné par les poids. Suspects : attention pleine à `head_dim` 512 sans noyau fusionné (P-12) ; le 12B (Unified) n'a pas le préfill par tranches (K-13). Prochain levier.
- Décodage borné par la bande passante des poids : ×1,6 à ×2,7 entre 16 et 4 bits selon la famille.
- `weights_bw_gbps` n'est qu'un indicateur (surestimé sur E2B/E4B, tables d'embeddings par couche).

## Portes K-3 / K-13 — `main` contre la branche, A/B/B/A — 2026-09-28

E2B 4 bits, cooldown 120 s, ordre A B B A (A = `main` c9543739 + commande `bench` seule, B = branche 22902cf7). Lignes brutes : [`benchmarks/gates-20260928.jsonl`](benchmarks/gates-20260928.jsonl) (label `gate-A` / `gate-B` ; le champ `commit` y vaut 22902cf7 pour les deux : il lit le dépôt courant, pas le binaire).

| Point | Mesure | A (`main`) | B (branche) | Effet |
|---|---|---|---|---|
| texte 4 096 | préfill (tok/s) | 1 582 / 1 585 | 6 147 / 6 121 | ×3,9 |
| texte 4 096 | TTFT (ms) | 3 098 / 3 093 | 676 / 679 | ÷4,6 |
| texte 4 096 | pic MLX / empreinte (Mo) | 5 351 / 6 529 | 3 086 / 3 789 | −42 % / −42 % |
| texte 4 096 | décodage médian (tok/s) | 122,0 / 124,2 | 120,7 / 123,6 | bruit |
| texte 1 024 | préfill (tok/s) | 1 459 / 2 079 | 4 961 / 5 052 | ×2,4 à ×3,4 |
| texte 1 024 | pic MLX (Mo) | 3 699 | 2 964 | −20 % |
| image | décodage médian (tok/s) | 104,9 / 106,1 | 130,0 / 130,1 | +23 % |
| image | préfill (tok/s) / TTFT (ms) | 767-776 / 470-478 | 1 181-1 191 / 259-263 | +53 % / −45 % |
| image | empreinte (Mo) | 6 433 | 5 088 | −21 % |

**Portes** : K-13 — préfill ≥ +5 % ✅ (×3,9), décodage inchangé ✅, pic à 4k ≤ 50 % de la base ❌ de peu (58 %). K-3 — décodage image ≥ +5 % ✅ (+23 %), gain cumulé de K-3 (fp32), K-6 (vision bf16) et K-13 (préfill par tranches) : cette comparaison ne les sépare pas.

## `a4b/4bit-lean` : tranche de préfill 256 contre 512 — A/B/B/A — 2026-09-28

Lignes brutes : [`benchmarks/a4b-lean-20260928.jsonl`](benchmarks/a4b-lean-20260928.jsonl) (ordre 256, 512, 512, 256, puis `fast`).

| Variante | Préfill 1k / 4k (tok/s) | Décodage médian 4k | Pic MLX 4k | Empreinte 4k |
|---|---|---|---|---|
| lean, tranche 256 | 858-862 / 849-852 | 73,0-73,3 | 14 237 Mo | 15 135 Mo |
| lean, tranche 512 | 968-970 / 963-964 | 73,2-73,3 | 14 439 Mo | 15 116-15 118 Mo |
| fast | 985 / 966 | 74,5 | 14 439 Mo | 15 976 Mo |

**Décision** : `a4b/*-lean` passe à une tranche de 512 (+13 % de préfill, empreinte inchangée) ; l'économie de mémoire du profil lean (−5,4 % contre fast) vient des limites de cache, pas de la tranche.

Répété le 2026-10-01 sur le code de la 1.8.0 (`benchmarks/a4b-lean-prefill-20261001.jsonl`, A/B/B/A, 512 puis 256) :
tranche 512 → préfill 968-981 / 963-966 tok/s, TTFT 4k 4,26-4,27 s ; tranche 256 → 851-861 / 851 tok/s, TTFT 4k 4,83 s ;
décodage égal, pic MLX +200 Mo, sorties identiques. Même conclusion.

## DiffusionGemma 26B-A4B — profils `a4bdiff/*` — 2026-09-28

Lignes brutes : `benchmarks/diffusion-20260928-0940.jsonl` (bf16 ; les lignes 8/4 bits de ce fichier précèdent le correctif mémoire et ne comptent pas), `benchmarks/diffusion-20260928-1148.jsonl` (8bit-fast), `benchmarks/diffusion-20260928-1857.jsonl` (8bit-lean, 4 bits). Charges : d1 texte (25 jetons de prompt, 2 canvases), d2 image (`UI.png`, 305 jetons, 1 canvas), d3 contexte (773 jetons, 1 canvas).

| Profil | Charge | Pas médian (ms) | Passes / canvas | Débit (tok/s) | Mémoire active (Mo) | Empreinte (Mo) |
|---|---|---|---|---|---|---|
| `a4bdiff/16bit-fast` | d1 | 502 | 15.0 | 32.7 | 49255 | 51288 |
| `a4bdiff/16bit-fast` | d2 | 488 | 13.0 | 35.4 | 49263 | 52617 |
| `a4bdiff/16bit-fast` | d3 | 534 | 12.0 | 33.3 | 49263 | 52989 |
| `a4bdiff/16bit-lean` | d1 | 504 | 15.0 | 32.5 | 49255 | 51266 |
| `a4bdiff/16bit-lean` | d2 (remesuré 2026-10-01) | 490 | 8.0 | 51.0 | 48169 | 50113 |
| `a4bdiff/16bit-lean` | d3 | 538 | 16.0 | 27.4 | 48169 | 50245 |
| `a4bdiff/8bit-fast` | d1 | 521 | 16.5 | 29.6 | 26682 | 28696 |
| `a4bdiff/8bit-fast` | d2 | 521 | 10.0 | 43.3 | 26689 | 30015 |
| `a4bdiff/8bit-fast` | d3 | 543 | 11.0 | 36.1 | 26689 | 30396 |
| `a4bdiff/8bit-lean` | d1 | 528 | 16.5 | 29.1 | 26682 | 28561 |
| `a4bdiff/8bit-lean` | d2 (remesuré 2026-09-28) | 528 | 10.0 | 41.8 | 25596 | 27379 |
| `a4bdiff/8bit-lean` | d3 | 552 | 12.0 | 35.3 | 25596 | 27602 |
| `a4bdiff/4bit-fast` | d1 | 548 | 43.5 | 10.6 | 14647 | 16726 |
| `a4bdiff/4bit-fast` | d2 (remesuré 2026-10-01, pack) | 455 | 9.0 | 53.5 | 18786 | 21892 |
| `a4bdiff/4bit-fast` | d3 | 534 | 22.0 | 19.8 | 14655 | 18431 |
| `a4bdiff/4bit-lean` | d1 | 530 | 43.5 | 10.9 | 14648 | 16563 |
| `a4bdiff/4bit-lean` | d2 (remesuré 2026-10-01, pack) | 457 | 9.0 | 52.2 | 17693 | 19451 |
| `a4bdiff/4bit-lean` | d3 | 521 | 24.0 | 19.5 | 13561 | 15499 |

**Lecture** (M3 Max 96 Go, Release, cooldown 120 s, 2 passes ; A/A de l'instrument 0,0 à 0,7 %) :
- **Mémoire** (après les correctifs `2d6dbab5` experts MoE + `bf93a1ce` partage des modules) : bf16 49,3 Go actifs, 8 bits 26,7 Go (÷1,85), 4 bits 14,6 Go (÷3,4). Avant `bf93a1ce`, la quantification faisait monter la mémoire (8 bits 74,8 Go).
- **Pic au chargement (avant le correctif couche par couche)** : ~65 Go en 4 bits, ~77 Go en 8 bits, pour 49 Go de bf16 : le bf16 complet et tout le quantifié coexistaient. Voir plus bas.
- **Pas de débruitage** quasi constant (≈ 500-550 ms) quelle que soit la précision : 256 jetons par pas, calcul dominant.
- **4 bits uniforme** : 2,9× plus de passes en texte (43,5 contre 15), débit 10,6 tok/s contre 32,7. La cause est la précision des couches extrêmes, pas celle des embeddings (A/B `benchmarks/diffusion-4bit-variants-20260928.jsonl`, d1, 2 passes par variante) :

  | Variante (`--quant-variant`) | Passes/canvas | Pas médian | Débit | Actif MLX | Empreinte |
  |---|---|---|---|---|---|
  | 4 bits uniforme | 43,5 | 538-684 ms | 8,4-10,6 tok/s | 14 648 Mo | 16 726 Mo |
  | `4bit-sensitive8` (embeddings/tête + self_conditioning en 8 bits) | 44 | 486-523 ms | 11,2-11,7 tok/s | 15 668 Mo | 17 761 Mo |
  | **`4bit-mixed`** (couches 0-3 et 26-29 en 8 bits, sensibles en 8 bits) | **14,5** | **452-454 ms** | **38,5 tok/s** | 18 779 Mo | 20 871 Mo |

  `4bit-mixed` revient au nombre de passes du bf16 et le dépasse en débit ; il manque la porte « pic ≤ 18 Go » d'environ 3 Go. Qualité : voir ScreenSpot plus bas.
- **Suivi 4 bits** (`benchmarks/diffusion-4bit-followup-20260928.jsonl`, 2 passes chacun, sorties identiques entre passes) :
  - `--quant-variant 4bit-aggressive` (couches 0-1 et 28-29 en 8 bits), d1 : 20,5 passes/canvas, 27,0 tok/s, 17 227 Mo actifs, 19 303 Mo d'empreinte — sous 18 Go en actif seulement, et −30 % de débit contre le mixte pour −1,5 Go : le mixte reste le défaut.
  - `4bit-fast` (mixte), avec le correctif vision : d2 8 passes, 58,7 tok/s, empreinte 22,2 Go ; d3 9 passes, 47,4 tok/s, 22,6 Go. Remplace les lignes d2 perturbées par ollama.
- **Packs pré-quantifiés (K-D12)** — `gemma4-cli export-diffusion --model-path <bf16> --reference a4bdiff/8bit-fast --out <pack>` ; le pack se passe ensuite partout comme `--model-path` (format `gemma4-diffusion-prequantized-v1`, SHA-256 par fichier, une seule copie des modules partagés). Mesures d1 (`benchmarks/diffusion-packs-20260929.jsonl`, Lexar USB) :

  | Poids | Taille | Chargement | Pic au chargement | Passes/canvas | Débit | Sortie |
  |---|---|---|---|---|---|---|
  | 8 bits, à la volée depuis le bf16 | — | 64 s | 51,0 Go | 16,5 | 29,0 tok/s | référence |
  | **8 bits, pack** | 28,0 Go | **32 s** | **26,7 Go** | 16,5 | 28,9 tok/s | identique |
  | 4 bits mixte, à la volée | — | 62 s | 51,0 Go | 14,5 | 38,5 tok/s | référence |
  | **4 bits mixte, pack** | 19,7 Go | **11 s** | **18,8 Go** | 14,5 | 38,7 tok/s | identique |

  Porte « chargement ≤ ⅓ du bf16 (58 s) » : 4 bits tenue (11 s), 8 bits non (32 s, lecture de 28 Go limitée par le disque USB). Sorties identiques (début de réponse, passes, mémoire active) ; aller-retour bit-exact vérifié en test unitaire.
- **Pic de chargement corrigé** : la quantification se fait maintenant couche par couche (chaque couche de l'encodeur quantifiée, évaluée, reprise aussitôt par le décodeur). Pic mesuré (`benchmarks/diffusion-layerwise-quant-20260928.jsonl`) : 8 bits 77,2 → 51,0 Go, 4 bits mixte 68,7 → 51,0 Go, soit le bf16 seul ; mémoire en régime et passes inchangées.
- **8 bits** : le compromis mesuré aujourd'hui (−46 % de mémoire, −10 % de débit en texte).
- **`lean` + image (d2), premières lignes invalides (remplacées depuis dans le tableau)** : l'échauffement déchargeait la vision et la passe mesurée ignorait l'image, sans erreur (bug de bibliothèque, pas seulement du bench : tout appel avec image après un déchargement). Corrigé (`9ffc8871`, rechargement à la demande) ; remesuré (`benchmarks/diffusion-vision-reload-20260928.jsonl`) : `8bit-lean` d2 10 passes, même réponse que `8bit-fast`, 25,6 Go actifs contre 26,7.
- **Remesure d2 du 2026-10-01** (`benchmarks/diffusion-d2-lean-20261001.jsonl`, code de la 1.8.0, 4 bits sur le pack
  pré-quantifié) : les lignes d2 `16bit-lean`, `4bit-lean` et `4bit-fast` du tableau sont remplacées. `lean` et `fast`
  donnent **la même réponse** dans la même session (16 bits `[55, 115]`, 4 bits `[521, 85]`) : l'image est bien prise en
  compte. `16bit-fast` d2 sur le même code : 487 ms par pas, 8 passes, 53,7 tok/s, 49,3 Go actifs (la ligne du tableau,
  13 passes, précède les correctifs de diffusion). `lean` économise 1,1 Go actifs à réponse égale. La passe 1 de
  `16bit-lean` (40 tok/s) a tourné pendant un téléchargement sur le même disque ; la ligne retient la passe 2.
- **Qualité — ScreenSpot-100** (`gemma4-cli eval-screenspot`, les 100 cas du bench de référence reconstruits par `Scripts/quality/screenspot-sample.py` ; `benchmarks/screenspot-diffusion-20260928.jsonl`). Le bf16 retrouve 80/100 (79 dans la mesure d'origine) :

  | Config | Score | vs bf16 (perdus / gagnés) | Réponses identiques au bf16 | s/cas |
  |---|---|---|---|---|
  | bf16 (`16bit-fast`) | **80** | — | 100 | 6,9 |
  | 8 bits (`8bit-fast`) | 78 | −4 / +2 | 67 | 5,6 |
  | 4 bits mixte (`4bit-fast`) | 76 | −7 / +3 | 33 | 6,6 |
  | 4 bits uniforme (`--quant-variant 4bit-uniform`) | 77 | −6 / +3 | 35 | 8,8 |

  8 bits tient la porte (≤ 2 pts). Les 4 bits perdent 3-4 pts : au bord de l'écart-type d'un échantillon de 100 (~4 pts), donc ni tenue ni rejet nets de la porte ; le mixte n'est pas moins bon que l'uniforme et il est 25 % plus rapide par cas. BFCL non relancé : saturé à 95 % pour tous les modèles, il ne départage pas.

## Serveur — génération par lot (K-41) — 2026-10-02

E2B 4 bits (`e2b/4bit-fast`), M3 Max. Lignes brutes : `benchmarks/k41-batch-ceiling-20261002.jsonl` (plafond),
`benchmarks/k41-load-20261002.jsonl` (charge ; `-v1` et `-v2` : versions fautives, voir plus bas).

**Plafond** (`gemma4-cli bench-batch`, prompt 256, pas de décodage synchronisé) : B=1 9,2 ms/pas (109 tok/s),
B=2 9,1 ms (218), B=4 13,1 ms (306), **B=8 22,5 ms (356 tok/s, ×3,3)**, pic +1,2 Go.

**Charge** (8 clients × 4 requêtes, prompts courts, `max_tokens` 128, température 0, A/B/B/A) :

| | `--batch 1` | `--batch 8` |
|---|---|---|
| Débit agrégé | 127,7-127,9 tok/s | **267-271 tok/s (×2,1)** |
| TTFT p50 | 5,56-5,58 s | 0,16 s |
| TTFT p90 | 5,76-5,98 s | **1,44-1,71 s** |
| Réponses identiques au mode série | 32/32 | 4-6/32 |

Porte K-41 (×1,5, TTFT p90 ≤ +20 %) tenue. Les réponses du lot divergent de la génération seule après une
douzaine de jetons (logits au 1er pas : 0,84 % d'écart relatif, même argmax) et d'une passe à l'autre (20/32
identiques), selon la composition du lot.

**Défaut MLX trouvé en route** : les premières versions donnaient des lignes fausses (6 lignes sur 8 en boucle
« La mer / La mer ») : `MLXFast.RoPE` sur une entrée contiguë `[B > 1, H, 1, D]` calcule faux les lignes au-delà
de la première (écart 5,5-6 entre deux lignes identiques ; `RoPEBatchTests`). Contourné dans `RoPEWrapper`.
Reproduit en MLX Python 0.31.2 sur GPU (correct sur CPU) : défaut connu, [ml-explore/mlx#3494](https://github.com/ml-explore/mlx/issues/3494),
corrigé par mlx#3498 (inclus depuis MLX 0.32.0, plus reproduit en MLX 0.32.3). mlx-swift 0.31.6 embarque mlx-core 0.31.1 : le contournement
reste nécessaire jusqu'à la montée vers mlx-swift 0.32 (plan action-plans#602).

## DiffusionGemma — leviers K-D13 et variante de pas K-D15 — 2026-10-02

Pack 4 bits (`a4bdiff/4bit-*`), M3 Max. Lignes brutes : `benchmarks/diffusion-kd13-20261002.jsonl`,
`benchmarks/screenspot-kd15-20261002.jsonl`. Une application MLX (VoxtralApp) a pris le GPU par moments :
les temps de K-D13 (b) et (d) sont perturbés, les mémoires, scores et nombres de passes ne le sont pas.

**K-D13, leviers (A/B/B/A)** — aucun ne tient sa porte :

| Levier | Mesure | Porte | Verdict |
|---|---|---|---|
| (a) entropie compilée (`--compiled-entropy`) | pas −1,2 %, débit +0,3 %, sortie identique | +5 % | retiré |
| (b) `evalEveryNLayers` 8 | pic identique (19 996 Mo) | pic −10 % | retiré |
| (c) déchargement vision, d2 sur 2 canvases | pic −379 Mo, actif −1,09 Go, même réponse | pic −0,5 Go | manquée sur le pic ; déjà le réglage `lean`, gardé |
| (d) cache `lean` (1 Go) | cache 1 024 contre 1 763 Mo, empreinte −0,2 Go | empreinte ≤ actif + 1 Go | non tenue : ≈ 0,87 Go hors cache MLX |

**K-D15, variante de pas (ScreenSpot-100, `4bit-fast`, graine 0)** :

| Variante | Score | Passes / cas | Écart |
|---|---|---|---|
| référence (`entropy_bound` 0,1, `confidence_threshold` 0,005) | 78 | 5,99 | — |
| `entropy_bound` 0,2 | 78 | 5,35 | −10,7 % |
| `entropy_bound` 0,4 | 78 | 5,27 | −12,0 % |
| **`confidence_threshold` 0,02** | **77** | **4,17** | **−30,4 %** |

`confidence_threshold` 0,02 tient la porte (passes −≥ 15 %, score ± 1 pt) : variante publiée **hors profils**
(les profils gardent `generation_config.json`). À choisir quand le débit compte plus qu'un point de
ScreenSpot : `DiffusionGemmaPipeline.configureStepping(confidenceThreshold: 0.02)`, ou
`--confidence-threshold 0.02` sur `bench-diffusion` / `eval-screenspot`. Non mesurée sur texte libre (d1) ni
sur BFCL.

## Leviers K-15 à K-18 — 2026-09-29 (E2B 4 bits, M3 Max, A/B/B/A, cooldown 60 s)

Lignes brutes : `benchmarks/{ngram-k15,sampling-k17,vision-k18,noaudio-k16}-20260929.jsonl`. Une première série K-15 perturbée par une compilation est archivée (`benchmarks/archive/ngram-k15-20260929-perturbe.jsonl`).

| Fiche | Variante A | Variante B | Résultat | Porte |
|---|---|---|---|---|
| K-15 n-gramme (n = 5, 512 + 256 jetons) | ancien chemin CPU : 107,0-107,5 tok/s | historique GPU : 124,4-126,9 tok/s (sans n-gramme : 125,8-126,5) | **+17 %**, le coût du n-gramme disparaît ; mêmes jetons interdits à chaque pas sur 256 pas réels (`NoRepeatNGramDivergenceTests`) | ✅ |
| K-16 sans tour audio (image, 64 jetons) | 3 404 Mo actifs | 2 822 Mo actifs | **−583 Mo** (tour audio 0,61 Go en bf16 même dans le pack 4 bits), sortie et débit identiques | ✅ (≥ 0,3 Go) |
| K-17 échantillonnage | glouton : 125,8-126,4 tok/s | T 0,3 / top-p 0,95 : 118,6-119,5 tok/s | top-p coûte **5,6 %** (tri des 262 144 logits par jeton, `TopPSampler` amont) | au-dessus du seuil de 5 % ; pas de correction exacte sans synchronisation, voir PLAN |
| K-18 vision, image 624×1008 (2 457 patches) | préfill 252-273 ms | 243-258 ms | −3 % (dans le bruit : l'image remplit déjà 97 % des 2 520 patches) | — |
| K-18 vision, vidéo 70 jetons/frame (594 patches) | encodeur 193-194 ms/frame | **77-80 ms/frame** | **×2,5** ; contre une référence fp32, le nouveau chemin est plus précis (0,88 % contre 1,13 %) | ✅ (≥ ×2) |

K-18 sur l'image de référence : features à 3,8·10⁻⁴ (max, relatif) de l'ancien chemin ; description greedy identique sur 188 caractères puis un synonyme (« description » / « breakdown »), bascule d'argmax en bf16.
K-6 validé : `describe --quantize-bits 4` sur le bf16 d'E2B (encodeurs quantifiés compris) décrit correctement l'image (`benchmarks/describe-q4-20260929.txt`).

## LoRA — porte qualité director (K-29) — 2026-09-29

E2B bf16, `--mask-prompt --num-layers 16 --rank 8 --scale 20 --learning-rate 1e-4 --iterations 898 --seed 0 --val-batches 0`, dataset director de Fluxforge Studio (898/100, commit `f44da1aa`). Éval E7 : 30 briefs tenus à l'écart, `Scripts/quality/director-e7.py` (reproduit les références archivées : director-v3 29, base 17, professeur 27 ; et director-v3 relancé avec le binaire actuel : 29).

| Run | Troncature | Val loss finale | E7 | Durée | Pic MLX / empreinte |
|---|---|---|---|---|---|
| director-v3 (référence, avant audit) | non | 1,039 | 29/30 | ~2 h 40 | — |
| K-29 a (`benchmarks/lora-director-k29-*`) | 2048 (défaut éphémère) : 531/899 exemples coupés | 1,035 (valid. tronquée) | 27/30 | 2 h 17 | 36,3 / 76 Go |
| **K-29 b** (`benchmarks/lora-director-k29b-*`) | non (597 exemples > 2048, max 3 320) | **1,034** | **30/30** | 2 h 59 | 54,4 / 76 Go |
| **K-30 (défauts a + b)** (`benchmarks/lora-director-k30-*`) | non | **1,034** | **30/30** | **2 h 35** | **45,0 / 16,0 Go** |

La troncature par défaut (parité mlx-lm) coupait la fin des réponses sous `--mask-prompt` : retirée (`d3bd5a42`). K-29 b est la référence des A/B de K-30 (loss par pas dans `benchmarks/lora-director-k29b-20260929.jsonl`).

### K-30 : leviers mémoire de l'entraînement (director, pas 1-200, même graine que K-29 b)

| Variante | Perte @200 | Val @200 | Débit | Pic MLX | Empreinte |
|---|---|---|---|---|---|
| Baseline K-29 b | 1,147 | 1,1361 | 0,168 it/s (384 tok traités/s) | 50,3 Go | 76,0 Go |
| (b) cache MLX limité à 2 Go | 1,147 | 1,1361 | 0,180 (+7 %) | 50,4 Go | **15,8 Go** |
| (a) tête sur la réponse seule | 1,147 | 1,1361 | 0,191 (+13 %) | **40,7 Go** | 76,0 Go |
| **(a + b)** | **1,147** | **1,1361** | **0,211 (+26 %)** | **40,7 Go (−19 %)** | **15,9 Go (−79 %)** |

Pertes identiques à 4 décimales. (a + b) devient le défaut (`--full-head`, `--train-cache-limit-mb 0` pour revenir en arrière). Lignes : `benchmarks/lora-k30-{a,b,ab}-20260929.jsonl`.

## Lot I — résidence sous 4 Go (E2B 4 bits) — 2026-09-29

Mémoire disponible simulée (`GEMMA4_AVAILABLE_MB`), 2 passes, cooldown 30 s, sorties identiques (empreinte des jetons). `benchmarks/residency-lot-i-20260929.jsonl`.

| Configuration | Texte (1 024) : empreinte max | Image : empreinte max | Image : en régime | TTFT image |
|---|---|---|---|---|
| `4bit-fast` (audio, tours résidentes) | 3,39 Go | 5,06 Go | 5,06 Go | 253-264 ms |
| `4bit-lean` 6 Go, tours gardées | — | 4,36 Go | 4,07 Go | 255-264 ms |
| `4bit-lean` 6 Go, tours libérées (K-43) | 3,11 Go | 4,38 Go | 2,83 Go | 335-348 ms |
| + embedding de position sans one-hot | — | 4,17 Go | 2,83 Go | 318-319 ms |
| **`4bit-tiny`** (cache 256 Mo) | **3,11 Go** | **3,90 Go** | 2,83 Go | 315-319 ms |

`4bit-tiny` : texte 128 / 1 024 / 4 096 jetons = 2,92 / 3,11 / 3,28 Go, décodage 133 / 127 / 123 tok/s (= `fast`). L'embedding de position de l'encodeur vision lit la table au lieu d'un one-hot fp32 `[1, N, 2, 10240]` : pic MLX −133 Mo, TTFT −16 ms, sorties identiques. Détails et mode d'emploi : `docs/iOS.md`.

### K-31 : LoRA multimodal, base bf16 (TB3 LaTeX-OCR, E2B bf16, r16, lr 5e-5, 500 pas)

Jeu recréé et archivé (`Scripts/quality/latex-ocr-sample.py`, 500 train / 50 valid, SHA256SUMS) : l'échantillon du README (val 0,36, pic 24 Go) n'avait pas été conservé. Lignes : `benchmarks/lora-k31-*-20260929.jsonl`.

| Variante | Val @100 | Val @500 | Débit (500 pas) | Pic MLX | Empreinte | NaN |
|---|---|---|---|---|---|---|
| fp32 (tout le modèle, ancien défaut) | 0,596 | 0,428 | 1,11 it/s | 24,3 Go | 22,7 Go | non |
| base bf16 + LoRA fp32, tête promue en fp32 | 0,584 | — | 0,77 it/s (100 pas) | 15,9 Go | 12,9 Go | non |
| **base bf16 + LoRA fp32, tête au dtype de la table** | 0,585 | **0,430** | **1,38 it/s (+24 %)** | **14,4 Go (−41 %)** | **13,0 Go** | non |

Les couches LoRA sortent en fp32 (paramètres LoRA fp32) : sans retour au dtype de la table avant la tête liée, MLX promouvait ses 262 k × 1 536 poids en fp32 à chaque pas (débit ÷ 2).

### K-34 : `eval-mmlu` (E2B 4 bits, 5-shot)

Jeu recréé et archivé : `Scripts/quality/mmlu-sample.py` → 1 140 questions (20 par sujet, 57 sujets, graine 0) + 5 exemples « dev » par sujet (`/Volumes/Lexar/datasets/mmlu/mmlu_5shot_1140.json`, sha256 `447df0c0…`). Lignes : `benchmarks/mmlu-k34-20260929.jsonl`.

| Chemin | 200 questions | Score (200) | 1 140 questions |
|---|---|---|---|
| préfixe 5-shot re-préfillé à chaque question (`--no-prefix-cache`) | 74,7 s | 41,0 % (IC95 34,4-47,9) | — |
| **préfixe préfillé une fois par sujet + tête sur la dernière position** | **18,8 s (−75 %)** | 39,5 % (IC95 33,0-46,4) | **46,2 % (IC95 43,4-49,1), 106 s** |

Réponses concordantes à 195/200 : découper le calcul (préfixe puis suite) change les arrondis bf16 et fait basculer quelques argmax, comme en K-19 ; aucun repli de tokenisation (0 sur 1 140).

### K-32 : gradient checkpointing par couche (director, 200 pas, même graine)

| Variante | Perte @200 | Val @200 | Débit | Pic MLX | Empreinte |
|---|---|---|---|---|---|
| défauts K-30 | 1,1475 | 1,1378 | 0,202 it/s | 38,6 Go | 15,9 Go |
| **`--grad-checkpoint`** | **1,1475** | **1,1378** | 0,150 it/s (temps +35 %) | **21,1 Go (−45 %)** | 16,1 Go |

Pertes identiques ; porte « pic −30 %, temps ≤ +35 % » tenue, au bord pour le temps : option à activer quand la mémoire manque (Mac 32 Go, gros modèles), pas par défaut. Lignes : `benchmarks/lora-k32-{gc,base}-20260929.jsonl`.

