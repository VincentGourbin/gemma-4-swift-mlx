# OCR Bench 300 — DiffusionGemma vs Gemma 4 AR (extended)

Extension du mini-bench OCR à **300 images** (30 par catégorie × 10 catégories OCRBench). 10× plus large que le mini-bench (30) pour des résultats statistiquement plus robustes.

## Setup

| Component | Specification |
|-----------|--------------|
| **Hardware** | Mac Studio M3 Max 96 GB |
| **Source dataset** | [echo840/OCRBench](https://huggingface.co/datasets/echo840/OCRBench) |
| **Sample size** | **300 images** stratifiées (30/catégorie, `random_state=42`) |
| **Métrique** | Substring match case-insensitive de la ground-truth dans l'output |

## Résultats globaux

| Rang | Modèle | Score | RAM peak |
|---|---|---|---|
| 🥇 | **Gemma 4 26B-A4B 4-bit AR** | **84.3%** (253/300) | 16 GB |
| 🥈 | **DiffusionGemma 26B-A4B bf16** | **84.0%** (252/300) | 51 GB |
| 🥉 | **Gemma 4 E4B 4-bit AR** | **78.7%** (236/300) | 6 GB |

**DiffusionGemma est à 1 image** près de son équivalent AR sur cet échantillon plus large — confirme la parité observée sur 30 images (83.3% vs 86.7%).

## Détail par catégorie

| Catégorie | DiffusionGemma bf16 | 26B-A4B 4-bit | E4B 4-bit |
|---|:---:|:---:|:---:|
| Artistic Text Recognition | 30/30 (100%) ⭐ | 30/30 (100%) ⭐ | 30/30 (100%) ⭐ |
| **Doc-oriented VQA** | **29/30 (97%)** ⭐ | 26/30 (87%) | 21/30 (70%) |
| **Key Information Extraction** | **29/30 (97%)** ⭐ | 27/30 (90%) | 27/30 (90%) |
| Regular Text Recognition | 29/30 (97%) | 29/30 (97%) | 30/30 (100%) ⭐ |
| Irregular Text Recognition | 30/30 (100%) ⭐ | 30/30 (100%) ⭐ | 29/30 (97%) |
| **Scene Text-centric VQA** | **28/30 (93%)** ⭐ | 27/30 (90%) | 26/30 (87%) |
| Non-Semantic Text Recognition | 25/30 (83%) | **30/30 (100%)** ⭐ | 28/30 (93%) |
| Digit String Recognition | 22/30 (73%) | **24/30 (80%)** ⭐ | 20/30 (67%) |
| Handwriting Recognition | 23/30 (77%) | 23/30 (77%) | 22/30 (73%) |
| Handwritten Math Recognition | 7/30 (23%) | 7/30 (23%) | 3/30 (10%) |

## Patterns clés

### DiffusionGemma WINS (tâches VQA / compréhension contextuelle)

| Catégorie | DiffusionGemma | 26B-A4B AR | Diff |
|---|---|---|---|
| **Doc VQA** | 97% | 87% | **+10pts** ⭐ |
| **Key Info Extract** | 97% | 90% | **+7pts** ⭐ |
| **Scene Text VQA** | 93% | 90% | +3pts |

**Hypothèse** : le block-AR diffusion regarde tout le canvas en parallèle dès le premier denoising step, ce qui permet une meilleure intégration du contexte spatial pour les tâches d'extraction d'information depuis des documents complexes.

### 26B-A4B AR WINS (reconnaissance précise de caractères)

| Catégorie | 26B-A4B AR | DiffusionGemma | Diff |
|---|---|---|---|
| Non-Semantic Text | 100% | 83% | **-17pts** |
| Digit String | 80% | 73% | -7pts |

**Hypothèse** : la génération token-par-token AR est plus précise pour transcrire des séquences de caractères individuels (chiffres, lettres aléatoires) où chaque token doit être correct. Le block-AR diffusion peut introduire des erreurs locales (confusion 6/8, 0/O) car le canvas est convergé globalement.

### Parité parfaite

5 catégories où DiffusionGemma fait jeu égal avec 26B-A4B :

- **Artistic Text** (100%) — fonts décoratives
- **Irregular Text** (100%) — texte courbé/orienté
- **Regular Text** (97%) — texte simple
- **Handwriting** (77%) — écriture cursive
- **Handwritten Math** (23%) — formules LaTeX

### Faiblesses universelles

- **Handwritten Math 23%** pour 26B-A4B et DiffusionGemma, **10%** pour E4B
- Cohérent avec le paper OCRBench : les modèles vision-LLM open-source plafonnent autour de 25-40% sur cette catégorie

## Comparaison avec mini-bench 30

| Modèle | Mini 30 | Extended 300 | Variation |
|---|---|---|---|
| Gemma 4 26B-A4B 4-bit | 86.7% | 84.3% | -2.4pts |
| DiffusionGemma bf16 | 83.3% | 84.0% | **+0.7pts** |
| Gemma 4 E4B 4-bit | 80.0% | 78.7% | -1.3pts |

L'écart entre DiffusionGemma et 26B-A4B AR **se resserre** sur l'échantillon large : 1 image (0.3%) au lieu de 3.3% sur 30 images. Confirme la parité réelle entre les deux paradigmes pour ce modèle.

## Temps de bench

| Modèle | Temps total |
|---|---|
| DiffusionGemma bf16 | ~50 min |
| 26B-A4B 4-bit AR | ~25 min |
| E4B 4-bit AR | ~28 min |

## Fichiers

- `meta_300.json` : 300 images stratifiées + ground truth
- `results_300/{diff,e4b,a4b}/*.txt` : 900 sorties brutes
- `score300.py` : script de scoring
- `run_bench_300.sh` : reproduction
