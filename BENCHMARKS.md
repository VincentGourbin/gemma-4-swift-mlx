

## LoRA — porte qualité director (K-29) — 2026-09-29

E2B bf16, `--mask-prompt --num-layers 16 --rank 8 --scale 20 --learning-rate 1e-4 --iterations 898 --seed 0 --val-batches 0`, dataset director de Fluxforge Studio (898/100, commit `f44da1aa`). Éval E7 : 30 briefs tenus à l'écart, `Scripts/quality/director-e7.py` (reproduit les références archivées : director-v3 29, base 17, professeur 27 ; et director-v3 relancé avec le binaire actuel : 29).

| Run | Troncature | Val loss finale | E7 | Durée | Pic MLX / empreinte |
|---|---|---|---|---|---|
| director-v3 (référence, avant audit) | non | 1,039 | 29/30 | ~2 h 40 | — |
| K-29 a (`benchmarks/lora-director-k29-*`) | 2048 (défaut éphémère) : 531/899 exemples coupés | 1,035 (valid. tronquée) | 27/30 | 2 h 17 | 36,3 / 76 Go |
| **K-29 b** (`benchmarks/lora-director-k29b-*`) | non (597 exemples > 2048, max 3 320) | **1,034** | **30/30** | 2 h 59 | 54,4 / 76 Go |

La troncature par défaut (parité mlx-lm) coupait la fin des réponses sous `--mask-prompt` : retirée (`d3bd5a42`). K-29 b est la référence des A/B de K-30 (loss par pas dans `benchmarks/lora-director-k29b-20260929.jsonl`).

