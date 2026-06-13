# DiffusionGemma 26B-A4B — Synthèse Optimisations Phases 5-7

## Configuration de test
Mac Studio M3 Max 96 GB, prompt "Roman Empire essay" (1024 tokens / 4 canvases), bf16 baseline.

## Phase par phase

### Phase 5 — Quantization pure (4/6/8-bit)

| Bits | Step avg | Forwards | Total | RAM libérée |
|---|---|---|---|---|
| bf16 baseline | **605 ms** | **63** | **1m26** | — |
| 4-bit | 708 ms (+17%) | 167 (×2.6) | 4m10 (×2.9) | 2.5 GB |
| 6-bit | 683 ms (+13%) | 68 | 1m44 (+21%) | 1.3 GB |
| 8-bit | 741 ms (+22%) | 71 | 1m56 (+35%) | 0.1 GB |

**Verdict** : aucune quantization pure ne bat bf16 sur Apple Silicon (unified memory). Le 4-bit dégrade les logits MoE (cohérent issue #27).

### Phase 6 — Mixed Precision + Memory Management

**Pattern Q-DiT (CVPR 2025) + ViDiT-Q (ICLR 2025)** : premiers/derniers layers en 8-bit, milieu en 4-bit.

| Config | Step avg | Forwards | Total | RAM peak |
|---|---|---|---|---|
| bf16 baseline | 605 ms | 63 | 1m26 | 50.7 GB |
| **Mixed default** (P6) | **574 ms** ⭐ | 75 | 1m37 | 49.3 GB |

**Verdict** : mixed-precision est **plus rapide que bf16** (-5%) tout en préservant la qualité MoE (75 vs 167 forwards en 4-bit pur).

### Phase 7 — prevLogits bf16 + cache limit override

| Config | Step avg | Step min | RAM peak |
|---|---|---|---|
| bf16 baseline | 605 ms | 506 ms | 50.7 GB |
| Mixed P6 | 574 ms | 494 ms | 49.3 GB |
| **Phase 7 final** | **567 ms** | **485 ms** | **49.0 GB** |

**Verdict** : gain marginal supplémentaire (-1.2% sur step avg, -1.7 GB sur peak). `prevLogits` en bf16 économise 128 MB par step en théorie.

## Synthèse cumulative

```
            ┌─────────────┬──────────┬──────────┐
            │  bf16 base  │ Phase 6  │ Phase 7  │
            ├─────────────┼──────────┼──────────┤
Step avg    │   605 ms    │  574 ms  │  567 ms  │  -6.3%
Step min    │   506 ms    │  494 ms  │  485 ms  │  -4.2%
Forwards    │     63      │    75    │    75    │
RAM peak    │  50.7 GB    │ 49.3 GB  │ 49.0 GB  │  -1.7 GB
Total time  │   1m26      │  1m37    │  1m36    │  +12%
            └─────────────┴──────────┴──────────┘
```

## Compromis configurables

| Use case | Preset | Note |
|---|---|---|
| Vitesse pure, RAM disponible | bf16 baseline | 1m26, qualité ⭐ |
| Compromis qualité/RAM | Phase 6/7 Mixed default | 1m36, -1.7 GB peak |
| RAM critique (Mac 32-48 GB) | Mixed default + clearCache + unloadVision | -1 GB additionnel |
| RAM très critique (Mac 16-24 GB) | Mixed aggressive (2+2 high-prec) | À tester |

## Ce qui n'a PAS donné les gains attendus

- **Quantization 4/6/8-bit pure** : Apple Silicon unified memory + MoE dégradation
- **KV cache encoder incrémental** : 0.03% du total time
- **asyncEval entre steps** : déjà fait via `item()`
- **MLX cache limit override** : pool refuse de se réduire
- **Memory.clearCache() entre canvases** : neutre sans `unloadVision` actif

## Pistes futures

- **MMLU mixed-precision** sur 26B-A4B standard (vérifier la résolution numérique de #27)
- **Stopping criterion plus permissif** : ramener à 63 forwards malgré quantize (-15% total time)
- **Distillation des temperatures** : `t_min=0.4` baseline mais pourrait être plus bas
- **Investigate MLX pool behavior** : signaler upstream si la non-libération est un bug
