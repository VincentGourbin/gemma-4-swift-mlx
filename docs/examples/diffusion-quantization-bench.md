# DiffusionGemma 26B-A4B — On-the-fly Quantization Benchmark

Profile sur 4 canvases (1024 tokens) prompt "Roman Empire essay", Mac Studio M3 Max 96 GB.

| Métrique | bf16 | 4-bit | 6-bit | 8-bit |
|---|---|---|---|---|
| Step avg | **605 ms** | 708 ms (+17%) | 683 ms (+13%) | 741 ms (+22%) |
| Step min | **506 ms** | 603 ms | 587 ms | 577 ms |
| Step std | 268 ms | **178 ms** | 280 ms | 289 ms |
| Total forwards | **63** | 167 (×2.6) | 68 | 71 |
| Total time | **1m26** | 4m10 (×2.9) | 1m44 (+21%) | 1m56 (+35%) |
| RAM après quantize | 49.2 GB | **46.7 GB** | 47.9 GB | 49.2 GB |
| Mémoire libérée | 0 | 2.5 GB | 1.3 GB | 0.1 GB |
| GPU peak | 50.7 GB | 50.3 GB | ~50 GB | 50.1 GB |

## Conclusion

**Sur Apple Silicon avec unified memory, le bf16 reste optimal pour ce modèle.**

Raisons :
- Pas de gain bande passante (RAM déjà unifiée avec GPU, pas de PCIe à traverser)
- `quantizedMM` ajoute de l'overhead CPU/GPU qui annule le gain
- MLX GPU pool ne libère pas agressivement les anciens bf16 (`Memory.clearCache()` ineffectif)
- MoE 4-bit dégrade les logits (cf. issue #27) → stopping criterion bien plus tardif → ×2.6 forwards

Pour cette architecture spécifique, prioriser :
1. ❌ Quantization 4/6/8-bit — pas de gain sur Apple Silicon
2. ✅ KV cache encoder incrémental — élimine le re-encoding O(N) du prompt cumulé
3. ✅ Garder bf16 et investir sur d'autres optim (kernels MLX-Fast, batch parallel)
