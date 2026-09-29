

### K-30 : leviers mémoire de l'entraînement (director, pas 1-200, même graine que K-29 b)

| Variante | Perte @200 | Val @200 | Débit | Pic MLX | Empreinte |
|---|---|---|---|---|---|
| Baseline K-29 b | 1,147 | 1,1361 | 0,168 it/s (384 tok traités/s) | 50,3 Go | 76,0 Go |
| (b) cache MLX limité à 2 Go | 1,147 | 1,1361 | 0,180 (+7 %) | 50,4 Go | **15,8 Go** |
| (a) tête sur la réponse seule | 1,147 | 1,1361 | 0,191 (+13 %) | **40,7 Go** | 76,0 Go |
| **(a + b)** | **1,147** | **1,1361** | **0,211 (+26 %)** | **40,7 Go (−19 %)** | **15,9 Go (−79 %)** |

Pertes identiques à 4 décimales. (a + b) devient le défaut (`--full-head`, `--train-cache-limit-mb 0` pour revenir en arrière). Lignes : `benchmarks/lora-k30-{a,b,ab}-20260929.jsonl`.

