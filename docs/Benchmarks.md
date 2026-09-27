# Mesurer

Toute affirmation de performance de ce dépôt vient d'une ligne produite par
`gemma4-cli bench`, prise selon le protocole ci-dessous. Les anciens chiffres de
`BENCHMARKS.md` (outil `profile`) ne mesurent pas le chemin de la bibliothèque :
ils restent comme historique, pas comme référence.

## Protocole

1. Binaire **Release** : `xcodebuild -scheme gemma4-cli -configuration Release …`
   (le champ `build` de chaque ligne le vérifie).
2. **Machine au repos** : aucun autre process MLX (`qwen38`, `yue2`, un autre
   `gemma4-cli`), sur secteur, pas d'indexation Spotlight en cours. Le skill
   `mlx-swift-audit` fournit `machine-check.sh` pour le vérifier.
3. **Refroidissement** : `--cooldown 120` avant chaque point de référence.
4. **Une variable à la fois**, en ordre **A/B/B/A** (`--label` pour les distinguer).
   Un écart inférieur à 5 %, ou inférieur à la dispersion des deux A, est du bruit.
5. **Parité** avant et après toute optimisation (sortie greedy identique, ou écart
   borné et justifié).
6. Chaque ligne JSON est recopiée **telle quelle** dans `BENCHMARKS.md` ; une ligne
   n'est jamais modifiée, une nouvelle mesure ajoute une ligne.

## Usage

```bash
B=.build/xcode/Build/Products/Release/gemma4-cli
M=~/Library/Caches/models/mlx-community/gemma-4-e2b-it-4bit

# Texte : prompts de 128, 1 024 et 4 096 jetons exactement, 128 jetons générés
$B bench --model-path $M --prompt-tokens 128,1024,4096 --max-tokens 128 \
  --cooldown 120 --repeats 2 --label A --out bench.jsonl

# Image (chemin multimodal, 280 jetons d'image)
$B bench --model-path $M --image docs/examples/vision-image-description/input_sample.jpg \
  --max-tokens 128 --cooldown 120 --out bench.jsonl
```

Une passe d'échauffement non chronométrée (compilation Metal) précède les points ;
`--no-warmup` la désactive.

## Champs

| Champ | Sens |
|---|---|
| `prefill_ms`, `prefill_tok_s` | traitement du prompt (`TokenIterator.promptPrefillTime`) |
| `ttft_ms` | du début de la requête au premier jeton |
| `decode_tok_s_median` | débit de décodage tiré de l'intervalle médian : **la** valeur à comparer (A/A, A/B) |
| `decode_tok_s`, `step_ms_median`, `step_ms_p90` | débit moyen sur la durée totale (sensible aux pas lents isolés : 8 % d'écart A/A observé contre 1,8 % en médiane), intervalles |
| `peak_mlx_mb`, `active_mlx_mb`, `cache_mlx_mb` | mémoire MLX (pic remis à zéro à chaque point) |
| `phys_footprint_mb`, `phys_footprint_peak_mb` | empreinte du process (ce que juge jetsam) ; le pic court depuis le lancement, chargement compris |
| `weights_bw_gbps` | taille des poids × tok/s : **indicateur**, surestimé sur E2B/E4B (tables d'embeddings par couche lues sur quelques lignes par jeton) |
| `commit`, `dep_mlx_swift*`, `host`, `os`, `build` | contexte, à garder avec la ligne |

Le nombre de jetons générés est fixe (`--max-tokens`) : un EOS n'arrête pas la mesure.
