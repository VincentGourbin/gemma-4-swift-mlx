# Profils de référence

`gemma4-cli references` les liste, `gemma4-cli bench --reference <famille/id>` en mesure un,
`Gemma4ReferenceProfile.all` les expose à une app et `Gemma4Pipeline.load(profile:)` en charge un.
Chaque profil fige **tous** les réglages qui comptent : poids, cache KV, tranche de préfill,
limites mémoire, modalités. Même profil + même graine = même sortie, temps comparable sur
matériel comparable.

Source : `Sources/Gemma4Swift/Configuration/Gemma4ReferenceProfile.swift`. Protocole de mesure :
[Benchmarks.md](Benchmarks.md). Décision : [knowledge/decisions/reference-profiles.md](knowledge/decisions/reference-profiles.md).

## Mesures (campagne du 2026-09-27)

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
- `a4b/*-lean` mesuré avec une tranche de 256 dans la table ci-dessus (préfill −12 à −22 %). **Corrigé** : tranche 512 depuis le 2026-09-28 (A/B : +13 % de préfill, empreinte inchangée, voir BENCHMARKS.md).
- **Préfill du 12B (~360 tok/s) et du 31B (~130 tok/s) quasi indépendant des bits** : le calcul n'est pas borné par les poids. Suspects : attention pleine à `head_dim` 512 sans noyau fusionné (P-12) ; le 12B (Unified) n'a pas le préfill par tranches (K-13). Prochain levier.
- Décodage borné par la bande passante des poids : ×1,6 à ×2,7 entre 16 et 4 bits selon la famille.
- `weights_bw_gbps` n'est qu'un indicateur (surestimé sur E2B/E4B, tables d'embeddings par couche).

Poids et rôle par famille :

| Famille | Profils | Poids (4 / 8 / 16 bits) | KV en `lean` | Remarque |
|---|---|---|---|---|
| `e2b` | 4/8/16 × fast/lean | `gemma-4-e2b-it-{4bit,8bit,bf16}` | bf16 (1 tête KV) | texte, image, vidéo, audio |
| `e4b` | 4/8/16 × fast/lean | `gemma-4-e4b-it-{4bit,8bit,bf16}` | bf16 (1 tête KV) | texte, image, vidéo, audio |
| `b12b` | 4/8/16 × fast/lean | `gemma-4-12B-it-{4bit,8bit,bf16}` | bf16 (MQA) | **8 bits conseillé** ; 4 bits dégradé (MMLU 37 % contre 57 %, 100 questions) ; profil 6 bits en attente d'un MMLU sur ≥ 1 000 questions |
| `a4b` (26B-A4B) | 4/8/16 × fast/lean | `gemma-4-26b-a4b-it-{4bit,8bit,bf16}` | 8 bits | pas d'audio |
| `b31b` | 4/8/16 × fast/lean | `gemma-4-31b-it-{4bit,8bit,bf16}` | 8 bits | pas d'audio |

## Ce que fait chaque réglage

| Réglage | `fast` | `lean` | Effet |
|---|---|---|---|
| `kvBits` | bf16 | 8 bits pour 26B-A4B et 31B | KV quantifié par mlx-swift-lm pendant la génération. Non appliqué sur les chemins avec interdiction de n-grammes. |
| `prefillStepSize` | 512 | 256 (512 pour 26B-A4B) | Taille des tranches de préfill ; le head ne calcule que le dernier jeton. |
| `cacheLimitMB` | 4 096 | min(1 024, max(256, dispo/6)) | Cache de buffers MLX. Sans limite, la mémoire du process peut exploser (Qwen38 : 74 Go). |
| `memoryLimitMB` | — | max(4 096, dispo − 2 048) | Seuil de libération du cache MLX, pas un plafond dur. |
| `clearCacheAfterAnswer` | non | oui | `Memory.clearCache()` à la fin de chaque réponse. |
| `multimodal` | oui | oui | Tours vision/audio chargées ; `textOnlyVariant()` pour s'en passer. |

`16bit-lean` garde les caches Mac : des limites serrées font thrasher un working set bf16
(YuE2 : +73 % de temps). « dispo » = mémoire physique − 8 Go sur macOS,
`os_proc_available_memory()` sur iOS, ou `GEMMA4_AVAILABLE_MB`.

## Choisir

`Gemma4ReferenceProfile.recommended(for:availableMB:)` prend le `fast` le plus large dont les
poids tiennent sous la moitié de la mémoire disponible, sinon le `lean` le plus petit.
`gemma4-cli references` marque le profil conseillé pour la machine courante.

## Ajouter ou changer un profil

1. Modifier `Gemma4ReferenceProfile.make` (uniquement des réglages existants).
2. Mesurer avec `gemma4-cli bench --reference <id>` selon [Benchmarks.md](Benchmarks.md)
   (A/B/B/A, refroidissement, machine au repos) ; vérifier `profile_weights_match: true`.
3. Contrôler la qualité à graine égale contre `16bit-fast`.
4. Ajouter les lignes à `BENCHMARKS.md` et la décision à
   `docs/knowledge/decisions/reference-profiles.md`.

## DiffusionGemma (`a4bdiff/*`)

Six profils (`gemma4-cli references --family a4bdiff`), quantification à la volée depuis le bf16
officiel (`google/diffusiongemma-26B-A4B-it`, ~48 Go). Mesures du 2026-09-28 :

| Profil | Charge | Pas médian (ms) | Passes / canvas | Débit (tok/s) | Mémoire active (Mo) | Empreinte (Mo) |
|---|---|---|---|---|---|---|
| `a4bdiff/16bit-fast` | d1 | 502 | 15.0 | 32.7 | 49255 | 51288 |
| `a4bdiff/16bit-fast` | d2 | 488 | 13.0 | 35.4 | 49263 | 52617 |
| `a4bdiff/16bit-fast` | d3 | 534 | 12.0 | 33.3 | 49263 | 52989 |
| `a4bdiff/16bit-lean` | d1 | 504 | 15.0 | 32.5 | 49255 | 51266 |
| `a4bdiff/16bit-lean` | d2 ⚠ à vérifier | 483 | 5.0 | 85.6 | 48169 | 50115 |
| `a4bdiff/16bit-lean` | d3 | 538 | 16.0 | 27.4 | 48169 | 50245 |
| `a4bdiff/8bit-fast` | d1 | 521 | 16.5 | 29.6 | 26682 | 28696 |
| `a4bdiff/8bit-fast` | d2 | 521 | 10.0 | 43.3 | 26689 | 30015 |
| `a4bdiff/8bit-fast` | d3 | 543 | 11.0 | 36.1 | 26689 | 30396 |
| `a4bdiff/8bit-lean` | d1 | 528 | 16.5 | 29.1 | 26682 | 28561 |
| `a4bdiff/8bit-lean` | d2 ⚠ à vérifier | 524 | 4.0 | 110.1 | 25596 | 27327 |
| `a4bdiff/8bit-lean` | d3 | 552 | 12.0 | 35.3 | 25596 | 27602 |
| `a4bdiff/4bit-fast` | d1 | 548 | 43.5 | 10.6 | 14647 | 16726 |
| `a4bdiff/4bit-fast` | d2 ⚠ perturbé (ollama) | 2017 | 17.0 | 7.5 | 14655 | 18117 |
| `a4bdiff/4bit-fast` | d3 | 534 | 22.0 | 19.8 | 14655 | 18431 |
| `a4bdiff/4bit-lean` | d1 | 530 | 43.5 | 10.9 | 14648 | 16563 |
| `a4bdiff/4bit-lean` | d2 ⚠ à vérifier | 502 | 7.0 | 68.6 | 13561 | 15508 |
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
- **Pic de chargement corrigé** : la quantification se fait maintenant couche par couche (chaque couche de l'encodeur quantifiée, évaluée, reprise aussitôt par le décodeur). Pic mesuré (`benchmarks/diffusion-layerwise-quant-20260928.jsonl`) : 8 bits 77,2 → 51,0 Go, 4 bits mixte 68,7 → 51,0 Go, soit le bf16 seul ; mémoire en régime et passes inchangées.
- **8 bits** : le compromis mesuré aujourd'hui (−46 % de mémoire, −10 % de débit en texte).
- ❌ **`lean` + image (d2), lignes ci-dessus invalides** : l'échauffement déchargeait la vision et la passe mesurée ignorait l'image, sans erreur (bug de bibliothèque, pas seulement du bench : tout appel avec image après un déchargement). Corrigé (`9ffc8871`, rechargement à la demande) ; remesuré (`benchmarks/diffusion-vision-reload-20260928.jsonl`) : `8bit-lean` d2 10 passes, même réponse que `8bit-fast`, 25,6 Go actifs contre 26,7.
- ⚠ `4bit-fast` d2 : les deux passes sont perturbées (un `ollama` actif pendant la mesure) ; à refaire.
- **Qualité — ScreenSpot-100** (`gemma4-cli eval-screenspot`, les 100 cas du bench de référence reconstruits par `Scripts/quality/screenspot-sample.py` ; `benchmarks/screenspot-diffusion-20260928.jsonl`). Le bf16 retrouve 80/100 (79 dans la mesure d'origine) :

  | Config | Score | vs bf16 (perdus / gagnés) | Réponses identiques au bf16 | s/cas |
  |---|---|---|---|---|
  | bf16 (`16bit-fast`) | **80** | — | 100 | 6,9 |
  | 8 bits (`8bit-fast`) | 78 | −4 / +2 | 67 | 5,6 |
  | 4 bits mixte (`4bit-fast`) | 76 | −7 / +3 | 33 | 6,6 |
  | 4 bits uniforme (`--quant-variant 4bit-uniform`) | 77 | −6 / +3 | 35 | 8,8 |

  8 bits tient la porte (≤ 2 pts). Les 4 bits perdent 3-4 pts : au bord de l'écart-type d'un échantillon de 100 (~4 pts), donc ni tenue ni rejet nets de la porte ; le mixte n'est pas moins bon que l'uniforme et il est 25 % plus rapide par cas. BFCL non relancé : saturé à 95 % pour tous les modèles, il ne départage pas.

**Choisir, en l'état** : `a4bdiff/16bit-*` pour la qualité de référence ; `a4bdiff/8bit-*`
(~27 Go en régime) ; `a4bdiff/4bit-*` (désormais quantification mixte, ~21 Go, 38,5 tok/s en texte ;
`bench-diffusion --quant-variant 4bit-uniform` pour l'ancien 4 bits) ; ScreenSpot bf16 80, 8 bits 78, 4 bits 76. Tous passent par ~51 Go au chargement
(quantification à la volée depuis le bf16) : sous 64 Go de RAM, il faut des poids pré-quantifiés.

