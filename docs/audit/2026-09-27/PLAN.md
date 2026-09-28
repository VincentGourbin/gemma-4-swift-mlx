# Plan — Stabilisation, performance et profils de référence — 2026-09-27

> Produit par le skill `mlx-swift-audit` sur `main` @ `c9543739` (mlx-swift 0.31.6, mlx-swift-lm 3.31.4).
> Rapports : [audit-stabilite.md](audit-stabilite.md) (S-01…S-37), [audit-performance.md](audit-performance.md)
> (P-01…P-16, matrice de profils, baseline), [scan.md](scan.md) (indices déterministes).
> Catalogue de référence : `~/.claude/skills/mlx-swift-audit/references/techniques.md` (T-x retenues, R-x rejetées).

## 0. Règles de mesure

Binaire Release · `~/.claude/skills/mlx-swift-audit/scripts/machine-check.sh --cooldown 120` vert
(aucun autre process MLX : le 27/09, un `qwen38` Release de 8,2 Go tournait) · un levier par
comparaison · A/B/B/A · gain < 5 % = bruit, le levier est retiré · une ligne JSON par mesure
recopiée telle quelle dans `BENCHMARKS.md` avec la révision des dépendances · parité greedy avant/après
sur le checkpoint réel · tests via `Scripts/run-tests.sh` uniquement (deadlock ABBA).

## 1. Faits vérifiés (ne pas re-dériver)

| Fait | Preuve |
|---|---|
| Aucun stream de la bibliothèque n'est annulable | aucun `onTermination` hors `Gemma4BenchUI` (`grep -rn onTermination Sources`) |
| Téléchargement partiel = « complet » dès 1 `.safetensors` | `Pipeline/Gemma4ModelCache.swift:93-99` |
| Préfill multimodal promu en fp32 (cache KV et décodage suivent) | `MLXArray(embedScale, dtype: .float32)` : `Gemma4MultimodalLLMModel.swift:116`, `Gemma4UnifiedMultimodalLLMModel.swift:209`, `Multimodal/Gemma4Model.swift:47` — la diffusion utilise `inputsEmbeds.dtype` |
| `kvBits` casse les couches KV-partagées E2B/E4B | `Gemma4Attention.swift:181-192` lit `state[0]/state[1]` = (poids packés, échelles) d'un `QuantizedKVCache` |
| `prepare` surchargé : `prefillStepSize` ignoré, logits sur toutes les positions | `Gemma4LLMModel.swift:76-85` et équivalents (P-01) |
| Aucune politique mémoire MLX sur l'inférence | `cacheLimit` seulement dans la diffusion (scan §4) |
| `asyncEval` mort dans l'outil de profil → chiffres actuels de `BENCHMARKS.md` hors chemin bibliothèque | `ProfileCommand.swift:185-186` (P-07) |
| 5 consommateurs externes (Fluxforge sur `main`, LTX et flux-2 `from: 1.5.0`, h3 `from: 1.0.0`, ToolsForge) ; aucun n'utilise TurboQuant, diffusion, `Gemma4MTPPipeline` | audit-stabilite §F |

## 2. Baseline (à remplir par K-10, avant toute fiche perf)

| Point | Workload | Préfill tok/s | Décodage tok/s (méd/p90) | TTFT | Pic phys_footprint | Ligne BENCHMARKS |
|---|---|---|---|---|---|---|
| B1 | E2B 4 bits, 128/1k/4k/8k jetons, 128 greedy | | | | | |
| B9 | E2B 4 bits, 8k/32k, footprint vs actif | | | | | |
| B3 | E2B 4 bits, 1 image vs texte même longueur | | | | | |
| B6 | E2B bf16 + drafter, 100 et 700 jetons (correction MTP) | | | | | |
| B4/B5 | vidéo 9 frames / audio 30 s | | | | | |
| B7/B8 | n-gramme n=5 ; temp 0 vs 0,3/topP 0,95 | | | | | |
| B11 | `multimodal: true` vs `false` (résidence) | | | | | |
| B2 | 12B 8 bits, 512/4k + `forwardCollectingHiddenStates` (LTX) | | | | | |
| B10 | 4 tours `continueChat` vs re-préfill | | | | | |

## 3. Fiches

Ordre imposé : correction/stabilité → hygiène → instrument + baseline → leviers perf → profils → 2.0.0.
Lots A-E : **1.8.0, strictement additive** (aucun défaut modifié, Fluxforge suit `main`). Lot F : 2.0.0.

### Lot A — Correction et stabilité (bloquant)

| Fiche | Objet | Source | Porte | Effort |
|---|---|---|---|---|
| K-1 | Streams annulables (`onTermination` → `task.cancel()`, `Task.checkCancellation` dans les boucles, MTP inclus) + `eval` des `MLXArray` avant traversée (`chatStreamMultimodal`, `processVideo/Audio`) | S-01, S-03, S-10 | test : consommateur qui s'arrête à 5 jetons → la tâche finit en < 1 pas ; appel suivant non bloqué | S |
| K-2 | Complétude du téléchargement (index safetensors + marqueur), chemin renvoyé, `--force`/`to:` | S-02, S-06, S-07 | test : dossier avec 1 shard sur 2 → `isDownloaded == false`, reprise effective | S |
| K-3 | Supprimer la promotion fp32 (`dtype: inputsEmbeds.dtype`) sur les 3 chemins multimodaux | P-02 | parité greedy image 32/32 vs avant ; dtype du cache = bf16 ; décodage image ≥ +5 % (attendu bien plus : 12B 12-13 → ~20 t/s) | S |
| K-4 | MTP : rollback correct au-delà de la fenêtre glissante (512) ; réduire les synchronisations | P-04 | `mtp-generate --compare` bit-exact à 700 jetons de prompt | M |
| K-5 | `kvBits` : ne jamais quantifier les couches sources des couches KV-partagées (ou refuser proprement) ; sortir `kvBits` de TurboQuant | P-05, S-13 | E2B `kvBits=8` : génération identique à fp16 sur 64 jetons (ou erreur explicite) | S-M |
| K-6 | Vision : `patches.asType(uint32)` si `input_proj` quantifié ; exclure les encodeurs de la quantification à la volée par défaut | P-03 (part. correction) | test : `--quantize-bits 4` + image E2B → description correcte | S |
| K-7 | `loadContainer` : fabrique `LLMModelFactory` privée par appel (plus de course sur le registre global) | S-05 | test : 2 chargements concurrents → bons types | S |
| K-8 | FFT audio sans pointeurs temporaires ; détokenisation en streaming (MTP, CLI) | S-09, S-11 | 0 warning `#TemporaryPointers` ; test UTF-8 multi-octets | S |
| K-9 | Deadlock ABBA : doc sur les 5 entrées d'entraînement + garde optionnelle `Gemma4ComputeGate` | S-04 | doc présente ; test de la garde (sérialisation) | S-M |
| K-4b | MTP : écart au greedy standard sur des quasi-égalités, même en contexte court et avec `--sequential-verify` (préexistant) ; trouver la source (préfill, dtype des logits, softcap…), corriger ou retirer « bit-exact » du README | trouvé en K-4 | `mtp-generate --compare` IDENTIQUE sur 256 jetons (prompt court et long), ou README corrigé | M |

### Lot B — Hygiène (sans risque)

| Fiche | Objet | Source | Porte | Effort |
|---|---|---|---|---|
| K-10a | Images en double à la racine, 1 051 fichiers de résultats sous `docs/examples/` (→ release asset ou `.local-runs`), chemins absolus ; `git gc` local | S-36, S-37 | `git ls-files | wc -l` en baisse ; liens de doc valides | S |
| K-10b | 0 warning dans `Gemma4Swift` (`GPU.*`→`Memory.*`, casts inutiles…), CI minimale (build + `run-tests.sh`), CHANGELOG 1.0→1.7.3 | S-35, S-32, S-34 | CI verte ; `-warnings-as-errors` sur la cible bibliothèque | M |
| K-10c | Docs vraies : macOS 15, 12B multimodal, diffusion hors `recommended()`/`--all`, prévenir h3 (`loadModelContainer` libre) | S-16, S-17, S-18, S-19 | relecture ; `download --all` sans diffusion | S |

### Lot C — Instrument et baseline

| Fiche | Objet | Source | Porte | Effort |
|---|---|---|---|---|
| K-11 | `gemma4-cli bench` : chemin `TokenIterator` de la bibliothèque, ligne JSON (phases, TTFT, méd/p90 par jeton, `phys_footprint`, pic actif, `weights_bw_gbps`, révisions), `--cooldown`, `--trace` ; corriger l'`asyncEval` mort | P-07, measurement §2 | A/A E2B 4 bits : dispersion ≤ 3 % | M |
| K-12 | Baseline §2 (B1→B10 ; 26B/31B si le disque le permet) | audit-performance §4 | tableau §2 rempli, lignes dans `BENCHMARKS.md` | M |

### Lot D — Leviers de performance (chacun A/B/B/A, retiré si < 5 %)

| Fiche | Objet | Source | Porte | Effort |
|---|---|---|---|---|
| K-13 | Préfill tranché (respecter `prefillStepSize`, balayer 256/512/1024/2048) + logits de la seule dernière position | P-01, T9 | pic préfill 4k/8k en baisse ≥ 20 % ou préfill ≥ +5 % ; parité | M |
| K-14 | Politique mémoire : `cacheLimit`/`memoryLimit`/`clearCache` par profil, posée après chargement | P-06, T1-T3 | B9 : footprint 32k ≤ actif + cacheLimit ; temps ±5 % | S |
| K-15 | n-gramme sans synchronisation CPU par jeton | P-08 | B7 : surcoût n=5 divisé par ≥ 2 | S-M |
| K-16 | Résidence : encodeurs vision/audio libérables (`releaseEncodersAfterPrefill`), `noAudioVariant` | P-10, T4-T5 | B11 : −X Go en lean, temps ±5 % | M |
| K-17 | Échantillonnage sur 262 k de vocabulaire (top-p/top-k côté GPU) | P-11 | B8 ≥ +5 % | S |
| K-18 | Vision : fin du padding systématique à 2 520 patches, calcul bf16 | P-03, T17 | B3/B4 encodeur ≥ +20 % ; parité description | M |
| K-19 | Réutilisation de préfixe entre tours hors `continueChat` (snapshot fin de dernier message, caches rotatifs) | P-09, T6-T7 | B10 : > 80 % de jetons réutilisés au tour 2, réponses identiques | L |

### Lot E — Standard des profils

| Fiche | Objet | Source | Porte | Effort |
|---|---|---|---|---|
| K-20 | `Gemma4ReferenceProfile` (`Sources/Gemma4Swift/Configuration/`), `.all`, `named`, `recommended(for:availableMB:)`, `applyGlobalPolicy()`, variantes `textOnly`/`noAudio`/`withMTP` ; `load(_:profile:)` additif ; CLI `references` et `--reference` (refuse un pack aux bits non conformes) | audit-performance §3 | build + tests ; `gemma4-cli references` liste la matrice | M |
| K-21 | Mesure de la matrice E2B/E4B/12B/26B-A4B/31B × 4/8/16 × fast/lean ; pour le 12B, MMLU sur ≥ 1 000 questions (bf16, 8 bits, 6 bits, 4 bits mixte) avant de décider du 6 bits ; `docs/References.md`, `docs/Weights.md` (poids recommandés par profil), `docs/Benchmarks.md`, `docs/knowledge/{index,log,decisions,pitfalls}` | Y `References.md`, standard | table mesurée complète, « Choosing » par classe de machine | L |

### Lot F — 2.0.0 (cassures annoncées, sur décision)

| Fiche | Objet | Source |
|---|---|---|
| K-22 | Code mort (~400 l.), doublons, surface publique réduite (935 déclarations), TurboQuant/diffusion en SPI ou produit séparé, BenchUI hors paquet | S-22, S-23, S-33 |
| K-23 | Collisions de noms avec MLXLLM/MLXVLM ; constructeur de prompt unique (6 aujourd'hui) ; boucles de génération unifiées | S-15, S-20, S-21 |
| K-24 | `Gemma4MTPPipeline` vs `MTPSpeculativeTokenIterator` amont ; 12B dans `chatStreamMultimodal` ; erreurs typées ; prompt système par défaut neutre (aujourd'hui « Tu es un assistant utile. ») | S-25, S-16, S-08, S-12, S-34 |

### Lot G — Serveur d'inférence (détail : audit-annexes-serveur.md §2)
Prérequis : K-1 (annulation), K-9 (gate ABBA). Paquet imbriqué `Server/` recommandé : la bibliothèque garde un graphe sans swift-nio.

| Fiche | Objet | Source | Porte | Effort |
|---|---|---|---|---|
| K-36 | `Gemma4ChatEngine` dans la bibliothèque (aucune dépendance) : messages OpenAI (rôles `tool` compris, parties image/audio), `tools`, `enable_thinking`, profil, événements typés (texte, pensée, appel d'outil, usage, fin), annulable, sous gate K-9 ; `Gemma4Pipeline` relaie `.toolCall` | § 2.2, § 2.3-1 | tests : ids rendus == rendu HF token à token pour 4 cas (texte, image, outils, tour `tool`) ; appel d'outil parsé sur fixture ; annulation en < 1 pas | L |
| K-37 | Paquet imbriqué `Server/` (Hummingbird 2, `swift-nio from: 2.100.0`) : routes v1, SSE + `: loading`, erreurs au format OpenAI, `--reference`, `--models-dir`, `--adapter` ; paquet témoin en CI | § 2.3-2/3, § 2.5 | le témoin résout exactement les 14 paquets actuels (ni hummingbird ni swift-nio) ; client `openai` Python : chat JSON, SSE, image, audio, outils OK | M |
| K-38 | Sécurité : `127.0.0.1` par défaut, hôte non-loopback ⇒ `--api-key` obligatoire, comparaison à temps constant, `/metrics` authentifié sans contenu, limites (32 Mio, 4 médias, 20 Mpx, 30 s d'audio, `max_tokens` du profil, file ≤ 16 → 429), pas de `file://`, pas de fichiers temporaires | § 2.3-7, défauts Q 1-3, 5 | tests d'intégration : 401, 413, 400 (`file://`, média trop grand), 429, refus de démarrer sur `0.0.0.0` sans clé | S-M |
| K-39 | Annulation et sérialisation de bout en bout : déconnexion → arrêt en < 1 pas ; file tenue jusqu'à la fin **réelle** du flux ; aucune route d'entraînement | § 2.3-5/6, défaut Q 4, A-13 | test : client coupé à 5 jetons, la requête suivante a un TTFT ≤ TTFT à vide + 1 pas ; 2 clients concurrents → sorties identiques au séquentiel | S (après K-1, K-9) |
| K-40 | Réutilisation de conversation (LRU par client, `conversation_id` ou préfixe implicite, snapshot en fin de prompt, extension stricte, médias dans la clé, budget en Go) | § 2.3-4, P-09, pièges 13/14/29 | boucle d'agent 4 tours : > 80 % des jetons du tour 2 servis du cache, réponses identiques au re-préfill, TTFT tour 2 −≥ 50 % | L (après K-19) |
| K-41 | *(option)* Lot multi-clients | § 2.3-8 | ≥ ×1,5 agrégé à 8 clients, TTFT p90 ≤ +20 % sur prompts ≤ 256 jetons ; sinon abandon | L |
| K-42 | *(option)* `/v1/messages` (Anthropic) pour Claude Code | wire format Q | session Claude Code de 10 tours sans erreur, cache de préfixe actif (31 k jetons d'outils) | M (après K-40) |

### Lot H — Fonctions annexes : LoRA, drafter MTP, eval (détail : audit-annexes-serveur.md §1)
Porte qualité de référence : ToolsForge ≥ 103/108 (Python 95,3 %, Swift 97,2 %).

| Fiche | Objet | Source | Porte | Effort |
|---|---|---|---|---|
| K-25 | Checkpoints sûrs et reprise : `adapter_config.json` écrit au démarrage, sauvegardes atomiques numérotées + `latest`, état Adam + pas + graine, `--resume`, arrêt propre sur SIGINT / annulation (sauvegarde finale) | A-02 | test : run interrompu au pas 60 → adaptateur chargeable par `loadAdapter` ; reprise 60→120 = loss@120 d'un run continu seedé ±1e-3 | M |
| K-26 | Corrections : écrêtage appliqué (ou option retirée) ; ids directs en multimodal et dans `lora eval` (boucle maison masquée) ; `mtp-train` sans suffixe de génération + `strippingTemplateArtifacts` ; format média aligné sur l'inférence, média sans tour user = erreur ; `--drafter-path` dossier trié ou refusé si ambigu | A-01, A-03, A-04, A-05, A-11 | tests : ids d'entraînement multimodal == ids d'inférence (égalité de tableaux) ; norme globale ≤ max après écrêtage ; `lora eval` == val loss finale de `train` sur le même fichier ±1e-3 | M |
| K-27 | Données et garde-fous : rapport des lignes rejetées (n° + raison), `--max-seq-length` (défaut 2048, troncatures comptées), `--fine-tune-type` strict, famille lue dans `config.json` (12B incluse), `full` refusé sur pack quantifié, mélange et drafter seedés, `precondition`/`fatalError` → `throw`, drafter↔cible vérifiés | A-06, A-07, A-08, A-12 | tests : JSONL à 3 lignes invalides → rapport de 3 ; deux runs seedés de 20 pas → loss identiques bit à bit | S-M |
| K-28 | Instrument d'entraînement : ligne JSON (§ 1.5), débit traité **et** entraîné, `phys_footprint`, `--val-batches` (défaut 25) ; jeux de porte recréés et archivés hors `/tmp` avec SHA-256 (ToolsForge, LaTeX-OCR 500, MMLU ≥ 1 000 questions stratifiées + script versionné) | A-09, A-15, A-21, A-14 | A/A TB2 : dispersion débit ≤ 3 %, loss@200 identiques | M |
| K-29 | Baseline TB1-TB5 (§ 1.5) | § 1.5 | TB1 ≥ 103/108 ; lignes recopiées dans `BENCHMARKS.md` | M |
| K-30 | Leviers (un par comparaison, A/B/B/A sur TB2 puis TB1) : (a) head + CE sur positions de réponse seulement ; (b) politique mémoire (`cacheLimit`, `clearCache` après validation et sauvegarde) ; (c) clés sans k/v sur couches partagées ; (d) forward cible hors VJP (drafter) ; (e) cache d'embeddings média | A-18, A-20, A-22, A-25, A-24 | chacun : pic −≥ 15 % ou débit +≥ 5 %, **et** TB1 ≥ 103/108, loss@200 ±2 % ; sinon retiré | M |
| K-31 | Multimodal sans conversion fp32 (après K-3) : poids bf16, perte et paramètres LoRA en fp32 | A-19 | TB3 : aucun NaN, val loss ≤ 0,378, pic ≤ 14,5 Go (contre 24 Go publiés) | M |
| K-32 | Gradient checkpointing par couche (wrapper `mlx_checkpoint`, local ou amont + `track`) | A-23 | E2B bf16, L = 2048, batch 1 : pic −≥ 30 %, temps ≤ +35 %, loss@50 identique ±1e-3 | M (L si amont) |
| K-33 | `Gemma4TrainingProfile` + `lora train --profile` + mesure de la matrice § 1.4 (E2B, E4B ; 12B/26B/31B selon disque) | § 1.4 | tableau mesuré (pic, débit, porte qualité) pour chaque profil retenu ; un profil non mesuré n'est pas publié | M-L |
| K-34 | `eval-mmlu` : n ≥ 1 000 et IC95 affiché, préfixe 5-shot en cache par sujet, logits de la dernière position, `cacheLimit`, petits défauts (choix > 4, EOS 50, division par zéro, `--verbose`) | A-14, A-26 | réponses identiques sur les 100 questions actuelles ; s/question −≥ 30 % (À MESURER) | M (après K-13/K-19) |
| K-35 | `lora fuse` complet : copie `chat_template.jinja` et `processor_config.json`, `config.json` cohérent avec les poids écrits, erreurs remontées ; vérifier la compatibilité mlx-lm annoncée | A-16 | modèle fusionné rechargé : 32 jetons greedy identiques à base + adaptateur | S |

## 4. Pièges à cocher
Catalogue §4 : 1, 3, 4, 6 (asyncEval mort — déjà rencontré), 7, 8, 9, 12, 13, 14 (positions multimodales, caches rotatifs, KV partagés), 15, 17, 20 (ABBA), 21.

## 5. Décisions

Tranchées par Vincent le 2026-09-27 :
- **26B-A4B et 31B dans la matrice mesurée** : oui (télécharger → mesurer → supprimer si le disque l'exige).
- **`docs/examples/`** : les fichiers de résultats sortent du dépôt (K-10a).
- **2.0.0 (lot F)** : pas maintenant ; on règle d'abord les lots A-E, G, H.
- **Serveur d'inférence** à exposer comme dans Qwen38 (lot G) ; **fonctions annexes** (LoRA, drafter MTP,
  eval) couvertes par le même standard (lot H).

Tranchées le 2026-09-27 (suite) :
- **Serveur** : paquet imbriqué `Server/` (K-37). Périmètre v1 = K-36 à K-40 ; K-41 (batch) et K-42
  (`/v1/messages`) en option.
- **Matrice d'entraînement** : toutes les familles (E2B, E4B, 12B, 26B-A4B, 31B).
- **Porte qualité LoRA** : dataset « director » de Fluxforge Studio
  (`Fluxforge Studio/Scripts/director/dataset/{train,valid}.jsonl`, 898/100, commit `f44da1aa`),
  éval E7 sur 30 briefs tenus à l'écart (`eval_holdout/`, `e7_eval_v3.py`, validation `director-tool`).
  Référence : `director-v3` (E2B 6 bits, rang 8, 16 couches) = **29/30 valides** (base 17/30, professeur 27/30),
  val loss 1,851 → 1,039 (commit `b8cf2932`). Deux tentatives tuées par le garde-fou mémoire (swap ≈ 90 %) :
  cas réel pour K-30/K-31/K-32. À archiver avec SHA-256 (K-28) ; ToolsForge reste secondaire s'il est retrouvé.
  Note : le consommateur entraîne sur **E2B 6 bits**, hors grille 4/8/16 → prévoir `lora-6bit-*` pour E2B.
- **qwen38** : défauts du serveur remontés dans VincentGourbin/qwen38-mlx-swift#2.

Ouvertes :
1. 12B : profil `6bit-*` ou non (MMLU ≥ 1 000 questions, K-21).

## 6. Hors plan
DiffusionGemma (a ses propres préréglages), iOS (pas de cible déclarée), noyau SDPA fusionné pour head_dim 256/512 (P-12, amont MLX).

## 7. Journal

```
## K-x — <titre> — <AAAA-MM-JJ> — validée|bloquée|retirée
- Fait : …
- Mesure : A <…> / B <…> / B <…> / A <…> (dispersion A : x %)
- Porte observée : <ligne recopiée>
- Parité : <ligne recopiée>
```

### Journal

## Lot A — 2026-09-27 — validé (branche `fix/lot-a-stabilite`)
Conditions : un `qwen38` Release (8,2 Go) occupait le GPU toute la soirée → **aucun chiffre de vitesse n'est une référence** ; corrections validées par tests (qui échouent sans le correctif) et essais fonctionnels. Suite finale : 234 tests swift-testing + 115 XCTest verts.

- **K-3** `d96bc8e8` — préfill multimodal sans promotion fp32. Porte : cache KV bf16 (test, échoue sans), description E2B 6 bits identique mot pour mot. Gain de vitesse **non mesuré** (machine occupée).
- **K-1** `bd7c6fff` — streams annulables, `eval` avant traversée. Porte : 3 tests d'intégration verts ; sur l'ancien code, « toujours .processing 3 s après » et l'appel suivant bloqué > 400 s.
- **K-2** `41d432f0` — complétude multi-shards, `--force`, chemin réel, snapshot HF. Constat réel : deux Mistral-Small-3.2-24B du dossier Fluxforge ont 8 shards sur 10 et passaient pour complets.
- **K-5** `bcb0cdda` + `4670b9b3` — `kvBits` et couches KV-partagées (crash « keys (1,1,13,16) » sans le correctif) ; `newCache` ne route plus vers TurboQuant. Le premier commit contenait un test instable (seuil sur l'écart max, 1 échec sur 4) corrigé au suivant. **Reste** : validation sur vrai modèle (E2B/E4B, 26B/31B) et mise à jour de la ligne d'état du README.
- **K-6** `134447b2` — encodeur vision au dtype des poids, `input_proj` quantifié (erreur 19 % → < 5 %). **Reste** : `describe --quantize-bits 4` sur E2B bf16 réel.
- **K-7** `b69f4312` — fabrique privée par appel. Piège : dictionnaire de closures `@Sendable` → `ModelTypeRegistry(creators:)` compile mais plante à l'exécution.
- **K-8** `90cef4b5` — FFT sans pointeurs temporaires ; `Gemma4StreamingDetokenizer`. **Bug amont trouvé** : `NaiveStreamingDetokenizer` (mlx-swift-lm) diffère par graphèmes et perd 🇷, les séquences ZWJ, les accents combinants — touche aussi nos `chatStream`. Gardé en `withKnownIssue`. À remonter (ASK).
- **K-9** `6e12f675` — `Gemma4ComputeGate` non bloquant (3 boucles de gradient exclusives, toutes les entrées d'inférence refusées pendant un entraînement).
- **K-4** `8d8708d8` — MTP au-delà de la fenêtre : texte corrompu dès le 23e caractère (prompt ~1 000 jetons) → IDENTIQUE. Divergence résiduelle préexistante sur quasi-égalités → **K-4b**.

Environnement de test : `~/Library/Caches/models/mlx-community/gemma-4-e2b-it-bf16/` = dossier de liens fichier par fichier vers le Lexar (un lien sur le **dossier** entier n'est pas parcouru par le chargeur mlx-swift-lm) ; drafter `google/gemma-4-E2B-it-assistant` téléchargé dans `~/Library/Caches/models/google/`.

## ASK — Lot A — 2026-09-27
- Remonter le bug `NaiveStreamingDetokenizer` à ml-explore/mlx-swift-lm (issue + éventuelle PR avec le correctif par scalaires) et le suivre avec `track` ?
- Ouvrir une PR `fix/lot-a-stabilite` → `main` maintenant (release 1.8.0 après lots B-E), ou attendre la fin des lots B-E ?

## Lots B, C, E (code) et K-13 — 2026-09-27 — prêts pour la mesure
- **K-8b** `44858e01` — chemins TokenIterator sans le défaut du détokeniseur amont (« 🇫 👩 » → « 🇫🇷 👩‍👩‍👧 »).
- **Lot B** : K-10a `e2dbd142` (1 279 → 289 fichiers suivis, chemins perso retirés), K-10b `bdb57823` (warnings 37 → 2, via le pattern MLX-001), `e89802e6` (CI de compilation, non vérifiée avant la PR), `8bc600c9` (CHANGELOG), K-10c `426293fd`.
- **K-11** `8bc600c9` — `gemma4-cli bench`. **K-13** `2b3f391a` — préfill par tranches (parité fp32 exacte ; réel : même premier jeton, écart 2,1 %/0,6 %). **K-20/K-14** `7bc3d06b` — 30 profils, `references`, `bench --reference`, `load(profile:)`.
- **Campagne** `Scripts/bench-campaign.sh` : A/A puis matrice, poids sur le Lexar (disque interne : 10 Go libres).
- Suite n-gramme : 2 échecs identiques sur `main` (E2B 6 bits) → préexistants, pas de régression.

### Reste à faire avec le GPU libre (dans l'ordre)
1. `python3 ~/.claude/skills/mac-awake/scripts/awake.py run -- Scripts/bench-campaign.sh --cleanup` (A/A ≤ 3 %, puis matrice) → lignes dans `BENCHMARKS.md`, table de `docs/References.md`.
2. Portes chiffrées de K-3 (décodage image), K-13 (pic préfill 4k ≤ 50 %, préfill ≥ +5 %), K-14 (footprint lean) à partir de ces lignes, en A/B/B/A contre `main` quand la porte l'exige.
3. Validations GPU restantes : K-5 `kvBits` sur 26B/31B (couvert par les profils lean), K-6 `describe --quantize-bits 4` sur E2B bf16.
4. Puis leviers conditionnés par la mesure : K-15 (n-gramme), K-17 (échantillonnage), K-18 (vision), K-16 (résidence), K-19 (conversation), lots G et H.

## Mesures — 2026-09-28 — campagne des profils et portes
- Campagne `22902cf7` : 30 profils mesurés (A/A 2,9 %), table dans `docs/References.md`.
- Portes (A/B/B/A contre `main`, E2B 4 bits) : **K-13** préfill 4k ×3,9, TTFT ÷4,6, décodage inchangé ; pic −42 % (porte −50 % manquée de peu). **K-3** décodage image +23 % (cumul K-3 + K-6 + K-13).
- Correction d'une lecture : le préfill du 12B et du 31B n'est pas anormal. 7 à 9 TFLOPS effectifs sur 12B, 26B-A4B et 31B, identique en 4 et 16 bits = borné par le calcul. T14 (déquantifier pour le préfill) écarté par la mesure.
- À corriger : `a4b/*-lean` (tranche 256 = −12 à −22 % de préfill) → tranche 512, à vérifier en A/B.

