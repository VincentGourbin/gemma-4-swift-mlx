# Audit de performance (vitesse et mémoire) — gemma-4-swift-mlx

Date : 2026-09-27 · Dépôt : `/Users/vincent/Developpements/gemma-4-swift-mlx` @ `c9543739` (`main`, tag le plus récent `1.7.2`)
Dépendances résolues (`Package.resolved`) : mlx-swift `0.31.6` (`0bb916c`), mlx-swift-lm `3.31.4` (`bd4b743`).
Audit **en lecture seule** : aucun fichier du dépôt modifié, aucun build, aucun banc. Référentiel : catalogue
`scratchpad/catalogue-techniques.md` (T1-T23 retenues, R1-R16 rejetées, pièges 1-25).

Légende : **VÉRIFIÉ** = lu dans le code (fichier:ligne cités, y compris dans les checkouts des dépendances
sous `.build/xcode/SourcePackages/checkouts/`) ; **À MESURER** = effet chiffré inconnu, protocole fourni ;
**À VÉRIFIER** = hypothèse plausible qu'un test court doit confirmer avant correction. Les chiffres
tirés de la mémoire ou de `BENCHMARKS.md` sont cités comme **indices**, pas comme mesures de référence
(ils ont été pris avec des boucles synchrones, voir P-07).

---

## 0. Carte des chemins d'inférence

| Chemin | Point d'entrée | Préfill | Décodage | Cache KV | Remarque |
|---|---|---|---|---|---|
| Texte, `ChatSession` | `Gemma4Pipeline.chat/chatStream` (`Gemma4Pipeline.swift:325-427`), CLI `generate`/`chat` | `prepare` surchargé → **prompt entier en un forward** (`Gemma4LLMModel.swift:76-85`) | `TokenIterator` amont, `asyncEval` pipeliné (`Evaluate.swift:726-743` amont) | `KVCacheSimple` (plein) + `RotatingKVCache(maxSize: slidingWindow)` (glissant) (`Gemma4LanguageModel.swift:189-199`) | nouvelle `ChatSession` à chaque `chatStream` |
| Texte, contournement (n-gramme / variables de gabarit) | `chatStreamBypassingSession` (`Gemma4Pipeline.swift:438-532`) | idem | `TokenIterator` mais **synchrone** si n-gramme (P-08) | idem, `cache: nil` (l. 496) | pas de réutilisation |
| Image (E2B/E4B/26B/31B) | `chatStreamMultimodal` (`Gemma4Pipeline.swift:578-695`) | `Gemma4MultimodalLLMModel.prepare` → prompt entier (`Gemma4MultimodalLLMModel.swift:253-260`) ; tour vision au 1er forward (l. 137-161) | `TokenIterator` | idem, `cache: nil` (l. 656) | **fuite fp32** (P-02) |
| Image/vidéo/audio CLI | `describe` (`Gemma4CLI.swift:~740-890`) | forward unique | **boucle maison synchrone** `.item()` (l. 857, 882, 886) | `newCache` | chiffres README pris ici |
| Unified 12B | `Gemma4UnifiedMultimodalLLMModel` | prompt entier (`:327-334`) + masque bidirectionnel `[T,T]` matérialisé (`Gemma4TextModel.swift:362-374`) | `TokenIterator` | idem | 2 `.item()` par préfill (`:175-176`) |
| Encodeur texte LTX | `forwardCollectingHiddenStates` (`Gemma4TextModel.swift:290-306`) | forward unique, 49 tenseurs retenus | — | aucun | pas de logits (bien) |
| MTP | `Gemma4MTPPipeline.runLoop` (`Gemma4MTPPipeline.swift:104-303`) | forward unique **+ logits complets évalués** (l. 161-162) | boucle synchrone draft→verify→walk | Rotating + Simple, `trimPromptCache` | **rollback cassé après 512 jetons** (P-04) |
| Profil CLI | `profile run/sweep` (`ProfileCommand.swift:150-190, 400-420`) | forward unique | `asyncEval` **code mort** (l. 185-186) | `newCache(kvBits)` → TurboQuant | source des chiffres `BENCHMARKS.md` |
| Diffusion | `DiffusionGemmaPipeline` | à part | à part | `EncoderKVCache` | seul chemin avec `cacheLimit`/`clearCache` (hors périmètre principal) |

Faits d'architecture utiles (VÉRIFIÉ, `config.json` de `gemma-4-e2b-it-bf16` sur `/Volumes/Lexar/models/mlx-community/`) :
vocabulaire **262 144**, `tie_word_embeddings = true`, `hidden 1536`, 35 couches dont 7 pleines
(indices 4, 9, …, 34), `head_dim 256` (glissant) / `global_head_dim 512` (plein), `sliding_window 512`,
20 couches KV-partagées, `final_logit_softcapping 30`, `embed_tokens_per_layer` = 262 144 × (35 × 256) =
**2,35 G paramètres** (le plus gros tenseur d'E2B). Aucun autre checkpoint Gemma n'est présent localement
(E4B/12B/26B/31B : valeurs équivalentes **À VÉRIFIER** dans leur `config.json`).

Fait MLX déterminant (VÉRIFIÉ, `mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/scaled_dot_product_attention.cpp:619-636`) :
le noyau SDPA fusionné « full » (préfill, L > 8) n'accepte que `head_dim ∈ {64, 80, 128}` ; le noyau
« vector » (L ≤ 8) `{64, 96, 128, 256}`. Gemma 4 (256 / 512) tombe donc **en repli non fusionné pour toutes
les couches au préfill**, et pour les couches pleines (512) **aussi au décodage** : les scores
`[B, H, L, S]` sont matérialisés. C'est ce qui rend le préfill non tranché (P-01) coûteux en mémoire.

---

## 1. Grille du catalogue (T1-T23, rejets, pièges)

| # | Technique | Statut Gemma | Preuve / justification |
|---|---|---|---|
| T1 | `Memory.cacheLimit` par étape | **Absente** (inférence) | Aucun `cacheLimit`/`memoryLimit` hors diffusion (`DiffusionGemmaCommand.swift:150-152`, `ProfileDiffusionCommand.swift:149-151`). L'amont n'en pose pas au chargement (grep vide dans mlx-swift-lm). → P-06 |
| T2 | Limites adaptatives (dispo/6, dispo−2 Go) | **Absente** | idem ; `Model.recommendedRAMGB` (`Gemma4Pipeline.swift:203-205`) n'est qu'informatif. → P-06 |
| T3 | `clearCache()` après réponse / entre étapes | **Partielle** | seulement `unload()` (`Gemma4Pipeline.swift:319`), `ProfileSweep` (`ProfileCommand.swift:478`), BenchUI, diffusion. Rien après une réponse ni après l'encodeur. → P-06 |
| T4 | Résidence par étape (encodeurs / drafter) | **Absente** (sauf diffusion) | Tours vision+audio toujours chargés si `multimodal: true`, défaut de `Pipeline.load` (`:255, :283`) ; précédent interne : `DiffusionGemmaEncoderModel.unloadVision()` (`:55-59`). → P-10 |
| T5 | Variante texte seul | **Partielle** | `load(multimodal: false)` instancie `Gemma4LLMModel` et le sanitizer écarte les tours (`Gemma4Registration.swift:38-46`). Pas de variante « sans audio », pas de refus explicite des images en mode texte (le `as?` échoue en `unsupportedModelFamily`, `Gemma4Pipeline.swift:634-639`). → P-10 |
| T6 | Réutilisation de conversation (snapshot KV) | **Partielle** | `ChatSession` amont garde le KV entre `continueChat` (`ChatSession.swift:615-627` amont) ; mais `chatStream` recrée une session à chaque appel (`Gemma4Pipeline.swift:402`), les chemins multimodal/contournement/MTP partent de `cache: nil` / `makeCache()` ; `RotatingKVCache` non trimmable au-delà de 512 ; masque Unified à offset 0 (`Gemma4TextModel.swift:366-367`). → P-09 |
| T7 | Seuls les nouveaux médias ré-encodés | **Absente** | conséquence de T6 ; `pendingPixelValues` consommé une fois par forward. → P-09 |
| T8 | Budget d'image exposé | **Partielle** | `maxSoftTokens` existe côté processeur (`ImageProcessor.swift:16, 35`), mais `VisionModel` rend toujours 280 (`VisionModel.swift:119`) et `multimodalChatIds` développe 280 (`Gemma4Processor.swift:75, 195`) ; la vidéo tronque à 70 (`Gemma4MultimodalLLMModel.swift:174`). Pas de redimensionnement implicite type `ChatSession` 512×512 (chemin multimodal hors `ChatSession`). → P-03 |
| T9 | Tranche de préfill + logits de la dernière position seulement | **Absente** | les 3 surcharges `prepare` rendent le prompt entier (`Gemma4LLMModel.swift:76-85`, `Gemma4MultimodalLLMModel.swift:253-260`, `Gemma4UnifiedMultimodalLLMModel.swift:327-334`) : `prefillStepSize` (512 par défaut, `Evaluate.swift:134` amont) est **ignoré** ; le head s'applique à toutes les positions (`Gemma4LanguageModel.swift:51`). → P-01 |
| T10 | KV 8 bits (lean) | **Absente / cassée** | TurboQuant désactivé sur E2B/E4B/12B par l'heuristique (`Gemma4LanguageModel.swift:133-169`) ; le KV natif (`GenerateParameters.kvBits`, recommandé par `README.md:865-874`) **casse les couches KV-partagées**. → P-05 |
| T11 | KV préalloué, écriture en place | **Déjà appliquée (amont)** | `KVCacheSimple` : pas de 256, `zeros` + écriture en place (`KVCache.swift:375-412` amont). Pas de cache maison hors TurboQuant/diffusion. Rien à faire. |
| T12 | Head quantifié via `quantizedMatmul`, jamais dé-quantifié | **Appliquée** | head lié : `embedTokens.asLinear` (`Gemma4LanguageModel.swift:51`) → `QuantizedEmbedding.asLinear` = `quantizedMM` (`mlx-swift/Source/MLXNN/Quantized.swift:213-217`). Drafter : `MaskedEmbedder` (sparse). Mais le head tourne sur **toutes** les positions au préfill (P-01). |
| T13 | Quantification mixte par voie | **Partielle** | les packs 12B-4bit mlx-community sont déjà mixtes (4 bits attention / 8 bits MLP, `memory/project_benchmarks_12b.md`) ; OTF uniforme seulement (`Gemma4OnTheFlyQuantization.swift:62-122`). Candidat : `embed_tokens_per_layer` (2,35 G param. sur E2B) plus bas que le reste. → P-13 |
| T14 | Dé-quantifier vers bf16 pour une étape bornée calcul | **Absente** | aucun chemin ; candidat préfill long / encodeurs en `fast` seulement. À MESURER après P-01 (sinon le repli SDPA domine). Priorité basse. |
| T15 | `asyncEval` pipeliné | **Appliquée dans la bibliothèque, morte ailleurs** | `TokenIterator` amont OK ; **code mort** dans `ProfileCommand.swift:185-186` (`asyncEval(token)` puis `.item()` immédiat, piège 6) ; `describe` synchrone (`Gemma4CLI.swift:882, 886`) ; MTP synchrone ; n-gramme casse le pipeline. → P-07, P-08, P-04 |
| T16 | `eval` par couche dans les longues boucles | **N/A inférence** / partielle diffusion | pas de boucle de graphe différé multi-couches hors diffusion (`evalEveryNLayers` en diffusion). À garder en tête pour l'encodeur audio (12 blocs, un seul `eval`). |
| T17 | Fuites de dtype fp32 | **Présentes** | `MLXArray(embedScale, dtype: .float32)` au préfill multimodal (`Gemma4MultimodalLLMModel.swift:116`, `Gemma4UnifiedMultimodalLLMModel.swift:209`) ; vision (`VisionPatchEmbedder.swift:40`) ; audio (`AudioAttention.swift:102-104`). Le chemin diffusion, écrit plus tard, fait bien `dtype: inputsEmbeds.dtype` (`DiffusionGemmaEncoderModel.swift:99`). → P-02, P-03, P-17 |
| T18 | `compile(shapeless:)` d'une activation élémentaire | **Déjà (amont)** / candidat mineur | `geluApproximate` compilé côté MLXNN (cf. CLAUDE.md, deadlock ABBA) ; seul compile maison : entropie diffusion (`EntropyBoundSampler.swift:70`). Candidat mineur : softcap `tanh(x/c)*c` sur 262 k logits (`Gemma4LanguageModel.swift:54-56`) — gain attendu < 5 %, non prioritaire. |
| T19 | Réutilisation d'un cache entre étapes | **Appliquée (MTP)** | le drafter lit le KV cible (`Gemma4MTPPipeline.swift:194-201`). |
| T20 | Politique de calcul selon borne BP / calcul | **Absente** | pas de distinction préfill/décodage ; à décider avec P-01/T14. |
| T21 | Nombre de pas (analogue : taille de bloc MTP) | **Partielle** | `--block-size` exposé (`Gemma4CLI.swift` Generate), défaut 4, jamais balayé avec taux d'acceptation publié. À MESURER après P-04. |
| T22 | `pread` / `F_NOCACHE` | **N/A (faible)** | poids 3,6-5 Go résidents ; seule la mise en garde « mesures à chaud » s'applique (poids sur disque externe Lexar). |
| T23 | Reprise / porte GPU iOS | **N/A pour l'instant** | `Package.swift:6` déclare `.iOS(.v17)` : le profil `lean` doit prévoir `os_proc_available_memory` ; pas de porte GPU (piège 22). |
| R1-R3 | compile du pas de décodage / par couche | **À ne pas tenter** | décodage Gemma borné dispatch (35-60 couches, repli SDPA 512) : même diagnostic que Y/Q. |
| R4 | Hadamard partagé | cohérent | TurboQuant utilise une rotation dense (README l. 874) : même famille de coût, README recommande déjà de l'abandonner. |
| R12, R16 | `MLX_MAX_OPS_PER_BUFFER`, `sysctl iogpu.wired_limit_mb` | **À ne pas tenter** | sans effet mesuré. Le levier câblé propre est l'API `wiredMemoryTicket` amont (`Evaluate.swift:534` amont), non utilisée (P-14). |

Pièges du catalogue pertinents, vérifiés ici : **2** (`asType(weight.dtype)` sur un `QuantizedLinear` —
`VisionPatchEmbedder.swift:60`, P-03), **6** (`asyncEval` mort — P-07), **7** (pas de `cacheLimit` — P-06),
**8** (`gemma4-cli` lui-même a perturbé Y et Q : l'inverse vaut ici), **9** (Release), **12** (repli
silencieux de pack : le futur `--reference` doit refuser un `--model-path` dont les bits ne correspondent pas),
**13-14** (gabarit / état positionnel — P-09), **15** (redimensionnement implicite : absent ici), **17**
(`memoryLimit` n'est pas un plafond), **20** (ABBA : jamais d'entraînement LoRA/drafter pendant un banc).

---

## 2. Constats

Ordre = priorité proposée (correction d'abord, puis gain attendu / effort). Seuil de rétention : **5 %**
(catalogue §2.1) ; toute mesure en A/B/B/A, binaire Release, machine au repos, refroidissement 120 s.

### P-01 — Préfill non tranché et logits calculés sur toute la séquence
- **Constat (VÉRIFIÉ)** : les trois wrappers surchargent `prepare` pour rendre le prompt entier
  (`Gemma4LLMModel.swift:76-85`, `Gemma4MultimodalLLMModel.swift:253-260`,
  `Gemma4UnifiedMultimodalLLMModel.swift:327-334`) au lieu du découpage par défaut de `LLMModel`
  (`MLXLLM/LLMModel.swift`, boucle `while y.tokens.size > prefillStepSize` + `eval(cache)`). Le
  `TokenIterator` fait alors `step(previous: prompt entier)` (`Evaluate.swift:675-688` amont) ; le head
  lié s'applique à **toutes** les positions (`Gemma4LanguageModel.swift:51`) avant le `logits[:, -1]`
  (`Evaluate.swift:697`) : MLX ne pousse pas le slice à travers le matmul. Même schéma dans MTP
  (`Gemma4MTPPipeline.swift:161-162`, `eval(prefillOut.logits…)` **matérialise** les logits),
  `profile` (`ProfileCommand.swift:158-160`), `describe` (`Gemma4CLI.swift:857`). S'y ajoute le repli SDPA
  (§0) : scores `[H, L, L]` matérialisés à chaque couche.
- **Ordre de grandeur (calculé, VÉRIFIÉ sur la config E2B)** : logits d'un prompt de 4 000 jetons =
  4 000 × 262 144 × 2 o = **2,1 Go bf16**, et 4 000 × 1 536 × 262 144 ≈ 1,6 T MAC de head inutiles
  (head ≈ 403 M param., soit ~17 % des 2,3 G paramètres effectifs par jeton). Scores d'une couche pleine
  (8 têtes) à L = 4 000 : 8 × 16 M × 2 o ≈ 256 Mo par couche, contre ≈ 33 Mo avec des tranches de 512.
- **Gain attendu** : pic de préfill fortement réduit (T9 : Q 512 → 16,1 Go contre 52,5 Go à 4 096) ;
  préfill plus rapide (Q : plus petit = plus rapide ET plus léger). Montant **À MESURER**.
- **Source catalogue** : T9 (tranche + `keepLastOnly`), T1.
- **Correction** : (1) texte : réimplémenter le découpage dans `prepare` (tranches de `windowSize`,
  `asyncEval(cache)`), en ne laissant **qu'un jeton** au `TokenIterator` (le head ne tourne que sur 1
  position) ; (2) multimodal : calculer `inputsEmbeds` (tours + `maskedScatter`) et `perLayerInputs` une
  fois, puis faire avancer `languageModel.model` par tranches d'embeddings ; (3) Unified : tranches alignées
  sur les bornes de blocs vision, et `createCausalMask(n: T, offset: 0)` (`Gemma4TextModel.swift:366-367`)
  doit prendre l'offset du cache ; (4) MTP : ne garder que `preNorm[-1]` et appliquer le head sur la seule
  dernière position ; (5) `forwardWithIntermediates` inchangé pour la vérif MTP (bs positions nécessaires).
- **Mesure** : banc B1 (128 / 1k / 4k / 8k jetons, E2B 4 bits puis 12B 8 bits), balayage
  `prefillStepSize` 256 / 512 / 1024 / 2048 en A/B/B/A (ordre 512, 2048, 256, 1024, 1024, 256, 2048, 512).
- **Porte** : à 4k jetons, pic MLX actif du préfill **≤ 50 %** de la base et préfill tok/s **≥ base + 5 %** ;
  décodage inchangé (± 3 %) ; parité : premiers 64 jetons greedy identiques sur ≥ 7/8 prompts, écart max
  des logits du dernier jeton rel. < 1e-2 (le découpage change l'ordre des réductions bf16).
- **Risque** : moyen (Unified : blocs bidirectionnels ; KV-partage : les couches partagées lisent le cache de
  la couche source — compatible tranches, à tester) ; ne pas casser `forwardCollectingHiddenStates` (LTX).
- **Effort** : M.

### P-02 — Fuite fp32 au préfill multimodal, propagée au cache KV et au décodage
- **Constat (VÉRIFIÉ code ; dtype effectif À VÉRIFIER en run)** :
  `inputsEmbeds * MLXArray(embedScale, dtype: .float32)` (`Gemma4MultimodalLLMModel.swift:116`,
  `Gemma4UnifiedMultimodalLLMModel.swift:209`, et le code apparemment mort `Multimodal/Gemma4Model.swift:47`)
  promeut les embeddings bf16 en fp32 ; tout le préfill multimodal passe en fp32 (Linear/`quantizedMM`
  promeuvent). `KVCacheSimple` alloue son tampon au dtype des premières clés (`KVCache.swift:403-404`
  amont) → **cache fp32** ; au décodage, SDPA(q bf16, K fp32) promeut la sortie en fp32 et le flux résiduel
  reste fp32 pour toute la génération. Le chemin texte fait `dtype: h.dtype` (`Gemma4TextModel.swift:326`),
  la diffusion aussi (`DiffusionGemmaEncoderModel.swift:99`).
- **Indices** (non référence) : Unified 4 bits 20,8 t/s texte contre 12-13 t/s avec une image de 260
  jetons (`memory/project_gemma4_unified.md`) alors que `BENCHMARKS.md` §2 montre un décodage quasi plat
  entre 28 et 127 jetons de contexte ; E2B vidéo 50,8 t/s contre 97 t/s texte (README).
- **Gain attendu** : décodage après média ramené près du texte à contexte égal (jusqu'à ~×1,5-2 d'après
  les indices) ; KV ÷ 2 ; préfill multimodal plus rapide. **À MESURER**.
- **Source** : T17 (Q : ×2,74 en décodage réel).
- **Correction** : `MLXArray(embedScale, dtype: inputsEmbeds.dtype)` aux deux endroits ; supprimer ou aligner
  `Gemma4Model.swift` ; ajouter une assertion de test « dtype du cache == dtype des poids après préfill image ».
- **Mesure** : B3 (1 image + prompt court, E2B 4 bits et 12B 8 bits) contre un prompt texte de même
  longueur ; relever `cache[0].state[0].dtype`.
- **Porte** : décodage après image **≥ 90 %** du décodage texte à contexte égal ; parité qualité : sortie
  greedy comparée à mlx-vlm Python (même image, 64 jetons) et test OCR du 12B (`BENCHMARKS.md` §7) inchangé.
- **Risque** : faible à moyen (numérique : la référence Python ne promeut vraisemblablement pas — À VÉRIFIER
  dans `mlx_vlm/models/gemma4`) ; rupture d'ids nulle mais sorties greedy différentes → à annoncer à
  LTX/Fluxforge (Fluxforge suit `main`).
- **Effort** : S.

### P-03 — Encodeur vision : padding systématique à 2 520 patches, calcul fp32, piège `uint32`
- **Constat (VÉRIFIÉ)** : `VisionModel` complète toujours à `maxPatches = 280 × 3 × 3 = 2 520`
  (`Gemma4VisionConfig.swift:54-56`, `VisionModel.swift:96-97`) et construit un masque `[B,1,2520,2520]`
  (`:102-106`) ; une frame vidéo à 70 jetons (630 patches réels) coûte donc **×4 jetons, ×16 sur la partie
  attention** ; les frames passent une par une (`Gemma4MultimodalLLMModel.swift:169-176`).
  `MLX.where(mask, MLXArray(Float(0.0)), posEmb)` (`VisionPatchEmbedder.swift:40`) promeut en fp32 → tout
  l'encodeur tourne en fp32 (les `VisionRMSNorm` rendent `x.dtype`, `VisionNorms.swift:22, 38`).
  `patches.asType(inputProj.weight.dtype)` (`VisionPatchEmbedder.swift:60`) : si `input_proj` est quantifié,
  `weight` est `uint32` (piège 2) → entrée détruite. Or la quantification OTF par défaut **n'exclut rien**
  (`Gemma4OnTheFlyQuantization.swift:36`, contrairement à ce que dit `memory/project_benchmarks_12b.md`).
- **Gain attendu** : encodeur image ÷ 1 à 2 (fp32 → bf16), vidéo jusqu'à ÷ 4-10 (padding). **À MESURER**.
- **Source** : T17, T8 ; piège 2.
- **Correction** : n'encoder que les patches réels (le masque exclut déjà le padding : résultat identique
  aux arrondis près), `MLXArray(0, dtype: posEmb.dtype)`, `asType(computeDType)` explicite (dtype des
  `scales` si quantifié) ; batcher les frames vidéo (même taille) ; exposer `imageSoftTokens` (280/140/70)
  en réutilisant le mécanisme vidéo.
- **Mesure** : B3/B4 — temps de l'encodeur par image et par frame (phase dédiée du profiler), pic.
- **Porte** : encodeur vidéo **≥ ×2** plus rapide ; écart des features rel. < 1e-2 contre la version paddée ;
  descriptions greedy identiques sur l'image de référence ; `describe --quantize-bits 4` (sans
  `--quantize-text-only`) sur E2B produit une description cohérente (aujourd'hui **À VÉRIFIER** : bug
  probable).
- **Risque** : moyen (le pooler suppose 2 520 positions — à adapter) ; **Effort** : M.

### P-04 — MTP : rollback du cache inopérant au-delà de la fenêtre glissante, boucle synchrone
- **Constat (VÉRIFIÉ, logique)** : le cache MTP contient des `RotatingKVCache(maxSize: 512)`
  (`Gemma4LanguageModel.swift:197`) ; `RotatingKVCache.isTrimmable` vaut `offset < maxCacheSize`
  (`KVCache.swift:717-719` amont) et `trimPromptCache` ne fait **rien** si une seule couche n'est pas
  trimmable (`KVCache.swift:1871-1875` amont). Dès que le contexte atteint 512 jetons (prompt long ou
  génération longue), `Gemma4MTPPipeline.swift:286-289` ne retire plus les brouillons rejetés, **y compris
  dans les caches pleins** → le KV contient des jetons faux : la promesse « bit-exact » du README ne tient
  plus (acceptation et sortie faussées). Côté perf : logits complets du préfill évalués (l. 161-162) ;
  par round `eval(drafts)` + (bs−1) `.item()` (l. 209-216), `eval` des logits `[1, bs, 262 144]`
  (l. 248), puis `argMax` + `eval` + bs `.item()` (l. 252-258) ; aucun recouvrement CPU/GPU ; le chat MTP
  CLI re-préremplit tout l'historique à chaque tour (`Gemma4CLI.swift:544-549`).
- **Gain attendu** : correction d'abord ; ensuite ~2 synchronisations par round au lieu de ~2·bs+2,
  logits jamais matérialisés. **À MESURER**.
- **Source** : T15 (piège 6), T19, T21 ; piège 13-14.
- **Correction** : rollback sûr pour les couches glissantes (snapshot `state`/`metaState` avant la vérif,
  ou cache glissant non rotatif + masque de fenêtre pendant MTP, ou reprise du schéma de
  `MTPSpeculativeTokenIterator` amont — `MTPSpeculativeTokenIterator.swift` — à lire) ; `argMax` dans le même
  graphe que la vérif, un seul `eval` de `(targets, preNorm)`, `asArray` au lieu de `.item()` en boucle ;
  head sur la dernière position au préfill (P-01).
- **Mesure** : B6 — `mtp-generate --compare` sur prompt de 100 et de 700 jetons, 256 jetons générés ;
  tok/s MTP contre greedy standard (`TokenIterator`).
- **Porte** : **bit-exact** sur les deux longueurs (bloquant) ; puis tok/s MTP ≥ base MTP + 5 % ; profil
  `+mtp` retenu seulement si tok/s MTP ≥ 1,05 × greedy standard sur le corpus de référence.
- **Risque** : moyen ; **Effort** : M.

### P-05 — Quantification KV native incompatible avec le KV-partage (E2B/E4B)
- **Constat (VÉRIFIÉ, logique)** : avec `GenerateParameters(kvBits:)` (`ChatSession`, `generate`), le
  `TokenIterator` convertit les `KVCacheSimple` en `QuantizedKVCache` (`maybeQuantizeKVCache`,
  `KVCache.swift:2001-…` amont). Les couches KV-partagées lisent alors `cache.state[0]`, `state[1]`
  comme clés/valeurs (`Gemma4Attention.swift:181-192`) alors que `QuantizedKVCache.state` rend
  `[kq, kscales, kbiases, vq, …]` (`KVCache.swift:1011-1027` amont) → attention sur les poids packés et
  les échelles. Même lecture dans MTP (`Gemma4MTPPipeline.swift:315-322`). Le README recommande
  précisément ce réglage (`README.md:865-874`). Sur 12B/26B/31B (sans KV-partage), les couches pleines
  passeraient en `quantizedScaledDotProductAttention` via `attentionWithCacheUpdate` (OK a priori) ; les
  `RotatingKVCache` ne sont pas convertis (sinon `fatalError`, `KVCache.swift:785-790` amont — non atteint).
- **Gain attendu** : débloque T10 (Q : lean 12,2 Go contre 16,1 Go à 32 k). **À MESURER** ; sur E2B/E4B
  le KV plein est petit (1 tête KV) — le gain y sera faible, il compte surtout pour 26B/31B (GQA 4).
- **Correction** : dans la branche partagée, détecter `QuantizedKVCacheProtocol` et appeler
  `quantizedScaledDotProductAttention` sur les tuples quantifiés ; corriger le README ; test d'intégration
  « kvBits 8 = sortie cohérente sur E2B ».
- **Mesure / porte** : B9 bis — greedy 64 jetons KV8 contre KV bf16 identique sur ≥ 7/8 prompts ;
  pic à 8 k / 32 k (31B : **≥ 10 %** de pic en moins pour retenir le champ `kvBits: 8` en lean).
- **Risque** : moyen ; **Effort** : S-M.

### P-06 — Aucune politique mémoire MLX sur les chemins d'inférence
- **Constat (VÉRIFIÉ)** : pas de `Memory.cacheLimit`, `memoryLimit`, ni `clearCache` après réponse
  (voir T1-T3). Avec préfill non tranché (P-01) et tampons de tailles variables, le cache de l'allocateur
  grossit sans borne (catalogue : 74 Go de `phys_footprint` pour ≈ 8,4 Go actifs sur Bonsai 2).
- **Gain attendu** : `phys_footprint` ≈ actif + limite ; indispensable pour `lean`/iOS. **À MESURER**.
- **Source** : T1, T2, T3 ; pièges 7, 17.
- **Correction** : `Gemma4ReferenceProfile.applyGlobalPolicy()` (après chargement) :
  fast = `cacheLimit` 4 Go (Mac), lean = `min(1 Go, max(256 Mo, dispo/6))` et `memoryLimit =
  max(4 Go, dispo − 2 Go)`, `clearCache` après réponse en lean ; `availableMemoryMB()` iOS
  (`os_proc_available_memory`) / macOS (physique − 8 Go) / override `GEMMA4_AVAILABLE_MB`.
- **Mesure** : B9 — E2B 4 bits, contextes 1k/8k/32k, `phys_footprint` (TASK_VM_INFO) et pic MLX actif,
  sans puis avec limite ; `GEMMA4_AVAILABLE_MB=16384` pour simuler un Mac 16 Go.
- **Porte** : pic `phys_footprint` ≤ pic actif + `cacheLimit` + 10 % ; tok/s dans ± 5 % de la base (sinon
  la limite est trop basse : thrash, cf. R13).
- **Risque** : faible (défauts inchangés tant qu'aucun profil n'est appliqué) ; **Effort** : S.

### P-07 — Instruments de banc biaisés : `asyncEval` mort, boucles synchrones hors bibliothèque
- **Constat (VÉRIFIÉ)** : `ProfileCommand.swift:185-186` (et 161-162, 407-408, 418-419) fait
  `asyncEval(token)` puis `token.item()` immédiatement : aucun recouvrement (piège 6). `describe` (chemins E2B/E4B et Unified `runUnified`) décode en synchrone (`Gemma4CLI.swift:857-887, 1106-1130`). Tous les chiffres de
  `BENCHMARKS.md` §1-3 et du README (vision/vidéo/audio) viennent de ces boucles, pas du `TokenIterator`
  qu'utilisent l'API et les consommateurs. Pas de `phys_footprint`, seulement `GPU.peakMemory`.
- **Conséquence** : la base de référence doit être reprise avec un instrument aligné sur la bibliothèque
  avant tout A/B (sinon on optimise un chemin que personne n'emprunte).
- **Correction** : sous-commande `gemma4-cli bench` (ou `profile run --library-path`) qui passe par
  `TokenIterator`/`Gemma4Pipeline`, découpe les phases (chargement, encodeur, préfill, décodage), et écrit
  **une ligne JSON** par mesure (modèle, pack, profil, `prompt_tokens`, `prefill_ms`, `ttft_ms`,
  décodage médiane/p90 ms, tok/s, pic MLX actif, pic `phys_footprint`, `cache_mb`, `weights_gb`,
  `weights_bw_gbps`, commit, révisions mlx-swift/mlx-swift-lm) ; sorties sous `.local-runs/bench.noindex/`.
- **Porte** : tok/s `bench` et `ChatSession` (même prompt, greedy) à ± 3 % ; l'ancien `profile run` est
  alors re-mesuré une fois pour chiffrer le biais historique (information, pas porte).
- **Risque** : nul ; **Effort** : S-M. **Prérequis de toutes les autres mesures.**

### P-08 — Processeur n-gramme : synchronisation GPU→CPU à chaque jeton
- **Constat (VÉRIFIÉ, documenté dans le code)** : `didSample` fait `asArray` (`NoRepeatNGramLogitProcessor.swift:116-124`),
  ce qui force l'évaluation du jeton dans `convertToToken` avant le `asyncEval` du pas suivant : le
  pipelining du `TokenIterator` est désactivé (`:30-38` le reconnaît). LTX l'utilise en greedy avec n = 5.
- **Gain attendu** : Q a mesuré −13 à −22 % ms/pas en rétablissant le recouvrement (T15). **À MESURER** ici.
- **Correction** : historique côté GPU (tableau `MLXArray` des derniers jetons, comparaison vectorisée des
  fenêtres de taille n−1, `-inf` posé par un `putAlong` à nombre d'indices fixe) — l'automate de canal de
  pensée reste à porter (état sur GPU ou décision CPU différée d'un pas, à étudier).
- **Mesure** : B7 — E4B 4 bits, greedy 256 jetons, n-gramme 5 contre sans.
- **Porte** : tok/s ≥ base + 5 % ; `NoRepeatNGramTests` + `NoRepeatNGramIntegrationTests` verts ; captions
  LTX de référence identiques octet pour octet.
- **Risque** : moyen (contrat HF + canal de pensée) ; **Effort** : M. Gain réservé aux consommateurs n-gramme.

### P-09 — Pas de réutilisation de préfixe entre requêtes / tours (hors `continueChat`)
- **Constat (VÉRIFIÉ)** : voir T6/T7. `ChatSession` rend au tour suivant les seuls nouveaux messages sur un
  KV qui ne contient pas le `<turn|>` de fin (le jeton EOS n'est jamais réinjecté) : frontière de gabarit
  potentiellement différente du rendu complet (piège 13) — **À VÉRIFIER** par parité « tour 2 réutilisé »
  contre « historique complet re-préfillé ». Unified : masque bidirectionnel à offset 0 incompatible avec
  un cache non vide.
- **Gain attendu** : Q K-6 : préfill 84-96 s contre 145-253 s sur 4 tours, 47 % de jetons réutilisés.
  Pour Fluxforge (captions répétées, même système) le préfixe commun est court : gain probablement faible
  (**À MESURER** avant d'investir).
- **Correction** : moteur de snapshot à la Q (fin du dernier message, `trim`, vérif des offsets), avec
  couches glissantes non trimmables au-delà de 512 → snapshot d'état plutôt que `trim` ; nouveaux médias
  seulement (T7).
- **Mesure** : B10 — rejeu 4 tours (texte puis avec image au tour 3).
- **Porte** : ≥ 80 % de jetons réutilisés dès le tour 2, TTFT ÷ 3 au tour 4, réponses identiques au jeton.
- **Risque** : élevé (KV-partage, fenêtre glissante, MTP) ; **Effort** : L. À planifier après P-01/P-04.

### P-10 — Encodeurs vision/audio toujours résidents
- **Constat (VÉRIFIÉ)** : `Pipeline.load(multimodal: true)` par défaut ; tour audio chargée même sans
  audio ; aucune libération après préfill. Taille **estimée** pour E2B bf16 (à partir de la config :
  vision 16 × 768 ≈ 0,17 G param., audio 12 blocs Conformer 1 024 ≈ 0,3 G param.) : ≈ 0,9 Go bf16 —
  **À MESURER** (delta `Memory.activeMemory` après `eval`, piège 18).
- **Source** : T4, T5.
- **Correction** : variantes `textOnlyVariant()` (existe via `multimodal: false`) et `noAudioVariant()`
  (clés `audio_tower.*`/`embed_audio.*` retirées avant `verify`, piège 16) ; en lean, libération des tours
  après l'encodage (modèle `unloadVision()` de la diffusion) avec rechargement paresseux à la prochaine
  requête média.
- **Mesure / porte** : B11 — gain ≥ 0,3 Go (E2B) pour retenir une variante ; aucune différence de sortie
  texte ; temps de rechargement documenté.
- **Risque** : moyen (calcul silencieux sur zéros si mal gardé) ; **Effort** : M.

### P-11 — Échantillonnage par défaut coûteux sur 262 k de vocabulaire
- **Constat (VÉRIFIÉ)** : défauts `temperature 0.3, topP 0.95` (`Gemma4Pipeline.swift:335, 401, 488, 644`)
  → `TopPSampler` amont : cast fp32 + `logSoftmax` + tri sur 262 144 logits par jeton (`Evaluate.swift:257-270` amont).
- **Gain attendu** : inconnu (quelques % à 10 % du pas sur E2B ?). **À MESURER** (B8 : temp 0 contre 0,3/0,95).
- **Correction** (si > 5 %) : pré-filtre `topK` (p. ex. 64) avant `topP`, ou ne rien faire et documenter.
- **Porte** : ≥ 5 % tok/s, distribution de sortie inchangée (test statistique simple sur 1 000 tirages).
- **Risque** : faible ; **Effort** : S.

### P-12 — Repli SDPA non fusionné (head_dim 256/512)
- **Constat (VÉRIFIÉ)** : §0. Structurel, non corrigeable dans ce dépôt ; atténué par P-01. À noter dans
  les profils (tranche de préfill plus petite en lean) et comme demande amont (noyau `sdpa_full` 256).
- **Effort** : — (suivi amont).

### P-13 — Choix de pack et quantification mixte
- **Constat** : (VÉRIFIÉ, `BENCHMARKS.md` §4) 12B : 4 bits = 32-37 % MMLU contre 57 % bf16 et 58 % 8 bits ;
  26B-A4B 4 bits −6 pts contre Python (issue #27) ; 31B 4 bits = parité Python. E2B/E4B : `embed_tokens_per_layer`
  est le plus gros tenseur (VÉRIFIÉ, config E2B) — candidat à une quantification plus basse que le reste (T13).
- **Correction** : dans les profils, 12B `4bit-*` marqués « qualité dégradée » (et non recommandés pour
  l'encodeur LTX) ; essai « 8 bits + PLE 4 bits » en OTF.
- **Porte** : poids −≥ 15 %, MMLU plain ≥ −1 pt contre 8 bits pur, parité greedy 7/8. **Effort** : M.

### P-14 — Mémoire câblée non utilisée (26B/31B)
- **Constat (VÉRIFIÉ)** : l'API `wiredMemoryTicket` de `generate` amont n'est pas utilisée. Seul levier
  « câblé » propre (R16 : le `sysctl` seul est sans effet). Pertinent quand les poids approchent le working
  set recommandé (31B 8 bits sur 64 Go). **À MESURER**, priorité basse. **Effort** : S.

### P-15 — Petites synchronisations et divers (non prioritaires)
- 2 `.item()` par préfill Unified (`Gemma4UnifiedMultimodalLLMModel.swift:175-176`), comptage audio
  (`Gemma4MultimodalLLMModel.swift:196, 213`) : une fois par requête, négligeable.
- `tokenizer.decode` jeton par jeton dans MTP (`Gemma4MTPPipeline.swift:330`) : CPU, et incorrect pour les
  caractères multi-jetons (qualité, pas perf).
- `Multimodal/Gemma4Model.swift` : aucune instanciation trouvée (grep) → code mort à retirer ou aligner (P-02).

### P-16 — Encodeur audio : sortie d'attention non recastée
- **Constat (VÉRIFIÉ)** : q/k/v castés en fp32 (`AudioAttention.swift:102-104`), `context` jamais
  recasté avant `post` (l. 167) → le reste du Conformer tourne en fp32. **À VÉRIFIER** contre mlx-vlm
  (le fp32 de l'attention est peut-être voulu, le recast de sortie probablement aussi).
- **Gain** : encodeur audio (30 s ≈ 750 jetons) une fois par requête — faible. **Effort** : S.

---

## 3. Standard de profils de référence pour Gemma

### 3.1 Modèles et packs référencés (code + doc, sans réseau)

| Famille (`Model.Family`) | Packs mlx-community référencés (`Gemma4Pipeline.swift:21-59`) | Autres formats cités | Drafter MTP | Modalités |
|---|---|---|---|---|
| E2B | `gemma-4-e2b-it-{4bit,6bit,8bit,bf16}` (3,6 / 4,2 / 5,2 / 10 Go) | README : `mxfp4`, `mxfp8`, `nvfp4`, `5-bit` (non énumérés) | `google/gemma-4-E2B-it-assistant` | texte, image, vidéo, audio |
| E4B | `gemma-4-e4b-it-{4bit,6bit,8bit,bf16}` (5 / 6,5 / 8 / 19 Go) | idem | `google/gemma-4-E4B-it-assistant` | idem |
| 12B Unified | `gemma-4-12B-it-{4bit,6bit,8bit,bf16}` (6,8 / 9,5 / 12,7 / 24 Go ; 4bit = mixte 4/8) | OTF mxfp4 depuis bf16 (`BENCHMARKS.md` §1) | — | texte, image, vidéo, audio (sans encodeur) |
| 26B-A4B (MoE) | `gemma-4-26b-a4b-it-{4bit,6bit,8bit,bf16}` (14 / 21 / 27 / 52 Go) | idem | — | texte, image, vidéo |
| 31B | `gemma-4-31b-it-{4bit,6bit,8bit,bf16}` (17 / 25 / 33 / 63 Go) | idem | — | texte, image, vidéo |
| DiffusionGemma 26B-A4B | `google/diffusiongemma-26B-A4B-it` (bf16 seul) | OTF + `DiffusionMemoryConfig` | — | hors standard (a ses préréglages, `DiffusionMemoryConfig.swift:36-75`) |

Les tailles viennent de `estimatedSizeGB` (`Gemma4Pipeline.swift:99-124`), à recouper au téléchargement.
Le standard garde les trois largeurs **4 / 8 / 16** ; le 6 bits reste disponible mais hors grille (sauf
décision contraire pour 12B, voir 3.3).

### 3.2 Champs du type (chaque champ = un bouton existant, ou créé d'abord par le constat indiqué)

| Champ | Type | Bouton | État |
|---|---|---|---|
| `family` | `Gemma4Pipeline.Model.Family` | enum existant | existe |
| `bits`, `kind` | `.four/.eight/.sixteen`, `.fast/.lean` | — | nouveau (id `"\(bits)bit-\(kind)"`) |
| `model` (pack) | `Gemma4Pipeline.Model` | enum existant → repo HF + `estimatedSizeGB` | existe |
| `otfQuantization` | `(bits, groupSize, mode, excludeEncoders)?` | `Gemma4OnTheFlyQuantization.apply` | existe (nil par défaut : on préfère le pack) |
| `kvBits` | `Int?` | `GenerateParameters.kvBits` | **cassé sur E2B/E4B → P-05** |
| `prefillStepSize` | `Int` | `GenerateParameters.prefillStepSize` | **ignoré aujourd'hui → P-01** |
| `cacheLimitMB`, `memoryLimitMB` | `Int?` | `MLX.Memory.cacheLimit/memoryLimit` | API MLX existe, câblage **P-06** |
| `clearCacheAfterAnswer` | `Bool` | `MLX.Memory.clearCache()` | idem **P-06** |
| `modalities` | `.all / .noAudio / .textOnly` | `load(multimodal:)` | `.textOnly` existe ; `.noAudio` **P-10** |
| `releaseEncodersAfterPrefill` | `Bool` | modèle `unloadVision()` | **P-10** |
| `imageSoftTokens` | `Int` (280 / 140 / 70) | `ImageProcessor.maxSoftTokens` | partiel **P-03** |
| `mtp` | `nil` ou `(drafter, blockSize)` | `Gemma4MTPPipeline` | existe, **bloqué par P-04** ; variante `withMTP()` plutôt que champ de base (drafter = téléchargement séparé, greedy seulement) |
| `summary` | `String` | — | nouveau |

Règle Q/Y reprise : `applyGlobalPolicy()` pose les réglages process-wide (limites mémoire) **après** le
chargement ; les autres champs sont lus par le `Gemma4Pipeline` à l'appel.

### 3.3 Matrice proposée (valeurs initiales ; **toutes les colonnes chiffrées sont À MESURER**)

Valeurs communes : `fast` = `cacheLimit 4096 Mo`, `memoryLimit nil`, `clearCacheAfterAnswer false`,
`prefillStepSize 512` (balayage B1), `modalities .all`, `releaseEncoders false`, `imageSoftTokens 280`,
`kvBits nil`. `lean` = `cacheLimit min(1024, max(256, dispo/6))`, `memoryLimit max(4096, dispo−2048)`,
`clearCacheAfterAnswer true`, `prefillStepSize 256`, `modalities .all` (décision K-5 de Q : la vision reste
dans `lean`, « sans » = variante), `releaseEncoders true`, `imageSoftTokens 280` (140 si la qualité tient,
B3), `kvBits` selon famille. `16bit-lean` garde les **caches Mac** (T2 : bf16 + limites mobiles = +73 %) et
ne diffère de `16bit-fast` que par la résidence et la tranche.

| Famille | Id | Pack | kvBits | Particularités | Temps / pic | Pour qui |
|---|---|---|---|---|---|---|
| E2B | 4bit-fast | `e2b4bit` | nil | — | À MESURER | Mac, débit max |
| E2B | 4bit-lean | `e2b4bit` | nil (1 tête KV : gain < 100 Mo, `BENCHMARKS.md` §3) | limites adaptatives, tours libérées | À MESURER (cible iPhone 8 Go) | iOS / Mac 8-16 Go |
| E2B | 8bit-fast / -lean | `e2b8bit` | nil | idem | À MESURER | qualité vision (README : 8 bits « Fiat 600 » vs 4 bits « classic car ») |
| E2B | 16bit-fast / -lean | `e2bBf16` | nil | lean = caches Mac | À MESURER | enhancer LTX (bf16 retenu par LTX, `memory/project_ltx25_consumer.md`), cible MTP |
| E4B | 4 / 8 / 16 × fast/lean | `e4b4bit`, `e4b8bit`, `e4bBf16` | nil | idem E2B | À MESURER | meilleur rapport qualité/taille |
| 12B | 8bit-fast / -lean | `b12b8bit` | nil (MQA 1 tête : TQ refusé, gain 171 Mo) | `modalities .textOnly` recommandé pour l'encodeur LTX | À MESURER | **profil recommandé 12B** |
| 12B | 16bit-fast / -lean | `b12bBf16` | nil | — | À MESURER | encodeur LTX (référence) |
| 12B | 4bit-fast / -lean | `b12b4bit` (mixte 4/8) | nil | `summary` : « qualité dégradée : MMLU −20 pts » | À MESURER | démo mémoire seulement |
| 26B-A4B | 4 / 8 / 16 × fast/lean | `a4b4bit`, `a4b8bit`, `a4bBf16` | lean : **8** (GQA 4, à valider P-05) | pas d'audio (`.noAudio` implicite) ; 4 bits : −6 pts (issue #27) | À MESURER | Mac 32-64 Go |
| 31B | 4 / 8 / 16 × fast/lean | `b31b4bit`, `b31b8bit`, `b31bBf16` | lean : **8** (TQ4 : −735 Mo à 12 k, `BENCHMARKS.md` §3) | `wiredMemory` à évaluer (P-14) | À MESURER | Mac 32-96 Go |

Variantes (non comptées comme profils, comme `textOnlyVariant()` de Q) : `textOnlyVariant()`,
`noAudioVariant()`, `withMTP(drafter:blockSize:)` (E2B/E4B, greedy, après P-04 ; `blockSize` balayé 2-6
avec taux d'acceptation, T21).

Question ouverte à trancher par Vincent : faut-il un `6bit-*` pour le 12B (MMLU 50 %, −7 pts) plutôt que
de marquer le 4 bits « dégradé » ? La grille 4/8/16 suffit si la réponse est non.

### 3.4 Placement et exposition

- **Type** : `Sources/Gemma4Swift/Configuration/Gemma4ReferenceProfile.swift` (convention de nommage du
  dépôt : préfixe `Gemma4`, cf. `Gemma4TextConfig.swift`). `public struct Gemma4ReferenceProfile: Sendable,
  Identifiable, Equatable` avec `family`, `bits`, `kind`, champs 3.2, `id`, `summary`,
  `static let all: [Gemma4ReferenceProfile]`, `static func named(_ id: String, family:) -> Self?`,
  `static func recommended(for family:, availableMB:) -> Self`, `applyGlobalPolicy()`,
  `static func availableMemoryMB()` (override `GEMMA4_AVAILABLE_MB`), `textOnlyVariant()`,
  `noAudioVariant()`, `withMTP(...)`.
- **API** (strictement additive — Fluxforge suit la branche `main`, LTX les tags ; `memory/project_fluxforge_consumer.md`,
  `project_ltx25_consumer.md`) : nouvelles surcharges `Gemma4Pipeline.load(_ model:, profile:)` et
  `load(from:profile:)` qui stockent le profil et appellent `applyGlobalPolicy()` après chargement ;
  `chatStream`/`chatStreamMultimodal` lisent `profile?.prefillStepSize/kvBits` **seulement si un profil a
  été posé** ; sans profil, comportement 1.7.x inchangé (aucun défaut modifié). `Gemma4Registration.loadContainer`
  et `forwardCollectingHiddenStates` (LTX) ne changent pas. Pas de nouveau `import MLXLMCommon` exigé côté
  consommateur.
- **CLI** : `gemma4-cli references [--family e2b]` (une ligne par profil : tous les champs + `summary` +
  ligne `weights: gemma4-cli download <pack> (repo, taille)`) ; `--reference <id>` sur `generate`,
  `chat`, `describe`, `bench` (P-07) ; la famille est déduite du `config.json` de `--model-path`, et un
  pack dont la quantification ne correspond pas aux `bits` du profil est **refusé** (piège 12).
  Chaque profil a ses équivalents en options explicites (`--prefill-step`, `--kv-bits`, `--cache-limit-mb`,
  `--text-only`, `--no-audio`).
- **Docs** : `docs/References.md` (table mesurée, « What each setting does », « Choosing » par classe de
  machine, « Adding or changing a profile », équivalents CLI), ligne brute par profil dans `BENCHMARKS.md`,
  décision dans `docs/knowledge/decisions/reference-profiles.md`.

---

## 4. Mesures de référence à faire en premier (baseline), dans l'ordre

Conditions pour tous les points : Release (`xcodebuild -configuration Release`), `pgrep -fl "gemma4|serve|train|lora|yue2|qwen38"`
vide, 120 s de refroidissement, sorties sous `.local-runs/bench.noindex/`, processus le plus gourmand
journalisé avant chaque point, machine M3 Max 96 Go (celle de Vincent), révisions des dépendances notées,
**mesures à chaud** (poids sur disque externe Lexar : une passe d'amorçage jetée), premier point d'un
script jeté (montée en fréquence).

0. **Instrument** (P-07) : créer `gemma4-cli bench` (ligne JSON, chemin `TokenIterator`, phases,
   `phys_footprint`). Contrôle : A/A sur E2B 4 bits, dispersion ≤ 3 %.
1. **B1 texte** — E2B 4 bits : prompts de 128 / 1k / 4k / 8k jetons (tronqués exactement au tokenizer),
   128 jetons greedy : préfill ms, TTFT, décodage médiane/p90, pic actif, pic `phys_footprint`,
   `weights_bw_gbps`. Base de P-01, P-06, P-12.
2. **B9 mémoire** — E2B 4 bits à 8k et 32k : `phys_footprint` contre actif (ampleur du cache non borné) ;
   refaire avec `GEMMA4_AVAILABLE_MB=16384` une fois P-06 codé.
3. **B3 image** — E2B 4 bits, 1 image de référence + prompt court, contre prompt texte de même longueur :
   encodeur ms, préfill, décodage, dtype du cache. Base de P-02/P-03.
4. **B6 MTP** — E2B bf16 + drafter : `mtp-generate --compare` à 100 et 700 jetons de prompt (test de
   correction P-04, **bloquant** pour tout profil `withMTP`), tok/s contre greedy `TokenIterator`.
5. **B4 vidéo / B5 audio** — E2B 4 bits : 9 frames, 30 s d'audio : ms d'encodeur, pic.
6. **B7 n-gramme** — E4B 4 bits, greedy 256 jetons, n = 5 contre sans (P-08).
7. **B8 échantillonnage** — E2B 4 bits, temp 0 contre 0,3/topP 0,95 (P-11).
8. **B11 résidence** — `load(multimodal: true)` contre `false`, delta d'actif après `eval` (P-10).
9. **B2 12B** — 12B 8 bits : B1 à 512 / 4k jetons + `forwardCollectingHiddenStates` à 512 jetons (usage LTX).
10. **B10 multi-tours** — rejeu 4 tours `continueChat` contre re-préfill complet (parité + TTFT, P-09).
11. 26B / 31B : seulement si l'espace disque le permet (télécharger / mesurer / supprimer), B1 + B9 + KV 8 bits (P-05).

Ordre de correction proposé après la base : P-07 → P-02 (S, gain probablement le plus gros sur les
médias) → P-01 → P-06 → P-04 (bloquant MTP) → P-05 → P-03 → P-08 → P-10 → P-11 → P-13 → P-09 → P-14/P-16.

---

## 5. Résumé

1. Audit en lecture seule de `main` @ `c9543739` (mlx-swift 0.31.6, mlx-swift-lm 3.31.4), 16 constats P-01…P-16.
2. P-01 : les trois wrappers surchargent `prepare` et préremplissent le prompt entier ; `prefillStepSize` est ignoré.
3. Le head lié (262 144 lignes) tourne sur toutes les positions : ~2,1 Go de logits bf16 à 4k jetons (E2B).
4. MLX n'a pas de SDPA fusionné pour head_dim 256/512 au préfill : scores L×L matérialisés à chaque couche.
5. P-02 : `embedScale` en fp32 au préfill multimodal → cache KV et décodage en fp32 (indices : 20,8 → 12-13 t/s sur 12B avec image).
6. P-03 : l'encodeur vision complète toujours à 2 520 patches (×4 pour une frame vidéo) et calcule en fp32.
7. P-03 bis : `asType(inputProj.weight.dtype)` + OTF qui n'exclut plus rien → probable image détruite en `--quantize-bits` (À VÉRIFIER).
8. P-04 : MTP — `trimPromptCache` ne fait rien dès 512 jetons (RotatingKVCache) : sortie non bit-exacte sur contexte long.
9. P-05 : `kvBits` natif (recommandé par le README) casse les couches KV-partagées d'E2B/E4B.
10. P-06 : aucune `cacheLimit`/`memoryLimit`/`clearCache` sur l'inférence (seulement la diffusion).
11. P-07 : `asyncEval` mort dans `profile` et boucles synchrones dans `describe` : les chiffres publiés ne mesurent pas le chemin bibliothèque.
12. P-08 : le n-gramme (LTX, n = 5) force une synchronisation CPU par jeton.
13. T11 (KV préalloué) et T12 (head via quantizedMM) déjà couverts par l'amont ; R1-R3, R12, R16 à ne pas retenter.
14. Réutilisation de conversation partielle (`continueChat` seulement) ; chantier L, après P-01/P-04.
15. Encodeurs vision+audio toujours résidents (≈ 0,9 Go bf16 estimé sur E2B, À MESURER).
16. Standard de profils : `Gemma4ReferenceProfile` dans `Sources/Gemma4Swift/Configuration/`, ids `<bits>bit-<fast|lean>` par famille.
17. Familles : E2B, E4B, 12B Unified, 26B-A4B, 31B × packs mlx-community 4/8/bf16 ; DiffusionGemma hors standard.
18. `kvBits` 8 seulement en lean 26B/31B (GQA 4) ; nil pour E2B/E4B/12B (1 tête KV).
19. 12B 4 bits marqué « qualité dégradée » (MMLU −20 pts) ; 12B 8 bits recommandé.
20. MTP en variante `withMTP()`, bloquée par P-04 ; `textOnlyVariant()` / `noAudioVariant()` comme chez Q.
21. API strictement additive (`load(_:profile:)`, `applyGlobalPolicy()`), aucun défaut modifié : Fluxforge suit `main`.
22. CLI : `gemma4-cli references`, `--reference <id>`, refus d'un pack aux bits incompatibles.
23. Premier travail : instrument `gemma4-cli bench` (JSON, phases, `phys_footprint`) puis base B1-B11.
24. Ordre de correction : P-07, P-02, P-01, P-06, P-04, P-05, P-03, P-08, P-10, P-11, P-13, P-09.
25. Fichier : `audit-performance.md`
