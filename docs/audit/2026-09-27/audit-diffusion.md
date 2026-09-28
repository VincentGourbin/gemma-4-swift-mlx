# Audit du chemin DiffusionGemma — correction, performance, profils, patterns itératifs

Date : 2026-09-28 · Dépôt : `gemma-4-swift-mlx`, branche `fix/lot-a-stabilite` @ `124de4f5` (arbre sale :
`Gemma4ReferenceProfile.swift`, `ReferenceProfileTests.swift`, sans lien avec la diffusion).
mlx-swift 0.31.6, mlx-swift-lm 3.31.4. **Audit en lecture seule** : aucun build, aucun GPU (une campagne
de mesure tourne). Seul ce fichier a été écrit.

Sources lues : `Sources/Gemma4Swift/Diffusion/**` (23 fichiers, 2 792 lignes), `Sources/Gemma4CLI/{DiffusionGemmaCommand,ProfileDiffusionCommand}.swift`,
`docs/DIFFUSIONGEMMA.md`, `docs/examples/diffusion-{optim-phases,quantization-bench}.md`, tests `Diffusion*`,
`docs/audit/2026-09-27/{PLAN,audit-performance,scan}.md`, `docs/References.md`, `Gemma4ReferenceProfile.swift`,
le catalogue `~/.claude/skills/mlx-swift-audit/references/{techniques,measurement,profiles-standard}.md`,
YuE2 (`docs/References.md`, `docs/Benchmarks.md`, `docs/knowledge/log.md`, `Synthesis/CachedNAR.swift`),
la référence Python HF (`transformers/models/diffusion_gemma/{generation,modular}_diffusion_gemma.py` ; copie
locale lue dans `~/Developpements/h3-swift-mlx/.local-runs/parity-venv/lib/python3.12/site-packages/transformers/` :
arrêt `generation_diffusion_gemma.py:521-530` (comparaison **puis** `roll`), encodeur `modular_diffusion_gemma.py:758, 778`
(`DynamicCache(config=…)`, `create_sliding_window_causal_mask`), `cache_utils.py:190, 230-232`
(`DynamicSlidingWindowLayer` ne garde que les `sliding_window − 1` derniers jetons) — D-04 et D-05 sont donc
VÉRIFIÉS contre la référence), `config.json` du pack AR
`gemma-4-26b-a4b-it-4bit` (Lexar) pour les dimensions (même squelette que DiffusionGemma).

Légende : **VÉRIFIÉ** = lu dans le code (fichier:ligne) ou dans une mesure publiée du dépôt ;
**À MESURER** = effet chiffré inconnu, protocole fourni ; **À VÉRIFIER** = hypothèse à confirmer par un test
court ; **CALCUL** = ordre de grandeur dérivé des dimensions, pas une mesure.

---

## A. Carte du chemin DiffusionGemma

### A.1 Dimensions utiles (VÉRIFIÉ, `config.json` 26B-A4B ; DiffusionGemma = même tronc)

hidden 2 816 · 30 couches (25 glissantes, 5 pleines) · `sliding_window` **1 024** · 16 têtes Q ·
8 têtes KV (glissantes, `head_dim` 256) / 2 têtes KV (pleines, `global_head_dim` 512, K=V) · MoE 128 experts,
top-8, `moe_intermediate` 704 · MLP dense 2 112 · vocabulaire 262 144 · canvas 256.

Répartition des 25,8 G paramètres (CALCUL) : experts 3 × 128 × 2 816 × 704 × 30 = **22,8 G (88 %)** ;
attention ≈ 1,11 G ; MLP dense ≈ 0,54 G ; `embed_tokens` 0,74 G ; self-conditioning 18 M ; routeurs 11 M ;
vision SigLIP ≈ 0,4 G. Paramètres actifs par jeton ≈ 3,1 G (experts 1,43 G).

### A.2 Étapes

| # | Étape | Code | Fréquence | Ce qui est calculé | Cache |
|---|---|---|---|---|---|
| 0 | Chargement | `DiffusionGemmaLoader.load` (`Pipeline/DiffusionGemmaLoader.swift:38-47`) | 1 × | tous les `*.safetensors` en dict, `DiffusionWeightSanitizer` **duplique** chaque poids `decoder.*` vers `encoder.language_model.*` (même `MLXArray`, `DiffusionWeightSanitizer.swift:106-111`), `update(verify: .all)`, `eval(model)` | poids bf16 ≈ 50 Go, **partagés** encodeur/décodeur |
| 0b | Quantification à la volée (option) | `DiffusionOnTheFlyQuantization.apply` / `applyMixedPrecision` (`:35-109`, `:192-261`) | 1 × | `MLXNN.quantize` avec filtre `Linear || Embedding` | voir D-01/D-02 |
| 1 | Encodeur (canvas 0) | `encodePrompt` → `DiffusionGemmaEncoderModel.callAsFunction` (`TextModel/DiffusionGemmaEncoderModel.swift:75-121`) | 1 × par requête | embeddings, vision (SigLIP + `embed_vision` + `maskedScatter`) si image, 30 couches MoE sur **tout le prompt en un passage**, masque causal `[T,T]` matérialisé (`DiffusionGemmaEncoderTextModel.swift:96-104`) | `EncoderKVCache` : K/V de **toutes** les positions pour **toutes** les couches (`EncoderKVCache.swift:22-68`) |
| 1' | Encodeur incrémental (canvas N ≥ 1) | `DiffusionGemmaPipeline.swift:117-128` | 1 × par canvas | seulement les 256 jetons commités (argmax du canvas précédent), concaténés au cache (2 `concatenated` par couche : `DiffusionGemmaEncoderTextAttention.swift:134-135` et `DiffusionGemmaEncoderTextModel.swift:126-127`) | cache étendu (croît de 256 par canvas) |
| 2 | Init canvas | `EntropyBoundSampler.initializeCanvas` (`Sampling/EntropyBoundSampler.swift:48-55`) | 1 × par canvas | `randInt` uniforme `[B, 256]` | — |
| 3 | **Boucle de débruitage** | `DiffusionGemmaPipeline.swift:150-204` | ≤ 48 × par canvas (≈ 16 mesurés) | voir A.3 | `prevLogits` (bf16) d'un pas au suivant |
| 4 | Commit | `:210-216` | 1 × par canvas | `fullIds = concat(fullIds, argmax)`, test EOS côté CPU (`containsEOS`, `canvas.eval()` + `asArray`, `:248-259`) | — |
| 5 | Nettoyage | `:221` | 1 × par canvas (sauf dernier) | `MLX.Memory.clearCache()` **inconditionnel** | — |

### A.3 Un pas de débruitage (ce qui est recalculé à chaque pas)

1. `embed_tokens(canvas) * sqrt(H)` (`DiffusionGemmaDecoderTextModel.swift:82-83`).
2. Self-conditioning (dès le 2ᵉ pas) : `softmax(prevLogits, fp32)` sur `[1, 256, 262 144]`, puis
   `probs @ embed_tokens.weight` (matmul dense 256 × 262 144 × 2 816 ≈ **0,38 TFLOP**) (`:93-121`),
   MLP SwiGLU + normes (`DiffusionGemmaSelfConditioning.swift:45-53`).
3. 30 couches décodeur : Q/K/V du canvas, **`concatenated([encoderKV, canvasKV])` par couche et par pas**
   (`DiffusionDecoderAttention.swift:149-151`), SDPA **non fusionné** (head_dim 256/512 hors des noyaux
   fusionnés, cf. audit-performance §0), MLP dense + MoE (`SwitchGLU`, tri des indices dès 64 affectations).
   Masque : `nil` (`DiffusionGemmaPipeline.swift:156` → `DiffusionAttentionMask.create` renvoie `(nil, nil)`,
   `DiffusionAttentionMask.swift:60-62`).
4. Head lié `embed_tokens.asLinear` sur **les 256 positions** (≈ 0,38 TFLOP) → `asType(.float32)` →
   softcap `tanh(x/c)*c` (`DiffusionGemmaForBlockDiffusion.swift:54-60`).
5. Température `logits / T(step)` (`LinearTemperatureSchedule.swift:32-35`).
6. `argMax` ; `categorical` (Gumbel : 67 M tirages aléatoires par pas) (`DiffusionGemmaPipeline.swift:163-166`).
7. `accept` : entropie par position (logSoftmax + exp + mul + sum sur 67 M éléments), `argSort`, `cumsum`,
   `putAlong`, `where` (`EntropyBoundSampler.swift:90-117`).
8. `shouldStop` : **deuxième** calcul complet de l'entropie sur les mêmes logits
   (`StableConfidentStopping.swift:60`) + comparaison d'historique.
9. **Synchronisation** : `shouldStop.all().item(Bool.self)` (`DiffusionGemmaPipeline.swift:181`).
10. `renoise` (`randInt` + `where`), `prevLogits = scaled.asType(.bfloat16)` (`:185-194`).

Coût relatif (CALCUL, 256 positions) : décodeur ≈ 2 × 3,1 G × 256 ≈ 1,6 TFLOP ; head + self-conditioning
≈ 0,76 TFLOP (**~⅓ des FLOP du pas**) ; post-traitement des logits ≈ 20 passes sur un tenseur fp32 de
268 Mo ≈ 5 Go de trafic (≈ 13 ms, 2-3 % du pas). Mesuré (phases 5-7, bf16) : **pas moyen 605 ms, min 506 ms**,
63 forwards pour 4 canvases (≈ 16/canvas sur 48 max), soit ≈ 27 jetons/s effectifs hors chargement.
Le pas de diffusion est, pour le matériel, **un préfill de 256 jetons + deux GEMM vocabulaire** : c'est dans
cette zone (128 → 1 k jetons) que le tableau `docs/References.md` montre que les bits comptent encore sur ce MoE
(préfill `a4b` 128 jetons : 161 tok/s bf16 contre 416 en 4 bits ; à 4 k, 808 contre 965).

### A.4 Synchronisations CPU/GPU par pas

| Point | Fichier:ligne | Fréquence |
|---|---|---|
| `shouldStop.all().item(Bool)` | `DiffusionGemmaPipeline.swift:181` | 1 / pas (seule sync du chemin bibliothèque) |
| `onStep` (si fourni) : la CLI fait `argmaxCanvas.eval()` + `asArray` + décodage | `DiffusionGemmaCommand.swift:297-311` | 1 / pas, `--stream-steps` seulement (après la sync ci-dessus : coût = copie 1 Ko + détokenisation) |
| `containsEOS` : `eval` + `asArray` | `DiffusionGemmaPipeline.swift:251-252` | 1 / canvas |
| `evalEveryNLayers` (défaut 0) | `DiffusionGemmaDecoderTextModel.swift:159-164` | 0 par défaut |
| `ProfileDiffusionCommand` : `argmaxCanvas.eval()` supplémentaire avant le `.item()` | `ProfileDiffusionCommand.swift:359` | 1 / pas (outil seulement) |

### A.5 dtype par étape (VÉRIFIÉ)

| Tenseur | dtype | Preuve |
|---|---|---|
| Poids | bf16 (checkpoint) | — |
| Embeddings × échelle | dtype de l'entrée (pas de fuite fp32, contrairement à l'ancien chemin AR) | `DiffusionGemmaEncoderModel.swift:99`, `DiffusionGemmaDecoderTextModel.swift:83, 115` |
| Logits | **fp32** voulu (softcap, température, softmax, entropie, catégorielle) | `DiffusionGemmaForBlockDiffusion.swift:59` |
| `prevLogits` (self-conditioning) | bf16 (Phase 7) — conforme à Python (`processed_logits.to(embeddings_dtype)`) | `DiffusionGemmaPipeline.swift:194` |
| `probs` du self-conditioning | fp32 puis cast au dtype des poids / échelles | `DiffusionGemmaDecoderTextModel.swift:94-113` |
| Cache encodeur | bf16 | — |
| `ProfileDiffusionCommand` : `prevLogits = scaled` | **fp32** (≠ pipeline) | `ProfileDiffusionCommand.swift:374` |

### A.6 Politique mémoire actuelle

- `DiffusionMemoryConfig` (`Pipeline/DiffusionMemoryConfig.swift:12-89`) : `mixedPrecision`,
  `unloadVisionAfterFirstCanvas`, `clearCacheBetweenCanvases` ; presets `disabled/light/moderate/aggressive/extreme`,
  `recommended(forRAMGB:)` (96 Go → `light`).
- **Seul `mixedPrecision` est câblé** (`DiffusionGemmaRegistration.swift:91-93`). Les deux booléens ne sont lus
  nulle part : le pipeline appelle `clearCache()` entre canvases sans condition (`DiffusionGemmaPipeline.swift:221`)
  et n'appelle jamais `unloadVision()` ; `DiffusionGemmaContainer.makePipeline()` ne transmet pas la config
  (`DiffusionGemmaRegistration.swift:39-41`). Seul `ProfileDiffusionCommand.swift:315-319` décharge la vision.
- Aucun `cacheLimit` / `memoryLimit` dans la bibliothèque ; `--cache-limit-gb` dans les deux commandes CLI
  seulement (`DiffusionGemmaCommand.swift:150-154`, `ProfileDiffusionCommand.swift:149-154`).
- `evalEveryNLayers` / `clearCacheOnEval` : propriétés publiques du décodeur, réglées uniquement par
  `profile-diffusion`.

### A.7 Points d'entrée publics et consommateurs

- Bibliothèque : `DiffusionGemmaRegistration.load(from:|modelId:, memoryConfig:, includeVision:)` →
  `DiffusionGemmaContainer` → `makePipeline()` ; `DiffusionGemmaLoader.load` ; `DiffusionGemmaPipeline.generate(promptIds:pixelValues:maxBlocks:seed:onCanvas:onStep:)`
  (synchrone dans un `actor`, non `throws`) ; `DiffusionOnTheFlyQuantization.*` ; tous les modules `TextModel/*` et
  samplers sont `public`.
- `Gemma4Pipeline.load` refuse la famille `.a4bDiff` (`Gemma4Pipeline.swift:293-297`) ; `Gemma4ReferenceProfile.all`
  l'exclut (`Gemma4ReferenceProfile.swift:120-121`).
- CLI : `gemma4-cli diffusion`, `gemma4-cli profile-diffusion`.
- Consommateurs : **aucun externe** (audit-stabilite §F : Fluxforge, LTX, flux-2, h3, ToolsForge n'utilisent pas la
  diffusion). Interne : `gemma4-bench-ui` (5 sites : `BenchViewModel.swift:452`, `Agent/WebAgentLoop.swift:289`,
  `Agent/AgentStepViewModel.swift:322`, `IOSSim/IOSAgentStepViewModel.swift:303`, `VQAGame/VQAGameViewModel.swift:143`),
  tous en `Task.detached` avec `nonisolated(unsafe)` sur le modèle et les `pixelValues`. **Conséquence : la surface
  peut changer sans casser personne ; rester additif quand même (1.8.0).**

---

## B. Constats D-01…

Ordre : correction d'abord (ce qui fausse les mesures passées ou la sortie), puis stabilité, puis leviers.
Seuil de rétention 5 %, A/B/B/A, Release, machine au repos (`machine-check.sh --cooldown 120`).

### D-01 — La quantification à la volée ne quantifie pas les experts MoE (88 % des poids) — **VÉRIFIÉ**
- **Constat** : les deux filtres ne retiennent que `m is Linear || m is Embedding`
  (`DiffusionOnTheFlyQuantization.swift:86-88` et `:217-219`). Les experts sont des `SwitchLinear`
  (`Gemma4Experts.swift:10-21` → `SwitchGLU`) ; `SwitchLinear: Module, Quantizable` n'hérite **pas** de `Linear`
  (`.build/xcode/SourcePackages/checkouts/mlx-swift-lm/Libraries/MLXLMCommon/SwitchLayers.swift:222`). Les 22,8 G
  paramètres d'experts restent en bf16 quel que soit `--quantize-bits` ou `--mixed-precision`.
- **Corroboration par la mesure publiée** (`docs/examples/diffusion-quantization-bench.md`) : mémoire libérée
  4 bits 2,5 Go, 6 bits 1,3 Go, 8 bits 0,1 Go sur ≈ 50 Go. CALCUL avec D-02 : partie non-experts ≈ 2,4 G param.
  = 4,8 Go bf16 ; quantifiée deux fois (encodeur + décodeur) : 4 bits 2 × 1,35 = 2,7 Go (libère 2,1), 6 bits
  2 × 1,95 = 3,9 (libère 0,9), 8 bits 2 × 2,55 = 5,1 (libère −0,3). Ordre de grandeur et monotonie identiques aux
  chiffres mesurés.
- **Conséquence** : le verdict des phases 5-7 (« aucune quantification ne bat bf16 sur Apple Silicon », « le pool
  MLX refuse de se réduire », « `clearCache` ineffectif ») **ne teste pas la quantification** : il mesure 10 % du
  modèle quantifié, dupliqué. La doc publique promet l'inverse de la réalité : aide CLI « 48 Go → ~14 Go en 4-bit,
  3-4x speedup » (`DiffusionGemmaCommand.swift:65`), en-tête `DiffusionOnTheFlyQuantization.swift:7-9`,
  `DIFFUSIONGEMMA.md` « économiser ~70 % de RAM », presets `moderate/aggressive/extreme` « cible Mac 64/32-48/16-24 Go »
  (`DiffusionMemoryConfig.swift:52-74`) qui ne tiennent sur aucune de ces machines (≈ 48 Go résidents).
- **Gain attendu** (À MESURER) : mémoire 4 bits ≈ 50 → **≈ 15 Go**, 8 bits ≈ 27 Go (CALCUL ; cohérent avec les pics
  AR mesurés `a4b/4bit` 14,4 Go, `a4b/8bit` 26,5 Go). Vitesse : le pas est un préfill de 256 jetons, régime où le
  4 bits reste nettement plus rapide que bf16 sur ce MoE (References.md : ×2,6 à 128 jetons, ×1,3 à 1 k,
  ×1,2 à 4 k) → attendu **−20 à −45 % par pas**, hypothèse à confirmer. À surveiller : nombre de forwards (D-03).
- **Correction** : filtre sur `Quantizable` (ou `m is Linear || m is Embedding || m is SwitchLinear`), avec la règle de
  divisibilité sur la dernière dimension ; `MixedPrecisionConfig` s'applique aussi aux experts.
- **Source** : T13, piège catalogue nouveau (§D, IT-5). Porte : voir K-D3/K-D11.
- **Risque** : faible en code ; qualité à mesurer (D-03).

### D-02 — Poids liés encodeur/décodeur quantifiés deux fois — **VÉRIFIÉ (code), effet CALCUL**
- **Constat** : le sanitizer insère le même `MLXArray` sous `decoder.X` et `encoder.language_model.X`
  (`DiffusionWeightSanitizer.swift:106-111`) : en bf16, un seul tampon (d'où ≈ 49 Go et non ≈ 95). Mais
  `apply` quantifie le modèle entier, et `applyMixedPrecision` parcourt séparément `encoder.languageModel.layers`
  et `decoder.layers` puis les deux `embed_tokens` (`:234-251`) : chaque `quantizeSingle` produit un **nouveau**
  triplet (poids packés, échelles, biais) → deux copies quantifiées. Une fois D-01 corrigé, un 4 bits pèserait
  ≈ 2 × 14 = 28 Go au lieu de 14.
- **Correction** : quantifier une fois et partager. Le plus simple et le plus sûr : quantifier au niveau du
  dictionnaire (`MLX.quantized` par clé source **avant** duplication dans le sanitizer, structure du modèle
  quantifiée d'abord par `MLXNN.quantize` avec le même filtre), puis `update(parameters:)` — les deux voies
  reçoivent les mêmes trois `MLXArray`. Idem pour `embed_tokens` (encodeur, décodeur, head lié). Les `layer_scalar`
  restent propres à chaque voie.
- **Test sans GPU** : sur une config minuscule, après quantification, `encoder.…q_proj.weight` et
  `decoder.…q_proj.weight` sont le **même** tampon (identité ou somme `nbytes` des paramètres = attendu analytique).
- **Source** : T4 (« le budget = une seule copie par voie »), IT-6.

### D-03 — Routeur MoE quantifié en 4 bits : cause probable des ×2,6 forwards — **VÉRIFIÉ (code), effet À MESURER**
- **Constat** : `Gemma4Router.proj` est un `Linear` (`TextModel/Gemma4Router.swift:21, 31`) → quantifié en 4 bits par
  `apply(bits: 4)` et, en précision mixte, dans les couches basses (4-25). La recette du pack de référence
  `mlx-community/gemma-4-26b-a4b-it-4bit` garde **les 30 `router.proj` en 8 bits** (VÉRIFIÉ, `config.json` du pack :
  30 surcharges `{"bits": 8, "group_size": 64}`).
- **Mesure publiée** : 4 bits « pur » = 167 forwards contre 63 (×2,6), total ×2,9 ; mixte `default` = 75 forwards
  (+19 %) pour un pas −5 % → total +12 %. Un routage bruité change les experts choisis, donc les logits, donc
  l'entropie → l'arrêt adaptatif (entropie moyenne < 0,005) arrive plus tard. Le pas plus lent en 4 bits (+17 %)
  s'explique par D-01 (experts bf16 + petites matmuls quantifiées en plus).
- **Correction** : routeur toujours ≥ 8 bits (ou bf16 : 0,36 M param./couche, coût nul), comme le pack mlx-community.
- **Porte** : à bits égaux, forwards / canvas ≤ bf16 + 15 % et score ScreenSpot-100 ≥ bf16 − 2 pts.

### D-04 — Critère d'arrêt : « stable » est toujours vrai avec `stability_threshold = 1` — **VÉRIFIÉ (code et Python)**
- **Constat** : `shouldStop` ajoute l'argmax courant à l'historique **puis** compare l'historique à l'argmax courant
  (`StableConfidentStopping.swift:40-57`). Avec le seuil par défaut 1 (`DiffusionGenerationConfig.swift:43`),
  l'historique ne contient que l'argmax courant → `stable` vaut toujours vrai, l'arrêt se réduit à « confiant ».
  Python compare **avant** de mettre à jour :
  `stable = (self.argmax_canvas_history == argmax_canvas[None]).all(-1).all(0)` puis `roll` et écriture.
- **Conséquence** : le port s'arrête potentiellement **plus tôt** que la référence ; les 63 forwards bf16 de la base
  (et toutes les comparaisons de forwards des phases 5-7) sont sur un critère différent de Python. Corriger
  **augmentera** probablement le nombre de forwards (≥ +1 par canvas) : c'est une correction, pas un levier.
- **Correction** : comparer à l'historique précédent (initialisé à une valeur impossible, p. ex. −1, de taille
  `stabilityThreshold`) puis décaler. Test unitaire : au 1ᵉʳ pas, jamais stable ; deux argmax égaux consécutifs + entropie
  basse → stop.
- **Effort** : S. À mesurer ensuite : forwards/canvas et scores (K-D10).

### D-05 — Fenêtre glissante ignorée côté encodeur et côté décodeur — **VÉRIFIÉ (code), écart à Python VÉRIFIÉ sur extraits**
- **Constat** :
  - Encodeur : `slidingMask = mask` (masque causal plein) pour les 25 couches glissantes
    (`DiffusionGemmaEncoderTextModel.swift:104`) ; Python : `create_sliding_window_causal_mask` pour
    `sliding_attention`.
  - Cache : `EncoderKVCache` garde toutes les positions pour toutes les couches ; Python : `DynamicCache(config=…)`,
    dont les couches glissantes ne gardent que les `sliding_window − 1` derniers jetons (le masque décodeur
    tronque `kv_length` à `get_max_length()` de la couche glissante).
  - Décodeur : le pipeline passe `decoderAttentionMask: nil` (`DiffusionGemmaPipeline.swift:156`) → `(nil, nil)`
    (`DiffusionAttentionMask.swift:60-62`) → les couches glissantes voient **tout** le cache encodeur. Python
    construit toujours un masque (`ones` + pad `canvas_length`) et donc la tranche glissante.
- **Conséquence** : identique à Python tant que prompt + jetons générés < 1 024 ; **au-delà** (agent web : capture
  280 jetons + `pageText` 2 000 caractères + gabarit ; 4 canvases de 256 après un prompt de 100 jetons), sortie
  différente de la référence. Perf : 25 couches sur 30 lisent tout le contexte à chaque pas, cache encodeur non borné
  (≈ 8 Ko/jeton/couche glissante en bf16, CALCUL) — à 4 k jetons : ≈ 0,8 Go de cache et 0,8 Go recopiés par pas au
  lieu de ≈ 0,2 Go.
- **Correction** : masque glissant à l'encodeur (`createCausalMask(n:offset:windowSize:)` ou équivalent) ; entrées de
  cache glissantes tronquées aux `sliding_window − 1` dernières positions après chaque extension ; au décodeur,
  aucune tranche à faire si le cache est déjà tronqué (le masque `nil` redevient correct : bidirectionnel sur
  « fenêtre + canvas »). Garder la position RoPE sur l'offset **total** (pas sur la longueur tronquée) : aujourd'hui
  `seqLength` lit la 1ʳᵉ couche remplie (`EncoderKVCache.swift:43-50`) — qui est glissante (couche 0) : il faut un
  offset explicite, sinon les positions deviennent fausses après troncature (piège 14/36).
- **Test sans GPU** : config minuscule (fenêtre 8, 2 couches), prompt de 20 jetons : sorties décodeur égales à une
  référence naïve avec masques explicites ; à 6 jetons, identiques à l'ancien chemin.
- **Gain perf** : À MESURER sur D3 (contexte 2,5-4 k) ; attendu faible à 1 k, sensible à 4 k.

### D-06 — Encodeur : blocs vision bidirectionnels non implémentés — **VÉRIFIÉ (commentaire du code)**
- `use_bidirectional_attention = "vision"` (défaut du checkpoint) : Python rend l'attention bidirectionnelle à
  l'intérieur des blocs d'image ; le port fait du causal pur (en-tête `DiffusionGemmaEncoderTextModel.swift:13-14`,
  `:97-103`). Les scores publiés (ScreenSpot 79 %, OCRBench 80,8 %) ont été obtenus **avec** cet écart : la correction
  peut les améliorer. Même mécanisme que le 12B Unified (masque bidirectionnel `[T,T]`, `Gemma4TextModel.swift:362-374`)
  → réutiliser. Porte : ScreenSpot-100 ≥ 79 % (pas de régression), OCRBench-100 idem.

### D-07 — Génération non annulable, hors garde K-9 — **VÉRIFIÉ**
- `generate` est une boucle synchrone sans `Task.checkCancellation()` ni rappel d'annulation
  (`DiffusionGemmaPipeline.swift:87-234`). `BenchViewModel.swift:522` fait `continuation.onTermination = { task.cancel() }`,
  sans effet : la génération continue jusqu'à `maxBlocks × 48` pas (jusqu'à ≈ 2 min en bf16) en tenant le modèle
  et le GPU. Même défaut que S-01 (lot A K-1) sur le chemin AR.
- `Gemma4ComputeGate.shared.beginInference()` n'est appelé nulle part dans la diffusion (grep : seulement
  `Gemma4Pipeline`, `Gemma4MTPPipeline`, entraînements). Le pas utilise des fonctions compilées
  (`geluApproximate` dans `Gemma4MLP`/`SwitchGLU` et le self-conditioning, `compiledTokenEntropy`) : une
  diffusion pendant un `lora train` dans le même processus (BenchUI) est exposée au deadlock ABBA décrit dans
  `CLAUDE.md`.
- **Correction (additive)** : `generate` vérifie l'annulation à chaque pas (avant le forward, comme
  `CachedNAR.solve` de YuE2, `Synthesis/CachedNAR.swift:104-113`) et rend un résultat partiel marqué annulé (ou une
  surcharge `throws`) ; `beginInference()/endInference()` autour de la boucle. Test : annulation après 1 pas → retour
  en < 2 pas ; entraînement en cours → refus explicite.

### D-08 — `MLXArray` qui traversent une frontière sans `eval` — **VÉRIFIÉ**
- `DiffusionGenerationResult.generatedIds` est une tranche paresseuse de `fullIds` (`DiffusionGemmaPipeline.swift:226`)
  rendue hors de l'actor ; `onStep`/`onCanvas` reçoivent des tableaux paresseux (le canvas de `onCanvas` est déjà
  évalué par `containsEOS`, pas `argmaxCanvas` de `onStep` si l'appelant l'utilise avant la sync). BenchUI passe des
  `pixelValues` non évalués à un `Task.detached` via `nonisolated(unsafe)` (`BenchViewModel.swift:446-449`).
- Correction : `eval(generatedIds, fullIds)` avant de rendre ; documenter que `onStep` reçoit un tableau évalué
  (l'évaluer avant l'appel, la sync du pas l'a déjà calculé : coût nul) ; BenchUI : `eval(pixelValues)` avant le
  détachement. Source : pattern MLX « eval avant traversée » (lot A K-1, mémoire `feedback_mlx_array_threading`).

### D-09 — Entrées invalides : `precondition` / `fatalError` / corruption silencieuse — **VÉRIFIÉ**
- `maskedScatter` : `precondition(B == 1)` (`DiffusionGemmaEncoderModel.swift:154`) → crash avec une image et B > 1 ;
  ne place que les `K = source.dim(1)` premières positions d'image de la **première** image (`:161-173`) : deux
  images → la seconde est ignorée sans erreur ; moins de 280 jetons image dans le prompt → `argSort` complète avec
  des positions **non-image** qui sont écrasées. Le README de la doc affirme pourtant « multi-image testé » (chemin AR).
- `fatalError("inputs ou inputsEmbeds requis")` (`DiffusionGemmaEncoderTextModel.swift:86`),
  `EncoderKVCache.get` `fatalError` (`EncoderKVCache.swift:63-66`), `precondition` de longueur de masque
  (`DiffusionAttentionMask.swift:66-69`), `encoderCache!` (`DiffusionGemmaPipeline.swift:119, 154`).
- Correction : valider en tête de `generate` (nombre de jetons image = 280 × nombre d'images, B == 1 si image) et
  lever une erreur typée ; `maskedScatter` générique (même approche que le chemin AR `Gemma4Model.maskedScatter`).

### D-10 — `DiffusionMemoryConfig` à moitié câblé, `unloadVision` contradictoire — **VÉRIFIÉ**
- Voir A.6. De plus, le pipeline affirme que décharger la vision par affectation directe sur `@ModuleInfo` **plante**
  au canvas suivant (`DiffusionGemmaPipeline.swift:133-138`), alors que `unloadVision()` fait exactement cette
  affectation (`DiffusionGemmaEncoderModel.swift:55-59`) et que `profile-diffusion` l'appelle (`:317`). **À VÉRIFIER**
  (test sans poids : forward encodeur incrémental après `unloadVision()` sur une config minuscule). Correction propre :
  `update(modules:)` avec des modules vides, ou le mécanisme de résidence YuE2 (zéros non évalués, `WeightResidency`).
- Gain : ≈ 0,6-0,8 Go (vision SigLIP bf16) en `lean`, seulement pour les requêtes avec image. Porte : −≥ 0,5 Go de pic
  sur D2, temps ± 3 %.

### D-11 — Outil de profil hors du chemin bibliothèque — **VÉRIFIÉ** (piège catalogue 33)
- `ProfileDiffusionCommand` recopie la boucle (`:288-389`) et a dérivé : `prevLogits` en fp32 (`:374`, le pipeline
  est en bf16 depuis la Phase 7), EOS codé en dur `{1, 106}` au lieu de `eos_token_id` (`:385`, oublie 50),
  `eval` supplémentaire par pas (`:359`), décharge la vision (le pipeline non). Les chiffres « Phase 7 » ne mesurent
  donc pas exactement ce qu'exécutent la CLI `diffusion` et BenchUI.
- Les « Total time » des phases 5-7 incluent le chargement (et la quantification) : 63 × 0,605 s = 38 s de
  débruitage pour un total de 86 s. Les comparaisons de total mélangent chargement et génération.
- Correction : l'instrument appelle `DiffusionGemmaPipeline.generate` et chronomètre via `onStep` (déjà synchronisé
  par le pas) ; une ligne JSON au format de `gemma4-cli bench` (voir C.5).

### D-12 — Entropie calculée deux fois par pas, post-traitement non fusionné — **VÉRIFIÉ, gain CALCUL faible**
- `accept` et `shouldStop` recalculent chacun `tokenEntropy(scaled)` (`EntropyBoundSampler.swift:95-97`,
  `StableConfidentStopping.swift:60`) ; softcap, température, entropie font ≈ 20 passes sur 268 Mo par pas.
  `useCompiledEntropy` est `false` par défaut et jamais mesuré (aucune ligne publiée).
- Correction exacte : calculer l'entropie une fois et la passer aux deux (API additive `shouldStop(argmaxCanvas:entropy:)`) ;
  option : fusionner `softcap → /T → logSoftmax → entropie` dans une fonction compilée `shapeless` (T18).
- **Gain attendu : 1-4 % du pas** (CALCUL) → probablement sous le seuil de 5 % ; ne coder que la déduplication
  (exacte, triviale) et mesurer le compile une fois. Source : T18 (Y SnakeBeta −30 % sur une chaîne élémentaire
  dominante, ici non dominante).

### D-13 — Le self-conditioning coûte un GEMM vocabulaire complet par pas — **VÉRIFIÉ, levier HYPOTHÈSE**
- `softmax(prevLogits) @ embed_tokens.weight` : 256 × 262 144 × 2 816 ≈ 0,38 TFLOP par pas, autant que le head
  (`DiffusionGemmaDecoderTextModel.swift:93-114`). Les distributions sont très piquées après quelques pas (entropie
  moyenne < 0,005 à l'arrêt).
- Levier **approximatif** (change la numérique, porte qualité obligatoire) : top-k (p. ex. 64) des probabilités,
  `take` des lignes d'embedding, somme pondérée (≈ 0,1 % du GEMM). Gain potentiel 10-15 % du pas (CALCUL).
  À ne tenter qu'après D-01/D-03 et la base ; rejet si ScreenSpot-100 perd > 1 pt ou forwards +5 %.
- Levier **exact** à écarter : quantifier `embed_tokens` en 8 bits accélère-t-il ce GEMM borné calcul ? Probablement
  non (T14, « préfill 4 = 16 bits au-delà de 1 k jetons ») ; se mesure gratuitement avec K-D11.

### D-14 — Recopie du contexte encodeur à chaque pas et à chaque couche — **VÉRIFIÉ, levier REJETÉ par précédent**
- `concatenated([encoderEntry.keys, keys])` par couche et par pas (`DiffusionDecoderAttention.swift:150-151`) :
  CALCUL ≈ 0,5 Go recopiés par pas à 2,5 k jetons de contexte (≈ 1-3 ms, < 1 % du pas), borné à ≈ 0,2 Go après D-05.
  Même structure que le NAR de YuE2 : tampons K/V contigus (O8) mesurés **−1,6 %, retirés**. Ne pas coder.
  Idem pour la double concaténation de l'encodeur incrémental (1 × par canvas).

### D-15 — Pas de politique mémoire dans la bibliothèque — **VÉRIFIÉ**
- Aucun `cacheLimit`/`memoryLimit` hors CLI (A.6) ; le catalogue (T1/T2) et le lot E (`applyGlobalPolicy`) les
  posent pour l'AR. La conclusion Phase 7 « le pool refuse de se réduire » est à relire à la lumière de D-01 : la
  mémoire ne pouvait pas baisser, les experts bf16 étaient toujours référencés.
- Correction : champs `cacheLimitMB`/`memoryLimitMB` du profil diffusion (C), posés après chargement **et**
  quantification. `16bit-lean` garde les caches Mac (YuE2 : limites serrées = +73 % en bf16).

### D-16 — Nombre de pas : réglages existants jamais balayés — **VÉRIFIÉ (réglages), gain À MESURER**
- `maxDenoisingSteps` (48) n'est qu'un plafond (≈ 16 pas atteints). Les vrais leviers sont `confidenceThreshold`
  (0,005), `entropyBound` (0,1 : plus haut = plus de positions acceptées par pas), `stabilityThreshold` (1),
  `tMin/tMax` — tous dans `DiffusionGenerationConfig` et exposés en CLI pour `tMin/tMax/maxSteps` seulement.
- Analogue mesuré : YuE2 ODE 32 → 24 → 16 pas = 66-69 → 55,3 → 43,9 s (T21), qualité à valider à l'écoute.
- Règle : **hors profils de référence** (qui gardent `generation_config.json`, comme `odeSteps = nil` chez YuE2) ;
  variante documentée seulement si ScreenSpot-100 et BFCL-100 tiennent (± 1 pt) avec ≥ −15 % de forwards.

### D-17 — Synchronisation par pas : ne pas pipeliner — **VÉRIFIÉ, levier REJETÉ par calcul**
- Une seule sync par pas (`.item()`), sur un pas de ≈ 500 ms. Pipeliner (lancer le pas N+1 avant de lire l'arrêt du
  pas N) gaspille un forward complet à chaque fin de canvas (≈ 500 ms × 4) pour économiser la construction du graphe
  CPU (quelques ms × 16 pas/canvas) → perte nette. Phase 5 a mesuré `asyncEval` entre pas : ≈ 0. Garder tel quel ;
  vérifier seulement dans une trace que la part CPU du pas est < 5 % (sinon, rouvrir).

### D-18 — Pic transitoire et `evalEveryNLayers` jamais mesuré — **À MESURER**
- Commentaires : « ~440 Mo de pic par forward décodeur » ; `evalEveryNLayers = 8` proposé (motif Flux2/LTX), aucune
  ligne publiée. YuE2 `evalPerLayer` : numériquement identique, +40-60 Mo à L = 1 024, sert surtout de granularité
  pour une porte GPU (iOS). Pour la diffusion Mac : à mesurer en `lean` uniquement ; porte : pic −≥ 10 % pour
  ≤ +3 % de temps, sinon retiré.

### D-19 — Encodeur non tranché — **VÉRIFIÉ, priorité basse**
- Tout le prompt en un passage + masque `[T,T]` matérialisé (`DiffusionGemmaEncoderTextModel.swift:100`). Coût en
  temps négligeable (Phase 6 : encodeur incrémental 0,03 % du total), mais pic proportionnel à T² × couches avec le
  repli SDPA non fusionné. Transposer K-13 (tranches de 512 via `priorCache` : l'encodeur sait déjà étendre son
  cache) **si** D3 montre un pic d'encodeur > pic de débruitage. Porte K-13 : pic −≥ 20 %.

### D-20 — Débit multi-requêtes : lot de canvases — **HYPOTHÈSE**
- Le pas à 256 jetons est dans la zone où le MoE est mal amorti (tableau préfill `a4b`). Un lot B = 2-4 (requêtes
  distinctes, ou graines pour un vote) partagerait la lecture des experts. Le code accepte `B` mais l'arrêt est
  `.all()` sur le lot, l'EOS est testé sur tout le lot et la vision impose B = 1. Hors plan tant qu'aucun consommateur
  ne sert plusieurs requêtes (lot G serveur : K-41 est déjà « option »).

---

## C. Standard de profils diffusion `<bits>bit-fast|lean`

### C.1 Principe
Mêmes règles que `Gemma4ReferenceProfile` et YuE2 : une configuration nommée qui fige **tous** les réglages qui
comptent, chaque champ = un réglage **existant** ; ce qui manque est un constat (C.3), pas un champ inventé.
Poids source unique : `google/diffusiongemma-26B-A4B-it` bf16 (~50 Go, sur `/Volumes/Lexar`), 4 et 8 bits par
quantification à la volée (puis pack exporté, K-D12). Le débruitage (`generation_config.json`) est **identique dans
les six profils** : la vitesse ne s'achète pas avec la qualité dans un profil de référence (D-16).

### C.2 Champs (réglages existants)

| Champ | Réglage existant | `fast` | `lean` |
|---|---|---|---|
| `bits` | `nil` (bf16) / `DiffusionOnTheFlyQuantization.apply(bits:groupSize:mode:excludedPathPrefixes:)` / `applyMixedPrecision(config:)` | — | — |
| `quantization` | `.none` · `.uniform(bits, groupSize 64, .affine, excluded: multimodalEncoderPrefixes)` · `.mixed(MixedPrecisionConfig)` | selon bits | selon bits |
| `includeVision` | `DiffusionGemmaRegistration.load(includeVision:)` | oui | oui (variante `textOnly` = `false`) |
| `unloadVisionAfterFirstCanvas` | `DiffusionMemoryConfig` (**non câblé**, D-10) | non | oui |
| `clearCacheBetweenCanvases` | `DiffusionMemoryConfig` (**non câblé**, pipeline : toujours oui) | non | oui |
| `cacheLimitMB` | `Memory.cacheLimit` (CLI `--cache-limit-gb` seulement) | 4 096 | `min(1 024, max(256, dispo/6))`, sauf 16 bits : 4 096 |
| `memoryLimitMB` | `Memory.memoryLimit` (**absent** du chemin diffusion) | — | `max(4 096, dispo − 2 048)`, sauf 16 bits |
| `evalEveryNLayers` / `clearCacheOnEval` | propriétés du décodeur | 0 / non | 0 ou 8 selon D-18 |
| `useCompiledEntropy` | `EntropyBoundSampler` | selon K-D13 | idem |
| `generation` | `DiffusionGenerationConfig` du checkpoint (`nil` = fichier) | checkpoint | checkpoint |
| `maxBlocks` | argument par appel | — (appelant) | — |

`applyGlobalPolicy()` pose `cacheLimit`/`memoryLimit` après chargement **et** quantification ; le reste est lu par
`load(profile:)` et `makePipeline(profile:)`.

### C.3 Ce qui manque (constats, pas des champs)
1. Quantification des experts et du routeur 8 bits (D-01, D-03) — sans elle, `4bit`/`8bit` ne sont pas des profils
   mais un bf16 à 95 %.
2. Quantification unique partagée encodeur/décodeur (D-02).
3. Câblage de `DiffusionMemoryConfig` dans le pipeline, `unloadVision` sûr (D-10).
4. `memoryLimit`/`cacheLimit` dans la bibliothèque (D-15).
5. Pack pré-quantifié exporté (sinon chaque chargement 4/8 bits paie ≈ 50 Go de lecture bf16 + la quantification ;
   le premier run d'un banc est à jeter, catalogue §2.1).

### C.4 Matrice proposée (valeurs initiales, **toutes À MESURER**)

| Id | Poids | Quantification | Vision | Mémoire | Attendu (CALCUL / hypothèse) |
|---|---|---|---|---|---|
| `a4bdiff/16bit-fast` | bf16 ~50 Go | aucune | résidente | cache 4 Go, pas de clear | référence qualité ; pas ≈ 0,5-0,6 s ; pic ≈ 51 Go |
| `a4bdiff/16bit-lean` | bf16 | aucune | déchargée après canvas 0 | caches Mac (règle bf16), clear entre canvases, `evalEveryNLayers` selon D-18 | pic −0,5 à −1 Go, temps ± 3 % |
| `a4bdiff/8bit-fast` | 8 bits affine g64 (experts compris), routeur 8, vision bf16 | uniforme 8 | résidente | cache 4 Go | ≈ 27 Go ; pas −10 à −30 % |
| `a4bdiff/8bit-lean` | idem | uniforme 8 | déchargée | limites adaptatives | ≈ 26 Go |
| `a4bdiff/4bit-fast` | 4 bits g64 (experts), **routeur 8**, `embed_tokens`/head 8 bits (candidat), couches 0-3 et 26-29 en 8 bits si la qualité l'exige (`MixedPrecisionConfig.default` étendu aux experts) | uniforme 4 ou mixte | résidente | cache 4 Go | ≈ 15-17 Go ; pas −20 à −45 % ; porte forwards |
| `a4bdiff/4bit-lean` | idem | idem | déchargée | limites adaptatives | ≈ 15 Go ; seul profil pour un Mac 32 Go |

Choix du 4 bits : **uniforme + routeur 8** (recette mlx-community, 26B-A4B AR : 78,9 % OCRBench) contre **mixte
default** : trancher par la mesure K-D11 (scores + forwards), comme le 12B (K-21). Le groupe 32 / `mxfp4` n'entre
pas dans la matrice (un réglage de plus = une mesure de plus, rien ne l'appelle).

Ce que la matrice doit mesurer (par profil, workloads D1-D3 ci-dessous) : chargement (bf16 → quantifié, ou pack),
pas médian / p90 (ms), forwards par canvas, jetons/s effectifs, pic MLX, `phys_footprint`, ScreenSpot-100,
BFCL-100 (ou OCRBench-100), `profile_weights_match`.

Workloads : **D1** texte, 4 canvases, prompt court fixe (« Why is the sky blue? … 4 paragraphs », celui de
`profile-diffusion`) ; **D2** une image (capture ScreenSpot fixe) + consigne `CLICK:` ; **D3** contexte long
(≈ 2,5 k jetons : capture + `pageText` 2 000 caractères, comme l'agent web), 2 canvases — c'est lui qui exerce D-05.
Graine fixe (0).

### C.5 Exposition — recommandation : **type frère `DiffusionReferenceProfile`**

Ne pas étendre `Gemma4ReferenceProfile` :
- ses champs structurants n'ont pas de sens ici — `model: Gemma4Pipeline.Model` (packs mlx-community, n'existe qu'en
  bf16 pour la diffusion), `kvBits` et `prefillStepSize` (passent par `GenerateParameters`, que la diffusion n'utilise
  pas), `multimodal` (`load(multimodal:)`) ;
- `all` exclut volontairement `.a4bDiff` et les tests de `ReferenceProfileTests` portent sur 30 profils AR ; ajouter
  6 entrées aux champs optionnels vides rendrait `apply(to: GenerateParameters)` trompeur.

Type frère dans `Sources/Gemma4Swift/Configuration/DiffusionReferenceProfile.swift` :
- **réutilise** `Gemma4ReferenceProfile.Bits`, `.Kind`, `availableMemoryMB()` et le format d'id
  (`qualifiedID = "a4bdiff/4bit-fast"`), pour que `gemma4-cli references` et une app les listent ensemble ;
- champs de C.2 ; `applyGlobalPolicy()` ; `static let all` (6) ; `named(_:)` ; `recommended(availableMB:)`
  (même règle : le `fast` le plus large sous la moitié de la mémoire, sinon le `lean` le plus petit) ;
  `textOnlyVariant()` (`includeVision = false`) ;
- `DiffusionGemmaRegistration.load(from:profile:)` et `DiffusionGemmaContainer.makePipeline()` qui lit le profil
  (additif ; l'actuel `load(memoryConfig:)` reste).
- Option de factorisation (plus tard) : extraire un `ReferenceMemoryPolicy` (cacheLimitMB, memoryLimitMB,
  `applyGlobalPolicy`) commun aux deux types.
- `DiffusionMemoryConfig.recommended(forRAMGB:)` devient un alias du profil conseillé, et ses presets
  `moderate/aggressive/extreme` sont dépréciés (leurs cibles RAM sont fausses, D-01).

### C.6 Commande de mesure — recommandation : **étendre `gemma4-cli bench`**, pas `profile-diffusion`

- `profile-diffusion` réimplémente la boucle (D-11) : le corriger reviendrait à le réécrire. Le garder comme outil
  de trace Chrome (en le rebranchant sur `DiffusionGemmaPipeline.generate` + `onStep`), mais **ne pas** en faire
  l'instrument de référence.
- `gemma4-cli bench --reference a4bdiff/<id> --workload d1|d2|d3 --cooldown 120 --passes 2` : même format de ligne
  JSON que K-11 (commit du binaire +modifs, révisions, machine, `phys_footprint`, pic MLX), champs diffusion :
  `load_ms`, `quantize_ms`, `encode_ms`, `step_ms_median`, `step_ms_p90`, `forwards`, `forwards_per_canvas`,
  `canvases`, `tokens`, `tokens_per_s` (hors chargement), `eos_reached`. Temps de pas lus dans `onStep` (le pas
  est déjà synchronisé par `.item()` : aucune sync ajoutée).
- `Scripts/bench-campaign.sh` : ajouter la famille `a4bdiff` en fin de campagne, poids sur le Lexar
  (`--model-path /Volumes/Lexar/models/google/diffusiongemma-26B-A4B-it`, dossier réel avec liens **par fichier**,
  piège 37). Disque interne presque plein : les packs exportés (K-D12) vont aussi sur le Lexar.
- Qualité : scripts existants `docs/examples/ui-grounding-bench` (ScreenSpot 100) et `function-calling-bench`
  (BFCL 100), sous-ensembles fixes, lancés sur `16bit-fast` puis sur chaque profil retenu.

---

## D. Catalogue de patterns « modèles itératifs / diffusion »

Format proche des T-x. Préfixe **IT-**. Sources : **G** = gemma-4-swift-mlx (phases 5-7, ce audit), **Y** = YuE2
(NAR = flow matching, solveur midpoint, 32 pas × 2 évaluations, cache de préfixe). Statut : **MESURÉ** (chiffre et
source), **CALCUL**, **HYPOTHÈSE**.

### D.1 Retenus

#### IT-1. Conditionnement calculé une fois, réutilisé à chaque pas
- **Problème** : le contexte (prompt, préfixe AR) ne change pas pendant les N pas d'un bloc.
- **Mécanisme** : K/V du contexte (après normes et RoPE) calculés une fois et relus par chaque évaluation
  (`Y Synthesis/CachedNAR.swift:9-22`, G `EncoderKVCache`) ; extension incrémentale d'un bloc au suivant.
- **Gain MESURÉ** : Y O2 (réutiliser le cache de la phase sémantique comme préfixe NAR) −17 % sur chanson courte,
  **−1,2 %** sur la complète (`Y log.md:11`) ; G encodeur incrémental : 0,03 % du total (Phase 6). Leçon : le
  gain dépend du rapport longueur de contexte / (pas × taille de bloc) ; mesurer à la longueur réelle.
- **Risque** : positions (offset RoPE) et fenêtres glissantes du cache — un cache qui ne tronque pas les couches
  glissantes diverge de la référence au-delà de la fenêtre (G D-05).
- **Applicabilité** : tout modèle itératif conditionné (diffusion texte bloc-AR, flow matching audio, DiT vidéo avec
  texte encodé une fois).

#### IT-2. Une seule synchronisation par pas, annulation et porte GPU au même endroit
- **Mécanisme** : un `eval`/`.item()` par pas (Y `eval(state)` fin de pas, G `.item()` du critère d'arrêt) ;
  test d'annulation **avant** chaque évaluation et à mi-pas (Y `CachedNAR.swift:104-113`), attente de la porte GPU
  (Y `YuE2GPUGate`) au même point.
- **Gain MESURÉ** : G `asyncEval` entre pas : ≈ 0 (Phase 5). Y : un incident iOS ne coûte plus qu'un pas.
- **Rejet associé** (CALCUL, G D-17) : pipeliner le pas N+1 avant de lire l'arrêt du pas N quand l'arrêt est
  adaptatif : un forward gaspillé par bloc > gain de construction de graphe.
- **Applicabilité** : toutes les boucles de pas ; indispensable dès qu'un consommateur annule (UI, serveur).

#### IT-3. Le nombre de pas est le premier levier, et il est de la qualité
- **Mécanisme** : pas fixes (Y `odeSteps`) ou arrêt adaptatif (G entropie + stabilité).
- **Gain MESURÉ** : Y 32 → 24 → 16 pas = 66-69 → 55,3 → 43,9 s (`Y docs/Benchmarks.md` « ODE steps ») ; 8 bits packé
  20 pas 64,4 s contre 83,5-86,2 s à 32. G : arrêt adaptatif ≈ 16 pas atteints sur 48 max (63 forwards / 4 blocs).
- **Règle** : hors profils de référence (le profil garde les pas du checkpoint) ; variante publiée seulement après
  validation qualité (écoute Y, scores de tâche G).
- **Piège** : un critère d'arrêt mal porté change le nombre de pas sans que rien ne casse (G D-04 : « stable »
  toujours vrai) → tester le critère contre la référence au pas près.

#### IT-4. Compter les pas dans toute mesure de quantification (arrêt adaptatif)
- **Problème** : avec un arrêt sur l'entropie, un bruit de quantification retarde la convergence.
- **MESURÉ** (G, `diffusion-quantization-bench.md`) : pas +17 % mais **forwards ×2,6** → total ×2,9 (4 bits, routeur
  4 bits, experts bf16) ; mixte : pas −5 %, forwards +19 %, total +12 %.
- **Règle** : la métrique d'une quantification sur un itératif adaptatif est **temps par pas × pas**, jamais le pas
  seul ; porte « forwards ≤ référence + x % ».

#### IT-5. Vérifier la couverture réelle de la quantification (octets, pas modules)
- **Problème** : un filtre `Linear || Embedding` rate les couches MoE (`SwitchLinear` n'est pas un `Linear`).
- **MESURÉ indirectement** (G D-01) : 2,5 / 1,3 / 0,1 Go libérés sur 50 en 4 / 6 / 8 bits ; le verdict
  « la quantification ne sert à rien sur Apple Silicon » en a été tiré à tort.
- **Règle** : après quantification, comparer la somme des `nbytes` des paramètres au CALCUL attendu
  (param. × (bits + 32/groupe)/8) ; filtrer sur `Quantizable`. Test unitaire sur une config minuscule.
- **Applicabilité** : tout MoE (Gemma 26B, Qwen MoE, Mixtral…) quantifié à la volée.

#### IT-6. Poids liés entre deux voies : quantifier une fois, partager le triplet
- **Problème** : encodeur et décodeur partagent leurs poids ; la quantification par module crée deux copies.
- **Mécanisme** : quantifier au niveau du dictionnaire (une clé source → un triplet) avant duplication, ou
  réaffecter les tableaux de la voie A à la voie B.
- **Statut** : CALCUL (G D-02 : 4 bits 14 Go au lieu de 28 une fois D-01 corrigé). À MESURER.

#### IT-7. Précision par sensibilité : routeur et étage itéré plus hauts
- **MESURÉ** : Y NAR 4 bits rel 0,15 (budget 0,10) → rejeté, erreur composée sur 64 évaluations (`Y docs/Weights.md`) ;
  Y AR 8 bits seul −30,2 % de temps AR, rel 0,005 (`Y log.md:14`). Recette mlx-community 26B-A4B 4 bits : routeurs
  en 8 bits (VÉRIFIÉ, `config.json`). G : mixte `default` (4 premières / 4 dernières couches 8 bits) −5 % de pas.
- **Règle** : routeurs MoE ≥ 8 bits toujours ; une étape itérée N fois tolère moins de bits qu'une étape unique.

#### IT-8. Politique de calcul selon la borne de l'étape, mesurée à la taille réelle du pas
- **MESURÉ** : Y NAR 8 bits packé 58-61 s → déquantifié bf16 53-55 s (+1,4 Go) (`Y log.md:35`) ; G AR préfill
  4 = 16 bits au-delà de 1 k jetons (7-9 TFLOPS, T14 écarté, PLAN §7) mais ×2,6 à 128 jetons (References.md).
- **Règle** : un pas de diffusion traite un bloc (256 jetons chez G, L frames chez Y) : se placer sur la courbe
  préfill tok/s(L) du modèle **à cette taille** avant de choisir packé / déquantifié / bf16. HYPOTHÈSE G : à 256
  jetons sur MoE, packé 4 bits gagne (K-D11).

#### IT-9. État inter-pas dans le dtype des poids
- **MESURÉ** : G `prevLogits` bf16 (Phase 7) : pas −1,2 %, pic −1,7 Go (combiné) ; conforme à la référence Python.
  Y : constantes scalaires castées au dtype de l'état (`MLXArray(dt/2).asType(dtype)`, `CachedNAR.swift:111`).
- **Règle** : tout tenseur porté d'un pas à l'autre au dtype des poids ; fp32 seulement pour softmax/entropie
  locales ; jamais une constante fp32 qui promeut l'état (piège 26).

#### IT-10. Mémoire : `cacheLimit` par étape, `clearCache` entre blocs, pas entre pas
- **MESURÉ** : G `clearCache` entre pas : neutre (Phase 5) ; entre blocs : neutre sans déchargement ; Y
  `cacheLimit` NAR 4 Go : stable, sans régression (`Y log.md:10`) ; Y limites fixes serrées : +32 % (4 bits),
  +73 % (bf16) → limites adaptatives, bf16 garde les caches Mac.
- **Applicabilité** : profils `lean`.

#### IT-11. Résidence par étape (le budget = max des étapes)
- **MESURÉ** : Y 8 bits 11,5 → 4,3 Go, 4 bits 10,4 → 3,3-3,5 Go, **sans coût temps** (A/B/B/A) ; G : vision
  SigLIP ≈ 0,6 Go libérable après le 1ᵉʳ bloc (non câblé, D-10).
- **Piège** : affecter `nil` à un `@ModuleInfo` n'est pas la bonne API (G commentaire pipeline) ; zéros non évalués
  ou `update(modules:)`.

#### IT-12. Checkpoint au pas + reprise bit-exacte
- **MESURÉ** : Y `NARCheckpoint` 192 Ko, reprise au pas près bit-exacte ; un crash Metal (GPU retiré sous iOS) ne
  coûte plus qu'un pas (`Y log.md:30`). G : absent (canvas de 256 jetons ≈ 10 s en bf16 : checkpoint au bloc suffirait).
- **Applicabilité** : iOS et tout pas > 1 s.

#### IT-13. `eval` par couche dans le pas : granularité, pas vitesse
- **MESURÉ** : Y `evalPerLayer` identique numériquement, +40-60 Mo de pic à L = 1 024, −10 à −25 Mo à 256 ; utile
  pour la porte GPU. G `evalEveryNLayers` : non mesuré (D-18).

#### IT-14. Compile d'une chaîne élémentaire dominante
- **MESURÉ** : Y `SnakeBeta` compilée −30 % sur le VAE (`Y log.md:13`). G entropie compilée : option, jamais mesurée ;
  post-traitement des logits ≈ 2-3 % du pas (CALCUL) → probablement sous le seuil. Règle : compiler seulement ce qui
  domine une trace.

### D.2 Rejetés (et pourquoi)

| # | Technique | Mesure | Raison | Source |
|---|---|---|---|---|
| IT-R1 | Tampons K/V contigus pour éviter la concaténation du contexte à chaque évaluation | Y O8 : 0 % court, −1,6 % complet | < 5 % → retiré. Même structure chez G (D-14) : ne pas coder | `Y log.md:12` |
| IT-R2 | Tuilage des requêtes de l'étape itérée | Y O6 : pas d'allocation `[16, L, L]` persistante (SDPA fusionné) | garde-fou négatif, rien codé. **Chez G, SDPA non fusionné (head_dim 256/512)** : à reprofiler avant de rejeter | `Y log.md:15` |
| IT-R3 | CFG batché (deux branches dans un lot) | Y O9 : CFG inactif dans la config mesurée (`guidance == 1`) | ne batcher que ce qui est actif dans le workload mesuré ; G n'a pas de CFG | `Y log.md:15` |
| IT-R4 | `asyncEval` / pipeliner entre pas | G Phase 5 : ≈ 0 | le `.item()` du critère d'arrêt synchronise déjà ; un forward gaspillé par bloc | `diffusion-optim-phases.md` |
| IT-R5 | `clearCache` entre chaque pas | G Phase 5 : neutre | tampons réutilisés d'un pas au suivant (formes identiques) | idem |
| IT-R6 | « Quantification inutile sur mémoire unifiée » | G Phase 5 | **conclusion invalide** : experts non quantifiés, poids dupliqués (D-01/D-02). À rouvrir, pas à citer | ce document |
| IT-R7 | `Memory.cacheLimit` pour forcer la libération des bf16 après quantification | G Phase 7 : « le pool refuse » | les bf16 étaient toujours référencés (D-01) ; le levier mémoire est la couverture, pas le cache | idem |
| IT-R8 | Compile du pas complet / par couche | Y R1 : 103-113 contre 114-115 tok/s ; Q R2/R3 | borné dispatch (AR) ; pour un pas de diffusion borné calcul, rien n'indique un gain | catalogue R1-R3 |
| IT-R9 | Étape itérée en 4 bits | Y NAR rel 0,15 | erreur composée ; G : à re-mesurer une fois D-01/D-03 corrigés (le 4 bits G n'a jamais été testé) | `Y docs/Weights.md` |
| IT-R10 | Backend alternatif (Core AI GPU) pour l'étape itérée | Y : 5-8× plus lent que MLX | règle « > 2× MLX ⇒ pas d'asset » | `Y log.md:17-18` |

### D.3 Hypothèses à confirmer (pour le catalogue, marquées comme telles)
- **IT-H1** Self-conditioning parcimonieux (top-k des probabilités × lignes d'embedding) : −10-15 % du pas (CALCUL),
  approximatif (G D-13).
- **IT-H2** Lot de blocs/requêtes pour amortir la lecture des experts MoE à petite taille de pas (G D-20).
- **IT-H3** Pack pré-quantifié exporté une fois (Y `docs/Weights.md`) : supprime ≈ 50 Go de lecture bf16 et la
  quantification à chaque chargement (G C.3-5) — gain de chargement, pas de pas.

---

## E. Plan — fiches K-D

Ordre : correction → stabilité → instrument → base → quantification → leviers → profils. Chaque fiche : un commit,
test qui échoue sans le correctif (piège 38), `Scripts/run-tests.sh`. Les fiches **sans GPU** se codent pendant la
campagne en cours ; les fiches **GPU** se lancent sur le M3 Max 96 Go au repos (`machine-check.sh --cooldown 120`),
bf16 officiel sur `/Volumes/Lexar/models/google/diffusiongemma-26B-A4B-it` (dossier réel, liens par fichier si
besoin d'un chemin dans `~/Library/Caches/models`), packs exportés sur le Lexar.

### E.1 Sans GPU (code + tests sur configs minuscules et poids aléatoires seedés)

| Fiche | Objet | Source | Porte | Effort |
|---|---|---|---|---|
| K-D1 | Critère d'arrêt comparé à l'historique **précédent** (Python) | D-04 | test : 1ᵉʳ pas jamais stable ; 2 argmax égaux + entropie basse → stop ; échoue sur l'ancien code | S |
| K-D2 | Fenêtre glissante : masque glissant encodeur, entrées glissantes du cache tronquées à `window − 1`, offset RoPE explicite (plus `seqLength` de la couche 0) | D-05 | test fenêtre 8 : sortie décodeur = référence naïve (erreur rel. L2 < 1e-3, piège 40) à 20 jetons ; inchangée à 6 jetons | M |
| K-D3 | Quantification correcte : filtre `Quantizable` (experts), routeur ≥ 8 bits, triplet unique partagé encodeur/décodeur/head, `MixedPrecisionConfig` étendu aux experts | D-01, D-02, D-03 | test : `nbytes` total = CALCUL ± 2 % ; poids encodeur et décodeur = même tampon ; `router.proj` en 8 bits | M |
| K-D4 | `generate` annulable (test par pas), sous `Gemma4ComputeGate`, `eval` des résultats avant retour et des tableaux passés à `onStep` ; BenchUI : `eval(pixelValues)` avant `Task.detached` | D-07, D-08 | test : annulation après 1 pas → fin en < 2 pas ; entraînement actif → refus typé | S |
| K-D5 | Entrées validées : nombre de jetons image = 280 × images, B == 1 avec image, `maskedScatter` multi-images ou erreur ; `fatalError`/`precondition` → erreurs typées | D-09 | tests : 2 images → erreur explicite (ou 2 images placées) ; 279 jetons image → erreur | S |
| K-D6 | `DiffusionMemoryConfig` câblé (`makePipeline`, `clearCacheBetweenCanvases`, `unloadVisionAfterFirstCanvas` via `update(modules:)`), presets honnêtes | D-10 | test sans poids : forward incrémental après déchargement OK ; le flag `false` n'appelle pas `clearCache` | S-M |
| K-D7 | Entropie calculée une fois par pas (exact) | D-12 | test : `shouldStop` avec entropie fournie = ancien résultat | S |
| K-D8 | Instrument : `gemma4-cli bench --reference a4bdiff/<id> --workload d1|d2|d3` sur `DiffusionGemmaPipeline.generate` (ligne JSON C.6) ; `profile-diffusion` rebranché sur le pipeline (plus de boucle recopiée) | D-11 | build ; exécution à blanc sur config minuscule ; champs JSON présents | M |
| K-D9 | `DiffusionReferenceProfile` (C.5), `load(from:profile:)`, `gemma4-cli references` liste `a4bdiff/*` | C | tests : 6 profils, `named`, `recommended` (16/32/64/96 Go simulés via `GEMMA4_AVAILABLE_MB`) | M |
| K-D10a | Docs vraies : aide CLI `--quantize-bits`, en-tête `DiffusionOnTheFlyQuantization`, `DIFFUSIONGEMMA.md` (~70 %), `diffusion-optim-phases.md` annoté (verdict Phase 5 invalidé par D-01) | D-01, D-15 | relecture | S |
| K-D16 | Encodeur : blocs vision bidirectionnels (réutiliser le masque du 12B Unified) | D-06 | test : masque = référence sur une séquence texte/image/texte | M |

### E.2 Avec GPU (mesure ; dans cet ordre)

| Fiche | Objet | Source | Porte | Effort |
|---|---|---|---|---|
| K-D10 | A/A de l'instrument (`16bit-fast`, D1) puis **base** bf16 D1/D2/D3 **avant et après** K-D1/K-D2/K-D16 (effets de correction chiffrés : forwards/canvas, scores) + ScreenSpot-100 et BFCL-100 sur `16bit-fast` corrigé | K-D8 | A/A ≤ 3 % (médiane du pas) ; lignes `BENCHMARKS.md` ; ScreenSpot ≥ 77 % (tolérance −2 pts sur 79 % publié) | M |
| K-D12 | Export des packs 4 et 8 bits (K-D3) sur le Lexar (safetensors + SHA-256, `format=gemma4-diffusion-prequantized-v1`) et chargement direct | IT-H3, C.3-5 | rechargement = sorties identiques à la quantification à la volée (même graine) ; chargement ≤ ⅓ du bf16 | M |
| K-D11 | Quantification corrigée : 8 bits uniforme, 4 bits uniforme + routeur 8, 4 bits mixte (experts compris) ; A/B/B/A contre `16bit-fast` sur D1/D2 | D-01, D-03, IT-4, IT-8 | 4 bits : pic ≤ 18 Go **et** (pas × forwards) ≤ −15 % **et** ScreenSpot ≥ bf16 − 2 pts, BFCL ≥ bf16 − 1 pt ; sinon mixte ; sinon 8 bits seul publié | M |
| K-D13 | Leviers un par un (A/B/B/A, retirés si < 5 %) : (a) `useCompiledEntropy` ; (b) `evalEveryNLayers` 8 en `lean` (porte : pic −≥ 10 %, temps ≤ +3 %) ; (c) déchargement vision D2 (pic −≥ 0,5 Go) ; (d) `cacheLimit`/`memoryLimit` lean (empreinte ≤ actif + cacheLimit, temps ± 5 %) | D-12, D-18, D-10, D-15 | chacun sa porte | M |
| K-D14 | Matrice 6 profils × D1/D2/D3, 2 passes, + qualité sur les profils 8/4 bits ; section « DiffusionGemma » de `docs/References.md` ; décision `docs/knowledge/decisions/reference-profiles.md` | C | table complète ; un profil non mesuré n'est pas publié | M-L |
| K-D15 | Variante de pas (hors profils) : balayage `entropyBound` 0,1/0,2/0,4, `confidenceThreshold` 0,005/0,02 sur `4bit-fast` | D-16, IT-3 | publiée seulement si forwards −≥ 15 % à scores ± 1 pt | M |
| K-D17 | *(option)* D-19 encodeur tranché si D3 montre un pic encodeur > pic du débruitage ; D-13 self-conditioning top-k ; D-20 lot de canvases | D-19, D-13, D-20 | portes des constats | M chacun |

Dépendances : K-D8 → K-D10 → K-D11 → K-D13 → K-D14 ; K-D3 → K-D12 → K-D11 ; K-D1/K-D2/K-D16 avant K-D10
(sinon la base mesure un critère et des masques différents de la référence). Disque : bf16 50 Go + packs ≈ 42 Go
sur le Lexar ; rien sur le disque interne.

Mise à jour du PLAN : la ligne « Hors plan : DiffusionGemma (a ses propres préréglages) » (§6) devient ce lot ;
ces préréglages sont justement à revoir (D-01, D-10).

---

## Résumé

1. Chemin : encodeur (prompt complet puis +256 jetons par canvas, cache K/V) → ≤ 48 pas de débruitage par canvas de 256 jetons (≈ 16 atteints) → commit ; une sync `.item()` par pas.
2. Un pas = préfill de 256 jetons du 26B-A4B + deux GEMM vocabulaire (head, self-conditioning ≈ ⅓ des FLOP) ; mesuré 505-605 ms en bf16.
3. **D-01 (majeur)** : la quantification à la volée ne touche pas les experts MoE (`SwitchLinear` n'est pas un `Linear`) → 88 % des poids restent bf16 ; les 2,5/1,3/0,1 Go libérés publiés le confirment.
4. **D-02** : encodeur et décodeur partagent leurs poids en bf16 mais sont quantifiés en deux copies.
5. **D-03** : le routeur passe en 4 bits (le pack mlx-community le garde en 8) : cause probable des ×2,6 forwards.
6. Le verdict des phases 5-7 « quantifier ne sert à rien sur Apple Silicon » est donc invalide : 4 bits attendu ≈ 15 Go au lieu de 50, pas plus rapide (à mesurer).
7. **D-04** : critère d'arrêt : « stable » toujours vrai (ajout avant comparaison, Python compare avant) → arrêt plus précoce que la référence.
8. **D-05** : fenêtre glissante (1 024) ignorée à l'encodeur, dans le cache et au décodeur → sortie différente au-delà de 1 k jetons de contexte ; **D-06** : blocs vision bidirectionnels absents.
9. Stabilité : génération non annulable (BenchUI annule pour rien), hors `Gemma4ComputeGate` (risque ABBA), tableaux non évalués qui traversent, `precondition`/corruption silencieuse multi-image.
10. `DiffusionMemoryConfig` à moitié câblé ; `profile-diffusion` a dérivé du pipeline (prevLogits fp32, EOS) ; « total » publié inclut le chargement.
11. Rejetés par précédent ou calcul : buffers K/V contigus (YuE2 −1,6 %), pipelining inter-pas, `clearCache` par pas ; entropie en double = 1-4 % seulement.
12. Profils : type frère `DiffusionReferenceProfile` (réutilise `Bits`/`Kind`/`availableMemoryMB`), 6 profils bf16/8/4 × fast/lean, débruitage du checkpoint inchangé.
13. Mesure : extension de `gemma4-cli bench` sur le pipeline bibliothèque, workloads D1 texte / D2 image / D3 contexte long, qualité ScreenSpot-100 + BFCL-100.
14. Catalogue : 14 patterns IT retenus, 10 rejetés, 3 hypothèses, sources G et YuE2 distinguées (mesuré/calcul/hypothèse).
15. Plan : 11 fiches sans GPU (K-D1…K-D9, K-D10a, K-D16) codables pendant la campagne ; 6 fiches GPU (K-D10 base → K-D12 packs → K-D11 quantification → K-D13 leviers → K-D14 matrice → K-D15 pas).
16. Porte phare K-D11 : 4 bits ≤ 18 Go, (pas × forwards) −15 %, ScreenSpot ≥ bf16 − 2 pts.
