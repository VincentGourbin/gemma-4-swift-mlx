# Audit de stabilisation et de nettoyage — gemma-4-swift-mlx

- Date : 2026-09-27, HEAD `c9543739` (main, propre). Dernier tag distant : `1.7.3` (`fd57edea`) ; `c9543739` (bump deps) n'est pas taggé. Le tag `1.7.3` n'est pas encore récupéré en local (`git describe` → `1.7.2-2-g…`).
- Périmètre : lecture seule. Aucun build ni test lancé. Les warnings cités viennent du **dernier log de build existant** (`.build/xcode/Logs/Build/E16435DD-….xcactivitylog`, 15/09, postérieur au dernier commit touchant `Sources/` du 12/09) ; ils ne sont donc pas re-mesurés.
- Dépendances résolues (Package.resolved = checkouts) : mlx-swift 0.31.6, mlx-swift-lm 3.31.4 (`bd4b743`), swift-transformers 1.3.4, swift-mlx-profiler 1.5.0.
- Consommateurs externes trouvés en local (`grep -rl "import Gemma4Swift"` dans `~/Developpements`) : **5, pas 2** — Fluxforge Studio (épinglé sur la branche `main`), ltx-video-swift-mlx (`from: 1.5.0`), flux-2-swift-mlx (`from: 1.5.0`), h3-swift-mlx (`from: 1.0.0`), ToolsForge (`script macos/`). Toute cassure d'API impose donc une 2.0.0.

Légende : **VÉRIFIÉ** = constaté dans le code ou par une commande reproductible ; **À VÉRIFIER** = déduit du code, l'effet à l'exécution reste à mesurer.
Effort : S (< ½ j), M (½–2 j), L (> 2 j).

---

## Synthèse des constats

| ID | Sév. | Thème | Titre court | Statut |
|----|------|-------|-------------|--------|
| S-01 | haute | D | Aucun stream n'est annulable : la génération continue après l'arrêt du consommateur | VÉRIFIÉ |
| S-02 | haute | D | Téléchargement interrompu considéré comme « téléchargé », jamais repris | VÉRIFIÉ (logique) |
| S-03 | haute | D | `chatStreamMultimodal` fait traverser un graphe MLX paresseux vers une autre file | VÉRIFIÉ |
| S-04 | haute | D/F | Le deadlock ABBA (`evalLock`) n'est ni empêché ni signalé dans l'API publique | VÉRIFIÉ |
| S-05 | moyenne | D | L'enregistrement global last-write-wins rend deux `loadContainer` concurrents non déterministes | VÉRIFIÉ (logique) |
| S-06 | moyenne | D | `download(modelId:)` renvoie un répertoire qui peut ne pas exister | VÉRIFIÉ (logique) |
| S-07 | moyenne | D | CLI `download --force` sans effet, paramètre `to:` ignoré | VÉRIFIÉ |
| S-08 | moyenne | D | `load(from:)` garde l'ancien modèle en mémoire et l'état devient incohérent | VÉRIFIÉ |
| S-09 | moyenne | D | Pointeurs temporaires (comportement indéfini) dans la FFT audio | VÉRIFIÉ (warning compilateur) |
| S-10 | moyenne | D | `processVideo` et `processAudio` rendent des `MLXArray` paresseux depuis un contexte async | VÉRIFIÉ |
| S-11 | moyenne | D | Décodage token par token (MTP, CLI) : texte potentiellement corrompu | VÉRIFIÉ code / effet À VÉRIFIER |
| S-12 | moyenne | D | `precondition` / `as!` sur des entrées d'API publique | VÉRIFIÉ |
| S-13 | moyenne | D/F | `GenerateParameters.kvBits` active TurboQuant, ce qui contredit le README | VÉRIFIÉ code / effet À VÉRIFIER |
| S-14 | moyenne | D | `prepare()` ignore `windowSize` : pas de prefill par tranches sur le chemin texte | VÉRIFIÉ code / impact À VÉRIFIER |
| S-15 | moyenne | F | Collisions de noms publics avec MLXLLM, MLXVLM et MLXLMCommon | VÉRIFIÉ |
| S-16 | moyenne | F | 12B Unified : annoncé `anyToAny`, mais multimodal impossible via `Gemma4Pipeline` ; docs périmées | VÉRIFIÉ |
| S-17 | moyenne | F | `Model` inclut DiffusionGemma, que `load` refuse ; il sort pourtant de `recommended()` et de `download --all` | VÉRIFIÉ |
| S-18 | moyenne | F | Plateforme : Package exige macOS 15, la doc annonce macOS 14 | VÉRIFIÉ |
| S-19 | moyenne | F | Consommateur h3-swift-mlx charge via la fonction libre `loadModelContainer`, le chemin piégé | VÉRIFIÉ |
| S-20 | moyenne | B | Six constructeurs de prompt divergents (artefacts jinja non corrigés sur 3 chemins) | VÉRIFIÉ |
| S-21 | moyenne | B | Boucles de génération dupliquées (Pipeline ×3, CLI ×2 manuelles, MTP) | VÉRIFIÉ |
| S-22 | basse | B | Code mort confirmé (≈ 400 lignes) | VÉRIFIÉ |
| S-23 | basse | B | Doublons : téléchargeurs, entrées LoRA, chargeurs diffusion, `systemRAMGB` ×3 | VÉRIFIÉ |
| S-24 | basse | B | Commandes CLI de diagnostic exposées en production | VÉRIFIÉ |
| S-25 | moyenne | C | Redondance avec le Gemma 4 amont : quoi garder, quoi retirer | VÉRIFIÉ (inventaire) |
| S-26 | basse | D | `Gemma4TokenFilter` : `@unchecked Sendable` sur un état mutable non protégé | VÉRIFIÉ |
| S-27 | basse | D | Globaux `nonisolated(unsafe)` mutables publics (`customModelsDirectory`) | VÉRIFIÉ |
| S-28 | basse | D | `snapshots.last` sur un ordre de répertoire non spécifié | VÉRIFIÉ |
| S-29 | basse | D | Mémoire : `Memory.cacheLimit` jamais fixé, `unload()` ne libère pas pendant un stream | VÉRIFIÉ code / impact À VÉRIFIER |
| S-30 | basse | D | `try!`, `fatalError` inatteignable, `print` dans la bibliothèque | VÉRIFIÉ |
| S-31 | moyenne | E | Sous-systèmes sans aucun test (encodeurs, TurboQuant, diffusion, MTP pipeline, 12B) | VÉRIFIÉ |
| S-32 | basse | E | Test tautologique ; suite qui mute un global sans `.serialized` ; pas de CI | VÉRIFIÉ |
| S-33 | basse | F | Surface publique de 935 déclarations, modules expérimentaux entièrement publics | VÉRIFIÉ |
| S-34 | basse | F | Incohérences de nommage et de langue, pas de CHANGELOG | VÉRIFIÉ |
| S-35 | basse | G | 107 warnings au dernier build, dont 40 dans la bibliothèque | VÉRIFIÉ (log du 15/09) |
| S-36 | basse | A | Hygiène : images en double, 1 051 fichiers de résultats de bench suivis, chemins absolus | VÉRIFIÉ |
| S-37 | basse | A | Hygiène locale : pack `.git` de 578 Mio d'objets inatteignables, branches de suivi périmées | VÉRIFIÉ |

---

## A. Hygiène du dépôt

### S-36 — Fichiers suivis superflus ou en double (basse, VÉRIFIÉ)
**Prémisse corrigée :** `default.profraw` (5,5 Mo) et `gemma4_gemma-4-e2b-it-4bit_trace.json` ne sont **pas** suivis. `.gitignore` couvre `*.profraw` et `*_trace.json` ; `git status --ignored` les liste en `!!`. Ce sont des résidus locaux, rien à faire dans git (un `rm` local suffit).

Constats réels :
1. **Images en double à la racine** : `UI.png` (1,1 Mo) et `input_sample.jpg` (1,0 Mo) sont identiques octet pour octet à `docs/examples/vision-image-description/` (même SHA-1 `8df9117c…` et `051bc537…`). Seule la copie de `docs/` est référencée (par `docs/examples/vision-image-description/README.md:19,25`). Rien dans `README.md`, `Sources/` ou `Tests/` ne pointe vers les copies de la racine.
   - Preuve : `shasum UI.png docs/examples/vision-image-description/UI.png`.
   - Correction : `git rm UI.png input_sample.jpg`. Risque : aucun. Effort S.
2. **1 266 fichiers suivis, dont 1 051 sous `docs/examples/`**, 1 004 rien que dans `ocr-bench/`. Ce sont 900 fichiers de résultats par échantillon (`results_300/{a4b,diff,e4b}/…`, `results/…`), soit 4,8 Mo. `summary_1000.csv` et les `*_summary.json` en portent déjà l'agrégat.
   - Preuve : `git ls-files docs/examples | awk -F/ '{print $3}' | sort | uniq -c`.
   - Correction : garder README + summaries + scripts, déplacer les résultats bruts en asset de Release (ou archive `.tar.zst`). Risque : aucun pour l'API. Effort S.
3. **Trois emplacements de benchmarks** : `BENCHMARKS_python_scripts/` (6 scripts à la racine), `benchmarks/` (16 résultats + `run_benchmarks.sh`) et `docs/benchmarks/`. S'y ajoutent `BENCHMARKS.md` et `examples/birdcall-adapter/` (3 fichiers, dont un `.py`).
   - Correction : regrouper sous `docs/benchmarks/{scripts,results}` et `docs/examples/`, et garder la racine limitée au code, au manifeste et à la doc principale. Effort S.
4. **Chemins absolus personnels dans des fichiers suivis** : `git grep -c "/Users/vincent"` touche 1 030 fichiers. Les scripts concernés ne tournent donc que sur ta machine : `benchmarks/run_benchmarks.sh:18-19` (média dans `~/Pictures`, `~/Downloads`), `docs/examples/ocr-bench/run_bench*.sh`, `function-calling-bench/run_bfcl100.py:6`, `toolathlon-bench/proxy.py:31`, `ui-grounding-bench/run_ss.py:10`. Aucun secret trouvé (`git grep -E "hf_[A-Za-z0-9]{30,}"` : 0 résultat).
   - Correction : dériver le chemin du CLI de `$(git rev-parse --show-toplevel)` ou d'une variable d'environnement. Effort S.
5. `docs/examples/toolathlon-bench/__pycache__/` est ignoré via un `.gitignore` local : OK.

### S-37 — Hygiène git locale (basse, VÉRIFIÉ)
- Le pack local `pack-3075f525….pack` pèse **578 Mio**. Ses plus gros blobs (107, 105, 82, 80, 69 Mo) sont **inatteignables** (`reach=0`). Le dépôt distant ne fait que 4,2 Mo (`gh api repos/VincentGourbin/gemma-4-swift-mlx --jq .size` → 4236). Il s'agit donc d'un reliquat local (historique réécrit), sans impact sur les consommateurs.
  - Correction : `git reflog expire --expire=now --all && git gc --prune=now`, après avoir vérifié qu'aucune stash ou branche locale n'en a besoin. Effort S.
- Des branches de suivi périmées subsistent (`origin/chore/bump-deps-swift-nio-cve`, `origin/fix/model-factory-vlm-collision`, absentes de `git ls-remote`). Correction : `git fetch --prune --tags`, ce qui récupère aussi `1.7.3`.
- Tag `multimodal-lora-v1` non semver : il trie en tête de `git tag --sort=-v:refname`. Sans conséquence pour SwiftPM, mais il pollue les listings.

---

## B. Code mort, expérimental et dupliqué

### S-22 — Code mort confirmé (basse, VÉRIFIÉ)
Méthode : pour chaque déclaration, j'ai compté les occurrences avec `grep -rlw` dans `Sources/` et `Tests/` hors du fichier qui la déclare, puis relu les occurrences à l'intérieur de ce fichier. Les symboles suivants ne sont référencés **que par leur propre déclaration**. Aucun des 5 consommateurs locaux ne les utilise non plus (grep croisé).

| Symbole | Fichier | Lignes | Remarque |
|---|---|---|---|
| `Gemma4MultimodalModel` (+ `getInputEmbeddings`) | `Multimodal/Gemma4Model.swift` | 91 | Ancien wrapper d'avril, audio « à implémenter en Phase 4 ». Remplacé par `Gemma4MultimodalLLMModel` |
| `TurboQuantProdCodec` | `TurboQuant/TurboQuantProdCodec.swift` | 138 | Codec QJL jamais instancié |
| `DiffusionGemmaRegistration`, `DiffusionGemmaContainer`, `makePipeline` | `Diffusion/Pipeline/DiffusionGemmaRegistration.swift` | 145 | Second chemin de chargement ; le CLI et la BenchUI utilisent `DiffusionGemmaLoader` |
| `rhtPaddedDim`, `nextPowerOfTwo`, `isPowerOfTwo` | `TurboQuant/TurboQuantUtils.swift:84,91,341` | — | RHT désactivé (« butterfly cassé », cf. mémoire) |
| `weightedSumFromScores` | `TurboQuant/TurboQuantMSECodec.swift:167` | — | |
| `innerState` | `TurboQuant/TurboQuantKVCache.swift:106` | — | |
| `poolMask` (variable) | `VisionEncoder/VisionModel.swift:112` | — | Warning « never used » |

- Correction : supprimer les trois fichiers et les fonctions listées. **Risque d'API** : ces symboles sont `public`, donc leur suppression est techniquement une cassure SemVer. Aucun consommateur connu ne les utilise. À faire en 2.0.0, ou avec `@available(*, deprecated)` en 1.8.0 puis suppression. Effort S.
- **À VÉRIFIER** : `continueChatStream` n'est appelé nulle part dans le dépôt, mais Fluxforge l'utilise (`grep` → `.continueChatStream(`). Il est **vivant** et ne doit pas être retiré.

### S-20 — Constructeurs de prompt divergents (moyenne, VÉRIFIÉ)
Six chemins construisent un prompt Gemma 4, avec des résultats différents :
1. `Gemma4Pipeline.chat` / `chatStream` sans option → `ChatSession` → processor mlx-swift-lm. **Garde l'artefact swift-jinja `<bos>\n`**, comme la mémoire le reconnaît (« Le chemin texte (ChatSession) porte encore le premier »).
2. `chatStreamBypassingSession` sans `templateVariables` → `context.processor.prepare` (`Gemma4Pipeline.swift:481-486`) : même artefact.
3. `chatStreamBypassingSession` avec `templateVariables` → `Gemma4Processor.textChatIds` : artefact **corrigé**. Pour un même prompt, passer `templateVariables: [:]` au lieu de `nil` change donc les ids.
4. `chatStreamMultimodal` → `Gemma4Processor.multimodalChatIds` : corrigé.
5. `Gemma4MTPPipeline.runLoop` → `context.tokenizer.applyChatTemplate(messages:)` brut (`Gemma4MTPPipeline.swift:150-153`) : non corrigé, sans système.
6. LoRA → `applyGemma4ChatTemplate` construit à la main (`LoRA/Gemma4LoRAData.swift:70`) ou `chatFormatter` du CLI. Le CLI `describe` construit aussi ses ids lui-même (`Gemma4CLI.swift` ~l.780-810).

Conséquence : pour un même texte, les ids diffèrent selon le chemin, alors que le README et la mémoire LTX présentent la parité HF comme un objectif.
- Correction : un seul point d'entrée `Gemma4Processor.chatIds(messages:system:templateVariables:media:)` appliquant `strippingTemplateArtifacts`, utilisé par tous les chemins. Le chemin `ChatSession` peut passer par un `UserInputProcessor` Gemma 4 dédié, renvoyé par le `ModelContext` du loader.
- **Risque** : cassure de comportement, pas d'API. Sur les chemins 1, 2 et 5, les ids changent d'un `\n` : même type de rupture que celle assumée en 1.3.0, à annoncer (LTX mesure ses captions à l'octet près). Effort M.

### S-21 — Boucles de génération dupliquées (moyenne, VÉRIFIÉ)
- `Gemma4Pipeline.swift` : trois blocs `AsyncThrowingStream { Task { … perform … switch generation … MainActor.run { state = .ready } } }` quasi identiques (412-426, 467-531, 607-694), plus un quatrième pour `continueChatStream` (719-733).
- `Gemma4CLI.swift` : deux boucles de décodage **manuelles** (`Describe.run` ~l.843-890 et `runUnified` ~l.1100-1135). Elles réimplémentent argmax et échantillonnage (sans `topP`, alors que l'affichage l'annonce), le décodage token par token (S-11) et le budget thinking ×3. Elles contournent `TokenIterator`, ce qui les rend plus lentes (pas d'`asyncEval` pipeliné) et non comparables à la bibliothèque.
- Correction : un helper privé `makeStream(container:build:)` dans le Pipeline, qui porte aussi la correction S-01 en un seul endroit ; faire passer le CLI `describe` par `Gemma4Pipeline` ou par `TokenIterator`. Risque : aucun pour l'API. Effort M.

### S-23 — Doublons de moindre gravité (basse, VÉRIFIÉ)
- **Téléchargement** : `Gemma4ModelDownloader` (enum, one-shot) et `Gemma4DownloadManager` (`@Observable`) partagent `runFileLoop`, ce qui est acceptable et documenté. En revanche `Gemma4CLI/LocalModelDownloader.swift` est un alias qui **perd des paramètres** (S-07), et `LocalTokenizerLoader.swift` est un `typealias` d'une ligne : les deux sont à supprimer.
- **LoRA** : deux points d'entrée publics, `Gemma4LoRATrain.train(...)` (namespace) et la fonction libre `trainLoRA(...)` (`Gemma4TrainingLoop.swift:153`), idem pour `trainMultimodalLoRA`. S'y ajoutent les fonctions libres publiques `loadGemma4TrainingData`, `loadGemma4MultimodalJSONL`, `applyGemma4ChatTemplate` et `maskedScatter`, qui polluent l'espace de noms global des consommateurs.
- **Diffusion** : `DiffusionGemmaLoader` et `DiffusionGemmaRegistration` (le second est mort, S-22).
- **`systemRAMGB` ×3** : `Gemma4ModelCache.swift:22`, `Gemma4Pipeline.swift:208` (`Model.systemRAMGB`), `DiffusionGemmaRegistration.swift:125`.
- **`Model.displayName` / `quantization`** : deux `switch` dupliqués (`Gemma4Pipeline.swift:81-97` et `153-160`).
- Correction : garder un seul symbole et marquer les autres `@available(*, deprecated, renamed:)`. Risque : aucun en dépréciation. Effort S.

### S-24 — Commandes CLI de diagnostic exposées (basse, VÉRIFIÉ)
`mtp-smoke` (« valide le chargement »), `mtp-forward` (« validation Jalon B »), `mtp-diag-verify` (« Diagnostic ») et `profile sweep` (« TurboQuant vs Standard », sur une fonctionnalité en cours de migration) figurent dans `subcommands` (`Gemma4CLI.swift:48`). Côté API, `Gemma4MTPPipeline.mtpStream(sequentialVerify:)` est un paramètre public documenté « DIAGNOSTIC ONLY — perf catastrophique ».
- Correction : `shouldDisplay: false` ou un groupe `debug`, et sortir `sequentialVerify` de l'API publique (`@_spi(Diagnostics)`). Risque : faible (le CLI n'est pas une API). Effort S.

### Modules expérimentaux (voir aussi S-13 et S-33)
- **TurboQuant** : le README (l.9 et 865-876) le dit « Migrating… retained for reference but not used in the default inference pipeline ». C'est faux dès que `kvBits` est passé (S-13). 7 fichiers, 64 déclarations publiques, `MLXFastKernel` déprécié à 7 endroits.
- **Diffusion** : 184 déclarations publiques, `precondition(B == 1, "… (Phase 6)")` (`DiffusionGemmaEncoderModel.swift:154`), `TODO Phase 10` (`DiffusionGemmaPipeline.swift:138`). Aucun consommateur ne l'utilise ; elle sert au CLI et à la BenchUI.
- **Gemma4BenchUI** : 17 fichiers et 7 400 lignes, livrés comme **produit** du paquet (`Package.swift:10`), dont un pilote de simulateur iOS et un agent web. C'est un outil de labo, pas une partie de la bibliothèque.
- Correction proposée : sortir la BenchUI dans un paquet séparé (ou un dossier `Apps/` avec son propre `Package.swift`), et exposer TurboQuant et Diffusion sous `@_spi(Experimental)` ou comme produits séparés `Gemma4SwiftExperimental`. Aucun consommateur connu n'importe ces symboles : vérifié par grep sur les 5 dépôts. Effort M (L si découpage en cibles).

---

## C. Redondance avec le Gemma 4 amont (mlx-swift-lm 3.31.4)

### S-25 — Inventaire et arbitrage (moyenne, VÉRIFIÉ par lecture de `.build/checkouts/mlx-swift-lm`)
Ce que l'amont fournit :
- `MLXLLM/Models/Gemma4Text.swift` (776 l.) : texte seul, enregistré sous `gemma4_text`.
- `MLXVLM/Models/Gemma4.swift` (3 085 l.) : `Gemma4` (vision E2B/E4B/26B/31B) et `Gemma4Unified` (12B, vision + `embed_audio`, bidirectionnel `use_bidirectional_attention == "vision"` l.1249), plus un `public struct Gemma4Processor`. Point important : la variante E2B/E4B **jette l'audio** (`sanitize` filtre `audio_tower`/`embed_audio`, l.1448 et 2113). Pas de Conformer amont.
- `MLXVLM/Models/Gemma4Assistant.swift` : drafter MTP, branché sur le générique `MLXLMCommon/MTPSpeculativeTokenIterator` et `MTPDrafterModelFactory` (intégré à `generate`/`ChatSession`).
- `MLXLMCommon/Adapters/LoRA/*` : LoRA/DoRA génériques ; `QuantizedKVCache` natif (`maybeQuantizeKVCache`).

| Brique de ce dépôt | Couvert en amont ? | Verdict |
|---|---|---|
| Décodeur texte (`TextModel/`) | Oui (MLXLLM) | **Garder.** LTX dépend de l'arbre de modules `Gemma4LLMModel` (clés de poids, `LTX25TextEncoderAssets.swift:167`) et de `forwardCollectingHiddenStates`, que l'amont n'a pas. Ajouter un test de parité de logits avec `MLXLLM.Gemma4TextModel` pour détecter les divergences (À VÉRIFIER) |
| Vision SigLIP E2B/E4B | Oui (MLXVLM) | Garder : le lien avec audio et vidéo passe par le même wrapper, et le paquet ne lie pas MLXVLM (ce qui est voulu, cf. CLAUDE.md) |
| Audio Conformer E2B/E4B | **Non** | Garder : c'est une valeur propre du dépôt |
| 12B Unified multimodal | Oui (MLXVLM) | Garder pour l'instant (encodeur LTX). À réévaluer si LTX peut consommer `MLXVLM.Gemma4Unified` |
| `Gemma4MTPPipeline` (actor, 359 l.) | **Oui**, mieux intégré (`MTPSpeculativeTokenIterator` via `GenerateParameters`) | **Candidat au retrait.** Aucun consommateur ne l'utilise (grep). Mais l'amont exige MLXVLM pour le drafter. Deux options : (a) déprécier et documenter la voie amont, (b) garder uniquement l'entraînement du drafter (`Gemma4DrafterTraining`), qui n'a pas d'équivalent amont |
| `Gemma4AssistantDraftModel` | Oui (même nom public, S-15) | Suit le sort du point précédent |
| LoRA (boucle custom) | Partiellement | Garder : la mémoire documente que la boucle amont ne reproduisait pas mlx-lm (97,2 % contre l'écart initial). Supprimer seulement les doublons d'entrée (S-23) |
| TurboQuant | Remplacé par `QuantizedKVCache` natif (le README le recommande déjà) | **Retirer** du chemin `newCache(parameters:)` (S-13), puis sortir le module ou le passer en SPI |
| NoRepeatNGram, `templateVariables`, `systemPrompt` multimodal, `forwardCollectingHiddenStates`, téléchargement, `Gemma4Pipeline` | Non | Garder : c'est ce qu'utilisent les consommateurs |
| Diffusion | Non | Garder, mais hors de la surface stable (S-33) |

- Risque : retirer `Gemma4MTPPipeline` ou `Gemma4AssistantDraftModel` casse l'API publique, donc à faire en 2.0.0. Effort M (L avec le test de parité).

---

## D. Robustesse

### S-01 — Streams non annulables (haute, VÉRIFIÉ)
- Fichiers :
  - `Pipeline/Gemma4Pipeline.swift:412-426`, `467-531`, `607-694`, `719-733` ;
  - `Pipeline/Gemma4MTPPipeline.swift:42-59`, `70-87`.
- Constat : chaque stream crée un `Task { … }` non structuré. Aucun `continuation.onTermination` n'annule ce Task, et la boucle ne teste jamais `Task.isCancelled`.
  - Preuve : `grep -rn "onTermination\|Task.isCancelled\|checkCancellation" Sources` ne trouve qu'une occurrence, dans `Gemma4BenchUI/BenchViewModel.swift:522` (**l'app de bench le fait, la bibliothèque non**).
- Conséquences :
  - Si le consommateur `break` ou annule son Task (Fluxforge, flux-2, h3), la génération continue jusqu'à `maxTokens` en occupant le GPU, et le `ModelContainer` reste verrouillé : l'appel suivant attend.
  - `unload()` n'interrompt rien.
  - La boucle MTP (`runLoop`) ne s'arrête pas non plus.
- Correction :
  ```swift
  let task = Task { … }
  continuation.onTermination = { _ in task.cancel() }
  ```
  Ajouter aussi `try Task.checkCancellation()` à chaque tour de la boucle `for await generation in stream` et de la boucle MTP. Le `for await` sur `MLXLMCommon.generate` s'arrête déjà à l'annulation du Task qui l'itère.
- Risque : aucun pour l'API. Changement de comportement voulu : un stream abandonné s'arrête. Effort S.

### S-02 — Téléchargement partiel pris pour complet, jamais repris (haute, VÉRIFIÉ par lecture)
- Fichiers :
  - `Pipeline/Gemma4ModelCache.swift:93-99` (`hasModelFiles`) ;
  - `Pipeline/Gemma4ModelDownloader.swift:45-48` ;
  - `Download/Gemma4DownloadManager.swift:59-62`.
- Constat :
  - `hasModelFiles` renvoie `true` dès qu'il existe `config.json` **et un seul** `*.safetensors`.
  - Les fichiers arrivent dans l'ordre du manifeste HF (alphabétique : `config.json` avant `model-00001-of-0000N.safetensors`), et chaque fichier est déplacé atomiquement à la fin de son transfert.
  - Après une annulation ou une coupure au 2ᵉ shard d'un modèle multi-shards, `isDownloaded` vaut donc `true`. Alors :
    - `Gemma4ModelDownloader.download` fait `return modelDir` sans réseau ;
    - `Gemma4DownloadManager.download` et `retry` font `task.markCompleted()` : l'UI de Fluxforge affiche « Downloaded » ;
    - `Gemma4Pipeline.load(downloadIfNeeded: true)` ne relance pas ;
    - le chargement échoue ensuite sur des poids manquants.
  - La logique de reprise fichier par fichier de `runFileLoop` (`!force && fileExists → skip`) existe, mais le retour anticipé l'empêche de servir.
- Correction :
  - `hasModelFiles` lit `model.safetensors.index.json` quand il existe et vérifie la présence de tous les shards de `weight_map` ;
  - le téléchargement écrit un marqueur `.complete` en fin de `runFileLoop`, et `isDownloaded` l'exige (tout en tolérant l'absence de marqueur pour les caches HF et les téléchargements antérieurs, via l'index).
- Risque : aucun pour l'API. Des modèles aujourd'hui « téléchargés » à tort repasseront en « non téléchargés », ce qui est l'effet voulu. Effort S-M.
- À VÉRIFIER : reproduction réelle (couper le réseau au 2ᵉ shard de `gemma-4-31b-it-4bit`).

### S-03 — Graphe MLX paresseux transmis à `container.perform` (haute, VÉRIFIÉ)
- Fichiers : `Pipeline/Gemma4Pipeline.swift:597` (`nonisolated(unsafe) let pixelsCapture = pixelValues`), `:626`, `:640` ; `Pipeline/ImageProcessor.swift:46-70`, dont la version synchrone renvoie `chw.expandedDimensions(axis: 0)`, un graphe non évalué.
- Constat :
  - La règle du dépôt (mémoire `feedback_mlx_array_threading`, doc mlx-swift) interdit de construire un graphe sur un thread et de l'évaluer sur un autre. C'est pourtant ce que produit la séquence suivante : `processImage` synchrone sur le main actor, puis `chatStreamMultimodal(pixelValues:)`, qui capture l'array et l'évalue dans `container.perform` (isolation du `ModelContainer`).
  - Fluxforge fait exactement cela (mémoire : `VLMCaptionService.swift:185`, version synchrone suivie de `chatStreamMultimodal`). Les surcharges `async` évaluent bien (`ImageProcessor.swift:176,199`), mais `chatStreamMultimodal` ne se protège pas contre un appelant qui ne les utilise pas.
  - Même motif pour les `pendingX` publics de `Gemma4MultimodalLLMModel` (`:26-36`), que h3-swift-mlx affecte lui-même.
- Correction : `eval(pixelValues)` en tête de `chatStreamMultimodal`, sur le thread appelant, avant la capture. Le coût est nul si l'array est déjà matérialisé. Documenter la même précondition sur les `pendingX`.
- Risque : aucun pour l'API. Le coût de prétraitement se paie dans l'appel. Effort S.

### S-04 — Deadlock ABBA : l'API publique n'offre aucune protection (haute, VÉRIFIÉ)
- Fichiers : `LoRA/Gemma4LoRATrain.swift:100` (`public static func train`), `:311` (`trainMultimodal`), `Speculative/Gemma4DrafterTraining.swift:139` (`trainDrafter`), fonctions libres `trainLoRA` et `trainMultimodalLoRA` ; côté inférence, `Gemma4Pipeline` (`@MainActor`) et `Gemma4MTPPipeline` (actor).
- Constat :
  - CLAUDE.md et README (l.249-254) décrivent un gel du process si un gradient tourne pendant un forward sur un autre thread.
  - La bibliothèque n'a aucun garde-fou : `grep evalLock Sources` → seulement `Gemma4BenchUI/VQAGame/VQAGameViewModel.swift:52`.
  - Les doc-comments des 5 points d'entrée d'entraînement ne mentionnent pas le risque.
- Correction (du moins au plus robuste) :
  1. `- Warning:` dans la doc de chaque entrée d'entraînement et de `Gemma4Pipeline` (S) ;
  2. un verrou coopératif de processus, `Gemma4ComputeGate` (actor avec file FIFO), pris par chaque entraînement et chaque stream d'inférence du paquet. La bibliothèque sérialise alors elle-même ses propres appels, ce qui ne protège pas des appels MLX directs du consommateur (M) ;
  3. corriger en amont dans mlx-swift (ordre des verrous), c'est-à-dire ouvrir l'issue et le suivre avec le skill `track`.
- Risque : l'option 2 sérialise inférence et entraînement dans un même process. C'est le comportement recommandé, mais un changement de performance pour qui les parallélise aujourd'hui (et risque de gel). Effort S (doc) / M (verrou).

### S-05 — Enregistrement global : course entre deux chargements (moyenne, VÉRIFIÉ par lecture)
- Fichier : `Pipeline/Gemma4Registration.swift:30-59`, `86-94`.
- Constat :
  - `loadContainer(multimodal:)` fait `await register(multimodal:)` (qui mute `LLMTypeRegistry.shared`, global, last-write-wins), puis `await LLMModelFactory.shared.loadContainer`.
  - Deux chargements concurrents avec des `multimodal` différents peuvent s'intercaler. Exemple : Fluxforge charge l'enhancer E2B multimodal pendant que LTX charge un modèle texte. L'un obtient alors le mauvais type (`Gemma4LLMModel` au lieu de `Gemma4MultimodalLLMModel`) et échoue plus tard en `unsupportedModelFamily`.
  - Les défauts sont en outre incohérents : `register(multimodal: false)`, mais `loadContainer(multimodal: true)` et `Pipeline.load(multimodal: true)`.
- Correction : ne plus toucher au registre global dans `loadContainer`. `LLMModelFactory` a un init public (`LLMModelFactory.swift:526`, `init(typeRegistry:modelRegistry:)`) et `ModelTypeRegistry` aussi (`init(creators:)`). On peut donc construire une fabrique privée par appel :
  ```swift
  let factory = LLMModelFactory(
      typeRegistry: ModelTypeRegistry(creators: gemma4Creators(multimodal: multimodal)),
      modelRegistry: LLMRegistry.shared)
  ```
  `register()` reste disponible pour ceux qui chargent eux-mêmes.
- Risque : aucun pour l'API (la signature est inchangée). Changement : `loadContainer` n'écrase plus les entrées globales, alors qu'un consommateur pourrait compter sur cet effet de bord. Documenter l'appel à `register()` pour ce cas. Effort S.

### S-06 — `download(modelId:)` peut renvoyer un répertoire inexistant (moyenne, VÉRIFIÉ par lecture)
- Fichiers : `Pipeline/Gemma4ModelDownloader.swift:40-48` ; `Pipeline/Gemma4ModelCache.swift:101-130`.
- Constat : `isDownloaded(modelId:)` accepte aussi le cache HF (`~/.cache/huggingface/hub/models--…/snapshots/*`), mais `download` renvoie toujours `modelsDirectory/org/model`. Si le modèle n'existe que dans le cache HF, les appelants reçoivent un chemin vide : CLI `MtpForward`, `MtpSmoke`, `MtpDiagVerify`, `MtpDrafterLoader`, `DiffusionGemmaCommand`.
- Correction : renvoyer `Gemma4ModelCache.localPath(forModelId:)` dans la branche « déjà présent ». Risque : aucun. Effort S.

### S-07 — CLI `download --force` sans effet (moyenne, VÉRIFIÉ)
- Fichiers : `Gemma4CLI/LocalModelDownloader.swift:7-19` ; `Gemma4CLI/Gemma4CLI.swift:201-227`.
- Constat : `Download.run` calcule `toDownload` en tenant compte de `force`, puis appelle `LocalModelDownloader.download(modelId:to:token:)`. Ce wrapper **ignore `to:`** et **ne transmet pas `force`** : `Gemma4ModelDownloader.download` voit le modèle en cache et retourne sans rien télécharger. `--force` est donc inopérant. Combiné à S-02, c'est le seul moyen de réparer un téléchargement partiel, et il ne marche pas.
- Correction : supprimer `LocalModelDownloader` et appeler `Gemma4ModelDownloader.download(model, token:, force: force)`. Risque : aucun (CLI). Effort S.

### S-08 — Rechargement : double empreinte mémoire et état incohérent (moyenne, VÉRIFIÉ)
- Fichier : `Pipeline/Gemma4Pipeline.swift:283-298`, `315-320`, `412-426`, `529`, `692`.
- Constat :
  1. `load(from:)` fait `state = .unloaded` mais **garde `container`** pendant le chargement du nouveau modèle : le pic mémoire vaut deux modèles (31B + 26B = plus de 30 Go). Si le chargement échoue, `state == .unloaded` alors que `chat()` fonctionne encore sur l'ancien modèle.
  2. Chaque stream termine par `self?.state = .ready` inconditionnellement. Après `unload()`, ou si un autre stream est encore en cours, l'état redevient `.ready` avec `container == nil` ; `isReady` ment.
  3. `State.error(String)` n'est jamais affecté (`grep "\.error(" Gemma4Pipeline.swift` : 0).
  4. `continueChat` lève `modelNotLoaded` quand il n'y a pas de session, message trompeur (le test `PipelineTests:87` fige ce comportement).
- Correction : `container = nil` et `Memory.clearCache()` avant de charger, avec rétablissement de l'état en cas d'échec ; compteur de générations actives, `.ready` seulement si `container != nil && active == 0` ; nouveau cas d'erreur `noActiveSession`.
- Risque : ajouter un cas à `Gemma4PipelineError` casse les `switch` exhaustifs des consommateurs (enum non `@frozen`, mais paquet sans library evolution). À faire en 2.0.0 ou en réutilisant `invalidInput`. Effort S.

### S-09 — Pointeurs temporaires dans la FFT audio (moyenne, VÉRIFIÉ par warning)
- Fichier : `Pipeline/AudioProcessor.swift:213`.
- Constat : `DSPSplitComplex(realp: &realPart, imagp: &imagPart)` : le pointeur n'est valide que le temps de l'`init`, puis il est utilisé par `vDSP_ctoz` et `vDSP_fft_zrip`. C'est un comportement indéfini, signalé par le compilateur (`[#TemporaryPointers]`, deux warnings). Ça marche aujourd'hui par chance (buffer d'array stable). L'audio est utilisé par flux-2 et h3 (`Gemma4AudioProcessor.processAudio`).
- Correction : imbriquer `realPart.withUnsafeMutableBufferPointer { r in imagPart.withUnsafeMutableBufferPointer { i in var split = DSPSplitComplex(realp: r.baseAddress!, imagp: i.baseAddress!) … } }`. Risque : aucun. Garder le test `AudioProcessorTests` comme non-régression numérique. Effort S.

### S-10 — Processeurs async qui rendent des graphes paresseux (moyenne, VÉRIFIÉ)
- Fichiers : `VideoProcessor/Gemma4VideoProcessor.swift:46-137` (`concatenated(padded)`, écritures par slice, rendus sans `eval`) ; `Pipeline/AudioProcessor.swift:51-79` (`MLXArray(melData).reshaped`, `zeros`).
- Constat : ce sont des fonctions `nonisolated async`. Le graphe est construit sur le pool coopératif puis consommé par l'appelant ailleurs, ce qui enfreint la même règle que S-03. `ImageProcessor` async a été corrigé (1.6.0), pas ces deux-là.
- Correction : `eval(...)` avant le `return`. Risque : aucun. Effort S.

### S-11 — Décodage token par token (moyenne ; VÉRIFIÉ code, effet À VÉRIFIER)
- Fichiers : `Pipeline/Gemma4MTPPipeline.swift:325-331` (`yieldToken` → `tokenizer.decode(tokenIds: [id])`) ; `Gemma4CLI/Gemma4CLI.swift:867` et `:1113`.
- Constat : décoder chaque id isolément perd les caractères UTF-8 multi-octets répartis sur plusieurs tokens byte-fallback (émojis, certains accents) et, selon le tokenizer, l'espace initial `▁`. L'amont utilise `NaiveStreamingDetokenizer`, et `Gemma4Pipeline` en profite via `generateTask`, mais pas le MTP ni le CLI `describe`.
- Correction : `NaiveStreamingDetokenizer` (MLXLMCommon), ou un décodage incrémental avec diff de préfixe. Risque : aucun pour l'API. Effort S.
- À VÉRIFIER : générer un texte avec émojis et accents en `mtp-generate` et comparer au chemin `chatStream`.

### S-12 — Crashs sur entrées d'API publique (moyenne, VÉRIFIÉ)
- `Gemma4MTPPipeline.swift:113` : `precondition(blockSize >= 2)` sur un paramètre public de `mtpStream`, qui tue le process au lieu de lever une erreur.
- `NoRepeatNGramLogitProcessor.swift:83` : `precondition(ngramSize >= 1)` dans un `init` public. Le Pipeline valide en amont (`:449`, `:592`), mais un appel direct au processeur crashe.
- `Speculative/Gemma4AssistantDraftModel.swift:284-289` : `fatalError` si `bind`/`setSharedKV` n'ont pas été appelés, `precondition(blockSize >= 2)`.
- `LoRA/Gemma4TrainingLoop.swift:349`, `:469` : `model as! Gemma4MultimodalLLMModel` dans `public func trainMultimodalLoRA(model: Module, …)`. Passer un modèle texte fait crasher.
- `TextModel/Gemma4TextModel.swift:196`, `:206`, `:328` ; `Gemma4Attention.swift:252` ; `TurboQuantKVCache.swift:153` ; `EncoderKVCache.swift:65` : `fatalError` d'invariants internes. Acceptable pour des erreurs de programmation, mais pas quand l'invariant dépend du checkpoint chargé : `embed_tokens_per_layer` absent pour une config E2B avec des poids incomplets.
- Correction : `throws` ou `Result` sur les entrées publiques (`mtpStream` renvoie déjà un stream qui peut lever ; `trainMultimodalLoRA` → `guard let … else throw`).
- Risque : `NoRepeatNGramLogitProcessor.init` qui devient `throws` casse l'API. Préférer un `init?` supplémentaire ou un clamp documenté en 1.x. Effort S.

### S-13 — `kvBits` redirigé vers TurboQuant, contrairement au README (moyenne ; VÉRIFIÉ code, effet À VÉRIFIER)
- Fichiers : `Pipeline/Gemma4LLMModel.swift:63-66` (`newCache(parameters:)` → `makeCache(kvBits: Float(parameters.kvBits))`) ; `TextModel/Gemma4LanguageModel.swift:174-199` ; `README.md:865-876`.
- Constat :
  - Le README recommande `GenerateParameters(kvBits: 4, kvGroupSize: 64, quantizedKVStart: 5000)` pour le cache quantifié **natif** et affirme que TurboQuant « is not used in the default inference pipeline ».
  - En réalité, tout `kvBits` passé au `ModelContainer` active `TurboQuantKVCache` sur les couches full attention quand l'heuristique le juge viable (26B, 31B). Sur E2B/E4B, il imprime `[TurboQuant] Desactive…` sur stdout (`print` dans la bibliothèque) et retombe sur `KVCacheSimple`, que `maybeQuantizeKVCache` amont quantifiera ensuite.
  - Selon le modèle, le même paramètre donne donc deux implémentations différentes. `kvBits: 8` n'est pas borné (TurboQuant 8 bits : codebook 256 entrées, non testé).
- Correction : `newCache(parameters:)` ne consulte plus `kvBits` (laisser `maybeQuantizeKVCache` amont faire son travail) ; TurboQuant uniquement via `makeCache(kvBits:)` explicite, en SPI expérimentale ; supprimer le `print`.
- Risque : changement de comportement pour qui passe `kvBits` sur 26B/31B (cache natif au lieu de TurboQuant). C'est l'objectif affiché par le README. Effort S.
- À VÉRIFIER : que `attentionWithCacheUpdate` gère bien `QuantizedKVCache` sur les couches à `global_head_dim` 512 et en K=V (générer 8K tokens sur 31B 4-bit avec `kvBits: 4`).

### S-14 — Pas de prefill par tranches sur le chemin texte (moyenne ; VÉRIFIÉ code, impact À VÉRIFIER)
- Fichiers : `Pipeline/Gemma4LLMModel.swift:76-86` ; `Pipeline/Gemma4MultimodalLLMModel.swift:253`.
- Constat : `prepare(_:cache:windowSize:)` est surchargé et renvoie `.tokens(input.text)` entier, en ignorant `windowSize`. L'implémentation amont par défaut (`MLXLLM/LLMModel.swift:21-24`) découpe par `prefillStepSize` (512). Un prompt long (8K-32K, cas de LTX et Fluxforge) est donc prérempli en un seul forward, avec un pic d'activations proportionnel à T. Pour le multimodal c'est nécessaire (les `pendingX` sont consommés au premier forward, masque bidirectionnel du 12B), mais pas pour `Gemma4LLMModel`.
- Correction : supprimer la surcharge dans `Gemma4LLMModel` (hériter du défaut `LLMModel`). Pour le multimodal, faire le premier forward avec les médias puis découper le reste.
- Risque : aucun pour l'API ; à valider numériquement (prefill découpé avec `RotatingKVCache` en fenêtre glissante). Effort S (texte) / M (multimodal).
- À VÉRIFIER : mesurer le pic GPU avec un prompt de 16K sur E4B, avant et après.

### S-26 — `Gemma4TokenFilter` faussement `Sendable` (basse, VÉRIFIÉ)
- Fichier : `Pipeline/Gemma4TokenFilter.swift:26-40`. Classe `public final` à état mutable (`channel`, `thinkingTokens`, `pendingText`), déclarée `@unchecked Sendable` sans aucun verrou. h3-swift-mlx l'utilise.
- Correction : retirer la conformance, ou protéger l'état avec un `Mutex` (Synchronization, macOS 15, compatible avec la plateforme minimale actuelle).
- Risque : retirer `Sendable` peut casser un consommateur qui l'envoie entre tâches, alors que le `Mutex` ne casse rien. Effort S.

### S-27 — Globaux mutables `nonisolated(unsafe)` (basse, VÉRIFIÉ)
- `Gemma4ModelCache.swift:10` : `nonisolated(unsafe) public static var customModelsDirectory: URL?` est public, écrit par Fluxforge et par les tests, et lu depuis n'importe quel thread (téléchargements). Data race formelle.
- `Gemma4Pipeline.swift:240` : `nonisolated(unsafe) private var currentSession` est inutile (la classe est `@MainActor`, et le compilateur le signale : « has no effect »). Le `@unchecked Sendable` de la classe `@MainActor` (`:16`) est lui aussi redondant.
- Caches TurboQuant (`TurboQuantUtils.swift:14-17`, `TurboQuantDecodeKernels.swift:141`) : protégés par `NSLock`, OK. En revanche ils stockent des `MLXArray` créés sur un thread et réutilisés sur d'autres, et il reste À VÉRIFIER qu'ils sont `eval`-ués avant d'entrer dans le cache.
- Correction : `Mutex<URL?>` derrière une propriété calculée (API source-compatible). Supprimer les annotations inutiles. Effort S.

### S-28 — « Dernier snapshot » pris au hasard (basse, VÉRIFIÉ)
- Fichier : `Pipeline/Gemma4ModelCache.swift:119-126`. Le commentaire dit « Prendre le dernier snapshot (le plus recent) », mais `contentsOfDirectory` ne garantit aucun ordre et les noms sont des hashes de commit.
- Correction : trier par date de modification, ou lire `refs/main` du cache HF. Effort S.

### S-29 — Gestion mémoire (basse ; VÉRIFIÉ code, impact À VÉRIFIER)
- `Memory.cacheLimit` n'est fixé nulle part dans la bibliothèque (seulement par les commandes CLI diffusion : `DiffusionGemmaCommand.swift:152`, `ProfileDiffusionCommand.swift:151`). Le cache de buffers MLX peut donc croître jusqu'à la limite mémoire, ce qui gêne une app hôte (Fluxforge, qui fait aussi tourner LTX).
- `unload()` (`Gemma4Pipeline.swift:315-320`) met `container = nil`, mais les Tasks de stream en cours capturent `container` fortement. Faute d'annulation (S-01), la mémoire n'est libérée qu'en fin de génération.
- Les `pendingAudioFeatures` d'un modèle sans tour audio (26B/31B : `audioTower == nil`, `Gemma4MultimodalLLMModel.swift:204`) ne sont jamais remis à `nil` et resteront attachés au modèle.
- Correction : paramètre optionnel `cacheLimit` sur `Gemma4Pipeline` (ou documentation) ; S-01 ; remettre tous les `pendingX` à `nil` en fin de `prepareMultimodalEmbeds` quoi qu'il arrive (`defer`). Effort S.

### S-30 — `try!`, `fatalError` inatteignable, `print` (basse, VÉRIFIÉ)
- `Pipeline/Gemma4UnifiedVideoProcessor.swift:116-117` : deux `try!` (aller-retour JSON pour surcharger `num_soft_tokens`). Fragile dès qu'un champ requis est ajouté à `Gemma4UnifiedVisionConfig`. Correction : un `init` mémberwise ou `with(numSoftTokens:)`.
- `LoRA/Gemma4LoRAData.swift:146` : `fatalError("Type de fichier non supporte")`, inatteignable aujourd'hui (l'appelant filtre `jsonl`/`txt`), mais la fonction est réutilisable. Correction : `throw`.
- `print` dans la bibliothèque : 23 dans `Gemma4LoRATrain.swift`, 6 dans `Gemma4DrafterTraining.swift`, 1 dans `Gemma4LanguageModel.swift:183`, 1 dans `Gemma4OnTheFlyQuantization.swift`. Correction : `os.Logger` (sous-système `Gemma4Swift`). Effort S.

---

## E. Tests

### S-31 — Sous-systèmes sans aucun test (moyenne, VÉRIFIÉ)
Preuve : `grep -rlw <symbole> Tests | wc -l` = 0 pour chacun des symboles suivants :
- **Encodeur audio Conformer** : `AudioEncoder`, `ConformerBlock`, `AudioAttention`, `SubSampleConvProjection`. Seul le prétraitement mel est testé (`AudioProcessorTests`).
- **Encodeur vision SigLIP** : `VisionModel`, `VisionAttention`, `VisionPooler`, `VisionPatchEmbedder`. Seul `ClippableLinear` est testé.
- **12B Unified** : `Gemma4UnifiedMultimodalLLMModel`, `Gemma4UnifiedVisionEmbedder`, `Gemma4BidirectionalMask`, `computeVisionTokenMask`, `Gemma4UnifiedAudioProcessor`. Le masque bidirectionnel, dont l'absence faisait halluciner le modèle (mémoire), n'a **aucun** test.
- **MoE** : `Gemma4Router`, `Gemma4Experts`.
- **TurboQuant** : `TurboQuantKVCache`, `TurboQuantMSECodec`, bit-packing. Seule l'heuristique de viabilité est testée.
- **Diffusion** : `EntropyBoundSampler`, `StableConfidentStopping`, `DiffusionAttentionMask`, `DiffusionGemmaPipeline`, `DiffusionWeightSanitizer`, `DiffusionGemmaLoader`.
- **Pipelines** : `Gemma4MTPPipeline`, `Gemma4LoRAInference`, `Gemma4OnTheFlyQuantization`, `CGImageLoader`.
- `Gemma4Pipeline` n'est testé que sur ses états (`PipelineTests` : 6 tests sans modèle). Les chemins de génération ne sont couverts que par les 3 suites d'intégration conditionnelles.

Tests d'intégration conditionnés par `GEMMA4_INTEGRATION_MODEL_PATH` : `NoRepeatNGramIntegrationTests`, `MultimodalSystemPromptTests`, `TemplateVariablesTests`. Aucun sur l'audio, la vidéo, le MTP, LoRA-inference, le 12B ni l'annulation.

Priorités proposées (sans modèle, poids aléatoires et petites configs, comme `ForwardCollectingHiddenStatesTests`) :
1. `Gemma4BidirectionalMask` : forme et contenu du masque (S) ;
2. forme de sortie Conformer et SigLIP sur config minuscule (S) ;
3. aller-retour TurboQuant MSE et bit-packing (S) ;
4. annulation d'un stream, test de non-régression de S-01 (S) ;
5. `hasModelFiles` avec index partiel, non-régression de S-02 (S) ;
6. intégration conditionnelle audio E2B et MTP greedy (M).

### S-32 — Qualité et fiabilité de la suite (basse, VÉRIFIÉ)
- `DownloadCoordinatorTests.swift:58-97` (« cancelAll resumes pending continuation… ») : **tautologique**. Il accepte le succès, la `CancellationError`, `Gemma4DownloadError.cancelled` et toute autre erreur (`catch { }`). Seul `#expect(await coordinator.isCancelled)` compte, et il est trivialement vrai après `cancelAll()`. Il dépend en plus d'un `Task.sleep(10 ms)` et d'une vraie connexion à `0.0.0.0`.
- `ModelCacheTests` (`@Suite("Model Cache")`, **non** `.serialized`) : 4 tests mutent le global `Gemma4ModelCache.customModelsDirectory`. Ils ne tiennent que grâce à `SWT_EXPERIMENTAL_MAXIMUM_PARALLELIZATION_WIDTH=1` ; le jour où le wrapper disparaît (correctif amont du deadlock), ils deviennent flaky. `DownloadIntegrationTests` a bien `.serialized` sur une de ses suites.
- 13 fichiers XCTest et 23 swift-testing coexistent.
- **Pas de CI** : aucun `.github/`. Le deadlock, les warnings et les régressions d'API ne sont détectés qu'en local.
- Correction :
  - réécrire le test d'annulation avec un `URLProtocol` stub qui ne répond jamais, et exiger `Gemma4DownloadError.cancelled` ;
  - `.serialized` sur `ModelCacheTests` ;
  - un workflow GitHub Actions `macos-15` (build CLI + `Scripts/run-tests.sh` sans modèle) et une vérification d'API (`swift package diagnose-api-breaking-changes 1.7.3`).
- Effort S (tests) / M (CI, runner Apple Silicon requis pour Metal).

---

## F. API publique

### Ce que les consommateurs utilisent (à ne pas casser), relevé par grep dans les 5 dépôts
- **Fluxforge Studio** (épinglé `main`, pas de tag) :
  - `Gemma4Pipeline` : `.load`, `.unload`, `.chatStream`, `.chatStreamMultimodal`, `.continueChatStream`, `.loadAdapter`, et le cas `Model.e2b6bit` ;
  - `Gemma4ModelCache` : `isDownloaded`, `localPath`, `modelsDirectory`, `customModelsDirectory` ;
  - `Gemma4DownloadManager.shared` : `download`, `cancel`, `delete`, `status` ;
  - `Gemma4ImageProcessor.processImage`, `Gemma4LoRAInference.loadAdapter`.
- **ltx-video-swift-mlx** :
  - `Gemma4LLMModel(config:)`, `forwardCollectingHiddenStates`, l'**arbre de modules et les clés de poids** de `Gemma4LLMModel` (quantification et `applyWeights` maison) ;
  - `Gemma4TextConfig`, `Gemma4Config`, `Gemma4Pipeline` (enhancer) ;
  - le module importe aussi `MLXVLM`.
- **flux-2-swift-mlx** :
  - `Gemma4Pipeline` ;
  - `Gemma4Processor` : constantes d'ids et `multimodalChatIds` ;
  - `Gemma4ImageProcessor`, `Gemma4VideoProcessor.processVideo` / `formatTimestamp` / `VideoFrames`, `Gemma4AudioProcessor.processAudio` / `AudioFeatures`, `Gemma4TokenFilter`, `Gemma4ModelCache`.
- **h3-swift-mlx** : `Gemma4Registration.register(multimodal:)`, `loadModelContainer` (fonction libre, voir S-19), `Gemma4MultimodalLLMModel` et ses `pendingX`, `Gemma4TokenFilter`.
- **ToolsForge** : `Gemma4Pipeline` (`.load`, `.chat`, `.chatStream`, `.fuseAdapter`, `.unload`), `Gemma4ModelCache.isDownloaded`.

Aucun n'utilise TurboQuant, Diffusion, `Gemma4MTPPipeline`, `Gemma4LoRATrain` ni les fonctions libres LoRA.

### S-15 — Collisions de noms publics avec l'amont (moyenne, VÉRIFIÉ)
Commande : `comm -12` entre les types publics de `Sources/Gemma4Swift` et ceux de `mlx-swift-lm/Libraries/{MLXLLM,MLXVLM,MLXLMCommon}` + `mlx-swift/Source`.

| Notre type | Collision avec |
|---|---|
| `Gemma4TextModel` (`TextModel/Gemma4TextModel.swift:102`) | `MLXLLM.Gemma4TextModel` (`Gemma4Text.swift:685`) |
| `Gemma4Processor` (enum) | `MLXVLM.Gemma4Processor` (struct, `Gemma4.swift:2612`) |
| `Gemma4AssistantDraftModel`, `Gemma4AssistantDraftInner` | `MLXVLM` (`Gemma4Assistant.swift:112,143`) |
| `ProportionalRoPE` (`RoPE/ProportionalRoPE.swift:18`) | `MLXLMCommon.ProportionalRoPE` (`RoPEUtils.swift:164`) |
| `RoPELayer` (protocole, `RoPE/RoPEFactory.swift:7`) | `MLXLMCommon.RoPELayer` (typealias, `RoPEUtils.swift:408`) |
| `ProcessedImage`, `ProcessedVideo`, `ProcessedAudio` (imbriqués dans nos processeurs Unified) | imbriqués dans `MLXLMCommon.UserInput` : pas de conflit réel, car imbriqués des deux côtés |

- Conséquence : tout fichier consommateur qui importe `Gemma4Swift` avec `MLXLLM` ou `MLXVLM` (flux-2 utilise `Gemma4Processor` et LTX importe `MLXVLM`) obtient « ambiguous use » dès qu'il nomme le type sans qualification, et doit écrire `Gemma4Swift.Gemma4Processor`. Ça compile aujourd'hui par chance, selon les imports de chaque fichier.
- Correction : ne **pas** renommer en 1.x. Documenter la qualification, puis en 2.0.0 renommer ou rendre `internal` `ProportionalRoPE` et `RoPELayer` (usage interne uniquement ; aucun consommateur ne les utilise). Pour `Gemma4Processor`, garder le nom (flux-2 l'utilise) et ajouter un `typealias Gemma4SwiftProcessor` pour faciliter la désambiguïsation. Effort S.

### S-16 — 12B Unified : capacités annoncées contre capacités réelles (moyenne, VÉRIFIÉ)
- `Gemma4Pipeline.swift:188-190` déclare `.b12b → .anyToAny` (image, audio, vidéo).
- Or `chatStreamMultimodal` exige `as? Gemma4MultimodalLLMModel` (`:634-639`) : un 12B chargé en multimodal (`Gemma4UnifiedMultimodalLLMModel`, qui n'en hérite pas, `Gemma4UnifiedMultimodalLLMModel.swift:17`) lève `unsupportedModelFamily`. Idem pour `Gemma4MTPPipeline` (`:128-141`). Seul le CLI sait piloter le 12B multimodal (`Gemma4CLI.swift:658`, `:916`, `:1066`).
- Docs périmées :
  - `Gemma4Registration.swift:24-27` : « le path multimodal du 12B n'est pas encore branche… tombe en text-only », alors que les lignes 50-54 le branchent ;
  - `Gemma4Pipeline.swift:46-48` : « schema different (todo). Pour l'instant : text-only ».
- Correction : protocole commun `Gemma4MultimodalInput` (image, audio, vidéo en attente) implémenté par les deux wrappers, utilisé par `chatStreamMultimodal`. En attendant, corriger la doc et les capacités (ou lever une erreur explicite).
- Risque : modifier `capabilities` change le comportement des UIs qui filtrent dessus. Effort M.

### S-17 — DiffusionGemma dans `Gemma4Pipeline.Model` (moyenne, VÉRIFIÉ)
- `Gemma4Pipeline.swift:59` : `case a4bDiffBf16` est dans `Model`, mais `load()` le refuse (`:261-266`).
- `Model.recommended(forRAMGB:)` (`:213-217`) ne filtre que `isInstructionTuned` (toujours vrai) et la RAM : sur une machine de 96 Go, il **recommande** DiffusionGemma, que le pipeline ne sait pas charger.
- `gemma4-cli download --all` télécharge les 52 Go de la diffusion (`Gemma4CLI.swift:167`).
- Ajouter ce cas a cassé tout `switch` exhaustif consommateur sur `Model` : l'enum publique non gelée évolue sans que la version l'annonce.
- Correction : filtrer `!isDiffusion` dans `recommended` et `--all` (S) ; en 2.0.0, sortir la diffusion dans son propre enum.
- Risque : le filtrage ne casse rien.

### S-18 — Plateforme minimale incohérente (moyenne, VÉRIFIÉ)
- `Package.swift:6` : `.macOS(.v15), .iOS(.v17)`. Changé par `d382353c` (intégration de swift-mlx-profiler).
- `README.md:26`, `CLAUDE.md:90`, `AGENTS.md:90` annoncent « macOS 14+ (Sonoma) ».
- Un consommateur sous macOS 14 ne peut pas résoudre le paquet.
- Correction : aligner la doc sur macOS 15, ou redescendre si rien n'exige 15. À VÉRIFIER : `Mutex`, `swift-mlx-profiler` et `AVURLAsset` (le warning `init(url:)` est déprécié en 15, `Gemma4VideoProcessor.swift:51`).
- Risque : redescendre ne casse rien, monter si. Effort S.

### S-19 — Un consommateur utilise le chemin de chargement piégé (moyenne, VÉRIFIÉ)
- `h3-swift-mlx/Sources/H3PromptEnhancer/MultimodalContextIR.swift:70-71` : `await Gemma4Registration.register(multimodal: true)` puis `loadModelContainer(…)`, la fonction libre que CLAUDE.md interdit.
- h3 dépend de `from: 1.0.0`. Dès que son process lie MLXVLM, il reçoit `MLXVLM.Gemma4` et son `as? Gemma4MultimodalLLMModel` (l.372) échoue.
- Par ailleurs, le README ne documente ni `Gemma4Registration.loadContainer` ni ce piège : la section « Library Integration » (`README.md:546-575`) ne montre que `Gemma4Pipeline`.
- Correction :
  - prévenir h3 : `Gemma4Registration.loadContainer(from:multimodal:)` ;
  - ajouter au README un encart « Charger soi-même un ModelContainer » ;
  - dans la doc de `register()`, un `- Warning:` explicite (déjà partiellement présent l.15-18).
- Risque : aucun. Effort S.

### S-33 — Surface publique trop large (basse, VÉRIFIÉ)
- 935 déclarations `public`/`open`. Répartition : Pipeline 244, Diffusion 184, Configuration 153, LoRA 66, TurboQuant 64, TextModel 47, Speculative 43, VisionEncoder 35, Download 34, AudioEncoder 24. Des briques internes sont publiques : `VisionMLP`, `VisionTransformerModel`, tous les codecs TurboQuant, les couches de diffusion, `ScaledLinear`, etc.
- Chaque symbole public est une promesse SemVer, et les 5 consommateurs n'en utilisent qu'une trentaine.
- Correction : en 2.0.0, rendre `internal` (ou `@_spi(Experimental)`) ce qui n'est ni utilisé par un consommateur ni documenté dans le README. Préparer en 1.x avec `@available(*, deprecated, message: "deviendra interne en 2.0")`. Effort M.

### S-34 — Incohérences de nommage, de langue et de doc (basse, VÉRIFIÉ)
- Noms de fichiers différents des types : `ImageProcessor.swift` → `Gemma4ImageProcessor`, `AudioProcessor.swift` → `Gemma4AudioProcessor`, `Multimodal/Gemma4Model.swift` → `Gemma4MultimodalModel`.
- Langue mélangée dans les messages d'erreur publics : `Gemma4DownloadError` en anglais, `Gemma4PipelineError`, `Gemma4LoRADataError` et `MTPError` en français et en anglais.
- `Gemma4Pipeline.chat` / `chatStream` injectent par défaut le prompt système **« Tu es un assistant utile. »** (`:338`, `:404`, `:458`), alors que `chatStreamMultimodal` n'en injecte aucun : comportement par défaut asymétrique et francophone dans une bibliothèque documentée en anglais. Le changer modifie les sorties de ToolsForge et Fluxforge, donc à annoncer.
- `isEOS` du MTP : le commentaire dit « <pad>=0 » alors que l'ensemble vaut `[1, 106, 50]` (`Gemma4MTPPipeline.swift:334-336`).
- Pas de `CHANGELOG.md`, alors que le dépôt a déjà livré des ruptures volontaires (ids multimodaux en 1.3.0) et compte 5 consommateurs.
- Correction : CHANGELOG (Keep a Changelog) à partir des messages de PR 1.0 → 1.7.3 (S) ; le reste en 2.0.0.

---

## G. Warnings et marqueurs

### S-35 — Warnings (basse, VÉRIFIÉ d'après le log du 15/09)
Commande : `gunzip -c .build/xcode/Logs/Build/E16435DD-….xcactivitylog | strings | grep -oE "Sources/…swift:L:C: warning: …" | sort -u`.
- **107 warnings uniques** : Gemma4Swift 40, Gemma4CLI 51, Gemma4BenchUI 16.
- Par type : 36 dépréciations (`GPU.*` → `Memory.*`, `MLXFastKernel` → `MLXFast.MLXFastKernel`, `AVAsset(url:)`, `ignoringOtherApps`) ; 19 `nonisolated(unsafe)` inutiles ; 19 `try` sans appel qui lève ; 6 `var` jamais mutées ; 10 casts toujours réussis ou sans effet (`as? any LanguageModel`, `as! Module`, dans `Gemma4LoRAInference.swift:25,46,64` et `Gemma4LoRATrain.swift:159,253,358,368,427,476`).
- Les warnings qui signalent un **vrai risque** :
  - `AudioProcessor.swift:213` `#TemporaryPointers` ×2 → S-09 ;
  - `Gemma4Pipeline.swift:346` et `:706` `sending 'session' risks causing data races` : `ChatSession` non-Sendable utilisé depuis `@MainActor` dans un appel async. Correction : faire de la session un état confiné, ou appeler `respond` depuis un contexte isolé cohérent ;
  - `Gemma4LoRATrain.swift:148` : `tokenizer` initialisé mais jamais utilisé (reliquat, à vérifier que ce n'est pas un oubli fonctionnel) ;
  - `Gemma4Pipeline.swift:319` : `GPU.clearCache()` déprécié.
- Correction : passer les 27 `GPU.*` en `Memory.*` (sed mécanique) et `MLXFastKernel` en `MLXFast.MLXFastKernel`, supprimer les annotations et les casts inutiles. Viser 0 warning dans `Gemma4Swift` et l'imposer en CI (`-warnings-as-errors` sur la cible bibliothèque seulement). Effort S-M.

### TODO / FIXME / HACK (VÉRIFIÉ)
`grep -rn "TODO\|FIXME\|HACK\|XXX" Sources Tests Scripts` ne renvoie que 3 occurrences :
- `Pipeline/Gemma4Pipeline.swift:48` : « (todo). Pour l'instant : text-only », **périmé** (S-16) ;
- `Diffusion/Pipeline/DiffusionGemmaPipeline.swift:138` : « TODO Phase 10 » ;
- `TurboQuant/TurboQuantMSECodec.swift:31` : « TODO: reimplementer le WHT via Metal kernel », sans objet si TurboQuant est retiré (S-13, S-25).

Marqueurs de phase dans des messages de crash : `DiffusionGemmaEncoderModel.swift:154` (`"… (Phase 6)"`), `DiffusionGemmaCommand.swift:20` (« experimental Phase 4 »).

---

## Plan de correction suggéré

**1.7.4 / 1.8.0 (sans cassure d'API)**
1. S-01 annulation des streams et S-03 `eval(pixelValues)` en tête de `chatStreamMultimodal` : S.
2. S-02 complétude du téléchargement (index + marqueur), S-06 chemin renvoyé, S-07 `--force` : S.
3. S-05 fabrique privée par appel dans `loadContainer` : S.
4. S-09 pointeurs FFT, S-10 `eval` dans les processeurs async, S-11 détokeniseur en streaming : S.
5. S-04 avertissements dans la doc et `Gemma4ComputeGate` (optionnel) : S/M.
6. S-13 sortir `kvBits` de TurboQuant, S-17 filtrer la diffusion de `recommended` et `--all`, S-18 doc de plateforme, S-19 prévenir h3 et documenter `loadContainer` : S.
7. S-35 warnings, S-32 tests à corriger, CI minimale, CHANGELOG : M.
8. Hygiène S-36/S-37 : S.

**2.0.0 (cassures annoncées)**
- S-22/S-23/S-33 : suppression du code mort, des doublons et des fonctions libres, réduction de la surface ; TurboQuant et Diffusion en SPI ou dans un produit séparé ; BenchUI hors du paquet.
- S-15 : noms en collision rendus internes ou renommés.
- S-16 : protocole multimodal commun, 12B dans `chatStreamMultimodal`.
- S-20 : constructeur de prompt unique, avec les ids texte alignés sur HF.
- S-25 : décider du sort de `Gemma4MTPPipeline` face à `MTPSpeculativeTokenIterator`.
- S-08/S-12/S-34 : erreurs typées, prompt système par défaut neutre.
