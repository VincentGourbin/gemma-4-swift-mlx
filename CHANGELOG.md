# Journal des modifications

Toutes les modifications notables de ce projet sont consignées dans ce fichier.

Le format s'appuie sur [Keep a Changelog 1.1.0](https://keepachangelog.com/fr/1.1.0/),
et le projet adhère au [versionnage sémantique](https://semver.org/lang/fr/).

Les entrées sont écrites du point de vue d'un consommateur de la bibliothèque
`Gemma4Swift` ; les changements propres au CLI `gemma4-cli` sont signalés comme tels.

## [Non publié]

### Sécurité

- `swift-nio` sort entièrement du graphe de dépendances, ce qui rend sans objet
  CVE-2026-28980 (GHSA-rj37-6j9x-74q6, `HTTPDecoder` NIOHTTP1 sans borne sur les
  en-têtes) et CVE-2026-43671 (GHSA-r3rc-9hpw-54v9, écriture hors bornes dans
  `ByteBuffer`). Résolution refaite par `swift package update` : `EventSource` 1.5.1
  place NIO derrière un *package trait* désactivé par défaut, et `swift-atomics` /
  `swift-system` partent avec lui. Aucune borne de `Package.swift` modifiée ;
  `mlx-swift` passe de 0.31.4 à 0.31.6 dans son `.upToNextMinor`, `mlx-swift-lm`
  reste en 3.31.4. (#53, c9543739)

### Ajouté

- `Gemma4ReferenceProfile` : profils de référence `<bits>bit-<fast|lean>` (5 familles × 4/8/16
  bits × fast/lean), avec poids recommandés, `kvBits`, tranche de préfill, limites mémoire MLX
  et vidage du cache après réponse. `Gemma4Pipeline.load(profile:)` et `apply(profile:)` ; sans
  profil, comportement inchangé. CLI : `gemma4-cli references`, `bench --reference`.
  Valeurs initiales non mesurées (`docs/References.md`).
- Préfill par tranches (`prefillStepSize` enfin respecté) et head calculé sur le seul dernier
  jeton du prompt, texte et multimodal E2B/E4B (Unified inchangé).
- CLI : `gemma4-cli bench`, instrument de mesure du chemin de génération de la
  bibliothèque (`loadContainer` + `TokenIterator` de mlx-swift-lm, `asyncEval`
  compris) : préfill, TTFT, intervalle par jeton (médiane, p90), pics MLX et
  `phys_footprint`, une ligne JSON par point avec commit, versions des dépendances
  et machine. Remplace `profile` comme référence de mesure (sa boucle maison avait
  un `asyncEval` mort). Protocole : `docs/Benchmarks.md`.
- `Gemma4ComputeGate` : garde de processus, non bloquante, entre entraînement
  (gradient) et inférence, contre le deadlock mlx-swift gradient × forward.
  Erreurs `GateError.trainingInProgress`, `.inferenceInProgress(count:)`,
  `.trainingAlreadyRunning`. (6e12f675)
- `Gemma4StreamingDetokenizer` : détokenisation incrémentale exacte (différence par
  scalaires Unicode, attente sur U+FFFD), qui ne perd plus les drapeaux, séquences
  ZWJ, accents combinants ni caractères en repli octet par octet. (90cef4b5)
- `Gemma4ModelCache.localPath(modelId:)` : emplacement local réel d'un modèle, cache
  Hugging Face compris. (41d432f0)
- `Gemma4LanguageModel.makeCache(kvBits:slidingCapacity:)` : paramètre additif
  `slidingCapacity` pour dimensionner les caches glissants. (8d8708d8)
- `MTPError.cacheRollbackFailed` : nouveau cas d'une enum publique, levé quand le
  retrait des brouillons rejetés du cache est partiel. Un `switch` exhaustif sur
  `MTPError` côté consommateur doit l'ajouter. (8d8708d8)
- CI GitHub Actions : compilation de la bibliothèque, du CLI et des tests sur macOS
  (compilation seulement, les tests restent sur Apple Silicon via
  `Scripts/run-tests.sh`). Ce workflow n'a pas encore tourné. (e89802e6)

### Modifié

- **Rupture de comportement — streams annulables.** Tous les `AsyncThrowingStream`
  de `Gemma4Pipeline` (`chatStream`, `chatStreamMultimodal`, `continueChatStream`)
  et de `Gemma4MTPPipeline` (`mtpStream`, `mtpStreamFromTokens`) annulent désormais
  leur génération quand le consommateur arrête d'itérer. Auparavant la génération
  courait jusqu'à `maxTokens` en tenant le `ModelContainer`, et l'appel suivant
  attendait. (bd7c6fff)
- **Rupture de comportement — garde entraînement / inférence.** Une inférence
  (`chat`, `chatStream`, `chatStreamMultimodal`, `continueChat`,
  `continueChatStream`, boucle MTP) échoue avec `trainingInProgress` pendant un
  entraînement LoRA ou drafter, et un entraînement échoue avec
  `inferenceInProgress` pendant une inférence. Les inférences restent parallèles
  entre elles. Ne couvre pas les appels MLX directs d'un consommateur. (6e12f675)
- **Rupture de comportement — `kvBits` ne route plus vers TurboQuant.**
  `Gemma4LLMModel.newCache(parameters:)` ignore désormais `kvBits` : la
  quantification native du KV se fait pendant la génération (comportement
  documenté de mlx-swift-lm), pour toutes les familles. TurboQuant reste disponible
  explicitement via `languageModel.makeCache(kvBits:)` ; `gemma4-cli profile
  --kvBits` l'appelle ainsi et garde son comportement. (bcb0cdda)
- `Gemma4Registration.loadContainer(from:using:multimodal:)` n'écrit plus dans
  `LLMTypeRegistry.shared` : il charge via une `LLMModelFactory` privée dont le
  registre ne connaît que les types Gemma 4. Deux chargements concurrents avec des
  `multimodal:` différents reçoivent chacun le bon type. `register()` garde son rôle
  pour qui charge lui-même via `LLMModelFactory.shared`. (b69f4312)
- Détection des téléchargements incomplets : quand
  `model.safetensors.index.json` existe, chaque shard de son `weight_map` doit être
  présent (un index illisible vaut incomplet). Un téléchargement coupé est donc
  repris au lieu de passer pour complet. Un shard en lien symbolique vers un disque
  démonté compte comme présent. (41d432f0)
- `download(modelId:)` renvoie l'emplacement réel d'un modèle déjà présent
  (snapshot du cache HF compris) ; le snapshot HF retenu est le plus récemment
  modifié. (41d432f0)
- `chatStreamMultimodal` et `chatStream` quand il contourne `ChatSession`
  (n-gramme, variables de gabarit) génèrent des jetons bruts et détokenisent avec
  `Gemma4StreamingDetokenizer`. Conséquence : ces chemins ne passent plus par
  l'analyse d'appels d'outils de l'amont (dont les événements `.toolCall` étaient
  de toute façon ignorés). (44858e01)
- Le dépôt allège son historique suivi : 990 sorties brutes du bench OCR retirées
  (toujours consultables au commit c9543739), doublons d'images à la racine
  retirés, chemins personnels remplacés par des variables d'environnement dans
  `benchmarks/`. (e2dbd142)
- Avertissements de compilation de la bibliothèque ramenés de 37 à 2 (API mémoire
  `GPU.*` → `Memory.*`, `MLXFast.MLXFastKernel`, `AVURLAsset`, etc.). (bdb57823)
- Documentation d'audit (stabilité, performance, plan de fiches) ajoutée sous
  `docs/audit/2026-09-27/`. (2848d71d, f0acb128)

### Corrigé

- Le prefill multimodal (image, vidéo, audio) ne promeut plus les embeddings, le
  cache KV et tout le décodage en fp32 : ils restent au dtype des poids. (d96bc8e8)
- L'encodeur vision reste au dtype des poids ; avec `input_proj` quantifié à la
  volée (`--quantize-bits`), les patches ne sont plus tronqués en entiers. Les packs
  mlx-community n'étaient pas touchés. (134447b2)
- `kvBits` ne fait plus planter les couches à KV partagé (E2B/E4B), qui lisaient un
  `QuantizedKVCache` comme des (K, V) bruts. (bcb0cdda, 4670b9b3)
- MTP : les brouillons rejetés ne restent plus dans le cache une fois la fenêtre
  glissante (512) dépassée — la sortie ne se corrompt plus sur les longs prompts.
  (8d8708d8)
- `pixelValues` et les sorties de `processVideo` / `processAudio` sont évalués
  avant de changer de thread. (bd7c6fff)
- La boucle MTP et `describe` (CLI) ne perdent plus de caractères à la
  détokenisation. (90cef4b5)
- FFT audio : plus de pointeurs temporaires échappés (comportement indéfini).
  (90cef4b5)
- CLI : `gemma4-cli download --force` était sans effet ; il retélécharge
  désormais. (41d432f0)

### Retiré

- CLI : wrapper interne `LocalModelDownloader`, remplacé par
  `Gemma4ModelDownloader`. Aucune API publique de la bibliothèque retirée.
  (41d432f0)

### Limites connues

- `chat`, `chatStream` par défaut et `continueChat(Stream)` passent par
  `ChatSession` et gardent le défaut du détokeniseur amont
  (`NaiveStreamingDetokenizer`, différence par graphèmes : drapeaux, séquences ZWJ
  et accents combinants peuvent être perdus) jusqu'à sa correction dans
  mlx-swift-lm.
- MTP : même en contexte court, la génération spéculative peut diverger de la
  génération standard sur des quasi-égalités de logits. Défaut préexistant ;
  l'affirmation « bit-exact » du README est fausse en général.
- Deux avertissements « sending 'session' risks causing data races » subsistent
  (la `ChatSession` non `Sendable` conservée par le pipeline pour `continueChat`).
- Plusieurs correctifs restent à valider sur de vrais modèles au-delà d'E2B
  (`kvBits` sur E4B / 26B / 31B, `--quantize-bits 4` sur E2B bf16).

## [1.7.3] - 2026-09-12

### Ajouté

- `Gemma4Registration.loadContainer(from:using:multimodal:)` : charge via
  `LLMModelFactory.shared` directement, sans passer par `ModelFactoryRegistry`.
  (#50)

### Corrigé

- Dans tout processus qui lie `MLXVLM`, la fonction libre
  `MLXLMCommon.loadModelContainer(...)` pouvait rendre `MLXVLM.Gemma4` au lieu du
  modèle de ce paquet, quel que soit `multimodal:` (ordre VLM puis LLM de
  `ModelFactoryRegistry`). Symptôme : `unsupportedModelFamily` dans
  `chatStreamMultimodal`. `Gemma4Pipeline.load(from:multimodal:)` et le CLI
  passent par la nouvelle API ; aucune migration pour qui charge via le pipeline.
  Qui appelle lui-même `loadModelContainer` pour un Gemma 4 doit passer à
  `Gemma4Registration.loadContainer(...)`. (#50)

## [1.7.2] - 2026-09-12

### Corrigé

- Les `switch` sur `MLXLMCommon.Generation` absorbent les futurs cas via
  `@unknown default` : la compilation ne casse plus quand mlx-swift-lm est résolu
  sur une branche qui ajoute `.rejectedToolCall`. (#49)

## [1.7.1] - 2026-09-10

### Corrigé

- `Gemma4ModelCache.diskSize(for:)` suit les poids en lien symbolique (modèle
  relocalisé sur un disque externe) au lieu de rapporter la taille du lien, et
  compte 0 quand la cible est absente (disque démonté). (#47, #48)

## [1.7.0] - 2026-08-30

### Ajouté

- `Gemma4UnifiedVisionConfig.maxModelPatches` (== `numSoftTokens`), distinct de
  `maxPatches` qui compte des patches fins de 16 px. (#46)
- Couverture de tests de `Gemma4UnifiedImageProcessor`. (#46)

### Modifié

- `Gemma4UnifiedImageProcessor` (12B Unified) padde ses patches à
  `maxModelPatches` (280) au lieu de 2520 : le tenseur de sortie change de forme
  (~70 Mo → ~7,7 Mo par image, embedder vision ~5,6× plus rapide mesuré). (#46)
- `Gemma4UnifiedVisionConfig` rejette désormais, avec un `DecodingError`, une
  géométrie de patches incohérente ou des dimensions nulles. Les checkpoints
  mlx-community publiés sont cohérents. (#46)

### Corrigé

- Le modèle Unified lit les positions de padding dans les position ids au lieu de
  traiter une vingtaine de lignes de padding comme de l'image. (#46)

## [1.6.0] - 2026-08-30

### Ajouté

- Surcharges `async` de `Gemma4ImageProcessor.processImage(...:priority:)` et
  `Gemma4UnifiedImageProcessor.processImage(...:priority:)`, qui sortent le
  prétraitement image du thread appelant (et donc du main thread). `priority` est
  sans valeur par défaut : les appels synchrones existants sont inchangés. Demande
  de Fluxforge Studio. (#44)

### Modifié

- Conversion pixels → tenseur de `Gemma4ImageProcessor` vectorisée (MLX au lieu
  d'une boucle Swift scalaire), sortie bit à bit identique. (#44)

### Corrigé

- Les images très petites ou de ratio extrême ne retombent plus silencieusement en
  48×48. (#44)
- Rendu CoreGraphics sans pointeur temporaire échappé (comportement indéfini).
  (#44)

## [1.5.0] - 2026-08-20

### Ajouté

- `noRepeatNGramIncludesThinking` sur `chatStream` et `chatStreamMultimodal`
  (défaut `true`, comportement inchangé) : à `false`, les jetons du canal de pensée
  n'alimentent plus la fenêtre d'interdiction n-gramme. (#43)

## [1.4.0] - 2026-08-18

### Ajouté

- `templateVariables` sur `chatStream` et `chatStreamMultimodal` (transmis comme
  `additionalContext` au chat template), notamment pour `enable_thinking`. Défaut
  `nil` = rendu inchangé. Le stream reste brut : le filtrage du raisonnement est
  à la charge de l'appelant. (#42)

## [1.3.0] - 2026-08-18

### Ajouté

- `systemPrompt` sur `chatStreamMultimodal` : tour `<|turn>system` distinct, comme
  le chat template Gemma 4. (#41)
- `Gemma4Processor.strippingTemplateArtifacts` public. (#41)

### Modifié

- **Rupture de comportement — ids multimodaux.** À appel identique,
  `chatStreamMultimodal` ne produit plus les mêmes ids qu'en 1.2.x : le `\n`
  parasite émis après `<bos>` (et le `\n` en trop entre tours avec système) par
  swift-jinja est retiré, pour obtenir exactement les ids du rendu HF. Le chemin
  texte (`ChatSession`) porte encore le premier artefact. (#41)
- `chatStreamMultimodal` refuse un désaccord entre le nombre de marqueurs image et
  `pixelValues.dim(0)`, et un marqueur de modalité dans le prompt système, au lieu
  de recopier des embeddings au hasard. (#41)

### Corrigé

- `buildMultimodalPrompt` et `applyGemma4ChatTemplate` émettaient des marqueurs de
  tour Gemma 3 (`<start_of_turn>`), absents du vocabulaire Gemma 4 et tokenisés en
  texte littéral. Touchait l'évaluation LoRA multimodale. (#41)

## [1.2.0] - 2026-08-17

### Ajouté

- Fenêtre d'interdiction n-gramme configurable :
  `NoRepeatNGramLogitProcessor(ngramSize:includePromptInWindow:)` et
  `noRepeatNGramIncludesPrompt` sur `chatStream` / `chatStreamMultimodal` (défaut
  `true` = parité HF). À `false`, le modèle peut citer son prompt verbatim. (#40)
- `Scripts/run-tests.sh`, point d'entrée obligatoire des tests, qui contourne un
  deadlock mlx-swift figeant la suite complète. (#39)

## [1.1.0] - 2026-08-14

### Ajouté

- `forwardCollectingHiddenStates` (convention HF `output_hidden_states`) sur
  `Gemma4TextModel`, `Gemma4LanguageModel`, `Gemma4LLMModel` et
  `Gemma4UnifiedMultimodalLLMModel`, pour un usage en encodeur texte. (#35, #36)
- `noRepeatNGramSize` sur `chatStream` et `chatStreamMultimodal` (parité
  `no_repeat_ngram_size` de HF), via `NoRepeatNGramLogitProcessor`. (#38)

### Corrigé

- Le nettoyage des K/V des couches KV-shared ne touche plus les tours vision et
  audio : `load(multimodal: true)` sur E2B échouait en `keyNotFound`. (#37)

## [1.0.0] - 2026-08-07

Première version publiée sous tag semver : les projets aval peuvent dépendre du
paquet par version (`from: "1.0.0"`) au lieu de `branch: "main"`.

### Ajouté

- État initial : portage Swift/MLX de Gemma 4 (familles E2B, E4B, 26B-A4B MoE,
  31B, et 12B Unified) — génération texte, vision, vidéo image par image, audio
  (E2B/E4B), filtre du mode thinking, décodage spéculatif MTP, LoRA, TurboQuant
  (expérimental), DiffusionGemma, outillage de profiling, CLI `gemma4-cli`.

### Modifié

- `mlx-swift-lm` (3.31.4) et `mlx-swift` (0.31.4) bornés en `.upToNextMinor` au
  lieu d'une dépendance de branche. (#32)

### Corrigé

- Chargement des checkpoints `gemma-4-e4b-it-4bit` actuels, en texte comme en
  multimodal : `ScaledLinear` quantifiable (`per_layer_model_projection`), et
  `k_proj` / `v_proj` / `k_norm` absents des couches KV-shared.

[Non publié]: https://github.com/VincentGourbin/gemma-4-swift-mlx/compare/1.7.3...HEAD
[1.7.3]: https://github.com/VincentGourbin/gemma-4-swift-mlx/compare/1.7.2...1.7.3
[1.7.2]: https://github.com/VincentGourbin/gemma-4-swift-mlx/compare/1.7.1...1.7.2
[1.7.1]: https://github.com/VincentGourbin/gemma-4-swift-mlx/compare/1.7.0...1.7.1
[1.7.0]: https://github.com/VincentGourbin/gemma-4-swift-mlx/compare/1.6.0...1.7.0
[1.6.0]: https://github.com/VincentGourbin/gemma-4-swift-mlx/compare/1.5.0...1.6.0
[1.5.0]: https://github.com/VincentGourbin/gemma-4-swift-mlx/compare/1.4.0...1.5.0
[1.4.0]: https://github.com/VincentGourbin/gemma-4-swift-mlx/compare/1.3.0...1.4.0
[1.3.0]: https://github.com/VincentGourbin/gemma-4-swift-mlx/compare/1.2.0...1.3.0
[1.2.0]: https://github.com/VincentGourbin/gemma-4-swift-mlx/compare/1.1.0...1.2.0
[1.1.0]: https://github.com/VincentGourbin/gemma-4-swift-mlx/compare/1.0.0...1.1.0
[1.0.0]: https://github.com/VincentGourbin/gemma-4-swift-mlx/releases/tag/1.0.0
