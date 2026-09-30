# Journal des modifications

Toutes les modifications notables de ce projet sont consignées dans ce fichier.

Le format s'appuie sur [Keep a Changelog 1.1.0](https://keepachangelog.com/fr/1.1.0/),
et le projet adhère au [versionnage sémantique](https://semver.org/lang/fr/).

Les entrées sont écrites du point de vue d'un consommateur de la bibliothèque
`Gemma4Swift` ; les changements propres au CLI `gemma4-cli` sont signalés comme tels.

## [Non publié]

## [1.8.0] - 2026-09-30

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

- `GEMMA4_MODELS_DIR` : racine des modèles déplaçable (ex. disque externe), comme
  `QWEN38_MODELS_DIR` ; priorité `customModelsDirectory` → variable → `~/Library/Caches/models`,
  modèles déjà présents dans l'emplacement par défaut toujours trouvés.
  `Gemma4ModelCache.defaultModelsDirectory`, `environmentModelsDirectory`,
  `isOnUnmountedVolume(_:)` ; `Gemma4DownloadError.volumeNotMounted` (nouveau cas) : un
  téléchargement vers un disque externe non monté échoue au lieu d'écrire sur le disque interne.
- `Gemma4ReferenceProfile` : profils de référence `<bits>bit-<fast|lean>` (5 familles × 4/8/16
  bits × fast/lean), avec poids recommandés, `kvBits`, tranche de préfill, limites mémoire MLX
  et vidage du cache après réponse. `Gemma4Pipeline.load(profile:)` et `apply(profile:)` ; sans
  profil, comportement inchangé. CLI : `gemma4-cli references`, `bench --reference`.
  Matrice mesurée (30 profils, `docs/References.md`) ; voir plus bas `tiny`, audio et tours.
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
  `Scripts/run-tests.sh`), et build iOS (`generic/platform=iOS`). Déclenché par les pull
  requests et `main`. (e89802e6, 198d0054)

#### Inférence et profils

- `Gemma4ChatEngine` : moteur de conversation sans dépendance serveur — messages de rôles
  `system`/`user`/`assistant`/`tool`, appels d'outils au format Gemma 4, canal de pensée
  séparé (`reasoning`), images, annulation (`Gemma4ChatRun.cancel()`), usage
  (`Gemma4ChatUsage`, dont `cachedPromptTokens`). Protocole `Gemma4ChatBackend` pour
  brancher un autre moteur. (4762bed5)
- Réutilisation du préfixe de conversation entre tours (`setReusesConversation`,
  `configureConversationCache(capacity:budgetBytes:)`, LRU par budget d'octets) : TTFT
  −78 % au tour suivant. (29f3c27f, 90ff0c3b)
- Profil `e2b/4bit-tiny` : E2B 4 bits sous 4 Go d'empreinte (texte 3,1 Go, image 3,9 Go),
  pour iPhone/iPad ; `Gemma4ReferenceProfile.recommended()` le choisit sous 6 Go disponibles.
  Guide : `docs/iOS.md`. (198d0054)
- Chargement sans tour audio : `Gemma4Pipeline.load(from:multimodal:audio:)`,
  `Gemma4Registration.loadContainer(..., audio:)`, `Gemma4ReferenceProfile.audio` et
  `withAudioVariant()` (−0,6 Go). Un audio envoyé sans tour est refusé
  (`audioTowerUnavailable`), jamais ignoré. (9a8ac015)
- Tours vision/audio libérables après le préfill et rechargées à la demande
  (`releaseEncodersAfterPrefill`, `releaseEncoders()`, `restoreEncodersIfNeeded()`) :
  −1,24 Go en régime, +80 ms à l'image suivante. (b1e665f3)
- `NoRepeatNGramLogitProcessor` : historique tenu sur le GPU, plus de synchronisation par
  jeton (+17 % de débit avec n-gramme) ; `forceHostHistory` pour l'ancien chemin. (1ab24fc2)

#### Entraînement (LoRA, drafter MTP)

- `Gemma4TrainingProfile` : 12 profils `lora-<bits>bit-<fast|lean>` sur les 5 familles,
  11 publiés avec leur mesure (pic, empreinte, débit, perte). Porte qualité E7 sur E4B
  `lora-16bit-fast` : 29/30 (base 19/30). CLI : `lora train --reference`, `lora profiles`.
  (1a17f7d4, 5944a7b0, e5fe4746)
- Checkpoints sûrs et reprise exacte : `adapter_config.json` écrit au démarrage,
  sauvegardes atomiques numérotées, état de l'optimiseur (`Gemma4ResumableAdam`, calcul
  identique à l'Adam de MLX), `Gemma4TrainingCheckpoint`, `TrainingConfig.resume`. (8e6b26a9)
- `TrainingConfig` : `seed` (init LoRA, dropout et mélange reproductibles, `SeededGenerator`),
  `maxSeqLength`, `validationBatches`, `metricsURL` (une ligne JSON par rapport,
  `Gemma4TrainingMetrics`), `responseOnlyHead`, `memoryPolicy`
  (`Gemma4TrainingMemoryPolicy`), `multimodalFloat32`, `gradientCheckpointing` (pic −45 %,
  temps +35 %, pertes identiques). (413c3604, f71ed1a4, fc110440, 1335189e, 1aa8ce24)
- `Gemma4LoRAInference.fuseAndSave` : modèle fusionné complet et rechargeable (gabarit et
  `processor_config.json` copiés, `config.json` cohérent). (4db495c6)
- `Gemma4LoRADefaults.ModelFamily.from(directory:)` (famille lue dans `config.json`) et
  famille `b12b`. (b5a4ddef)
- `Gemma4Processor.droppingGenerationPrompt(_:)` : retire l'invite de génération quelle que
  soit sa longueur selon le gabarit. (bd867d40)

#### DiffusionGemma

- Profils `a4bdiff/{16,8,4}bit-{fast,lean}` (`DiffusionReferenceProfile`), 4 bits en
  précision mixte (couches 0-3 et 26-29 en 8 bits : autant de passes que le bf16, 38,7 tok/s).
  (d9a03081, 50bef8fc)
- Packs pré-quantifiés (`gemma4-diffusion-prequantized-v1`, SHA-256 par fichier) :
  `export-diffusion`, chargement direct (4 bits : 11 s et 18,8 Go de pic au lieu de 62 s et
  51 Go). Publiés sur Hugging Face (`VincentGOURBIN/diffusiongemma-26B-A4B-it-gemma4swift-8bit`
  et `-4bit-mixed`), `DiffusionReferenceProfile.weightsRepository`, CLI
  `download diff-8bit` / `diff-4bit`. (3c869e5e, 06f93ced)
- CLI : `bench-diffusion`, `eval-screenspot` (ScreenSpot-100 : bf16 80, 8 bits 78,
  4 bits 76). (b94a0e1e, 5c50f47e)

#### Serveur et outils

- Paquet imbriqué `Server/` (produit séparé, la bibliothèque ne dépend toujours d'aucun
  serveur) : `gemma4-server`, API OpenAI (`/v1/chat/completions` avec SSE, outils, pensée,
  images ; `/v1/models`, `/healthz`, `/metrics`), loopback par défaut, clé d'API obligatoire
  hors loopback, file bornée (429), annulation à la déconnexion, cache de conversation.
  (ef54c1bc, 90ff0c3b)
- CLI : `eval-mmlu` avec préfixe 5-shot en cache (−75 % de temps), IC95 de Wilson, jeu
  archivé de 1 140 questions, `--out`. (f936bc97)
- CLI : `bench --no-repeat-ngram`, `--no-audio`, `--reference` ; `output_sha` pour la
  parité A/B. (9a853c44, 5074ef36)

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
- README revu : 5 familles (12B ajouté), vitesses remplacées par la matrice mesurée des profils
  (`gemma4-cli bench`), 12B en 8 bits conseillé, KV quantifié natif, exemples d'API corrigés
  (`Gemma4Registration.loadContainer`, ordre des arguments de `TrainingConfig`), MTP non bit-exact.
- Le dépôt allège son historique suivi : 990 sorties brutes du bench OCR retirées
  (toujours consultables au commit c9543739), doublons d'images à la racine
  retirés, chemins personnels remplacés par des variables d'environnement dans
  `benchmarks/`. (e2dbd142)
- Avertissements de compilation de la bibliothèque ramenés de 37 à 2 (API mémoire
  `GPU.*` → `Memory.*`, `MLXFast.MLXFastKernel`, `AVURLAsset`, etc.). (bdb57823)
- Documentation d'audit (stabilité, performance, plan de fiches) ajoutée sous
  `docs/audit/2026-09-27/`. (2848d71d, f0acb128)

- **Rupture de comportement — gabarit de chat normalisé.** Le contrôle d'espaces du
  `chat_template.jinja` est corrigé à la volée (défaut de swift-jinja) : le rendu est
  désormais identique à celui de Hugging Face, donc les ids de prompt changent légèrement
  par rapport à 1.7.x. (4762bed5)
- **Rupture de comportement — profils `lean`** : sans tour audio et avec tours vision
  libérées après le préfill (`withAudioVariant()` pour l'audio). (b1e665f3)
- **Rupture de comportement — entraînement LoRA, défauts** : tête et perte sur la seule
  réponse avec `maskPrompt` (+26 % de débit, perte identique), cache MLX borné à 2 Go
  (empreinte 76 → 16 Go sur director), pas de troncature par défaut, multimodal en base bf16
  avec LoRA en fp32 (pic 24 → 14 Go ; `multimodalFloat32` pour l'ancien chemin).
  (fc110440, d3bd5a42, 1335189e)
- Vision : seuls les patches réels passent dans l'encodeur (plus de padding systématique à
  2 520), vidéo ×2,5 par frame ; `padToMaxPatches` pour l'ancien comportement. Sorties à
  3,8·10⁻⁴ près. (af8e647b)
- Quantification à la volée : les experts MoE (`SwitchLinear`) sont enfin quantifiés,
  routeur en 8 bits (26B-A4B, DiffusionGemma). (2d6dbab5)
- `swift-mlx-profiler` 1.5.1, dépendance inconditionnelle (compile pour iOS). (cf2b8f6c)

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

- LoRA sur 26B-A4B : l'entraînement mourait au premier pas (`gatherMM`, VJP par rapport aux
  indices du routeur) ; indices hors gradient. (16dd66f4)
- LoRA, multimodal et `mtp-train` sur 12B, 26B-A4B et 31B : l'invite de génération de ces
  gabarits (`<|channel>thought\n<channel|>` après `<|turn>model\n`) n'était retirée qu'en
  partie, et le masque de réponse ne gardait que 4 jetons par exemple (au lieu de ~750).
  E2B/E4B inchangés. (bd867d40)
- LoRA : l'écrêtage de gradient était affiché mais jamais appliqué ; mélange non
  reproductible ; lignes rejetées sans explication ; full fine-tune accepté sur un pack
  quantifié. (413c3604)
- LoRA multimodal : ids directs au format de l'inférence (plus d'aller-retour
  décodage/encodage) ; `lora eval` masqué comme l'entraînement. (f235d67c)
- Drafter MTP : gabarit aligné sur l'inférence, poids chargés sans ambiguïté, entrées
  validées (`DrafterTrainingError`), tirage seedé. (1f754c00)
- DiffusionGemma : critère d'arrêt, fenêtre glissante de l'encodeur, blocs image
  bidirectionnels, une seule copie des poids partagés, quantification couche par couche
  (pic 77 → 51 Go), image après déchargement de la vision plus ignorée, génération
  annulable et sous `Gemma4ComputeGate`. (b80f4ea0, 4a0b9ccd, fabe68ee, b326083a, bf93a1ce,
  50bef8fc, 9ffc8871, f6b30ffc)

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
  le README ne l'annonce plus « bit-exact ».
- Deux avertissements « sending 'session' risks causing data races » subsistent
  (la `ChatSession` non `Sendable` conservée par le pipeline pour `continueChat`).
- Plusieurs correctifs restent à valider sur de vrais modèles au-delà d'E2B
  (`kvBits` sur E4B / 26B / 31B, `--quantize-bits 4` sur E2B bf16).
- Entraînement : `b31b/lora-4bit-lean` diverge au lr de 1e-4 (non publié). Seul
  `e4b/lora-16bit-fast` a passé la porte E7 ; les autres profils sont mesurés sur 50 pas.
- Échantillonnage top-p : 5,6 % du débit de décodage (tri sur 262 k logits), issue #54.
- mlx-swift 0.32 : la montée demande un correctif (`prepare` du protocole, sinon médias
  ignorés sans erreur), prêt dans `docs/audit/2026-09-27/mlx-swift-0.32-migration.patch` ;
  en attente d'une release de mlx-swift-lm.
- `e2b/4bit-tiny` n'a pas été mesuré sur un iPhone réel.

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

[Non publié]: https://github.com/VincentGourbin/gemma-4-swift-mlx/compare/1.8.0...HEAD
[1.8.0]: https://github.com/VincentGourbin/gemma-4-swift-mlx/compare/1.7.3...1.8.0
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
