# Audit complémentaire — fonctions annexes (entraînement, évaluation) et serveur d'inférence

- Date : 2026-09-27, `main` @ `c9543739` (mlx-swift 0.31.6, mlx-swift-lm 3.31.4). Complète
  [audit-stabilite.md](audit-stabilite.md) (S-xx) et [audit-performance.md](audit-performance.md) (P-xx) sans les
  répéter ; les renvois S-/P-/K- pointent vers ces rapports et vers [PLAN.md](PLAN.md).
- Lecture seule. Aucun build ni banc du dépôt. Les seules commandes exécutées hors lecture sont trois
  **expériences SwiftPM jetables** dans le scratchpad de la session (paquets jouets, § 2.5), pour trancher la
  question des dépendances.
- Sources lues : `Sources/Gemma4Swift/LoRA/*`, `Speculative/Gemma4DrafterTraining.swift`,
  `Sources/Gemma4CLI/{LoRACommand,MtpTrainCommand,MtpDrafterLoader,MtpGenerateCommand,EvalCommand}.swift`, mémoire
  `feedback_lora_training.md` et `project_lora_status.md`, le serveur `qwen38 serve` (`Qwen38Server.swift`,
  `Qwen38BatchScheduling.swift`, README § « OpenAI-compatible server », `docs/reference/claude-code-wire-format.md`),
  le gabarit `chat_template.jinja` et `generation_config.json` du pack `gemma-4-e2b-it-bf16` (disque Lexar),
  le parseur d'outils amont (`MLXLMCommon/Tool/*`).
- Légende : **VÉRIFIÉ** = constaté dans le code ou par une commande reproductible ; **À VÉRIFIER** = déduit du
  code, effet à confirmer ; **À MESURER** = chiffre à produire par la baseline. Effort : S (< ½ j), M (½–2 j), L (> 2 j).

> **Mémoire périmée à corriger.** L'index `MEMORY.md` annonce « 14 % vs 87 % gap to fix » ; les fiches elles-mêmes
> (`feedback_lora_training.md`, `project_lora_status.md`) disent l'inverse : bug 5 (KV-partage absent à
> l'entraînement) corrigé, **97,2 % Swift contre 95,3 % Python** sur le classifieur ToolsForge (108 exemples de test),
> 14,3 % étant l'état *avant* correction. La porte qualité retenue ici est donc « ne pas régresser sous 95,3 % ».

---

## Partie 1 — Fonctions annexes

### 1.0 Carte

| Fonction | Entrée bibliothèque | Commande CLI | Consommateurs externes |
|---|---|---|---|
| LoRA/DoRA/full texte | `Gemma4LoRATrain.train` (`Gemma4LoRATrain.swift:100`) → `trainLoRA` (`Gemma4TrainingLoop.swift:153`) | `lora train` | aucun (audit-stabilite § F) |
| LoRA multimodal (image/audio) | `trainMultimodal` (`:311`) → `trainMultimodalLoRA` (`TrainingLoop:333`) | `lora train --multimodal` | aucun |
| Inférence avec adaptateur | `Gemma4LoRAInference.loadAdapter/fuseAdapter/unloadAdapter`, `Gemma4Pipeline.loadAdapter` | `lora generate`, `lora fuse`, `lora bench-multimodal`, `lora eval` | `loadAdapter` utilisé (audit-stabilite § F) |
| Drafter MTP | `Gemma4DrafterTraining.trainDrafter` (`:139`) | `mtp-train` | aucun |
| MMLU 5-shot | — | `eval-mmlu` (`EvalCommand.swift`) | — |
| Diagnostics MTP, profil, diffusion | — | `mtp-smoke/forward/diag-verify`, `profile`, `diffusion` | déjà traités : S-24, P-07, hors plan |

Aucun consommateur externe n'appelle l'entraînement : les corrections du lot H peuvent changer des défauts CLI
sans casser d'API (seuls des ajouts côté bibliothèque).

### 1.1 Synthèse des constats

| ID | Sév. | Thème | Titre court | Statut |
|---|---|---|---|---|
| A-01 | haute | correction | `gradClipMaxNorm` accepté, affiché, **jamais appliqué** | VÉRIFIÉ |
| A-02 | haute | reprise | Run interrompu = adaptateur inutilisable (config écrite à la fin, fichier écrasé en place, pas de reprise, pas d'arrêt propre) | VÉRIFIÉ code / effet À VÉRIFIER |
| A-03 | haute | correction | Multimodal et `lora eval` réintroduisent l'aller-retour decode→encode (bug 4 de la mémoire) ; `lora eval` sans masquage | VÉRIFIÉ |
| A-04 | moyenne | correction | Format des médias différent entre entraînement multimodal et inférence ; médias perdus en silence | VÉRIFIÉ code / effet À MESURER |
| A-05 | moyenne | correction | `mtp-train` au format chat réintroduit les bugs 1 et 3 (suffixe `<|turn>model\n`, `\n` parasite) | VÉRIFIÉ |
| A-06 | moyenne | données | Validation des données silencieuse, pas de longueur maximale, options inconnues avalées | VÉRIFIÉ |
| A-07 | moyenne | correction | `--fine-tune-type full` sur un pack quantifié n'entraîne presque rien, sans avertissement | VÉRIFIÉ code / effet À VÉRIFIER |
| A-08 | moyenne | mesure | Entraînements non reproductibles (mélange non seedé) : « loss à N pas » incomparable | VÉRIFIÉ |
| A-09 | moyenne | mesure | Le tok/s rapporté compte les jetons **entraînés**, pas les jetons traités | VÉRIFIÉ |
| A-10 | basse | correction | Masque de perte : une cible de padding par échantillon court quand batch > 1 | VÉRIFIÉ code / parité mlx-lm À VÉRIFIER |
| A-11 | moyenne | drafter | `--drafter-path <dossier>` mélange `drafter.safetensors` et `drafter.best.safetensors` dans un ordre non déterministe | VÉRIFIÉ |
| A-12 | basse | drafter | `precondition`/`fatalError` sur entrées, pas de contrôle drafter↔cible, pas de reprise | VÉRIFIÉ |
| A-13 | moyenne | concurrence | L'entraînement tient le `ModelContainer` pendant des heures (famine, et ABBA si second conteneur) | VÉRIFIÉ code |
| A-14 | basse | évaluation | `eval-mmlu` sur 100 questions : IC95 ≈ ±10 pts ; petits défauts d'outil | VÉRIFIÉ |
| A-15 | moyenne | mesure | Jeux de données des portes qualité introuvables (`/tmp`) : aucune porte rejouable aujourd'hui | VÉRIFIÉ |
| A-16 | basse | fuse | `lora fuse` : `chat_template.jinja` et `processor_config.json` non copiés, erreurs avalées | VÉRIFIÉ (fichiers) / effet À VÉRIFIER |
| A-17 | basse | CLI | `lora generate` / `bench-multimodal` : boucles manuelles, sorties par défaut dans `/tmp`, division par zéro | VÉRIFIÉ |
| A-18 | haute | mémoire | Logits pleine vocabulaire en fp32 sur **toutes** les positions à chaque pas, même avec `--mask-prompt` | VÉRIFIÉ code / gain À MESURER |
| A-19 | haute | mémoire | Multimodal : tout le modèle converti en fp32 (≈ 2× les poids) | VÉRIFIÉ code / cause des NaN À MESURER |
| A-20 | moyenne | mémoire | Aucune politique mémoire MLX pendant l'entraînement | VÉRIFIÉ code / impact À MESURER |
| A-21 | moyenne | vitesse | Validation sur **tout** le jeu, dès le pas 0 puis tous les 50 pas | VÉRIFIÉ code / coût À MESURER |
| A-22 | moyenne | mémoire | Adaptateurs k_proj/v_proj morts sur les couches KV-partagées (E2B : toutes les couches adaptées) | VÉRIFIÉ |
| A-23 | moyenne | mémoire | Pas de gradient checkpointing (non exposé en Swift par mlx-swift) | VÉRIFIÉ |
| A-24 | moyenne | vitesse/mémoire | Multimodal : encodeurs gelés recalculés à chaque pas ; tous les médias du jeu gardés en mémoire | VÉRIFIÉ code / À MESURER |
| A-25 | moyenne | drafter | Forward complet de la cible tracé dans la VJP à chaque pas | VÉRIFIÉ code / À MESURER |
| A-26 | basse | évaluation | `eval-mmlu` re-préremplit le préfixe 5-shot à chaque question, logits sur toutes les positions | VÉRIFIÉ code / À MESURER |

### 1.2 Stabilité et correction

#### A-01 — Écrêtage de gradient fantôme (haute, VÉRIFIÉ)
- `TrainingConfig.gradClipMaxNorm` (`Gemma4LoRATrain.swift:53, 71, 87`) n'est lu qu'une fois, pour le profil
  (`:136`). `trainLoRA` et `trainMultimodalLoRA` n'ont pas de paramètre d'écrêtage (`TrainingLoop.swift:153-166, 333-345`) ;
  `grep -rn gradClip Sources` ne trouve aucun autre usage.
- Le CLI force **0,3 en full** et l'affiche (`LoRACommand.swift:168, 174-175, 262`) : l'utilisateur croit écrêter.
- Correction : `clipGradNorm(gradients:maxNorm:)` (MLXOptimizers) entre `valueAndGrad` et `optimizer.update` dans les
  deux boucles, ou retrait du champ et refus de l'option. Effort S.

#### A-02 — Checkpoints non réutilisables, pas de reprise (haute ; VÉRIFIÉ code, effet À VÉRIFIER)
- `adapter_config.json` n'est écrit **qu'après** la boucle (`Gemma4LoRATrain.swift:270-279`, `:441-446`). Or
  `LoRAContainer.from(directory:)` exige ce fichier (mlx-swift-lm `LoRAContainer.swift:120-123`).
- La sauvegarde intermédiaire (tous les 50 pas, figé dans le CLI : `LoRACommand.swift:165, 259`) **écrase le même
  fichier** `adapters.safetensors` (`TrainingLoop.swift:227-234`) : pas d'historique, pas de « meilleur », et une
  coupure pendant l'écriture laisse le seul exemplaire corrompu (écriture non atomique, À VÉRIFIER dans `MLX.save`).
- Ni pas courant, ni état Adam, ni graine ne sont sauvegardés : aucune reprise possible (pas d'équivalent au
  `--resume-adapter-file` de mlx-lm).
- Arrêt : le callback CLI renvoie toujours `.more` (`LoRACommand.swift:186-189, 279-282`) et aucune boucle ne teste
  `Task.isCancelled`. Un Ctrl-C tue le process sans sauvegarde finale.
- Conséquence : un run de 1 300 pas interrompu au pas 1 000 laisse un `adapters.safetensors` du pas 1 000 **sans
  configuration**, que `lora generate`/`Gemma4Pipeline.loadAdapter` refusent de charger.

#### A-03 — Aller-retour decode→encode réintroduit (haute, VÉRIFIÉ)
- La mémoire (bug 4) établit que `applyChatTemplate → decode → encode` n'est pas fidèle en Swift, et le chemin texte
  tokenise désormais directement (`LoRACommand.swift:95-137`).
- Le chemin **multimodal** refait l'aller-retour : le formatteur décode les ids (`LoRACommand.swift:213-224`), puis
  `preprocessMultimodalSamples` ré-encode le texte (`:319-322`).
- `lora eval` aussi : formatteur qui décode (`:416-424`), puis `LoRATrain.evaluate` amont qui ré-encode
  (`Gemma4LoRATrain.swift:467-482`). En outre, cette évaluation amont **ne masque pas le prompt** : la « test loss »
  n'est pas comparable à la perte de validation de l'entraînement (masquée).
- Correction : ids directs partout ; `lora eval` réutilise `evaluateTraining` (boucle maison, masquage identique).

#### A-04 — Médias : format d'entraînement ≠ format d'inférence (moyenne ; VÉRIFIÉ code, effet À MESURER)
- Entraînement : `boi + 280×image + eoi` inséré juste après `<|turn>user\n`, **sans** saut de ligne avant le texte
  (`LoRACommand.swift:327-356`).
- Inférence (`lora bench-multimodal`, pipeline) : `Gemma4Processor.buildMultimodalPrompt` joint médias et texte par
  `"\n"` (`Gemma4Processor.swift:106-110`). Un jeton de différence à chaque exemple (sixième constructeur de prompt, S-20).
- Si aucun tour `user` n'est trouvé, les jetons média ne sont pas insérés mais `pixelValues`/`audioFeatures` sont
  conservés (`:336-374`) : `maskedScatter` reçoit alors des embeddings sans positions.
- `for j in 0 ..< tokens.count - 2` (`:328`) piège (plage invalide) si le texte fait moins de 2 jetons.
- Ne pas conclure que cela explique les 50 % du jeu birdcall (README) : à mesurer après alignement.

#### A-05 — `mtp-train` au format chat : bugs 1 et 3 de nouveau (moyenne, VÉRIFIÉ)
- `parseAndTokenizeJsonlLine` appelle `tokenizer.applyChatTemplate(messages:)` (`MtpTrainCommand.swift:249-256`) :
  génération de prompt active par défaut (suffixe `<|turn>model\n`, bug 1) et pas de
  `Gemma4Processor.strippingTemplateArtifacts` (`\n` parasite après `<bos>`, bug 3).
- Les messages dont `content` n'est pas une chaîne (parties multimodales) sont supprimés en silence (`compactMap`, `:251-255`).
- Impact : le drafter apprend sur des séquences que la cible ne voit jamais à l'inférence (effet sur le taux
  d'acceptation À MESURER).

#### A-06 — Données : rejets silencieux, pas de longueur maximale (moyenne, VÉRIFIÉ)
- Lignes JSONL ne commençant pas exactement par `{` ignorées (`Gemma4LoRAData.swift:162, 201`, `LoRACommand.swift:103`,
  `:645`) ; lignes non décodables → `try?` puis `nil` (`Gemma4LoRAData.swift:209-225`). Aucun compte des rejets.
- Le CLI texte fait l'inverse : `JSONDecoder().decode` lève sur la première ligne invalide, sans numéro de ligne (`:112`).
- **Aucune troncature** : `paddedLength = maxLength` (`TrainingLoop.swift:100-103`) ; `grep -i "maxSeq|truncat"` vide.
  Un exemple de 20 k jetons part tel quel (OOM probable). mlx-lm tronque à 2 048 par défaut.
- `--fine-tune-type` inconnu → `lora` sans message (`LoRACommand.swift:153, 247`).
- Famille déduite du **chemin** (`Gemma4LoRAConfig.swift:37-45`) : 12B absent de l'énumération, tout modèle inconnu
  prend les défauts E2B (8 couches) ; commentaires de couches périmés (« 26 layers » pour E2B, qui en a 35 d'après
  `config.json`). `trainMultimodalLoRA` plante sur le 12B unifié (`as!`, déjà S-12).

#### A-07 — Full fine-tune sur pack quantifié (moyenne ; VÉRIFIÉ code, effet À VÉRIFIER)
- `QuantizedLinear.init` appelle `self.freeze()` (mlx-swift `MLXNN/Quantized.swift:201`) ; le mode full ne dégèle
  rien (`Gemma4LoRATrain.swift:153-157`). Sur un pack 4/8 bits, seuls normes et paramètres non quantifiés
  s'entraînent. Le pourcentage imprimé (`:168-176`) le trahit, mais rien ne refuse ni n'avertit.
- Correction : refuser `full` si le modèle contient des `QuantizedLinear` (ou proposer QLoRA).

#### A-08 — Non-reproductibilité (moyenne, VÉRIFIÉ)
- `permutation.shuffle()` utilise le générateur système, non seedé (`TrainingLoop.swift:83, 90, 312, 318`) ;
  `MLXRandom.seed(0)` (`Gemma4LoRATrain.swift:151, 363`) ne couvre que l'initialisation LoRA et le dropout.
- Drafter : `chunks.randomElement()!` (`Gemma4DrafterTraining.swift:199`), aucune graine.
- Conséquence : deux runs identiques n'ont pas la même courbe ; une comparaison A/B de « loss à N pas » mesure le
  mélange, pas le levier. Prérequis de toute baseline d'entraînement.

#### A-09 — Débit rapporté trompeur (moyenne, VÉRIFIÉ)
- `tokenCount += tokens.item(Int.self)` où `tokens` = nombre de positions **masquées** (`TrainingLoop.swift:35, 195, 202`).
  Avec `--mask-prompt` (recommandé), le « tok/s » ne compte que les jetons de réponse : pour un classifieur dont
  la réponse fait une dizaine de jetons sur quelques centaines, le débit réel est sous-estimé d'un ordre de grandeur.
- Correction : publier `tokens_processed_per_s` (B × L) **et** `tokens_trained_per_s`.

#### A-10 — Masque : cible de padding avec batch > 1 (basse ; VÉRIFIÉ code, parité mlx-lm À VÉRIFIER)
- `mask = (steps >= offset) && (steps <= total)` avec `steps = 1…L-1` (`TrainingLoop.swift:28-32`). La cible d'indice
  `i` est le jeton `i+1` ; `steps = total` sélectionne donc le jeton d'indice `total`, qui est du padding (`0 = <pad>`)
  pour tout échantillon plus court que le plus long du lot (`:106-110`).
- Sans effet à batch 1 (configuration validée). À comparer à mlx-lm (`default_loss`) avant de corriger : si l'amont
  a la même inégalité, le signaler en amont plutôt que diverger.

#### A-11 — Chargement d'un drafter fine-tuné non déterministe (moyenne, VÉRIFIÉ)
- `mtp-train` écrit `drafter.safetensors` **et** `drafter.best.safetensors` dans le même dossier
  (`Gemma4DrafterTraining.swift:259-262, 277-279`).
- `--drafter-path <dossier>` charge tous les `.safetensors` du dossier via `contentsOfDirectory` (ordre non spécifié)
  et fusionne les clés, dernier lu gagnant (`MtpGenerateCommand.swift:101-121`, `Gemma4CLI.swift:461-470`). Les
  deux fichiers ayant les mêmes clés, le résultat est l'un ou l'autre selon l'ordre du système de fichiers.
- Le README passe un **fichier** (`--drafter-path ./my-drafter/drafter.best.safetensors`) : correct ; le dossier est le piège.

#### A-12 — Drafter : garde-fous (basse, VÉRIFIÉ)
- `precondition(L >= 3)` dans une fonction publique (`Gemma4DrafterTraining.swift:38`) alors que `--seq-len` n'est
  pas validé ; `fatalError` bibliothèque (`:52`) et CLI (`MtpTrainCommand.swift:130, 140, 183`).
- Aucun contrôle que le drafter correspond à la cible (`bind` se contente de brancher l'embedding, `Gemma4AssistantDraftModel.swift:109-113`) :
  un drafter E2B sur une cible E4B échoue plus loin sur une forme.
- La bibliothèque jette en silence les échantillons plus courts que `seqLen` (`chunkify`, `:156-166`).
- Pas de sauvegarde d'état d'optimiseur ni de reprise ; la validation lève sur `NSError` anonyme.

#### A-13 — Conteneur monopolisé, ABBA (moyenne, VÉRIFIÉ code)
- Les trois entraînements s'exécutent entièrement dans `container.perform` (`Gemma4LoRATrain.swift:146, 353`,
  `MtpTrainCommand.swift:128`). Toute inférence sur le **même** conteneur attend la fin (famine de plusieurs heures,
  pas un gel). Une inférence sur un **autre** conteneur dans le même process, pendant ce temps, est exactement le
  scénario ABBA (`evalLock` × `compile`) : le forward d'entraînement passe par `geluApproximate` compilé
  (`Gemma4MLP.swift:27`, `Gemma4DecoderLayer.swift:140`) à l'intérieur de `valueAndGrad`.
- Déjà couvert par S-04/K-9 pour la protection ; à ajouter : le gate K-9 doit aussi être pris par l'entraînement
  **pour toute sa durée**, et le serveur (§ 2) ne doit exposer aucune route d'entraînement.

#### A-14 — `eval-mmlu` : puissance statistique et petits défauts (basse, VÉRIFIÉ)
- Le protocole publié utilise **100 questions** (10 sujets × 10, `BENCHMARKS.md:195`). Écart-type binomial à p ≈ 0,5 :
  5 pts, soit un IC95 d'environ ±10 pts. Le « −20 pts » du 12B 4 bits est significatif ; le « −7 pts » du 6 bits
  (question ouverte n° 1 du PLAN) **ne l'est pas**.
- Défauts d'outil : affichage `?` au-delà de 4 choix (`EvalCommand.swift:229-231`, MMLU-Pro en a 10) ; CoT sans arrêt
  sur l'id 50 (`<|tool_response>`, pourtant EOS dans `generation_config.json`) (`:281`) ; division par zéro si
  aucune question (`:244`) ; `--verbose` est une `@Option` Bool au lieu d'un `@Flag` (`:61-62`) ; un sujet absent
  du jeu dev passe silencieusement en 0-shot (`:174`).
- Le jeu est produit par `/tmp/benchwork/fetch_mmlu.py` selon `BENCHMARKS.md:203`, alors que le dépôt contient
  `BENCHMARKS_python_scripts/fetch_mmlu_5shot.py` : documenter le script versionné.

#### A-15 — Portes qualité non rejouables (moyenne, VÉRIFIÉ)
- `ls /tmp/toolsforge*` → rien : le jeu ToolsForge (1 301 train / 108 test) de la porte 97,2 % a disparu avec `/tmp`.
- `/tmp/mmlu_5shot.json` absent ; birdcall et LaTeX-OCR absents localement (`/Volumes/Lexar/datasets` n'en contient pas).
- Tant qu'ils ne sont pas recréés et archivés hors `/tmp` (avec empreinte SHA-256), aucune fiche du lot H ne peut
  prouver l'absence de régression qualité.

#### A-16 — `lora fuse` incomplet (basse ; VÉRIFIÉ fichiers, effet À VÉRIFIER)
- Copie `config.json`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `generation_config.json`
  avec `try?` (`LoRACommand.swift:480-490`). Le pack `gemma-4-e2b-it-bf16` contient aussi **`chat_template.jinja`**
  et **`processor_config.json`** : le modèle fusionné perd son gabarit de chat et la configuration du processeur.
- Les poids sont écrits dans la hiérarchie du module Swift (`parameters().flattened()`, `:474-478`) avec le
  `config.json` d'origine (bloc `quantization` compris pour un pack quantifié) : rechargement et compatibilité
  mlx-lm À VÉRIFIER. Le README affirme « Adapters trained in Swift work in Python mlx-lm and vice versa » : À VÉRIFIER.

#### A-17 — Commandes d'inférence avec adaptateur (basse, VÉRIFIÉ)
- `lora generate` et `lora bench-multimodal` réimplémentent le décodage (`LoRACommand.swift:546-590, 692-737`) :
  pas de top-p, pas d'`asyncEval`, arrêt sur ids codés en dur `1, 106, 50` (corrects : ce sont les EOS de
  `generation_config.json` — `<eos>`, `<turn|>`, `<|tool_response>` — mais non lus). Déjà S-21.
- Prompt système par défaut en français (`:513`, S-34) ; sortie par défaut `/tmp/birdcall-bench-results.jsonl`
  (`:621`) ; précision en `Double(correct)/Double(total)` sans garde (`:762`).

### 1.3 Performance et mémoire

#### A-18 — Logits pleine vocabulaire en fp32 sur toutes les positions (haute ; VÉRIFIÉ code, gain À MESURER)
- `model(inputs, cache: nil).asType(.float32)` (`TrainingLoop.swift:25`) matérialise `[B, L, 262 144]` en bf16 puis
  en fp32, puis `crossEntropy` et son gradient de même forme. Ordre de grandeur à B = 1 : L = 512 → 0,5 Gio par copie
  fp32 ; L = 2 048 → 2 Gio par copie, plusieurs copies vivantes au pic (logits, softmax, gradient).
- Avec `--mask-prompt`, seules les positions de réponse comptent. Pour le classifieur ToolsForge, c'est une petite
  fraction de la séquence : le head (262 144 × 1 536) et la CE sont calculés pour rien sur le reste.
- Levier : rassembler les états cachés des positions masquées **avant** `embedTokens.asLinear` (et le softcapping
  final, `Gemma4LanguageModel.swift:54`), puis CE fp32 sur ce seul sous-ensemble. Sans masque : CE par tranches.
  Numériquement identique (mêmes termes sommés). Porte : pic −≥ 15 % ou débit +≥ 5 %, loss@200 identique ±1e-3 (seedé).

#### A-19 — Multimodal : modèle entier converti en fp32 (haute ; VÉRIFIÉ code, cause À MESURER)
- `(model as! Module).apply { array.dtype.isFloatingPoint ? array.asType(.float32) : array }`
  (`Gemma4LoRATrain.swift:356-360`), justifié par des NaN en bf16 « > 300 tokens avec images ».
- Effet : ≈ 2× les poids en mémoire (E2B bf16 ≈ 10 Go → ≈ 20 Go), cohérent avec les **23-24 Go de pic** publiés
  (README, birdcall et LaTeX-OCR). Les couches LoRA et le calcul entier passent en fp32 : débit dégradé (T17).
- Hypothèse à tester en premier : la fuite fp32 P-02 (`embedScale` fp32 au chemin multimodal) fait cohabiter bf16 et
  fp32 dans le même graphe ; corriger K-3 puis retester bf16 + perte fp32 + paramètres LoRA fp32 seulement.

#### A-20 — Pas de politique mémoire (moyenne ; VÉRIFIÉ code, impact À MESURER)
- Aucune occurrence de `cacheLimit`, `memoryLimit` ou `clearCache` dans `LoRA/`, `Speculative/`, `LoRACommand`,
  `MtpTrainCommand`, `EvalCommand` (grep vide). Les longueurs changent à chaque pas : le cache de buffers MLX
  accumule des tailles différentes (piège 7). Le « GPU pic » affiché (`MLX.GPU.peakMemory`, API dépréciée, S-35)
  n'est pas le `phys_footprint`.

#### A-21 — Validation non plafonnée (moyenne ; VÉRIFIÉ code, coût À MESURER)
- Validation au pas 0 puis tous les `stepsPerEval` (CLI : 50) sur **tout** le jeu (`TrainingLoop.swift:213-217, 256-271`),
  avec un `.item()` par lot. mlx-lm plafonne à 25 lots (`val_batches`). Sur ToolsForge (1 300 pas), 27 validations
  complètes. Ajouter `--val-batches` (défaut 25, `-1` = tout) et `Memory.clearCache()` après chaque validation.

#### A-22 — Adaptateurs morts sur les couches KV-partagées (moyenne, VÉRIFIÉ)
- E2B : 35 couches dont 20 KV-partagées (`num_kv_shared_layers: 20`, `config.json` du pack bf16) → couches 15-34.
  LoRA s'applique aux `numLayers` **dernières** couches (`LoRAContainer.swift:107`) : 8 (défaut) ou 16 (configuration
  validée) → **toutes** partagées. Les clés par défaut sont toutes les `Linear` (`LoRAModel.swift:28-40`), donc
  `k_proj`/`v_proj` reçoivent un adaptateur dont le gradient est nul (la mémoire relève 32 `lora_b` nuls sur 144).
- Coût : calcul du forward LoRA, états Adam, fichier. Faible mais gratuit à retirer : `keys` explicites sans
  `self_attn.k_proj`/`v_proj` quand toutes les couches ciblées sont partagées (E4B : même logique, nombre de couches
  partagées À VÉRIFIER dans son `config.json`).

#### A-23 — Pas de gradient checkpointing (moyenne, VÉRIFIÉ)
- mlx-swift n'expose pas `checkpoint` en Swift : `grep "func checkpoint" Source/MLX` vide ; seul `mlx_checkpoint`
  existe dans `Cmlx/include/mlx/c/transforms.h:32`, et `Cmlx` n'est pas un produit du paquet (`Package.swift:309-315`).
- Sans lui, les activations de toutes les couches adaptées restent vivantes jusqu'au backward : c'est le premier
  bouton d'un profil `lean` et la condition d'un entraînement 26B/31B à longueur utile. Options : wrapper local via
  `import Cmlx` (importabilité transitive À VÉRIFIER), ou contribution amont (issue à suivre avec `track`).

#### A-24 — Multimodal : recalcul des encodeurs, médias résidents (moyenne ; VÉRIFIÉ code, À MESURER)
- À chaque pas, la tour vision (ou audio) et son projecteur tournent sur l'échantillon (`TrainingLoop.swift:367-392`)
  alors qu'ils sont gelés (`stopGradient`). Sur 1 époque, chaque média est encodé une fois de trop à la validation ;
  sur plusieurs époques, une fois par époque.
- À l'inverse, `preprocessMultimodalSamples` garde **tous** les `pixelValues` fp32 et features audio du jeu en
  mémoire pendant tout l'entraînement (`LoRACommand.swift:290-377`), sous forme de graphes paresseux (S-03).
- Levier : encoder une fois, stocker les embeddings (E2B : 280 × 1 536 bf16 ≈ 0,8 Mio par image) en mémoire ou sur
  disque selon le profil ; libérer les tours (T4) pendant la boucle.

#### A-25 — Drafter : forward cible dans la VJP (moyenne ; VÉRIFIÉ code, À MESURER)
- `drafterLoss` exécute le forward complet de la cible (35 couches + logits `[B, L, 262 144]` pour un `argMax`)
  **à l'intérieur** de la fermeture de `valueAndGrad` (`Gemma4DrafterTraining.swift:41-48`, appelée `:178-187`).
  `stopGradient` coupe la rétropropagation, mais le graphe de la cible est tracé et retenu avec celui du drafter.
- Levier : calculer `preNormHidden`, `argmax` et le KV partagé hors VJP, `eval`, puis les passer en entrées.
  Référence publiée : 2 000 pas, batch 4, 11 min (README, « en session »).

#### A-26 — `eval-mmlu` : préfixe recalculé (basse ; VÉRIFIÉ code, À MESURER)
- Chaque question re-préremplit le préfixe 5-shot de son sujet (`EvalCommand.swift:172-210`) et calcule les logits
  sur toutes les positions (P-01). Un snapshot de cache par sujet (T6, K-19) et la seule dernière position
  divisent le coût. Utile surtout si l'on passe à ≥ 1 000 questions (A-14).

### 1.4 Profils d'entraînement proposés

Même principe que les profils d'inférence (audit-performance § 3) : **chaque champ = un bouton existant ou créé
par la fiche indiquée**, réglages figés, mesurés. Type proposé : `Gemma4TrainingProfile` dans
`Sources/Gemma4Swift/Configuration/`, ids `lora-<bits>bit-<fast|lean>` où `bits` = largeur des poids de **base**
(16 = LoRA sur bf16 ; 8/4 = QLoRA sur pack quantifié), `named(_:family:)`, `all`, `applyGlobalPolicy()` ;
CLI `lora train --profile lora-16bit-fast` avec équivalents explicites.

| Champ | Bouton | État |
|---|---|---|
| `basePack` | `Gemma4Pipeline.Model` | existe |
| `rank`, `scale`, `numLayers`, `lr` | `TrainingConfig` | existe |
| `keys` (sans k/v sur couches partagées) | `LoRAConfiguration.loraParameters.keys` | existe côté amont, non exposé au CLI (A-22) |
| `batchSize` | `TrainingConfig.batchSize` | existe (batch > 1 : corriger A-10 d'abord) |
| `maxSeqLength` | — | **à créer** (A-06) |
| `responseOnlyHead` | — | **à créer** (A-18) |
| `gradientCheckpointing` | — | **à créer** (A-23) |
| `cacheLimitMB`, `clearCacheAfterEval` | `MLX.Memory` | API existe, câblage **à créer** (A-20) |
| `valBatches` | — | **à créer** (A-21) |
| `multimodalPrecision` | aujourd'hui fp32 forcé | **à rendre réglable** (A-19) |
| `mediaEmbeddingCache` | — | **à créer** (A-24) |
| `seed` | `MLXRandom.seed` + mélange seedé | **à compléter** (A-08) |

Valeurs communes : `fast` = batch le plus grand qui tient, `maxSeqLength 2048`, `cacheLimit 4096 Mo`, pas de
checkpointing, embeddings média en mémoire. `lean` = batch 1, `maxSeqLength 1024`, `cacheLimit min(1024, dispo/6)`,
checkpointing actif, `responseOnlyHead` actif, embeddings média sur disque, `clearCacheAfterEval`.
`responseOnlyHead` et `valBatches 25` valent pour les deux dès qu'ils ont passé leur porte.
**Tous les temps, pics et qualités sont À MESURER** ; la seule configuration mesurée en qualité est la ligne E2B
marquée « validé ».

| Famille | Id | Base | Rang / couches / clés | Batch · L max · ckpt | Pic / tok/s | Remarque |
|---|---|---|---|---|---|---|
| E2B | lora-16bit-fast | `e2bBf16` | r8 s20, 16 couches (15-34, toutes partagées), sans k/v | 1 (validé) puis 4 À MESURER · 2048 · non | À MESURER | **validé 97,2 %** à batch 1, lr 1e-4, 1 époque (mémoire) |
| E2B | lora-16bit-lean | `e2bBf16` | idem | 1 · 1024 · oui | À MESURER | cible Mac 16 Go |
| E2B | lora-8bit-lean | `e2b8bit` | idem | 1 · 1024 · oui | À MESURER | QLoRA 8 bits : porte qualité ≥ 95,3 % à confirmer |
| E2B | lora-4bit-lean | `e2b4bit` | r8, 8 couches (README : lr 1e-5) | 1 · 1024 · oui | À MESURER | README : « noisier gradients » ; qualité À MESURER |
| E4B | lora-16bit-fast / -lean | `e4bBf16` (19 Go) | r8 s20, 12 couches (README), clés selon `num_kv_shared_layers` À VÉRIFIER | fast 1-4 · 2048 / lean 1 · 1024 · oui | À MESURER | « Higher quality base » (README) |
| E4B | lora-8bit-lean | `e4b8bit` | idem | 1 · 1024 · oui | À MESURER | Mac 16 Go |
| 12B | lora-16bit-fast | `b12bBf16` (24 Go) | r8, 16 couches À VÉRIFIER ; famille à ajouter (A-06) | 1 · 2048 · non | À MESURER | multimodal impossible aujourd'hui (S-12/S-16) |
| 12B | lora-8bit-lean | `b12b8bit` | idem | 1 · 1024 · oui | À MESURER | 4 bits déconseillé comme base (MMLU −20 pts) |
| 26B-A4B | lora-16bit-fast | `a4bBf16` (52 Go) | attention seule : les experts `SwitchLinear` ne sont pas des `Linear` (clés par défaut À VÉRIFIER) | 1 · 2048 · oui | À MESURER | Mac 96 Go |
| 26B-A4B | lora-4bit-lean | `a4b4bit` (14 Go) | idem | 1 · 1024 · oui | À MESURER | 4 bits : −6 pts (issue #27) |
| 31B | lora-8bit-fast | `b31b8bit` (33 Go) | r8, 16 couches | 1 · 2048 · oui | À MESURER | bf16 (63 Go) hors de portée d'un entraînement confortable |
| 31B | lora-4bit-lean | `b31b4bit` (17 Go) | idem | 1 · 1024 · oui | À MESURER | après A-23 seulement |

Le drafter MTP a un seul profil utile : `mtp-e2b-bf16` (cible bf16, seqLen 256, batch 4, lr 1e-4, 2 000 pas —
valeurs du README) ; le rendre reproductible (A-08) avant de le mesurer.

### 1.5 Baseline d'entraînement (avant tout levier du lot H)

Conditions : celles du PLAN § 0 (Release, machine au repos, `pgrep -fl "gemma4|serve|train|lora|yue2|qwen38"`
vide — l'entraînement Gemma est lui-même le process qui a perturbé YuE2 et Qwen38), graine fixée (après A-08),
jeux de données archivés hors `/tmp` avec SHA-256 (A-15). Une ligne JSON par run :
`profile, family, pack, seed, dataset_sha, steps, tokens_processed_per_s, tokens_trained_per_s, step_ms_median,
step_ms_p90, peak_phys_footprint_mb, peak_mlx_active_mb, loss@50/@200/@final, val_loss, gate, revisions`.

| Point | Workload | Mesures | Porte qualité |
|---|---|---|---|
| TB1 | E2B bf16, ToolsForge 1 301/108, r8, 16 couches, lr 1e-4, batch 1, `--mask-prompt`, 1 300 pas | débit, pic, loss@200, loss finale (réf. mémoire ≈ 0,008) | précision test **≥ 95,3 % (≥ 103/108)** ; `lora_b` k/v nuls (32/144) |
| TB2 | TB1 tronqué à 200 pas, 2 runs (A/A) | dispersion débit, loss@200 | loss@200 identiques (bit à bit si A-08) — c'est le point de comparaison rapide des leviers |
| TB3 | E2B bf16 multimodal LaTeX-OCR 500 images, r16, lr 5e-5, 500 pas | pic (réf. 24 Go publié), débit, NaN | val loss ≤ 0,36 × 1,05 |
| TB4 | `mtp-train` E2B bf16, batch 4, 2 000 pas | durée (réf. 11 min), pic | acceptation ≥ 22,4 % et sortie greedy bit-exacte (README) |
| TB5 | `eval-mmlu` E2B 4 bits, 100 puis ≥ 1 000 questions | s/question, pic | exactitude inchangée sur les 100 questions actuelles |

Quand un levier est testé : TB2 A/B/B/A pour le débit et le pic, puis **TB1 complet** pour la porte qualité ; un
levier qui fait passer TB1 sous 103/108 est rejeté quel que soit le gain.

---

## Partie 2 — Serveur d'inférence

### 2.1 Ce que fait `qwen38 serve` (lu, pas seulement résumé)

- Hummingbird 2, un acteur `Qwen38InferenceServer`, routes `GET /healthz`, `GET /v1/models`, `GET /metrics` (JSON,
  pas Prometheus), `POST /v1/chat/completions` JSON et SSE (`Qwen38Server.swift:550-560`).
- Requête OpenAI : `messages` (contenu chaîne ou parties `text`/`image_url`), `tools`/`tool_choice`, rôle `tool`,
  `tool_calls` d'un tour assistant, dernier rôle `user|tool|assistant` (continuation d'un tour coupé), champs non
  standard à plat ou sous `extra` (`enable_thinking`, `conversation_id`, `mtp`…). Réponse avec `reasoning_content`,
  `tool_calls` retypés contre le schéma JSON, bloc `usage` avec `cached_tokens`.
- Diffusion : commentaire `: loading` toutes les 10 s tant qu'aucun événement n'est arrivé (`mergingHeartbeat`,
  `:360-390`) ; avec outils, `content` bufferisé et livré en un fragment pour ne jamais laisser fuir un
  `<tool_call>` brut.
- Catalogue de modèles limité à un répertoire racine explicite (`Qwen38ModelCatalog`) : un client ne peut pas
  transformer `model` en chemin arbitraire. Un seul modèle résident, changement de modèle = déchargement.
- File FIFO (`FIFORequestQueue`), cache de conversation LRU par client (budget 12 Go par défaut, `:475`), préfixe
  implicite par ids rendus quand il n'y a pas de `conversation_id`.
- Lot (`--batch-size N`, fenêtre 30 ms, regroupement par longueur de prompt proche, `--batch-max-prompt-tokens 256`) :
  ×2,16 agrégé à 8 clients sur prompts courts ; le préfill groupé dégrade le TTFT (×4,9 à ~1 200 jetons). Le verrou
  d'exécution du lot n'est relâché qu'à la fin **réelle** de la génération (correctif d'un crash `EXC_BAD_ACCESS`).
- `docs/reference/claude-code-wire-format.md` : Claude Code parle `/v1/messages` (Anthropic), toujours en
  streaming, avec **~31 700 jetons de schémas d'outils par requête** — irréaliste sans cache de préfixe.

**Défauts de Q à ne pas recopier** (VÉRIFIÉS par lecture) :
1. Adresse **`0.0.0.0` codée en dur** (`:564`), pas d'option d'hôte.
2. `/healthz` et `/metrics` non authentifiés (`:550, 552`) ; `/metrics` publie les sessions avec `lastToken`
   (48 derniers caractères générés) : fuite de contenu sur le LAN même avec `--api-key`.
3. `image_url` en `file://` accepté (`:1086`) : un client distant fait lire au serveur n'importe quelle image locale.
4. Chemin non groupé : `defer { Task { await queue.release() } }` (`:644`) relâche la FIFO au **retour du handler**,
   donc avant la fin du flux SSE ; la sérialisation réelle repose alors sur le runtime. Le chemin groupé a été
   corrigé par une porte de complétion, pas celui-ci.
5. Comparaison de la clé par `==` (`:1249`), pas à temps constant ; corps accepté jusqu'à 64 Mio (`:609`) ;
   images base64 écrites en fichiers temporaires (`:1087-1090`).

### 2.2 Ce que Gemma 4 permet (vérifié dans le gabarit et l'amont)

- **Outils : oui.** Le `chat_template.jinja` du pack E2B déclare `<|tool>…<tool|>` quand `tools` est fourni (l.179),
  rend les `tool_calls` d'un tour assistant, accepte les messages `role: "tool"` au format OpenAI Chat Completions
  (« forward-scan consecutive role:tool messages », l.263-275) et l'ancien `tool_responses`.
- Appels émis sous la forme `<|tool_call>call:nom{clé:<|"|>valeur<|"|>}<tool_call|>` ; l'amont a le parseur
  `ToolCallFormat.gemma4` (`MLXLMCommon/Tool/ToolCallFormat.swift:85-87, 124-126`), **inféré automatiquement** du
  `model_type` par `LLMModelFactory` (`LLMModelFactory.swift:592-593`, `ToolCallFormat.swift:195-196`) — donc actif
  sur nos conteneurs chargés par `Gemma4Registration.loadContainer`.
- `generation_config.json` : `eos_token_id = [1, 106, 50]`, soit `<eos>`, `<turn|>` et **`<|tool_response>`** : le
  modèle s'arrête de lui-même après un appel d'outil, en attendant la réponse.
- **Défaut existant** : `Gemma4Pipeline` jette les événements `.toolCall` (`Gemma4Pipeline.swift:518, 681` :
  `case .info, .toolCall: break`). Comme le parseur amont retire le texte de l'appel du flux, un appel d'outil
  émis via le pipeline disparaît sans trace (aujourd'hui peu probable : le pipeline passe `tools: nil`). VÉRIFIÉ code.
- Parties de contenu : le gabarit rend `type: image|audio|video` en `<|image|>`/`<|audio|>`/`<|video|>` (l.299-304,
  331-339). Raisonnement : `enable_thinking` (l.179-186), canal `<|channel>thought … <channel|>` ; le gabarit
  **retire la pensée** des tours modèle passés (`strip_thinking`, l.148, 319) — conséquence directe pour le cache
  de conversation (piège 13).
- Par famille : E2B/E4B/12B texte+image+audio(+vidéo) ; 26B/31B sans audio ; 12B multimodal indisponible via
  `Gemma4Pipeline` (S-16).

### 2.3 Conception proposée

**Principe : le moteur dans la bibliothèque, le HTTP à part.** Comme Q l'a fait avec `Qwen38Brain` (« conversation
OpenAI en entrée, événements typés en sortie », sans serveur HTTP), la partie qui touche MLX vit dans
`Gemma4Swift` sans nouvelle dépendance ; le serveur n'est qu'un adaptateur HTTP.

1. **`Gemma4ChatEngine`** (acteur, bibliothèque) — entrée : messages (`system|user|assistant|tool`, parties
   `text|image|audio|video`), `tools`, variables de gabarit (`enable_thinking`), paramètres, profil
   (`Gemma4ReferenceProfile`, K-20). Rendu par `chat_template.jinja` en **ids directs** +
   `strippingTemplateArtifacts` + expansion des médias (généralise `Gemma4Processor.multimodalChatIds`, un seul
   constructeur de prompt — S-20). Sortie : `.text`, `.reasoning`, `.toolCall(name, argumentsJSON)`,
   `.usage(prompt, cached, completion, ttft, tok/s)`, `.finish(stop|length|tool_calls)`. Annulable (K-1), `eval`
   des médias avant traversée (S-03/S-10), pris sous le gate K-9, 12B multimodal pris en charge (S-16).
   `Gemma4Pipeline` peut ensuite s'appuyer dessus (et relayer `.toolCall` au lieu de le jeter).
2. **Routes v1** : `POST /v1/chat/completions` (JSON et SSE, `: loading` pendant chargement/préfill long,
   `usage` dans le dernier fragment, `reasoning_content`, `tool_calls` livrés entiers comme Q) ; parties OpenAI
   `image_url` (**`data:` base64 uniquement**) et `input_audio` (`{data, format: wav|mp3}`) ; vidéo
   (`video_url` data:, convention vLLM) hors v1. `GET /v1/models` (catalogue sous `--models-dir`, validé par
   `config.json` `model_type` gemma4*, avec le profil appliqué), `GET /healthz` (minimal), `GET /metrics`
   (authentifié, compteurs seulement). Hors v1 : `/v1/messages` (Anthropic, pour Claude Code — n'a de sens qu'avec
   le cache de conversation vu les 31 k jetons d'outils), `/v1/completions`, embeddings.
3. **Profils** : `gemma4-server --reference 4bit-fast --models-dir …` ; famille déduite du `config.json` ; pack
   aux bits non conformes refusé (piège 12) ; `applyGlobalPolicy()` après chargement ; le profil plafonne
   `max_tokens`, la résolution image et la résidence des encodeurs. `--adapter <dir>` au démarrage uniquement
   (`loadAdapter` modifie le modèle : jamais pendant une génération).
4. **Réutilisation de conversation** (après K-19) : LRU par client, clé `conversation_id` ou préfixe implicite
   par ids rendus (comme Q P6.1), budget en Go. Spécificités Gemma : (a) **snapshot à la fin du prompt**, pas
   après la génération — le gabarit retire la pensée, donc les jetons générés ≠ le rendu du tour suivant
   (piège 13) ; (b) **extension stricte** seulement : les `RotatingKVCache` (fenêtre 512) ne reviennent pas en
   arrière (piège 29) ; (c) la clé inclut l'empreinte des médias ; (d) KV-partagé et positions multimodales
   restaurés avec le cache (piège 14). Le KV d'E2B est petit (1 tête KV) : snapshots peu coûteux, À MESURER.
5. **Sérialisation / ABBA** : un seul modèle résident, file FIFO **tenue jusqu'à la fin réelle du flux** (porte de
   complétion — défaut 4 de Q), gate K-9 partagé. Le process serveur n'expose **aucune** route d'entraînement ni
   de fusion ; un entraînement se lance dans un autre process (et le serveur peut refuser de démarrer si un
   `gemma4-cli lora|mtp-train` tourne, par `pgrep`, option). Profondeur de file bornée → HTTP 429.
6. **Annulation** : la déconnexion du client fait lever `writer.write` → fin de l'itération → `onTermination` du
   flux → `task.cancel()` → `Task.checkCancellation` par jeton (K-1) → libération de la file. Sans K-1, un client
   qui coupe laisse la génération courir jusqu'à `max_tokens` en bloquant toute la file : **K-1 est bloquant pour
   le serveur**.
7. **Sécurité** : écoute **`127.0.0.1` par défaut** ; `--host 0.0.0.0` n'est accepté qu'avec `--api-key`
   (refus de démarrer sinon) ; comparaison à temps constant ; `/metrics` authentifié et sans contenu généré ;
   limites par défaut : corps 32 Mio, 4 médias par requête, image ≤ 20 Mpx après décodage, audio ≤ 30 s,
   `max_tokens` ≤ plafond du profil, file ≤ 16 ; pas de `file://` ; décodage des médias en mémoire, sans fichier
   temporaire ; pas de CORS ; journal d'usage sur stderr sans contenu.
8. **Lot (batch)** : **hors v1**. mlx-swift-lm 3.31.4 n'a pas d'itérateur groupé (`grep` sur `Batch.*Iterator`
   ne trouve que `LoraTrain.swift`) ; KV-partagé, caches rotatifs et médias rendent le chantier L, et le gain Q
   (×2,16) ne vaut que pour des prompts courts. À ouvrir seulement si un banc 8 clients le justifie.

### 2.4 Contrainte de dépendance : options comparées

Contexte vérifié : `c9543739` a fait sortir swift-nio du graphe (EventSource 1.5.1 met NIO derrière un trait
désactivé) ; `Package.resolved` compte aujourd'hui **14 paquets**, sans NIO. Qwen38, qui embarque Hummingbird 2.27,
en résout **34**, dont la famille swift-nio (nio 2.103, nio-extras, nio-http2, nio-ssl, nio-transport-services),
swift-log/metrics/distributed-tracing/service-lifecycle/configuration et async-http-client (tous les paquets ne
viennent pas de Hummingbird, mais NIO si).

### 2.5 Expériences SwiftPM (Swift 6.4, Xcode 27.0 27A266a — scratchpad, paquets jouets)

| # | Montage | Résultat | Statut |
|---|---|---|---|
| E1 | Paquet `lib` (tools 6.0) : produit `Lib` sans dépendance + exécutable `server` qui dépend de `heavy` ; consommateur n'utilise que `Lib` | le consommateur **clone et résout `heavy`** (présent dans son `Package.resolved` et `.build/checkouts`), mais ne le **compile pas** (0 objet `Heavy.build`) | VÉRIFIÉ |
| E2 | Même montage, tools 6.1, dépendance conditionnée par un trait `Server` désactivé par défaut (SE-0450) | consommateur : **seul `lib` résolu** ; racine sans trait : rien ; `swift package resolve --traits Server` : `heavy` | VÉRIFIÉ |
| E2b | `xcodebuild -scheme server` à la racine | build sans le trait (sortie « trait off ») ; `xcodebuild -help` n'a **aucune** option de trait | VÉRIFIÉ |
| E2c | Paquet enveloppe qui déclare `.package(url: lib, traits: ["Server"])`, construit par `xcodebuild` | trait **actif** (« trait ON ») | VÉRIFIÉ |
| E3 | Paquet imbriqué `Server/Package.swift` (dépend de `.package(path: "..")` + `heavy`), non référencé par le manifeste racine | consommateur du tag : **seul `lib` résolu** ; `xcodebuild -scheme lib-server` dans `Server/` : BUILD SUCCEEDED | VÉRIFIÉ |

Conclusion sur la question posée : la « target-based dependency resolution » ne suffit pas — avec SwiftPM 6.4, les
dépendances d'un produit non utilisé sont **récupérées et résolues** par le consommateur (E1), seulement pas
compilées. Les consommateurs verraient donc swift-nio réapparaître dans leur `Package.resolved` (et dans leurs
alertes de sécurité), et ses bornes de version entreraient dans leur résolution.

| Option | Consommateurs (5, bibliothèque seule) | Build du serveur (`xcodebuild` obligatoire, Metal) | Coûts / risques | Verdict |
|---|---|---|---|---|
| (a) cible serveur dans `Package.swift` racine | **NIO réapparaît** dans leur graphe résolu (E1) ; conflits de bornes possibles | direct | annule le bénéfice de `c9543739` pour les 5 consommateurs | **rejetée** |
| (a') cible racine + trait `Server` (tools 6.1) | graphe inchangé (E2) | impossible à la racine avec `xcodebuild` (E2b) → exige un paquet enveloppe (E2c) | tools-version 6.1 imposée à tous les consommateurs ; `#if Server` dans le code ; comportement des consommateurs en **projet Xcode** (Fluxforge) À VÉRIFIER | possible mais plus complexe que (b) pour le même résultat |
| (b) dépôt séparé `gemma4-server` | graphe inchangé | direct | versions croisées (tag bibliothèque ↔ serveur), CI double ; oblige une API publique propre (bien) | **bonne** |
| (b') paquet imbriqué `Server/` dans ce dépôt | graphe inchangé (E3) | direct (E3) | sources clonées par les consommateurs (quelques Ko) ; même tag pour les deux ; `Package.resolved` propre au serveur, scanné séparément | **recommandée** |
| (c) serveur maison Network.framework (`NWListener`) dans ce paquet | graphe inchangé | direct | écrire et sécuriser un parseur HTTP/1.1 (en-têtes bornés, keep-alive, chunked, SSE) : exactement la classe de CVE qu'on vient de fuir ; M-L d'effort et une surface d'attaque maison | repli seulement si « aucune dépendance nouvelle, nulle part » devient une règle |

**Recommandation : (b')**, un paquet `Server/` (produit exécutable `gemma4-server`, cible bibliothèque
`Gemma4Server` testable) qui dépend de `.package(path: "..")`, de Hummingbird 2 et **explicitement de
`swift-nio` `from: "2.100.0"`** (plancher des deux CVE, pour qu'aucune résolution ne redescende). Arguments :
(1) les 5 consommateurs ne voient rien (E3) ; (2) aucun changement de tools-version ni de manifeste racine ;
(3) serveur et bibliothèque évoluent dans le même commit, testés par la même CI, sans jeu de tags croisés ;
(4) le `Package.resolved` du serveur est scanné à part : une alerte NIO future concerne le serveur, pas la
bibliothèque ; (5) si le rythme de publication diverge, l'extraction en dépôt séparé (b) est mécanique. Garde-fou
à ajouter en CI : un paquet témoin qui dépend du tag et vérifie que son `Package.resolved` ne contient ni
`hummingbird` ni `swift-nio`. Le moteur (§ 2.3, point 1) reste dans la bibliothèque, sans dépendance.

---

## 3. Fiches proposées (format du PLAN)

Ordre : H (correction → instrument → baseline → leviers → profils) peut démarrer dès maintenant, indépendamment ;
G dépend de K-1, K-9 et K-20 (lot A et E du PLAN) ; K-40 dépend de K-19.

### Lot H — Fonctions annexes (entraînement, évaluation)

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

### Lot G — Serveur d'inférence

| Fiche | Objet | Source | Porte | Effort |
|---|---|---|---|---|
| K-36 | `Gemma4ChatEngine` dans la bibliothèque (aucune dépendance) : messages OpenAI (rôles `tool` compris, parties image/audio), `tools`, `enable_thinking`, profil, événements typés (texte, pensée, appel d'outil, usage, fin), annulable, sous gate K-9 ; `Gemma4Pipeline` relaie `.toolCall` | § 2.2, § 2.3-1 | tests : ids rendus == rendu HF token à token pour 4 cas (texte, image, outils, tour `tool`) ; appel d'outil parsé sur fixture ; annulation en < 1 pas | L |
| K-37 | Paquet imbriqué `Server/` (Hummingbird 2, `swift-nio from: 2.100.0`) : routes v1, SSE + `: loading`, erreurs au format OpenAI, `--reference`, `--models-dir`, `--adapter` ; paquet témoin en CI | § 2.3-2/3, § 2.5 | le témoin résout exactement les 14 paquets actuels (ni hummingbird ni swift-nio) ; client `openai` Python : chat JSON, SSE, image, audio, outils OK | M |
| K-38 | Sécurité : `127.0.0.1` par défaut, hôte non-loopback ⇒ `--api-key` obligatoire, comparaison à temps constant, `/metrics` authentifié sans contenu, limites (32 Mio, 4 médias, 20 Mpx, 30 s d'audio, `max_tokens` du profil, file ≤ 16 → 429), pas de `file://`, pas de fichiers temporaires | § 2.3-7, défauts Q 1-3, 5 | tests d'intégration : 401, 413, 400 (`file://`, média trop grand), 429, refus de démarrer sur `0.0.0.0` sans clé | S-M |
| K-39 | Annulation et sérialisation de bout en bout : déconnexion → arrêt en < 1 pas ; file tenue jusqu'à la fin **réelle** du flux ; aucune route d'entraînement | § 2.3-5/6, défaut Q 4, A-13 | test : client coupé à 5 jetons, la requête suivante a un TTFT ≤ TTFT à vide + 1 pas ; 2 clients concurrents → sorties identiques au séquentiel | S (après K-1, K-9) |
| K-40 | Réutilisation de conversation (LRU par client, `conversation_id` ou préfixe implicite, snapshot en fin de prompt, extension stricte, médias dans la clé, budget en Go) | § 2.3-4, P-09, pièges 13/14/29 | boucle d'agent 4 tours : > 80 % des jetons du tour 2 servis du cache, réponses identiques au re-préfill, TTFT tour 2 −≥ 50 % | L (après K-19) |
| K-41 | *(option)* Lot multi-clients | § 2.3-8 | ≥ ×1,5 agrégé à 8 clients, TTFT p90 ≤ +20 % sur prompts ≤ 256 jetons ; sinon abandon | L |
| K-42 | *(option)* `/v1/messages` (Anthropic) pour Claude Code | wire format Q | session Claude Code de 10 tours sans erreur, cache de préfixe actif (31 k jetons d'outils) | M (après K-40) |

### Décisions à prendre (ASK)
1. Serveur : paquet imbriqué `Server/` (recommandé) ou dépôt séparé `gemma4-server` ?
2. Périmètre v1 : `/v1/chat/completions` + `models`/`healthz`/`metrics` seulement, lot et `/v1/messages` plus tard ?
3. Profils d'entraînement : matrice E2B/E4B seulement, ou 12B/26B/31B (téléchargements de 24-63 Go) ?
4. Recréer le jeu ToolsForge (porte 97,2 %) : la source existe-t-elle encore côté ToolsForge ?

---

## 4. Résumé

1. Complément en lecture seule : entraînement LoRA/multimodal/drafter, évaluation, serveur ; 26 constats A-01…A-26.
2. Mémoire périmée : l'index dit « 14 % vs 87 % » ; la réalité documentée est 97,2 % Swift contre 95,3 % Python.
3. A-01 : `gradClipMaxNorm` est affiché (0,3 par défaut en full) mais n'est jamais appliqué.
4. A-02 : un run interrompu laisse un adaptateur sans `adapter_config.json`, écrasé en place, sans reprise possible.
5. A-03/A-05 : l'aller-retour decode→encode (bug 4) revient en multimodal et dans `lora eval` ; `mtp-train` refait les bugs 1 et 3.
6. A-04 : entraînement et inférence multimodaux diffèrent d'un jeton (`\n` après le média).
7. A-08/A-09 : mélange non seedé et tok/s qui ne compte que les jetons entraînés : les chiffres actuels ne se comparent pas.
8. A-15 : les jeux des portes qualité (ToolsForge, MMLU) ont disparu de `/tmp` : rien n'est rejouable aujourd'hui.
9. A-14 : MMLU sur 100 questions = IC95 ±10 pts ; le « −7 pts » du 12B 6 bits n'est pas significatif.
10. A-18 : logits 262 k fp32 sur toutes les positions à chaque pas, même quand seule la réponse compte.
11. A-19 : le multimodal convertit tout le modèle en fp32, ce qui explique l'essentiel des 23-24 Go publiés.
12. A-22 : sur E2B, les 16 couches adaptées sont toutes KV-partagées : adaptateurs k/v morts.
13. A-23 : pas de gradient checkpointing (mlx-swift ne l'expose pas en Swift) : bloquant pour `lean` et 26B/31B.
14. A-11 : `--drafter-path <dossier>` mélange au hasard `drafter` et `drafter.best`.
15. Profils `lora-<bits>bit-<fast|lean>` par famille proposés (§ 1.4) ; seule la ligne E2B bf16 batch 1 est validée en qualité.
16. Baseline TB1-TB5 ; porte : ToolsForge ≥ 103/108, TB2 A/A pour les leviers.
17. Gemma 4 gère les outils : gabarit (`<|tool>`, rôle `tool`, `tool_calls`), parseur amont `.gemma4` actif, arrêt sur `<|tool_response>`.
18. `Gemma4Pipeline` jette aujourd'hui les événements `.toolCall` (l. 518, 681).
19. Serveur : moteur `Gemma4ChatEngine` sans dépendance dans la bibliothèque, HTTP à part.
20. À ne pas recopier de Q : `0.0.0.0` codé en dur, `/metrics` non authentifié qui publie du texte généré, `file://`, file relâchée avant la fin du flux.
21. K-1 (annulation) et K-9 (gate) sont bloquants pour le serveur ; le process serveur n'entraîne jamais.
22. Expérience SwiftPM 6.4 : un produit non utilisé fait quand même **résoudre et cloner** ses dépendances chez le consommateur.
23. Les traits (SE-0450) élaguent bien, mais `xcodebuild` ne sait pas activer un trait racine : il faut de toute façon un paquet enveloppe.
24. Recommandation : paquet imbriqué `Server/` (Hummingbird + swift-nio ≥ 2.100.0), graphe des 5 consommateurs inchangé (vérifié).
25. Fiches : lot H K-25…K-35, lot G K-36…K-42 ; 4 décisions à prendre.
