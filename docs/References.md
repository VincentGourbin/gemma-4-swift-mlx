# Profils de référence

`gemma4-cli references` les liste, `gemma4-cli bench --reference <famille/id>` en mesure un,
`Gemma4ReferenceProfile.all` les expose à une app et `Gemma4Pipeline.load(profile:)` en charge un.
Chaque profil fige **tous** les réglages qui comptent : poids, cache KV, tranche de préfill,
limites mémoire, modalités. Même profil + même graine = même sortie, temps comparable sur
matériel comparable.

Source : `Sources/Gemma4Swift/Configuration/Gemma4ReferenceProfile.swift`. Protocole de mesure :
[Benchmarks.md](Benchmarks.md). Décision : [knowledge/decisions/reference-profiles.md](knowledge/decisions/reference-profiles.md).

## État : valeurs initiales, non mesurées

La matrice ci-dessous vient de l'audit du 2026-09-27 et des mesures antérieures de
`BENCHMARKS.md`. Les colonnes temps et pic se rempliront avec `gemma4-cli bench`, machine au
repos ; un profil garde son identifiant si une valeur change après mesure.

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
| `prefillStepSize` | 512 | 256 | Taille des tranches de préfill ; le head ne calcule que le dernier jeton. |
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
