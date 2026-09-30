# Décision — profils de référence 4/8/16 bits × fast/lean

**Contexte** (2026-09-27, demande de Vincent) : faire du standard de profils de YuE2 et Qwen38
le standard de ce dépôt, avec les poids recommandés par quantisation, pour toutes les familles
(E2B, E4B, 12B, 26B-A4B, 31B).

**Décision** : `Gemma4ReferenceProfile.all` (30 profils), exposé par `gemma4-cli references`,
`bench --reference` et `Gemma4Pipeline.load(profile:)`. API strictement additive : sans profil,
le pipeline se comporte comme avant (Fluxforge suit `main`). Chaque champ est un réglage
existant : `kvBits` (utilisable depuis K-5), `prefillStepSize` (effectif depuis K-13), limites
mémoire MLX, vidage du cache, modalités. DiffusionGemma est hors standard.

**Choix motivés** : KV 8 bits en `lean` seulement pour 26B-A4B et 31B (plusieurs têtes KV ; une
seule tête sur E2B/E4B/12B, gain négligeable). `16bit-lean` garde les caches Mac (YuE2 : limites
serrées + bf16 = +73 %). 12B : 8 bits conseillé ; 4 bits marqué dégradé (MMLU 37 % contre 57 %).

**Ouvert** : table mesurée (temps, pic) avec `gemma4-cli bench` ; profil 6 bits pour le 12B
(MMLU sur ≥ 1 000 questions) ; E2B 6 bits en entraînement (Fluxforge) ; résidence des encodeurs
(K-16) et MTP (`withMTP`, après K-4b) comme champs futurs.
