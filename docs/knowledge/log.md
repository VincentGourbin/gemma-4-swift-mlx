# Journal

- 2026-09-27 — **Audit** (skill `mlx-swift-audit`) : 37 constats de stabilité, 16 de performance,
  26 sur les fonctions annexes et le serveur ; plan K-1 à K-42 (`docs/audit/2026-09-27/PLAN.md`).
- 2026-09-27 — **Lot A** (K-1 à K-9) sur `fix/lot-a-stabilite` : streams annulables, téléchargement
  multi-shards, fp32 multimodal, `kvBits` + KV partagé, encodeur vision, fabrique privée, FFT et
  détokenisation, garde entraînement/inférence, MTP au-delà de la fenêtre. Aucun chiffre de
  vitesse : `qwen38` occupait le GPU. Détail et SHA : journal du PLAN.
- 2026-09-27 — **K-8b** : contournement du détokeniseur amont sur les chemins TokenIterator.
- 2026-09-27 — **Lot B** : 990 sorties brutes OCR hors dépôt, chemins personnels retirés des
  journaux publics, warnings 37 → 2, CI de compilation, CHANGELOG, DiffusionGemma hors de
  `recommended()`, macOS 15 dans la doc.
- 2026-09-27 — **K-11** : `gemma4-cli bench`. Constat : « poids × tok/s » dépasse le plafond de
  la puce sur E2B (404 Go/s) — tables d'embeddings par couche lues partiellement ; indicateur,
  pas une mesure.
- 2026-09-27 — **K-13** : préfill par tranches, head sur le dernier jeton. Parité fp32 exacte ;
  sur E2B réel (~1 000 jetons, gabarit de chat), même premier jeton, écart des logits 2,1 %
  (6 bits) / 0,6 % (bf16), marge top-1/top-2 de 0,5 logit : une sortie greedy longue peut
  bifurquer sur une quasi-égalité, sans bug.
- 2026-09-27 — **K-20 / K-14** : `Gemma4ReferenceProfile` (30 profils), `references`,
  `bench --reference`, `Gemma4Pipeline.load(profile:)` et `apply(profile:)` ; politique mémoire
  par profil. Valeurs non mesurées.
- 2026-09-27 — Suite n-gramme d'intégration : 2 échecs identiques sur `main` et sur la branche
  avec E2B 6 bits ; la suite a été écrite pour E4B 4 bits. Pas de régression.
