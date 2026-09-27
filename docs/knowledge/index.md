---
okf_version: "0.1"
---

# Base de connaissances — gemma-4-swift-mlx

Journal d'ingénierie (bundle OKF, même format que YuE2 et Qwen38). `log.md` porte
l'historique horodaté ; les sous-répertoires accumulent les conclusions durables. Rapports
d'audit et plan : `docs/audit/2026-09-27/`.

## Décisions
- [Profils de référence](decisions/reference-profiles.md) — 5 familles × 4/8/16 bits × fast/lean, valeurs initiales non mesurées ; 12B : 8 bits conseillé, 6 bits en attente d'un MMLU significatif.

## Pièges
- [Constante fp32 qui promeut le graphe](pitfalls/fp32-scalar-promotes-graph.md) — `x * MLXArray(s, dtype: .float32)` fait passer préfill, cache KV et décodage en fp32.
- [Détokeniseur de streaming amont](pitfalls/upstream-streaming-detokenizer.md) — `NaiveStreamingDetokenizer` diffère par graphèmes et perd des scalaires.
- [Dossier de modèle derrière un lien](pitfalls/symlinked-model-directory.md) — le chargeur ne parcourt pas une racine qui est un lien symbolique.
