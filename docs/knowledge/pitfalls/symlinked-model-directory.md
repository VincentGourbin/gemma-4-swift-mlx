# Piège — dossier de modèle derrière un lien symbolique

**Symptôme** (2026-09-27) : « Key language_model.model.norm.weight not found in … RMSNorm » en
chargeant `~/Library/Caches/models/mlx-community/gemma-4-e2b-it-bf16`, lien vers le Lexar.
**Cause** : le chargeur de mlx-swift-lm énumère les poids avec `FileManager.enumerator(at:)`,
qui ne descend pas dans une racine qui est un lien : aucun poids n'est lu.
**Correctif** : un vrai dossier avec un lien par fichier (les shards en lien sont suivis, même
pendants pour le calcul de complétude : `41d432f0`).
**Règle** : lier les fichiers, jamais le dossier racine d'un modèle.
