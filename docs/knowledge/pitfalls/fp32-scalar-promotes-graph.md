# Piège — une constante fp32 promeut tout le graphe

**Symptôme** (2026-09-27) : avec une image, le cache KV d'E2B était en fp32 et le décodage suivait.
**Cause** : `inputsEmbeds * MLXArray(embedScale, dtype: .float32)` sur les trois chemins
multimodaux (le texte utilisait déjà le dtype d'entrée) ; même défaut dans le masquage des
positions de l'encodeur vision (`MLXArray(Float(0.0))`).
**Correctif** : `d96bc8e8` (K-3), `134447b2` (K-6). Tests `MultimodalDtypeTests`,
`VisionPatchEmbedderDtypeTests` (échouent sans le correctif).
**Règle** : une constante MLX prend le dtype du tenseur qu'elle modifie.
