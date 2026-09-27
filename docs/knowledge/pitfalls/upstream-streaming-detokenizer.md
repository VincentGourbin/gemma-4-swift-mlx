# Piège — `NaiveStreamingDetokenizer` perd des scalaires

**Symptôme** (2026-09-27) : « Recopie 🇫🇷 👩‍👩‍👧 𠀋 𝔘𝔫𝔦 » rendu « 🇫 👩 𠀋 𝔘𝔫𝔦 » par `chatStreamMultimodal`.
**Cause** : mlx-swift-lm (`Tokenizer.swift:96-97` @ `bd4b7434`) calcule le nouveau texte par
`String.count`, qui compte des graphèmes : un scalaire qui fusionne avec le graphème précédent
(2e indicateur régional, séquence ZWJ, accent combinant) ne fait pas grandir le compte et est
perdu. Décoder token par token perd, lui, les caractères en repli octet par octet.
**Correctif** : `Gemma4StreamingDetokenizer` (différence par scalaires), `90cef4b5` (MTP, CLI) et
`44858e01` (chemins TokenIterator du pipeline). Les chemins `ChatSession` gardent le défaut.
**Règle** : une différence de texte incrémentale se fait sur les scalaires Unicode, jamais sur
`String.count`. Test amont gardé en `withKnownIssue` : il signalera la correction.
