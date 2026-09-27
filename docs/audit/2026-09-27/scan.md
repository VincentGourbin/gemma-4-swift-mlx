# Scan mlx-swift-audit — `/Users/vincent/Developpements/gemma-4-swift-mlx`

Révision : `c9543739 chore(deps): met a jour les dependances, swift-nio sort du graphe (#52) (#53)` · 1266 fichiers suivis · 174 fichiers Swift

## 1. Volume Swift par module

| Module | Lignes |
|---|---|
| `Sources/Gemma4Swift` | 14935 |
| `Sources/Gemma4BenchUI` | 7382 |
| `Tests/Gemma4SwiftTests` | 6168 |
| `Sources/Gemma4CLI` | 4905 |
| `Package.swift` | 69 |

## 2. Dépendances (Package.swift / Package.resolved)

- `.package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.31.4")),`
- `.package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),`
- `.package(url: "https://github.com/ml-explore/mlx-swift-lm", .upToNextMinor(from: "3.31.4")),`
- `.package(url: "https://github.com/apple/swift-argument-parser", from: "1.2.0"),`
- `.package(url: "https://github.com/VincentGourbin/swift-mlx-profiler", from: "1.4.0"),`
- résolu : `mlx-swift` → `0bb916c67f4b9e5c682cbe02a42c701c93ab5021`
- résolu : `mlx-swift-lm` → `bd4b7434e6bdb588c7ef55706ff8904cb7fd4c57`
- résolu : `swift-mlx-profiler` → `689df79754fe69e32360ecd54c55d9e0a15d4017`

## 3. Hygiène du dépôt

Plus gros fichiers suivis :

| Taille | Fichier |
|---|---|
| 1130 Ko | `docs/examples/vision-image-description/UI.png` |
| 1130 Ko | `UI.png` |
| 1042 Ko | `input_sample.jpg` |
| 1042 Ko | `docs/examples/vision-image-description/input_sample.jpg` |
| 437 Ko | `docs/examples/ocr-bench/summary_1000.csv` |
| 421 Ko | `docs/examples/ui-grounding-bench/wikipedia-search-bar-demo.gif` |
| 250 Ko | `docs/examples/ocr-bench/meta_1000.json` |
| 188 Ko | `docs/examples/function-calling-bench/cases_100.json` |
| 78 Ko | `docs/examples/ocr-bench/meta_300.json` |
| 53 Ko | `docs/examples/turboquant_paper.txt` |
| 48 Ko | `docs/examples/ui-grounding-bench/diff_summary.json` |
| 48 Ko | `Sources/Gemma4CLI/Gemma4CLI.swift` |
| 47 Ko | `Sources/Gemma4BenchUI/IOSSim/IOSAgentStepViewModel.swift` |
| 42 Ko | `docs/examples/ui-grounding-bench/a4b_summary.json` |
| 41 Ko | `docs/examples/ui-grounding-bench/e4b_summary.json` |

Fichiers suspects suivis (artefacts, poids, traces) :

- aucun

Fichiers non-code à la racine :

- `UI.png`
- `input_sample.jpg`

## 4. Indices perf / mémoire

| Motif | Occ. (Sources / Tests) | Sens | Catalogue |
|---|---|---|---|
| `cache-limit` | 2 / 0 | Pose d'une limite de cache MLX | T1/T2, piège 7 |
| `memory-limit` | 0 / 0 | Seuil GC MLX (pas un plafond dur) | T2, piège 17 |
| `clear-cache` | 14 / 0 | Libération du cache MLX | T3 |
| `async-eval` | 4 / 0 | Pipelining du décodage | T15, piège 6 |
| `item-call` | 62 / 49 | Synchronisation GPU→CPU (coûteuse dans une boucle) | T15 |
| `as-array` | 16 / 6 | Copie GPU→CPU | T15 |
| `concat` | 43 / 1 | Concat (cache KV recopié à chaque pas ?) | T11, piège 3 |
| `fp32-cast` | 34 / 3 | Passage en fp32 (fuite de dtype ?) | T17 |
| `compile` | 1 / 0 | compile() MLX (utile ? deadlock ABBA avec vjp) | T18, R1-R3, piège 20 |
| `dequantize` | 2 / 0 | Dé-quantification (jamais le head entier) | T12, T14 |
| `quantized-kv` | 40 / 0 | KV cache quantifié | T10 |
| `prefill-step` | 2 / 0 | Tranche de préfill | T9 |
| `kv-trim` | 3 / 0 | Réutilisation de préfixe KV | T6 |
| `eval-params` | 0 / 0 | eval global des paramètres (pic au chargement) | piège 4 |
| `resize-image` | 9 / 1 | Redimensionnement d'image implicite ? | T8, piège 15 |

<details><summary><code>cache-limit</code> — 2 hors tests</summary>

- `Sources/Gemma4CLI/DiffusionGemmaCommand.swift:152` — `MLX.Memory.cacheLimit = bytes`
- `Sources/Gemma4CLI/ProfileDiffusionCommand.swift:151` — `MLX.Memory.cacheLimit = bytes`

</details>

<details><summary><code>clear-cache</code> — 14 hors tests</summary>

- `Sources/Gemma4BenchUI/Agent/WebAgentLoop.swift:115` — `MLX.GPU.clearCache()`
- `Sources/Gemma4BenchUI/ModelRegistry.swift:98` — `MLX.GPU.clearCache()`
- `Sources/Gemma4BenchUI/ModelRegistry.swift:126` — `MLX.GPU.clearCache()`
- `Sources/Gemma4BenchUI/ModelRegistry.swift:175` — `MLX.GPU.clearCache()`
- `Sources/Gemma4BenchUI/ModelRegistry.swift:187` — `MLX.GPU.clearCache()`
- `Sources/Gemma4CLI/ProfileCommand.swift:478` — `MLX.GPU.clearCache()`
- `Sources/Gemma4CLI/ProfileDiffusionCommand.swift:88` — `@Flag(name: .customLong("clear-cache-on-eval"), help: "Avec --eval-every-n-layers : appelle MLX.Memory.clearCache() apres chaque eval.")`
- `Sources/Gemma4CLI/ProfileDiffusionCommand.swift:388` — `MLX.Memory.clearCache()`
- `Sources/Gemma4Swift/Diffusion/Pipeline/DiffusionGemmaPipeline.swift:221` — `MLX.Memory.clearCache()`
- `Sources/Gemma4Swift/Diffusion/Pipeline/DiffusionOnTheFlyQuantization.swift:107` — `MLX.Memory.clearCache()`
- `Sources/Gemma4Swift/Diffusion/Pipeline/DiffusionOnTheFlyQuantization.swift:258` — `MLX.Memory.clearCache()`
- `Sources/Gemma4Swift/Diffusion/TextModel/DiffusionGemmaDecoderTextModel.swift:162` — `MLX.Memory.clearCache()`
- … 2 de plus

</details>

<details><summary><code>async-eval</code> — 4 hors tests</summary>

- `Sources/Gemma4CLI/ProfileCommand.swift:161` — `asyncEval(firstToken)`
- `Sources/Gemma4CLI/ProfileCommand.swift:185` — `asyncEval(token)`
- `Sources/Gemma4CLI/ProfileCommand.swift:407` — `asyncEval(firstToken)`
- `Sources/Gemma4CLI/ProfileCommand.swift:418` — `asyncEval(token)`

</details>

<details><summary><code>item-call</code> — 62 hors tests</summary>

- `Sources/Gemma4CLI/EvalCommand.swift:277` — `var nextTok = argMax(prefillOut[0..., prefillOut.dim(1) - 1, 0...], axis: -1).item(Int32.self)`
- `Sources/Gemma4CLI/EvalCommand.swift:284` — `nextTok = argMax(out[0..., 0, 0...], axis: -1).item(Int32.self)`
- `Sources/Gemma4CLI/Gemma4CLI.swift:857` — `var nextToken = argMax(prefillOutput[0..., prefillOutput.dim(1) - 1, 0...], axis: -1).item(Int32.self)`
- `Sources/Gemma4CLI/Gemma4CLI.swift:882` — `nextToken = argMax(output[0..., 0, 0...], axis: -1).item(Int32.self)`
- `Sources/Gemma4CLI/Gemma4CLI.swift:886` — `nextToken = MLXRandom.categorical(log(probs)).item(Int32.self)`
- `Sources/Gemma4CLI/Gemma4CLI.swift:1106` — `var nextToken = argMax(prefillOutput[0..., prefillOutput.dim(1) - 1, 0...], axis: -1).item(Int32.self)`
- `Sources/Gemma4CLI/Gemma4CLI.swift:1126` — `nextToken = argMax(output[0..., 0, 0...], axis: -1).item(Int32.self)`
- `Sources/Gemma4CLI/Gemma4CLI.swift:1130` — `nextToken = MLXRandom.categorical(log(probs)).item(Int32.self)`
- `Sources/Gemma4CLI/LoRACommand.swift:570` — `var nextToken = argMax(prefillOutput[0..., prefillOutput.dim(1) - 1, 0...], axis: -1).item(Int32.self)`
- `Sources/Gemma4CLI/LoRACommand.swift:580` — `nextToken = argMax(output[0..., 0, 0...], axis: -1).item(Int32.self)`
- `Sources/Gemma4CLI/LoRACommand.swift:584` — `nextToken = MLXRandom.categorical(log(probs)).item(Int32.self)`
- `Sources/Gemma4CLI/LoRACommand.swift:717` — `var nextToken = argMax(prefillOutput[0..., prefillOutput.dim(1) - 1, 0...], axis: -1).item(Int32.self)`
- … 50 de plus

</details>

<details><summary><code>as-array</code> — 16 hors tests</summary>

- `Sources/Gemma4BenchUI/Agent/AgentStepViewModel.swift:329` — `let outIds = r.generatedIds.asArray(Int32.self).map { Int($0) }`
- `Sources/Gemma4BenchUI/Agent/WebAgentLoop.swift:296` — `let outIds = result.generatedIds.asArray(Int32.self).map { Int($0) }`
- `Sources/Gemma4BenchUI/BenchViewModel.swift:486` — `let tokens = canvas.asArray(Int32.self).map { Int($0) }`
- `Sources/Gemma4BenchUI/BenchViewModel.swift:498` — `let tokens = argmax.asArray(Int32.self).map { Int($0) }`
- `Sources/Gemma4BenchUI/IOSSim/IOSAgentStepViewModel.swift:308` — `let outIds = r.generatedIds.asArray(Int32.self).map { Int($0) }`
- `Sources/Gemma4BenchUI/VQAGame/VQAGameViewModel.swift:150` — `let outIds = r.generatedIds.asArray(Int32.self).map { Int($0) }`
- `Sources/Gemma4CLI/DiffusionGemmaCommand.swift:293` — `let tokens = canvas.asArray(Int32.self).map { Int($0) }`
- `Sources/Gemma4CLI/DiffusionGemmaCommand.swift:299` — `let tokens = argmaxCanvas.asArray(Int32.self).map { Int($0) }`
- `Sources/Gemma4CLI/DiffusionGemmaCommand.swift:321` — `let tokens = canvas.asArray(Int32.self).map { Int($0) }`
- `Sources/Gemma4CLI/DiffusionGemmaCommand.swift:331` — `let allTokens = result.generatedIds.asArray(Int32.self).map { Int($0) }`
- `Sources/Gemma4CLI/EvalCommand.swift:357` — `let candidateLogits = lastLogits.take(candidateIdx, axis: 0).asArray(Float.self)`
- `Sources/Gemma4CLI/ProfileDiffusionCommand.swift:384` — `let arr = argmaxCanvas.asArray(Int32.self)`
- … 4 de plus

</details>

<details><summary><code>concat</code> — 43 hors tests</summary>

- `Sources/Gemma4CLI/Gemma4CLI.swift:707` — `pixelValues = concatenated(padded, axis: 0)`
- `Sources/Gemma4CLI/ProfileDiffusionCommand.swift:379` — `fullIds = concatenated([fullIds, argmaxCanvas], axis: -1)`
- `Sources/Gemma4Swift/AudioEncoder/AudioRelPositionalEncoding.swift:45` — `return concatenated([sin(scaledTime), cos(scaledTime)], axis: -1).asType(hiddenStates.dtype)`
- `Sources/Gemma4Swift/Diffusion/Pipeline/DiffusionGemmaPipeline.swift:210` — `fullIds = concatenated([fullIds, argmaxCanvas], axis: -1)`
- `Sources/Gemma4Swift/Diffusion/TextModel/DiffusionAttentionMask.swift:102` — `let slidingMask = concatenated([slidingSlice, canvasTrue], axis: -1)`
- `Sources/Gemma4Swift/Diffusion/TextModel/DiffusionDecoderAttention.swift:150` — `finalKeys = concatenated([encoderEntry.keys, keys], axis: 2)`
- `Sources/Gemma4Swift/Diffusion/TextModel/DiffusionDecoderAttention.swift:151` — `finalValues = concatenated([encoderEntry.values, values], axis: 2)`
- `Sources/Gemma4Swift/Diffusion/TextModel/DiffusionGemmaEncoderTextAttention.swift:134` — `attentionKeys = concatenated([prior.keys, newKeys], axis: 2)`
- `Sources/Gemma4Swift/Diffusion/TextModel/DiffusionGemmaEncoderTextAttention.swift:135` — `attentionValues = concatenated([prior.values, newValues], axis: 2)`
- `Sources/Gemma4Swift/Diffusion/TextModel/DiffusionGemmaEncoderTextModel.swift:126` — `let mergedKeys = concatenated([prior.keys, newKeys], axis: 2)`
- `Sources/Gemma4Swift/Diffusion/TextModel/DiffusionGemmaEncoderTextModel.swift:127` — `let mergedValues = concatenated([prior.values, newValues], axis: 2)`
- `Sources/Gemma4Swift/LoRA/Gemma4TrainingLoop.swift:377` — `var imageFeatures = concatenated(allFeatures, axis: 1)`
- … 31 de plus

</details>

<details><summary><code>fp32-cast</code> — 34 hors tests</summary>

- `Sources/Gemma4Swift/AudioEncoder/AudioAttention.swift:102` — `var q = qProj(hiddenStates).asType(.float32).reshaped(B, T, numHeads, headDim)`
- `Sources/Gemma4Swift/AudioEncoder/AudioAttention.swift:103` — `var k = kProj(hiddenStates).asType(.float32).reshaped(B, T, numHeads, headDim)`
- `Sources/Gemma4Swift/AudioEncoder/AudioAttention.swift:104` — `let v = vProj(hiddenStates).asType(.float32).reshaped(B, T, numHeads, headDim)`
- `Sources/Gemma4Swift/Diffusion/TextModel/DiffusionGemmaDecoderTextModel.swift:94` — `var probs = softmax(scLogits.asType(.float32), axis: -1)`
- `Sources/Gemma4Swift/Diffusion/TextModel/DiffusionGemmaForBlockDiffusion.swift:59` — `return applyLogitSoftcapping(logits.asType(.float32))`
- `Sources/Gemma4Swift/LoRA/Gemma4LoRATrain.swift:359` — `array.dtype.isFloatingPoint ? array.asType(.float32) : array`
- `Sources/Gemma4Swift/LoRA/Gemma4TrainingLoop.swift:25` — `let logits = model(inputs, cache: nil).asType(.float32)`
- `Sources/Gemma4Swift/Multimodal/Gemma4Model.swift:47` — `inputsEmbeds = inputsEmbeds * MLXArray(languageModel.model.embedScale, dtype: .float32)`
- `Sources/Gemma4Swift/Pipeline/Gemma4MultimodalLLMModel.swift:116` — `inputsEmbeds = inputsEmbeds * MLXArray(languageModel.model.embedScale, dtype: .float32)`
- `Sources/Gemma4Swift/Pipeline/Gemma4UnifiedImageProcessor.swift:99` — `let rgb = raw.asType(.float32) / MLXArray(Float(255.0))`
- `Sources/Gemma4Swift/Pipeline/Gemma4UnifiedMultimodalLLMModel.swift:209` — `inputsEmbeds = inputsEmbeds * MLXArray(languageModel.model.embedScale, dtype: .float32)`
- `Sources/Gemma4Swift/Pipeline/ImageProcessor.swift:68` — `let chw = rgb.transposed(2, 0, 1).asType(.float32) / MLXArray(Float(255.0))`
- … 22 de plus

</details>

<details><summary><code>compile</code> — 1 hors tests</summary>

- `Sources/Gemma4Swift/Diffusion/Sampling/EntropyBoundSampler.swift:70` — `nonisolated(unsafe) static let compiledTokenEntropy: @Sendable (MLXArray) -> MLXArray = MLX.compile(shapeless: true) { logits -> MLXArray in`

</details>

<details><summary><code>dequantize</code> — 2 hors tests</summary>

- `Sources/Gemma4Swift/TurboQuant/TurboQuantMSECodec.swift:127` — `public func dequantize(_ state: TurboQuantMSEState) -> MLXArray {`
- `Sources/Gemma4Swift/TurboQuant/TurboQuantProdCodec.swift:73` — `public func dequantize(_ state: TurboQuantProdState) -> MLXArray {`

</details>

<details><summary><code>quantized-kv</code> — 40 hors tests</summary>

- `Sources/Gemma4CLI/EvalCommand.swift:56` — `var kvBits: Int?`
- `Sources/Gemma4CLI/EvalCommand.swift:154` — `let kvBitsParam = self.kvBits`
- `Sources/Gemma4CLI/EvalCommand.swift:158` — `if kvBitsParam != nil { correctByCfg["TQ-kv"] = 0; totalByCfg["TQ-kv"] = 0 }`
- `Sources/Gemma4CLI/EvalCommand.swift:166` — `if let kv = kvBitsParam {`
- `Sources/Gemma4CLI/EvalCommand.swift:201` — `kvBits: kv`
- `Sources/Gemma4CLI/EvalCommand.swift:208` — `kvBits: kv`
- `Sources/Gemma4CLI/EvalCommand.swift:235` — `(kvBitsParam != nil ? " TQ=\(ltr(answers["TQ-kv"]))" : "") +`
- `Sources/Gemma4CLI/EvalCommand.swift:266` — `kvBits: Int?`
- `Sources/Gemma4CLI/EvalCommand.swift:268` — `let kvBitsArg = kvBits`
- `Sources/Gemma4CLI/EvalCommand.swift:272` — `let params = kvBitsArg != nil ? GenerateParameters(kvBits: kvBitsArg) : nil`
- `Sources/Gemma4CLI/EvalCommand.swift:343` — `kvBits: Int?`
- `Sources/Gemma4CLI/EvalCommand.swift:345` — `let kvBitsArg = kvBits`
- … 28 de plus

</details>

<details><summary><code>prefill-step</code> — 2 hors tests</summary>

- `Sources/Gemma4Swift/Pipeline/Gemma4Pipeline.swift:505` — `prefillStepSize: params.prefillStepSize,`
- `Sources/Gemma4Swift/Pipeline/Gemma4Pipeline.swift:663` — `prefillStepSize: params.prefillStepSize,`

</details>

<details><summary><code>kv-trim</code> — 3 hors tests</summary>

- `Sources/Gemma4BenchUI/Agent/WebBrowserHost.swift:278` — `var raw = (el.innerText || el.value || el.getAttribute('aria-label') || el.title || '').trim();`
- `Sources/Gemma4BenchUI/Agent/WebBrowserHost.swift:364` — `var txt = (el.innerText || el.value || el.getAttribute('aria-label') || '').trim();`
- `Sources/Gemma4Swift/Pipeline/Gemma4MTPPipeline.swift:288` — `trimPromptCache(cache, numTokens: toTrim)`

</details>

<details><summary><code>resize-image</code> — 9 hors tests</summary>

- `Sources/Gemma4CLI/ProfileCommand.swift:337` — `for targetSize in sizes {`
- `Sources/Gemma4CLI/ProfileCommand.swift:341` — `if targetSize > maxCtx {`
- `Sources/Gemma4CLI/ProfileCommand.swift:342` — `print("  \(String(format: "%7d", targetSize))    SKIP (depasse max_position_embeddings = \(maxCtx))")`
- `Sources/Gemma4CLI/ProfileCommand.swift:351` — `print("  \(String(format: "%7d", targetSize))    SKIP (pic precedent \(String(format: "%.1f", lastPeakGB)) Go > safe max \(String(format: "%`
- `Sources/Gemma4CLI/ProfileCommand.swift:360` — `r.kvBits == kvBits && abs(r.contextTokens - targetSize) < max(50, targetSize / 10)`
- `Sources/Gemma4CLI/ProfileCommand.swift:364` — `print("  \(String(format: "%7d", targetSize))    \(cfgLabel.padding(toLength: 12, withPad: " ", startingAt: 0))  [deja fait, skip]")`
- `Sources/Gemma4CLI/ProfileCommand.swift:370` — `targetTokens: targetSize,`
- `Sources/Gemma4Swift/Pipeline/ImageProcessor.swift:52` — `let (bestW, bestH) = try targetSize(`
- `Sources/Gemma4Swift/Pipeline/ImageProcessor.swift:75` — `private static func targetSize(`

</details>

## 5. Indices stabilité

| Motif | Occ. (Sources / Tests) | Sens | Catalogue |
|---|---|---|---|
| `try-bang` | 2 / 5 | try! : crash sur erreur |  |
| `fatal` | 20 / 0 | Arrêt dur (entrée utilisateur ?) |  |
| `try-q` | 109 / 10 | Erreur avalée silencieusement ? |  |
| `unchecked-sendable` | 26 / 1 | Sendable non vérifié | feedback MLXArray/thread |
| `nonisolated-unsafe` | 60 / 6 | État global non isolé |  |
| `detached` | 9 / 0 | Task.detached (task-locals perdus, eval avant transfert) |  |
| `todo` | 2 / 0 | Dette marquée |  |
| `value-and-grad` | 3 / 0 | Gradient : ne pas mélanger avec l'inférence (ABBA) | piège 20 |
| `free-load` | 0 / 0 | Chargement via ModelFactoryRegistry (VLM d'abord) |  |
| `print` | 504 / 0 | print() dans une bibliothèque (profiler plutôt) | measurement |

<details><summary><code>try-bang</code> — 2 hors tests</summary>

- `Sources/Gemma4Swift/Pipeline/Gemma4UnifiedVideoProcessor.swift:116` — `let data = try! JSONSerialization.data(withJSONObject: json)`
- `Sources/Gemma4Swift/Pipeline/Gemma4UnifiedVideoProcessor.swift:117` — `return try! JSONDecoder().decode(Gemma4UnifiedVisionConfig.self, from: data)`

</details>

<details><summary><code>fatal</code> — 20 hors tests</summary>

- `Sources/Gemma4CLI/MtpDiagVerifyCommand.swift:52` — `fatalError("Expected Gemma4LLMModel, got \(type(of: context.model))")`
- `Sources/Gemma4CLI/MtpForwardCommand.swift:116` — `fatalError("Expected Gemma4LLMModel, got \(type(of: context.model))")`
- `Sources/Gemma4CLI/MtpForwardCommand.swift:156` — `fatalError("Cannot extract shared K/V from intermediates (firstShared=\(firstSharedIdx))")`
- `Sources/Gemma4CLI/MtpTrainCommand.swift:130` — `fatalError("Expected Gemma4LLMModel for training")`
- `Sources/Gemma4CLI/MtpTrainCommand.swift:140` — `fatalError("Cannot find concrete full_attention and sliding_attention layers")`
- `Sources/Gemma4CLI/MtpTrainCommand.swift:183` — `fatalError("Aucun sample n'a au moins \(sl) tokens")`
- `Sources/Gemma4Swift/Diffusion/TextModel/DiffusionGemmaEncoderTextModel.swift:86` — `fatalError("inputs ou inputsEmbeds requis")`
- `Sources/Gemma4Swift/Diffusion/TextModel/EncoderKVCache.swift:65` — `fatalError("EncoderKVCache.get(layerIdx: \(layerIdx)) : cache non rempli pour cette couche")`
- `Sources/Gemma4Swift/LoRA/Gemma4LoRAData.swift:146` — `fatalError("Type de fichier non supporte: \(url.pathExtension)")`
- `Sources/Gemma4Swift/Speculative/Gemma4AssistantDraftModel.swift:163` — `fatalError("sharedKVStates manque pour layer_type=\(layerType)")`
- `Sources/Gemma4Swift/Speculative/Gemma4AssistantDraftModel.swift:198` — `fatalError("Aucun LM head disponible (ni masked_embedding, ni tied, ni lm_head)")`
- `Sources/Gemma4Swift/Speculative/Gemma4AssistantDraftModel.swift:233` — `fatalError("sharedKVStates manque pour layer_type=\(layerType)")`
- … 8 de plus

</details>

<details><summary><code>try-q</code> — 109 hors tests</summary>

- `Sources/Gemma4BenchUI/Agent/AgentPrompts.swift:135` — `let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],`
- `Sources/Gemma4BenchUI/Agent/AgentPrompts.swift:174` — `let primary = try? NSRegularExpression(`
- `Sources/Gemma4BenchUI/Agent/AgentPrompts.swift:189` — `let fallback = try? NSRegularExpression(`
- `Sources/Gemma4BenchUI/Agent/AgentStepViewModel.swift:114` — `try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)`
- `Sources/Gemma4BenchUI/Agent/AgentStepViewModel.swift:123` — `try? manifest.write(to: dir.appendingPathComponent("manifest.md"), atomically: true, encoding: .utf8)`
- `Sources/Gemma4BenchUI/Agent/AgentStepViewModel.swift:130` — `try? content.write(to: dir.appendingPathComponent(relativePath), atomically: true, encoding: .utf8)`
- `Sources/Gemma4BenchUI/Agent/AgentStepViewModel.swift:138` — `try? png.write(to: dir.appendingPathComponent(relativePath))`
- `Sources/Gemma4BenchUI/Agent/AgentStepViewModel.swift:206` — `try? await Task.sleep(nanoseconds: 200_000_000)`
- `Sources/Gemma4BenchUI/Agent/AgentStepViewModel.swift:208` — `try? await Task.sleep(nanoseconds: 600_000_000) // marge pour rendu JS`
- `Sources/Gemma4BenchUI/Agent/AgentStepViewModel.swift:213` — `try? await Task.sleep(nanoseconds: 500_000_000)`
- `Sources/Gemma4BenchUI/Agent/AgentStepViewModel.swift:273` — `let pixels = try? await Gemma4ImageProcessor.processImage(cg, priority: .userInitiated)`
- `Sources/Gemma4BenchUI/Agent/AgentStepViewModel.swift:405` — `try? await Task.sleep(nanoseconds: 900_000_000)`
- … 97 de plus

</details>

<details><summary><code>unchecked-sendable</code> — 26 hors tests</summary>

- `Sources/Gemma4Swift/Configuration/Gemma4UnifiedConfig.swift:10` — `public struct Gemma4UnifiedConfig: Decodable, @unchecked Sendable {`
- `Sources/Gemma4Swift/Configuration/Gemma4VisionConfig.swift:70` — `public struct AnyCodable: Codable, @unchecked Sendable {`
- `Sources/Gemma4Swift/Diffusion/Configuration/DiffusionGemmaConfig.swift:14` — `public struct DiffusionGemmaConfig: Decodable, @unchecked Sendable {`
- `Sources/Gemma4Swift/Diffusion/Configuration/DiffusionGemmaTextConfig.swift:20` — `public struct DiffusionGemmaTextConfig: Decodable, @unchecked Sendable {`
- `Sources/Gemma4Swift/Diffusion/Pipeline/DiffusionGemmaPipeline.swift:30` — `public struct DiffusionGenerationResult: @unchecked Sendable {`
- `Sources/Gemma4Swift/Diffusion/Pipeline/DiffusionGemmaRegistration.swift:17` — `public struct DiffusionGemmaContainer: @unchecked Sendable {`
- `Sources/Gemma4Swift/Diffusion/Sampling/EntropyBoundSampler.swift:31` — `public final class EntropyBoundSampler: @unchecked Sendable {`
- `Sources/Gemma4Swift/Diffusion/Sampling/StableConfidentStopping.swift:17` — `public final class StableConfidentStopping: @unchecked Sendable {`
- `Sources/Gemma4Swift/Diffusion/TextModel/DiffusionAttentionMask.swift:33` — `public struct Mapping: @unchecked Sendable {`
- `Sources/Gemma4Swift/Diffusion/TextModel/DiffusionGemmaEncoderTextModel.swift:25` — `public struct DiffusionEncoderOutput: @unchecked Sendable {`
- `Sources/Gemma4Swift/Diffusion/TextModel/DiffusionGemmaForBlockDiffusion.swift:18` — `public struct DiffusionForwardOutput: @unchecked Sendable {`
- `Sources/Gemma4Swift/Diffusion/TextModel/EncoderKVCache.swift:22` — `public struct EncoderKVCache: @unchecked Sendable {`
- … 14 de plus

</details>

<details><summary><code>nonisolated-unsafe</code> — 60 hors tests</summary>

- `Sources/Gemma4BenchUI/Agent/AgentStepViewModel.swift:308` — `nonisolated(unsafe) let unsafeIds = MLXArray(ids.map { Int32($0) }).reshaped(1, -1)`
- `Sources/Gemma4BenchUI/Agent/AgentStepViewModel.swift:309` — `nonisolated(unsafe) let unsafePixels: MLXArray? = pixels`
- `Sources/Gemma4BenchUI/Agent/AgentStepViewModel.swift:310` — `nonisolated(unsafe) let unsafeModel = model`
- `Sources/Gemma4BenchUI/Agent/AgentStepViewModel.swift:311` — `nonisolated(unsafe) let unsafeTokenizer = dtok`
- `Sources/Gemma4BenchUI/Agent/WebAgentLoop.swift:284` — `nonisolated(unsafe) let unsafeIds = MLXArray(ids.map { Int32($0) }).reshaped(1, -1)`
- `Sources/Gemma4BenchUI/Agent/WebAgentLoop.swift:285` — `nonisolated(unsafe) let unsafePixels: MLXArray? = pixels`
- `Sources/Gemma4BenchUI/Agent/WebAgentLoop.swift:286` — `nonisolated(unsafe) let unsafeModel = diff`
- `Sources/Gemma4BenchUI/Agent/WebAgentLoop.swift:287` — `nonisolated(unsafe) let unsafeTokenizer = dtok`
- `Sources/Gemma4BenchUI/BenchViewModel.swift:447` — `nonisolated(unsafe) let unsafeModel = model`
- `Sources/Gemma4BenchUI/BenchViewModel.swift:448` — `nonisolated(unsafe) let unsafeTokenizer = tokenizer`
- `Sources/Gemma4BenchUI/BenchViewModel.swift:449` — `nonisolated(unsafe) let unsafePixels = pixelValues`
- `Sources/Gemma4BenchUI/BenchViewModel.swift:477` — `nonisolated(unsafe) let nonisolatedTokenizer = unsafeTokenizer`
- … 48 de plus

</details>

<details><summary><code>detached</code> — 9 hors tests</summary>

- `Sources/Gemma4BenchUI/Agent/AgentStepViewModel.swift:321` — `let out: (text: String, forwards: Int) = await Task.detached {`
- `Sources/Gemma4BenchUI/Agent/WebAgentLoop.swift:288` — `let outText: String = await Task.detached {`
- `Sources/Gemma4BenchUI/BenchViewModel.swift:451` — `let task = Task.detached {`
- `Sources/Gemma4BenchUI/IOSSim/IOSAgentStepViewModel.swift:302` — `let out: (text: String, forwards: Int) = await Task.detached {`
- `Sources/Gemma4BenchUI/VQAGame/VQAGameViewModel.swift:142` — `let result: (text: String, steps: Int) = await Task.detached {`
- `Sources/Gemma4Swift/Pipeline/Gemma4UnifiedImageProcessor.swift:173` — `try await Task.detached(priority: priority) {`
- `Sources/Gemma4Swift/Pipeline/Gemma4UnifiedImageProcessor.swift:187` — `try await Task.detached(priority: priority) {`
- `Sources/Gemma4Swift/Pipeline/ImageProcessor.swift:169` — `let transferred = try await Task.detached(priority: priority) {`
- `Sources/Gemma4Swift/Pipeline/ImageProcessor.swift:192` — `let transferred = try await Task.detached(priority: priority) {`

</details>

<details><summary><code>todo</code> — 2 hors tests</summary>

- `Sources/Gemma4Swift/Diffusion/Pipeline/DiffusionGemmaPipeline.swift:138` — `// (~600 MB sur 50 GB, negligeable). TODO Phase 10.`
- `Sources/Gemma4Swift/TurboQuant/TurboQuantMSECodec.swift:31` — `// TODO: reimplementer le WHT via Metal kernel pour O(D log D)`

</details>

<details><summary><code>value-and-grad</code> — 3 hors tests</summary>

- `Sources/Gemma4Swift/LoRA/Gemma4TrainingLoop.swift:171` — `let lossValueGrad = valueAndGrad(model: model) { model, arrays in`
- `Sources/Gemma4Swift/LoRA/Gemma4TrainingLoop.swift:352` — `let lossValueGrad = valueAndGrad(model: model) { model, arrays in`
- `Sources/Gemma4Swift/Speculative/Gemma4DrafterTraining.swift:178` — `let lossValueGrad = valueAndGrad(model: drafter) { (drafter: Gemma4AssistantDraftModel, arrays: [MLXArray]) -> [MLXArray] in`

</details>

<details><summary><code>print</code> — 504 hors tests</summary>

- `Sources/Gemma4BenchUI/Agent/AgentStepViewModel.swift:125` — `print("[WebAgent] session log dir: \(dir.path)")`
- `Sources/Gemma4BenchUI/Agent/AgentStepViewModel.swift:212` — `print("[AgentStep] cookie banner auto-dismissed: \(dismissed)")`
- `Sources/Gemma4BenchUI/Agent/AgentStepViewModel.swift:335` — `print("[AgentStep] step \(n) attempt \(attempt + 1) raw: \(out.text.replacingOccurrences(of: "\n", with: " | "))")`
- `Sources/Gemma4BenchUI/Agent/AgentStepViewModel.swift:342` — `print("[AgentStep] step \(n) attempt \(attempt + 1) — action non parseable, retry seed")`
- `Sources/Gemma4BenchUI/Agent/AgentStepViewModel.swift:360` — `print("[AgentStep] step \(n) action: \(a.kind)")`
- `Sources/Gemma4BenchUI/Agent/AgentStepViewModel.swift:362` — `print("[AgentStep] step \(n) !! NO ACTION after retries — raw was:\n\(lastRawOut)")`
- `Sources/Gemma4BenchUI/Agent/AgentStepViewModel.swift:410` — `print("[AgentStep] cookie banner auto-dismissed after click: \(dismissed)")`
- `Sources/Gemma4BenchUI/Agent/AgentStepViewModel.swift:451` — `print("[AgentStep] cookie banner auto-dismissed after click_and_type: \(dismissed)")`
- `Sources/Gemma4BenchUI/Agent/WebAgentLoop.swift:139` — `print("[Agent] Goal: \(goal)")`
- `Sources/Gemma4BenchUI/Agent/WebAgentLoop.swift:140` — `print("[Agent] Diag dir: \(diagDir.path)")`
- `Sources/Gemma4BenchUI/Agent/WebAgentLoop.swift:149` — `print("[Agent] === Step \(n) ===")`
- `Sources/Gemma4BenchUI/Agent/WebAgentLoop.swift:150` — `print("[Agent] URL: \(browser.currentURL) | Title: \(browser.pageTitle)")`
- … 492 de plus

</details>

## 6. Alertes synthétiques

- 🟡 `cacheLimit` posé dans très peu de fichiers : vérifier que chaque chemin d'inférence en a un.
- 🟠 compile() ET gradients dans la même bibliothèque : risque de deadlock ABBA mlx-swift (piège 20).
- 🟢 Candidat(s) profils : `Sources/Gemma4BenchUI/IOSSim/IOSAppPreset.swift`
- 🟡 Module `Gemma4BenchUI` : aucun test ne l'importe ni ne porte son nom.
- 🟡 Module `Gemma4CLI` : aucun test ne l'importe ni ne porte son nom.

## 7. Standard documentaire (references/knowledge-structure.md)

| Élément | Présent | Rôle |
|---|---|---|
| `BENCHMARKS.md` | ✅ | Lignes de mesure brutes, jamais éditées |
| `docs/References.md` | ❌ | Table des profils de référence mesurés |
| `docs/Weights.md` | ❌ | Poids / packs recommandés par profil |
| `docs/Benchmarks.md` | ❌ | Protocole de mesure |
| `docs/knowledge/index.md` | ❌ | Index de la base de connaissance |
| `docs/knowledge/log.md` | ❌ | Journal horodaté |
| `docs/knowledge/decisions` | ❌ | Décisions |
| `docs/knowledge/pitfalls` | ❌ | Pièges |
| `CHANGELOG.md` | ❌ | Journal des versions |
| `PLAN.md` | ❌ | Plan / chantiers |

