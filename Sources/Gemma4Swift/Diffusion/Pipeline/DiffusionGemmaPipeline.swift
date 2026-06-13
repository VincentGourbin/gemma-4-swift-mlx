// Pipeline de generation DiffusionGemma block-AR.
//
// Algorithme :
//   for canvas in 0..<max_new_canvases:
//       encoder_cache = encoder(input_ids)
//       canvas = sampler.initialize_canvas(B)
//       prev_logits = nil
//       stopping.reset()
//       for step in max_denoising_steps...1:
//           logits = model.denoiseStep(canvas, encoder_cache, prev_logits)
//           logits = temperature.apply(logits, step)
//           argmax_canvas = argmax(logits)
//           denoiser_canvas = sample(logits)
//           canvas = sampler.accept(canvas, denoiser_canvas, logits)
//           if stopping.shouldStop(argmax_canvas, logits).all() : break
//           canvas = sampler.renoise(canvas, B)
//           prev_logits = logits
//       input_ids = cat(input_ids, argmax_canvas)
//       if EOS in argmax_canvas : break
//
// Simplification Phase 4 : on re-encode TOUT le prompt a chaque canvas.
// Phase 5+ : KV cache incremental cote encoder (faster).

import Foundation
import MLX
import MLXNN
import MLXRandom

/// Sortie d'une generation block-AR.
public struct DiffusionGenerationResult: @unchecked Sendable {
    /// Tokens generes (apres le prompt initial).
    public let generatedIds: MLXArray
    /// Concatenation prompt + tokens generes.
    public let fullIds: MLXArray
    /// Nombre de forwards decoder reellement executes.
    public let totalDecoderSteps: Int
    /// Nombre de canvases utilises.
    public let canvases: Int
}

/// Pipeline de generation DiffusionGemma block-AR.
public actor DiffusionGemmaPipeline {
    public let model: DiffusionGemmaForBlockDiffusion
    public let genConfig: DiffusionGenerationConfig
    public let sampler: EntropyBoundSampler
    public let temperatureSchedule: LinearTemperatureSchedule
    public let stopping: StableConfidentStopping

    public init(
        model: DiffusionGemmaForBlockDiffusion,
        genConfig: DiffusionGenerationConfig
    ) {
        self.model = model
        self.genConfig = genConfig
        let vocab = model.config.textConfig.base.vocabSize
        let canvas = model.config.textConfig.canvasLength
        self.sampler = EntropyBoundSampler(
            entropyBound: genConfig.entropyBound,
            vocabSize: vocab,
            canvasLength: canvas
        )
        self.temperatureSchedule = LinearTemperatureSchedule(
            tMin: genConfig.tMin,
            tMax: genConfig.tMax,
            maxDenoisingSteps: genConfig.maxDenoisingSteps
        )
        self.stopping = StableConfidentStopping(
            stabilityThreshold: genConfig.stabilityThreshold,
            confidenceThreshold: genConfig.confidenceThreshold
        )
    }

    /// Genere des canvases successifs jusqu'a EOS ou maxBlocks atteint.
    ///
    /// - Parameters:
    ///   - promptIds : `[B, T_prompt]` int. Tokens du prompt initial.
    ///   - maxBlocks : nombre max de canvases (chacun de `canvas_length` tokens).
    ///   - seed : seed pour le PRNG (sampler + initialize_canvas).
    ///   - onCanvas : callback optionnel apres chaque canvas commit. Equivalent
    ///     du `streamer.put(canvas)` cote Python.
    ///   - onStep : callback optionnel apres CHAQUE step de denoising. Recoit
    ///     `(canvasIdx, step, argmaxCanvas)` — le draft courant decode avec
    ///     argmax. Equivalent du `streamer.put_draft(argmax_canvas)` cote
    ///     Python (= passer par le VAE a chaque step en diffusion image).
    ///     Permet d'observer la convergence du denoising en live.
    /// - Returns: tokens generes + stats.
    public func generate(
        promptIds: MLXArray,
        pixelValues: MLXArray? = nil,
        maxBlocks: Int = 4,
        seed: UInt64 = 0,
        onCanvas: ((Int, MLXArray) -> Void)? = nil,
        onStep: ((_ canvasIdx: Int, _ step: Int, _ argmaxCanvas: MLXArray) -> Void)? = nil
    ) -> DiffusionGenerationResult {
        var key = MLXRandom.key(seed)
        let batchSize = promptIds.dim(0)
        let eosSet = Set(genConfig.eosTokenIds.map { Int32($0) })

        var fullIds = promptIds
        var totalSteps = 0
        var canvasesUsed = 0

        // KV cache encoder incremental : maintenu entre les canvases.
        // - Canvas 0 : on encode le prompt complet (+ vision si fourni)
        // - Canvas N+1 : on encode juste les 256 nouveaux tokens (argmax du canvas N)
        //   et on append au cache. Pas de re-encode du prompt initial.
        var encoderCache: EncoderKVCache? = nil

        for canvasIdx in 0 ..< maxBlocks {
            // 1) Encoder forward incremental
            let deltaIds: MLXArray
            let pixelsForCall: MLXArray?
            if encoderCache == nil {
                // Premier appel : encode tout le prompt + vision
                deltaIds = fullIds
                pixelsForCall = pixelValues
            } else {
                // Append : seulement les nouveaux tokens du dernier canvas commit
                let promptLen = encoderCache!.seqLength
                deltaIds = fullIds[0..., promptLen ..< fullIds.dim(1)]
                pixelsForCall = nil
            }
            let encOut = model.encodePrompt(
                promptIds: deltaIds,
                pixelValues: pixelsForCall,
                priorCache: encoderCache
            )
            encoderCache = encOut.kvCache
            // encOut.lastHiddenState : pas utilise dans le pipeline de generation
            // (l'encoder text n'a pas de role direct dans le denoising, c'est le
            // KV cache qui est utilise par le decoder cross-attention).

            // Apres le 1er forward (qui a encode les soft-tokens vision dans le
            // KV cache), on peut decharger vision_tower + embed_vision pour
            // liberer ~600 MB. Pattern LTX unloadAfterUse.
            if canvasIdx == 0 && pixelValues != nil && model.encoder.hasVisionLoaded {
                model.encoder.unloadVision()
            }

            // 2) Init canvas + stopping
            let (k1, k2) = splitKey(key: &key)
            var canvas = sampler.initializeCanvas(batchSize: batchSize, key: k1)
            var argmaxCanvas = canvas
            var prevLogits: MLXArray? = nil
            stopping.reset()
            var rngKey = k2

            // 3) Inner denoising loop : steps decroissants
            var stepsExecuted = 0
            for step in (1 ... genConfig.maxDenoisingSteps).reversed() {
                // a) decoder forward
                let logits = model.denoiseStep(
                    canvasIds: canvas,
                    encoderCache: encoderCache!,
                    selfConditioningLogits: prevLogits,
                    decoderAttentionMask: nil
                )

                // b) temperature
                let scaled = temperatureSchedule.apply(logits, curStep: step)

                // c) argmax + sample
                argmaxCanvas = MLX.argMax(scaled, axis: -1).asType(.int32)
                let (kS, kN) = splitKey(key: &rngKey)
                let denoiserCanvas = MLXRandom.categorical(scaled, axis: -1, key: kS).asType(.int32)
                rngKey = kN

                // Streaming step-by-step : observer la convergence du denoising.
                // Equivalent du `streamer.put_draft(argmax_canvas)` Python.
                onStep?(canvasIdx, step, argmaxCanvas)

                // d) accept / stopping / renoise
                canvas = sampler.accept(
                    currentCanvas: canvas,
                    denoiserCanvas: denoiserCanvas,
                    logits: scaled
                )

                stepsExecuted += 1
                let shouldStop = stopping.shouldStop(argmaxCanvas: argmaxCanvas, logits: scaled)
                if shouldStop.all().item(Bool.self) {
                    break
                }

                let (kR, kNext) = splitKey(key: &rngKey)
                canvas = sampler.renoise(acceptedCanvas: canvas, batchSize: batchSize, key: kR)
                rngKey = kNext

                // prevLogits en bf16 plutot que fp32 : economie 128 MB par step.
                // Le decoder cast en fp32 pour la softmax interne, donc precision
                // preservee pour les positions piquees. Pour les positions tail
                // (probs ~ 1e-7), bf16 introduit du bruit numerique acceptable
                // car amplifie par la softmax fp32 dans le forward suivant.
                prevLogits = scaled.asType(.bfloat16)

                // Note : on a teste asyncEval(canvas, prevLogits, argmaxCanvas) ici
                // pour casser le graph entre steps. Resultat : gain ~0 sur le
                // temps et 0 sur la memoire. Le shouldStop.all().item(...) ligne
                // 152 force deja un eval qui materialise le graph du step en
                // cours. Les ~260 MB qui croissent par canvas ne sont pas une
                // fuite : c'est le KV cache encoder qui grossit avec la longueur
                // du prompt cumule (commit canvas precedents). O(N) attendu.
                // Vraie optim possible : encoder KV cache incremental + quantization.
            }

            totalSteps += stepsExecuted
            canvasesUsed += 1

            // 4) Commit canvas
            fullIds = concatenated([fullIds, argmaxCanvas], axis: -1)
            onCanvas?(canvasIdx, argmaxCanvas)

            // 5) Check EOS
            if containsEOS(canvas: argmaxCanvas, eosSet: eosSet) {
                break
            }

            // 6) Liberation du pic transient du denoising loop (~440 MB observe).
            //    Pattern Flux 2 clearCacheEveryNSteps mais ici entre canvases
            //    (entre steps : neutre testé Phase 5).
            MLX.Memory.clearCache()
        }

        let promptLen = promptIds.dim(1)
        let totalLen = fullIds.dim(1)
        let generatedIds = fullIds[0..., promptLen ..< totalLen]

        return DiffusionGenerationResult(
            generatedIds: generatedIds,
            fullIds: fullIds,
            totalDecoderSteps: totalSteps,
            canvases: canvasesUsed
        )
    }

    // MARK: - Helpers

    /// Split d'une cle PRNG en (k_use, k_next). Met a jour la cle courante.
    private func splitKey(key: inout MLXArray) -> (MLXArray, MLXArray) {
        let split = MLXRandom.split(key: key, into: 2)
        let useKey = split[0]
        let nextKey = split[1]
        key = nextKey
        return (useKey, nextKey)
    }

    /// Verifie si un token EOS est present dans le canvas. Sortie : Bool unique.
    private func containsEOS(canvas: MLXArray, eosSet: Set<Int32>) -> Bool {
        // Approche simple : eval + iteration cote CPU sur le canvas.
        // OK car canvas_length = 256, donc < 1 ms.
        canvas.eval()
        let array = canvas.asArray(Int32.self)
        for tok in array {
            if eosSet.contains(tok) {
                return true
            }
        }
        return false
    }
}
