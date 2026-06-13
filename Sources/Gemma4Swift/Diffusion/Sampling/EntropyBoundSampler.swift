// Port de EntropyBoundSampler (cf. arXiv:2505.24857)
//
// Algo :
//   1. Pour chaque position, calcule l'entropie de la distribution sur le vocab.
//   2. Trie les positions par entropie croissante.
//   3. Accepte les positions tant que cumsum(entropy) - max(entropy_accepted) <= entropy_bound.
//      => garantit que somme des MI restantes (~ somme des entropies independantes) est bornee
//         => les tokens acceptes peuvent etre echantillonnes "comme s'ils etaient independants"
//   4. Pour les positions acceptees : remplace current_canvas par denoiser_canvas
//      Pour les positions rejetees : re-bruit avec un canvas aleatoire
//
// Python ref (generation_diffusion_gemma.py) :
//   class EntropyBoundSampler:
//       def accept_canvas(self, current_canvas, denoiser_canvas, logits, cur_step):
//           token_entropy = Categorical(logits=logits).entropy()
//           sorted_e, sorted_idx = sort(token_entropy, descending=False)
//           cumulative = cumsum(sorted_e)
//           selection_mask = cumulative - sorted_e <= entropy_bound
//           accepted = scatter(selection_mask, sorted_idx)
//           return where(accepted, denoiser, current)
//       def renoise_canvas(self, accepted_canvas, cur_step):
//           random = randint(0, vocab_size, shape=canvas)
//           return where(~accepted, random, accepted)

import Foundation
import MLX
import MLXNN
import MLXRandom

/// EntropyBoundSampler : sampler bloc-AR pour diffusion discrete texte.
public final class EntropyBoundSampler: @unchecked Sendable {
    public let entropyBound: Float
    public let vocabSize: Int
    public let canvasLength: Int

    /// Masque d'acceptation produit par le dernier `accept(...)`, reutilise par `renoise(...)`.
    /// Shape : `[batch_size, canvas_length]`, bool.
    public private(set) var acceptedTokenMask: MLXArray?

    public init(entropyBound: Float, vocabSize: Int, canvasLength: Int) {
        self.entropyBound = entropyBound
        self.vocabSize = vocabSize
        self.canvasLength = canvasLength
    }

    /// Initialise un canvas aleatoire (uniforme sur le vocab).
    /// Shape de sortie : `[batchSize, canvasLength]`, dtype int32.
    public func initializeCanvas(batchSize: Int, key: MLXArray? = nil) -> MLXArray {
        MLXRandom.randInt(
            low: MLXArray(Int32(0)),
            high: MLXArray(Int32(vocabSize)),
            [batchSize, canvasLength],
            key: key
        )
    }

    /// Entropie par position : `H = -sum_v softmax(logits)_v * log_softmax(logits)_v`.
    /// `logits` shape : `[B, T, V]`. Retour : `[B, T]`.
    public static func tokenEntropy(_ logits: MLXArray) -> MLXArray {
        // log_softmax stable
        let logProbs = MLXNN.logSoftmax(logits, axis: -1)
        let probs = exp(logProbs)
        let mixed: MLXArray = probs * logProbs
        return MLXArray(0.0) - mixed.sum(axis: -1)
    }

    /// Version compilee de tokenEntropy via MLX.compile.
    /// Le JIT MLX fuse les kernels logSoftmax + exp + multiply + sum.
    /// Shapeless: true permet de varier la batch size sans recompiler.
    nonisolated(unsafe) static let compiledTokenEntropy: @Sendable (MLXArray) -> MLXArray = MLX.compile(shapeless: true) { logits -> MLXArray in
        let logProbs = MLXNN.logSoftmax(logits, axis: -1)
        let probs = exp(logProbs)
        let mixed: MLXArray = probs * logProbs
        return MLXArray(0.0) - mixed.sum(axis: -1)
    }

    /// Bascule entre tokenEntropy (eager) et compiledTokenEntropy (JIT fusion).
    /// Par defaut on garde l'ancienne pour ne pas changer la semantique a la
    /// volee, mais le sampler peut l'activer via useCompiledEntropy = true.
    public var useCompiledEntropy: Bool = false

    /// Etape "accept" : retourne le canvas mixant les tokens acceptes (du denoiser)
    /// avec ceux maintenus (du current). Met a jour `acceptedTokenMask`.
    ///
    /// - Parameters:
    ///   - currentCanvas : `[B, T]`, int. Canvas actuel (bruit + tokens precedemment acceptes).
    ///   - denoiserCanvas : `[B, T]`, int. Echantillon issu des logits cette etape.
    ///   - logits : `[B, T, V]`, float. Logits du denoiser cette etape.
    /// - Returns: `[B, T]`, int. Canvas accepte.
    public func accept(
        currentCanvas: MLXArray,
        denoiserCanvas: MLXArray,
        logits: MLXArray
    ) -> MLXArray {
        let entropy = useCompiledEntropy
            ? Self.compiledTokenEntropy(logits)
            : Self.tokenEntropy(logits)  // [B, T]

        // Tri ascendant par entropie le long de T
        let sortedIdx = argSort(entropy, axis: -1)        // [B, T]
        let sortedEntropy = takeAlong(entropy, sortedIdx, axis: -1)  // [B, T]

        // cumsum sur l'axe T
        let cumE = cumsum(sortedEntropy, axis: -1)

        // selection_mask = (cumE - sortedEntropy) <= entropy_bound
        // Python: cumulative_entropy - sorted_token_entropy <= entropy_bound
        let sortedSelection = (cumE - sortedEntropy) .<= MLXArray(entropyBound)  // [B, T] bool

        // scatter back vers l'ordre original via putAlong
        var accepted = MLXArray.zeros(like: sortedSelection).asType(.bool)
        accepted = putAlong(accepted, sortedIdx, values: sortedSelection, axis: -1)

        self.acceptedTokenMask = accepted

        return MLX.where(accepted, denoiserCanvas, currentCanvas)
    }

    /// Etape "renoise" : remplace les positions REJETEES par du bruit uniforme frais.
    /// Doit etre appele apres `accept(...)`.
    public func renoise(
        acceptedCanvas: MLXArray,
        batchSize: Int,
        key: MLXArray? = nil
    ) -> MLXArray {
        guard let mask = acceptedTokenMask else {
            return acceptedCanvas  // pas d'appel a accept() : rien a re-bruiter
        }
        let renoiseMask = logicalNot(mask)  // positions a re-bruiter
        let randomCanvas = initializeCanvas(batchSize: batchSize, key: key)
        return MLX.where(renoiseMask, randomCanvas, acceptedCanvas)
    }
}
