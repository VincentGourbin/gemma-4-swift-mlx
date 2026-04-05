// TurboQuant Prod Codec — MSE + QJL residuel pour une meilleure precision
// Port de turboquant.py : _TurboQuantProdCodec

import Foundation
import MLX
import MLXFast

/// Codec Prod : combine MSE (bits-1) + QJL 1-bit residuel
/// Le residuel capture l'erreur de quantisation MSE via projection aleatoire + sign bits
public final class TurboQuantProdCodec: @unchecked Sendable {
    public let dim: Int
    public let bits: Int

    /// MSE codec sous-jacent (bits-1 bits)
    public let mseCodec: TurboQuantMSECodec
    /// Matrice de projection pour QJL [D, D]
    public let projection: MLXArray
    /// Projection transposee [D, D]
    public let projectionT: MLXArray
    /// Query transform: [rotation_t | projection_t] concatenees [D, 2D]
    public let queryTransformT: MLXArray
    /// Scale QJL = sqrt(pi/2) / D
    public let scale: Float
    public let scaleArray: MLXArray

    public init(dim: Int, bits: Int, seed: UInt64) {
        self.dim = dim
        self.bits = bits
        self.mseCodec = TurboQuantMSECodec(dim: dim, bits: max(bits - 1, 0), seed: seed)
        self.projection = turboQuantProjectionMatrix(dim: dim, seed: seed &+ 1)
        self.projectionT = dim > 0 ? projection.transposed() : projection

        if dim > 0 {
            self.queryTransformT = concatenated([mseCodec.rotationT, projectionT], axis: -1)
        } else {
            self.queryTransformT = MLXArray.zeros([0, 0])
        }

        self.scale = dim > 0 ? sqrt(Float.pi / 2.0) / Float(dim) : 0.0
        self.scaleArray = MLXArray([scale])
    }

    // MARK: - Quantize

    /// Quantise des vecteurs via MSE + QJL residuel
    public func quantize(_ vectors: MLXArray) -> TurboQuantProdState {
        let vectorsF32 = vectors.asType(.float32)
        let norms = norm(vectorsF32, axis: -1)
        let unitVectors = vectorsF32 / maximum(norms[.ellipsis, .newAxis], MLXArray(TURBOQUANT_EPS))

        // MSE quantize + estimation du vecteur reconstruit
        let (mseIndices, mseUnit) = mseCodec.quantizeUnitWithEstimate(unitVectors)

        // Residuel = vrai - estime
        let residual = unitVectors - mseUnit
        let residualNorms = norm(residual, axis: -1)

        // Projeter le residuel → stocker les signes (1 bit par dimension)
        let projected = matmul(residual, projectionT)
        let signBits = MLX.where(projected .>= 0, MLXArray(UInt32(1)), MLXArray(UInt32(0)))

        return TurboQuantProdState(
            norms: norms.asType(.float16),
            mseIndices: mseIndices,
            residualNorms: residualNorms.asType(.float16),
            qjlSigns: turboQuantPackLowbit(signBits.asType(.uint32), bits: 1)
        )
    }

    // MARK: - Dequantize

    /// Reconstruit les vecteurs depuis l'etat Prod (MSE + QJL)
    public func dequantize(_ state: TurboQuantProdState) -> MLXArray {
        let mseUnit = mseCodec.dequantizeUnit(state.mseIndices)

        let signBitsRaw = turboQuantUnpackLowbit(state.qjlSigns, bits: 1, length: dim).asType(.float32)
        let signs = signBitsRaw * 2.0 - 1.0

        let qjlUnit = scale
            * state.residualNorms[.ellipsis, .newAxis].asType(.float32)
            * matmul(signs, projection)

        return state.norms[.ellipsis, .newAxis].asType(.float32) * (mseUnit + qjlUnit)
    }

    // MARK: - Query Preparation

    /// Prepare les queries : rotation pour MSE + projection pour QJL
    /// Retourne (q_rot, q_proj) pour le scoring
    public func prepareQueries(_ queries: MLXArray) -> (MLXArray, MLXArray) {
        if mseCodec.useRHT {
            let qRot = mseCodec.rotateForward(queries)
            let qProj = matmul(queries, projectionT)
            return (qRot, qProj)
        }
        // Fused: une seule matmul avec [rotation_t | projection_t]
        let transformed = matmul(queries, queryTransformT)
        return (transformed[.ellipsis, 0 ..< dim], transformed[.ellipsis, dim...])
    }

    // MARK: - Scoring

    /// Score: MSE score + QJL score
    public func scorePrepared(
        _ preparedQueries: (MLXArray, MLXArray),
        state: TurboQuantProdState
    ) -> MLXArray {
        let (mseQueries, projQueries) = preparedQueries

        // MSE score
        var mseScore: MLXArray
        if mseCodec.bits > 0 {
            let mseState = TurboQuantMSEState(norms: state.norms, indices: state.mseIndices)
            mseScore = mseCodec.scorePrepared(mseQueries, state: mseState)
        } else {
            mseScore = MLXArray.zeros([
                projQueries.dim(0), projQueries.dim(1), projQueries.dim(2),
                projQueries.dim(3), state.norms.dim(2),
            ])
        }

        // QJL score
        let signBitsRaw = turboQuantUnpackLowbit(state.qjlSigns, bits: 1, length: dim).asType(.float32)
        let signs = signBitsRaw * 2.0 - 1.0

        let qjlScore = scale
            * state.residualNorms.asType(.float32)[0..., 0..., .newAxis, .newAxis, 0...]
            * MLX.einsum("bhmld,bhtd->bhmlt", projQueries, signs)

        let norms = state.norms.asType(.float32)[0..., 0..., .newAxis, .newAxis, 0...]
        return mseScore + norms * qjlScore
    }

    /// Score queries non-preparees
    public func score(_ queries: MLXArray, state: TurboQuantProdState) -> MLXArray {
        scorePrepared(prepareQueries(queries), state: state)
    }
}
