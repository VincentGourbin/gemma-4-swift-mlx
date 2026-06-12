// Bidirectional attention overlay pour les blocs vision (gemma4_unified).
//
// Port du Python LanguageModel._block_sequence_ids_for_mask /
// _apply_blockwise_bidirectional_overlay.
//
// Pour chaque bloc CONTIGU de tokens vision dans la sequence, on autorise
// l'attention bidirectionnelle entre les positions de CE bloc. Le reste
// (text-text, text-vision, vision-text vers le futur) reste causal.

import Foundation
import MLX

public enum Gemma4BidirectionalMask {

    /// Pour un mask [B, T] (true = token vision), retourne block_sequence_ids
    /// [B, T] int32 : id de bloc pour chaque position vision, -1 pour text/audio.
    ///
    /// Exemple : visionMask = [0,0,1,1,1,0,0,1,1,0]
    ///          block_ids  = [-1,-1,0,0,0,-1,-1,1,1,-1]
    public static func blockSequenceIds(visionMask: MLXArray) -> MLXArray {
        let isVision = visionMask
        // prev[b, t] = isVision[b, t-1] (avec 0 prepended)
        let B = isVision.dim(0)
        let T = isVision.dim(1)
        let zerosPrefix = MLXArray.zeros([B, 1], type: Bool.self)
        let prev: MLXArray
        if T <= 1 {
            prev = zerosPrefix
        } else {
            prev = concatenated([zerosPrefix, isVision[0..., 0 ..< (T - 1)]], axis: 1)
        }
        // starts[b, t] = isVision[b, t] AND NOT prev[b, t]
        let starts = isVision .&& logicalNot(prev)
        // group_ids = cumsum(starts) - 1
        let cumStarts = cumsum(starts.asType(.int32), axis: 1)
        let groupIds = cumStarts - MLXArray(Int32(1))
        // Pour les positions text : -1
        let minusOne = MLXArray.zeros(like: groupIds) - MLXArray(Int32(1))
        return MLX.where(isVision, groupIds, minusOne)
    }

    /// Construit l'overlay bidirectionnel a partir des block_sequence_ids.
    /// Retourne un mask bool [B, T, T] : true ou les 2 positions sont dans LE MEME bloc vision.
    public static func overlay(blockSequenceIds: MLXArray) -> MLXArray {
        // qBlocks [B, T, 1], kBlocks [B, 1, T]
        let qBlocks = expandedDimensions(blockSequenceIds, axis: -1)
        let kBlocks = expandedDimensions(blockSequenceIds, axis: -2)
        let notMinus = qBlocks .!= MLXArray(Int32(-1))
        let sameBlock = (qBlocks .== kBlocks) .&& notMinus
        return sameBlock
    }

    /// Compose un mask causal materialise (bool [T, T]) avec l'overlay bidirectionnel.
    /// `causal` shape [T, T] bool. `overlayBT` shape [B, T, T] bool.
    /// Retourne [B, T, T] bool (true = autorise).
    public static func compose(causal: MLXArray, overlay overlayBT: MLXArray) -> MLXArray {
        // Broadcast causal [T, T] -> [1, T, T] pour OR.
        let causalB = expandedDimensions(causal, axis: 0)
        return causalB .|| overlayBT
    }
}
