// Cache K/V minimal cote encoder DiffusionGemma.
//
// Apres le forward de l'encoder, on collecte les K/V de chaque couche pour
// les rendre accessibles en lecture seule au decoder. Le decoder ne modifie
// jamais ce cache : il fait juste cat([encoderKV, canvasKV], dim=2) pour
// chaque couche.
//
// Convention shape (post-transpose Gemma4Attention) :
//   keys, values : `[B, numKVHeads, T_encoder, headDim]`
//
// L'`offset` (== T_encoder en pratique) sert a passer aux RoPE en aval si
// besoin, et a documenter le `cache_seq_length` que Python lit via
// `past_key_values.get_seq_length(layer_idx=0)`.
//
// Cette struct n'est PAS un Module : c'est un container leger transmis
// explicitement aux forwards du decoder. Il n'est jamais entraine.

import Foundation
import MLX

/// Cache encoder lu (jamais ecrit) par le decoder DiffusionGemma.
public struct EncoderKVCache: @unchecked Sendable {
    /// Tableau indexable par layerIdx. Une entree par couche du text model.
    /// `nil` autorise pour les couches qui n'ont pas (encore) ete remplies.
    public var entries: [Entry?]

    public struct Entry: @unchecked Sendable {
        public let keys: MLXArray
        public let values: MLXArray

        public init(keys: MLXArray, values: MLXArray) {
            self.keys = keys
            self.values = values
        }
    }

    public init(numLayers: Int) {
        self.entries = Array(repeating: nil, count: numLayers)
    }

    /// Longueur de cache effective (lue par le decoder pour calculer position_ids).
    /// Convention : on prend la longueur depuis la premiere couche remplie.
    public var seqLength: Int {
        for entry in entries {
            if let entry = entry {
                return entry.keys.dim(2)
            }
        }
        return 0
    }

    /// Vrai si toutes les couches sont remplies (cache complet).
    public var isComplete: Bool {
        entries.allSatisfy { $0 != nil }
    }

    /// Set d'une couche specifique.
    public mutating func set(layerIdx: Int, keys: MLXArray, values: MLXArray) {
        entries[layerIdx] = Entry(keys: keys, values: values)
    }

    /// Lecture des K/V d'une couche. Trap si non remplie.
    public func get(layerIdx: Int) -> Entry {
        guard let entry = entries[layerIdx] else {
            fatalError("EncoderKVCache.get(layerIdx: \(layerIdx)) : cache non rempli pour cette couche")
        }
        return entry
    }
}
