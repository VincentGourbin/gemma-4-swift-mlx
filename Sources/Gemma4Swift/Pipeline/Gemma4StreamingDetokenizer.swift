// Detokenisation incrementale pour les boucles de generation manuelles (MTP, CLI)

import Foundation
import MLXLMCommon

/// Detokeniseur incremental qui compare des **scalaires Unicode**, pas des graphemes.
///
/// Decoder chaque token isolement casse les caracteres repartis sur plusieurs
/// tokens (repli octet par octet : 𠀋, 𝔘…). `MLXLMCommon.NaiveStreamingDetokenizer`
/// corrige ce cas mais calcule le nouveau texte par `String.count`, qui compte des
/// graphemes : un scalaire qui fusionne avec le grapheme precedent (2e indicateur
/// regional d'un drapeau 🇫🇷, sequence ZWJ 👩‍👩‍👧, accent combinant) ne fait pas
/// grandir le compte et est perdu. Ici, la difference porte sur les scalaires.
public struct Gemma4StreamingDetokenizer {
    private let tokenizer: any Tokenizer
    private var segmentTokens: [Int] = []
    private var emittedScalars = 0

    public init(tokenizer: any Tokenizer) {
        self.tokenizer = tokenizer
    }

    /// Ajoute un token et rend le texte nouvellement complet, ou `nil` si le token
    /// n'acheve pas encore un caractere (ou n'ajoute rien).
    public mutating func append(token: Int) -> String? {
        segmentTokens.append(token)
        let scalars = tokenizer.decode(tokenIds: segmentTokens).unicodeScalars

        // Caractere UTF-8 incomplet : le decodeur rend U+FFFD en attendant la suite.
        if scalars.last == "\u{FFFD}" { return nil }
        guard scalars.count > emittedScalars else { return nil }

        var fresh = String.UnicodeScalarView()
        fresh.append(contentsOf: scalars.dropFirst(emittedScalars))
        emittedScalars = scalars.count

        // Borne le cout (chaque appel redecode le segment) : nouveau segment a chaque
        // fin de ligne, amorce par le dernier token comme le fait l'amont.
        if fresh.last == "\n" {
            segmentTokens = [token]
            emittedScalars = tokenizer.decode(tokenIds: segmentTokens).unicodeScalars.count
        }
        return String(fresh)
    }
}
