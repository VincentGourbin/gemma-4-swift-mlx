// Generation par lot (K-41) : plusieurs requetes texte decodees dans le meme forward.
//
// Plafond mesure (gemma4-cli bench-batch, E2B 4 bits, prompt 256) : un pas de decodage pour
// 8 sequences coute 22,5 ms contre 9,2 ms pour une seule, soit x3,3 de debit agrege.
//
// Prompts de longueurs differentes : completes a gauche, masques par ligne
// (`Gemma4TextModel.batchPadding`). Les caches glissants gardent toutes les positions
// (capacite >= longueur totale) pour que l'indice d'une cle reste sa position absolue.
// Lot statique : il demarre ensemble et tourne jusqu'a la fin de sa derniere ligne ; une
// ligne terminee continue d'etre calculee sans rien produire.

import Foundation
import MLX
import MLXLMCommon

public enum Gemma4BatchGeneration {

    public struct Request: Sendable {
        public var ids: [Int]
        public var maxTokens: Int
        public var temperature: Float
        public var topP: Float
        public var topK: Int

        public init(ids: [Int], maxTokens: Int, temperature: Float = 0, topP: Float = 1, topK: Int = 0) {
            self.ids = ids
            self.maxTokens = maxTokens
            self.temperature = temperature
            self.topP = topP
            self.topK = topK
        }
    }

    public enum BatchError: Error, LocalizedError, Equatable {
        case unsupportedModel
        case emptyRequest(Int)

        public var errorDescription: String? {
            switch self {
            case .unsupportedModel: return "generation par lot : modele Gemma 4 texte ou multimodal attendu"
            case .emptyRequest(let row): return "generation par lot : prompt vide (ligne \(row))"
            }
        }
    }

    /// Modele de langue d'un modele charge (texte ou multimodal, hors 12B unified).
    static func languageModel(of model: any LanguageModel) -> Gemma4LanguageModel? {
        if let m = model as? Gemma4LLMModel { return m.languageModel }
        if let m = model as? Gemma4MultimodalLLMModel { return m.languageModel }
        return nil
    }

    /// Lot accepte par ce modele ?
    public static func supports(_ model: any LanguageModel) -> Bool {
        languageModel(of: model) != nil
    }

    /// Fin d'une ligne.
    public struct RowResult: Sendable, Equatable {
        /// Jetons livres (le jeton de fin n'est pas compte, comme TokenIterator).
        public let tokens: Int
        /// Arretee par un jeton de fin.
        public let hitStopToken: Bool
    }

    /// Decode toutes les lignes ensemble. `onToken(ligne, jeton)` rend `false` pour arreter la
    /// ligne (fin de reponse cote appelant, client parti, appel d'outil) ; une ligne s'arrete
    /// aussi sur un jeton de fin ou a `maxTokens`.
    public static func run(
        model: any LanguageModel,
        requests: [Request],
        stopTokens: Set<Int> = Set(Gemma4Processor.eosTokenIds.map(Int.init)),
        onToken: (Int, Int) -> Bool
    ) throws -> [RowResult] {
        guard let lm = languageModel(of: model) else { throw BatchError.unsupportedModel }
        for (row, request) in requests.enumerated() where request.ids.isEmpty {
            throw BatchError.emptyRequest(row)
        }
        let count = requests.count
        let longest = requests.map(\.ids.count).max() ?? 0
        let padding = requests.map { longest - $0.ids.count }
        let budget = requests.map(\.maxTokens).max() ?? 0

        // Prompts completes a gauche (jeton 0 : masque, sa valeur ne compte pas).
        var flat: [Int32] = []
        flat.reserveCapacity(count * longest)
        for (request, pad) in zip(requests, padding) {
            flat += Array(repeating: 0, count: pad) + request.ids.map(Int32.init)
        }
        let prompt = MLXArray(flat).reshaped(count, longest)
        let cache: [KVCache?] = lm.makeCache(slidingCapacity: longest + budget + 1).map { $0 }
        let samplers = requests.map {
            GenerateParameters(temperature: $0.temperature, topP: $0.topP, topK: $0.topK).sampler()
        }

        // Sans padding (requete seule, prompts de meme longueur), les masques causaux
        // ordinaires suffisent : on garde le chemin d'attention standard (memes arrondis).
        lm.model.batchPadding = padding.contains { $0 > 0 } ? padding : nil
        defer { lm.model.batchPadding = nil }

        var produced = Array(repeating: 0, count: count)
        var hitStop = Array(repeating: false, count: count)
        var active = Array(repeating: true, count: count)
        // Meme decoupage que le chemin standard (`prepare` puis TokenIterator) : prefill des
        // longest-1 premieres positions, puis la derniere comme un pas de decodage. Tout
        // prefiller d'un coup change le noyau de la derniere position, donc les arrondis bf16.
        var logits: MLXArray
        if longest > 1 {
            let head = lm(inputs: prompt[0..., ..<(longest - 1)], cache: cache, logitsFrom: longest - 2)
            eval(head)
            logits = lm(inputs: prompt[0..., (longest - 1)...], cache: cache)    // [B, 1, V]
        } else {
            logits = lm(inputs: prompt, cache: cache)
        }
        while active.contains(true) {
            let last = logits[0..., -1, 0...]                                      // [B, V]
            let tokens = concatenated(
                (0 ..< count).map { samplers[$0].sample(logits: last[$0 ..< $0 + 1]).reshaped(1) }, axis: 0)
            eval(tokens)
            let values = tokens.asArray(Int32.self).map(Int.init)
            for row in 0 ..< count where active[row] {
                let token = values[row]
                // Un jeton de fin n'est ni livre ni compte (comme TokenIterator).
                if stopTokens.contains(token) {
                    hitStop[row] = true
                    active[row] = false
                    continue
                }
                produced[row] += 1
                if !onToken(row, token) || produced[row] >= requests[row].maxTokens {
                    active[row] = false
                }
            }
            guard active.contains(true), !Task.isCancelled else { break }
            logits = lm(inputs: tokens.reshaped(count, 1), cache: cache)
        }
        return zip(produced, hitStop).map { RowResult(tokens: $0, hitStopToken: $1) }
    }
}
