// Processeur multimodal Gemma 4 — Gere l'expansion des tokens image/audio/video

import Foundation
import MLX
import MLXLMCommon

/// Processeur multimodal qui prepare les prompts avec les bons tokens speciaux.
/// Expand <|image|> en boi + image_token*N + eoi, et <|audio|> en boa + audio_token*N + eoa.
public struct Gemma4Processor {

    // Token strings (tels que definis dans le tokenizer.json)
    public static let boiToken = "<|image>"   // 255999
    public static let eoiToken = "<image|>"   // 258882
    public static let imageToken = "<|image|>" // 258880
    public static let boaToken = "<|audio>"   // 256000
    public static let eoaToken = "<audio|>"   // 258883
    public static let audioToken = "<|audio|>" // 258881
    public static let videoToken = "<|video|>" // 258884

    // Marqueurs de tour Gemma 4 (le format Gemma 3 <start_of_turn> n'existe pas
    // dans ce vocabulaire — il se tokeniserait en texte litteral).
    public static let bosToken = "<bos>"           // 2
    public static let turnStartToken = "<|turn>"   // 105
    public static let turnEndToken = "<turn|>"     // 106

    // Token IDs — multimodal (de config.json)
    public static let imageTokenId: Int32 = 258880
    public static let audioTokenId: Int32 = 258881
    public static let videoTokenId: Int32 = 258884
    public static let boiTokenId: Int32 = 255999
    public static let eoiTokenId: Int32 = 258882
    public static let boaTokenId: Int32 = 256000
    public static let eoaTokenId: Int32 = 258883

    // Token IDs — structure de tour
    public static let bosTokenId: Int32 = 2
    public static let turnStartTokenId: Int32 = 105
    public static let turnEndTokenId: Int32 = 106
    public static let newlineTokenId: Int32 = 107
    public static let doubleNewlineTokenId: Int32 = 108

    // Token IDs — thinking/channel (de tokenizer.json added_tokens)
    public static let thinkTokenId: Int32 = 98        // <|think|>
    public static let channelStartTokenId: Int32 = 100 // <|channel>
    public static let channelEndTokenId: Int32 = 101   // <channel|>

    // EOS tokens (de generation_config.json)
    public static let eosTokenIds: Set<Int32> = [1, 106, 50]

    /// Construit le prompt avec le chat template Gemma 4 et expand les tokens multimodaux
    /// - Parameters:
    ///   - userPrompt: le texte de l'utilisateur
    ///   - systemPrompt: prompt systeme optionnel
    ///   - hasImage: si true, insere un placeholder image
    ///   - numImageTokens: nombre de soft tokens par image (280 par defaut)
    ///   - hasAudio: si true, insere un placeholder audio
    ///   - numAudioTokens: nombre de tokens audio
    ///   - hasVideo: si true, insere un placeholder video
    ///   - numVideoFrames: nombre de frames video
    ///   - softTokensPerFrame: tokens par frame video (70 par defaut, ref Python)
    ///   - videoTimestamps: timestamps en secondes pour chaque frame video
    /// - Returns: le prompt avec les tokens expandes, pret pour la tokenisation
    public static func buildMultimodalPrompt(
        userPrompt: String,
        systemPrompt: String? = nil,
        hasImage: Bool = false,
        numImageTokens: Int = 280,
        hasAudio: Bool = false,
        numAudioTokens: Int = 0,
        hasVideo: Bool = false,
        numVideoFrames: Int = 0,
        softTokensPerFrame: Int = 70,
        videoTimestamps: [Double]? = nil
    ) -> String {
        var parts: [String] = []

        // Image: boi + image_token * N + eoi
        if hasImage {
            let imageExpanded = boiToken + String(repeating: imageToken, count: numImageTokens) + eoiToken
            parts.append(imageExpanded)
        }

        // Video: timestamp MM:SS + boi + video_token * N + eoi (ref Python)
        if hasVideo && numVideoFrames > 0 {
            for i in 0 ..< numVideoFrames {
                let ts = videoTimestamps.map { Gemma4VideoProcessor.formatTimestamp($0[i]) } ?? "00:00"
                let frameExpanded = ts + "\n" + boiToken + String(repeating: videoToken, count: softTokensPerFrame) + eoiToken
                parts.append(frameExpanded)
            }
        }

        // Audio: boa + audio_token * N + eoa
        if hasAudio && numAudioTokens > 0 {
            let audioExpanded = boaToken + String(repeating: audioToken, count: numAudioTokens) + eoaToken
            parts.append(audioExpanded)
        }

        // Texte utilisateur
        parts.append(userPrompt)

        // Construire le prompt complet avec le chat template Gemma 4
        let content = parts.joined(separator: "\n")

        // Format de tour Gemma 4 : <|turn>role\n ... <turn|>\n — les marqueurs
        // Gemma 3 (<start_of_turn>) n'existent pas dans le vocabulaire Gemma 4
        // et se tokeniseraient en texte litteral.
        var fullPrompt = bosToken
        if let sys = systemPrompt {
            fullPrompt += "\(turnStartToken)system\n\(sys)\(turnEndToken)\n"
        }
        fullPrompt += "\(turnStartToken)user\n\(content)\(turnEndToken)\n"
        fullPrompt += "\(turnStartToken)model\n"

        return fullPrompt
    }

    /// Repare les sauts de ligne parasites que swift-jinja insere dans le rendu
    /// du chat template Gemma 4.
    ///
    /// Le template Gemma 4 fait suivre le `{%- endif %}` du bloc systeme d'une
    /// ligne vide puis d'un commentaire `{#- Pre-scan ... -#}`. Le `-` initial
    /// de ce tag commentaire avale les espaces qui le precedent : jinja2 honore
    /// ce controle d'espaces sur les commentaires, swift-jinja non, et le saut
    /// de ligne survit. Verifie en rendant le meme fichier avec jinja2 — le
    /// resultat est identique aux quatre combinaisons de `trim_blocks` /
    /// `lstrip_blocks`, et le parasite n'apparait qu'en retirant le `-` du tag
    /// commentaire. Ce n'est donc pas `trim_blocks` : le correctif amont porte
    /// sur le controle d'espaces des commentaires.
    ///
    /// Une seule cause, deux symptomes selon qu'il y a un tour systeme ou non :
    ///
    /// - sans systeme : `<bos>` `\n` `<|turn>` au lieu de `<bos>` `<|turn>` ;
    /// - avec systeme : `<turn|>` `\n\n` `<|turn>` au lieu de `<turn|>` `\n`
    ///   `<|turn>`, le tokenizer fusionnant les deux sauts en un token 108.
    ///
    /// Les ids attendus sont ceux du rendu HF du meme `chat_template.jinja`,
    /// mesures token par token (voir `MultimodalSystemPromptTests`).
    public static func strippingTemplateArtifacts(_ ids: [Int]) -> [Int] {
        var out = ids

        // <bos> \n <|turn>  →  <bos> <|turn>
        if out.count >= 3, out[0] == Int(bosTokenId), out[1] == Int(newlineTokenId),
            out[2] == Int(turnStartTokenId)
        {
            out.remove(at: 1)
        }

        // <turn|> \n\n <|turn>  →  <turn|> \n <|turn>
        for i in out.indices.dropFirst().dropLast()
        where out[i] == Int(doubleNewlineTokenId)
            && out[i - 1] == Int(turnEndTokenId)
            && out[i + 1] == Int(turnStartTokenId)
        {
            out[i] = Int(newlineTokenId)
        }

        return out
    }

    /// Construit les ids d'un tour image + texte : rendu du chat template du
    /// modele, puis expansion de chaque marqueur `<|image|>` en
    /// `boi + image_token × numImageTokens + eoi`.
    ///
    /// Le placement du role systeme est delegue au template — pour Gemma 4 il
    /// rend un tour `<|turn>system ... <turn|>` distinct, il ne fusionne pas le
    /// systeme dans le premier tour utilisateur. On ne prefixe donc rien a la
    /// main : c'est le `chat_template.jinja` qui decide.
    ///
    /// - Parameters:
    ///   - userPrompt: texte du tour utilisateur (le marqueur image est ajoute
    ///     devant par cette methode).
    ///   - systemPrompt: contenu du tour systeme. `nil` = aucun tour systeme.
    ///   - templateVariables: variables passees au chat template en
    ///     `additionalContext` — p.ex. `["enable_thinking": true]`, que le
    ///     template Gemma 4 traduit par un `<|think|>` en tete du tour systeme
    ///     (qu'il cree au besoin). `nil` = rendu inchange.
    ///   - numImageTokens: soft tokens par image (280 pour Gemma 4).
    /// - Throws: `Gemma4PipelineError.invalidInput` si `systemPrompt` rend
    ///   lui-meme un marqueur multimodal — l'expansion doit rester cantonnee au
    ///   tour utilisateur, sinon `maskedScatter` recoit plus de positions a
    ///   remplir que d'embeddings disponibles.
    public static func multimodalChatIds(
        userPrompt: String,
        systemPrompt: String? = nil,
        tokenizer: any Tokenizer,
        templateVariables: [String: any Sendable]? = nil,
        numImageTokens: Int = 280
    ) throws -> [Int] {
        let userMessage = ["role": "user", "content": "\(imageToken)\n\(userPrompt)"]
        var messages: [[String: String]] = []
        if let systemPrompt {
            messages.append(["role": "system", "content": systemPrompt])
        }
        messages.append(userMessage)

        let ids = strippingTemplateArtifacts(
            try tokenizer.applyChatTemplate(
                messages: messages, tools: nil, additionalContext: templateVariables))

        // Le tour utilisateur peut legitimement porter plusieurs marqueurs (N
        // images empilees sur l'axe batch de pixelValues) ; le tour systeme,
        // jamais. On compare donc au rendu du seul tour utilisateur plutot que
        // d'imposer un marqueur unique. Le comptage couvre toutes les modalites :
        // un long prompt systeme qui documente les marqueurs du modele glisserait
        // sinon des ids speciaux bruts dans la sequence, et fausserait le compte
        // du masked_scatter si l'appelant fournit aussi de l'audio ou de la video.
        if systemPrompt != nil {
            let userOnly = try tokenizer.applyChatTemplate(
                messages: [userMessage], tools: nil, additionalContext: templateVariables)
            let markers = [
                imageTokenId, audioTokenId, videoTokenId,
                boiTokenId, eoiTokenId, boaTokenId, eoaTokenId,
            ].map(Int.init)
            let fromSystem = markers.reduce(into: 0) { total, marker in
                total += ids.count(where: { $0 == marker })
                    - userOnly.count(where: { $0 == marker })
            }
            guard fromSystem == 0 else {
                throw Gemma4PipelineError.invalidInput(
                    "systemPrompt contient \(fromSystem) marqueur(s) multimodal : "
                        + "les marqueurs doivent rester dans le tour utilisateur.")
            }
        }

        var expanded: [Int] = []
        for id in ids {
            guard id == Int(imageTokenId) else {
                expanded.append(id)
                continue
            }
            expanded.append(Int(boiTokenId))
            expanded.append(contentsOf: repeatElement(Int(imageTokenId), count: numImageTokens))
            expanded.append(Int(eoiTokenId))
        }
        return expanded
    }

    /// Equivalent texte de `multimodalChatIds` : meme rendu de chat template,
    /// memes reparations d'artefacts, sans marqueur image.
    ///
    /// Sert au chemin texte quand il doit contourner `ChatSession` — celle-ci
    /// ne sait transporter ni `LogitProcessor` ni variables de template.
    public static func textChatIds(
        userPrompt: String,
        systemPrompt: String? = nil,
        tokenizer: any Tokenizer,
        templateVariables: [String: any Sendable]? = nil
    ) throws -> [Int] {
        var messages: [[String: String]] = []
        if let systemPrompt {
            messages.append(["role": "system", "content": systemPrompt])
        }
        messages.append(["role": "user", "content": userPrompt])

        return strippingTemplateArtifacts(
            try tokenizer.applyChatTemplate(
                messages: messages, tools: nil, additionalContext: templateVariables))
    }

    /// Tokenise le prompt multimodal et retourne les input_ids
    /// Le tokenizer doit reconnaitre les tokens speciaux <|image|>, <|audio|>, etc.
    public static func tokenize(
        prompt: String,
        tokenizer: any Tokenizer
    ) -> MLXArray {
        let tokens = tokenizer.encode(text: prompt)
        return MLXArray(tokens.map { Int32($0) })
    }

    /// Verifie que les input_ids contiennent le bon nombre de tokens image/audio
    public static func validateTokenCounts(
        inputIds: MLXArray,
        expectedImageTokens: Int = 0,
        expectedAudioTokens: Int = 0
    ) -> (imageCount: Int, audioCount: Int) {
        let ids = inputIds.asType(.int32)
        let imageCount = (ids .== imageTokenId).sum().item(Int.self)
        let audioCount = (ids .== audioTokenId).sum().item(Int.self)
        return (imageCount, audioCount)
    }
}
