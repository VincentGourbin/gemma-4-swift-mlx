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

    // Token IDs — multimodal (de config.json)
    public static let imageTokenId: Int32 = 258880
    public static let audioTokenId: Int32 = 258881
    public static let videoTokenId: Int32 = 258884
    public static let boiTokenId: Int32 = 255999
    public static let eoiTokenId: Int32 = 258882
    public static let boaTokenId: Int32 = 256000
    public static let eoaTokenId: Int32 = 258883

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

        var fullPrompt = "<bos>"
        if let sys = systemPrompt {
            fullPrompt += "<start_of_turn>system\n\(sys)<end_of_turn>\n"
        }
        fullPrompt += "<start_of_turn>user\n\(content)<end_of_turn>\n<start_of_turn>model\n"

        return fullPrompt
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
