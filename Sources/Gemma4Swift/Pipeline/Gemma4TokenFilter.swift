// Filtrage du thinking mode Gemma 4
// Gere les tokens <|think|>, <|channel>thought et <|channel>response

import Foundation

/// Mode de gestion du thinking de Gemma 4
public enum Gemma4ThinkingMode: Sendable {
    /// Filtre automatiquement les blocs thinking — seule la reponse est emise (defaut)
    case disabled
    /// Laisse passer tous les tokens bruts (thinking + response)
    case enabled
    /// Separe thinking et response dans des objets distincts
    case structured
}

/// Reponse structuree separant thinking et reponse
public struct Gemma4StructuredResponse: Sendable {
    /// Contenu du bloc thinking (nil si pas de thinking)
    public let thinking: String?
    /// Contenu de la reponse finale
    public let response: String
}

/// Filtre stateful pour le stream de tokens Gemma 4.
/// Detecte les blocs `<|channel>thought` et `<|channel>response` et filtre selon le mode choisi.
public final class Gemma4TokenFilter: @unchecked Sendable {
    public let mode: Gemma4ThinkingMode

    /// Canal courant detecte
    enum Channel {
        case none       // Pas encore dans un canal
        case thinking   // Dans <|channel>thought...
        case response   // Dans <|channel>response... (ou hors canal)
        case detecting  // Juste apres <|channel>, en attente du nom du canal
    }

    private var channel: Channel = .none
    private var thinkingTokens: [String] = []
    private var responseTokens: [String] = []
    private var pendingText: String = ""

    /// Nombre de tokens de thinking generes (filtres)
    public var thinkingTokenCount: Int { thinkingTokens.count }
    /// Nombre de tokens de reponse visibles
    public var responseTokenCount: Int { responseTokens.count }

    public init(mode: Gemma4ThinkingMode = .disabled) {
        self.mode = mode
    }

    /// Traite un token genere et retourne le texte a afficher (peut etre vide si filtre)
    /// - Parameters:
    ///   - tokenId: ID du token genere
    ///   - text: texte decode du token
    /// - Returns: texte a afficher a l'utilisateur (vide si filtre)
    public func process(tokenId: Int32, text: String) -> String {
        switch mode {
        case .enabled:
            return text

        case .disabled, .structured:
            return filterToken(tokenId: tokenId, text: text)
        }
    }

    /// Retourne true si le token est un EOS
    public func isEOS(_ tokenId: Int32) -> Bool {
        Gemma4Processor.eosTokenIds.contains(tokenId)
    }

    /// Construit la reponse structuree finale (pour mode .structured)
    public func structuredResponse() -> Gemma4StructuredResponse {
        let thinking = thinkingTokens.isEmpty ? nil : thinkingTokens.joined()
        let response = responseTokens.joined()
        return Gemma4StructuredResponse(thinking: thinking, response: response)
    }

    // MARK: - Private

    private func filterToken(tokenId: Int32, text: String) -> String {
        // Detecter les tokens speciaux de canal
        if tokenId == Gemma4Processor.channelStartTokenId {
            // <|channel> — passer en mode detection du nom de canal
            channel = .detecting
            pendingText = ""
            return ""
        }

        if tokenId == Gemma4Processor.channelEndTokenId {
            // <channel|> — fin du canal courant
            channel = .none
            return ""
        }

        if tokenId == Gemma4Processor.thinkTokenId {
            // <|think|> — active le thinking (le prochain <|channel> sera thought)
            return ""
        }

        // Si on detecte le nom du canal
        if channel == .detecting {
            pendingText += text
            if pendingText.contains("thought") {
                channel = .thinking
                pendingText = ""
                return ""
            } else if pendingText.contains("response") {
                channel = .response
                pendingText = ""
                return ""
            }
            // Pas encore assez de texte pour identifier le canal
            // Si on a accumule trop sans match, traiter comme response
            if pendingText.count > 20 {
                channel = .response
                let buffered = pendingText
                pendingText = ""
                responseTokens.append(buffered)
                return mode == .disabled ? buffered : ""
            }
            return ""
        }

        // Router selon le canal courant
        switch channel {
        case .thinking:
            thinkingTokens.append(text)
            return "" // Toujours masque en mode disabled, capture en structured

        case .response, .none:
            responseTokens.append(text)
            return text

        case .detecting:
            return "" // Ne devrait pas arriver
        }
    }
}
