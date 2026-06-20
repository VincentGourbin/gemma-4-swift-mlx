// Prompts pour l'agent web :
//   - E4B planning : decide la prochaine action JSON, voit le screenshot
//   - Diffusion grounding : convertit "click on X" en coords (x, y)
//
// On reutilise le format ScreenSpot pour le grounding (qui a atteint 79% global
// dans nos benchs ScreenSpot v1 100 cas, 100% sur ios/text).

import Foundation

enum AgentPrompts {

    /// Prompt E4B planning. L'image est passee separement via pixelValues.
    /// `pageText` (~2k chars) aide E4B a "lire" la page (le vision encoder
    /// 280 soft tokens voit la mise en page mais pas tous les details texte).
    static func e4bPlanning(
        goal: String,
        currentURL: String,
        pageTitle: String,
        history: [String],
        pageText: String
    ) -> String {
        let historyBlock = history.isEmpty
            ? "(no actions yet — this is the first step)"
            : history.enumerated().map { idx, h in "  \(idx + 1). \(h)" }.joined(separator: "\n")

        // Detecter une boucle : derniere(s) action(s) identique(s) repetees
        let lastN = history.suffix(3)
        let isLooping = lastN.count >= 2 && lastN.allSatisfy { line in
            line.hasPrefix(lastN.first ?? "")
                || (lastN.first?.hasPrefix("scrolled") == true && line.hasPrefix("scrolled"))
        }
        let loopWarning = isLooping
            ? "\n⚠ LOOP DETECTED: your last actions are repetitive. Pick a DIFFERENT action this turn (do not scroll again).\n"
            : ""

        let trimmedText = pageText
            .replacingOccurrences(of: "\n\n", with: "\n")
            .replacingOccurrences(of: "\n\n", with: "\n")

        return """
        You are a web navigation agent. Your overall goal is:
        GOAL: \(goal)

        Current page URL: \(currentURL)
        Current page title: \(pageTitle.isEmpty ? "(unknown)" : pageTitle)

        Actions already taken:
        \(historyBlock)
        \(loopWarning)
        Visible text of the page (first 2000 chars, may help reading dense lists):
        ---
        \(trimmedText.prefix(2000))
        ---

        You are looking at a screenshot of the current page in a browser viewport.
        Decide the SINGLE NEXT action to take. Pick the most direct action that
        makes progress toward the goal.

        IMPORTANT decision rules — read carefully:
        1. If the visible text or screenshot ALREADY contains enough information
           to answer the GOAL, emit {"action":"done","summary":"..."} with a
           thorough synthesis. DO NOT keep scrolling once you have the answer.
        2. If a search input is visible and useful, prefer clicking it, then
           typing, then pressEnter — that beats scrolling through long lists.
        3. NEVER repeat scroll more than twice in a row. If your last 2 actions
           were both scroll, you MUST pick another action (done / click / navigate).
        4. Click targets MUST be visible. Describe them precisely enough for a
           vision model to locate (e.g. "Gemma 4 12B model card link in the list",
           "search bar at top of page with placeholder 'Search models'").

        Available actions (respond with exactly ONE JSON object on a single line,
        no extra text, no markdown fences):

          {"action":"navigate","url":"https://..."}
          {"action":"click","target":"short description of what to click"}
          {"action":"type","text":"text to type into the currently focused input"}
          {"action":"pressEnter"}
          {"action":"scroll","direction":"down"}    // or "up"
          {"action":"done","summary":"final answer / synthesis here"}

        Output ONLY the JSON, on one line, nothing else.
        """
    }

    /// Prompt grounding pour DiffusionGemma : meme format que le bench
    /// ScreenSpot qui a donne 79% (100% ios/text).
    static func diffusionGrounding(target: String, imageWidth: Int, imageHeight: Int) -> String {
        return """
        Look at this UI screenshot (\(imageWidth)x\(imageHeight) pixels).
        Goal: click on \(target).

        Where on the screen should I click to accomplish this goal? Respond with the predicted click position in normalized coordinates between 0 and 1, in the exact format:
          CLICK: (x=0.XX, y=0.XX)

        where x is the horizontal position (0=left, 1=right) and y is the vertical position (0=top, 1=bottom). Reply with only the CLICK: line, no extra explanation.
        """
    }

    // MARK: - Parsing

    /// Extrait la premiere occurrence d'un objet JSON {"action":...} dans le texte
    /// genere par E4B (qui peut entourer son JSON de prose ou de fences).
    static func extractActionJSON(from raw: String) -> String? {
        let s = raw
            .replacingOccurrences(of: "```json", with: "")
            .replacingOccurrences(of: "```", with: "")
        guard let start = s.firstIndex(of: "{") else { return nil }
        // Balance les accolades pour trouver la fin de l'objet
        var depth = 0
        var i = start
        var inString = false
        var escape = false
        while i < s.endIndex {
            let c = s[i]
            if escape { escape = false; i = s.index(after: i); continue }
            if c == "\\" { escape = true; i = s.index(after: i); continue }
            if c == "\"" { inString.toggle() }
            if !inString {
                if c == "{" { depth += 1 }
                if c == "}" {
                    depth -= 1
                    if depth == 0 {
                        return String(s[start ... i])
                    }
                }
            }
            i = s.index(after: i)
        }
        return nil
    }

    /// Decode un AgentAction depuis le JSON extrait.
    static func parseAction(from json: String) -> AgentAction? {
        guard let data = json.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let action = obj["action"] as? String
        else { return nil }

        switch action.lowercased() {
        case "navigate":
            if let url = obj["url"] as? String, !url.isEmpty {
                return .navigate(url: url)
            }
        case "click":
            if let target = obj["target"] as? String, !target.isEmpty {
                return .click(target: target)
            }
        case "type":
            if let text = obj["text"] as? String {
                return .type(text: text)
            }
        case "pressenter", "press_enter", "enter":
            return .pressEnter
        case "scroll":
            let dir = (obj["direction"] as? String)?.lowercased() ?? "down"
            return .scroll(direction: dir == "up" ? .up : .down)
        case "done", "finish":
            let summary = obj["summary"] as? String ?? "(no summary)"
            return .done(summary: summary)
        default:
            return nil
        }
        return nil
    }

    /// Parse la sortie Diffusion `CLICK: (x=0.XX, y=0.XX)` -> coordonnees normalisees [0, 1].
    static func parseClickCoords(from raw: String) -> (x: Double, y: Double)? {
        let cleaned = raw
            .replacingOccurrences(of: "<eos>", with: " ")
            .replacingOccurrences(of: "<turn|>", with: " ")
            .replacingOccurrences(of: "<|turn>", with: " ")

        // Pattern principal : CLICK: (x=0.XX, y=0.XX)
        let primary = try? NSRegularExpression(
            pattern: #"CLICK\s*:?\s*\(?\s*x\s*=\s*([0-9.]+)\s*,\s*y\s*=\s*([0-9.]+)"#,
            options: [.caseInsensitive]
        )
        if let m = primary?.firstMatch(in: cleaned, range: NSRange(cleaned.startIndex..., in: cleaned)),
           let xr = Range(m.range(at: 1), in: cleaned),
           let yr = Range(m.range(at: 2), in: cleaned),
           let x = Double(cleaned[xr]),
           let y = Double(cleaned[yr]),
           x <= 1 && y <= 1
        {
            return (x, y)
        }

        // Fallback : x=0.XX, y=0.XX (sans CLICK:)
        let fallback = try? NSRegularExpression(
            pattern: #"x\s*[=:]\s*([0-9.]+)\s*,\s*y\s*[=:]\s*([0-9.]+)"#,
            options: [.caseInsensitive]
        )
        if let m = fallback?.firstMatch(in: cleaned, range: NSRange(cleaned.startIndex..., in: cleaned)),
           let xr = Range(m.range(at: 1), in: cleaned),
           let yr = Range(m.range(at: 2), in: cleaned),
           let x = Double(cleaned[xr]),
           let y = Double(cleaned[yr]),
           x <= 1 && y <= 1
        {
            return (x, y)
        }
        return nil
    }
}
