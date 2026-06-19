// Types partages entre WebAgentLoop et AgentView.
// AgentAction = action atomique decidee par l'E4B planning.
// AgentEvent = evenement emis par la loop (thoughts, actions, observations, erreurs).
// AgentStep = trace complete d'un step (pour l'affichage de la timeline).

import AppKit
import Foundation

enum ScrollDirection: String, Codable, Sendable {
    case up, down
}

enum AgentAction: Sendable {
    case navigate(url: String)
    case click(target: String)
    case type(text: String)
    case pressEnter
    case scroll(direction: ScrollDirection)
    case done(summary: String)

    var shortLabel: String {
        switch self {
        case .navigate(let url):   return "navigate(\(url.prefix(40)))"
        case .click(let t):        return "click(\(t.prefix(40)))"
        case .type(let t):         return "type(\(t.prefix(40)))"
        case .pressEnter:          return "pressEnter"
        case .scroll(let d):       return "scroll(\(d.rawValue))"
        case .done:                return "done"
        }
    }

    var icon: String {
        switch self {
        case .navigate:  return "globe"
        case .click:     return "hand.point.up.fill"
        case .type:      return "keyboard"
        case .pressEnter: return "return"
        case .scroll:    return "arrow.up.arrow.down"
        case .done:      return "checkmark.circle.fill"
        }
    }
}

enum AgentEvent: Sendable {
    case status(String)
    case thought(String)
    case planRaw(String)
    case action(AgentAction)
    case grounding(target: String, x: Double, y: Double, elapsed: TimeInterval)
    case observation(String)
    case screenshot(NSImage)
    case error(String)
    case done(summary: String)
}

struct AgentStep: Identifiable, Sendable {
    let id = UUID()
    let n: Int
    var thought: String = ""
    var planRaw: String = ""
    var action: AgentAction? = nil
    var groundingCoords: (x: Double, y: Double)? = nil
    var groundingElapsed: TimeInterval? = nil
    var observation: String = ""
    var screenshot: NSImage? = nil
    var error: String? = nil
}

extension AgentStep: Equatable {
    static func == (lhs: AgentStep, rhs: AgentStep) -> Bool {
        lhs.id == rhs.id
    }
}
