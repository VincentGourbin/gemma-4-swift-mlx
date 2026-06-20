// Controleur pour piloter l'iOS Simulator depuis l'app benchUI.
//
// Phase 1 (mini POC) :
//   - launchSimulator()        : open -a Simulator (lance l'app + boot le device par defaut)
//   - bootedDevice()           : xcrun simctl list devices booted -j -> nom + UDID
//   - screenshot()             : xcrun simctl io booted screenshot - --type=png (PNG en stdin)
//   - tap(normalizedX,Y)       : CGEvent au point relatif a la fenetre Simulator visible
//   - simulatorWindowFrame()   : CGWindowList pour reperer la fenetre du Simulator
//   - checkAccessibilityPermission() : verifie si notre app a le droit AX (necessaire
//                                      pour les CGEvent injectes dans une autre app focusee)
//
// Limites :
//   - L'arbre UIKit/SwiftUI de l'app iOS dans le simulator n'est PAS expose via
//     macOS AX. Donc pas d'equivalent direct de findNavigationButtons() ici.
//   - On peut quand meme demander a l'AX macOS la frame de la fenetre Simulator
//     pour convertir des coords normalisees [0,1] -> pixels ecran absolus.

import AppKit
import ApplicationServices
import Foundation
import SwiftUI

@MainActor
final class IOSSimulatorHostController: ObservableObject {

    enum SimConnectionState: Equatable {
        case unknown
        case notRunning
        case running(deviceName: String, udid: String)
        case bootedNoDevice
        case error(String)

        var statusLabel: String {
            switch self {
            case .unknown:      return "État inconnu"
            case .notRunning:   return "Simulator non lancé"
            case .running(let d, _): return "Connecté : \(d)"
            case .bootedNoDevice:    return "Simulator lancé, aucun device booté"
            case .error(let e): return "Erreur : \(e)"
            }
        }
        var color: Color {
            switch self {
            case .running: return .green
            case .bootedNoDevice: return .orange
            case .notRunning: return .gray
            case .error: return .red
            case .unknown: return .gray
            }
        }
        var isReadyForActions: Bool {
            if case .running = self { return true }
            return false
        }
    }

    @Published var connection: SimConnectionState = .unknown
    @Published var hasAccessibilityPermission: Bool = false
    @Published var lastScreenshot: NSImage?
    @Published var lastTapPoint: CGPoint? = nil  // dernier tap effectue (coords ecran)
    @Published var isBusy: Bool = false
    @Published var logLines: [String] = []

    private let simulatorBundleID = "com.apple.iphonesimulator"

    // MARK: - Logging
    private func log(_ msg: String) {
        let ts = String(format: "%.3f", Date().timeIntervalSinceReferenceDate.truncatingRemainder(dividingBy: 100_000))
        logLines.append("[\(ts)] \(msg)")
        if logLines.count > 50 { logLines.removeFirst(logLines.count - 50) }
        print("[iOSSim] \(msg)")
    }

    // MARK: - Etat / connexion

    func refreshState() async {
        await MainActor.run {
            hasAccessibilityPermission = AXIsProcessTrusted()
        }
        let running = NSWorkspace.shared.runningApplications.contains { $0.bundleIdentifier == simulatorBundleID }
        if !running {
            connection = .notRunning
            log("Simulator app not running")
            return
        }
        if let dev = await bootedDevice() {
            connection = .running(deviceName: dev.name, udid: dev.udid)
            log("Connected to \(dev.name) (\(dev.udid))")
        } else {
            connection = .bootedNoDevice
            log("Simulator app running but no device booted")
        }
    }

    /// Demande l'autorisation Accessibility (ouvre System Settings si necessaire).
    func requestAccessibilityPermission() {
        let options: [String: Any] = ["AXTrustedCheckOptionPrompt": true]
        let trusted = AXIsProcessTrustedWithOptions(options as CFDictionary)
        hasAccessibilityPermission = trusted
        log("AX permission requested: \(trusted)")
    }

    // MARK: - Lancement

    func launchSimulator() async {
        isBusy = true
        defer { isBusy = false }
        // 1. open -a Simulator (lance l'app si pas deja lancee)
        if !NSWorkspace.shared.runningApplications.contains(where: { $0.bundleIdentifier == simulatorBundleID }) {
            log("Lancement de l'app Simulator…")
            let conf = NSWorkspace.OpenConfiguration()
            do {
                if let url = NSWorkspace.shared.urlForApplication(withBundleIdentifier: simulatorBundleID) {
                    try await NSWorkspace.shared.openApplication(at: url, configuration: conf)
                } else {
                    log("⚠ Bundle Simulator introuvable")
                    connection = .error("Bundle Simulator non trouvé sur le système")
                    return
                }
            } catch {
                log("⚠ Echec open: \(error.localizedDescription)")
                connection = .error("open échoué : \(error.localizedDescription)")
                return
            }
        }
        // 2. Attendre 1s que l'app se prepare puis verifier qu'un device boot
        try? await Task.sleep(nanoseconds: 1_000_000_000)
        for _ in 0 ..< 15 {
            if let dev = await bootedDevice() {
                connection = .running(deviceName: dev.name, udid: dev.udid)
                log("Device booted : \(dev.name)")
                return
            }
            try? await Task.sleep(nanoseconds: 1_000_000_000)
        }
        // 3. Si toujours rien, tenter de booter le premier iPhone disponible
        log("Aucun device booté après 15s — tentative de boot manuel…")
        if let chosen = await firstAvailableiPhoneUDID() {
            let res = await runProcess("/usr/bin/xcrun", ["simctl", "boot", chosen])
            log("simctl boot \(chosen) -> exit \(res.exitCode)")
            try? await Task.sleep(nanoseconds: 2_000_000_000)
            if let dev = await bootedDevice() {
                connection = .running(deviceName: dev.name, udid: dev.udid)
                return
            }
        }
        connection = .bootedNoDevice
    }

    /// Retourne le device booté (premier trouvé) ou nil.
    func bootedDevice() async -> (name: String, udid: String)? {
        let res = await runProcess("/usr/bin/xcrun", ["simctl", "list", "devices", "booted", "-j"])
        guard res.exitCode == 0,
              let data = res.stdout.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let devices = json["devices"] as? [String: [[String: Any]]]
        else { return nil }
        for (_, arr) in devices {
            for entry in arr {
                if let state = entry["state"] as? String, state == "Booted",
                   let name = entry["name"] as? String,
                   let udid = entry["udid"] as? String
                {
                    return (name, udid)
                }
            }
        }
        return nil
    }

    private func firstAvailableiPhoneUDID() async -> String? {
        let res = await runProcess("/usr/bin/xcrun", ["simctl", "list", "devices", "available", "-j"])
        guard res.exitCode == 0,
              let data = res.stdout.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let devices = json["devices"] as? [String: [[String: Any]]]
        else { return nil }
        // Cherche un iPhone disponible
        for (_, arr) in devices {
            for entry in arr {
                if let name = entry["name"] as? String,
                   let udid = entry["udid"] as? String,
                   name.lowercased().contains("iphone")
                {
                    return udid
                }
            }
        }
        return nil
    }

    // MARK: - Screenshot

    /// Screenshot du device booté via simctl io booted screenshot.
    /// Passe par un fichier temporaire car sur iOS 26+ le mode stdout (-) ne
    /// fonctionne plus quand le device a plusieurs displays.
    func screenshot() async -> NSImage? {
        guard case .running = connection else {
            log("⚠ screenshot impossible : pas connecté")
            return nil
        }
        let tmpPath = "/tmp/gemma4-iosim-shot-\(UUID().uuidString).png"
        defer { try? FileManager.default.removeItem(atPath: tmpPath) }
        // On force le display "internal" pour avoir l'ecran principal du device
        // (les iPhone recents exposent aussi un external display dans simctl).
        let res = await runProcess("/usr/bin/xcrun",
                                   ["simctl", "io", "booted", "screenshot",
                                    "--type=png", "--display=internal",
                                    tmpPath])
        if res.exitCode != 0 || !FileManager.default.fileExists(atPath: tmpPath) {
            log("⚠ screenshot exit \(res.exitCode) — \(res.stderr.replacingOccurrences(of: "\n", with: " ").prefix(180))")
            return nil
        }
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: tmpPath)),
              let img = NSImage(data: data)
        else {
            log("⚠ screenshot PNG illisible (\(tmpPath))")
            return nil
        }
        lastScreenshot = img
        log("screenshot \(Int(img.size.width))×\(Int(img.size.height)) (\(data.count / 1024) KB)")
        return img
    }

    // MARK: - Frame fenêtre Simulator

    /// Cherche dans la liste des fenêtres macOS celle appartenant au Simulator,
    /// retourne sa frame écran (origin en haut-gauche, points). Permet de
    /// convertir des coords normalisées en pixels écran absolus pour les taps.
    func simulatorWindowFrame() -> CGRect? {
        let opts: CGWindowListOption = [.optionOnScreenOnly, .excludeDesktopElements]
        guard let raw = CGWindowListCopyWindowInfo(opts, kCGNullWindowID) as? [[String: Any]] else { return nil }
        for w in raw {
            guard let owner = w[kCGWindowOwnerName as String] as? String,
                  owner == "Simulator"
            else { continue }
            // Skip statusbar, ombres, etc. : on prend la plus grosse fenêtre
            if let boundsDict = w[kCGWindowBounds as String] as? [String: CGFloat] {
                let x = boundsDict["X"] ?? 0
                let y = boundsDict["Y"] ?? 0
                let w = boundsDict["Width"] ?? 0
                let h = boundsDict["Height"] ?? 0
                // Heuristique : ignore les minuscules (status bar)
                if w > 200, h > 200 {
                    return CGRect(x: x, y: y, width: w, height: h)
                }
            }
        }
        return nil
    }

    // MARK: - Tap

    /// Tap dans la zone visible de la fenêtre Simulator à (nx, ny) normalisés [0,1].
    /// Requiert que notre app ait la permission Accessibility (System Settings).
    func tap(normalizedX nx: Double, normalizedY ny: Double) async {
        guard case .running = connection else {
            log("⚠ tap impossible : pas connecté")
            return
        }
        if !hasAccessibilityPermission {
            log("⚠ tap probablement bloqué : permission Accessibility manquante")
        }
        guard let frame = simulatorWindowFrame() else {
            log("⚠ tap impossible : fenêtre Simulator introuvable")
            return
        }
        // Clamp [0, 1] : evite que le LLM produise (1.5, 2.0) et fasse cliquer
        // CGEvent en dehors de la fenetre Simulator (sur le Finder ou autre).
        let cnx = nx.isFinite ? min(1.0, max(0.0, nx)) : 0.5
        let cny = ny.isFinite ? min(1.0, max(0.0, ny)) : 0.5
        if cnx != nx || cny != ny {
            log(String(format: "⚠ tap coords clampees (%.3f,%.3f) -> (%.3f,%.3f)", nx, ny, cnx, cny))
        }
        let cx = frame.origin.x + cnx * frame.size.width
        let cy = frame.origin.y + cny * frame.size.height
        let pt = CGPoint(x: cx, y: cy)
        lastTapPoint = pt
        log(String(format: "tap norm=(%.3f,%.3f) -> screen=(%.0f,%.0f)", cnx, cny, cx, cy))

        // Activer Simulator d'abord
        if let app = NSWorkspace.shared.runningApplications.first(where: { $0.bundleIdentifier == simulatorBundleID }) {
            app.activate(options: [.activateIgnoringOtherApps])
            try? await Task.sleep(nanoseconds: 150_000_000)
        }

        // Envoie un mouseDown puis mouseUp à la position absolue
        guard let down = CGEvent(mouseEventSource: nil, mouseType: .leftMouseDown, mouseCursorPosition: pt, mouseButton: .left),
              let up   = CGEvent(mouseEventSource: nil, mouseType: .leftMouseUp,   mouseCursorPosition: pt, mouseButton: .left)
        else {
            log("⚠ CGEvent creation failed")
            return
        }
        down.post(tap: .cghidEventTap)
        try? await Task.sleep(nanoseconds: 50_000_000)
        up.post(tap: .cghidEventTap)
    }

    /// Capture composite haut+bas pour donner au modele la VUE COMPLETE de la
    /// page. Workflow :
    ///   1) screenshot du viewport courant (= zone TOP)
    ///   2) swipe vers le haut (= scroll page vers le bas)
    ///   3) screenshot de la nouvelle position (= zone BOTTOM)
    ///   4) swipe vers le bas pour remettre la page au scroll initial
    ///   5) assemble verticalement les 2 captures avec un separateur visuel
    ///      jaune labelle "═══ SCROLLED DOWN ═══"
    ///
    /// Le caller doit ensuite re-mapper les coords proposees par le modele :
    ///   y_composite < 0.5 → zone TOP, tap direct
    ///   y_composite > 0.5 → zone BOTTOM, scroll_down d'abord puis tap
    func fullPageCapture() async -> (image: NSImage?, separatorYNormalized: Double) {
        guard case .running = connection else { return (nil, 0.5) }
        guard let topShot = await screenshot() else { return (nil, 0.5) }
        // Scroll page vers le bas (doigt monte de 0.75 a 0.25)
        await swipe(fromX: 0.5, fromY: 0.75, toX: 0.5, toY: 0.25, durationMs: 220)
        try? await Task.sleep(nanoseconds: 350_000_000)
        let bottomShot = await screenshot()
        // Remet au scroll initial
        await swipe(fromX: 0.5, fromY: 0.25, toX: 0.5, toY: 0.75, durationMs: 220)
        try? await Task.sleep(nanoseconds: 200_000_000)
        guard let bottom = bottomShot else { return (topShot, 1.0) }

        // Detection : si top == bottom, le scroll n'a rien change (page courte
        // qui tient entierement dans le viewport). On utilise juste le top
        // avec sepY=1.0 (= tout le screenshot est en zone HAUT pour le mapping
        // des taps : aucun re-scroll forcé du dispatcher).
        if !pageScrolled(from: topShot, to: bottom) {
            let downscaled = downscaleForVLM(topShot, targetWidth: 720) ?? topShot
            return (downscaled, 1.0)
        }

        // Assemble verticalement + downscale pour accelerer l'inference VLM
        let composite = composeTopBottom(top: topShot, bottom: bottom)
        let downscaled = downscaleForVLM(composite, targetWidth: 720) ?? composite
        let sepY = Double(topShot.size.height + 60) / Double(topShot.size.height + 120 + bottom.size.height)
        return (downscaled, sepY)
    }

    /// Compare 2 screenshots pour determiner si le scroll a effectivement
    /// change la page. On extrait une ligne pixels au tiers superieur de chaque
    /// image (zone susceptible de bouger au scroll) et on calcule la difference
    /// moyenne. Si < seuil → la page n'a pas bouge.
    private func pageScrolled(from a: NSImage, to b: NSImage, threshold: Double = 0.03) -> Bool {
        guard let ag = a.cgImage(forProposedRect: nil, context: nil, hints: nil),
              let bg = b.cgImage(forProposedRect: nil, context: nil, hints: nil),
              ag.width == bg.width, ag.height == bg.height,
              ag.width > 0, ag.height > 100
        else { return true }
        let W = ag.width, H = ag.height
        // On compare 3 lignes (haut, milieu, bas du tiers central) pour etre
        // robuste aux animations ponctuelles.
        let rows = [H / 3, H / 2, 2 * H / 3]
        var totalDiff: Double = 0
        var totalPixels: Int = 0
        for rowY in rows {
            guard let aRow = extractRowGray(ag, atY: rowY),
                  let bRow = extractRowGray(bg, atY: rowY)
            else { continue }
            for i in 0 ..< min(aRow.count, bRow.count) {
                totalDiff += abs(Double(aRow[i]) - Double(bRow[i]))
                totalPixels += 1
            }
        }
        guard totalPixels > 0 else { return true }
        let normalized = totalDiff / (Double(totalPixels) * 255.0)
        let didScroll = normalized > threshold
        print(String(format: "[iOSSim] pageScrolled? diff=%.4f threshold=%.4f -> %@",
                     normalized, threshold, didScroll ? "YES" : "NO (page courte)"))
        return didScroll
    }

    private func extractRowGray(_ image: CGImage, atY y: Int) -> [UInt8]? {
        let W = image.width
        var buf = [UInt8](repeating: 0, count: W)
        let cs = CGColorSpaceCreateDeviceGray()
        guard let ctx = CGContext(data: &buf, width: W, height: 1,
                                  bitsPerComponent: 8, bytesPerRow: W,
                                  space: cs, bitmapInfo: CGImageAlphaInfo.none.rawValue)
        else { return nil }
        // Draw with offset to extract row y
        ctx.draw(image, in: CGRect(x: 0, y: -y, width: W, height: image.height))
        return buf
    }

    /// Resize l'image a une largeur cible (preservant l'aspect ratio) pour
    /// reduire le cout d'inference VLM. iPhone screenshot natif = 1284x2778,
    /// composite = 1284x5676. Downscaler a 720 de large reduit l'image a
    /// ~5x moins de pixels tout en gardant texte lisible.
    private func downscaleForVLM(_ image: NSImage, targetWidth: CGFloat) -> NSImage? {
        let srcW = image.size.width
        let srcH = image.size.height
        guard srcW > targetWidth else { return image }
        let scale = targetWidth / srcW
        let newSize = CGSize(width: targetWidth, height: srcH * scale)
        let out = NSImage(size: newSize)
        out.lockFocus()
        defer { out.unlockFocus() }
        NSGraphicsContext.current?.imageInterpolation = .high
        image.draw(in: CGRect(origin: .zero, size: newSize),
                   from: CGRect(origin: .zero, size: image.size),
                   operation: .copy, fraction: 1.0)
        return out
    }

    /// Empile verticalement top + separator labelle + bottom.
    private func composeTopBottom(top: NSImage, bottom: NSImage) -> NSImage {
        let W = max(top.size.width, bottom.size.width)
        let sepH: CGFloat = 120
        let H = top.size.height + sepH + bottom.size.height
        let totalSize = CGSize(width: W, height: H)
        let out = NSImage(size: totalSize)
        out.lockFocus()
        defer { out.unlockFocus() }
        // top
        top.draw(in: CGRect(x: 0, y: bottom.size.height + sepH, width: top.size.width, height: top.size.height))
        // bottom
        bottom.draw(in: CGRect(x: 0, y: 0, width: bottom.size.width, height: bottom.size.height))
        // separator : bande jaune avec texte
        NSColor(red: 1.0, green: 0.78, blue: 0.0, alpha: 0.95).setFill()
        CGRect(x: 0, y: bottom.size.height, width: W, height: sepH).fill()
        // labels
        let attrs: [NSAttributedString.Key: Any] = [
            .font: NSFont.systemFont(ofSize: 28, weight: .bold),
            .foregroundColor: NSColor.black,
        ]
        let topLabel = NSAttributedString(string: "▲ ZONE HAUT (état initial, sans scroll)", attributes: attrs)
        let bottomLabel = NSAttributedString(string: "▼ ZONE BAS (après avoir scrollé la page vers le bas)", attributes: attrs)
        let topSize = topLabel.size()
        let bottomSize = bottomLabel.size()
        topLabel.draw(at: NSPoint(x: (W - topSize.width) / 2, y: bottom.size.height + sepH - topSize.height - 8))
        bottomLabel.draw(at: NSPoint(x: (W - bottomSize.width) / 2, y: bottom.size.height + 12))
        return out
    }

    /// Swipe depuis (nx1, ny1) vers (nx2, ny2) normalises. Effectue mouseDown +
    /// plusieurs mouseDragged + mouseUp pour produire un geste fluide.
    func swipe(fromX nx1: Double, fromY ny1: Double, toX nx2: Double, toY ny2: Double, durationMs: Int = 300) async {
        guard case .running = connection, let frame = simulatorWindowFrame() else { return }
        // Clamp [0, 1] : meme protection que tap().
        let cnx1 = nx1.isFinite ? min(1.0, max(0.0, nx1)) : 0.5
        let cny1 = ny1.isFinite ? min(1.0, max(0.0, ny1)) : 0.5
        let cnx2 = nx2.isFinite ? min(1.0, max(0.0, nx2)) : 0.5
        let cny2 = ny2.isFinite ? min(1.0, max(0.0, ny2)) : 0.5
        let from = CGPoint(x: frame.origin.x + cnx1 * frame.size.width,
                           y: frame.origin.y + cny1 * frame.size.height)
        let to = CGPoint(x: frame.origin.x + cnx2 * frame.size.width,
                        y: frame.origin.y + cny2 * frame.size.height)
        log(String(format: "swipe norm (%.2f,%.2f) -> (%.2f,%.2f) screen (%.0f,%.0f) -> (%.0f,%.0f)",
                   cnx1, cny1, cnx2, cny2, from.x, from.y, to.x, to.y))

        if let app = NSWorkspace.shared.runningApplications.first(where: { $0.bundleIdentifier == simulatorBundleID }) {
            app.activate(options: [.activateIgnoringOtherApps])
            try? await Task.sleep(nanoseconds: 150_000_000)
        }

        let steps = 20
        guard let down = CGEvent(mouseEventSource: nil, mouseType: .leftMouseDown,
                                 mouseCursorPosition: from, mouseButton: .left)
        else { return }
        down.post(tap: .cghidEventTap)
        let stepNs = UInt64(max(5, durationMs / steps)) * 1_000_000
        for i in 1 ... steps {
            let t = Double(i) / Double(steps)
            let pt = CGPoint(x: from.x + (to.x - from.x) * t,
                            y: from.y + (to.y - from.y) * t)
            if let drag = CGEvent(mouseEventSource: nil, mouseType: .leftMouseDragged,
                                  mouseCursorPosition: pt, mouseButton: .left)
            {
                drag.post(tap: .cghidEventTap)
            }
            try? await Task.sleep(nanoseconds: stepNs)
        }
        if let up = CGEvent(mouseEventSource: nil, mouseType: .leftMouseUp,
                            mouseCursorPosition: to, mouseButton: .left)
        {
            up.post(tap: .cghidEventTap)
        }
    }

    /// Tape du texte sur le Simulator. Utilise AppleScript System Events qui
    /// envoie les keystrokes a l'app focusee — assure-toi qu'un champ est
    /// actif avant d'appeler. Si pressEnter, envoie aussi une touche Return.
    func type(text: String, pressEnter: Bool = false) async {
        guard case .running = connection else { return }
        if let app = NSWorkspace.shared.runningApplications.first(where: { $0.bundleIdentifier == simulatorBundleID }) {
            app.activate(options: [.activateIgnoringOtherApps])
            try? await Task.sleep(nanoseconds: 200_000_000)
        }
        // Sanitize : retire les control chars sauf `\n` (qu'on transforme en
        // key code 36 plus loin). Puis split sur les newlines pour pouvoir
        // emettre un keystroke par ligne + Return entre. Cela empeche
        // l'injection AppleScript par un texte LLM contenant `";say ":` ou
        // similaire — chaque keystroke ne contient que la ligne courante
        // proprement echappee.
        let cleaned = text.unicodeScalars.filter { scalar in
            scalar == "\n" || !(scalar.value < 0x20)
        }
        let normalized = String(String.UnicodeScalarView(cleaned))
        let lines = normalized.components(separatedBy: "\n")
        var scriptParts: [String] = []
        for (i, line) in lines.enumerated() {
            if !line.isEmpty {
                let escaped = line
                    .replacingOccurrences(of: "\\", with: "\\\\")
                    .replacingOccurrences(of: "\"", with: "\\\"")
                scriptParts.append("tell application \"System Events\" to keystroke \"\(escaped)\"")
            }
            if i < lines.count - 1 {
                // Newline entre lignes -> Return
                scriptParts.append("delay 0.08")
                scriptParts.append("tell application \"System Events\" to key code 36")
            }
        }
        if pressEnter {
            scriptParts.append("delay 0.15")
            scriptParts.append("tell application \"System Events\" to key code 36")
        }
        let script = scriptParts.joined(separator: "\n")
        let res = await runProcess("/usr/bin/osascript", ["-e", script])
        log("type \"\(text.prefix(40))\"\(pressEnter ? " ⏎" : "") -> exit \(res.exitCode)")
    }

    /// Appuie sur la touche Home (key code 4 sur iOS via simctl)
    func pressHome() async {
        let res = await runProcess("/usr/bin/xcrun", ["simctl", "io", "booted", "biometric", "match"])
        // Pas une vraie touche home — simctl n'a pas de bouton home cliquable.
        // En alternative on peut utiliser swipe up depuis le bas de l'ecran.
        if res.exitCode != 0 {
            // fallback : swipe from bottom up to go to home
            await swipe(fromX: 0.5, fromY: 0.98, toX: 0.5, toY: 0.5, durationMs: 250)
        }
    }

    // MARK: - Process helper

    private struct ProcessResult {
        let exitCode: Int32
        let stdout: String
        let stderr: String
    }
    private struct ProcessResultData {
        let exitCode: Int32
        let stdoutData: Data
        let stderr: String
    }

    private func runProcess(_ path: String, _ args: [String]) async -> ProcessResult {
        await withCheckedContinuation { cont in
            DispatchQueue.global(qos: .userInitiated).async {
                let p = Process()
                p.launchPath = path
                p.arguments = args
                let outPipe = Pipe()
                let errPipe = Pipe()
                p.standardOutput = outPipe
                p.standardError = errPipe
                do {
                    try p.run()
                    p.waitUntilExit()
                } catch {
                    cont.resume(returning: .init(exitCode: -1, stdout: "", stderr: error.localizedDescription))
                    return
                }
                let out = outPipe.fileHandleForReading.readDataToEndOfFile()
                let err = errPipe.fileHandleForReading.readDataToEndOfFile()
                cont.resume(returning: .init(
                    exitCode: p.terminationStatus,
                    stdout: String(data: out, encoding: .utf8) ?? "",
                    stderr: String(data: err, encoding: .utf8) ?? ""
                ))
            }
        }
    }

    private func runProcessData(_ path: String, _ args: [String]) async -> ProcessResultData {
        await withCheckedContinuation { cont in
            DispatchQueue.global(qos: .userInitiated).async {
                let p = Process()
                p.launchPath = path
                p.arguments = args
                let outPipe = Pipe()
                let errPipe = Pipe()
                p.standardOutput = outPipe
                p.standardError = errPipe
                do {
                    try p.run()
                    p.waitUntilExit()
                } catch {
                    cont.resume(returning: .init(exitCode: -1, stdoutData: Data(), stderr: error.localizedDescription))
                    return
                }
                let out = outPipe.fileHandleForReading.readDataToEndOfFile()
                let err = errPipe.fileHandleForReading.readDataToEndOfFile()
                cont.resume(returning: .init(
                    exitCode: p.terminationStatus,
                    stdoutData: out,
                    stderr: String(data: err, encoding: .utf8) ?? ""
                ))
            }
        }
    }
}
