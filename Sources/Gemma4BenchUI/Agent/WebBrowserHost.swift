// Wrapper WKWebView pour l'agent web : navigation, screenshot, click JS,
// scroll, evaluation JavaScript arbitraire. Toutes les operations sont
// @MainActor (WKWebView est AppKit-bound).

import AppKit
import SwiftUI
import WebKit

@MainActor
final class WebBrowserHostController: ObservableObject {
    let webView: WKWebView
    @Published var currentURL: String = "https://huggingface.co/models"
    @Published var isLoading: Bool = false
    @Published var pageTitle: String = ""

    private let navDelegate = NavDelegate()

    init() {
        let config = WKWebViewConfiguration()
        // User agent desktop classique pour eviter les pages mobiles
        config.applicationNameForUserAgent = "Gemma4Agent/1.0"
        webView = WKWebView(frame: .zero, configuration: config)
        webView.navigationDelegate = navDelegate
        navDelegate.owner = self
    }

    func navigate(to urlString: String) {
        var s = urlString.trimmingCharacters(in: .whitespacesAndNewlines)
        if !s.hasPrefix("http://") && !s.hasPrefix("https://") {
            s = "https://" + s
        }
        guard let url = URL(string: s) else { return }
        currentURL = s
        webView.load(URLRequest(url: url))
    }

    /// Screenshot du viewport courant. Retourne un NSImage non-Retina (1024 px
    /// de large) pour passer ensuite via Gemma4ImageProcessor (qui re-resize a
    /// la taille SigLIP attendue).
    func screenshot(width: Int = 1024) async -> NSImage? {
        await withCheckedContinuation { cont in
            let config = WKSnapshotConfiguration()
            config.snapshotWidth = NSNumber(value: width)
            webView.takeSnapshot(with: config) { image, _ in
                cont.resume(returning: image)
            }
        }
    }

    /// Click via JS sur l'element situe en (nx, ny) normalises dans le
    /// viewport. Sequence : pointerdown -> mousedown -> pointerup -> mouseup ->
    /// click + focus() explicite. Si l'element cible n'est pas focusable, on
    /// remonte/descend dans l'arbre pour trouver le plus proche input/button/a.
    /// Retourne tagName#id.class :: text pour debug.
    @discardableResult
    func click(normalizedX nx: Double, normalizedY ny: Double) async -> String? {
        let cssW = Double(webView.frame.width)
        let cssH = Double(webView.frame.height)
        let cx = max(0, min(cssW, nx * cssW))
        let cy = max(0, min(cssH, ny * cssH))
        let js = """
        (function() {
            var hit = document.elementFromPoint(\(cx), \(cy));
            if (!hit) return null;

            // Heuristique : si hit n'est pas focusable, cherche un descendant
            // direct focusable (un wrapper div autour d'un input par exemple),
            // sinon remonte vers les ancetres proches.
            function isFocusable(n) {
                if (!n || !n.tagName) return false;
                var t = n.tagName;
                if (t === 'INPUT' || t === 'TEXTAREA' || t === 'SELECT'
                    || t === 'BUTTON' || t === 'A') return true;
                if (n.isContentEditable) return true;
                if (typeof n.tabIndex === 'number' && n.tabIndex >= 0) return true;
                return false;
            }
            var target = hit;
            if (!isFocusable(target)) {
                var inner = target.querySelector
                    ? target.querySelector('input,textarea,select,button,a,[contenteditable=""],[contenteditable="true"],[tabindex]')
                    : null;
                if (inner) {
                    target = inner;
                } else {
                    var p = target.parentElement;
                    for (var i = 0; i < 4 && p; i++) {
                        if (isFocusable(p)) { target = p; break; }
                        p = p.parentElement;
                    }
                }
            }

            // Sequence d'events (PointerEvent si supporte, sinon MouseEvent seul)
            var opts = { bubbles: true, cancelable: true, view: window,
                         clientX: \(cx), clientY: \(cy), button: 0, buttons: 1 };
            function fire(type, useMouse) {
                try {
                    if (!useMouse && typeof PointerEvent === 'function') {
                        var po = Object.assign({}, opts, {
                            pointerId: 1, pointerType: 'mouse', isPrimary: true
                        });
                        target.dispatchEvent(new PointerEvent(type, po));
                    } else {
                        target.dispatchEvent(new MouseEvent(type, opts));
                    }
                } catch (e) {
                    target.dispatchEvent(new MouseEvent(type, opts));
                }
            }
            fire('pointerdown', false);
            fire('mousedown',   true);
            fire('pointerup',   false);
            fire('mouseup',     true);
            fire('click',       true);

            // Focus explicite pour debloquer les inputs React/custom
            if (typeof target.focus === 'function') {
                try { target.focus({ preventScroll: false }); } catch (e) { target.focus(); }
            }

            return (target.tagName || '') + '#' + (target.id || '')
                + '.' + ((target.className || '') + '').slice(0, 80)
                + ' :: ' + ((target.innerText || target.value || '') + '').slice(0, 80);
        })();
        """
        return (try? await webView.evaluateJavaScript(js)) as? String
    }

    func scroll(deltaY: Double) async {
        _ = try? await webView.evaluateJavaScript("window.scrollBy(0, \(deltaY));")
    }

    /// Tape du texte. Si activeElement n'est pas un input/textarea/contenteditable,
    /// auto-focus le premier input "search-like" visible (type=search, role=search,
    /// aria-label/placeholder contenant 'search') avant de typer. Retourne une
    /// description courte de ce qui a ete tape (ou nil si echec).
    @discardableResult
    func type(text: String) async -> String? {
        // Encode le texte en literal JS valide via JSONSerialization. Empeche
        // toute injection : un texte produit par le LLM qui contiendrait des
        // guillemets, newlines, control chars ou `";alert(1);x="` est traite
        // comme une simple string.
        let userTextLiteral = Self.jsStringLiteral(text)
        let js = """
        (function() {
            var __t = \(userTextLiteral);
            function isTextInput(el) {
                if (!el || !el.tagName) return false;
                if (el.tagName === 'TEXTAREA') return true;
                if (el.isContentEditable) return true;
                if (el.tagName === 'INPUT') {
                    var t = (el.type || 'text').toLowerCase();
                    return ['text','search','email','url','tel','password','number',''].indexOf(t) >= 0;
                }
                return false;
            }
            function isVisible(el) {
                if (!el || !el.getBoundingClientRect) return false;
                var r = el.getBoundingClientRect();
                if (r.width < 1 || r.height < 1) return false;
                var st = window.getComputedStyle(el);
                if (st.display === 'none' || st.visibility === 'hidden' || st.opacity === '0') return false;
                return true;
            }

            var target = document.activeElement;
            if (!isTextInput(target) || !isVisible(target)) {
                // Auto-focus : preference search > visible input > body
                var candidates = Array.from(document.querySelectorAll(
                    'input[type=search], input[role=search], [role=search] input, ' +
                    'input[aria-label*=search i], input[placeholder*=search i], ' +
                    'input[aria-label*=recherch i], input[placeholder*=recherch i]'
                ));
                candidates = candidates.filter(isVisible);
                if (candidates.length === 0) {
                    // Fallback : tous les inputs texte visibles
                    candidates = Array.from(document.querySelectorAll('input, textarea, [contenteditable]'))
                        .filter(isTextInput).filter(isVisible);
                }
                if (candidates.length === 0) {
                    return 'NO_INPUT_FOUND';
                }
                target = candidates[0];
                try { target.focus({preventScroll: false}); } catch (e) { target.focus(); }
            }

            // Append text + dispatch events
            if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA') {
                var cur = target.value || '';
                target.value = cur + __t;
                target.dispatchEvent(new InputEvent('input', { bubbles: true, data: __t, inputType: 'insertText' }));
                target.dispatchEvent(new Event('change', { bubbles: true }));
                return 'TYPED_INPUT:' + ((target.placeholder || target.name || target.id || target.type || 'input') + '').slice(0,40);
            }
            document.execCommand('insertText', false, __t);
            return 'TYPED_CONTENTEDITABLE';
        })();
        """
        return (try? await webView.evaluateJavaScript(js)) as? String
    }

    /// Encode `s` comme un literal JS string echappe (guillemets compris).
    /// Implementation via JSONSerialization : produit `"..."` avec tous les
    /// caracteres speciaux (newlines, control chars, guillemets) correctement
    /// escapes selon la spec JSON, qui est un sous-ensemble strict de la spec
    /// JS pour les string literals.
    private static func jsStringLiteral(_ s: String) -> String {
        guard let data = try? JSONSerialization.data(withJSONObject: [s], options: []),
              let outer = String(data: data, encoding: .utf8),
              outer.count >= 2
        else {
            // Fallback ultra-defensif : double-quote vide
            return "\"\""
        }
        // outer = `["..."]` -> strip [...] pour ne garder que `"..."`
        return String(outer.dropFirst().dropLast())
    }

    func pressEnter() async {
        let js = """
        (function() {
            var el = document.activeElement;
            if (!el) return false;
            ['keydown','keypress','keyup'].forEach(function(t) {
                el.dispatchEvent(new KeyboardEvent(t, {
                    key: 'Enter', code: 'Enter', keyCode: 13, which: 13,
                    bubbles: true, cancelable: true
                }));
            });
            if (el.form) el.form.requestSubmit ? el.form.requestSubmit() : el.form.submit();
            return true;
        })();
        """
        _ = try? await webView.evaluateJavaScript(js)
    }

    /// Texte visible de la page (best-effort) — peut etre tronque par l'appelant.
    func pageText(maxChars: Int = 8_000) async -> String {
        let js = "(document.body && document.body.innerText) || ''"
        let raw = (try? await webView.evaluateJavaScript(js)) as? String ?? ""
        return String(raw.prefix(maxChars))
    }

    struct NavButton: Sendable {
        let label: String
        let x: Double  // normalise [0, 1]
        let y: Double
    }

    /// Detecte les boutons de navigation visibles du DOM (Suivant/Next/Submit/
    /// Valider/Continuer...) et retourne leur label + coords normalisees au
    /// centre. Permet d'injecter cette liste dans le prompt du modele, qui
    /// peut alors cliquer sans avoir a grounder visuellement le bouton.
    func findNavigationButtons() async -> [NavButton] {
        let js = """
        (function() {
            var keywords = [
                'suivant','next','continuer','continue','submit','valider','validate',
                'envoyer','send','ok','terminer','finish','done','soumettre','confirmer',
                'precedent','previous','back','retour','annuler','cancel','passer','skip'
            ];
            var W = window.innerWidth, H = window.innerHeight;
            function visible(el) {
                var r = el.getBoundingClientRect();
                if (r.width < 4 || r.height < 4) return false;
                var st = window.getComputedStyle(el);
                if (st.display === 'none' || st.visibility === 'hidden' || parseFloat(st.opacity) < 0.1) return false;
                if (r.right < 0 || r.bottom < 0 || r.left > W || r.top > H) return false;
                return true;
            }
            var seen = new Set();
            var out = [];
            var nodes = document.querySelectorAll('button, a, [role=button], [role=link], input[type=submit], input[type=button]');
            for (var i = 0; i < nodes.length; i++) {
                var el = nodes[i];
                if (!visible(el)) continue;
                var raw = (el.innerText || el.value || el.getAttribute('aria-label') || el.title || '').trim();
                if (!raw) continue;
                var low = raw.toLowerCase();
                var matched = null;
                for (var k = 0; k < keywords.length; k++) {
                    if (low === keywords[k] || low.indexOf(keywords[k]) >= 0 && low.length < 40) {
                        matched = keywords[k];
                        break;
                    }
                }
                if (!matched) continue;
                var r = el.getBoundingClientRect();
                var cx = (r.left + r.width / 2) / W;
                var cy = (r.top + r.height / 2) / H;
                var key = raw + '|' + Math.round(cx * 100) + '|' + Math.round(cy * 100);
                if (seen.has(key)) continue;
                seen.add(key);
                out.push({ label: raw.slice(0, 40), x: Math.round(cx * 1000) / 1000, y: Math.round(cy * 1000) / 1000 });
            }
            return JSON.stringify(out);
        })();
        """
        guard let raw = (try? await webView.evaluateJavaScript(js)) as? String,
              let data = raw.data(using: .utf8),
              let arr = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]]
        else { return [] }
        return arr.compactMap { obj in
            guard let label = obj["label"] as? String,
                  let x = obj["x"] as? Double,
                  let y = obj["y"] as? Double
            else { return nil }
            return NavButton(label: label, x: x, y: y)
        }
    }

    /// Vide les données persistantes du WKWebView : cache disque, cookies,
    /// localStorage, sessionStorage, IndexedDB, ServiceWorker, etc. Utile pour
    /// repartir d'une session vierge (ex : page "you have been blocked",
    /// cookies coincés, session expirée). Retourne le nombre de types de
    /// données effacees pour info.
    @discardableResult
    func clearAllBrowserData() async -> Int {
        let dataStore = webView.configuration.websiteDataStore
        let allTypes = WKWebsiteDataStore.allWebsiteDataTypes()
        await dataStore.removeData(ofTypes: allTypes, modifiedSince: Date(timeIntervalSince1970: 0))
        // Vide aussi la pile de navigation pour eviter de revenir sur une page
        // potentiellement cassee via le back/forward.
        webView.backForwardList.perform(NSSelectorFromString("_removeAllItems"))
        // Recharge sur about:blank pour visualiser que c'est repartit propre
        if let url = URL(string: "about:blank") {
            webView.load(URLRequest(url: url))
        }
        return allTypes.count
    }

    /// Tente de fermer un bandeau cookies / GDPR modal en cliquant sur le bouton
    /// de refus le plus probable. Best-effort — couvre les patterns frequents
    /// (texte FR + EN, attributs aria, ids/classes typiques). Retourne le label
    /// de l'element clique ou nil si rien trouve.
    @discardableResult
    func tryDismissCookieBanner() async -> String? {
        let js = """
        (function() {
            function visible(el) {
                if (!el) return false;
                var r = el.getBoundingClientRect();
                if (r.width < 1 || r.height < 1) return false;
                var st = window.getComputedStyle(el);
                if (st.display === 'none' || st.visibility === 'hidden' || st.opacity === '0') return false;
                return true;
            }
            // Tente une liste de regex sur le texte des boutons visibles
            var prefers = [
                /tout\\s*refuser/i, /refuser\\s*tout/i, /reject\\s*all/i, /refuse\\s*all/i,
                /\\brefuser\\b/i, /\\bdecline\\b/i, /\\bdécliner\\b/i, /\\bdeny\\b/i,
                /continuer\\s*sans\\s*accepter/i, /continue\\s*without\\s*accepting/i,
                /non\\s*merci/i, /no\\s*thanks/i,
                /tout\\s*accepter/i, /accept\\s*all/i, /j[''’]?accepte/i, /\\bok\\b/i
            ];
            var candidates = Array.prototype.slice.call(
                document.querySelectorAll('button, a, [role=button], input[type=button], input[type=submit], [class*=consent i] *, [id*=consent i] *')
            ).filter(visible);
            for (var p = 0; p < prefers.length; p++) {
                var re = prefers[p];
                for (var i = 0; i < candidates.length; i++) {
                    var el = candidates[i];
                    var txt = (el.innerText || el.value || el.getAttribute('aria-label') || '').trim();
                    if (txt && re.test(txt)) {
                        try { el.click(); } catch (e) {}
                        return 'cookie-dismiss[' + p + ']:' + txt.slice(0, 60);
                    }
                }
            }
            return null;
        })();
        """
        return (try? await webView.evaluateJavaScript(js)) as? String
    }
}

/// Bridge SwiftUI -> WKWebView.
struct WebBrowserView: NSViewRepresentable {
    let host: WebBrowserHostController
    func makeNSView(context: Context) -> WKWebView { host.webView }
    func updateNSView(_ nsView: WKWebView, context: Context) {}
}

// MARK: - Navigation delegate (loading state)

private final class NavDelegate: NSObject, WKNavigationDelegate {
    weak var owner: WebBrowserHostController?

    func webView(_ webView: WKWebView, didStartProvisionalNavigation navigation: WKNavigation!) {
        Task { @MainActor in self.owner?.isLoading = true }
    }

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        Task { @MainActor in
            self.owner?.isLoading = false
            self.owner?.pageTitle = webView.title ?? ""
            self.owner?.currentURL = webView.url?.absoluteString ?? self.owner?.currentURL ?? ""
        }
    }

    func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
        Task { @MainActor in self.owner?.isLoading = false }
    }
}
