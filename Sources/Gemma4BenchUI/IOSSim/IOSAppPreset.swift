// Presets d'app iOS pour le pipeline agent step-by-step.
// Chaque preset = un nom + un goal "starter" + des instructions specifiques
// a l'app, injectees dans le prompt systeme (section APPLICATION-SPECIFIC
// INSTRUCTIONS) pour aider DiffusionGemma a respecter le flow particulier
// (ordres, gestes, conventions UI...) de cette app.
//
// L'utilisateur peut toujours editer goal/appContext apres avoir applique
// un preset.

import Foundation

struct IOSAppPreset: Identifiable, Hashable, Sendable {
    let id: String
    let name: String
    let icon: String     // SF Symbol pour le menu
    let category: Category
    let goal: String
    let appContext: String

    enum Category: String, CaseIterable, Sendable {
        case custom = "Apps spécifiques"
        case generic = "Patterns génériques"
    }
}

extension IOSAppPreset {
    /// Liste des presets disponibles. Pour ajouter une app : copier un bloc
    /// existant en ajustant goal + appContext.
    static let all: [IOSAppPreset] = [

        // MARK: - Apps spécifiques

        IOSAppPreset(
            id: "decryptune",
            name: "Quizz Decryptune",
            icon: "music.note.list",
            category: .custom,
            goal: """
            Réponds aux 10 questions du quiz musical Decryptune en sélectionnant à chaque fois la réponse \
            que tu trouves la plus probable, puis termine quand le résultat final est affiché.
            """,
            appContext: """
            Quiz Decryptune (10 questions). Voici les écrans que tu vas rencontrer et
            l'action à proposer pour chacun :

            ÉCRAN "Round X/10" avec question + options + bouton Valider visible :
              ACTION = tap sur l'option choisie (les options sont au milieu/bas de l'écran).

            ÉCRAN "Round X/10" avec option déjà sélectionnée (entourée/cochée) :
              ACTION = tap sur le bouton "VALIDER" (gros bouton violet en bas).

            ÉCRAN "Pas tout à fait..." ou "Bravo !" (résultat de la question) :
              ACTION = tap sur le bouton "SUIVANT" (gros bouton en bas).

            ÉCRAN de score final après la 10e question :
              ACTION = done avec un récap des questions et de tes réponses.

            Règles :
              • Une seule ACTION par génération (tap, pas de tap_then_tap).
              • Si tu ne vois pas le bouton à cliquer, c'est qu'il est sous le fold :
                ACTION = scroll_down et l'écran suivant te le montrera.
              • Si la même action a échoué 2 fois (RECENT ACTIONS) → essaye une coord
                LÉGÈREMENT différente (±0.05 en y).
              • Garde "Question X/10" dans NOTES pour le compteur.
            """
        ),

        // MARK: - Patterns génériques

        IOSAppPreset(
            id: "settings_iOS",
            name: "Naviguer dans Réglages iOS",
            icon: "gearshape",
            category: .generic,
            goal: "Ouvre Réglages > Général puis trouve le numéro de version d'iOS.",
            appContext: """
            Tu navigues dans l'app Réglages iOS standard. Pattern à suivre :
            1) Sur l'écran d'accueil, tap l'icône Réglages (icône grise avec engrenages).
            2) Dans la liste, scroll_down pour trouver "Général".
            3) Tap sur "Général". Tu arrives sur une nouvelle liste.
            4) Cherche "Informations" en haut, tap dessus.
            5) Note la "Version" iOS affichée.
            6) Émets done avec le numéro de version.

            Pour revenir en arrière, tap le bouton "< Général" / "< Réglages" en haut-gauche.
            """
        ),

        IOSAppPreset(
            id: "scrollable_form",
            name: "Formulaire long défilable",
            icon: "list.bullet.rectangle",
            category: .generic,
            goal: "Remplis le formulaire avec des données plausibles puis soumets-le.",
            appContext: """
            Cette app présente un formulaire long qui s'étend sur plusieurs écrans (scrollable).
            Pattern :
            1) Lis les champs visibles. Si un champ texte est en haut, tap_and_type avec une
               valeur plausible.
            2) Si tous les champs visibles sont remplis et qu'il y a probablement d'autres
               champs en dessous, scroll_down pour les révéler.
            3) Recommence jusqu'à voir un bouton Valider / Soumettre / Submit en bas.
            4) Tap dessus.

            Ne re-remplis JAMAIS un champ déjà rempli. Ta NOTE doit indiquer ce qui est
            déjà rempli pour ne pas se perdre.
            """
        ),

        IOSAppPreset(
            id: "paginated_quiz",
            name: "Quiz paginé générique",
            icon: "questionmark.app",
            category: .generic,
            goal: "Réponds aux questions du quiz puis termine sur l'écran de résultat final.",
            appContext: """
            Cette app est un quiz multi-questions. Pour chaque question :
            1) Lis la question et les options.
            2) Tap la réponse choisie.
            3) Cherche un bouton Valider / Suivant / Submit. S'il n'est pas visible,
               scroll_down pour le révéler.
            4) Tap le bouton.

            Pour la question suivante, si tu ne vois pas tout de suite l'énoncé, scroll_up
            pour remonter en haut. Quand l'écran de résultat final est affiché, émets done.
            """
        ),

        IOSAppPreset(
            id: "home_app_launch",
            name: "Lancer une app depuis Home",
            icon: "iphone",
            category: .generic,
            goal: "Depuis l'écran d'accueil, ouvre l'app dont le nom est dans le goal.",
            appContext: """
            Tu es sur l'écran d'accueil iOS.
            Pattern :
            1) Identifie l'icône de l'app demandée parmi les icônes visibles.
            2) Si elle est visible, tap dessus.
            3) Si elle n'est pas visible, scroll_right pour passer à la page suivante du home,
               ou scroll_up depuis le milieu de l'écran pour ouvrir la recherche Spotlight
               puis tap_and_type le nom de l'app.
            4) Dès que l'app est ouverte, émets done.
            """
        ),
    ]

    static func presets(in category: Category) -> [IOSAppPreset] {
        all.filter { $0.category == category }
    }
}
