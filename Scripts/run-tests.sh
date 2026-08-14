#!/usr/bin/env bash
#
# Lance la suite de tests sans la parallelisation de swift-testing.
#
# Pourquoi : `xcodebuild ... test` sans filtre se fige indefiniment (0% CPU apres
# ~250 tests). Ce n'est pas un test lent, c'est un deadlock ABBA dans mlx-swift —
# deux verrous pris dans des ordres opposes :
#
#   - CompiledFunction.call (Transforms+Compile.swift:39) prend d'abord le NSLock
#     de la fonction compilee, puis le evalLock global dans innerCall (ligne 89) ;
#   - vjp / jvp (Transforms.swift:31 et 68), donc tout value_and_grad, prennent
#     d'abord le evalLock global, puis rappellent des fonctions compilees pendant
#     le tracing — et redemandent le NSLock par fonction.
#
# Un thread dans un gradient (DrafterTrainingTests, LoRATests) et un thread dans
# un forward qui passe par geluApproximate — une fonction `compile`d — suffisent.
# evalLock est un NSRecursiveLock, donc rien ne casse en mono-thread : seule
# l'execution parallele de swift-testing declenche le blocage.
#
# Le contournement : SWT_EXPERIMENTAL_MAXIMUM_PARALLELIZATION_WIDTH=1, lu par
# swift-testing a l'execution. xcodebuild ne transmet au runner que les variables
# prefixees TEST_RUNNER_, d'ou le prefixe ci-dessous.
#
# A retirer quand le deadlock sera corrige en amont dans mlx-swift.
#
# Usage :
#   Scripts/run-tests.sh
#   Scripts/run-tests.sh -only-testing:Gemma4SwiftTests/WeightSanitizerTests
#
# Les tests d'integration qui exigent un modele local s'activent avec :
#   GEMMA4_INTEGRATION_MODEL_PATH=~/Library/Caches/models/mlx-community/gemma-4-e4b-it-4bit \
#     Scripts/run-tests.sh -only-testing:Gemma4SwiftTests/NoRepeatNGramIntegrationTests

set -euo pipefail

# Controle des jobs : chaque tache de fond obtient son propre groupe de process,
# ce qui permet de tuer le chien de garde *et* son `sleep` en fin de script. Sans
# ca, le sleep survit, garde stdout ouvert, et un appel du type
# `Scripts/run-tests.sh | grep ...` reste bloque jusqu'a l'expiration du delai.
set -m

cd "$(dirname "$0")/.."

env_args=(TEST_RUNNER_SWT_EXPERIMENTAL_MAXIMUM_PARALLELIZATION_WIDTH=1)

# Relais de toutes les variables GEMMA4_* (activation des tests d'integration) :
# xcodebuild ne transmet au runner que ce qui est prefixe TEST_RUNNER_. Pas de
# liste blanche, sinon la prochaine variable ajoutee serait silencieusement
# ignoree et le test correspondant sauterait en se faisant passer pour vert.
while IFS= read -r line; do
    case "$line" in
        GEMMA4_*) env_args+=("TEST_RUNNER_${line}") ;;
    esac
done < <(env)

# Garde-fou. Le contournement repose sur deux maillons non garantis : xcodebuild
# qui relaie les variables TEST_RUNNER_, et un nom de variable que swift-testing
# annonce lui-meme comme EXPERIMENTAL. Si l'un des deux lache, le deadlock
# revient et la commande se fige sans rien dire.
#
# Les options natives d'xcodebuild ne rattrapent pas ce cas : verifie le
# 2026-08-14 avec `-test-timeouts-enabled YES
# -default-test-execution-time-allowance 60`, le process etait toujours bloque
# 87 s plus tard, sans « exceeded its execution time allowance ». D'ou ce
# chien de garde en horloge murale, sur la duree totale du run (la suite passe
# en ~1,2 s, ~35 s avec les tests d'integration).
timeout_seconds="${GEMMA4_TEST_TIMEOUT:-900}"

env "${env_args[@]}" \
    xcodebuild \
        -scheme Gemma4Swift-Package \
        -destination "platform=macOS" \
        -derivedDataPath .build/xcode \
        -skipMacroValidation \
        test "$@" &
build_pid=$!

(
    sleep "${timeout_seconds}"
    if kill -0 "${build_pid}" 2>/dev/null; then
        echo "" >&2
        echo "run-tests.sh : aucun resultat apres ${timeout_seconds}s." >&2
        echo "Le contournement du deadlock mlx-swift ne s'applique probablement plus" >&2
        echo "(cf. l'en-tete de ce script et CLAUDE.md). Arret du build." >&2
        kill -TERM "${build_pid}" 2>/dev/null || true
        sleep 5
        kill -KILL "${build_pid}" 2>/dev/null || true
        echo "Un process xctest peut survivre au build : verifier avec 'pgrep xctest'." >&2
    fi
) &
watchdog_pid=$!

status=0
wait "${build_pid}" || status=$?

# Le groupe entier, pour emporter le `sleep` avec le sous-shell.
kill -- -"${watchdog_pid}" 2>/dev/null || kill "${watchdog_pid}" 2>/dev/null || true
wait "${watchdog_pid}" 2>/dev/null || true

exit "${status}"
