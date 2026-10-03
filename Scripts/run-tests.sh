#!/usr/bin/env bash
#
# Lance la suite de tests sans la parallelisation de swift-testing.
#
# Pourquoi : les tests partagent l'etat global de MLX (MLXRandom.seed, limites
# Memory, Gemma4ComputeGate). En parallele (defaut de swift-testing), un voisin
# reensemence le generateur ou tient la garde : ~70 attentes echouent
# (GradientCheckpointingTests, LoRATrainingLoopTests, DiffusionPipelineTests).
#
# Jusqu'a mlx-swift 0.31, `xcodebuild ... test` nu se figeait meme indefiniment :
# deadlock ABBA entre CompiledFunction.call (verrou de la fonction, puis evalLock)
# et vjp / jvp (evalLock, puis fonctions compilees pendant le tracing). Corrige en
# mlx-swift 0.32 (#461) ; verifie le 2026-10-03 (trois runs paralleles, 3-7 s).
#
# Le contournement : SWT_EXPERIMENTAL_MAXIMUM_PARALLELIZATION_WIDTH=1, lu par
# swift-testing a l'execution. xcodebuild ne transmet au runner que les variables
# prefixees TEST_RUNNER_, d'ou le prefixe ci-dessous.
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
# annonce lui-meme comme EXPERIMENTAL. Un test qui se fige (deadlock, attente
# GPU) bloquerait la commande sans rien dire.
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
        echo "Un test est probablement fige (cf. l'en-tete de ce script et CLAUDE.md)." >&2
        echo "Arret du build." >&2
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
