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

cd "$(dirname "$0")/.."

env_args=(TEST_RUNNER_SWT_EXPERIMENTAL_MAXIMUM_PARALLELIZATION_WIDTH=1)

# Relais des variables d'activation des tests d'integration (xcodebuild n'en
# transmet aucune au runner sans le prefixe TEST_RUNNER_).
if [[ -n "${GEMMA4_INTEGRATION_MODEL_PATH:-}" ]]; then
    env_args+=("TEST_RUNNER_GEMMA4_INTEGRATION_MODEL_PATH=${GEMMA4_INTEGRATION_MODEL_PATH}")
fi

exec env "${env_args[@]}" \
    xcodebuild \
        -scheme Gemma4Swift-Package \
        -destination "platform=macOS" \
        -derivedDataPath .build/xcode \
        -skipMacroValidation \
        test "$@"
