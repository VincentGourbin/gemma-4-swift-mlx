#!/bin/bash
# TurboQuant Context Sweep — Benchmark complet avec download/cleanup automatique
# Telecharge chaque modele, lance le sweep, puis supprime pour economiser le disque
# Usage: ./docs/benchmarks/run_sweep.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CLI="$REPO_ROOT/.build/xcode/Build/Products/Release/gemma4-cli"
RESULTS_DIR="$SCRIPT_DIR/results"
FILLER="$SCRIPT_DIR/../examples/turboquant_paper.txt"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Configs
CONTEXT_SIZES="500,1000,2000,4000,8000,16000,32000,64000,96000"
KV_BITS="0,4,3"
GEN_TOKENS=200

# Modeles a benchmarker (shortcut|display_name|estimated_gb)
MODELS=(
    "e2b-4bit|E2B 4-bit|3.6"
    "e2b-8bit|E2B 8-bit|5.2"
    "e2b-bf16|E2B BF16|10.0"
    "e4b-4bit|E4B 4-bit|5.0"
    "e4b-8bit|E4B 8-bit|8.0"
    "e4b-bf16|E4B BF16|19.0"
    "a4b-4bit|26B-A4B 4-bit|14.0"
    "a4b-8bit|26B-A4B 8-bit|27.0"
    "a4b-bf16|26B-A4B BF16|52.0"
    "31b-4bit|31B 4-bit|17.0"
    "31b-8bit|31B 8-bit|33.0"
    "31b-bf16|31B BF16|63.0"
)

mkdir -p "$RESULTS_DIR"

# Build si necessaire
if [ ! -f "$CLI" ]; then
    echo "Building gemma4-cli..."
    cd "$REPO_ROOT"
    xcodebuild -scheme gemma4-cli -configuration Release \
        -destination "platform=macOS" -derivedDataPath .build/xcode \
        -skipMacroValidation build 2>&1 | tail -1
    echo ""
fi

# RAM disponible
RAM_GB=$(sysctl -n hw.memsize | awk '{printf "%.0f", $1/1073741824}')

echo "============================================================"
echo "  TurboQuant Context Sweep — $(date)"
echo "  RAM: ${RAM_GB} Go"
echo "  Context sizes: $CONTEXT_SIZES"
echo "  KV configs: $KV_BITS (0=Standard)"
echo "  Tokens a generer: $GEN_TOKENS"
echo "  Mode: download → benchmark → delete"
echo "============================================================"
echo ""

# Filler text arg
FILLER_ARG=""
if [ -f "$FILLER" ]; then
    FILLER_ARG="--filler-text $FILLER"
fi

TOTAL_MODELS=${#MODELS[@]}
CURRENT=0
SKIPPED=0

for entry in "${MODELS[@]}"; do
    IFS='|' read -r shortcut display_name size_gb <<< "$entry"
    CURRENT=$((CURRENT + 1))

    # Verifier si le modele tient en RAM (avec marge 30%)
    NEEDED=$(echo "$size_gb * 1.3" | bc | cut -d. -f1)
    if [ "$NEEDED" -gt "$RAM_GB" ]; then
        echo "[$CURRENT/$TOTAL_MODELS] SKIP $display_name (~${size_gb} Go, besoin ~${NEEDED} Go RAM, dispo ${RAM_GB} Go)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    echo "============================================================"
    echo "[$CURRENT/$TOTAL_MODELS] $display_name (~${size_gb} Go)"
    echo "============================================================"

    # 1. Telecharger
    echo "  Telechargement..."
    "$CLI" download "$shortcut" 2>&1 | grep -E "(Telechargement|telecharge|Erreur|Go)" || true

    # 2. Trouver le path local
    MODEL_PATH=""
    CACHE_DIR="$HOME/Library/Caches/models"
    for d in "$CACHE_DIR"/models--mlx-community--*/snapshots/*/; do
        if [ -f "${d}config.json" ] && echo "$d" | grep -qi "$(echo $shortcut | sed 's/-/.*/')"; then
            MODEL_PATH="${d%/}"
            break
        fi
    done

    if [ -z "$MODEL_PATH" ]; then
        echo "  ⚠ Modele non trouve apres download, skip"
        continue
    fi

    echo "  Path: $MODEL_PATH"

    # 3. Sweep
    echo "  Lancement du sweep..."
    "$CLI" profile sweep \
        --model-path "$MODEL_PATH" \
        --context-sizes "$CONTEXT_SIZES" \
        --kv-bits-list "$KV_BITS" \
        --generated-tokens "$GEN_TOKENS" \
        $FILLER_ARG \
        --output "$RESULTS_DIR/${shortcut}_sweep_${TIMESTAMP}.csv" \
    || echo "  ⚠ Sweep interrompu (probablement OOM sur grands contextes)"

    echo ""

    # 4. Supprimer le modele pour liberer le disque
    echo "  Nettoyage du modele..."
    # Trouver et supprimer le dossier models--mlx-community--XXX complet
    REPO_DIR=$(echo "$MODEL_PATH" | grep -o '.*/models--[^/]*')
    if [ -n "$REPO_DIR" ] && [ -d "$REPO_DIR" ]; then
        du -sh "$REPO_DIR" | awk '{print "  Suppression: " $2 " (" $1 ")"}'
        rm -rf "$REPO_DIR"
    fi
    echo "  OK"
    echo ""
done

echo "============================================================"
echo "  Benchmark termine!"
echo "  Modeles testes: $((CURRENT - SKIPPED))/$TOTAL_MODELS (${SKIPPED} skip RAM)"
echo "  Resultats dans: $RESULTS_DIR/"
echo ""
ls -lh "$RESULTS_DIR"/*_${TIMESTAMP}.csv 2>/dev/null || echo "  (aucun CSV)"
echo "============================================================"
