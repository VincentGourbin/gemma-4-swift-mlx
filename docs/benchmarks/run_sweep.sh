#!/bin/bash
# TurboQuant Context Sweep — Benchmark complet
# Usage: ./docs/benchmarks/run_sweep.sh
# Prerequis: gemma4-cli built en Release

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CLI="$REPO_ROOT/.build/xcode/Build/Products/Release/gemma4-cli"
RESULTS_DIR="$SCRIPT_DIR/results"
FILLER="$SCRIPT_DIR/../examples/turboquant_paper.txt"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Configs
CONTEXT_SIZES="500,1000,2000,4000,8000,16000"
KV_BITS="0,4,3"
GEN_TOKENS=200

# Modeles
E2B_PATH="$HOME/Library/Caches/models/google/gemma-4-E2B-it"
A4B_PATH="$HOME/Library/Caches/models/google/gemma-4-26B-A4B-it"

mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo "  TurboQuant Context Sweep — $(date)"
echo "  Context sizes: $CONTEXT_SIZES"
echo "  KV configs: $KV_BITS (0=Standard)"
echo "  Generated tokens: $GEN_TOKENS"
echo "============================================================"
echo ""

# Build si necessaire
if [ ! -f "$CLI" ]; then
    echo "Building gemma4-cli..."
    cd "$REPO_ROOT"
    xcodebuild -scheme gemma4-cli -configuration Release \
        -destination "platform=macOS" -derivedDataPath .build/xcode \
        -skipMacroValidation build 2>&1 | tail -1
    echo ""
fi

# E2B sweep
if [ -d "$E2B_PATH" ]; then
    echo ">>> E2B BF16 sweep"
    "$CLI" profile sweep \
        --model-path "$E2B_PATH" \
        --filler-text "$FILLER" \
        --context-sizes "$CONTEXT_SIZES" \
        --kv-bits-list "$KV_BITS" \
        --generated-tokens "$GEN_TOKENS" \
        --output "$RESULTS_DIR/e2b_sweep_${TIMESTAMP}.csv"
    echo ""
else
    echo "SKIP: E2B non trouve a $E2B_PATH"
fi

# 26B-A4B sweep
if [ -d "$A4B_PATH" ]; then
    echo ">>> 26B-A4B MoE BF16 sweep"
    "$CLI" profile sweep \
        --model-path "$A4B_PATH" \
        --filler-text "$FILLER" \
        --context-sizes "$CONTEXT_SIZES" \
        --kv-bits-list "$KV_BITS" \
        --generated-tokens "$GEN_TOKENS" \
        --output "$RESULTS_DIR/26b_a4b_sweep_${TIMESTAMP}.csv"
    echo ""
else
    echo "SKIP: 26B-A4B non trouve a $A4B_PATH"
fi

echo "============================================================"
echo "  Resultats dans: $RESULTS_DIR/"
ls -lh "$RESULTS_DIR"/*.csv 2>/dev/null || echo "  (aucun CSV genere)"
echo "============================================================"
