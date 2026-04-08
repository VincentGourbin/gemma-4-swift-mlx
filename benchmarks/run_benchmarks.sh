#!/usr/bin/env bash
# Benchmark complet Gemma 4 — toutes modalités, tous modèles d'une quantisation
# Usage: ./benchmarks/run_benchmarks.sh <quant> [model_filter]
# Ex:    ./benchmarks/run_benchmarks.sh 4bit
#        ./benchmarks/run_benchmarks.sh 8bit e2b

set -e

QUANT="${1:?Usage: $0 <quant> [model_filter]}"
FILTER="${2:-}"
CLI=".build/xcode/Build/Products/Release/gemma4-cli"
CACHE="$HOME/Library/Caches/models/mlx-community"
OUTDIR="benchmarks/results/${QUANT}"
mkdir -p "$OUTDIR"

# Images/audio de test
IMG="input_sample.jpg"
VIDEO_SRC="/Users/vincent/Pictures/FluxforgeStudio/Forge/un logo pour mon app_DF1EAD95/step_3797B1A8.mp4"
AUDIO_SRC="/Users/vincent/Downloads/Audio/Audio - Other/Audio_Obama.mp3"

# Copier la vidéo temporairement si elle existe
if [ -f "$VIDEO_SRC" ]; then
    cp "$VIDEO_SRC" /tmp/bench_video.mp4
    VIDEO="/tmp/bench_video.mp4"
else
    VIDEO=""
fi

# Familles avec audio
AUDIO_FAMILIES="e2b e4b"

echo "=== Gemma 4 Benchmarks — ${QUANT} ==="
echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Machine: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'Apple Silicon')"
echo "RAM: $(sysctl -n hw.memsize | awk '{print int($1/1024/1024/1024)}') GB"
echo ""

for family in e2b e4b a4b 31b; do
    case $family in
        e2b) model="gemma-4-e2b-it-${QUANT}" ;;
        e4b) model="gemma-4-e4b-it-${QUANT}" ;;
        a4b) model="gemma-4-26b-a4b-it-${QUANT}" ;;
        31b) model="gemma-4-31b-it-${QUANT}" ;;
    esac
    model_path="$CACHE/$model"

    # Filtrer si demandé
    if [ -n "$FILTER" ] && [[ "$family" != *"$FILTER"* ]]; then
        continue
    fi

    if [ ! -d "$model_path" ]; then
        echo "SKIP $model (not downloaded)"
        continue
    fi

    echo "=========================================="
    echo "MODEL: $model"
    echo "=========================================="
    RESULT_FILE="$OUTDIR/${family}.txt"
    echo "Model: mlx-community/$model" > "$RESULT_FILE"
    echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$RESULT_FILE"
    echo "" >> "$RESULT_FILE"

    # --- TEXT ---
    echo "--- Text Generation ---"
    echo "=== Text Generation ===" >> "$RESULT_FILE"
    $CLI generate --model-path "$model_path" \
        "Explain quantum computing in 3 sentences. Be precise and concise." \
        --max-tokens 200 --temperature 0.3 2>&1 | tee -a "$RESULT_FILE"
    echo "" >> "$RESULT_FILE"
    echo ""

    # --- VISION (single image) ---
    if [ -f "$IMG" ]; then
        echo "--- Vision (single image) ---"
        echo "=== Vision (single image) ===" >> "$RESULT_FILE"
        $CLI describe --model-path "$model_path" \
            --image "$IMG" \
            --prompt "Describe this image in detail. What type of vehicle is this? What color? What is the setting?" \
            --max-tokens 300 --temperature 0.3 2>&1 | tee -a "$RESULT_FILE"
        echo "" >> "$RESULT_FILE"
        echo ""
    fi

    # --- VIDEO ---
    if [ -n "$VIDEO" ]; then
        echo "--- Video ---"
        echo "=== Video ===" >> "$RESULT_FILE"
        $CLI describe --model-path "$model_path" \
            --video "$VIDEO" \
            --prompt "Describe this video in detail. What is happening frame by frame?" \
            --max-tokens 400 --temperature 0.3 2>&1 | tee -a "$RESULT_FILE"
        echo "" >> "$RESULT_FILE"
        echo ""
    fi

    # --- AUDIO (E2B/E4B only) ---
    if [[ " $AUDIO_FAMILIES " == *" $family "* ]] && [ -f "$AUDIO_SRC" ]; then
        echo "--- Audio (transcription) ---"
        echo "=== Audio (transcription) ===" >> "$RESULT_FILE"
        $CLI describe --model-path "$model_path" \
            --audio "$AUDIO_SRC" \
            --prompt "Transcribe the following speech segment in English into English text." \
            --max-tokens 200 --temperature 0.3 2>&1 | tee -a "$RESULT_FILE"
        echo "" >> "$RESULT_FILE"

        echo "--- Audio (comprehension) ---"
        echo "=== Audio (comprehension) ===" >> "$RESULT_FILE"
        $CLI describe --model-path "$model_path" \
            --audio "$AUDIO_SRC" \
            --prompt "Listen to this audio. Who is speaking? What is the context? Summarize the main message." \
            --max-tokens 300 --temperature 0.3 2>&1 | tee -a "$RESULT_FILE"
        echo "" >> "$RESULT_FILE"
        echo ""
    fi

    echo "Results saved to $RESULT_FILE"
    echo ""
done

# Cleanup
rm -f /tmp/bench_video.mp4

echo "=== Benchmarks ${QUANT} terminés ==="
echo "Résultats dans $OUTDIR/"
