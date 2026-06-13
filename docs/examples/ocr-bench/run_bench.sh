#!/bin/bash
# OCRBench mini-bench pour DiffusionGemma + (optionnel) modèles AR comparaison.
#
# Usage : ./run_bench.sh <model_kind>
#   model_kind = diff | e4b | a4b
#
# Lance la CLI sur les 30 images stratifiees + sauve les outputs dans
# /tmp/ocrbench/results/<model_kind>/<idx>.txt + un CSV resume.

set -e
MODEL_KIND="${1:-diff}"
META=/tmp/ocrbench/sample/meta.json
RESULTS=/tmp/ocrbench/results/$MODEL_KIND
mkdir -p "$RESULTS"
CLI=/Users/vincent/Developpements/gemma-4-swift-mlx/.build/xcode/Build/Products/Debug/gemma4-cli

# Build de la commande selon le model
case "$MODEL_KIND" in
  diff)
    BASE_CMD=("$CLI" diffusion --include-vision --max-blocks 1)
    ;;
  e4b)
    BASE_CMD=("$CLI" describe --model-path "$HOME/Library/Caches/models/mlx-community/gemma-4-e4b-it-4bit" --max-tokens 60 --temperature 0.1)
    ;;
  a4b)
    BASE_CMD=("$CLI" describe --model-path "$HOME/Library/Caches/models/mlx-community/gemma-4-26b-a4b-it-4bit" --max-tokens 60 --temperature 0.1)
    ;;
  *) echo "Unknown model_kind: $MODEL_KIND"; exit 1 ;;
esac

cd /Users/vincent/Developpements/gemma-4-swift-mlx

# Loop sur le meta.json
N=$(python3 -c "import json; print(len(json.load(open('$META'))))")
echo "Running $MODEL_KIND on $N images..."

for i in $(seq 0 $((N-1))); do
  IDX=$(printf "%03d" $i)
  ENTRY=$(python3 -c "import json; m=json.load(open('$META'))[$i]; print(m['image']+'|'+m['question']+'|'+'|'.join(m['answer']))")
  IMG=$(echo "$ENTRY" | cut -d'|' -f1)
  QUESTION=$(echo "$ENTRY" | cut -d'|' -f2)

  OUT="$RESULTS/$IDX.txt"
  if [ -f "$OUT" ]; then
    echo "[$IDX] cached"
    continue
  fi

  START=$(date +%s)
  case "$MODEL_KIND" in
    diff)
      "${BASE_CMD[@]}" --image "$IMG" "$QUESTION" 2>&1 > "$OUT" || echo "[$IDX] ERROR"
      ;;
    e4b|a4b)
      "${BASE_CMD[@]}" --image "$IMG" --prompt "$QUESTION" 2>&1 > "$OUT" || echo "[$IDX] ERROR"
      ;;
  esac
  ELAPSED=$(($(date +%s) - START))
  echo "[$IDX] $ELAPSED s"
done

echo "Done."
