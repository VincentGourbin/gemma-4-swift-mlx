#!/bin/bash
# OCRBench 300 images - enchaine les 3 modeles en sequence
set -e
MODEL_KIND="${1:-diff}"
META=/tmp/ocrbench/sample300/meta.json
RESULTS=/tmp/ocrbench/results300/$MODEL_KIND
mkdir -p "$RESULTS"
CLI=/Users/vincent/Developpements/gemma-4-swift-mlx/.build/xcode/Build/Products/Debug/gemma4-cli

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
  *) echo "Unknown: $MODEL_KIND"; exit 1 ;;
esac

cd /Users/vincent/Developpements/gemma-4-swift-mlx
N=$(python3 -c "import json; print(len(json.load(open('$META'))))")
echo "Running $MODEL_KIND on $N images..."

START_GLOBAL=$(date +%s)
for i in $(seq 0 $((N-1))); do
  IDX=$(printf "%03d" $i)
  OUT="$RESULTS/$IDX.txt"
  if [ -f "$OUT" ]; then continue; fi

  ENTRY=$(python3 -c "import json; m=json.load(open('$META'))[$i]; print(m['image']+'|'+m['question']+'|'+'|'.join(m['answer']))")
  IMG=$(echo "$ENTRY" | cut -d'|' -f1)
  QUESTION=$(echo "$ENTRY" | cut -d'|' -f2)

  case "$MODEL_KIND" in
    diff)
      "${BASE_CMD[@]}" --image "$IMG" "$QUESTION" 2>&1 > "$OUT" || true
      ;;
    e4b|a4b)
      "${BASE_CMD[@]}" --image "$IMG" --prompt "$QUESTION" 2>&1 > "$OUT" || true
      ;;
  esac

  if [ $((i % 30)) -eq 29 ]; then
    ELAPSED=$(($(date +%s) - START_GLOBAL))
    echo "[$IDX/$N] $ELAPSED s cumul"
  fi
done

TOTAL=$(($(date +%s) - START_GLOBAL))
echo "Done $MODEL_KIND in $TOTAL s"
