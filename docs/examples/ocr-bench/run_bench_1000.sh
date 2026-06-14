#!/bin/bash
set -e
MODEL_KIND="${1:-diff}"
META=/tmp/ocrbench/sample1000/meta.json
RESULTS=/tmp/ocrbench/results1000/$MODEL_KIND
mkdir -p "$RESULTS"
CLI=/Users/vincent/Developpements/gemma-4-swift-mlx/.build/xcode/Build/Products/Debug/gemma4-cli

case "$MODEL_KIND" in
  diff) BASE_CMD=("$CLI" diffusion --include-vision --max-blocks 1) ;;
  e4b)  BASE_CMD=("$CLI" describe --model-path "$HOME/Library/Caches/models/mlx-community/gemma-4-e4b-it-4bit" --max-tokens 60 --temperature 0.1) ;;
  a4b)  BASE_CMD=("$CLI" describe --model-path "$HOME/Library/Caches/models/mlx-community/gemma-4-26b-a4b-it-4bit" --max-tokens 60 --temperature 0.1) ;;
  *) echo "Unknown: $MODEL_KIND"; exit 1 ;;
esac

cd /Users/vincent/Developpements/gemma-4-swift-mlx
N=$(python3 -c "import json; print(len(json.load(open('$META'))))")
echo "[$(date +%T)] Running $MODEL_KIND on $N images..."

START=$(date +%s)
for i in $(seq 0 $((N-1))); do
  IDX=$(printf "%04d" $i)
  OUT="$RESULTS/$IDX.txt"
  [ -f "$OUT" ] && continue
  
  ENTRY=$(python3 -c "import json; m=json.load(open('$META'))[$i]; print(m['image']+'\t'+m['question'])")
  IMG=$(echo "$ENTRY" | cut -f1)
  QUESTION=$(echo "$ENTRY" | cut -f2)

  case "$MODEL_KIND" in
    diff) "${BASE_CMD[@]}" --image "$IMG" "$QUESTION" 2>&1 > "$OUT" || true ;;
    e4b|a4b) "${BASE_CMD[@]}" --image "$IMG" --prompt "$QUESTION" 2>&1 > "$OUT" || true ;;
  esac

  if [ $((i % 50)) -eq 49 ]; then
    ELAPSED=$(($(date +%s) - START))
    AVG=$(echo "scale=1; $ELAPSED / ($i + 1)" | bc)
    REMAIN=$(echo "scale=0; ($N - $i - 1) * $AVG / 1" | bc)
    echo "[$(date +%T)] $((i+1))/$N done, ${ELAPSED}s elapsed, avg ${AVG}s/img, est ${REMAIN}s remaining"
  fi
done

TOTAL=$(($(date +%s) - START))
echo "[$(date +%T)] Done $MODEL_KIND in $TOTAL s = $(echo "scale=1;$TOTAL/60" | bc) min"
