#!/usr/bin/env bash
#
# Campagne de mesure DiffusionGemma (K-D10, K-D11, K-D14) — machine au repos.
# 1. build Release + controle de la machine ; 2. telechargement du bf16 officiel
#    (~50 Go) dans $MODELS si absent ; 3. A/A de l'instrument (a4bdiff/16bit-fast, d1,
#    2 passes, dispersion du pas median <= 3 % sinon arret) ; 4. six profils
#    a4bdiff/{16,8,4}bit-{fast,lean} x charges d1/d2/d3, 2 passes, cooldown.
# Garder le Mac eveille :
#   python3 ~/.claude/skills/mac-awake/scripts/awake.py run -- Scripts/bench-diffusion-campaign.sh
set -euo pipefail
cd "$(dirname "$0")/.."
MODELS="${MODELS:-/Volumes/Lexar/models}"
COOLDOWN="${COOLDOWN:-120}"; REPEATS="${REPEATS:-2}"
PROFILES="${PROFILES:-16bit-fast 16bit-lean 8bit-fast 8bit-lean 4bit-fast 4bit-lean}"
WORKLOADS="${WORKLOADS:-d1 d2 d3}"
IMG="${IMG:-docs/examples/vision-image-description/UI.png}"
STAMP="$(date +%Y%m%d-%H%M)"; OUT="benchmarks/diffusion-${STAMP}.jsonl"; LOG="benchmarks/diffusion-${STAMP}.log"
CLI=".build/xcode/Build/Products/Release/gemma4-cli"
CHECK="$HOME/.claude/skills/mlx-swift-audit/scripts/machine-check.sh"
DIR="$MODELS/google/diffusiongemma-26B-A4B-it"
mkdir -p benchmarks; exec > >(tee -a "$LOG") 2>&1

echo "== build Release"
xcodebuild -scheme gemma4-cli -configuration Release -destination "platform=macOS" \
  -derivedDataPath .build/xcode -skipMacroValidation build -quiet || true
[ -x "$CLI" ] || { echo "binaire absent"; exit 1; }
echo "== machine"; "$CHECK" "$CLI" || { echo "Machine pas au repos : campagne annulee."; exit 1; }
echo "== poids"; [ -f "$DIR/config.json" ] || "$CLI" download diff-bf16 --models-dir "$MODELS"

bench() { # profil charge label
  local extra=(); [ "$2" != d1 ] && extra=(--image "$IMG")
  "$CLI" bench-diffusion --model-path "$DIR" --reference "a4bdiff/$1" --workload "$2" \
    ${extra[@]+"${extra[@]}"} --cooldown "$COOLDOWN" --repeats "$REPEATS" --label "$3" --out "$OUT"
}

echo "== validation A/A"
bench 16bit-fast d1 AA
python3 - "$OUT" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if '"label":"AA"' in l]
worst = 0.0
for key in ("step_ms_median",):
    v = [float(r[key]) for r in rows if key in r]
    if len(v) >= 2:
        spread = (max(v) - min(v)) / max(v) * 100; worst = max(worst, spread)
        print(f"  {key} {v} dispersion {spread:.1f} %")
    else:
        print(f"  champ {key} absent : {rows[0].keys() if rows else 'aucune ligne'}"); sys.exit(3)
print(f"  pire dispersion : {worst:.1f} % ({'OK' if worst <= 3 else 'TROP BRUITE'})")
sys.exit(0 if worst <= 3 else 2)
PY

for p in $PROFILES; do for w in $WORKLOADS; do echo "== $p $w"; bench "$p" "$w" baseline; done; done
echo "== fini : $OUT"
