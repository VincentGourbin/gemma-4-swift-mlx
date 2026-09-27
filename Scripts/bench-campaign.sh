#!/usr/bin/env bash
#
# Campagne de mesure des profils de reference (K-12, K-21) — a lancer machine au repos.
#
# 1. Build Release, controle de la machine (aucun autre process MLX, secteur).
# 2. Validation de l'instrument : A/A sur E2B 4 bits, dispersion <= 3 %.
# 3. Pour chaque (famille, bits) : telechargement dans $MODELS si absent, mesure des
#    profils fast et lean (--reference, --cooldown, --repeats), image pour E2B/E4B,
#    puis suppression des poids telecharges par la campagne si --cleanup.
#
# Duree : ~12 min par profil au protocole complet (3 tailles x 2 passes x 120 s de
# repos) : compter une nuit pour la matrice. Le disque interne est presque plein :
# les poids vont sur un disque externe ($MODELS). Garder le Mac eveille :
#   python3 ~/.claude/skills/mac-awake/scripts/awake.py run -- Scripts/bench-campaign.sh --cleanup
#
# Options (variables d'environnement) :
#   MODELS=/Volumes/Lexar/models        racine des poids
#   FAMILIES="e2b e4b 12b a4b 31b"      familles (raccourcis de `gemma4-cli download`)
#   BITS="4bit 8bit bf16"               quantisations
#   PROMPTS=128,1024,4096  MAX_TOKENS=128  COOLDOWN=120  REPEATS=2
#   SKIP_AA=1                           sauter la validation A/A
#   --cleanup                           supprimer les poids telecharges par la campagne
#   DRY_RUN=1                           afficher les commandes sans rien executer
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS="${MODELS:-/Volumes/Lexar/models}"
FAMILIES="${FAMILIES:-e2b e4b 12b a4b 31b}"
BITS="${BITS:-4bit 8bit bf16}"
PROMPTS="${PROMPTS:-128,1024,4096}"
MAX_TOKENS="${MAX_TOKENS:-128}"
COOLDOWN="${COOLDOWN:-120}"
REPEATS="${REPEATS:-2}"
CLEANUP=0
[ "${1:-}" = "--cleanup" ] && CLEANUP=1

STAMP="$(date +%Y%m%d-%H%M)"
OUT="benchmarks/campaign-${STAMP}.jsonl"
LOG="benchmarks/campaign-${STAMP}.log"
CLI=".build/xcode/Build/Products/Release/gemma4-cli"
CHECK="$HOME/.claude/skills/mlx-swift-audit/scripts/machine-check.sh"
IMG="docs/examples/vision-image-description/input_sample.jpg"
mkdir -p benchmarks
exec > >(tee -a "$LOG") 2>&1

if [ "${DRY_RUN:-0}" = 1 ]; then
  CHECK=/nonexistent
  echo "(essai a blanc : aucune commande executee)"
else
echo "== build Release"
xcodebuild -scheme gemma4-cli -configuration Release -destination "platform=macOS" \
  -derivedDataPath .build/xcode -skipMacroValidation build -quiet
fi

echo "== machine"
if [ -x "$CHECK" ]; then
  "$CHECK" "$CLI" || { echo "Machine pas au repos : campagne annulee."; exit 1; }
else
  echo "(machine-check.sh introuvable : controle manuel requis)"
fi

cli() { if [ "${DRY_RUN:-0}" = 1 ]; then echo "gemma4-cli $*" >&2; else "$CLI" "$@"; fi; }
bits_to_profile() { case "$1" in 4bit) echo 4bit ;; 8bit) echo 8bit ;; bf16) echo 16bit ;; *) echo "" ;; esac; }
family_to_profile() { case "$1" in 12b) echo b12b ;; 31b) echo b31b ;; *) echo "$1" ;; esac; }
repo_name() { # raccourci de famille -> nom du depot mlx-community
  case "$1" in 12b) echo "gemma-4-12B-it-$2" ;; a4b) echo "gemma-4-26b-a4b-it-$2" ;; *) echo "gemma-4-$1-it-$2" ;; esac
}

ensure_model() { # $1 raccourci (e2b-4bit) $2 dossier attendu ; echo 1 si telecharge ici
  if [ -f "$2/config.json" ]; then echo 0; return; fi
  cli download "$1" --models-dir "$MODELS" >&2
  echo 1
}

if [ "${SKIP_AA:-0}" != 1 ]; then
  echo "== validation A/A (E2B 4 bits)"
  dir="$MODELS/mlx-community/$(repo_name e2b 4bit)"
  fetched=$(ensure_model e2b-4bit "$dir")
  cli bench --model-path "$dir" --prompt-tokens "$PROMPTS" --max-tokens "$MAX_TOKENS" \
    --cooldown "$COOLDOWN" --repeats 2 --label AA --out "$OUT"
  [ "${DRY_RUN:-0}" = 1 ] || python3 - "$OUT" <<'PY'
import json, sys, collections
rows = [json.loads(l) for l in open(sys.argv[1]) if '"label":"AA"' in l]
by = collections.defaultdict(list)
for r in rows: by[r["prompt_tokens"]].append(r)
worst = 0.0
for size, rs in sorted(by.items()):
    # Mediane pour le decodage : la moyenne (decode_tok_s) bouge de 8 % sur quelques
    # pas lents isoles alors que l'intervalle median bouge de 1,8 % (A/A du 2026-09-27).
    for key in ("prefill_tok_s", "decode_tok_s_median"):
        v = [float(r[key]) for r in rs]
        spread = (max(v) - min(v)) / max(v) * 100
        worst = max(worst, spread)
        print(f"  {size:>5} jetons {key:<14} {v} dispersion {spread:.1f} %")
print(f"  pire dispersion : {worst:.1f} % ({'OK' if worst <= 3 else 'TROP BRUITE : refaire au repos'})")
sys.exit(0 if worst <= 3 else 2)
PY
fi

for fam in $FAMILIES; do
  for b in $BITS; do
    shortcut="$fam-$b"; dir="$MODELS/mlx-community/$(repo_name "$fam" "$b")"
    pb=$(bits_to_profile "$b"); pf=$(family_to_profile "$fam")
    echo "== $shortcut ($dir)"
    fetched=$(ensure_model "$shortcut" "$dir")
    for kind in fast lean; do
      cli bench --model-path "$dir" --reference "$pf/$pb-$kind" --prompt-tokens "$PROMPTS" \
        --max-tokens "$MAX_TOKENS" --cooldown "$COOLDOWN" --repeats "$REPEATS" --label baseline --out "$OUT"
      if [ "$fam" = e2b ] || [ "$fam" = e4b ]; then
        cli bench --model-path "$dir" --reference "$pf/$pb-$kind" --image "$IMG" \
          --max-tokens "$MAX_TOKENS" --cooldown "$COOLDOWN" --repeats "$REPEATS" --label baseline-image --out "$OUT"
      fi
    done
    if [ "$fetched" = 1 ] && [ "$CLEANUP" = 1 ]; then
      echo "  suppression de $dir"; [ "${DRY_RUN:-0}" = 1 ] || rm -rf "$dir"
    fi
  done
done

echo "== fini : $OUT (recopier les lignes dans BENCHMARKS.md, remplir docs/References.md)"
