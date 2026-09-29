#!/usr/bin/env bash
#
# Campagne de mesure des profils d'entrainement LoRA (K-33) — a lancer machine au repos.
#
# 1. Build Release (sauf SKIP_BUILD=1).
# 2. Telechargement de tous les packs de base manquants AVANT toute mesure (le reseau et
#    le disque ne tournent pas pendant un run chronometre).
# 3. Pour chaque profil de `gemma4-cli lora profiles` : controle de la machine, puis
#    `lora train --reference <id> --allow-unmeasured` sur le dataset director, graine 0,
#    STEPS pas, validation au pas 1 et au dernier pas (VAL_BATCHES lots), une ligne JSONL
#    par rapport dans benchmarks/k33-<stamp>/<famille>-<id>.jsonl.
#
# Aucun poids n'est supprime : la liste des packs telecharges ici est affichee a la fin.
# Un run qui echoue (memoire, crash) est consigne et la campagne continue.
#
# Garder le Mac eveille :
#   python3 ~/.claude/skills/mac-awake/scripts/awake.py run -- Scripts/bench-training-campaign.sh
#
# Options (variables d'environnement) :
#   MODELS=/Volumes/Lexar/models    racine des poids
#   DATA=<dossier train/valid.jsonl> defaut : dataset director de Fluxforge Studio
#   PROFILES="e2b/lora-16bit-fast ..." sous-ensemble (defaut : tous les candidats)
#   STEPS=50  VAL_BATCHES=10  COOLDOWN=60
#   SKIP_BUILD=1  DRY_RUN=1
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS="${MODELS:-${GEMMA4_MODELS_DIR:-/Volumes/Lexar/models}}"
DATA="${DATA:-$HOME/Developpements/Fluxforge Studio/Scripts/director/dataset}"
STEPS="${STEPS:-50}"
VAL_BATCHES="${VAL_BATCHES:-10}"
COOLDOWN="${COOLDOWN:-60}"
CLI=".build/xcode/Build/Products/Release/gemma4-cli"
CHECK="$HOME/.claude/skills/mlx-swift-audit/scripts/machine-check.sh"
STAMP="$(date +%Y%m%d-%H%M)"
OUTDIR="benchmarks/k33-${STAMP}"
ADAPTERS="${ADAPTERS:-/Volumes/Lexar/tmp/k33-adapters}"
mkdir -p "$OUTDIR" "$ADAPTERS"
exec > >(tee -a "$OUTDIR/campaign.log") 2>&1

run() { if [ "${DRY_RUN:-0}" = 1 ]; then echo "+ $*"; else "$@"; fi; }

if [ "${SKIP_BUILD:-0}" != 1 ] && [ "${DRY_RUN:-0}" != 1 ]; then
  echo "== build Release"
  xcodebuild -scheme gemma4-cli -configuration Release -destination "platform=macOS" \
    -derivedDataPath .build/xcode -skipMacroValidation build -quiet
fi

# "<famille>/<id> <depot>" pour chaque candidat
# (bash 3.2 de macOS : pas de mapfile)
ROWS=()
while read -r id repo; do
  if [ -n "${PROFILES:-}" ]; then
    case " $PROFILES " in *" $id "*) ;; *) continue ;; esac
  fi
  ROWS+=("$id $repo")
done < <("$CLI" lora profiles | awk '/^[a-z0-9]+\/lora-/ {print $1, $3}')
[ "${#ROWS[@]}" -gt 0 ] || { echo "aucun profil a mesurer"; exit 1; }

echo "== telechargements"
FETCHED=()
for row in "${ROWS[@]}"; do
  repo="${row#* }"
  dir="$MODELS/$repo"
  [ -f "$dir/config.json" ] && continue
  case " ${FETCHED[*]:-} " in *" $repo "*) continue ;; esac
  run "$CLI" download "$repo" --models-dir "$MODELS"
  FETCHED+=("$repo")
done

echo "== runs (${#ROWS[@]} profils, $STEPS pas)"
for row in "${ROWS[@]}"; do
  id="${row%% *}"; repo="${row#* }"
  name="${id//\//-}"
  echo "-- $id ($repo) $(date +%H:%M)"
  if [ "${DRY_RUN:-0}" != 1 ] && [ -x "$CHECK" ]; then
    "$CHECK" "$CLI" --cooldown "$COOLDOWN" || { echo "machine pas au repos : $id saute"; continue; }
  fi
  rm -rf "${ADAPTERS:?}/$name"
  if run "$CLI" lora train --model-path "$MODELS/$repo" --data "$DATA" \
      --output "$ADAPTERS/$name" --reference "$id" --allow-unmeasured \
      --iterations "$STEPS" --steps-per-eval "$STEPS" --steps-per-report 10 \
      --val-batches "$VAL_BATCHES" --mask-prompt --seed 0 \
      --metrics-out "$OUTDIR/$name.jsonl" > "$OUTDIR/$name.log" 2> "$OUTDIR/$name.err"; then
    echo "   ok : $(tail -1 "$OUTDIR/$name.log")"
  else
    echo "   ECHEC (code $?) : voir $OUTDIR/$name.err"
  fi
done

echo "== fini $(date +%H:%M) — resultats dans $OUTDIR"
[ "${#FETCHED[@]}" -gt 0 ] && printf 'Packs telecharges par la campagne (a supprimer sur decision) :\n%s\n' "${FETCHED[@]/#/  $MODELS/}"
exit 0
