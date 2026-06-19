#!/usr/bin/env bash
# Wrapper de soumission Toolathlon avec retry sur "Server is busy".
#
# Lance eval_client.py en boucle. Si le serveur renvoie "is busy",
# attend INTERVAL secondes et retente. Quand le job est accepté,
# eval_client.py continue à tourner jusqu'à completion ou kill.
#
# Usage :
#   ./run_with_retry.sh [TASK_NAME] [PROXY_PORT] [INTERVAL_SECONDS] [MAX_ATTEMPTS]
#
# Défauts :
#   TASK_NAME=find-alita-paper, PROXY_PORT=8001, INTERVAL=60, MAX_ATTEMPTS=120
#
# (120 × 60s = 2 h max d'attente, large pour qu'une fenêtre s'ouvre)

set -uo pipefail

TASK_NAME="${1:-find-alita-paper}"
PROXY_PORT="${2:-8001}"
INTERVAL="${3:-60}"
MAX_ATTEMPTS="${4:-120}"
TOOLATHLON_DIR="${TOOLATHLON_DIR:-/tmp/Toolathlon}"

if [[ ! -d "$TOOLATHLON_DIR" ]]; then
    echo "❌ Toolathlon repo introuvable à $TOOLATHLON_DIR"
    echo "   git clone --depth 1 https://github.com/hkust-nlp/Toolathlon.git $TOOLATHLON_DIR"
    exit 1
fi

# Préparer la task list dans un dossier de run isolé
RUN_DIR=$(mktemp -d -t toolathlon-run-XXXXXX)
echo "📁 Run dir: $RUN_DIR"
cp "$TOOLATHLON_DIR/eval_client.py" "$RUN_DIR/"
cp "$TOOLATHLON_DIR/simple_client_ws.py" "$RUN_DIR/"
echo "$TASK_NAME" > "$RUN_DIR/debug_tasks.txt"
mkdir -p "$RUN_DIR/results"

echo "🎯 Task: $TASK_NAME"
echo "🔌 Proxy: http://localhost:$PROXY_PORT (DiffusionGemma 26B-A4B bf16)"
echo "⏱  Retry every ${INTERVAL}s, max ${MAX_ATTEMPTS} attempts"
echo ""

cd "$RUN_DIR"

attempt=0
while (( attempt < MAX_ATTEMPTS )); do
    attempt=$(( attempt + 1 ))
    ts=$(date '+%H:%M:%S')
    echo "──────────────────────────────────────────"
    echo "[$ts] Attempt #${attempt}/${MAX_ATTEMPTS}"
    echo "──────────────────────────────────────────"

    # Capture la sortie tout en l'affichant
    output=$(python3 eval_client.py run \
        --mode private \
        --base-url "http://localhost:${PROXY_PORT}/v1" \
        --model-name diffusiongemma-26B-A4B-it \
        --output-dir ./results \
        --server-host 47.253.6.47 \
        --api-key dummy \
        --workers 1 \
        --server-port 8080 \
        --ws-proxy-port 8081 \
        --task-list-file ./debug_tasks.txt \
        --skip-container-restart 2>&1 | tee /dev/tty)
    rc=$?

    # Si on voit "is busy" (HTTP 400 du serveur) → on retry
    if echo "$output" | grep -qiE "Server is busy|is busy|processing another"; then
        wait_human=$(printf '%d:%02d' $(( INTERVAL / 60 )) $(( INTERVAL % 60 )))
        echo ""
        echo "⏳ Serveur busy. Sleep ${wait_human} avant retry…"
        sleep "$INTERVAL"
        continue
    fi

    # Si on voit "Workers limit exceeded" → bug client, on stoppe
    if echo "$output" | grep -qiE "version not supported|version missing|Workers limit"; then
        echo ""
        echo "❌ Erreur client. Stop."
        exit 2
    fi

    # Si rc=0 ET pas de "is busy" → job soumis et terminé. Stop.
    if (( rc == 0 )); then
        echo ""
        echo "✅ Job terminé. Résultats dans $RUN_DIR/results"
        exit 0
    fi

    # Autre erreur : on retry quand même, peut-être un flake réseau
    echo ""
    echo "⚠️  Sortie non-OK (rc=$rc). Sleep ${INTERVAL}s avant retry."
    sleep "$INTERVAL"
done

echo ""
echo "❌ Max attempts ($MAX_ATTEMPTS) atteint sans succès."
echo "   Le serveur est resté busy pendant $(( MAX_ATTEMPTS * INTERVAL / 60 )) minutes."
echo "   Suggestion : essaye plus tard ou contacte les auteurs Toolathlon"
echo "   (jlini@cse.ust.hk / junxianh@cse.ust.hk) pour un serveur dédié."
exit 1
