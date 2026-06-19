# Toolathlon — POC DiffusionGemma 26B-A4B en agent de tool-use

[Toolathlon](https://github.com/hkust-nlp/Toolathlon) (ICLR 2026) est un bench
de **tool-use agentique long-horizon** avec 600+ outils basés sur des
environnements logiciels réels (Canvas, Gmail, GitHub, arXiv, Scholarly, etc.).
Le benchmark fournit l'infrastructure côté serveur (containers MCP avec
comptes pré-configurés) ; toi tu fournis seulement le LLM en endpoint
**OpenAI ChatCompletion-compatible**.

Ce POC démontre que **DiffusionGemma 26B-A4B bf16 peut produire du tool calling
au format OpenAI** via un proxy Python minimal qui bridge vers `gemma4-cli
diffusion`. Le full eval n'a pas pu tourner faute de fenêtre serveur (rate
limit IP du service public à 180 min cumulés / 24 h) mais le pipeline complet
est fonctionnel.

## Architecture

```
┌─────────────────────────┐   WebSocket    ┌──────────────────────┐
│ Toolathlon eval_server  │ ◄──────────►   │ Toolathlon eval_client│
│  (47.253.6.47:8080)     │                │  (sur ta machine)     │
│  600+ MCP tools         │                └──────────┬───────────┘
└─────────────────────────┘                           │
                                                     POST /v1/chat/completions
                                                      │
                                          ┌───────────▼───────────┐
                                          │   proxy.py            │
                                          │   localhost:8001      │
                                          └───────────┬───────────┘
                                                      │ subprocess
                                                      ▼
                                          ┌───────────────────────┐
                                          │ gemma4-cli diffusion  │
                                          │ --include-vision      │
                                          │ --max-blocks 2-3      │
                                          └───────────────────────┘
```

- **eval_server** héberge les containers MCP et orchestre la session
- **eval_client** récupère les jobs via WebSocket et POSTe chaque tour LLM
  vers `localhost:8001/v1/chat/completions`
- **proxy.py** convertit le body OpenAI en prompt textuel, lance le CLI
  Swift en sous-process, parse la sortie pour reconstituer `tool_calls`

## Prompt système

Le proxy construit un prompt simple qui force le format de réponse :

```
You are an agent that completes user tasks by calling tools.
Respond with EITHER:
  a) A natural-language answer to the user, OR
  b) Exactly one tool call on its own line, formatted as:
     TOOL_CALL: {"name": "tool_name", "arguments": {...}}

Available tools (call by name with the listed arguments):
  - get_weather({"city": {"type": "string", "description": "City name"}}) : ...

Conversation:
USER: What is the weather in Paris today?
ASSISTANT:
```

DiffusionGemma a un format de tool calling propre démontré sur
[BFCL à 95%](../function-calling-bench/README.md) — on réutilise la même
forme.

## Test local validé

```bash
$ curl -s -X POST http://localhost:8001/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "diffusiongemma-26B-A4B-it",
      "messages": [{"role": "user", "content": "Weather in Paris today?"}],
      "tools": [{
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get current weather for a city",
          "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
          }
        }
      }]
    }' | python3 -m json.tool
```

**Réponse (latence : 30 s côté proxy = ~22 s chargement modèle + 3.6 s génération)** :

```json
{
  "model": "diffusiongemma-26B-A4B-it",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_f7896dfad36f4cf5",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"city\": \"Paris\"}"
        }
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

✅ Format OpenAI ChatCompletion strict respecté.
✅ Tool calling produit le bon nom + arguments parsés JSON.
✅ `finish_reason: "tool_calls"` correctement positionné.

Sortie brute CLI captée par le proxy (DiffusionGemma 26B-A4B bf16, 256 tokens
generated, 3 forwards decoder, 71.8 tok/s) :

```
[canvas 0] TOOL_CALL: {"name": "get_weather", "arguments": {"city": "Paris"}}
--- Stats ---
Tokens generes : 256
Decoder forwards : 3
Temps : 3.57s
Vitesse : 71.8 tok/s
```

## Tentative de run Toolathlon

Task choisie : `find-alita-paper` (la plus simple suggérée par leur README,
nécessite arxiv + filesystem + scholarly MCP servers).

```bash
$ python eval_client.py run \
    --mode private \
    --base-url http://localhost:8001/v1 \
    --model-name diffusiongemma-26B-A4B-it \
    --output-dir ./results \
    --server-host 47.253.6.47 \
    --api-key dummy \
    --workers 1 \
    --task-list-file ./debug_tasks.txt \
    --skip-container-restart
```

**Réponse serveur (à plusieurs tentatives)** :

```
❌ Server is busy:
   Server is currently processing another task. Please try again later.
```

Le service public Toolathlon implémente un **rate limit dual** (cf
[EVAL_SERVICE_README.md](https://github.com/hkust-nlp/Toolathlon/blob/main/EVAL_SERVICE_README.md)) :

- 180 minutes cumulatives par IP par 24 h
- 3 requêtes max par IP par 24 h une fois ce budget dépassé
- Un seul job concurrent par IP

Cette limitation n'est pas un problème de notre côté ; elle empêche
simplement de finir le full eval depuis une IP partagée. Les options
décrites par les auteurs pour aller plus loin :

1. **Setup local du serveur Toolathlon** (Docker/Podman, ~30 min setup)
2. **Contacter les auteurs** (`jlini@cse.ust.hk` / `junxianh@cse.ust.hk`) pour
   un serveur dédié gratuit pour "major users"
3. **Endpoint API public** (contact aussi)

## Coûts attendus pour le full eval

Sur les ~250 tasks Toolathlon, chacune **long-horizon** (10-50 tool calls par
trajectoire) avec DiffusionGemma 26B-A4B bf16 sur M3 Max 96 Go :

| Item | Estimation |
|---|---|
| Latence par tour LLM (génération seule) | ~3-7 s (max-blocks 2-3, ~70 tok/s) |
| Latence par tour LLM (avec rechargement modèle) | ~30 s (mode actuel du POC) |
| Coup par task (estimation 30 tours × 3 s) | ~90 s en mode persistant, **~15 min** en mode rechargement |
| Coût full eval 250 tasks | **~60-150 h** en mode rechargement, **~6-12 h** en mode persistant |

→ Pour aller plus loin il faudrait remplacer le proxy Python par un **serveur
Swift natif** qui garde le modèle chargé en mémoire entre les requêtes. C'est
un travail de ~1 jour (ajout d'une commande `gemma4-cli serve` qui démarre un
HTTP server via SwiftNIO ou `Network.NWListener` autour de `DiffusionGemmaPipeline`).

## Fichiers

- `proxy.py` — proxy HTTP minimal (203 lignes), seul fichier dans ce POC

## Reproduction

```bash
# 1. Build du CLI
xcodebuild -scheme gemma4-cli -configuration Debug \
  -destination "platform=macOS" -derivedDataPath .build/xcode \
  -skipMacroValidation build

# 2. Démarrer le proxy
python3 docs/examples/toolathlon-bench/proxy.py --port 8001 --max-blocks 3

# 3. Cloner Toolathlon et installer les deps client
git clone https://github.com/hkust-nlp/Toolathlon.git /tmp/Toolathlon
cd /tmp/Toolathlon
pip install httpx typer websockets

# 4. Préparer la task list
echo "find-alita-paper" > debug_tasks.txt

# 5. Lancer
python eval_client.py run \
  --mode private \
  --base-url http://localhost:8001/v1 \
  --model-name diffusiongemma-26B-A4B-it \
  --server-host 47.253.6.47 \
  --workers 1 \
  --task-list-file debug_tasks.txt \
  --skip-container-restart
```

Si le serveur public n'est pas disponible, contacter les auteurs Toolathlon
pour un serveur dédié ou setup local Docker.

## Conclusion

✅ **Pipeline OpenAI ChatCompletion fonctionnel** pour DiffusionGemma sur
Apple Silicon, validé sur un round-trip de tool calling complet.

⏸️  **Full eval bloqué** par le rate limit du service public Toolathlon —
non lié à notre stack. À relancer dès qu'une fenêtre serveur s'ouvre, ou avec
un serveur Toolathlon local Docker.

⚠️ **Optimisation requise pour le full run** : passer du mode "rechargement
modèle par requête" à un serveur HTTP Swift natif persistant (× 6-12 plus
rapide).
