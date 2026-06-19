#!/usr/bin/env python3
"""Toolathlon → DiffusionGemma proxy.

Mini-serveur HTTP qui expose `/v1/chat/completions` au format OpenAI et bridge
chaque tour vers `gemma4-cli diffusion` en passant le prompt construit. Sert
pour le POC Toolathlon — pas optimisé pour la prod (modèle rechargé à chaque
requête, ~8s overhead + génération).

Usage :
    python3 proxy.py [--cli PATH] [--port 8000] [--max-blocks 3]

Lance ensuite côté Toolathlon (extrait du repo) :
    python eval_client.py run --mode private \\
        --base-url http://localhost:8000/v1 \\
        --model-name diffusiongemma-26B-A4B-it \\
        --server-host 47.253.6.47 \\
        --workers 1 \\
        --task-list-file debug_tasks.txt \\
        --skip-container-restart
"""
import argparse
import json
import logging
import re
import subprocess
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_CLI = "/Users/vincent/Developpements/gemma-4-swift-mlx/.build/xcode/Build/Products/Debug/gemma4-cli"
DEFAULT_PORT = 8000
DEFAULT_MAX_BLOCKS = 3   # 3 × 256 = 768 tokens output max — suffisant pour tool_call JSON
DEFAULT_TIMEOUT = 600    # 10 min par requête (modele bf16 + chargement)
DEFAULT_RAW_LOG_DIR = None  # si set, on dump chaque (prompt, raw_stdout) sous <DIR>/req-<N>.txt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("toolathlon-proxy")


def build_prompt(messages: list, tools: list | None) -> str:
    """Construit un prompt textuel à partir d'un body OpenAI ChatCompletion.

    Le format demande explicitement au modèle de produire SOIT du texte SOIT
    une ligne `TOOL_CALL: {"name": "...", "arguments": {...}}`. C'est plus
    simple à parser que le format OpenAI natif (qui exige tool_calls JSON
    multi-call) et marche pour le POC.
    """
    parts = []
    parts.append(
        "You are an agent that completes user tasks by calling tools.\n"
        "Respond with EITHER:\n"
        "  a) A natural-language answer to the user (final answer), OR\n"
        "  b) Exactly one tool call on its own line, formatted as a function "
        "call:\n"
        '     TOOL_CALL: tool_name({"arg1": "value", "arg2": 42})\n'
        "Example:\n"
        '     TOOL_CALL: arxiv_local__search_papers({"query": "diffusion models", '
        '"max_results": 5})\n'
        "Pick one. Do not mix prose and TOOL_CALL on the same answer.\n"
    )
    if tools:
        parts.append("\nAvailable tools (call by name with the listed arguments):")
        for t in tools:
            fn = t.get("function", t) if isinstance(t, dict) else {}
            name = fn.get("name", "?")
            desc = fn.get("description", "")
            params = fn.get("parameters", {})
            sig = json.dumps(params.get("properties", {}), ensure_ascii=False)
            parts.append(f"  - {name}({sig}) : {desc}")
        parts.append("")

    parts.append("Conversation:")
    for m in messages:
        role = m.get("role", "user").upper()
        content = m.get("content")
        if isinstance(content, list):
            # OpenAI vision-style content blocks — keep only text parts
            content = " ".join(c.get("text", "") for c in content if c.get("type") == "text")
        if m.get("tool_calls"):
            # assistant just produced a tool call
            call = m["tool_calls"][0]
            fn = call.get("function", {})
            content = f'TOOL_CALL: {{"name": "{fn.get("name", "")}", "arguments": {fn.get("arguments", "{}")}}}'
        if role == "TOOL":
            # tool execution result
            tool_name = m.get("name", "?")
            content = f"[result of {tool_name}]: {content}"
        parts.append(f"{role}: {content or ''}")
    parts.append("ASSISTANT:")
    return "\n".join(parts)


def _extract_balanced_json(raw: str, from_idx: int) -> str | None:
    """Extrait le 1er JSON object equilibre apres from_idx (gere {...{...}...})."""
    jstart = raw.find("{", from_idx)
    if jstart < 0:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(jstart, len(raw)):
        ch = raw[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return raw[jstart : i + 1]
    return None


def _build_tool_call(name: str, args) -> dict:
    if isinstance(args, dict):
        args = json.dumps(args, ensure_ascii=False)
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": f"call_{uuid.uuid4().hex[:16]}",
                "type": "function",
                "function": {"name": name, "arguments": args or "{}"},
            }
        ],
    }


def parse_response(raw: str) -> dict:
    """Convertit la sortie texte du modele en reponse OpenAI ChatCompletion.

    Reconnait deux formats produits par DiffusionGemma :
      a) TOOL_CALL: {"name": "...", "arguments": {...}}
      b) TOOL_CALL: name_with__underscores({"arg": "..."})
    """
    raw = raw.strip()
    raw = re.sub(r"<\s*eos\s*>", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"<\s*\|?turn\|?\s*>", "", raw)
    raw = raw.strip()

    # DiffusionGemma tronque parfois le canvas en debut de generation,
    # produisant `_CALL:` au lieu de `TOOL_CALL:` (les premiers tokens "TOOL"
    # ont ete ecrases pendant la diffusion). On accepte les deux variantes.
    matches = list(re.finditer(r"(?:TOOL)?_CALL\s*:", raw, re.IGNORECASE))
    for m in reversed(matches):
        after = raw[m.end():].lstrip()

        # Format (b) — TOOL_CALL: name(<json>)
        nm = re.match(r"([A-Za-z_][\w]*)\s*\(", after)
        if nm:
            name = nm.group(1)
            args_str = _extract_balanced_json(after, nm.end() - 1)  # nm.end()-1 = pos du '('
            # _extract_balanced_json cherche le 1er '{' apres from_idx donc OK,
            # mais ici il faut chercher apres le '('
            args_str = _extract_balanced_json(after, nm.end())
            if args_str:
                try:
                    args = json.loads(args_str)
                    return _build_tool_call(name, args)
                except json.JSONDecodeError as e:
                    log.debug(f"format-b parse failed for {name}: {e}")
            # Cas args vides : TOOL_CALL: name() ou name({})
            after_paren = after[nm.end():].lstrip()
            if after_paren.startswith(")"):
                return _build_tool_call(name, {})

        # Format (a) — TOOL_CALL: {"name": "...", "arguments": {...}}
        json_str = _extract_balanced_json(raw, m.end())
        if json_str:
            try:
                call = json.loads(json_str)
                name = call.get("name")
                if name:
                    return _build_tool_call(name, call.get("arguments", {}))
            except json.JSONDecodeError as e:
                log.debug(f"format-a parse failed at offset {m.end()}: {e}")

    return {"role": "assistant", "content": raw}


class Handler(BaseHTTPRequestHandler):
    cli_path = DEFAULT_CLI
    max_blocks = DEFAULT_MAX_BLOCKS
    timeout = DEFAULT_TIMEOUT
    raw_log_dir = DEFAULT_RAW_LOG_DIR
    req_counter = 0

    def log_message(self, fmt, *args):
        log.info(fmt % args)

    def _json_reply(self, code: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            self._json_reply(200, {
                "object": "list",
                "data": [{
                    "id": "diffusiongemma-26B-A4B-it",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "google",
                }],
            })
        else:
            self._json_reply(404, {"error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
            self._json_reply(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(length))
        except json.JSONDecodeError:
            self._json_reply(400, {"error": "invalid JSON"})
            return

        messages = req.get("messages", [])
        tools = req.get("tools")
        prompt = build_prompt(messages, tools)

        log.info(f"Prompt length: {len(prompt)} chars, {len(messages)} messages, "
                 f"{len(tools) if tools else 0} tools")

        # --include-vision est nécessaire même sans image : le checkpoint a
        # vision_config != nil et instancie quand même le vision_tower.
        cmd = [
            self.cli_path, "diffusion",
            "--include-vision",
            "--max-blocks", str(self.max_blocks),
            prompt,
        ]
        t0 = time.time()
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
        except subprocess.TimeoutExpired:
            log.error(f"CLI timeout after {self.timeout}s")
            self._json_reply(504, {"error": "CLI timeout"})
            return
        elapsed = time.time() - t0

        # Find the generated text between markers
        out = res.stdout
        # Strip the load logs : output starts after "[canvas 0]" or "--- Generation ---"
        body = out
        for marker in ["[canvas 0]", "--- Generation ---"]:
            idx = body.find(marker)
            if idx >= 0:
                body = body[idx + len(marker):]
                break
        # Stop at stats marker
        for end in ["--- Stats ---", "--- Output complete ---"]:
            idx = body.find(end)
            if idx >= 0:
                body = body[:idx]
                break
        body = body.strip()

        message = parse_response(body)
        log.info(f"CLI ran in {elapsed:.1f}s, response role={message['role']} "
                 f"has_tool_call={'tool_calls' in message}")
        log.debug(f"Raw body (first 300 chars): {body[:300]}")

        # Dump raw I/O if requested (debug)
        if Handler.raw_log_dir:
            try:
                import os as _os
                _os.makedirs(Handler.raw_log_dir, exist_ok=True)
                Handler.req_counter += 1
                fname = _os.path.join(Handler.raw_log_dir,
                                      f"req-{Handler.req_counter:04d}.txt")
                with open(fname, "w") as f:
                    f.write("=== PROMPT ===\n")
                    f.write(prompt)
                    f.write("\n\n=== RAW STDOUT ===\n")
                    f.write(res.stdout)
                    f.write("\n\n=== EXTRACTED BODY ===\n")
                    f.write(body)
                    f.write("\n\n=== PARSED MESSAGE ===\n")
                    f.write(json.dumps(message, indent=2, ensure_ascii=False))
                log.info(f"raw dump → {fname}")
            except Exception as e:
                log.warning(f"raw log failed: {e}")

        reply = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.get("model", "diffusiongemma-26B-A4B-it"),
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if "tool_calls" in message else "stop",
            }],
            "usage": {
                "prompt_tokens": len(prompt) // 4,   # rough estimate
                "completion_tokens": len(body) // 4,
                "total_tokens": (len(prompt) + len(body)) // 4,
            },
        }
        self._json_reply(200, reply)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cli", default=DEFAULT_CLI, help="Path to gemma4-cli binary")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--max-blocks", type=int, default=DEFAULT_MAX_BLOCKS,
                   help="Diffusion max canvas blocks (256 tokens each)")
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    p.add_argument("--raw-log-dir", default=None,
                   help="If set, dump (prompt, raw stdout, parsed message) of each request to this dir")
    args = p.parse_args()

    Handler.cli_path = args.cli
    Handler.max_blocks = args.max_blocks
    Handler.timeout = args.timeout
    Handler.raw_log_dir = args.raw_log_dir

    log.info(f"DiffusionGemma OpenAI proxy listening on http://localhost:{args.port}")
    log.info(f"  CLI: {args.cli}")
    log.info(f"  max_blocks: {args.max_blocks} ({args.max_blocks * 256} tokens output max)")
    log.info(f"  timeout: {args.timeout}s per request")

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
