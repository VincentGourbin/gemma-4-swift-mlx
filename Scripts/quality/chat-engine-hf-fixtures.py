#!/usr/bin/env python3
"""Ids de reference HF pour Gemma4ChatEngine.promptIds (porte de K-36).

  python3 Scripts/quality/chat-engine-hf-fixtures.py /Volumes/Lexar/models/mlx-community/gemma-4-e2b-it-4bit
ecrit Tests/Gemma4SwiftTests/Fixtures/chat-engine-hf.json : pour chaque cas, les messages,
les outils, enable_thinking et les ids de `apply_chat_template` (transformers), avec chaque
<|image|> developpe en boi + image x 280 + eoi comme le processeur Gemma 4.
"""
import json, sys
from pathlib import Path
from transformers import AutoTokenizer

model = Path(sys.argv[1])
tok = AutoTokenizer.from_pretrained(model)
config = json.load(open(model / "config.json"))
image, boi, eoi = config["image_token_id"], config["boi_token_id"], config["eoi_token_id"]

weather = {"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}

cases = [
    {"name": "texte", "tools": [], "thinking": False, "messages": [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "What is the capital of France?"}]},
    {"name": "texte-pensee", "tools": [], "thinking": True, "messages": [
        {"role": "user", "content": "What is 17 * 23?"}]},
    {"name": "image", "tools": [], "thinking": False, "images": 1, "messages": [
        {"role": "user", "content": "<|image|>\nDescribe this image."}]},
    {"name": "outils", "tools": [weather], "thinking": False, "messages": [
        {"role": "user", "content": "Weather in Paris?"}]},
    {"name": "tour-outil", "tools": [weather], "thinking": False, "messages": [
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]},
        {"role": "tool", "content": "{\"temperature\": 18, \"sky\": \"clear\"}", "name": "get_weather", "tool_call_id": "call_1"}]},
]
out = []
for case in cases:
    ids = tok.apply_chat_template(case["messages"], tools=case["tools"] or None, add_generation_prompt=True,
                                  enable_thinking=case["thinking"], tokenize=True)
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    expanded = []
    for i in ids:
        expanded += [boi] + [image] * 280 + [eoi] if i == image else [i]
    out.append({**case, "ids": expanded})
path = Path(__file__).resolve().parents[2] / "Tests/Gemma4SwiftTests/Fixtures/chat-engine-hf.json"
json.dump(out, open(path, "w"), indent=1)
print(f"{len(out)} cas -> {path}")
