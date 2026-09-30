#!/usr/bin/env python3
"""Jeu de la porte TB3 (LoRA multimodal, K-28/K-31) : LaTeX-OCR, 500 exemples d'entrainement
+ 50 de validation, tires avec une graine fixe dans unsloth/LaTeX_OCR (train / test).

  python3 Scripts/quality/latex-ocr-sample.py /Volumes/Lexar/datasets/LaTeX_OCR
produit <dir>/tb3/{train,valid}.jsonl, <dir>/tb3/images/*.png et <dir>/tb3/SHA256SUMS
(format de `gemma4-cli lora train --multimodal`, comme l'exemple du README).
"""
import hashlib, io, json, random, sys
from pathlib import Path
import pyarrow.parquet as pq
from PIL import Image

root = Path(sys.argv[1]); out = root / "tb3"; (out / "images").mkdir(parents=True, exist_ok=True)
PROMPT = "Convert this mathematical expression to LaTeX."

def sample(split, n, seed):
    table = pq.read_table(root / f"{split}.parquet")
    idx = sorted(random.Random(seed).sample(range(table.num_rows), n))
    rows = table.take(idx).to_pylist()
    lines = []
    for i, row in zip(idx, rows):
        name = f"images/{split}_{i:06d}.png"
        path = out / name
        if not path.exists():
            Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB").save(path)
        lines.append(json.dumps({"messages": [{"role": "user", "content": PROMPT},
                                              {"role": "assistant", "content": row["text"]}],
                                 "image": name}, ensure_ascii=False))
    return lines

(out / "train.jsonl").write_text("\n".join(sample("train", 500, 0)) + "\n")
(out / "valid.jsonl").write_text("\n".join(sample("test", 50, 1)) + "\n")
sums = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(out)}"
        for p in sorted(out.rglob("*")) if p.is_file() and p.name != "SHA256SUMS"]
(out / "SHA256SUMS").write_text("\n".join(sums) + "\n")
print(f"{len(sums)} fichiers -> {out}")
