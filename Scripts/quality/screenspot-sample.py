#!/usr/bin/env python3
"""Reconstruit l'echantillon ScreenSpot-100 du bench de reference (79 % en bf16).

Les 100 cas sont identifies dans docs/examples/ui-grounding-bench/diff_summary.json
(debut d'instruction + bbox). Source : rootsautomation/ScreenSpot (parquet).

  python3 Scripts/quality/screenspot-sample.py /Volumes/Lexar/datasets/ScreenSpot
produit <dir>/sample/meta.json + <dir>/sample/img_XXX.png
"""
import io, json, sys
from pathlib import Path
import pyarrow.parquet as pq
from PIL import Image

root = Path(sys.argv[1])
repo = Path(__file__).resolve().parents[2]
wanted = json.load(open(repo / "docs/examples/ui-grounding-bench/diff_summary.json"))["details"]

rows = []
for f in sorted(root.glob("test-*.parquet")):
    rows += pq.read_table(f).to_pylist()

def same_bbox(a, b):
    return all(abs(x - y) < 1e-6 for x, y in zip(a, b))

out = root / "sample"
out.mkdir(exist_ok=True)
meta = []
for case in wanted:
    hits = [r for r in rows
            if r["instruction"][:40] == case["instruction"]
            and r["data_source"] == case["data_source"] and r["data_type"] == case["data_type"]
            and same_bbox(r["bbox"], case["bbox"])]
    if len(hits) != 1:
        sys.exit(f"cas {case['idx']} ({case['instruction']!r}) : {len(hits)} correspondance(s)")
    r = hits[0]
    img = Image.open(io.BytesIO(r["image"]["bytes"]))
    path = out / f"img_{case['idx']:03d}.png"
    if not path.exists():
        img.save(path)
    meta.append({"idx": case["idx"], "image": str(path), "width": img.width, "height": img.height,
                 "instruction": r["instruction"], "bbox": r["bbox"],
                 "data_source": r["data_source"], "data_type": r["data_type"]})
json.dump(meta, open(out / "meta.json", "w"), ensure_ascii=False, indent=1)
print(f"{len(meta)} cas -> {out / 'meta.json'}")
