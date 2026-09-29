#!/usr/bin/env python3
"""Jeu MMLU de `gemma4-cli eval-mmlu` (K-34) : 20 questions par sujet (57 sujets = 1 140,
IC95 ~ +/-3 pts au lieu de +/-10 sur 100), tirees avec une graine fixe dans cais/mmlu (test),
et les 5 exemples « dev » de chaque sujet pour le 5-shot.

  python3 Scripts/quality/mmlu-sample.py /Volumes/Lexar/datasets/mmlu
produit <dir>/mmlu_5shot_1140.json (+ SHA-256 affiche).
"""
import hashlib, json, random, sys
from collections import defaultdict
from pathlib import Path
import pyarrow.parquet as pq

root = Path(sys.argv[1]); per_subject = 20
dev = pq.read_table(root / "dev.parquet").to_pylist()
test = pq.read_table(root / "test.parquet").to_pylist()
subjects = defaultdict(lambda: {"dev": []})
for d in dev:
    subjects[d["subject"]]["dev"].append({"q": d["question"], "c": d["choices"], "a": d["answer"]})
by_subject = defaultdict(list)
for t in test: by_subject[t["subject"]].append(t)
rng = random.Random(0)
questions = []
for subject in sorted(by_subject):
    for t in rng.sample(by_subject[subject], min(per_subject, len(by_subject[subject]))):
        questions.append({"subject": subject, "question": t["question"], "choices": t["choices"], "answer": t["answer"]})
out = root / f"mmlu_5shot_{len(questions)}.json"
out.write_text(json.dumps({"subjects": subjects, "questions": questions}, ensure_ascii=False))
print(f"{len(questions)} questions, {len(subjects)} sujets -> {out}")
print("sha256", hashlib.sha256(out.read_bytes()).hexdigest())
