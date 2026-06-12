"""Telecharge un subset de MMLU pour eval Swift. Stratifie sur ~10 sujets."""
import json, random, sys
from datasets import load_dataset

random.seed(42)
N_PER_SUBJECT = 10
SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "college_biology", "college_chemistry", "elementary_mathematics",
    "global_facts", "high_school_geography", "world_religions"
]

out = []
for subj in SUBJECTS:
    ds = load_dataset("cais/mmlu", subj, split="test")
    indices = random.sample(range(len(ds)), min(N_PER_SUBJECT, len(ds)))
    for i in indices:
        item = ds[i]
        out.append({
            "subject": subj,
            "question": item["question"],
            "choices": item["choices"],
            "answer": item["answer"],  # 0..3
        })

with open("mmlu_mini.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"Sauve {len(out)} questions dans mmlu_mini.json")
