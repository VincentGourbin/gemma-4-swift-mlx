"""Telecharge MMLU avec dev examples (pour 5-shot)."""
import json, random
from datasets import load_dataset

random.seed(42)
N_PER_SUBJECT = 10
SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "college_biology", "college_chemistry", "elementary_mathematics",
    "global_facts", "high_school_geography", "world_religions"
]

out = {"subjects": {}, "questions": []}
for subj in SUBJECTS:
    dev = load_dataset("cais/mmlu", subj, split="dev")  # 5 examples par sujet
    test = load_dataset("cais/mmlu", subj, split="test")
    dev_items = [dict(q=dev[i]["question"], c=dev[i]["choices"], a=dev[i]["answer"]) for i in range(len(dev))]
    out["subjects"][subj] = {"dev": dev_items}
    indices = random.sample(range(len(test)), min(N_PER_SUBJECT, len(test)))
    for i in indices:
        item = test[i]
        out["questions"].append({
            "subject": subj,
            "question": item["question"],
            "choices": item["choices"],
            "answer": item["answer"],
        })

with open("mmlu_5shot.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"Sauve {len(out['questions'])} test questions + {len(SUBJECTS)} subjects (5 dev each)")
