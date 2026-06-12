import json, random
from datasets import load_dataset
random.seed(42)
N_PER_CAT = 15
val = load_dataset("TIGER-Lab/MMLU-Pro", split="validation")
test = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
val_by_cat = {}
for v in val:
    val_by_cat.setdefault(v['category'], []).append({
        "q": v['question'], "c": v['options'], "a": v['answer_index'],
        "cot": v['cot_content']
    })
by_cat = {}
for i, t in enumerate(test):
    by_cat.setdefault(t['category'], []).append(i)
questions = []
for cat, idxs in by_cat.items():
    sample = random.sample(idxs, min(N_PER_CAT, len(idxs)))
    for i in sample:
        t = test[i]
        questions.append({
            "subject": t['category'],
            "question": t['question'],
            "choices": t['options'],
            "answer": t['answer_index'],
        })
data = {"subjects": {cat: {"dev": items} for cat, items in val_by_cat.items()},
        "questions": questions}
with open("mmlu_pro_cot.json", "w") as f:
    json.dump(data, f, indent=2)
print(f"Sauve {len(questions)} test questions, {len(data['subjects'])} categories (avec cot_content)")
