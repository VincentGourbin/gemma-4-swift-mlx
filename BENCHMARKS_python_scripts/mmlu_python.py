"""MMLU 5-shot eval cote Python mlx_vlm (supporte 4-10 choix A-J)."""
import json, sys, time, argparse
import mlx.core as mx
from mlx_vlm import load

LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

def build_prefix(subj, dev_items):
    p = f"The following are multiple choice questions (with answers) about {subj.replace('_', ' ')}.\n\n"
    for d in dev_items:
        p += f"{d['q']}\n"
        for i, c in enumerate(d['c'][:len(LETTERS)]):
            p += f"{LETTERS[i]}. {c}\n"
        p += f"Answer: {LETTERS[d['a']]}\n\n"
    return p

def predict(model, tokenizer, prompt, candidate_tokens):
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array(tokens).reshape(1, -1)
    out = model.language_model(input_ids) if hasattr(model, 'language_model') else model(input_ids)
    out = out.logits if hasattr(out, 'logits') else out
    last = out[0, -1]
    best_idx, best_logit = 0, -1e9
    for i, tok in enumerate(candidate_tokens):
        l = float(last[tok].item())
        if l > best_logit: best_idx, best_logit = i, l
    return best_idx

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset", default="/tmp/mmlu_pro_5shot.json")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with open(args.dataset) as f:
        ds = json.load(f)
    questions = ds["questions"]
    if args.limit: questions = questions[:args.limit]

    print(f"Loading {args.model_path}...")
    t0 = time.perf_counter()
    model, processor = load(args.model_path)
    print(f"Loaded in {time.perf_counter()-t0:.1f}s")

    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor

    # Token IDs " A" ... " J"
    letter_tokens = [tokenizer.encode(f" {l}")[-1] for l in LETTERS]
    print(f"Tokens: " + " ".join(f"{LETTERS[i]}={letter_tokens[i]}" for i in range(len(LETTERS))))

    prefixes = {s: build_prefix(s, info["dev"]) for s, info in ds["subjects"].items()}

    correct = 0
    t_start = time.perf_counter()
    for idx, item in enumerate(questions):
        prefix = prefixes.get(item["subject"], "")
        prompt = prefix + item["question"] + "\n"
        for i, c in enumerate(item["choices"][:len(LETTERS)]):
            prompt += f"{LETTERS[i]}. {c}\n"
        prompt += "Answer:"
        n = min(len(item["choices"]), len(LETTERS))
        pred = predict(model, tokenizer, prompt, letter_tokens[:n])
        if pred == item["answer"]: correct += 1
        if (idx + 1) % 20 == 0:
            elapsed = time.perf_counter() - t_start
            eta = elapsed / (idx + 1) * (len(questions) - idx - 1)
            print(f"[{idx+1}/{len(questions)}] {correct}/{idx+1} = {correct/(idx+1)*100:.1f}% | {elapsed:.0f}s, ETA {eta:.0f}s")
    print(f"\n=== Python: {correct}/{len(questions)} = {correct/len(questions)*100:.1f}% ===")

if __name__ == "__main__":
    main()
