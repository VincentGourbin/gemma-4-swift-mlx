"""MMLU Pro 5-shot CoT eval (cache fix)."""
import json, time, argparse, re
import mlx.core as mx
from mlx_vlm import load

LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

def build_prefix(subj, dev_items):
    p = f"The following are multiple choice questions (with answers) about {subj.replace('_', ' ')}.\n\n"
    for d in dev_items:
        p += f"{d['q']}\n"
        for i, c in enumerate(d['c'][:len(LETTERS)]):
            p += f"{LETTERS[i]}. {c}\n"
        cot = d.get('cot') or ''
        if cot:
            p += f"{cot}\n\n"
        else:
            p += f"Answer: {LETTERS[d['a']]}\n\n"
    return p

def parse_letter(text, n):
    valid = LETTERS[:n]
    low = text.lower()
    for m in ["answer is (", "answer is *", "answer is **", "answer is "]:
        i = low.find(m)
        if i >= 0:
            tail = text[i + len(m):i + len(m) + 5]
            for ch in tail:
                if ch in valid: return valid.index(ch)
    matches = re.findall(r"\(([A-J])\)", text)
    if matches:
        for ch in reversed(matches):
            if ch in valid: return valid.index(ch)
    return -1

def predict_cot(lm, tokenizer, prompt, n_choices, max_tokens):
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array(tokens).reshape(1, -1)
    cache = lm.make_cache()
    out = lm(input_ids, cache=cache)
    logits = out.logits if hasattr(out, 'logits') else out
    nxt = int(mx.argmax(logits[0, -1], axis=-1).item())
    gen = [nxt]
    for _ in range(max_tokens - 1):
        if nxt in (1, 106): break
        out = lm(mx.array([nxt]).reshape(1, 1), cache=cache)
        logits = out.logits if hasattr(out, 'logits') else out
        nxt = int(mx.argmax(logits[0, 0], axis=-1).item())
        gen.append(nxt)
        if len(gen) % 32 == 0:
            if "the answer is" in tokenizer.decode(gen).lower():
                break
    return parse_letter(tokenizer.decode(gen), n_choices)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset", default="/tmp/mmlu_pro_cot.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=512)
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
    lm = model.language_model

    prefixes = {s: build_prefix(s, info["dev"]) for s, info in ds["subjects"].items()}
    correct = 0
    t_start = time.perf_counter()
    for idx, item in enumerate(questions):
        prefix = prefixes.get(item["subject"], "")
        prompt = prefix + item["question"] + "\n"
        for i, c in enumerate(item["choices"][:len(LETTERS)]):
            prompt += f"{LETTERS[i]}. {c}\n"
        prompt += "A: Let's think step by step."
        n = min(len(item["choices"]), len(LETTERS))
        pred = predict_cot(lm, tokenizer, prompt, n, args.max_tokens)
        if pred == item["answer"]: correct += 1
        if (idx + 1) % 10 == 0:
            elapsed = time.perf_counter() - t_start
            eta = elapsed / (idx + 1) * (len(questions) - idx - 1)
            print(f"[{idx+1}/{len(questions)}] {correct}/{idx+1} = {correct/(idx+1)*100:.1f}% | {elapsed:.0f}s, ETA {eta:.0f}s")
    print(f"\n=== Python CoT: {correct}/{len(questions)} = {correct/len(questions)*100:.1f}% ===")

if __name__ == "__main__":
    main()
