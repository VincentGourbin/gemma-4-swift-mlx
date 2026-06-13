#!/usr/bin/env python3
"""
Score OCRBench results.

Usage:
  python3 score.py <model_kind>

Reads /tmp/ocrbench/sample/meta.json (ground truth) and
/tmp/ocrbench/results/<model_kind>/<idx>.txt (CLI outputs).

Metric: case-insensitive substring match of any ground-truth answer
in the model output. Reports per-question_type and overall accuracy.
"""
import sys, json, os, re
from pathlib import Path

def normalize(s: str) -> str:
    """Lowercase + strip + remove special chars for matching."""
    s = s.lower().strip()
    s = re.sub(r'[^a-z0-9\sÀ-￿]+', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    return s

def extract_model_output(text: str) -> str:
    """Extract the actual answer from the CLI output (skip metadata)."""
    # diffusion CLI : tout apres "[canvas 0]" jusqu'a "--- Output complete ---"
    if "[canvas 0]" in text:
        try:
            start = text.index("[canvas 0]") + len("[canvas 0]")
            end = text.find("--- Output complete ---", start)
            if end == -1:
                end = text.find("--- Stats ---", start)
            return text[start:end].strip() if end > start else text[start:].strip()
        except ValueError:
            pass
    # describe CLI (AR multimodal) : sortie apres ligne "---" qui suit "  Prompt: ..."
    # Le pattern est :
    #   --- Generation multimodale ---
    #     Prompt: ...
    #   ---
    #   <reponse>
    #   --- Stats ---
    if "--- Generation multimodale ---" in text:
        try:
            # Trouver le 3eme "---" (apres Generation + Prompt)
            gen_idx = text.index("--- Generation multimodale ---")
            # Cherche le "---" suivant SUR SA PROPRE LIGNE
            search_from = gen_idx + len("--- Generation multimodale ---")
            lines = text[search_from:].split('\n')
            collected = []
            past_separator = False
            for line in lines:
                if line.strip() == '---' and not past_separator:
                    past_separator = True
                    continue
                if not past_separator:
                    continue
                if line.strip() == '--- Stats ---':
                    break
                collected.append(line)
            return '\n'.join(collected).strip()
        except ValueError:
            pass
    # generic fallback : skip header avant --- Stats ---
    if "--- Stats ---" in text:
        end = text.find("--- Stats ---")
        return text[:end].strip()
    return text.strip()

def score_one(answers: list[str], output: str) -> bool:
    """True if any ground-truth answer is contained in the normalized output."""
    norm_out = normalize(output)
    for ans in answers:
        if normalize(ans) in norm_out:
            return True
    return False

def main():
    model_kind = sys.argv[1] if len(sys.argv) > 1 else "diff"
    meta_path = Path("/tmp/ocrbench/sample/meta.json")
    results_dir = Path(f"/tmp/ocrbench/results/{model_kind}")

    if not meta_path.exists() or not results_dir.exists():
        print(f"ERROR: {meta_path} or {results_dir} not found")
        sys.exit(1)

    meta = json.load(meta_path.open())
    by_type: dict[str, dict[str, int]] = {}
    overall = {'correct': 0, 'total': 0}
    failures = []

    for entry in meta:
        idx = entry['idx']
        out_file = results_dir / f"{idx:03d}.txt"
        if not out_file.exists():
            print(f"[{idx:03d}] MISSING output")
            continue

        raw = out_file.read_text(errors='ignore')
        output = extract_model_output(raw)
        is_correct = score_one(entry['answer'], output)

        qt = entry['question_type']
        by_type.setdefault(qt, {'correct': 0, 'total': 0})
        by_type[qt]['total'] += 1
        overall['total'] += 1
        if is_correct:
            by_type[qt]['correct'] += 1
            overall['correct'] += 1
        else:
            failures.append({
                'idx': idx,
                'qt': qt,
                'question': entry['question'][:50],
                'expected': entry['answer'],
                'got': output[:100]
            })

    print(f"\n=== {model_kind.upper()} OCRBench Mini ===")
    print(f"Overall: {overall['correct']}/{overall['total']} = {100*overall['correct']/max(1,overall['total']):.1f}%")
    print()
    print(f"{'Category':<50} {'Acc':>10}")
    print('-' * 65)
    for qt in sorted(by_type):
        s = by_type[qt]
        acc = 100 * s['correct'] / s['total']
        print(f"{qt:<50} {s['correct']}/{s['total']} = {acc:>4.0f}%")

    print()
    print("Failures (first 10):")
    for f in failures[:10]:
        print(f"  [{f['idx']:03d}] [{f['qt'][:20]}] expected={f['expected']} got=\"{f['got'][:80]}\"")

if __name__ == '__main__':
    main()
