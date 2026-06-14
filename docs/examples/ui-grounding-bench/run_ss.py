#!/usr/bin/env python3
"""ScreenSpot GUI Grounding bench.

For each case: ask model to predict click coordinates (x, y) in normalized [0, 1].
Score: predicted point inside ground-truth bbox.
"""
import sys, json, os, subprocess, time, re
from pathlib import Path

CLI = "/Users/vincent/Developpements/gemma-4-swift-mlx/.build/xcode/Build/Products/Debug/gemma4-cli"
CASES = json.load(open("/tmp/screenspot/sample/meta.json"))

def build_prompt(case):
    return (
        f"Look at this UI screenshot ({case['width']}x{case['height']} pixels).\n"
        f"Goal: {case['instruction']}\n\n"
        "Where on the screen should I click to accomplish this goal? "
        "Respond with the predicted click position in normalized coordinates between 0 and 1, "
        "in the exact format:\n"
        "  CLICK: (x=0.XX, y=0.XX)\n\n"
        "where x is the horizontal position (0=left, 1=right) and y is the vertical position "
        "(0=top, 1=bottom). Reply with only the CLICK: line, no extra explanation."
    )

def run_model(prompt, image_path, kind):
    if kind == "diff":
        cmd = [CLI, "diffusion", "--include-vision", "--image", image_path, "--max-blocks", "1", prompt]
    elif kind == "a4b":
        cmd = [CLI, "describe",
               "--model-path", os.path.expanduser("~/Library/Caches/models/mlx-community/gemma-4-26b-a4b-it-4bit"),
               "--image", image_path,
               "--prompt", prompt,
               "--max-tokens", "60", "--temperature", "0.1"]
    elif kind == "e4b":
        cmd = [CLI, "describe",
               "--model-path", os.path.expanduser("~/Library/Caches/models/mlx-community/gemma-4-e4b-it-4bit"),
               "--image", image_path,
               "--prompt", prompt,
               "--max-tokens", "60", "--temperature", "0.1"]
    start = time.time()
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        return res.stdout + res.stderr, time.time() - start
    except subprocess.TimeoutExpired:
        return "TIMEOUT", time.time() - start

def extract_output(text, kind):
    if kind == "diff":
        if "[canvas 0]" in text:
            s = text.index("[canvas 0]") + len("[canvas 0]")
            e = text.find("--- Output complete ---", s)
            if e == -1: e = text.find("--- Stats ---", s)
            return text[s:e].strip() if e > s else text[s:].strip()
        return text.strip()
    else:
        # describe format
        if "--- Generation multimodale ---" in text:
            gen = text.index("--- Generation multimodale ---")
            lines = text[gen + 30:].split('\n')
            collected, past = [], False
            for line in lines:
                if line.strip() == '---' and not past: past = True; continue
                if not past: continue
                if line.strip() == '--- Stats ---': break
                collected.append(line)
            return '\n'.join(collected).strip()
        return text.strip()

def parse_click(output):
    """Extract (x, y) from output. Tries multiple formats."""
    output = re.sub(r'<eos>+', ' ', output)
    output = re.sub(r'<turn\|>|<\|turn>', ' ', output)

    # CLICK: (x=0.XX, y=0.XX)
    m = re.search(r'CLICK\s*:?\s*\(?\s*x\s*=\s*([\d.]+)\s*,\s*y\s*=\s*([\d.]+)', output, re.IGNORECASE)
    if m:
        return float(m.group(1)), float(m.group(2))

    # (x: 0.XX, y: 0.XX)
    m = re.search(r'x\s*[=:]\s*([\d.]+)\s*,\s*y\s*[=:]\s*([\d.]+)', output, re.IGNORECASE)
    if m:
        return float(m.group(1)), float(m.group(2))

    # (0.XX, 0.XX) or just two floats
    m = re.search(r'\(?\s*([\d.]+)\s*[,;]\s*([\d.]+)\s*\)?', output)
    if m:
        x, y = float(m.group(1)), float(m.group(2))
        # Si valeurs > 1, probablement en pixels — on les saute (besoin de dimensions)
        if x <= 1 and y <= 1:
            return x, y

    return None

def score_case(case, click):
    """Click inside bbox ?"""
    if click is None:
        return False
    x, y = click
    bb = case['bbox']  # [x1, y1, x2, y2] normalized
    return bb[0] <= x <= bb[2] and bb[1] <= y <= bb[3]

def main():
    kind = sys.argv[1]
    results_dir = Path(f"/tmp/screenspot/results/{kind}")
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running ScreenSpot on {kind}: {len(CASES)} cases", flush=True)
    summary = []
    for i, case in enumerate(CASES):
        prompt = build_prompt(case)
        out_file = results_dir / f"{i:03d}.txt"

        if out_file.exists():
            raw = out_file.read_text()
            elapsed = 0
        else:
            raw, elapsed = run_model(prompt, case['image'], kind)
            out_file.write_text(raw)

        output = extract_output(raw, kind)
        click = parse_click(output)
        correct = score_case(case, click)

        summary.append({
            'idx': i,
            'data_source': case['data_source'],
            'data_type': case['data_type'],
            'instruction': case['instruction'][:40],
            'bbox': case['bbox'],
            'click': click,
            'correct': correct,
            'elapsed': elapsed,
            'output': output[:100].replace('\n', ' ')
        })
        if (i+1) % 10 == 0 or i == 0:
            status = 'OK' if correct else 'FAIL'
            print(f"  [{i+1:03d}/{len(CASES)}] {case['data_source']:<8} {case['data_type']:<5} click={click} {status:<5} {elapsed:.1f}s", flush=True)

    # Aggregate
    by_source = {}
    by_type = {}
    by_st = {}  # source × type
    total_c = 0
    total_time = 0
    for s in summary:
        for d, k in [(by_source, s['data_source']), (by_type, s['data_type'])]:
            d.setdefault(k, {'c': 0, 't': 0})
            d[k]['t'] += 1
            if s['correct']: d[k]['c'] += 1
        sk = f"{s['data_source']}/{s['data_type']}"
        by_st.setdefault(sk, {'c': 0, 't': 0})
        by_st[sk]['t'] += 1
        if s['correct']: by_st[sk]['c'] += 1
        if s['correct']: total_c += 1
        total_time += s['elapsed']

    print(f"\n=== {kind.upper()} ScreenSpot ===")
    print(f"Overall: {total_c}/{len(summary)} = {100*total_c/len(summary):.1f}%")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"\nBy source:")
    for src, s in by_source.items():
        print(f"  {src:<10} {s['c']:>3}/{s['t']:<3} = {100*s['c']/s['t']:.0f}%")
    print(f"\nBy type:")
    for t, s in by_type.items():
        print(f"  {t:<10} {s['c']:>3}/{s['t']:<3} = {100*s['c']/s['t']:.0f}%")
    print(f"\nBy source × type:")
    for sk, s in sorted(by_st.items()):
        print(f"  {sk:<15} {s['c']:>3}/{s['t']:<3} = {100*s['c']/s['t']:.0f}%")

    with open(f'/tmp/screenspot/results/{kind}_summary.json', 'w') as f:
        json.dump({'kind': kind, 'overall': f"{total_c}/{len(summary)}",
                   'total_time_s': total_time,
                   'by_source': by_source, 'by_type': by_type, 'by_source_type': by_st,
                   'details': summary}, f, indent=2)

if __name__ == '__main__':
    main()
