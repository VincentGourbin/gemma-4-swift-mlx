#!/usr/bin/env python3
"""BFCL 100 cases bench. Same as run_bfcl.py but with parallel support."""
import sys, json, os, subprocess, time, re
from pathlib import Path

CLI = "/Users/vincent/Developpements/gemma-4-swift-mlx/.build/xcode/Build/Products/Debug/gemma4-cli"
CASES = json.load(open("/tmp/bfcl/sample100/cases.json"))

def build_prompt(case):
    funcs = case['function']
    funcs_str = json.dumps(funcs, indent=2)
    user_q = case['question'][0][0]['content']
    fmt = (
        "  FUNCTION_CALL: function_name(arg1=value1, arg2=value2, ...)\n"
        "  FUNCTION_CALL: name1(...) AND FUNCTION_CALL: name2(...)  # for parallel calls\n"
        "  NO_CALL: <brief explanation>\n"
    )
    return (
        "You are a function-calling assistant. "
        "Given the user's question and the available functions below, decide whether to call function(s).\n\n"
        f"Available functions:\n{funcs_str}\n\n"
        f"User question: {user_q}\n\n"
        f"Respond in ONE of these formats:\n{fmt}\n"
        "Reply with only the FUNCTION_CALL line(s) or NO_CALL line, no extra explanation."
    )

def run_model(prompt, kind):
    if kind == "diff":
        cmd = [CLI, "diffusion", "--include-vision", "--max-blocks", "1", prompt]
    elif kind == "a4b":
        cmd = [CLI, "generate",
               "--model-path", os.path.expanduser("~/Library/Caches/models/mlx-community/gemma-4-26b-a4b-it-4bit"),
               "--max-tokens", "200", "--temperature", "0.1",
               "--system", "You are a helpful assistant.", prompt]
    elif kind == "e4b":
        cmd = [CLI, "generate",
               "--model-path", os.path.expanduser("~/Library/Caches/models/mlx-community/gemma-4-e4b-it-4bit"),
               "--max-tokens", "200", "--temperature", "0.1",
               "--system", "You are a helpful assistant.", prompt]
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
        if "--- Stats ---" in text:
            end = text.find("--- Stats ---")
            before_stats = text[:end]
            lines = before_stats.split('\n')
            sep_idx = None
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].strip() == '---':
                    sep_idx = i
                    break
            if sep_idx is not None:
                return '\n'.join(lines[sep_idx + 1:]).strip()
        return text.strip()

def parse_fc(output):
    """Returns list of (name, args) or empty list if NO_CALL or invalid."""
    output = re.sub(r'<eos>+', ' ', output)
    output = re.sub(r'<turn\|>|<\|turn>', ' ', output)

    calls = []
    for m in re.finditer(r'FUNCTION_CALL\s*:\s*([\w\.]+)\s*\((.*?)\)(?=\s*(?:AND|FUNCTION_CALL|NO_CALL|$|\n))',
                        output, re.IGNORECASE | re.DOTALL):
        name = m.group(1)
        args_raw = m.group(2)
        args = {}
        for pair in re.finditer(r'(\w+)\s*=\s*("([^"]*)"|\'([^\']*)\'|([\d.\-+e]+)|(\[[^\]]*\])|(\{[^}]*\})|(true|false|True|False|None|null)|([\w\.]+))',
                                args_raw):
            k = pair.group(1)
            v = pair.group(3) or pair.group(4) or pair.group(5) or pair.group(6) or pair.group(7) or pair.group(8) or pair.group(9)
            args[k] = v
        calls.append((name, args))

    if calls:
        return calls
    if re.search(r'NO_CALL', output, re.IGNORECASE):
        return []  # explicit no-call
    return None  # invalid

def score_case(case, parsed):
    """parsed = list of (name, args) or None (invalid) or [] (no_call)"""
    cat = case['category']
    gt = case['ground_truth']

    if cat == 'irrelevance':
        return parsed == []  # must explicitly NO_CALL (empty list OK)

    if parsed is None or parsed == []:
        return False

    # gt is list of {name: {arg: [accepted]}}
    if not gt:
        return False

    # For parallel: gt has multiple entries; for simple/multiple: 1 entry
    expected_calls = gt  # list of dicts

    # Match each parsed call to an expected call (greedy by name)
    matched = []
    remaining_expected = list(expected_calls)
    for (pname, pargs) in parsed:
        for ge in remaining_expected:
            (ename, eargs) = list(ge.items())[0]
            if pname == ename or pname.endswith(ename.split('.')[-1]):
                # Check args : at least half match
                arg_match = 0
                for arg_name, accepted in eargs.items():
                    if arg_name not in pargs:
                        continue
                    pv = str(pargs[arg_name]).strip().strip('"\'').lower()
                    for acc in accepted:
                        if str(acc).strip().lower() == pv or str(acc).strip().lower() in pv:
                            arg_match += 1
                            break
                threshold = max(1, len(eargs) // 2)
                if arg_match >= threshold:
                    matched.append((ename, ge))
                    remaining_expected.remove(ge)
                    break

    if cat == 'parallel':
        # all expected must be matched (or at least majority)
        return len(matched) >= max(1, len(expected_calls) * 2 // 3)
    else:
        # simple or multiple : at least 1 match
        return len(matched) >= 1

def main():
    kind = sys.argv[1]
    results_dir = Path(f"/tmp/bfcl/results100/{kind}")
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running BFCL 100 on {kind}: {len(CASES)} cases", flush=True)
    summary = []
    for i, case in enumerate(CASES):
        prompt = build_prompt(case)
        out_file = results_dir / f"{i:03d}_{case['id']}.txt"

        if out_file.exists():
            raw = out_file.read_text()
            elapsed = 0
        else:
            raw, elapsed = run_model(prompt, kind)
            out_file.write_text(raw)

        output = extract_output(raw, kind)
        parsed = parse_fc(output)
        correct = score_case(case, parsed)

        n_calls = len(parsed) if parsed else 0
        summary.append({
            'id': case['id'], 'category': case['category'],
            'n_calls': n_calls, 'correct': correct,
            'elapsed': elapsed,
            'output_snippet': output[:120].replace('\n', ' ')
        })
        status = 'OK' if correct else 'FAIL'
        if (i+1) % 10 == 0 or i == 0:
            print(f"  [{i+1:03d}/{len(CASES)}] {case['category']:<12} {status:<5} {elapsed:.1f}s", flush=True)

    # Aggregate
    by_cat = {}
    for s in summary:
        by_cat.setdefault(s['category'], {'c': 0, 't': 0, 'time': 0})
        by_cat[s['category']]['t'] += 1
        by_cat[s['category']]['time'] += s['elapsed']
        if s['correct']:
            by_cat[s['category']]['c'] += 1

    total_c = sum(s['correct'] for s in summary)
    total_t = sum(s['elapsed'] for s in summary)
    print(f"\n=== {kind.upper()} BFCL 100 ===")
    print(f"Overall: {total_c}/{len(summary)} = {100*total_c/len(summary):.1f}%")
    print(f"Total time: {total_t:.1f}s ({total_t/60:.1f} min)")
    for cat, stats in by_cat.items():
        avg_t = stats['time']/stats['t'] if stats['t'] else 0
        print(f"  {cat:<15} {stats['c']:>3}/{stats['t']:<3} = {100*stats['c']/stats['t']:>4.0f}%  avg {avg_t:.1f}s/case")

    with open(f'/tmp/bfcl/results100/{kind}_summary.json', 'w') as f:
        json.dump({'kind': kind, 'overall': f"{total_c}/{len(summary)}",
                   'total_time_s': total_t, 'by_category': by_cat,
                   'details': summary}, f, indent=2)

if __name__ == '__main__':
    main()
