#!/usr/bin/env python3
"""
Score OCRBench 1000 + extract perf stats (tokens, time, tok/s).

Usage:
  python3 score1000.py <model_kind>          # one model
  python3 score1000.py compare               # all 3 models + perf comparison
"""
import sys, json, re, statistics
from pathlib import Path

def normalize(s):
    s = s.lower().strip()
    s = re.sub(r'[^a-z0-9\sÀ-￿]+', ' ', s)
    return re.sub(r'\s+', ' ', s)

def extract_output(text):
    if "[canvas 0]" in text:
        s = text.index("[canvas 0]") + len("[canvas 0]")
        e = text.find("--- Output complete ---", s)
        if e == -1: e = text.find("--- Stats ---", s)
        return text[s:e].strip() if e > s else text[s:].strip()
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
    if "--- Stats ---" in text:
        return text[:text.find("--- Stats ---")].strip()
    return text.strip()

def extract_perf(text, kind):
    """
    Parse the Stats section.
    Returns dict with: tokens, time_s, tok_per_s

    Diffusion format:
      --- Stats ---
      Tokens generes : 256
      Canvases : 1
      Decoder forwards : 7
      Temps : 6.54s
      Vitesse : 39.1 tok/s (0.93s/step)
      GPU pic : 50864 Mo

    Describe AR format:
      --- Stats ---
      Tokens: 20
      Temps: 2.70s, Vitesse: 7.4 t/s
      GPU pic: 6035 Mo
    """
    if kind == "diff":
        # DiffusionGemma format
        m_tok = re.search(r'Tokens generes\s*:\s*(\d+)', text)
        m_time = re.search(r'Temps\s*:\s*([\d.]+)\s*s', text)
        m_speed = re.search(r'Vitesse\s*:\s*([\d.]+)\s*tok/s', text)
        m_forwards = re.search(r'Decoder forwards\s*:\s*(\d+)', text)
        if not (m_tok and m_time and m_speed):
            return None
        return {
            'tokens': int(m_tok.group(1)),
            'time_s': float(m_time.group(1)),
            'tok_per_s': float(m_speed.group(1)),
            'decoder_forwards': int(m_forwards.group(1)) if m_forwards else None,
        }
    else:
        # describe AR format
        m_tok = re.search(r'Tokens\s*:\s*(\d+)', text)
        m_time = re.search(r'Temps\s*:\s*([\d.]+)\s*s', text)
        m_speed = re.search(r'Vitesse\s*:\s*([\d.]+)\s*t/s', text)
        if not (m_tok and m_time and m_speed):
            return None
        return {
            'tokens': int(m_tok.group(1)),
            'time_s': float(m_time.group(1)),
            'tok_per_s': float(m_speed.group(1)),
            'decoder_forwards': None,
        }

def score_one_kind(kind):
    meta = json.load(open("/tmp/ocrbench/sample1000/meta.json"))
    rd = Path(f"/tmp/ocrbench/results1000/{kind}")

    by_type = {}
    overall = {'c': 0, 't': 0}
    perfs = []

    for e in meta:
        out_file = rd / f"{e['idx']:04d}.txt"
        if not out_file.exists(): continue
        text = out_file.read_text(errors='ignore')
        ok = score_match(e['answer'], extract_output(text))
        qt = e['question_type']
        by_type.setdefault(qt, {'c': 0, 't': 0})
        by_type[qt]['t'] += 1
        overall['t'] += 1
        if ok:
            by_type[qt]['c'] += 1
            overall['c'] += 1
        p = extract_perf(text, kind)
        if p: perfs.append(p)

    return overall, by_type, perfs

def score_match(answers, output):
    norm = normalize(output)
    return any(normalize(a) in norm for a in answers)

def fmt_stats(values, unit=""):
    if not values: return "—"
    avg = statistics.mean(values)
    med = statistics.median(values)
    p95 = sorted(values)[int(len(values) * 0.95)] if len(values) > 1 else values[0]
    return f"avg={avg:.1f}{unit} med={med:.1f}{unit} p95={p95:.1f}{unit}"

def report_one(kind):
    overall, by_type, perfs = score_one_kind(kind)
    print(f"\n=== {kind.upper()} ===")
    print(f"Overall: {overall['c']}/{overall['t']} = {100*overall['c']/max(1,overall['t']):.1f}%")
    print()
    print(f"  {'Category':<50} {'Accuracy':<15}")
    print('  ' + '-' * 65)
    for qt in sorted(by_type):
        s = by_type[qt]
        acc = 100 * s['c'] / s['t']
        print(f"  {qt:<50} {s['c']:>4}/{s['t']:<4} = {acc:>4.0f}%")
    print()
    print(f"Perf stats (N={len(perfs)}):")
    print(f"  Tokens/run    : {fmt_stats([p['tokens'] for p in perfs])}")
    print(f"  Time/run      : {fmt_stats([p['time_s'] for p in perfs], 's')}")
    print(f"  Tok/s         : {fmt_stats([p['tok_per_s'] for p in perfs])}")
    if any(p.get('decoder_forwards') for p in perfs):
        print(f"  Forwards      : {fmt_stats([p['decoder_forwards'] for p in perfs if p['decoder_forwards']])}")

def compare():
    results = {}
    for kind in ['diff', 'a4b', 'e4b']:
        results[kind] = score_one_kind(kind)

    print("\n" + "=" * 80)
    print("COMPARAISON 3 MODELES — 1000 images")
    print("=" * 80)

    print(f"\n{'Model':<35} {'Score':<20} {'Accuracy':<10}")
    print('-' * 65)
    for kind in ['diff', 'a4b', 'e4b']:
        o, _, _ = results[kind]
        print(f"  {label(kind):<35} {o['c']:>4}/{o['t']:<4} ({100*o['c']/max(1,o['t']):.1f}%)")

    # By category
    all_types = set()
    for _, bt, _ in results.values():
        all_types.update(bt.keys())

    print(f"\n{'Category':<50} {'diff':<10} {'a4b':<10} {'e4b':<10}")
    print('-' * 80)
    for qt in sorted(all_types):
        line = f"  {qt[:48]:<50}"
        for kind in ['diff', 'a4b', 'e4b']:
            _, bt, _ = results[kind]
            if qt in bt:
                s = bt[qt]
                line += f"{s['c']:>3}/{s['t']:<3} ({100*s['c']/s['t']:>3.0f}%) "
            else:
                line += '  —          '
        print(line)

    # Perf comparison
    print(f"\n=== Performance Stats ===")
    for kind in ['diff', 'a4b', 'e4b']:
        _, _, perfs = results[kind]
        print(f"\n{label(kind)} (N={len(perfs)}):")
        print(f"  Time/img      : {fmt_stats([p['time_s'] for p in perfs], 's')}")
        print(f"  Tokens/img    : {fmt_stats([p['tokens'] for p in perfs])}")
        print(f"  Tok/s         : {fmt_stats([p['tok_per_s'] for p in perfs])}")
        if any(p.get('decoder_forwards') for p in perfs):
            fws = [p['decoder_forwards'] for p in perfs if p['decoder_forwards']]
            print(f"  Forwards      : {fmt_stats(fws)}")
        total_time = sum(p['time_s'] for p in perfs)
        print(f"  Total bench   : {total_time/60:.1f} min")

def label(kind):
    return {
        'diff': 'DiffusionGemma 26B-A4B bf16',
        'a4b':  'Gemma 4 26B-A4B 4-bit AR',
        'e4b':  'Gemma 4 E4B 4-bit AR',
    }[kind]

if __name__ == '__main__':
    arg = sys.argv[1] if len(sys.argv) > 1 else "compare"
    if arg == "compare":
        compare()
    else:
        report_one(arg)
