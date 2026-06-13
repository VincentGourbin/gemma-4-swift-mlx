#!/usr/bin/env python3
import sys, json, re
from pathlib import Path

def normalize(s): 
    s = s.lower().strip()
    s = re.sub(r'[^a-z0-9\sÀ-￿]+', ' ', s)
    return re.sub(r'\s+', ' ', s)

def extract(text):
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

def score(answers, output):
    norm = normalize(output)
    return any(normalize(a) in norm for a in answers)

def main():
    kind = sys.argv[1] if len(sys.argv) > 1 else "diff"
    meta = json.load(open("/tmp/ocrbench/sample300/meta.json"))
    rd = Path(f"/tmp/ocrbench/results300/{kind}")
    
    by_type = {}
    overall = {'c': 0, 't': 0}
    
    for e in meta:
        out_file = rd / f"{e['idx']:03d}.txt"
        if not out_file.exists(): continue
        text = out_file.read_text(errors='ignore')
        ok = score(e['answer'], extract(text))
        qt = e['question_type']
        by_type.setdefault(qt, {'c': 0, 't': 0})
        by_type[qt]['t'] += 1
        overall['t'] += 1
        if ok:
            by_type[qt]['c'] += 1
            overall['c'] += 1
    
    print(f"=== {kind.upper()} ===")
    print(f"Overall: {overall['c']}/{overall['t']} = {100*overall['c']/max(1,overall['t']):.1f}%")
    print()
    for qt in sorted(by_type):
        s = by_type[qt]
        print(f"  {qt:<50} {s['c']:>3}/{s['t']:<3} = {100*s['c']/s['t']:>4.0f}%")

main()
