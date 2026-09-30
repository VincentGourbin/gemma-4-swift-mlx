#!/usr/bin/env python3
"""Porte qualite LoRA (K-29) : eval E7 du « director » de Fluxforge Studio.

30 briefs tenus a l'ecart (jamais dans train/valid) ; pour chacun, prompt exact de
`director-tool prompt`, generation par gemma4-cli (base, ou base + adaptateur), puis
`director-tool validate`. Reference : director-v3 = 29/30 valides (base 17/30).

  python3 Scripts/quality/director-e7.py --condition adapter --adapter <dir> --out <dir>
  python3 Scripts/quality/director-e7.py --score-only <dir>   # re-noter des sorties existantes
  --base <dossier du modele>   autre base que E2B 6 bits (ex. E4B bf16, K-33)
"""
import argparse, json, subprocess, sys, time
from pathlib import Path

FLUXFORGE = Path("/Users/vincent/Developpements/Fluxforge Studio")
DIRECTOR = FLUXFORGE / "Scripts" / "director"
TOOL = FLUXFORGE / "Packages" / "DirectorCore" / ".build" / "out" / "Products" / "Debug" / "director-tool"
REPO = Path(__file__).resolve().parents[2]
CLI = REPO / ".build" / "xcode" / "Build" / "Products" / "Release" / "gemma4-cli"
BASE = Path("/Users/vincent/Pictures/FluxforgeStudio/Models/mlx-community/gemma-4-e2b-it-6bit")

sys.path.insert(0, str(DIRECTOR))
import teach  # group_env_path : environnement de validation du brief


def briefs():
    for path in sorted((DIRECTOR / "eval_holdout").glob("*.json")):
        brief = json.loads(path.read_text())
        env = teach.group_env_path(brief["env_profile"], bool(brief.get("subject_sheet")))
        yield brief, env


def prompt(brief, env):
    args = [str(TOOL), "prompt", "--brief", brief["brief"], "--env", str(env), "--language", brief["language"]]
    if brief.get("subject_sheet"): args += ["--subject-sheet", brief["subject_sheet"]]
    if brief.get("setting_sheet"): args += ["--setting-sheet", brief["setting_sheet"]]
    return json.loads(subprocess.run(args, capture_output=True, text=True, timeout=30, check=True).stdout)


def generate(system, user, adapter, base=BASE, max_tokens=2048):
    if adapter:
        args = [str(CLI), "lora", "generate", "--model-path", str(base), "--adapter-path", str(adapter)]
    else:
        args = [str(CLI), "generate", "--model-path", str(base)]
    args += ["--system", system, "--max-tokens", str(max_tokens), "--temperature", "0.3", user]
    out = subprocess.run(args, capture_output=True, text=True, timeout=400, check=True).stdout
    if "\n---\n" in out: out = out.split("\n---\n", 1)[1]
    elif "Generating...\n" in out: out = out.split("Generating...\n", 1)[1]
    return out.split("--- Stats ---", 1)[0].strip()


def valid(output_path, env):
    r = subprocess.run([str(TOOL), "validate", str(output_path), "--env", str(env)],
                       capture_output=True, text=True, timeout=30)
    return r.returncode == 0


def score(out_dir):
    ok, total, failures = 0, 0, []
    for brief, env in briefs():
        path = out_dir / f"{brief['id']}.txt"
        if not path.exists(): continue
        total += 1
        if valid(path, env): ok += 1
        else: failures.append(brief["id"])
    return {"valid": ok, "total": total, "failures": failures}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=["base", "adapter"])
    ap.add_argument("--adapter")
    ap.add_argument("--out")
    ap.add_argument("--score-only")
    ap.add_argument("--base", default=str(BASE))
    a = ap.parse_args()
    if a.score_only:
        print(json.dumps(score(Path(a.score_only)), ensure_ascii=False)); return
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    adapter = a.adapter if a.condition == "adapter" else None
    for i, (brief, env) in enumerate(briefs(), 1):
        path = out / f"{brief['id']}.txt"
        if path.exists(): continue
        p = prompt(brief, env); t0 = time.time()
        path.write_text(generate(p["system"], p["user"], adapter, Path(a.base)), encoding="utf-8")
        print(f"[{i}/30] {brief['id']} ({time.time() - t0:.0f} s)", flush=True)
    result = score(out)
    (out / "score.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
