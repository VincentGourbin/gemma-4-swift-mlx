"""Banc de routage d'intentions : embeddings EmbeddingGemma 2 contre LLM Gemma 4 (generation).

Jeux : CLINC150 (en, 150 intentions + hors perimetre), MASSIVE fr-FR (60 intentions),
Speech-MASSIVE fr-FR (memes requetes que le dev MASSIVE, enregistrees : routage depuis l'audio).
Les donnees restent hors du depot (Speech-MASSIVE est CC BY-NC-SA) : DATA = dossier de travail.

  python -I routing-bench.py prepare DATA        # jeux, wav, specs d'embedding (lots de 4000)
  python -I routing-bench.py llm-inputs DATA     # entrees de gemma4-cli eval-route (apres les embeddings)
  python -I routing-bench.py report DATA OUT.jsonl

Les dossiers attendus sous DATA : clinc/plus/*.parquet, massive/1.1/data/fr-FR.jsonl,
speech-massive/fr-FR/*.parquet (hf_hub_download, archive MASSIVE 1.1).
"""
import collections
import glob
import json
import os
import random
import re
import sys
import time

import numpy as np

SEED = 0
SHOTS = [5, 10, "full"]
LLM_SAMPLES = {"clinc": 1100, "massive": 1000, "speech": 500}
TOP_K = 5
CHUNK = 4000


# --- Preparation -----------------------------------------------------------------------------

def load_clinc(data):
    import pyarrow.parquet as pq
    names = None
    out = {}
    for split in ["train", "validation", "test"]:
        path = glob.glob(f"{data}/clinc/plus/{split}-*.parquet")[0]
        table = pq.read_table(path)
        if names is None:
            meta = json.loads(table.schema.metadata[b"huggingface"])
            names = meta["info"]["features"]["intent"]["names"]
        out[split] = [{"text": r["text"], "label": names[r["intent"]]} for r in table.to_pylist()]
    return out


def load_massive(data):
    rows = [json.loads(l) for l in open(f"{data}/massive/1.1/data/fr-FR.jsonl")]
    out = collections.defaultdict(list)
    for r in rows:
        out[r["partition"]].append({"id": r["id"], "text": r["utt"], "label": r["intent"]})
    return dict(out)


def load_speech(data):
    import pyarrow.parquet as pq
    wav_dir = f"{data}/work/speech-wav"
    os.makedirs(wav_dir, exist_ok=True)
    out = []
    for path in sorted(glob.glob(f"{data}/speech-massive/fr-FR/validation-*.parquet")):
        for r in pq.read_table(path, columns=["id", "intent_str", "utt", "audio"]).to_pylist():
            wav = f"{wav_dir}/{r['id']}.wav"
            if not os.path.exists(wav):
                open(wav, "wb").write(r["audio"]["bytes"])
            out.append({"id": r["id"], "text": r["utt"], "label": r["intent_str"], "audio": wav})
    return out


def shots(train, k, rng, exclude_label="oos"):
    by = collections.defaultdict(list)
    for i, r in enumerate(train):
        if r["label"] != exclude_label:
            by[r["label"]].append(i)
    if k == "full":
        return sorted(i for v in by.values() for i in v)
    return sorted(i for v in by.values() for i in rng.sample(v, min(k, len(v))))


def prepare(data):
    work = f"{data}/work"
    os.makedirs(f"{work}/specs", exist_ok=True)
    clinc, massive, speech = load_clinc(data), load_massive(data), load_speech(data)
    sets = {
        "clinc": {"train": clinc["train"], "calib": clinc["validation"], "test": clinc["test"]},
        # MASSIVE n'a pas de hors-perimetre : calibration (cascade) sur un tirage du train hors exemples.
        "massive": {"train": massive["train"], "test": massive["test"], "dev": massive["dev"]},
        "speech": {"test": speech},
    }
    rng = random.Random(SEED)
    pools = {}
    for name in ["clinc", "massive"]:
        train = sets[name]["train"]
        pools[name] = {str(k): shots(train, k, rng) for k in SHOTS}
        pools[name]["oos"] = [i for i, r in enumerate(train) if r["label"] == "oos"]
    # Calibration MASSIVE : 1000 du train hors des pools 5/10.
    used = set(pools["massive"]["5"]) | set(pools["massive"]["10"])
    rest = [i for i in range(len(massive["train"])) if i not in used]
    pools["massive"]["calib"] = sorted(rng.sample(rest, 1000))
    # Echantillons LLM (stratifies hors perimetre / dans le perimetre pour CLINC).
    test = sets["clinc"]["test"]
    oos = [i for i, r in enumerate(test) if r["label"] == "oos"]
    ins = [i for i, r in enumerate(test) if r["label"] != "oos"]
    n_oos = round(LLM_SAMPLES["clinc"] * len(oos) / len(test))
    samples = {
        "clinc": sorted(rng.sample(ins, LLM_SAMPLES["clinc"] - n_oos) + rng.sample(oos, n_oos)),
        "massive": sorted(rng.sample(range(len(massive["test"])), LLM_SAMPLES["massive"])),
        "speech": sorted(rng.sample(range(len(speech)), LLM_SAMPLES["speech"])),
    }
    json.dump({"sets": sets, "pools": pools, "samples": samples}, open(f"{work}/bench.json", "w"), ensure_ascii=False)

    # Specs d'embedding : tout le texte necessaire, et l'audio avec et sans prefixe de tache.
    entries = {}
    for name in ["clinc", "massive"]:
        for split, rows in sets[name].items():
            for i, r in enumerate(rows):
                entries[f"{name}|{split}|{i}"] = {"task": "classification", "parts": [{"text": r["text"]}]}
    for i, r in enumerate(speech):
        entries[f"speech|text|{i}"] = {"task": "classification", "parts": [{"text": r["text"]}]}
        entries[f"speech|audio|{i}"] = {"task": "classification", "parts": [{"audio": r["audio"]}]}
        entries[f"speech|audio+prefix|{i}"] = {
            "task": "none", "parts": [{"text": "task: classification | query: "}, {"audio": r["audio"]}]}
    keys = list(entries)
    for c in range(0, len(keys), CHUNK):
        json.dump({k: entries[k] for k in keys[c:c + CHUNK]},
                  open(f"{work}/specs/emb-{c // CHUNK:02d}.json", "w"), ensure_ascii=False)
    print(f"{len(entries)} embeddings en {(len(keys) + CHUNK - 1) // CHUNK} lots ; "
          f"clinc test {len(test)} ({len(oos)} hors perimetre), massive test {len(massive['test'])}, speech {len(speech)}")


# --- Vecteurs ----------------------------------------------------------------------------------

def load_vectors(work, tag):
    vectors, timings = {}, {}
    for path in sorted(glob.glob(f"{work}/vectors-{tag}/emb-*.json")):
        v = json.load(open(path))
        timings.update({k: t for k, t in v.pop("_timings_ms").items() if not k.startswith("_")})
        v.pop("_peak_mb", None)
        vectors.update(v)
    return vectors, timings


def matrix(vectors, keys):
    m = np.asarray([vectors[k] for k in keys], dtype=np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


# --- Classifieurs sur embeddings -------------------------------------------------------------

class Router:
    """Scores par classe ; `oos` n'est jamais une classe de score, il vient du seuil."""

    def __init__(self, method, x, y, x_neg=None):
        self.method = method
        self.labels = sorted(set(y))
        index = {l: i for i, l in enumerate(self.labels)}
        yi = np.asarray([index[l] for l in y])
        if method == "knn":
            self.x, self.yi = x, yi
        elif method == "centroid":
            c = np.stack([x[yi == i].mean(0) for i in range(len(self.labels))])
            self.c = c / np.linalg.norm(c, axis=1, keepdims=True)
        elif method == "logreg":
            from sklearn.linear_model import LogisticRegression
            xs, ys = x, yi
            if x_neg is not None:  # classe de rejet explicite, retiree des scores
                xs = np.concatenate([x, x_neg]); ys = np.concatenate([yi, np.full(len(x_neg), len(self.labels))])
            self.clf = LogisticRegression(C=10.0, max_iter=2000).fit(xs, ys)
            self.has_neg = x_neg is not None

    def scores(self, q):
        if self.method == "knn":
            sims = q @ self.x.T
            out = np.full((len(q), len(self.labels)), -1.0, dtype=np.float32)
            for i in range(len(self.labels)):
                out[:, i] = sims[:, self.yi == i].max(1)
            return out
        if self.method == "centroid":
            return q @ self.c.T
        p = self.clf.predict_proba(q)
        return p  # derniere colonne = rejet si has_neg

    def predict(self, q, threshold=None):
        s = self.scores(q)
        reject = None
        if self.method == "logreg" and self.has_neg:
            reject = s[:, -1]; s = s[:, :-1]
        order = np.argsort(-s, axis=1)
        top, second = s[np.arange(len(s)), order[:, 0]], s[np.arange(len(s)), order[:, 1]]
        pred = np.asarray([self.labels[i] for i in order[:, 0]], dtype=object)
        if reject is not None:
            pred[reject > top] = "oos"
        if threshold is not None:
            pred[top < threshold] = "oos"
        return pred, top, top - second, [[self.labels[j] for j in row[:TOP_K]] for row in order]


def metrics(gold, pred):
    gold = np.asarray(gold, dtype=object); pred = np.asarray(pred, dtype=object)
    ins = gold != "oos"
    out = {"n": int(len(gold)), "accuracy": float((gold == pred).mean())}
    out["in_scope_accuracy"] = float((gold[ins] == pred[ins]).mean())
    if (~ins).any():
        out["oos_recall"] = float((pred[~ins] == "oos").mean())
        out["oos_precision"] = float(((pred == "oos") & ~ins).sum() / max(1, (pred == "oos").sum()))
    return out


def best_threshold(router, q, gold):
    """Seuil sur le score de tete qui maximise l'exactitude globale (hors perimetre compris)."""
    pred, top, _, _ = router.predict(q)
    gold = np.asarray(gold, dtype=object)
    best = (-1, None)
    for t in np.linspace(0.0, 1.0, 201):
        acc = float((gold == np.where(top < t, "oos", pred)).mean())
        if acc > best[0]:
            best = (acc, float(t))
    return best[1]


# --- Entrees LLM -------------------------------------------------------------------------------

INSTRUCTION = {
    "clinc": ("Classify the user request into exactly one of the intents below. Answer with the intent name "
              "only, exactly as written, or `oos` if none of them applies.\n\nIntents:\n"),
    "massive": ("Classe la requete de l'utilisateur dans exactement une des intentions ci-dessous. Reponds "
                "uniquement par le nom de l'intention, exactement comme il est ecrit.\n\nIntentions :\n"),
}


EXAMPLES = {"clinc": "\n\nExamples:\n", "massive": "\n\nExemples :\n"}


def instruction(name, labels, examples=None):
    allowed = labels + (["oos"] if name == "clinc" else [])
    text = INSTRUCTION[name] + "\n".join(allowed)
    if examples:  # [(texte, etiquette)], pris dans le pool 5 exemples
        text += EXAMPLES[name] + "\n".join(f"{t} -> {l}" for t, l in examples)
    return text


def llm_inputs(data):
    work = f"{data}/work"
    bench = json.load(open(f"{work}/bench.json"))
    vectors, _ = load_vectors(work, "8bit")
    os.makedirs(f"{work}/llm", exist_ok=True)
    for name in ["clinc", "massive", "speech"]:
        base = "massive" if name == "speech" else name
        train = bench["sets"][base]["train"]
        labels = sorted({r["label"] for r in train if r["label"] != "oos"})
        test = bench["sets"][name]["test"]
        sample = bench["samples"][name]
        # Candidats de la cascade : regression logistique, exemples complets du train.
        idx = bench["pools"][base]["full"]
        router = Router("logreg", matrix(vectors, [f"{base}|train|{i}" for i in idx]), [train[i]["label"] for i in idx])
        qkind = "audio+prefix" if name == "speech" else "test"
        q = matrix(vectors, [f"{name}|{qkind}|{i}" for i in sample])
        _, _, _, top = router.predict(q)
        full, cascade = [], []
        for i, cands in zip(sample, top):
            item = {"id": str(i)}
            if name == "speech":
                item["audio"] = test[i]["audio"]
            else:
                item["text"] = test[i]["text"]
            full.append(item)
            cascade.append({**item, "instruction": instruction(base, sorted(cands)), "candidates": cands})
        json.dump({"instruction": instruction(base, labels), "items": full},
                  open(f"{work}/llm/{name}-full.json", "w"), ensure_ascii=False)
        json.dump({"instruction": instruction(base, labels), "items": cascade},
                  open(f"{work}/llm/{name}-top{TOP_K}.json", "w"), ensure_ascii=False)
        # Few-shot : 2 exemples par intention (les 2 premiers du pool 5) dans le prefixe commun.
        seen = collections.Counter()
        examples = []
        for i in bench["pools"][base]["5"]:
            if seen[train[i]["label"]] < 2:
                seen[train[i]["label"]] += 1
                examples.append((train[i]["text"], train[i]["label"]))
        if base == "clinc":
            examples += [(train[i]["text"], "oos") for i in bench["pools"]["clinc"]["oos"][:10]]
        json.dump({"instruction": instruction(base, labels, examples), "items": full},
                  open(f"{work}/llm/{name}-shots2.json", "w"), ensure_ascii=False)
        print(name, len(full), "items")


# Variantes de `oos` produites par E4B (« oops » surtout) : lues comme un rejet, comptees a part.
OOS_ALIASES = {"oops", "ooc", "oo", "none", "out_of_scope"}


def parse(output, allowed):
    text = output.strip().strip("`*\"' .").lower().replace(" ", "_")
    if text in allowed:
        return text
    if "oos" in allowed and text in OOS_ALIASES:
        return "oos"
    hits = [l for l in allowed if re.search(rf"(?<![a-z_]){re.escape(l)}(?![a-z_])", output.lower())]
    return max(hits, key=len) if hits else "invalid"


# --- Rapport ---------------------------------------------------------------------------------

def report(data, out_path):
    work = f"{data}/work"
    bench = json.load(open(f"{work}/bench.json"))
    lines = []

    def emit(**fields):
        fields["date"] = time.strftime("%Y-%m-%d")
        lines.append(fields)
        print(json.dumps(fields, ensure_ascii=False))

    for tag in sorted(os.path.basename(p)[len("vectors-"):] for p in glob.glob(f"{work}/vectors-*")):
        vectors, timings = load_vectors(work, tag)
        for name, base, qkinds in [("clinc", "clinc", ["test"]), ("massive", "massive", ["test"]),
                                   ("speech", "massive", ["text", "audio", "audio+prefix"])]:
            train = bench["sets"][base]["train"]
            test = bench["sets"][name]["test"]
            gold = [r["label"] for r in test]
            calib_rows = (bench["sets"]["clinc"]["calib"] if base == "clinc"
                          else [train[i] for i in bench["pools"]["massive"]["calib"]])
            calib_keys = ([f"clinc|calib|{i}" for i in range(len(calib_rows))] if base == "clinc"
                          else [f"massive|train|{i}" for i in bench["pools"]["massive"]["calib"]])
            calib_q = matrix(vectors, calib_keys)
            calib_gold = [r["label"] for r in calib_rows]
            neg = matrix(vectors, [f"clinc|train|{i}" for i in bench["pools"]["clinc"]["oos"]]) if base == "clinc" else None
            for k in SHOTS:
                idx = bench["pools"][base][str(k)]
                x = matrix(vectors, [f"{base}|train|{i}" for i in idx])
                y = [train[i]["label"] for i in idx]
                variants = [("knn", None), ("centroid", None), ("logreg", None)]
                if base == "clinc":
                    variants.append(("logreg+neg", neg))
                for method, x_neg in variants:
                    router = Router(method.split("+")[0], x, y, x_neg)
                    threshold = best_threshold(router, calib_q, calib_gold) if base == "clinc" else None
                    for qkind in qkinds:
                        keys = [f"{name}|{qkind}|{i}" for i in range(len(test))]
                        q = matrix(vectors, keys)
                        t0 = time.perf_counter()
                        pred, *_ = router.predict(q, threshold)
                        classify_us = (time.perf_counter() - t0) / len(q) * 1e6
                        emb_ms = np.asarray([timings[k_] for k_ in keys])
                        emit(dataset=name, embedder=tag, query=qkind, shots=k, method=method,
                             threshold=threshold, **metrics(gold, pred),
                             embed_ms_median=float(np.median(emb_ms)), embed_ms_p90=float(np.percentile(emb_ms, 90)),
                             classify_us=round(classify_us, 1))

        # Voix avec exemples vocaux : K clips par intention tires du dev Speech-MASSIVE, evaluation
        # sur les clips restants (memes requetes en texte pour comparaison), exemples texte en plus.
        speech = bench["sets"]["speech"]["test"]
        train = bench["sets"]["massive"]["train"]
        rng = random.Random(SEED)
        by = collections.defaultdict(list)
        for i, r in enumerate(speech):
            by[r["label"]].append(i)
        for k in [2, 5]:
            ex = sorted(i for v in by.values() for i in rng.sample(v, min(k, len(v) - 1)) if len(v) > 1)
            rest = [i for i in range(len(speech)) if i not in set(ex)]
            gold = [speech[i]["label"] for i in rest]
            text_idx = bench["pools"]["massive"]["10"]
            x_text = matrix(vectors, [f"massive|train|{i}" for i in text_idx])
            y_text = [train[i]["label"] for i in text_idx]
            x_audio = matrix(vectors, [f"speech|audio+prefix|{i}" for i in ex])
            y_audio = [speech[i]["label"] for i in ex]
            for exemplars, x, y in [("text10", x_text, y_text), (f"audio{k}", x_audio, y_audio),
                                    (f"text10+audio{k}", np.concatenate([x_text, x_audio]), y_text + y_audio)]:
                for method in ["knn", "logreg"]:
                    router = Router(method, x, y)
                    for qkind in ["text", "audio+prefix"]:
                        pred, *_ = router.predict(matrix(vectors, [f"speech|{qkind}|{i}" for i in rest]))
                        emit(dataset="speech-split", embedder=tag, query=qkind, exemplars=exemplars,
                             audio_shots=k, method=method, **metrics(gold, pred))

    # LLM : sortie brute -> etiquette, puis cascade sur les marges de la tete logistique.
    vectors, timings = load_vectors(work, "8bit")
    for path in sorted(glob.glob(f"{work}/llm/out-*.jsonl")):
        model, name, variant = re.match(r"out-(.+?)-(clinc|massive|speech)-(full|top\d+|shots\d+)\.jsonl", os.path.basename(path)).groups()
        base = "massive" if name == "speech" else name
        rows = [json.loads(l) for l in open(path)]
        header, rows = rows[0], rows[1:]
        test = bench["sets"][name]["test"]
        train = bench["sets"][base]["train"]
        labels = sorted({r["label"] for r in train if r["label"] != "oos"})
        allowed = labels + (["oos"] if base == "clinc" else [])
        spec = json.load(open(f"{work}/llm/{name}-{variant}.json"))
        cands = {it["id"]: it.get("candidates") for it in spec["items"]}
        gold, pred, ms = [], [], []
        for r in rows:
            ok = cands[r["id"]] + (["oos"] if base == "clinc" else []) if cands[r["id"]] else allowed
            gold.append(test[int(r["id"])]["label"]); pred.append(parse(r["output"], ok)); ms.append(r["ms"])
        invalid = float(np.mean([p == "invalid" for p in pred]))
        aliased = float(np.mean([r["output"].strip().strip("`*\"' .").lower() in OOS_ALIASES for r in rows]))
        emit(dataset=name, llm=model, variant=variant, **metrics(gold, pred), invalid_rate=invalid, oos_alias_rate=aliased,
             ms_median=float(np.median(ms)), ms_p90=float(np.percentile(ms, 90)),
             prefix_tokens=header["prefix_tokens"], prefix_ms=header["prefix_ms"], load_s=header["load_s"])
        if variant.startswith("top"):
            # Cascade : l'embedding decide si la marge est >= m, sinon la reponse du LLM top-k.
            # Base = meilleur routeur embedding : pour CLINC, classe de rejet apprise + seuil calibre.
            idx = bench["pools"][base]["full"]
            neg = None
            if base == "clinc":
                neg = matrix(vectors, [f"clinc|train|{i}" for i in bench["pools"]["clinc"]["oos"]])
            router = Router("logreg", matrix(vectors, [f"{base}|train|{i}" for i in idx]),
                            [train[i]["label"] for i in idx], neg)
            threshold = None
            if base == "clinc":
                calib = bench["sets"]["clinc"]["calib"]
                threshold = best_threshold(router, matrix(vectors, [f"clinc|calib|{i}" for i in range(len(calib))]),
                                           [r["label"] for r in calib])
            qkind = "audio+prefix" if name == "speech" else "test"
            sample = [int(r["id"]) for r in rows]
            q = matrix(vectors, [f"{name}|{qkind}|{i}" for i in sample])
            emb_pred, top, margin, _ = router.predict(q, threshold)
            emb_ms = np.asarray([timings[f"{name}|{qkind}|{i}"] for i in sample])
            for m in [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]:
                use_llm = margin < m
                final = np.where(use_llm, np.asarray(pred, dtype=object), emb_pred)
                latency = emb_ms + np.where(use_llm, np.asarray(ms), 0.0)
                emit(dataset=name, cascade=model, margin=m, llm_fraction=float(use_llm.mean()),
                     **metrics(gold, final), ms_median=float(np.median(latency)), ms_mean=float(latency.mean()),
                     ms_p90=float(np.percentile(latency, 90)))
    # Voix par transcription : le LLM transcrit (eval-route, consigne de transcription), l'embedder
    # 8 bits route le texte avec les exemples texte complets. Latence = transcription + embedding.
    asr_path = f"{work}/asr-vectors.json"
    if os.path.exists(asr_path):
        asr = json.load(open(asr_path))
        asr_ms = asr.pop("_timings_ms"); asr.pop("_peak_mb", None)
        speech = bench["sets"]["speech"]["test"]
        train = bench["sets"]["massive"]["train"]
        for k in SHOTS:
            idx = bench["pools"]["massive"][str(k)]
            router = Router("logreg", matrix(vectors, [f"massive|train|{i}" for i in idx]), [train[i]["label"] for i in idx])
            for model in ["e2b", "e4b"]:
                rows = [json.loads(l) for l in open(f"{work}/llm/asr-{model}.jsonl")][1:]
                ids = [r["id"] for r in rows]
                pred, *_ = router.predict(matrix(asr, [f"speech|asr-{model}|{i}" for i in ids]))
                norm = lambda t: re.sub(r"[^\w' ]", "", t.lower()).split()
                exact = float(np.mean([norm(r["output"]) == norm(speech[int(r["id"])]["text"]) for r in rows]))
                latency = np.asarray([r["ms"] for r in rows]) + np.asarray([asr_ms[f"speech|asr-{model}|{i}"] for i in ids])
                emit(dataset="speech", asr=model, shots=k, method="logreg",
                     **metrics([speech[int(i)]["label"] for i in ids], pred), transcript_exact=exact,
                     asr_ms_median=float(np.median([r["ms"] for r in rows])),
                     ms_median=float(np.median(latency)), ms_p90=float(np.percentile(latency, 90)))

    with open(out_path, "a") as f:
        for l in lines:
            f.write(json.dumps(l, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    command, data = sys.argv[1], sys.argv[2]
    if command == "prepare":
        prepare(data)
    elif command == "llm-inputs":
        llm_inputs(data)
    elif command == "report":
        report(data, sys.argv[3])
