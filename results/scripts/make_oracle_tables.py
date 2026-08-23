"""Oracle-over-layers test tables from eval_avg/ (per sentence pick its best layer,
then average). Main-table cells are fixed_projection@1.0 and saps@best-alpha.

    python make_oracle_tables.py [--csv out.csv]
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

EVAL_AVG = Path(__file__).resolve().parent.parent / "eval_avg"
PRIMARY = {"humblebrag": "humblebrag", "sarcasm": "sarcasm"}
METRICS = {"humblebrag": ["humblebrag", "entailment", "grammaticality", "clarity"],
           "sarcasm":    ["sarcasm", "entailment", "grammaticality", "clarity"]}

# best alpha per (task, method, model), selected on validation; fixed_projection = 1.0
BEST_ALPHA = {
    ("humblebrag", "saps"): {"Gemma-2-9B": 1.5, "Llama-3.1-8B": 1.4, "Mistral-7B": 1.4},
    ("sarcasm", "saps"): {"Gemma-2-9B": 1.2, "Llama-3.1-8B": 1.4, "Mistral-7B": 1.4},
}
MODEL_LABEL = {
    "google_gemma-2-9b-it": "Gemma-2-9B",
    "meta-llama_Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "mistralai_Mistral-7B-Instruct-v0.1": "Mistral-7B",
}


def scale_from_name(path):
    return float(path.stem.split("scale")[-1].replace("_avg_eval", "").replace("p", "."))


def oracle_over_layers(path, primary):
    by_sent = defaultdict(list)
    keys = None
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        ev = r.get("evaluation")
        if not ev:
            continue
        keys = keys or list(ev.keys())
        by_sent[r["sentence_id"]].append((r.get("layer", 0), ev))
    totals = defaultdict(list)
    for recs in by_sent.values():
        _, best = max(recs, key=lambda le: float(le[1].get(primary, -1) or -1))
        for m in keys:
            if best.get(m) is not None:
                totals[m].append(float(best[m]))
    means = {m: (sum(v) / len(v) if v else None) for m, v in totals.items()}
    n_layers = len({L for recs in by_sent.values() for L, _ in recs})
    return means, len(by_sent), n_layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    rows = []
    for task in ["humblebrag", "sarcasm"]:
        primary = PRIMARY[task]
        for method in ["fixed_projection", "saps"]:
            for model_dir in sorted((EVAL_AVG / task / method).glob("*")):
                model = MODEL_LABEL.get(model_dir.name, model_dir.name)
                for f in sorted(model_dir.glob("steer_random_scale*_avg_eval.jsonl")):
                    alpha = scale_from_name(f)
                    means, n_sent, n_layers = oracle_over_layers(f, primary)
                    rows.append({"task": task, "method": method, "model": model, "alpha": alpha,
                                 "n_sentences": n_sent, "n_layers": n_layers,
                                 **{f"oracle_{m}": means.get(m) for m in METRICS[task]}})

    for task in ["humblebrag", "sarcasm"]:
        p = PRIMARY[task]
        print(f"\n{task.upper()}  (oracle over layers, test)")
        print(f"{'method':17s} {'model':13s} {'a':>4} {'N':>4} {'L':>3}  {p:>7} {'ent':>6} {'gram':>6} {'clar':>6}")
        for r in rows:
            if r["task"] == task:
                print(f"{r['method']:17s} {r['model']:13s} {r['alpha']:>4.1f} {r['n_sentences']:>4} "
                      f"{r['n_layers']:>3}  {r[f'oracle_{p}']:>7.3f} {r['oracle_entailment']:>6.3f} "
                      f"{r['oracle_grammaticality']:>6.3f} {r['oracle_clarity']:>6.3f}")

    if args.csv:
        cols = ["task", "method", "model", "alpha", "n_sentences", "n_layers",
                "oracle_humblebrag", "oracle_sarcasm", "oracle_entailment",
                "oracle_grammaticality", "oracle_clarity"]
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print("\nwrote", args.csv)


if __name__ == "__main__":
    main()
