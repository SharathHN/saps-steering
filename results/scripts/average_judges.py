"""Average the per-judge eval files into *_avg_eval.jsonl (how eval_avg/ was made).

Per-judge inputs (large, with raw judge text) are not shipped; this documents and
reproduces the averaging if you have them.

    python average_judges.py --root <dir> --suffixes _oss_eval _qwen_eval _gemma_eval
"""

import argparse
import json
from pathlib import Path


def load_eval_file(path):
    records = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        ev = r.get("evaluation")
        if ev:
            records[(r["sentence_id"], r.get("layer", 0), r.get("scale", 0))] = (r, ev)
    return records


def average_evals(eval_list):
    result = {}
    for m in eval_list[0].keys():
        vals = []
        for e in eval_list:
            v = e.get(m)
            if v is None:
                continue
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                continue
        result[m] = sum(vals) / len(vals) if vals else None
    return result


def process_dir(root, suffixes):
    skip = tuple(suffixes) + ("_avg_eval",)
    for gen_file in sorted(p for p in root.rglob("*.jsonl") if not p.stem.endswith(skip)):
        stem = gen_file.stem
        loaded = [load_eval_file(gen_file.with_name(f"{stem}{suf}.jsonl"))
                  for suf in suffixes if gen_file.with_name(f"{stem}{suf}.jsonl").exists()]
        if not loaded:
            continue
        all_keys = set().union(*[set(d) for d in loaded])
        out_path = gen_file.with_name(f"{stem}_avg_eval.jsonl")
        with open(out_path, "w") as fout:
            for key in sorted(all_keys):
                present = [d[key][1] for d in loaded if key in d]
                if not present:
                    continue
                base = next(d[key][0] for d in loaded if key in d)
                rec = {k: v for k, v in base.items()
                       if k not in ("evaluation", "evaluation_raw_output")}
                rec["num_evaluators"] = len(present)
                rec["evaluation"] = average_evals(present)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print("wrote", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--suffixes", nargs="+", default=["_oss_eval", "_qwen_eval", "_gemma_eval"])
    args = ap.parse_args()
    process_dir(Path(args.root), args.suffixes)


if __name__ == "__main__":
    main()
