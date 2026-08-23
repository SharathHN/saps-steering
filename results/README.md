# Results — averaged evaluations and table scripts

For transparency the paper numbers are provided as the **averaged per-sentence
evaluation JSONL** they are computed from, plus the scripts that (a) produce that
JSONL from the individual judges and (b) turn it into the headline tables. No
pre-baked summary CSVs — regenerate them from the JSONL.

## `eval_avg/` — averaged evaluations (test set, N = 450)

```
eval_avg/<task>/<method>/<model>/steer_random_scale<alpha>_avg_eval.jsonl
  task   ∈ {humblebrag, sarcasm}
  method ∈ {saps, fixed_projection}
  model  ∈ {google_gemma-2-9b-it, meta-llama_Llama-3.1-8B-Instruct,
            mistralai_Mistral-7B-Instruct-v0.1}
```

One row per `(sentence, layer)`. Each row carries the generation fields plus:

| field | meaning |
|---|---|
| `evaluation` | mean judge scores: `{<task>, entailment, grammaticality, clarity}` (0–5) |
| `num_evaluators` | how many judges contributed to this row (2 or 3) |

`evaluation` is the **average of up to three LLM judges** (gpt-oss-120b, Qwen,
Gemma). Exactly the 12 main-table cells are shipped — one file per
(task, method, model): `fixed_projection` at α = 1.0 and `saps` at its best α.

## `scripts/`

- **`average_judges.py`** — how `evaluation` was formed: averages the per-judge
  eval files (`*_oss_eval.jsonl`, `*_qwen_eval.jsonl`, `*_gemma_eval.jsonl`) per
  `(sentence, layer, scale)` into `*_avg_eval.jsonl`. The per-judge inputs are
  large (they include each judge's raw text) and are not shipped; this documents
  and reproduces the averaging if you have them.

- **`make_oracle_tables.py`** — reads `eval_avg/` and prints the oracle-over-layers
  test tables (per sentence pick its best layer, then average). `*` marks the
  paper cells: `fixed_projection` @ α = 1.0 vs `saps` @ best α.

  ```bash
  python scripts/make_oracle_tables.py            # print
  python scripts/make_oracle_tables.py --csv oracle.csv   # also write CSV
  ```

## Notes

- **Method labels**: `saps` = our method (Scaled Autoregressive Projection
  Steering); `fixed_projection` = the baseline ("original" in the raw data).
- **Best α** (selected on the validation set) is recorded in `BEST_ALPHA` inside
  `make_oracle_tables.py`: humblebrag saps = {Gemma 1.5, Llama 1.4, Mistral 1.4};
  sarcasm saps = {Gemma 1.2, Llama 1.4, Mistral 1.4}; fixed_projection = 1.0.
- **N**: humblebrag has a negligible fraction of `(sentence, layer)` rows (≤0.07 %)
  where no judge returned a parseable score; sarcasm has none. Scores are means
  over the sentences scored at each layer.
