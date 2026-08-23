# SAPS: Scaled Autoregressive Projection Steering

Code and data for our EMNLP 2026 Findings paper. We steer instruction-tuned LLMs
toward a target style — humblebrag or sarcasm — by editing the residual stream at
inference time, and compare two ways of doing it:

- **SAPS** (ours): at each decoding step, re-project the last-token activation onto
  a learned style direction and pull it toward a target value, scaled by `alpha`.
- **Fixed Projection** (baseline): add a constant projection adjustment to every
  token position.

Both use the same learned direction; only the injection differs.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Weights are pulled from the Hugging Face Hub by TransformerLens; a CUDA GPU is
needed. Set `HF_HOME` / `HF_TOKEN` as usual.

## Usage

Two steps. Learn the direction once per (model, task), then generate.

```bash
# 1. learn + save the direction
python src/compute_direction.py \
    --model meta-llama/Llama-3.1-8B-Instruct --task humblebrag

# 2. generate (SAPS, layers 10-21, alpha 1.4, over the test set)
python src/generate.py \
    --model meta-llama/Llama-3.1-8B-Instruct --task humblebrag \
    --direction directions/meta-llama_Llama-3.1-8B-Instruct_humblebrag_random_seed42.pt \
    --method saps --alpha 1.4 --layer-start 10 --layer-end 21 \
    --input data/humblebrag/test.json
```

`--input` takes a sentence or a file (`.json` list, or `.txt` one per line).
Use `--method fixed_projection --alpha 1.0` for the baseline. Output is JSONL,
one row per (sentence, layer).

## Layout

```
prompts/     plain + style instructions per task
data/        train/val/test splits (humblebrag, sarcasm)
src/         steering.py, compute_direction.py, generate.py
directions/  saved directions (.pt)
outputs/     generations (.jsonl)
results/     averaged evaluations + table scripts (see results/README.md)
```

Directions are learned on `train`, numbers reported on `test`. `--seed` (default
42) fixes the random instruction sampling. `alpha` is the only steering
hyperparameter; the baseline in the paper uses `alpha = 1.0`.

MIT licensed.
