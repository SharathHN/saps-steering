"""Learn and save the steering direction for a task. Used by both methods.

    python src/compute_direction.py --model meta-llama/Llama-3.1-8B-Instruct --task humblebrag
"""

import argparse
import json
import os
import random

import torch
from transformer_lens import HookedTransformer

from steering import learn_style_direction

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--task", required=True, choices=["humblebrag", "sarcasm"])
    p.add_argument("--train-file", default=None)
    p.add_argument("--prompts-file", default=None)
    p.add_argument("--instruction-mode", default="random", choices=["fixed", "random"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", default="float16", choices=list(DTYPES))
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    train_file = args.train_file or os.path.join(ROOT, "data", args.task, "train.json")
    prompts_file = args.prompts_file or os.path.join(ROOT, "prompts", f"{args.task}.json")

    prompts = json.load(open(prompts_file))
    train_texts = json.load(open(train_file))

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = HookedTransformer.from_pretrained(args.model, device=args.device,
                                              torch_dtype=DTYPES[args.dtype])
    model.eval()

    steering_directions, target_projection = learn_style_direction(
        model, model.tokenizer, train_texts,
        prompts["plain"], prompts["style"], args.instruction_mode)

    model_slug = args.model.replace("/", "_")
    out_path = args.out or os.path.join(
        ROOT, "directions",
        f"{model_slug}_{args.task}_{args.instruction_mode}_seed{args.seed}.pt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    torch.save({
        "steering_directions": steering_directions,   # (n_layers, d_model)
        "target_projection": target_projection,        # (n_layers,)
        "model": args.model, "task": args.task,
        "instruction_mode": args.instruction_mode, "seed": args.seed,
        "n_train": len(train_texts),
        "n_layers": model.cfg.n_layers, "d_model": model.cfg.d_model,
    }, out_path)
    print("saved", out_path)


if __name__ == "__main__":
    main()
