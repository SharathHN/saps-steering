"""Steered generation with SAPS or Fixed Projection.

    python src/generate.py --model meta-llama/Llama-3.1-8B-Instruct --task humblebrag \
        --direction directions/<...>.pt --method saps --alpha 1.4 \
        --layer-start 10 --layer-end 21 --input data/humblebrag/test.json

--input is a sentence or a file (.json list, or .txt one per line).
"""

import argparse
import json
import os

import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

from steering import (build_chat_prompt, strip_chat_template,
                      capture_last_token_resid_post,
                      make_fixed_projection_hook, make_saps_hook)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--task", required=True, choices=["humblebrag", "sarcasm"])
    p.add_argument("--direction", required=True)
    p.add_argument("--method", required=True, choices=["saps", "fixed_projection"])
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--layer-start", type=int, required=True)
    p.add_argument("--layer-end", type=int, required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--prompts-file", default=None)
    p.add_argument("--max-new-tokens", type=int, default=60)
    p.add_argument("--dtype", default="float16", choices=list(DTYPES))
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=None)
    return p.parse_args()


def load_sentences(spec):
    if os.path.isfile(spec):
        if spec.endswith(".json"):
            return json.load(open(spec))
        return [ln.strip() for ln in open(spec) if ln.strip()]
    return [spec]


def main():
    args = parse_args()
    prompts_file = args.prompts_file or os.path.join(ROOT, "prompts", f"{args.task}.json")
    plain_instruction = json.load(open(prompts_file))["plain"]

    bundle = torch.load(args.direction, map_location="cpu")
    steering_directions = bundle["steering_directions"]
    target_projection = bundle["target_projection"]

    sentences = load_sentences(args.input)

    model = HookedTransformer.from_pretrained(args.model, device=args.device,
                                              torch_dtype=DTYPES[args.dtype])
    model.eval()
    directions = steering_directions.to(args.device)
    targets = target_projection.to(args.device)

    model_slug = args.model.replace("/", "_")
    alpha_slug = str(args.alpha).replace(".", "p")
    out_path = args.out or os.path.join(
        ROOT, "outputs",
        f"{model_slug}_{args.task}_{args.method}_alpha{alpha_slug}_L{args.layer_start}-{args.layer_end}.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w") as fout:
        for sent_id, sentence in enumerate(tqdm(sentences)):
            chat_plain = build_chat_prompt(model.tokenizer, sentence, plain_instruction)
            if args.method == "fixed_projection":
                resid_plain = capture_last_token_resid_post(model, chat_plain)

            for layer_idx in range(args.layer_start, args.layer_end + 1):
                if args.method == "saps":
                    hook = make_saps_hook(layer_idx, directions[layer_idx],
                                          targets[layer_idx], args.alpha)
                else:
                    proj = (resid_plain[layer_idx] * steering_directions[layer_idx]).sum().item()
                    c = (target_projection[layer_idx].item() - proj) * args.alpha
                    hook = make_fixed_projection_hook(layer_idx, directions[layer_idx], c)

                with model.hooks(fwd_hooks=[hook]):
                    out = model.generate(chat_plain, max_new_tokens=args.max_new_tokens,
                                         do_sample=False, verbose=False)

                fout.write(json.dumps({
                    "sentence_id": sent_id, "sentence": sentence,
                    "method": args.method, "alpha": args.alpha, "layer": layer_idx,
                    "output": strip_chat_template(out),
                }, ensure_ascii=False) + "\n")
            fout.flush()
    print("saved", out_path)


if __name__ == "__main__":
    main()
