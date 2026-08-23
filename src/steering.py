"""Shared steering primitives. Hook math is copied verbatim from the original
experiment code; only names were changed."""

import random

import torch
import torch.nn.functional as F
from tqdm import tqdm


def build_chat_prompt(tokenizer, sentence, instruction):
    prompt = f'''
{instruction}

Sentence:
"{sentence}"
'''
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def strip_chat_template(text):
    if "[/INST]" in text:
        return text.split("[/INST]", 1)[1].strip()
    if "assistant" in text:
        return text.split("assistant", 1)[1].strip()
    if "model" in text:
        return text.split("model", 1)[1].strip()
    return text.strip()


def capture_last_token_resid_post(model, prompt):
    toks = model.to_tokens(prompt)
    with torch.no_grad():
        _, cache = model.run_with_cache(
            toks, names_filter=lambda n: n.endswith("hook_resid_post"))
    acts = [cache[f"blocks.{l}.hook_resid_post"][:, -1, :].detach().cpu().float()
            for l in range(model.cfg.n_layers)]
    return torch.stack(acts, dim=0).squeeze(1)  # (n_layers, d_model)


def select_instruction(style_instructions, instruction_mode):
    if instruction_mode == "fixed":
        return style_instructions[0]
    return random.choice(style_instructions)


def learn_style_direction(model, tokenizer, train_texts,
                          plain_instruction, style_instructions, instruction_mode):
    """Per-layer unit direction = mean(resid_style - resid_plain) over train, normalized.
    target_projection = mean projection of styled activations onto that direction."""
    delta_accum = torch.zeros(model.cfg.n_layers, model.cfg.d_model)
    for sentence in tqdm(train_texts, desc=f"direction ({instruction_mode})"):
        instruction = select_instruction(style_instructions, instruction_mode)
        resid_plain = capture_last_token_resid_post(
            model, build_chat_prompt(tokenizer, sentence, plain_instruction))
        resid_style = capture_last_token_resid_post(
            model, build_chat_prompt(tokenizer, sentence, instruction))
        delta_accum += resid_style - resid_plain

    steering_directions = F.normalize(delta_accum / len(train_texts), dim=1)

    target_projection = torch.zeros(model.cfg.n_layers)
    for sentence in tqdm(train_texts, desc=f"target ({instruction_mode})"):
        instruction = select_instruction(style_instructions, instruction_mode)
        resid_style = capture_last_token_resid_post(
            model, build_chat_prompt(tokenizer, sentence, instruction))
        target_projection += (resid_style * steering_directions).sum(dim=1)
    target_projection /= len(train_texts)

    return steering_directions, target_projection


def make_fixed_projection_hook(layer_idx, direction, adjustment_scalar):
    """Baseline: add a constant adjustment_scalar * direction to every token position."""
    hook_name = f"blocks.{layer_idx}.hook_resid_post"

    def hook(resid, hook):
        h = resid
        u = direction.to(h.device, h.dtype)
        c = adjustment_scalar
        resid = resid.clone()
        resid = h + c * u
        return resid

    return (hook_name, hook)


def make_saps_hook(layer_idx, direction, target, alpha):
    """SAPS: each step, re-project the last-token activation toward target, scaled by alpha."""
    direction = direction / (direction.norm() + 1e-8)
    hook_name = f"blocks.{layer_idx}.hook_resid_post"

    def hook(resid, hook):
        h = resid[:, -1, :]
        u = direction.to(h.device, h.dtype)
        proj = (h * u).sum(dim=-1, keepdim=True)
        resid = resid.clone()
        resid = resid + alpha * (target - proj) * u
        return resid

    return (hook_name, hook)
