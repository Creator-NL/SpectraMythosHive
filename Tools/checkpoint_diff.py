"""
checkpoint_diff.py — Compare two PyTorch checkpoints (.pt) to see which layers
moved the most between them.

Usage:
    python Tools/checkpoint_diff.py before.pt after.pt
    python Tools/checkpoint_diff.py before.pt after.pt --top 40

Requires: torch
"""
import argparse
import os
import sys

import torch

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def load_state_dict(path, allow_full_checkpoint=False):
    strict_weights_only = (not allow_full_checkpoint) and (
        not os.path.basename(path).startswith("global_model")
    )
    obj = torch.load(path, map_location="cpu", weights_only=strict_weights_only)
    if isinstance(obj, dict):
        for key in ("model_state", "state_dict", "model_state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    return obj


def diff_checkpoints(before_path, after_path, top_n=20, allow_full_checkpoint=False):
    before = load_state_dict(before_path, allow_full_checkpoint=allow_full_checkpoint)
    after = load_state_dict(after_path, allow_full_checkpoint=allow_full_checkpoint)

    common_keys = set(before.keys()) & set(after.keys())
    only_before = set(before.keys()) - set(after.keys())
    only_after = set(after.keys()) - set(before.keys())

    if only_before:
        print(f"note: {len(only_before)} tensors only in BEFORE (e.g. {list(only_before)[:3]})")
    if only_after:
        print(f"note: {len(only_after)} tensors only in AFTER (e.g. {list(only_after)[:3]})")

    rows = []
    for key in common_keys:
        if not isinstance(before[key], torch.Tensor) or not isinstance(after[key], torch.Tensor):
            continue
        b = before[key].float()
        a = after[key].float()
        if b.shape != a.shape:
            print(f"note: shape mismatch on {key}: {tuple(b.shape)} -> {tuple(a.shape)}")
            continue
        delta_norm = (a - b).norm().item()
        rel_change = delta_norm / (b.norm().item() + 1e-12)
        rows.append((key, delta_norm, rel_change, tuple(b.shape)))

    rows.sort(key=lambda r: r[1], reverse=True)

    print(f"\n{'layer':50s} {'delta norm':>12s} {'delta %':>10s}  shape")
    print("-" * 95)
    for key, delta_norm, rel_change, shape in rows[:top_n]:
        print(f"{key[:50]:50s} {delta_norm:12.5f} {rel_change:9.3%}  {shape}")

    total = sum(r[1] for r in rows)
    print(f"\nTotal movement across {len(rows)} tensors: {total:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diff two PyTorch checkpoints layer by layer.")
    parser.add_argument("before", help="Path to the earlier checkpoint (.pt)")
    parser.add_argument("after", help="Path to the later checkpoint (.pt)")
    parser.add_argument("--top", type=int, default=20, help="How many layers to show (default 20)")
    parser.add_argument("--allow-full-checkpoint", action="store_true", help="Allow loading trusted local optimizer-included checkpoints")
    args = parser.parse_args()
    diff_checkpoints(args.before, args.after, args.top, allow_full_checkpoint=args.allow_full_checkpoint)
