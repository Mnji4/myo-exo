#!/usr/bin/env python3
"""Analyze exported policy synergy datasets and propose muscle groups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def corrcoef_safe(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    std = np.std(x, axis=0)
    keep = std > 1e-8
    corr = np.eye(x.shape[1], dtype=np.float64)
    if np.count_nonzero(keep) >= 2:
        sub = np.corrcoef(x[:, keep], rowvar=False)
        corr[np.ix_(keep, keep)] = sub
    return np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)


def greedy_groups(score: np.ndarray, *, threshold: float, names: list[str]) -> list[list[int]]:
    remaining = set(range(score.shape[0]))
    groups: list[list[int]] = []
    while remaining:
        seed = max(remaining, key=lambda idx: float(np.sum(score[idx, list(remaining)])))
        group = [int(seed)]
        remaining.remove(seed)
        changed = True
        while changed:
            changed = False
            best_idx = None
            best_val = float(threshold)
            for idx in sorted(remaining):
                val = float(np.mean(score[idx, group]))
                if val > best_val:
                    best_val = val
                    best_idx = int(idx)
            if best_idx is not None:
                group.append(best_idx)
                remaining.remove(best_idx)
                changed = True
        groups.append(sorted(group))
    groups.sort(key=lambda g: (len(g), -sum(g)), reverse=True)
    return groups


def named_groups(groups: list[list[int]], names: list[str]) -> list[dict[str, object]]:
    return [{"indices": group, "names": [names[idx] for idx in group]} for group in groups]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, nargs="+", required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--matrix", choices=["activation", "length", "combined"], default="combined")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    loaded = []
    names: list[str] | None = None
    for path in args.dataset:
        data = np.load(path, allow_pickle=True)
        metadata = data["metadata"].item()
        current_names = list(metadata["actuator_names"])
        if names is None:
            names = current_names
        elif names != current_names:
            raise ValueError(f"actuator names mismatch in {path}")
        item = {
            "path": str(path),
            "metadata": metadata,
            "activation_corr": corrcoef_safe(data["activations"]),
            "length_corr": corrcoef_safe(data["actuator_length"]),
            "activation_samples": int(data["activations"].shape[0]),
        }
        loaded.append(item)
    assert names is not None

    activation_stack = np.stack([item["activation_corr"] for item in loaded], axis=0)
    length_stack = np.stack([item["length_corr"] for item in loaded], axis=0)
    activation_stable = np.min(np.abs(activation_stack), axis=0) * np.sign(np.mean(activation_stack, axis=0))
    length_stable = np.min(np.abs(length_stack), axis=0) * np.sign(np.mean(length_stack, axis=0))
    combined = 0.65 * np.abs(activation_stable) + 0.35 * np.abs(length_stable)
    np.fill_diagonal(combined, 1.0)

    if args.matrix == "activation":
        grouping_score = np.abs(activation_stable)
    elif args.matrix == "length":
        grouping_score = np.abs(length_stable)
    else:
        grouping_score = combined
    np.fill_diagonal(grouping_score, 1.0)

    groups = greedy_groups(grouping_score, threshold=float(args.threshold), names=names)
    summary = {
        "datasets": [
            {
                "path": item["path"],
                "label": item["metadata"].get("label", ""),
                "global_step": item["metadata"].get("global_step", None),
                "samples": item["activation_samples"],
                "phase_start": item["metadata"].get("phase_start", None),
                "phase_end": item["metadata"].get("phase_end", None),
            }
            for item in loaded
        ],
        "threshold": float(args.threshold),
        "matrix": str(args.matrix),
        "groups": named_groups(groups, names),
    }
    np.save(args.outdir / "activation_corr_stack.npy", activation_stack.astype(np.float32))
    np.save(args.outdir / "length_corr_stack.npy", length_stack.astype(np.float32))
    np.save(args.outdir / "activation_stable_corr.npy", activation_stable.astype(np.float32))
    np.save(args.outdir / "length_stable_corr.npy", length_stable.astype(np.float32))
    np.save(args.outdir / "combined_group_score.npy", combined.astype(np.float32))
    (args.outdir / "groups.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
