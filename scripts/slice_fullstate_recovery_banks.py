#!/usr/bin/env python3
"""Merge selected pre-failure tails from complete-state rollout banks."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tail-start", type=int, default=50)
    parser.add_argument("--tail-end", type=int, default=10)
    args = parser.parse_args()
    if args.tail_start <= args.tail_end or args.tail_end < 0:
        raise ValueError("require tail-start > tail-end >= 0")

    merged: dict[str, list[np.ndarray]] = {}
    metadata: dict[str, np.ndarray] = {}
    for path in args.inputs:
        with np.load(path, allow_pickle=True) as payload:
            row_count = len(payload["qpos"])
            start = max(0, row_count - int(args.tail_start))
            end = max(start + 1, row_count - int(args.tail_end))
            for key in payload.files:
                value = np.asarray(payload[key])
                if value.ndim > 0 and len(value) == row_count:
                    merged.setdefault(key, []).append(value[start:end])
                elif key not in metadata:
                    metadata[key] = value
    if not merged:
        raise ValueError("inputs contain no row-aligned state arrays")
    result = {key: np.concatenate(parts, axis=0) for key, parts in merged.items()}
    result.update(metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **result)
    print({"output": str(args.output), "states": len(result["qpos"])})


if __name__ == "__main__":
    main()
