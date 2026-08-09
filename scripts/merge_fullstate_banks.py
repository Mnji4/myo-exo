#!/usr/bin/env python3
"""Merge full-state reset banks with an equal row quota per source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files if key != "metadata"}


def select_rows(size: int, count: int) -> np.ndarray:
    if size <= 0:
        raise ValueError("bank is empty")
    return np.rint(np.linspace(0, size - 1, num=count)).astype(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, action="append", required=True)
    parser.add_argument("--rows-per-bank", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    banks = [load(path) for path in args.bank]
    common = set(banks[0])
    for bank in banks[1:]:
        common &= set(bank)
    common = {
        key
        for key in common
        if all(
            bank[key].ndim >= 1
            and int(bank[key].shape[0]) == int(bank["qpos"].shape[0])
            for bank in banks
        )
    }
    required = {"qpos", "qvel", "act", "phase"}
    if not required.issubset(common):
        raise ValueError(f"banks do not share required keys: {sorted(required - common)}")

    merged: dict[str, list[np.ndarray]] = {key: [] for key in sorted(common)}
    source_id: list[np.ndarray] = []
    for idx, bank in enumerate(banks):
        rows = select_rows(int(bank["qpos"].shape[0]), int(args.rows_per_bank))
        for key in merged:
            merged[key].append(bank[key][rows])
        source_id.append(np.full(rows.size, idx, dtype=np.int64))
    arrays = {key: np.concatenate(parts, axis=0) for key, parts in merged.items()}
    arrays["bank_source_id"] = np.concatenate(source_id, axis=0)
    metadata = {
        "sources": [str(path) for path in args.bank],
        "rows_per_bank": int(args.rows_per_bank),
        "contains_activation": True,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays, metadata=np.asarray(metadata, dtype=object))
    print(json.dumps({"out": str(args.out), "states": int(arrays["qpos"].shape[0]), **metadata}, indent=2))


if __name__ == "__main__":
    main()
