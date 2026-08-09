#!/usr/bin/env python3
"""Combine matched no-Exo and assisted hard-constrained solutions."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--noexo", type=Path, required=True)
    parser.add_argument("--assisted", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.noexo, allow_pickle=False) as source:
        noexo = {key: np.asarray(source[key]) for key in source.files}
    with np.load(args.assisted, allow_pickle=False) as source:
        assisted = {key: np.asarray(source[key]) for key in source.files}
    if not np.array_equal(noexo["phase"], assisted["phase"]):
        raise ValueError("solution phases differ")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        phase=assisted["phase"],
        target_torque=assisted["target_torque"],
        muscle_maps=assisted["muscle_maps"],
        exo_maps=assisted["exo_maps"],
        muscle_only_activation=noexo["activation"],
        muscle_only_excitation=noexo["excitation"],
        assisted_activation=assisted["activation"],
        assisted_excitation=assisted["excitation"],
        assisted_exo_control=assisted["exo_control"],
    )


if __name__ == "__main__":
    main()
