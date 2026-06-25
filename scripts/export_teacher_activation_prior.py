#!/usr/bin/env python3
"""Export phase-aligned 22-muscle ctrl targets from an existing MyoAssist policy."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


MYOASSIST_ROOT = Path("/home/lzn/myoassist")
if str(MYOASSIST_ROOT) not in sys.path:
    sys.path.insert(0, str(MYOASSIST_ROOT))

from rl_train.envs.environment_handler import EnvironmentHandler
from rl_train.train.train_configs.config import TrainSessionConfigBase
from rl_train.utils.data_types import DictionableDataclass


DEFAULT_SESSION_CONFIG = (
    MYOASSIST_ROOT
    / "rl_train/results/train_session_swing_limb_symmetry24_120k/session_config.json"
)
DEFAULT_MODEL = (
    MYOASSIST_ROOT
    / "rl_train/results/train_session_swing_limb_symmetry24_120k/trained_models/best_model.zip"
)


def get_reference_index(env: Any, info: dict[str, Any]) -> int:
    if hasattr(env, "_get_reference_index"):
        return int(env._get_reference_index())
    if hasattr(env, "_imitation_index") and env._imitation_index is not None:
        return int(env._imitation_index)
    return int(info.get("reference_index", -1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-config", type=Path, default=DEFAULT_SESSION_CONFIG)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=Path("results/teacher_activation_prior/teacher_activation_prior.npz"))
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    os.chdir(MYOASSIST_ROOT)
    with args.session_config.open("r", encoding="utf-8") as f:
        raw_config = json.load(f)
    raw_config["env_params"]["num_envs"] = 1
    raw_config["env_params"]["seed"] = int(args.seed)
    raw_config["env_params"]["flag_random_ref_index"] = True
    raw_config["env_params"]["reference_episode_length"] = int(args.steps)
    raw_config["ppo_params"]["device"] = "cpu"
    config = DictionableDataclass.create(TrainSessionConfigBase, raw_config)
    config_type = EnvironmentHandler.get_config_type_from_session_id(config.env_params.env_id)
    config = DictionableDataclass.create(config_type, raw_config)

    env = EnvironmentHandler.create_environment(config, is_rendering_on=False, is_evaluate_mode=True)
    model = EnvironmentHandler.get_stable_baselines3_model(config, env, trained_model_path=str(args.model))
    unwrapped = env.unwrapped

    rows: list[dict[str, Any]] = []
    target_activations: list[np.ndarray] = []
    target_actions: list[np.ndarray] = []
    reference_indices: list[int] = []
    for episode in range(int(args.episodes)):
        obs, info = env.reset()
        for step in range(int(args.steps)):
            ref_index = get_reference_index(unwrapped, info)
            action, _ = model.predict(obs, deterministic=True)
            obs, _reward, done, truncated, info = env.step(action)
            ctrl = np.asarray(unwrapped.sim.data.ctrl[:22], dtype=np.float32).copy()
            act = np.asarray(unwrapped.sim.data.act[:22], dtype=np.float32).copy()
            action_arr = np.asarray(action, dtype=np.float32).reshape(-1)[:22].copy()
            reference_indices.append(ref_index)
            target_activations.append(ctrl)
            target_actions.append(action_arr)
            row = {
                "episode": episode,
                "step": step,
                "reference_index": ref_index,
                "done": bool(done),
                "truncated": bool(truncated),
                "mean_ctrl": float(np.mean(ctrl)),
                "max_ctrl": float(np.max(ctrl)),
                "mean_act": float(np.mean(act)),
                "max_act": float(np.max(act)),
            }
            for i, value in enumerate(ctrl):
                row[f"ctrl_{i:02d}"] = float(value)
            rows.append(row)
            if done or truncated:
                break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        reference_indices=np.asarray(reference_indices, dtype=np.int64),
        target_activations=np.asarray(target_activations, dtype=np.float32),
        target_actions=np.asarray(target_actions, dtype=np.float32),
        metadata={
            "session_config": str(args.session_config),
            "model": str(args.model),
            "episodes": int(args.episodes),
            "steps": int(args.steps),
        },
    )
    csv_path = args.csv or args.out.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"out": str(args.out), "csv": str(csv_path), "rows": len(rows)}, ensure_ascii=False))
    env.close()


if __name__ == "__main__":
    main()
