#!/usr/bin/env python3
"""Export a MyoAssist SB3 teacher policy as a phase-indexed muscle prior."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--myoassist-root",
        type=Path,
        default=Path("/home/lzn/myoassist"),
    )
    parser.add_argument(
        "--session-config",
        type=Path,
        default=Path("/home/lzn/myoassist/rl_train/results/train_session_swing_limb_symmetry24_120k/session_config.json"),
    )
    parser.add_argument(
        "--teacher",
        type=Path,
        default=Path("/home/lzn/myoassist/rl_train/results/train_session_swing_limb_symmetry24_120k/trained_models/best_model.zip"),
    )
    parser.add_argument("--out", type=Path, default=Path("data/myoassist_teacher_prior_swing_limb_symmetry24.npz"))
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--reset-indices", type=int, nargs="+", default=[344, 762, 1427, 2010])
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def load_config(myoassist_root: Path, session_config: Path) -> Any:
    sys.path.insert(0, str(myoassist_root))
    from rl_train.envs.environment_handler import EnvironmentHandler
    from rl_train.train.train_configs.config import TrainSessionConfigBase

    base_config = EnvironmentHandler.get_session_config_from_path(str(session_config), TrainSessionConfigBase)
    config_type = EnvironmentHandler.get_config_type_from_session_id(base_config.env_params.env_id)
    config = EnvironmentHandler.get_session_config_from_path(str(session_config), config_type)
    return config


def actuator_names(env: Any) -> list[str]:
    sim = getattr(env, "sim", None)
    if sim is None:
        return []
    names = []
    for idx in range(int(sim.model.nu)):
        try:
            names.append(sim.model.actuator(idx).name)
        except Exception:
            try:
                names.append(sim.model.id2name(idx, "actuator") or str(idx))
            except Exception:
                names.append(str(idx))
    return names


def main() -> None:
    args = parse_args()
    args.out = args.out.resolve()
    if args.summary_csv is not None:
        args.summary_csv = args.summary_csv.resolve()
    myoassist_root = args.myoassist_root.resolve()
    os.chdir(myoassist_root)
    config = load_config(myoassist_root, args.session_config)
    config.env_params.num_envs = 1
    config.env_params.seed = int(args.seed)
    config.env_params.flag_random_ref_index = False
    config.env_params.reference_reset_noise_scale = 0.0
    config.env_params.reference_episode_length = int(args.horizon)
    config.env_params.custom_max_episode_steps = int(args.horizon)
    config.ppo_params.device = "cpu"

    from rl_train.envs.environment_handler import EnvironmentHandler

    rows: list[dict[str, Any]] = []
    actions: list[np.ndarray] = []
    activations: list[np.ndarray] = []
    reference_indices: list[int] = []
    episode_starts: list[int] = []
    episode_steps: list[int] = []
    done_flags: list[bool] = []
    actuator_name_list: list[str] = []

    env = None
    try:
        for reset_index in args.reset_indices:
            config.env_params.reference_reset_start = int(reset_index)
            config.env_params.reference_reset_end = int(reset_index) + 1
            env = EnvironmentHandler.create_environment(config, is_rendering_on=False, is_evaluate_mode=True)
            model = EnvironmentHandler.get_stable_baselines3_model(config, env, trained_model_path=str(args.teacher))
            if not actuator_name_list:
                actuator_name_list = actuator_names(env)
            reset_out = env.reset()
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            for step in range(int(args.horizon)):
                action, _ = model.predict(obs, deterministic=True)
                action = np.asarray(action, dtype=np.float32).reshape(-1)
                activation = np.clip(0.5 * (action + 1.0), 0.0, 1.0).astype(np.float32)
                out = env.step(action)
                if len(out) == 5:
                    obs, reward, done, truncated, info = out
                    terminal = bool(done or truncated)
                else:
                    obs, reward, done, info = out
                    truncated = False
                    terminal = bool(done)
                sim = getattr(env, "sim", None)
                actual_ctrl = np.asarray(sim.data.ctrl, dtype=np.float32).copy() if sim is not None else activation
                actual_act = np.asarray(sim.data.act, dtype=np.float32).copy() if sim is not None else activation
                reference_index = int((int(reset_index) + step) % 1000000000)
                actions.append(action)
                activations.append(actual_ctrl)
                reference_indices.append(reference_index)
                episode_starts.append(int(reset_index))
                episode_steps.append(step)
                done_flags.append(terminal)
                rows.append(
                    {
                        "reset_index": int(reset_index),
                        "step": step,
                        "reference_index": reference_index,
                        "reward": float(reward),
                        "done": terminal,
                        "mean_raw_action": float(np.mean(action)),
                        "min_raw_action": float(np.min(action)),
                        "max_raw_action": float(np.max(action)),
                        "mean_target_activation": float(np.mean(activation)),
                        "mean_ctrl": float(np.mean(actual_ctrl)),
                        "max_ctrl": float(np.max(actual_ctrl)),
                        "mean_act_state": float(np.mean(actual_act)),
                    }
                )
                if terminal:
                    break
            env.close()
            env = None
    finally:
        if env is not None:
            env.close()

    if not activations:
        raise RuntimeError("No teacher rollout samples exported")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        target_activations=np.stack(activations, axis=0).astype(np.float32),
        raw_teacher_actions=np.stack(actions, axis=0).astype(np.float32),
        reference_indices=np.asarray(reference_indices, dtype=np.int64),
        episode_starts=np.asarray(episode_starts, dtype=np.int64),
        episode_steps=np.asarray(episode_steps, dtype=np.int64),
        done=np.asarray(done_flags, dtype=bool),
        actuator_names=np.asarray(actuator_name_list, dtype=object),
        metadata=np.asarray(
            {
                "teacher": str(args.teacher),
                "session_config": str(args.session_config),
                "reset_indices": list(map(int, args.reset_indices)),
                "horizon": int(args.horizon),
                "source_scale": "activation01",
                "action_mapping": "teacher_activation = 0.5 * (raw_teacher_action + 1)",
            },
            dtype=object,
        ),
    )

    summary_csv = args.summary_csv or args.out.with_suffix(".csv")
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "summary_csv": str(summary_csv),
                "samples": len(activations),
                "act_dim": int(np.stack(activations).shape[1]),
                "mean_activation": float(np.mean(np.stack(activations))),
                "max_activation": float(np.max(np.stack(activations))),
                "terminated_episodes": int(sum(done_flags)),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
