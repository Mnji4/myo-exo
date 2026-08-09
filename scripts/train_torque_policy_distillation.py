#!/usr/bin/env python3
"""Distill a muscle policy into a six-dimensional joint-torque policy."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from myo_exo_train.checkpoint import build_sac_actor_for_checkpoint  # noqa: E402
from myo_exo_train.env.model import build_muscle_model  # noqa: E402
from myo_exo_train.env.reference import load_reference_from_config  # noqa: E402
from myo_exo_train.env.runner import MJWarpMuscleRunner  # noqa: E402
from myo_exo_train.evaluation import load_config  # noqa: E402


JOINTS = (
    "hip_flexion_r",
    "knee_angle_r",
    "ankle_angle_r",
    "hip_flexion_l",
    "knee_angle_l",
    "ankle_angle_l",
)


class TorquePolicy(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, len(JOINTS)),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)


def joint_actuator_specs(
    model: mujoco.MjModel, data: mujoco.MjData, device: torch.device
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    specs = []
    for joint_name in JOINTS:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise KeyError(f"missing joint {joint_name}")
        dof = int(model.jnt_dofadr[joint_id])
        actuator_indices = []
        moment_offsets = []
        for actuator in range(int(model.nu)):
            rowadr = int(data.moment_rowadr[actuator])
            rownnz = int(data.moment_rownnz[actuator])
            columns = np.asarray(data.moment_colind[rowadr : rowadr + rownnz])
            matches = np.flatnonzero(columns == dof)
            if matches.size:
                actuator_indices.append(actuator)
                moment_offsets.append(int(matches[0]))
        specs.append(
            (
                torch.tensor(actuator_indices, dtype=torch.long, device=device),
                torch.tensor(moment_offsets, dtype=torch.long, device=device),
            )
        )
    return specs


def current_joint_torque(
    runner: MJWarpMuscleRunner,
    specs: list[tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    values = []
    for actuator_indices, moment_offsets in specs:
        moment_indices = runner.actuator_moment_rowadr.index_select(1, actuator_indices)
        moment_indices = moment_indices + moment_offsets.unsqueeze(0)
        moments = runner.actuator_moment.gather(1, moment_indices)
        forces = runner.actuator_force.index_select(1, actuator_indices)
        values.append(torch.sum(forces * moments, dim=1))
    return torch.stack(values, dim=1)


@torch.no_grad()
def collect_dataset(
    *,
    actor: nn.Module,
    normalizer: nn.Module,
    runner: MJWarpMuscleRunner,
    specs: list[tuple[torch.Tensor, torch.Tensor]],
    frames: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    observations = []
    torques = []
    obs = runner.obs()
    for _ in range(int(frames)):
        normalized_obs = normalizer.normalize(obs)
        action, _, _, _ = actor.get_action_and_value(normalized_obs, deterministic=True)
        next_obs, _reward, done, _terms = runner.step(action)
        valid = ~done
        if bool(valid.any().item()):
            observations.append(normalized_obs[valid].detach().cpu())
            torques.append(current_joint_torque(runner, specs)[valid].detach().cpu())
        obs = next_obs
    return torch.cat(observations, dim=0), torch.cat(torques, dim=0)


def train_policy(
    observations: torch.Tensor,
    torques: torch.Tensor,
    *,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
) -> tuple[TorquePolicy, dict[str, object]]:
    generator = torch.Generator().manual_seed(20260722)
    permutation = torch.randperm(observations.shape[0], generator=generator)
    split = max(1, int(0.9 * observations.shape[0]))
    train_indices = permutation[:split]
    validation_indices = permutation[split:]
    if validation_indices.numel() == 0:
        validation_indices = train_indices[-1:]

    torque_mean = torques[train_indices].mean(dim=0)
    torque_std = torques[train_indices].std(dim=0).clamp_min(1.0)
    model = TorquePolicy(int(observations.shape[1]), hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.0e-5)
    train_obs = observations[train_indices]
    train_targets = (torques[train_indices] - torque_mean) / torque_std

    model.train()
    for _ in range(int(epochs)):
        order = torch.randperm(train_obs.shape[0], generator=generator)
        for start in range(0, train_obs.shape[0], int(batch_size)):
            batch = order[start : start + int(batch_size)]
            prediction = model(train_obs[batch].to(device))
            loss = torch.mean(torch.square(prediction - train_targets[batch].to(device)))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        validation_prediction = (
            model(observations[validation_indices].to(device)).cpu() * torque_std + torque_mean
        )
    error = validation_prediction - torques[validation_indices]
    metrics = {
        "dataset_samples": int(observations.shape[0]),
        "train_samples": int(train_indices.numel()),
        "validation_samples": int(validation_indices.numel()),
        "validation_rmse_nm": torch.sqrt(torch.mean(torch.square(error), dim=0)).tolist(),
        "validation_mae_nm": torch.mean(torch.abs(error), dim=0).tolist(),
        "validation_total_rmse_nm": float(torch.sqrt(torch.mean(torch.square(error))).item()),
        "torque_mean": torque_mean.tolist(),
        "torque_std": torque_std.tolist(),
    }
    return model, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-output", type=Path)
    parser.add_argument("--dataset-input", type=Path)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--collection-frames", type=int, default=640)
    parser.add_argument("--episode-steps", type=int, default=240)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    config = copy.deepcopy(load_config(args.config))
    config["reset"]["episode_steps"] = int(args.episode_steps)
    config["reset"]["phase_indices"] = []
    config["reset"]["phase_index_jitter"] = 0
    config.setdefault("recovery_reset", {})["enabled"] = False
    config.setdefault("offline_recovery_reset", {})["enabled"] = False
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if args.dataset_input is not None:
        dataset = torch.load(args.dataset_input, map_location="cpu")
        observations = dataset["normalized_obs"].float()
        torques = dataset["joint_torque"].float()
    else:
        model, data = build_muscle_model(config)
        reference = load_reference_from_config(
            args.reference, model, float(config["control"]["control_hz"]), device, config
        )
        runner = MJWarpMuscleRunner(
            model=model,
            data=data,
            config=config,
            reference=reference,
            nworld=int(args.num_envs),
            nconmax=128,
            njmax=512,
            seed=20260722,
            device=device,
        )
        actor, normalizer, _ = build_sac_actor_for_checkpoint(
            checkpoint=checkpoint,
            model=model,
            config=config,
            obs_dim=runner.obs_dim,
            act_dim=runner.act_dim,
            device=device,
        )
        actor.eval()
        specs = joint_actuator_specs(model, data, device)
        observations, torques = collect_dataset(
            actor=actor,
            normalizer=normalizer,
            runner=runner,
            specs=specs,
            frames=int(args.collection_frames),
        )
    torque_policy, metrics = train_policy(
        observations,
        torques,
        hidden_dim=int(args.hidden_dim),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        device=device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": torque_policy.state_dict(),
        "obs_dim": int(observations.shape[1]),
        "hidden_dim": int(args.hidden_dim),
        "joint_names": JOINTS,
        "torque_mean": metrics["torque_mean"],
        "torque_std": metrics["torque_std"],
        "source_checkpoint": str(args.checkpoint),
        "source_global_step": int(checkpoint.get("global_step", 0)),
        "metrics": metrics,
    }
    torch.save(payload, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    if args.dataset_output is not None:
        args.dataset_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"normalized_obs": observations, "joint_torque": torques}, args.dataset_output)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
