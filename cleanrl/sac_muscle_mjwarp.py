#!/usr/bin/env python3
"""SAC for MJWarp batched 22-muscle MyoAssist training."""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import mujoco
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cleanrl.ppo_muscle_mjwarp import (  # noqa: E402
    DEFAULT_REFERENCE_PATH,
    FOOT_SITE_NAMES,
    MJWarpMuscleRunner,
    ObsNormalizer,
    TRACK_JOINTS,
    append_csv,
    build_muscle_model,
    evaluate,
    foot_obs_feature_dim,
    frame_stack_feature_dim,
    frame_stack_prev_steps,
    load_config,
    load_reference,
    load_reference_from_config,
    muscle_action_to_activation,
    reference_curriculum_for_update,
    reference_pool_schedule_for_step,
    render_policy_video,
    write_json,
)


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


def swap_side_name(name: str) -> str | None:
    """Swap MuJoCo left/right name tokens while leaving sagittal scalar names intact."""
    swapped = str(name)
    if swapped.startswith("r_"):
        swapped = "l_" + swapped[2:]
    elif swapped.startswith("l_"):
        swapped = "r_" + swapped[2:]
    marker = "\u0000SIDE_R\u0000"
    swapped = swapped.replace("_r", marker).replace("_l", "_r").replace(marker, "_l")
    return swapped if swapped != name else None


def name_mirror_perm(names: list[str]) -> list[int]:
    name_to_idx = {name: idx for idx, name in enumerate(names)}
    perm: list[int] = []
    for idx, name in enumerate(names):
        paired = swap_side_name(name)
        perm.append(name_to_idx.get(paired, idx) if paired is not None else idx)
    for idx, mirrored_idx in enumerate(perm):
        if perm[mirrored_idx] != idx:
            raise ValueError(f"mirror permutation is not involutive at {names[idx]} -> {names[mirrored_idx]}")
    return perm


def mj_names_in_qpos_order(model: mujoco.MjModel) -> list[str]:
    names = [""] * int(model.nq)
    for joint_id in range(int(model.njnt)):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        joint_type = int(model.jnt_type[joint_id])
        qpos_width = 7 if joint_type == int(mujoco.mjtJoint.mjJNT_FREE) else 4 if joint_type == int(mujoco.mjtJoint.mjJNT_BALL) else 1
        start = int(model.jnt_qposadr[joint_id])
        for offset in range(qpos_width):
            names[start + offset] = joint_name if qpos_width == 1 else f"{joint_name}:{offset}"
    if any(name == "" for name in names):
        raise ValueError("could not build complete qpos mirror names")
    return names


def mj_names_in_qvel_order(model: mujoco.MjModel) -> list[str]:
    names = [""] * int(model.nv)
    for joint_id in range(int(model.njnt)):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        joint_type = int(model.jnt_type[joint_id])
        qvel_width = 6 if joint_type == int(mujoco.mjtJoint.mjJNT_FREE) else 3 if joint_type == int(mujoco.mjtJoint.mjJNT_BALL) else 1
        start = int(model.jnt_dofadr[joint_id])
        for offset in range(qvel_width):
            names[start + offset] = joint_name if qvel_width == 1 else f"{joint_name}:{offset}"
    if any(name == "" for name in names):
        raise ValueError("could not build complete qvel mirror names")
    return names


def actuator_names(model: mujoco.MjModel) -> list[str]:
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, idx) for idx in range(int(model.nu))]


def offset_perm(offset: int, perm: list[int]) -> list[int]:
    return [offset + int(index) for index in perm]


def build_sagittal_mirror_spec(
    model: mujoco.MjModel,
    config: dict[str, Any],
    *,
    obs_dim: int,
    future_steps: int,
    device: torch.device,
) -> dict[str, torch.Tensor | dict[str, Any]]:
    qpos_perm = name_mirror_perm(mj_names_in_qpos_order(model))
    qvel_perm = name_mirror_perm(mj_names_in_qvel_order(model))
    act_perm = name_mirror_perm(actuator_names(model))
    track_perm = name_mirror_perm(list(TRACK_JOINTS))
    foot_site_perm = name_mirror_perm(list(FOOT_SITE_NAMES))
    foot_groups = max(1, int(foot_obs_feature_dim(config) // len(FOOT_SITE_NAMES)))
    foot_feature_perm = []
    for group in range(foot_groups):
        foot_feature_perm.extend([group * len(FOOT_SITE_NAMES) + idx for idx in foot_site_perm])
    state_feature_perm = qpos_perm + [
        int(model.nq) + idx for idx in qvel_perm
    ] + [
        int(model.nq) + int(model.nv) + idx for idx in act_perm
    ] + [
        int(model.nq) + int(model.nv) + int(model.na) + idx for idx in foot_feature_perm
    ]
    history_steps = frame_stack_prev_steps(config)
    ref_extra = 1 if bool(config.get("post_reference", {}).get("include_reference_valid_obs", False)) else 0

    obs_perm: list[int] = []
    cursor = 0
    obs_perm.extend(offset_perm(cursor, qpos_perm))
    cursor += int(model.nq)
    obs_perm.extend(offset_perm(cursor, qvel_perm))
    cursor += int(model.nv)
    obs_perm.extend(offset_perm(cursor, act_perm))
    cursor += int(model.na)
    obs_perm.extend(offset_perm(cursor, track_perm))
    cursor += len(TRACK_JOINTS)
    obs_perm.extend(offset_perm(cursor, track_perm))
    cursor += len(TRACK_JOINTS)
    obs_perm.extend([cursor, cursor + 1])  # phase sin/cos: structural mirror only, no half-cycle shift.
    cursor += 2
    obs_perm.extend(offset_perm(cursor, foot_feature_perm))
    cursor += len(foot_feature_perm)
    obs_perm.extend(range(cursor, cursor + ref_extra))
    cursor += ref_extra
    for _ in range(history_steps):
        obs_perm.extend(offset_perm(cursor, state_feature_perm))
        cursor += len(state_feature_perm)
    future_perm = track_perm + [len(TRACK_JOINTS) + idx for idx in foot_feature_perm]
    for _ in range(int(future_steps)):
        obs_perm.extend(offset_perm(cursor, future_perm))
        cursor += len(TRACK_JOINTS) + len(foot_feature_perm)
    if cursor < int(obs_dim):
        obs_perm.extend(range(cursor, int(obs_dim)))
        cursor = int(obs_dim)
    if cursor != int(obs_dim):
        raise ValueError(f"mirror obs layout mismatch: built {cursor}, runner obs_dim is {obs_dim}")
    for idx, mirrored_idx in enumerate(obs_perm):
        if obs_perm[mirrored_idx] != idx:
            raise ValueError(f"obs mirror permutation is not involutive at dim {idx} -> {mirrored_idx}")

    obs_sign = torch.ones(int(obs_dim), dtype=torch.float32, device=device)
    act_sign = torch.ones(int(model.nu), dtype=torch.float32, device=device)
    return {
        "obs_perm": torch.tensor(obs_perm, dtype=torch.long, device=device),
        "obs_sign": obs_sign,
        "act_perm": torch.tensor(act_perm, dtype=torch.long, device=device),
        "act_sign": act_sign,
        "metadata": {
            "future_steps": int(future_steps),
            "phase_mirror": "unchanged",
            "action_perm_names": actuator_names(model),
            "track_joint_perm": [TRACK_JOINTS[idx] for idx in track_perm],
            "foot_site_perm": [FOOT_SITE_NAMES[idx] for idx in foot_site_perm],
        },
    }


def actor_obs_dim_from_state_dict(state_dict: dict[str, torch.Tensor]) -> int:
    for key in (
        "backbone.0.weight",
        "base_actor.backbone.0.weight",
        "base_encoder.0.weight",
        "base_actor.base_encoder.0.weight",
    ):
        if key in state_dict:
            return int(state_dict[key].shape[1])
    raise KeyError("could not infer actor obs_dim from checkpoint state dict")


def policy_architecture(config: dict[str, Any]) -> str:
    return str(config.get("policy", {}).get("architecture", "mlp_sac") or "mlp_sac")


def gated_ref_obs_spec(
    model: mujoco.MjModel,
    config: dict[str, Any],
    *,
    obs_dim: int,
    device: torch.device,
) -> dict[str, torch.Tensor | dict[str, Any]]:
    future_steps = max(0, int(config.get("imitation", {}).get("reference_future_steps", 0) or 0))
    terrain_dim = max(0, int(config.get("terrain_context", {}).get("num_preview_samples", 0) or 0)) if bool(
        config.get("terrain_context", {}).get("include_height_preview", False)
    ) else 0
    ref_extra = 1 if bool(config.get("post_reference", {}).get("include_reference_valid_obs", False)) else 0
    foot_dim = foot_obs_feature_dim(config)
    history_steps = frame_stack_prev_steps(config)
    history_step_dim = frame_stack_feature_dim(config, nq=int(model.nq), nv=int(model.nv), na=int(model.na))

    cursor = 0
    qpos = list(range(cursor, cursor + int(model.nq)))
    cursor += int(model.nq)
    qvel = list(range(cursor, cursor + int(model.nv)))
    cursor += int(model.nv)
    act = list(range(cursor, cursor + int(model.na)))
    cursor += int(model.na)
    ref_q = list(range(cursor, cursor + len(TRACK_JOINTS)))
    cursor += len(TRACK_JOINTS)
    ref_dq = list(range(cursor, cursor + len(TRACK_JOINTS)))
    cursor += len(TRACK_JOINTS)
    phase = list(range(cursor, cursor + 2))
    cursor += 2
    foot = list(range(cursor, cursor + foot_dim))
    cursor += foot_dim
    ref_valid = list(range(cursor, cursor + ref_extra))
    cursor += ref_extra
    history: list[int] = []
    for _ in range(history_steps):
        history.extend(range(cursor, cursor + history_step_dim))
        cursor += history_step_dim
    future: list[int] = []
    future_step_dim = len(TRACK_JOINTS) + foot_dim
    for _ in range(future_steps):
        future.extend(range(cursor, cursor + future_step_dim))
        cursor += future_step_dim
    terrain = list(range(cursor, cursor + terrain_dim))
    cursor += terrain_dim
    if cursor != int(obs_dim):
        raise ValueError(f"gated ref obs layout mismatch: built {cursor}, runner obs_dim is {obs_dim}")

    base_indices = qpos + qvel + act + foot + history + terrain
    ref_indices = ref_q + ref_dq + phase + ref_valid + future
    return {
        "base_indices": torch.tensor(base_indices, dtype=torch.long, device=device),
        "ref_indices": torch.tensor(ref_indices, dtype=torch.long, device=device),
        "metadata": {
            "base_dim": len(base_indices),
            "ref_dim": len(ref_indices),
            "qpos": [qpos[0], qpos[-1] + 1] if qpos else [0, 0],
            "qvel": [qvel[0], qvel[-1] + 1] if qvel else [0, 0],
            "act": [act[0], act[-1] + 1] if act else [0, 0],
            "ref_q_error": [ref_q[0], ref_q[-1] + 1] if ref_q else [0, 0],
            "ref_dq_error": [ref_dq[0], ref_dq[-1] + 1] if ref_dq else [0, 0],
            "phase": [phase[0], phase[-1] + 1] if phase else [0, 0],
            "foot": [foot[0], foot[-1] + 1] if foot else [0, 0],
            "reference_valid": [ref_valid[0], ref_valid[-1] + 1] if ref_valid else [cursor, cursor],
            "history": [history[0], history[-1] + 1] if history else [cursor, cursor],
            "future": [future[0], future[-1] + 1] if future else [cursor, cursor],
            "terrain": [terrain[0], terrain[-1] + 1] if terrain else [cursor, cursor],
            "future_steps": future_steps,
            "history_steps": history_steps,
            "history_step_dim": history_step_dim,
            "terrain_dim": terrain_dim,
            "reference_extra_dim": ref_extra,
        },
    }


def ref_gate_for_step(config: dict[str, Any], global_step: int, run_start_global_step: int) -> float:
    policy_cfg = config.get("policy", {})
    gate = float(policy_cfg.get("ref_gate", 1.0))
    schedule = policy_cfg.get("ref_gate_schedule", [])
    if not isinstance(schedule, list) or not schedule:
        return max(0.0, min(1.0, gate))
    schedule_step = int(global_step)
    if str(config.get("reward_schedule_mode", "relative")) == "relative":
        schedule_step = max(0, int(global_step) - int(run_start_global_step))
    for item in sorted(schedule, key=lambda x: int(x.get("after_steps", 0))):
        start = int(item.get("after_steps", 0))
        if schedule_step < start:
            continue
        if "gate" in item:
            gate = float(item["gate"])
        duration = int(item.get("duration_steps", 0) or 0)
        if duration > 0 and "end_gate" in item:
            start_gate = float(item.get("gate", gate))
            end_gate = float(item["end_gate"])
            progress = max(0.0, min(1.0, float(schedule_step - start) / float(duration)))
            gate = start_gate + progress * (end_gate - start_gate)
    return max(0.0, min(1.0, float(gate)))


def set_actor_ref_gate(actor: nn.Module, gate: float) -> None:
    if hasattr(actor, "set_ref_gate"):
        actor.set_ref_gate(gate)  # type: ignore[attr-defined]
    elif isinstance(actor, SymmetricSACActor) and hasattr(actor.base_actor, "set_ref_gate"):
        actor.base_actor.set_ref_gate(gate)  # type: ignore[attr-defined]


def mask_ref_obs_for_q(obs: torch.Tensor, ref_indices: torch.Tensor | None, ref_gate: float) -> torch.Tensor:
    if ref_indices is None or int(ref_indices.numel()) == 0 or float(ref_gate) >= 0.999999:
        return obs
    masked = obs.clone()
    masked.index_copy_(
        1,
        ref_indices,
        masked.index_select(1, ref_indices) * float(ref_gate),
    )
    return masked


def reward_weights_for_step(config: dict[str, Any], global_step: int, run_start_global_step: int) -> dict[str, float]:
    weights = {str(k): float(v) for k, v in config.get("reward", {}).items()}
    schedule = config.get("reward_schedule", [])
    if not isinstance(schedule, list) or not schedule:
        return weights
    schedule_step = int(global_step)
    if str(config.get("reward_schedule_mode", "relative")) == "relative":
        schedule_step = max(0, int(global_step) - int(run_start_global_step))
    for item in sorted(schedule, key=lambda x: int(x.get("after_steps", 0))):
        if schedule_step >= int(item.get("after_steps", 0)):
            weights.update({str(k): float(v) for k, v in item.get("weights", {}).items()})
    return weights


def reset_phase_schedule_for_step(config: dict[str, Any], global_step: int, run_start_global_step: int) -> dict[str, Any] | None:
    schedule = config.get("reset_phase_schedule", [])
    if not isinstance(schedule, list) or not schedule:
        return None
    schedule_step = int(global_step)
    if str(config.get("reset_phase_schedule_mode", "absolute")) == "relative":
        schedule_step = max(0, int(global_step) - int(run_start_global_step))
    current = schedule[0]
    for item in sorted(schedule, key=lambda x: int(x.get("after_steps", 0))):
        if schedule_step >= int(item.get("after_steps", 0)):
            current = item
    return {
        "name": str(current.get("name", "")),
        "after_steps": int(current.get("after_steps", 0)),
        "phase_windows": list(current.get("phase_windows", [])),
        "phase_indices": list(current.get("phase_indices", [])),
    }


def apply_reset_phase_stage(runner: MJWarpMuscleRunner, stage: dict[str, Any] | None) -> None:
    if stage is None:
        runner.phase_choices = runner.build_phase_choices(
            runner.config["reset"].get("phase_windows", []),
            runner.config["reset"].get("phase_indices", []),
            int(runner.config["reset"].get("phase_index_jitter", 0) or 0),
            int(runner.reference["length"]),
        )
        return
    runner.phase_choices = runner.build_phase_choices(
        stage.get("phase_windows", []),
        stage.get("phase_indices", []),
        int(runner.config["reset"].get("phase_index_jitter", 0) or 0),
        int(runner.reference["length"]),
    )


def apply_reward_schedule(config: dict[str, Any], runner: MJWarpMuscleRunner, global_step: int, run_start_global_step: int) -> dict[str, float]:
    weights = reward_weights_for_step(config, global_step, run_start_global_step)
    config["reward"] = dict(weights)
    runner.reward_weights = dict(weights)
    return weights


def future_obs_dropout_prob_for_step(config: dict[str, Any], global_step: int, run_start_global_step: int) -> float:
    imitation = config.get("imitation", {})
    prob = float(imitation.get("future_obs_dropout_prob", 0.0) or 0.0)
    schedule = imitation.get("future_obs_dropout_schedule", [])
    if isinstance(schedule, list):
        schedule_step = int(global_step)
        if str(config.get("reward_schedule_mode", "relative")) == "relative":
            schedule_step = max(0, int(global_step) - int(run_start_global_step))
        for item in sorted(schedule, key=lambda x: int(x.get("after_steps", 0))):
            if schedule_step >= int(item.get("after_steps", 0)):
                prob = float(item.get("prob", prob))
    return max(0.0, min(1.0, prob))


def set_future_obs_dropout_prob(config: dict[str, Any], prob: float) -> None:
    config.setdefault("imitation", {})["current_future_obs_dropout_prob"] = max(0.0, min(1.0, float(prob)))


def configured_video_phases(
    config: dict[str, Any],
    reference: dict[str, Any],
    fallback_phase: int,
    *,
    global_step: int | None = None,
    run_start_global_step: int = 0,
    video_every: int = 0,
) -> list[int]:
    phases = config.get("video", {}).get("phase_indices", [])
    if not isinstance(phases, list) or not phases:
        return [int(fallback_phase) % int(reference["length"])]
    selected = [int(phase) % int(reference["length"]) for phase in phases]
    phase_mode = str(config.get("video", {}).get("phase_mode", "all")).lower()
    if phase_mode in {"round_robin", "one_per_event"} and len(selected) > 1:
        if global_step is None or int(video_every) <= 0:
            event_index = 0
        else:
            event_index = max(0, (int(global_step) - int(run_start_global_step)) // int(video_every))
        return [selected[event_index % len(selected)]]
    return selected


def mem_available_gb() -> float:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return float(line.split()[1]) / (1024.0 * 1024.0)
    except OSError:
        return float("inf")
    return float("inf")


def pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        import os

        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def active_lock_pid(lock_file: Path) -> int | None:
    if not lock_file.exists():
        return None
    try:
        pid = int(lock_file.read_text(encoding="utf-8").strip())
    except ValueError:
        return None
    return pid if pid_is_running(pid) else None


def maybe_launch_checkpoint_video_export(
    *,
    config: dict[str, Any],
    args: argparse.Namespace,
    checkpoint_path: Path,
    global_step: int,
    nworld: int,
    active_process: subprocess.Popen | None,
) -> tuple[subprocess.Popen | None, dict[str, Any]]:
    export_cfg = config.get("checkpoint_video_export", {})
    if not bool(export_cfg.get("enabled", False)):
        return active_process, {"global_step": int(global_step), "status": "disabled"}

    if active_process is not None and active_process.poll() is None:
        return active_process, {
            "global_step": int(global_step),
            "status": "skipped_running_process",
            "pid": int(active_process.pid),
        }

    outdir = Path(export_cfg.get("outdir", args.outdir / "videos_from_checkpoints"))
    if not outdir.is_absolute():
        outdir = args.outdir / outdir if str(outdir).startswith("videos") else ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)
    lock_file = outdir / "export.lock"
    lock_pid = active_lock_pid(lock_file)
    if lock_pid is not None:
        return active_process, {
            "global_step": int(global_step),
            "status": "skipped_lock_active",
            "pid": int(lock_pid),
        }

    min_mem_gb = float(export_cfg.get("min_available_memory_gb", 2.0))
    available_gb = mem_available_gb()
    if available_gb < min_mem_gb:
        return active_process, {
            "global_step": int(global_step),
            "status": "skipped_low_memory",
            "mem_available_gb": available_gb,
            "min_available_memory_gb": min_mem_gb,
        }

    every_steps = int(export_cfg.get("every_steps", args.checkpoint_every) or 0)
    if every_steps > 0 and int(global_step) % every_steps >= int(nworld):
        return active_process, {"global_step": int(global_step), "status": "skipped_interval"}

    phases = export_cfg.get("phase_indices", config.get("video", {}).get("phase_indices", []))
    command = [
        sys.executable,
        str(ROOT / "scripts" / "render_sac_checkpoint_videos.py"),
        "--config",
        str(args.config.resolve()),
        "--checkpoint",
        str(checkpoint_path.resolve()),
        "--outdir",
        str(outdir.resolve()),
        "--reference",
        str(args.reference.resolve()),
        "--device",
        str(export_cfg.get("device", "cpu")),
        "--video-steps",
        str(int(export_cfg.get("video_steps", args.video_steps))),
        "--video-height",
        str(int(export_cfg.get("video_height", args.video_height))),
        "--video-width",
        str(int(export_cfg.get("video_width", args.video_width))),
        "--video-camera-distance",
        str(float(export_cfg.get("video_camera_distance", args.video_camera_distance))),
        "--video-camera-height",
        str(float(export_cfg.get("video_camera_height", args.video_camera_height))),
        "--lock-file",
        str(lock_file.resolve()),
    ]
    for phase in phases if isinstance(phases, list) else []:
        command.extend(["--phase", str(int(phase))])
    if args.video_activation_prior_execution_mix is not None:
        command.extend(["--video-activation-prior-execution-mix", str(float(args.video_activation_prior_execution_mix))])

    log_path = outdir / f"export_step_{int(global_step):09d}.log"
    with log_path.open("ab") as log_file:
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return process, {
        "global_step": int(global_step),
        "status": "started",
        "pid": int(process.pid),
        "checkpoint": str(checkpoint_path),
        "log": str(log_path),
        "outdir": str(outdir),
        "mem_available_gb": available_gb,
    }


class SACActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, logstd_init: float, initial_action_mean: float):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mean = nn.Linear(256, act_dim)
        self.logstd = nn.Linear(256, act_dim)
        nn.init.zeros_(self.mean.weight)
        nn.init.constant_(self.mean.bias, float(initial_action_mean))
        nn.init.zeros_(self.logstd.weight)
        nn.init.constant_(self.logstd.bias, float(logstd_init))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(obs)
        mean = self.mean(h)
        logstd = torch.clamp(self.logstd(h), LOG_STD_MIN, LOG_STD_MAX)
        return mean, logstd

    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        mean, logstd = self(obs)
        if deterministic:
            action = torch.tanh(mean)
            logprob = torch.zeros(action.shape[0], dtype=torch.float32, device=action.device)
            return action, logprob
        std = torch.exp(logstd)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        logprob = normal.log_prob(x_t)
        logprob -= torch.log(torch.clamp(1.0 - torch.square(action), min=1e-6))
        return action, logprob.sum(dim=1)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if action is None:
            action, logprob = self.get_action(obs, deterministic=deterministic)
        else:
            _mean, _logstd = self(obs)
            logprob = torch.zeros(action.shape[0], dtype=torch.float32, device=action.device)
        entropy = torch.zeros(action.shape[0], dtype=torch.float32, device=action.device)
        value = torch.zeros((action.shape[0], 1), dtype=torch.float32, device=action.device)
        return action, logprob, entropy, value


class GatedRefSACActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        *,
        base_indices: torch.Tensor,
        ref_indices: torch.Tensor,
        logstd_init: float,
        initial_action_mean: float,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        initial_ref_gate: float = 1.0,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.register_buffer("base_indices", base_indices.detach().clone().long())
        self.register_buffer("ref_indices", ref_indices.detach().clone().long())
        self.register_buffer("ref_gate", torch.tensor(float(initial_ref_gate), dtype=torch.float32))
        self.base_encoder = nn.Sequential(
            nn.Linear(int(self.base_indices.numel()), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(latent_dim)),
            nn.ReLU(),
        )
        self.ref_encoder = nn.Sequential(
            nn.Linear(int(self.ref_indices.numel()), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(latent_dim)),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(int(latent_dim) * 2, int(hidden_dim)),
            nn.ReLU(),
        )
        self.mean = nn.Linear(int(hidden_dim), act_dim)
        self.logstd = nn.Linear(int(hidden_dim), act_dim)
        nn.init.zeros_(self.mean.weight)
        nn.init.constant_(self.mean.bias, float(initial_action_mean))
        nn.init.zeros_(self.logstd.weight)
        nn.init.constant_(self.logstd.bias, float(logstd_init))

    def set_ref_gate(self, gate: float) -> None:
        self.ref_gate.fill_(max(0.0, min(1.0, float(gate))))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        base_obs = obs.index_select(-1, self.base_indices)
        ref_obs = obs.index_select(-1, self.ref_indices)
        z_base = self.base_encoder(base_obs)
        z_ref = self.ref_encoder(ref_obs) * self.ref_gate.to(dtype=z_base.dtype, device=z_base.device)
        h = self.head(torch.cat([z_base, z_ref], dim=-1))
        mean = self.mean(h)
        logstd = torch.clamp(self.logstd(h), LOG_STD_MIN, LOG_STD_MAX)
        return mean, logstd

    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        mean, logstd = self(obs)
        if deterministic:
            action = torch.tanh(mean)
            logprob = torch.zeros(action.shape[0], dtype=torch.float32, device=action.device)
            return action, logprob
        std = torch.exp(logstd)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        logprob = normal.log_prob(x_t)
        logprob -= torch.log(torch.clamp(1.0 - torch.square(action), min=1e-6))
        return action, logprob.sum(dim=1)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if action is None:
            action, logprob = self.get_action(obs, deterministic=deterministic)
        else:
            logprob = torch.zeros(action.shape[0], dtype=torch.float32, device=action.device)
        entropy = torch.zeros(action.shape[0], dtype=torch.float32, device=action.device)
        value = torch.zeros((action.shape[0], 1), dtype=torch.float32, device=action.device)
        return action, logprob, entropy, value


class SymmetricSACActor(nn.Module):
    def __init__(
        self,
        base_actor: SACActor,
        *,
        obs_perm: torch.Tensor,
        obs_sign: torch.Tensor,
        act_perm: torch.Tensor,
        act_sign: torch.Tensor,
    ):
        super().__init__()
        self.base_actor = base_actor
        self.register_buffer("obs_perm", obs_perm.detach().clone().long())
        self.register_buffer("obs_sign", obs_sign.detach().clone().float())
        self.register_buffer("act_perm", act_perm.detach().clone().long())
        self.register_buffer("act_sign", act_sign.detach().clone().float())

    def mirror_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return obs.index_select(-1, self.obs_perm) * self.obs_sign

    def mirror_action(self, action: torch.Tensor) -> torch.Tensor:
        return action.index_select(-1, self.act_perm) * self.act_sign

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean0, logstd0 = self.base_actor(obs)
        mean_m, logstd_m = self.base_actor(self.mirror_obs(obs))
        mean = 0.5 * (mean0 + self.mirror_action(mean_m))
        logstd = 0.5 * (logstd0 + self.mirror_action(logstd_m))
        return mean, torch.clamp(logstd, LOG_STD_MIN, LOG_STD_MAX)

    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        mean, logstd = self(obs)
        if deterministic:
            action = torch.tanh(mean)
            logprob = torch.zeros(action.shape[0], dtype=torch.float32, device=action.device)
            return action, logprob
        std = torch.exp(logstd)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        logprob = normal.log_prob(x_t)
        logprob -= torch.log(torch.clamp(1.0 - torch.square(action), min=1e-6))
        return action, logprob.sum(dim=1)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if action is None:
            action, logprob = self.get_action(obs, deterministic=deterministic)
        else:
            logprob = torch.zeros(action.shape[0], dtype=torch.float32, device=action.device)
        entropy = torch.zeros(action.shape[0], dtype=torch.float32, device=action.device)
        value = torch.zeros((action.shape[0], 1), dtype=torch.float32, device=action.device)
        return action, logprob, entropy, value


class SoftQNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=1)).squeeze(1)


class SymmetricSoftQNetwork(nn.Module):
    def __init__(
        self,
        base_q: SoftQNetwork,
        *,
        obs_perm: torch.Tensor,
        obs_sign: torch.Tensor,
        act_perm: torch.Tensor,
        act_sign: torch.Tensor,
    ):
        super().__init__()
        self.base_q = base_q
        self.register_buffer("obs_perm", obs_perm.detach().clone().long())
        self.register_buffer("obs_sign", obs_sign.detach().clone().float())
        self.register_buffer("act_perm", act_perm.detach().clone().long())
        self.register_buffer("act_sign", act_sign.detach().clone().float())

    def mirror_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return obs.index_select(-1, self.obs_perm) * self.obs_sign

    def mirror_action(self, action: torch.Tensor) -> torch.Tensor:
        return action.index_select(-1, self.act_perm) * self.act_sign

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q0 = self.base_q(obs, action)
        q1 = self.base_q(self.mirror_obs(obs), self.mirror_action(action))
        return 0.5 * (q0 + q1)


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int, device: torch.device):
        self.capacity = int(capacity)
        self.device = device
        self.obs = torch.empty((self.capacity, obs_dim), dtype=torch.float32, device=device)
        self.next_obs = torch.empty((self.capacity, obs_dim), dtype=torch.float32, device=device)
        self.action = torch.empty((self.capacity, act_dim), dtype=torch.float32, device=device)
        self.reward = torch.empty(self.capacity, dtype=torch.float32, device=device)
        self.done = torch.empty(self.capacity, dtype=torch.float32, device=device)
        self.next_idx = 0
        self.size = 0

    @torch.no_grad()
    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        n = int(obs.shape[0])
        if n >= self.capacity:
            self.obs.copy_(obs[-self.capacity :].detach())
            self.action.copy_(action[-self.capacity :].detach())
            self.reward.copy_(reward[-self.capacity :].detach())
            self.next_obs.copy_(next_obs[-self.capacity :].detach())
            self.done.copy_(done[-self.capacity :].float().detach())
            self.next_idx = 0
            self.size = self.capacity
            return
        first = min(n, self.capacity - self.next_idx)
        dst = slice(self.next_idx, self.next_idx + first)
        self.obs[dst].copy_(obs[:first].detach())
        self.action[dst].copy_(action[:first].detach())
        self.reward[dst].copy_(reward[:first].detach())
        self.next_obs[dst].copy_(next_obs[:first].detach())
        self.done[dst].copy_(done[:first].float().detach())
        remain = n - first
        if remain > 0:
            self.obs[:remain].copy_(obs[first:].detach())
            self.action[:remain].copy_(action[first:].detach())
            self.reward[:remain].copy_(reward[first:].detach())
            self.next_obs[:remain].copy_(next_obs[first:].detach())
            self.done[:remain].copy_(done[first:].float().detach())
        self.next_idx = (self.next_idx + n) % self.capacity
        self.size = min(self.capacity, self.size + n)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = torch.randint(0, self.size, (int(batch_size),), device=self.device)
        return self.obs[idx], self.action[idx], self.reward[idx], self.next_obs[idx], self.done[idx]


def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for src_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.mul_(1.0 - tau)
            target_param.data.add_(tau * src_param.data)


def save_checkpoint(
    path: Path,
    *,
    actor: nn.Module,
    qf1: nn.Module,
    qf2: nn.Module,
    qf1_target: nn.Module,
    qf2_target: nn.Module,
    actor_optimizer: optim.Optimizer,
    q_optimizer: optim.Optimizer,
    alpha_optimizer: optim.Optimizer,
    log_alpha: torch.Tensor,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor_state_dict": actor.state_dict(),
            "qf1_state_dict": qf1.state_dict(),
            "qf2_state_dict": qf2.state_dict(),
            "qf1_target_state_dict": qf1_target.state_dict(),
            "qf2_target_state_dict": qf2_target.state_dict(),
            "actor_optimizer_state_dict": actor_optimizer.state_dict(),
            "q_optimizer_state_dict": q_optimizer.state_dict(),
            "alpha_optimizer_state_dict": alpha_optimizer.state_dict(),
            "log_alpha": log_alpha.detach().clone(),
            **payload,
        },
        path,
    )


def load_shape_compatible_state_dict(module: nn.Module, state_dict: dict[str, torch.Tensor]) -> bool:
    current = module.state_dict()
    exact = True
    with torch.no_grad():
        for key, source in state_dict.items():
            if key not in current:
                exact = False
                continue
            target = current[key]
            source = source.to(device=target.device, dtype=target.dtype)
            if tuple(source.shape) == tuple(target.shape):
                target.copy_(source)
                continue
            exact = False
            if source.ndim != target.ndim:
                continue
            target.zero_()
            slices = tuple(slice(0, min(int(a), int(b))) for a, b in zip(target.shape, source.shape))
            target[slices].copy_(source[slices])
    module.load_state_dict(current)
    return exact


def load_shape_compatible_q_state_dict(
    module: nn.Module,
    state_dict: dict[str, torch.Tensor],
    *,
    old_obs_dim: int,
    new_obs_dim: int,
    act_dim: int,
) -> bool:
    current = module.state_dict()
    exact = True
    with torch.no_grad():
        for key, source in state_dict.items():
            if key not in current:
                exact = False
                continue
            target = current[key]
            source = source.to(device=target.device, dtype=target.dtype)
            if tuple(source.shape) == tuple(target.shape):
                target.copy_(source)
                continue
            exact = False
            if key in {"net.0.weight", "base_q.net.0.weight"} and source.ndim == 2 and target.ndim == 2:
                target.zero_()
                obs_cols = min(int(old_obs_dim), int(new_obs_dim), int(source.shape[1]), int(target.shape[1]))
                target[:, :obs_cols].copy_(source[:, :obs_cols])
                old_action_start = int(old_obs_dim)
                new_action_start = int(new_obs_dim)
                action_cols = min(
                    int(act_dim),
                    max(0, int(source.shape[1]) - old_action_start),
                    max(0, int(target.shape[1]) - new_action_start),
                )
                if action_cols > 0:
                    target[:, new_action_start : new_action_start + action_cols].copy_(
                        source[:, old_action_start : old_action_start + action_cols]
                    )
                continue
            if source.ndim != target.ndim:
                continue
            target.zero_()
            slices = tuple(slice(0, min(int(a), int(b))) for a, b in zip(target.shape, source.shape))
            target[slices].copy_(source[slices])
    module.load_state_dict(current)
    return exact


@torch.no_grad()
def symmetric_module_self_test(
    actor: SymmetricSACActor,
    qf: SymmetricSoftQNetwork,
    *,
    obs_dim: int,
    act_dim: int,
    device: torch.device,
) -> dict[str, float]:
    obs = torch.randn((1024, int(obs_dim)), dtype=torch.float32, device=device)
    action = torch.randn((1024, int(act_dim)), dtype=torch.float32, device=device)
    obs_roundtrip = actor.mirror_obs(actor.mirror_obs(obs))
    action_roundtrip = actor.mirror_action(actor.mirror_action(action))
    mean, logstd = actor(obs)
    mean_from_mirror, logstd_from_mirror = actor(actor.mirror_obs(obs))
    q = qf(obs, action)
    q_from_mirror = qf(actor.mirror_obs(obs), actor.mirror_action(action))
    return {
        "obs_roundtrip_max_abs": float((obs_roundtrip - obs).abs().max().item()),
        "action_roundtrip_max_abs": float((action_roundtrip - action).abs().max().item()),
        "actor_mean_equivariance_max_abs": float((actor.mirror_action(mean_from_mirror) - mean).abs().max().item()),
        "actor_logstd_equivariance_max_abs": float((actor.mirror_action(logstd_from_mirror) - logstd).abs().max().item()),
        "q_invariance_max_abs": float((q_from_mirror - q).abs().max().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "muscle_2d_mjwarp_teacher_stage1_swing_hip_sac.json")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE_PATH)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--nworld", type=int, default=None)
    parser.add_argument("--nconmax", type=int, default=128)
    parser.add_argument("--njmax", type=int, default=512)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--episode-steps", type=int, default=None)
    parser.add_argument("--qpos-noise", type=float, default=None)
    parser.add_argument("--qvel-noise", type=float, default=None)
    parser.add_argument("--eval-every", type=int, default=8192)
    parser.add_argument("--eval-worlds", type=int, default=16)
    parser.add_argument("--eval-steps", type=int, default=12)
    parser.add_argument("--video-every", type=int, default=8192)
    parser.add_argument("--video-steps", type=int, default=12)
    parser.add_argument("--video-phase", type=int, default=344)
    parser.add_argument("--video-height", type=int, default=720)
    parser.add_argument("--video-width", type=int, default=1280)
    parser.add_argument("--video-camera-distance", type=float, default=7.0)
    parser.add_argument("--video-camera-height", type=float, default=0.9)
    parser.add_argument("--video-activation-prior-execution-mix", type=float, default=None)
    parser.add_argument("--render-only-video", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=8192)
    parser.add_argument("--log-every", type=int, default=8192)
    args = parser.parse_args()

    if args.device != "cuda":
        raise SystemExit("MJWarp SAC is intended for --device cuda")
    device = torch.device(args.device)
    config = load_config(args.config)
    if args.episode_steps is not None:
        config["reset"]["episode_steps"] = int(args.episode_steps)
    if args.qpos_noise is not None:
        config["reset"]["qpos_noise"] = float(args.qpos_noise)
    if args.qvel_noise is not None:
        config["reset"]["qvel_noise"] = float(args.qvel_noise)

    sac_cfg = config.get("sac", config.get("ppo", {}))
    seed = int(args.seed if args.seed is not None else config["seed"])
    args.seed = seed
    total_timesteps = int(args.total_timesteps if args.total_timesteps is not None else sac_cfg["total_timesteps"])
    nworld = int(args.nworld if args.nworld is not None else sac_cfg["num_envs"])
    batch_size = int(sac_cfg.get("batch_size", 1024))
    learning_starts = int(sac_cfg.get("learning_starts", 1024))
    train_freq = int(sac_cfg.get("train_freq", 1))
    gradient_steps = int(sac_cfg.get("gradient_steps", 2))
    gamma = float(sac_cfg.get("gamma", 0.99))
    tau = float(sac_cfg.get("tau", 0.005))
    target_entropy = float(sac_cfg.get("target_entropy", -float("nan")))
    symmetric_policy = bool(sac_cfg.get("symmetric_policy", False))
    architecture = policy_architecture(config)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if args.outdir is None:
        args.outdir = ROOT / "results" / f"mjwarp_muscle_sac_{time.strftime('%Y%m%d-%H%M%S')}"
    args.outdir.mkdir(parents=True, exist_ok=True)

    model, data = build_muscle_model(config)
    reference = load_reference_from_config(args.reference, model, float(config["control"]["control_hz"]), device, config)
    runner = MJWarpMuscleRunner(
        model=model,
        data=data,
        config=config,
        reference=reference,
        nworld=nworld,
        nconmax=int(args.nconmax),
        njmax=int(args.njmax),
        seed=seed,
        device=device,
    )
    obs_normalizer = ObsNormalizer(
        runner.obs_dim,
        device,
        enabled=bool(sac_cfg.get("normalize_observations", True)),
        clip=float(sac_cfg.get("obs_norm_clip", 10.0)),
    )
    mirror_spec: dict[str, torch.Tensor | dict[str, Any]] | None = None
    if symmetric_policy:
        mirror_spec = build_sagittal_mirror_spec(
            model,
            config,
            obs_dim=runner.obs_dim,
            future_steps=int(config.get("imitation", {}).get("reference_future_steps", 0)),
            device=device,
        )
    gated_spec: dict[str, torch.Tensor | dict[str, Any]] | None = None
    current_ref_gate = ref_gate_for_step(config, 0, 0)
    if architecture == "gated_ref_sac":
        gated_spec = gated_ref_obs_spec(model, config, obs_dim=runner.obs_dim, device=device)
        base_actor = GatedRefSACActor(
            runner.obs_dim,
            runner.act_dim,
            base_indices=gated_spec["base_indices"],
            ref_indices=gated_spec["ref_indices"],
            logstd_init=float(sac_cfg.get("actor_logstd_init", -0.5)),
            initial_action_mean=float(sac_cfg.get("initial_actor_action_mean", -0.2)),
            hidden_dim=int(config.get("policy", {}).get("hidden_dim", 256)),
            latent_dim=int(config.get("policy", {}).get("latent_dim", 128)),
            initial_ref_gate=current_ref_gate,
        ).to(device)
    else:
        base_actor = SACActor(
            runner.obs_dim,
            runner.act_dim,
            logstd_init=float(sac_cfg.get("actor_logstd_init", -0.5)),
            initial_action_mean=float(sac_cfg.get("initial_actor_action_mean", -0.2)),
        ).to(device)
    if mirror_spec is None:
        actor: nn.Module = base_actor
        qf1: nn.Module = SoftQNetwork(runner.obs_dim, runner.act_dim).to(device)
        qf2: nn.Module = SoftQNetwork(runner.obs_dim, runner.act_dim).to(device)
        qf1_target: nn.Module = SoftQNetwork(runner.obs_dim, runner.act_dim).to(device)
        qf2_target: nn.Module = SoftQNetwork(runner.obs_dim, runner.act_dim).to(device)
    else:
        actor = SymmetricSACActor(
            base_actor,
            obs_perm=mirror_spec["obs_perm"],
            obs_sign=mirror_spec["obs_sign"],
            act_perm=mirror_spec["act_perm"],
            act_sign=mirror_spec["act_sign"],
        ).to(device)
        qf1 = SymmetricSoftQNetwork(
            SoftQNetwork(runner.obs_dim, runner.act_dim).to(device),
            obs_perm=mirror_spec["obs_perm"],
            obs_sign=mirror_spec["obs_sign"],
            act_perm=mirror_spec["act_perm"],
            act_sign=mirror_spec["act_sign"],
        ).to(device)
        qf2 = SymmetricSoftQNetwork(
            SoftQNetwork(runner.obs_dim, runner.act_dim).to(device),
            obs_perm=mirror_spec["obs_perm"],
            obs_sign=mirror_spec["obs_sign"],
            act_perm=mirror_spec["act_perm"],
            act_sign=mirror_spec["act_sign"],
        ).to(device)
        qf1_target = SymmetricSoftQNetwork(
            SoftQNetwork(runner.obs_dim, runner.act_dim).to(device),
            obs_perm=mirror_spec["obs_perm"],
            obs_sign=mirror_spec["obs_sign"],
            act_perm=mirror_spec["act_perm"],
            act_sign=mirror_spec["act_sign"],
        ).to(device)
        qf2_target = SymmetricSoftQNetwork(
            SoftQNetwork(runner.obs_dim, runner.act_dim).to(device),
            obs_perm=mirror_spec["obs_perm"],
            obs_sign=mirror_spec["obs_sign"],
            act_perm=mirror_spec["act_perm"],
            act_sign=mirror_spec["act_sign"],
        ).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    symmetry_test: dict[str, float] | None = None
    if isinstance(actor, SymmetricSACActor) and isinstance(qf1, SymmetricSoftQNetwork):
        symmetry_test = symmetric_module_self_test(
            actor,
            qf1,
            obs_dim=runner.obs_dim,
            act_dim=runner.act_dim,
            device=device,
        )
        print(json.dumps({"symmetry_test": symmetry_test}, ensure_ascii=False), flush=True)
    actor_optimizer = optim.Adam(actor.parameters(), lr=float(sac_cfg.get("policy_lr", sac_cfg.get("learning_rate", 3e-4))), eps=1e-5)
    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()),
        lr=float(sac_cfg.get("q_lr", sac_cfg.get("learning_rate", 3e-4))),
        eps=1e-5,
    )
    log_alpha = torch.tensor(np.log(float(sac_cfg.get("alpha", 0.2))), dtype=torch.float32, device=device, requires_grad=True)
    alpha_optimizer = optim.Adam([log_alpha], lr=float(sac_cfg.get("alpha_lr", sac_cfg.get("learning_rate", 3e-4))), eps=1e-5)
    if not np.isfinite(target_entropy):
        target_entropy = -float(runner.act_dim)

    replay = ReplayBuffer(
        int(sac_cfg.get("buffer_size", 250000)),
        runner.obs_dim,
        runner.act_dim,
        device,
    )
    global_step = 0
    start_env_step = 1
    resumed_from: str | None = None
    partial_resume = False
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        old_obs_dim = actor_obs_dim_from_state_dict(checkpoint["actor_state_dict"])
        new_obs_dim = int(runner.obs_dim)
        exact_actor = load_shape_compatible_state_dict(actor, checkpoint["actor_state_dict"])
        exact_qf1 = load_shape_compatible_q_state_dict(
            qf1,
            checkpoint["qf1_state_dict"],
            old_obs_dim=old_obs_dim,
            new_obs_dim=new_obs_dim,
            act_dim=runner.act_dim,
        )
        exact_qf2 = load_shape_compatible_q_state_dict(
            qf2,
            checkpoint["qf2_state_dict"],
            old_obs_dim=old_obs_dim,
            new_obs_dim=new_obs_dim,
            act_dim=runner.act_dim,
        )
        exact_qf1_target = load_shape_compatible_q_state_dict(
            qf1_target,
            checkpoint["qf1_target_state_dict"],
            old_obs_dim=old_obs_dim,
            new_obs_dim=new_obs_dim,
            act_dim=runner.act_dim,
        )
        exact_qf2_target = load_shape_compatible_q_state_dict(
            qf2_target,
            checkpoint["qf2_target_state_dict"],
            old_obs_dim=old_obs_dim,
            new_obs_dim=new_obs_dim,
            act_dim=runner.act_dim,
        )
        partial_resume = not all([exact_actor, exact_qf1, exact_qf2, exact_qf1_target, exact_qf2_target])
        if not partial_resume:
            actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
            q_optimizer.load_state_dict(checkpoint["q_optimizer_state_dict"])
            alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer_state_dict"])
        with torch.no_grad():
            log_alpha.copy_(checkpoint["log_alpha"].to(device))
        if "obs_normalizer" in checkpoint:
            obs_normalizer.load_state_dict(checkpoint["obs_normalizer"])
        global_step = int(checkpoint.get("global_step", 0))
        start_env_step = int(checkpoint.get("env_step", 0)) + 1
        resumed_from = str(args.resume)
    run_start_global_step = int(global_step)
    if args.resume is not None:
        if bool(config.get("resume_schedule_from_checkpoint", True)):
            run_start_global_step = int(checkpoint.get("run_start_global_step", checkpoint.get("run_config", {}).get("run_start_global_step", global_step)))
    current_ref_gate = ref_gate_for_step(config, global_step, run_start_global_step)
    config.setdefault("policy", {})["current_ref_gate"] = float(current_ref_gate)
    set_actor_ref_gate(actor, current_ref_gate)
    current_reward_weights = apply_reward_schedule(config, runner, global_step, run_start_global_step)

    run_config = {
        "config": config,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "reference": {
            "path": reference["path"],
            "length": reference["length"],
            "metadata": reference["metadata"],
            "source_index_count": int(len(reference["source_indices"])),
            "activation_prior": reference["activation_prior_metadata"],
            "emg_prior": reference["emg_prior_metadata"],
            "reference_names": reference.get("reference_names", []),
            "reference_offsets": reference.get("reference_offsets", []),
        },
        "obs_dim": runner.obs_dim,
        "act_dim": runner.act_dim,
        "action_mapping": "activation = 0.5 * (clamp(action, -1, 1) + 1)",
        "normalize_observations": obs_normalizer.enabled,
        "target_entropy": target_entropy,
        "symmetric_policy": symmetric_policy,
        "policy_architecture": architecture,
        "gated_ref_obs_spec": gated_spec["metadata"] if gated_spec is not None else None,
        "mirror_spec": mirror_spec["metadata"] if mirror_spec is not None else None,
        "symmetry_test": symmetry_test,
        "resumed_from": resumed_from,
        "partial_resume": partial_resume,
        "run_start_global_step": run_start_global_step,
    }
    write_json(args.outdir / "run_config.json", run_config)

    current_reference_pool_stage = reference_pool_schedule_for_step(config, reference, global_step, run_start_global_step)
    if current_reference_pool_stage is not None:
        runner.set_phase_choices_from_windows(current_reference_pool_stage["phase_windows"])
        runner.reset(torch.ones(nworld, dtype=torch.bool, device=device))
    current_reset_phase_stage = reset_phase_schedule_for_step(config, global_step, run_start_global_step)
    if current_reset_phase_stage is not None:
        apply_reset_phase_stage(runner, current_reset_phase_stage)
        runner.reset(torch.ones(nworld, dtype=torch.bool, device=device))

    current_future_obs_dropout_prob = future_obs_dropout_prob_for_step(config, global_step, run_start_global_step)
    set_future_obs_dropout_prob(config, current_future_obs_dropout_prob)
    if args.render_only_video:
        set_future_obs_dropout_prob(config, 0.0)
        original_video_phase = int(args.video_phase)
        video_rows = []
        for video_phase in configured_video_phases(
            config,
            reference,
            original_video_phase,
            global_step=global_step,
            run_start_global_step=run_start_global_step,
            video_every=int(args.video_every),
        ):
            args.video_phase = int(video_phase)
            video_row = render_policy_video(
                agent=actor,
                obs_normalizer=obs_normalizer,
                config=config,
                reference=reference,
                args=args,
                device=device,
                update=start_env_step,
                global_step=global_step,
            )
            append_csv(args.outdir / "video_metrics.csv", video_row)
            print(json.dumps({"video": video_row}, ensure_ascii=False), flush=True)
            video_rows.append(video_row)
        args.video_phase = original_video_phase
        print(json.dumps({"render_only_video_done": video_rows}, ensure_ascii=False), flush=True)
        return
    next_obs_raw = runner.obs()
    next_done = torch.zeros(nworld, dtype=torch.float32, device=device)
    train_stats: dict[str, float] = {}
    env_step = start_env_step
    active_video_export: subprocess.Popen | None = None
    while global_step < total_timesteps:
        update_start = time.perf_counter()
        current_reward_weights = apply_reward_schedule(config, runner, global_step, run_start_global_step)
        next_reference_pool_stage = reference_pool_schedule_for_step(config, reference, global_step, run_start_global_step)
        if next_reference_pool_stage != current_reference_pool_stage:
            current_reference_pool_stage = next_reference_pool_stage
            if current_reference_pool_stage is not None:
                runner.set_phase_choices_from_windows(current_reference_pool_stage["phase_windows"])
            elif current_reset_phase_stage is None:
                apply_reset_phase_stage(runner, None)
        next_reset_phase_stage = reset_phase_schedule_for_step(config, global_step, run_start_global_step)
        if next_reset_phase_stage != current_reset_phase_stage:
            current_reset_phase_stage = next_reset_phase_stage
            if current_reference_pool_stage is None:
                apply_reset_phase_stage(runner, current_reset_phase_stage)
                runner.reset(torch.ones(nworld, dtype=torch.bool, device=device))
        current_future_obs_dropout_prob = future_obs_dropout_prob_for_step(config, global_step, run_start_global_step)
        set_future_obs_dropout_prob(config, current_future_obs_dropout_prob)
        current_ref_gate = ref_gate_for_step(config, global_step, run_start_global_step)
        config.setdefault("policy", {})["current_ref_gate"] = float(current_ref_gate)
        set_actor_ref_gate(actor, current_ref_gate)
        reference_curriculum = reference_curriculum_for_update(config, max(1, env_step))
        config.setdefault("reference_curriculum", {})["current_phase_lead_steps"] = int(reference_curriculum["phase_lead_steps"])
        config.setdefault("reference_curriculum", {})["current_phase_tolerance_steps"] = int(reference_curriculum["phase_tolerance_steps"])
        config.setdefault("reference_curriculum", {})["current_swing_exaggeration_scale"] = float(
            reference_curriculum["swing_exaggeration_scale"]
        )
        runner.set_reference_curriculum(
            phase_lead_steps=int(reference_curriculum["phase_lead_steps"]),
            phase_tolerance_steps=int(reference_curriculum["phase_tolerance_steps"]),
            swing_exaggeration_scale=float(reference_curriculum["swing_exaggeration_scale"]),
        )
        obs_normalizer.update(next_obs_raw)
        obs = obs_normalizer.normalize(next_obs_raw)
        if global_step < learning_starts:
            action = torch.empty((nworld, runner.act_dim), dtype=torch.float32, device=device).uniform_(-1.0, 1.0)
            logprob = torch.zeros(nworld, dtype=torch.float32, device=device)
        else:
            with torch.no_grad():
                action, logprob = actor.get_action(obs, deterministic=False)
        with torch.no_grad():
            obs_before = next_obs_raw
            next_obs_raw, reward, done, terms = runner.step(action)
        replay.add(obs_before, action, reward, next_obs_raw, done)
        next_done = done.float()
        global_step += nworld

        learned = 0
        if replay.size >= max(batch_size, learning_starts) and env_step % train_freq == 0:
            for _ in range(gradient_steps):
                b_obs_raw, b_action, b_reward, b_next_obs_raw, b_done = replay.sample(batch_size)
                b_obs = obs_normalizer.normalize(b_obs_raw)
                b_next_obs = obs_normalizer.normalize(b_next_obs_raw)
                q_b_obs = mask_ref_obs_for_q(
                    b_obs,
                    gated_spec["ref_indices"] if gated_spec is not None else None,
                    current_ref_gate,
                )
                q_b_next_obs = mask_ref_obs_for_q(
                    b_next_obs,
                    gated_spec["ref_indices"] if gated_spec is not None else None,
                    current_ref_gate,
                )
                with torch.no_grad():
                    next_action, next_logprob = actor.get_action(b_next_obs)
                    target_q = torch.min(qf1_target(q_b_next_obs, next_action), qf2_target(q_b_next_obs, next_action))
                    alpha = log_alpha.exp()
                    next_q_value = b_reward + (1.0 - b_done) * gamma * (target_q - alpha * next_logprob)
                q1 = qf1(q_b_obs, b_action)
                q2 = qf2(q_b_obs, b_action)
                q_loss = F.mse_loss(q1, next_q_value) + F.mse_loss(q2, next_q_value)
                q_optimizer.zero_grad()
                q_loss.backward()
                nn.utils.clip_grad_norm_(list(qf1.parameters()) + list(qf2.parameters()), float(sac_cfg.get("max_grad_norm", 10.0)))
                q_optimizer.step()

                pi, pi_logprob = actor.get_action(b_obs)
                min_q_pi = torch.min(qf1(q_b_obs, pi), qf2(q_b_obs, pi))
                alpha = log_alpha.exp().detach()
                actor_loss = (alpha * pi_logprob - min_q_pi).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), float(sac_cfg.get("max_grad_norm", 10.0)))
                actor_optimizer.step()

                alpha_loss = -(log_alpha * (pi_logprob + target_entropy).detach()).mean()
                alpha_optimizer.zero_grad()
                alpha_loss.backward()
                alpha_optimizer.step()

                soft_update(qf1, qf1_target, tau)
                soft_update(qf2, qf2_target, tau)
                learned += 1
                train_stats = {
                    "q_loss": float(q_loss.detach().item()),
                    "actor_loss": float(actor_loss.detach().item()),
                    "alpha_loss": float(alpha_loss.detach().item()),
                    "alpha": float(log_alpha.exp().detach().item()),
                    "sample_logprob": float(pi_logprob.detach().mean().item()),
                    "q1_mean": float(q1.detach().mean().item()),
                    "q2_mean": float(q2.detach().mean().item()),
                }

        should_log = args.log_every > 0 and (global_step % int(args.log_every) < nworld or global_step >= total_timesteps)
        if should_log:
            row: dict[str, Any] = {
                "global_step": global_step,
                "env_step": env_step,
                "mean_reward": float(reward.mean().item()),
                "replay_size": int(replay.size),
                "learned_gradient_steps": int(learned),
                "seconds_step": time.perf_counter() - update_start,
                "samples_per_sec_step": float(nworld / max(time.perf_counter() - update_start, 1e-9)),
                "done_rate": float(next_done.mean().item()),
                "fall_rate": float(terms["fall_done"].mean().item()),
                "qvel_done_rate": float(terms["qvel_done"].mean().item()),
                "policy_logprob": float(logprob.mean().item()),
                "activation_mean": float(muscle_action_to_activation(action).mean().item()),
                "action_clip_fraction": float((torch.abs(action) > 0.999).float().mean().item()),
                "reference_phase_lead_steps": int(reference_curriculum["phase_lead_steps"]),
                "reference_phase_tolerance_steps": int(reference_curriculum["phase_tolerance_steps"]),
                "reference_swing_exaggeration_scale": float(reference_curriculum["swing_exaggeration_scale"]),
                "reference_pool_stage": "" if current_reference_pool_stage is None else str(current_reference_pool_stage["name"]),
                "reference_pool_count": 0 if current_reference_pool_stage is None else len(current_reference_pool_stage["references"]),
                "reset_phase_stage": "" if current_reset_phase_stage is None else str(current_reset_phase_stage["name"]),
                "reset_phase_choice_count": 0 if runner.phase_choices is None else int(runner.phase_choices.numel()),
                "future_obs_dropout_prob": float(current_future_obs_dropout_prob),
                "ref_gate": float(current_ref_gate),
                **train_stats,
            }
            for key, value in sorted(current_reward_weights.items()):
                row[f"reward_weight_{key}"] = float(value)
            for key in [
                "tracking_qpos_penalty",
                "tracking_qvel_penalty",
                "tracking_foot_site_penalty",
                "tracking_swing_foot_site_penalty",
                "tracking_swing_hip_penalty",
                "tracking_swing_limb_penalty",
                "tracking_activation_symmetry_penalty",
                "tracking_future_foot_site_penalty",
                "terminal_swing_landing_penalty",
                "tracking_pelvis_penalty",
                "pelvis_tx_vel_ref",
                "pelvis_ty_vel_ref",
                "pelvis_tangent_vel_ref",
                "pelvis_normal_vel_ref",
                "foot_slip",
                "foot_tangent_delta_abs",
                "foot_normal_delta_abs",
                "tracking_energy_penalty",
                "activation_smooth",
                "upright",
                "height",
                "pelvis_height_above_terrain",
                "alive",
                "fall",
                "batch_swing_side_balance_penalty",
                "swing_hip_abs_err",
                "swing_foot_forward_delta",
                "reference_valid",
            ]:
                if key in terms:
                    row[f"reward_mean_{key}"] = float(terms[key].mean().item())
            append_csv(args.outdir / "train_metrics.csv", row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

        if args.eval_every > 0 and (global_step % int(args.eval_every) < nworld or global_step >= total_timesteps):
            set_future_obs_dropout_prob(config, 0.0)
            eval_row = evaluate(
                agent=actor,
                obs_normalizer=obs_normalizer,
                model=model,
                data=data,
                config=config,
                reference=reference,
                args=args,
                device=device,
                update=env_step,
                global_step=global_step,
            )
            set_future_obs_dropout_prob(config, current_future_obs_dropout_prob)
            append_csv(args.outdir / "eval_metrics.csv", eval_row)
            print(json.dumps({"eval": eval_row}, ensure_ascii=False), flush=True)

        if args.video_every > 0 and (global_step % int(args.video_every) < nworld or global_step >= total_timesteps):
            set_future_obs_dropout_prob(config, 0.0)
            original_video_phase = int(args.video_phase)
            for video_phase in configured_video_phases(
                config,
                reference,
                original_video_phase,
                global_step=global_step,
                run_start_global_step=run_start_global_step,
                video_every=int(args.video_every),
            ):
                args.video_phase = int(video_phase)
                video_row = render_policy_video(
                    agent=actor,
                    obs_normalizer=obs_normalizer,
                    config=config,
                    reference=reference,
                    args=args,
                    device=device,
                    update=env_step,
                    global_step=global_step,
                )
                append_csv(args.outdir / "video_metrics.csv", video_row)
                print(json.dumps({"video": video_row}, ensure_ascii=False), flush=True)
            args.video_phase = original_video_phase
            set_future_obs_dropout_prob(config, current_future_obs_dropout_prob)

        if args.checkpoint_every > 0 and (global_step % int(args.checkpoint_every) < nworld or global_step >= total_timesteps):
            checkpoint_path = args.outdir / f"agent_step_{global_step}.pt"
            payload = {
                "global_step": global_step,
                "env_step": env_step,
                "run_start_global_step": run_start_global_step,
                "run_config": run_config,
                "obs_normalizer": obs_normalizer.state_dict(),
            }
            save_checkpoint(
                checkpoint_path,
                actor=actor,
                qf1=qf1,
                qf2=qf2,
                qf1_target=qf1_target,
                qf2_target=qf2_target,
                actor_optimizer=actor_optimizer,
                q_optimizer=q_optimizer,
                alpha_optimizer=alpha_optimizer,
                log_alpha=log_alpha,
                payload=payload,
            )
            save_checkpoint(
                args.outdir / "latest.pt",
                actor=actor,
                qf1=qf1,
                qf2=qf2,
                qf1_target=qf1_target,
                qf2_target=qf2_target,
                actor_optimizer=actor_optimizer,
                q_optimizer=q_optimizer,
                alpha_optimizer=alpha_optimizer,
                log_alpha=log_alpha,
                payload=payload,
            )
            active_video_export, export_row = maybe_launch_checkpoint_video_export(
                config=config,
                args=args,
                checkpoint_path=checkpoint_path,
                global_step=global_step,
                nworld=nworld,
                active_process=active_video_export,
            )
            append_csv(args.outdir / "video_export_metrics.csv", export_row)
            print(json.dumps({"video_export": export_row}, ensure_ascii=False), flush=True)
        env_step += 1


if __name__ == "__main__":
    main()
