"""Policy evaluation, rendering, metrics, and asynchronous video export."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]

from myo_exo_train.env.model import (
    RESET_JOINTS,
    TRACK_JOINTS,
    apply_non_muscle_ctrl_override,
    build_muscle_model,
    freejoint_root_id,
    model_foot_sensor_names,
    muscle_action_mapping_mode,
    policy_action_to_ctrl,
    semantic_qpos_index,
    sensor_adr_or_none,
    terrain_forward_axis,
)
from myo_exo_train.env.observation import (
    ObsNormalizer,
    build_policy_obs_tensor,
    current_terrain_height_np,
    policy_task_context_features,
    reference_index,
    reset_reference_phase_from_x,
)
from myo_exo_train.env.runner import MJWarpMuscleRunner

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

def resolve_root_path(value: str | Path) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else ROOT / path

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

    export_mode = str(export_cfg.get("mode", "")).lower()
    if export_mode == "hard_switch_moe":
        uphill_value = str(export_cfg.get("uphill_checkpoint", ""))
        if uphill_value == "__current_U__":
            uphill_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_U{checkpoint_path.suffix}")
        else:
            uphill_path = resolve_root_path(uphill_value)
        stair_value = str(export_cfg.get("stair_checkpoint", "__current__"))
        if stair_value == "__current_S__":
            stair_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_S{checkpoint_path.suffix}")
        elif stair_value == "__current__":
            stair_path = checkpoint_path
        else:
            stair_path = resolve_root_path(stair_value)
        phases = export_cfg.get("phase_indices", config.get("video", {}).get("phase_indices", [0]))
        if not isinstance(phases, list) or not phases:
            phases = [0]
        commands: list[str] = ["set -euo pipefail", f"echo $$ > {shlex.quote(str(lock_file.resolve()))}"]
        for phase in phases:
            out_path = outdir / f"step_{int(global_step):09d}_hard_switch_moe_phase{int(phase)}.mp4"
            cmd_parts = [
                shlex.quote(sys.executable),
                shlex.quote(str(ROOT / "scripts" / "eval_hard_switch_flatup_stair_flatup_moe.py")),
                "--config",
                shlex.quote(str(args.config.resolve())),
                "--reference",
                shlex.quote(str(args.reference.resolve())),
                "--uphill-checkpoint",
                shlex.quote(str(uphill_path.resolve())),
                "--stair-checkpoint",
                shlex.quote(str(stair_path.resolve())),
                "--out",
                shlex.quote(str(out_path.resolve())),
                "--phase",
                str(int(phase)),
                "--steps",
                str(int(export_cfg.get("video_steps", args.video_steps))),
                "--height",
                str(int(export_cfg.get("video_height", args.video_height))),
                "--width",
                str(int(export_cfg.get("video_width", args.video_width))),
                "--camera-distance",
                str(float(export_cfg.get("video_camera_distance", args.video_camera_distance))),
                "--camera-height",
                str(float(export_cfg.get("video_camera_height", args.video_camera_height))),
                "--camera-azimuth",
                str(float(export_cfg.get("video_camera_azimuth", getattr(args, "video_camera_azimuth", 135.0)))),
                "--camera-elevation",
                str(float(export_cfg.get("video_camera_elevation", getattr(args, "video_camera_elevation", -30.0)))),
                "--device",
                shlex.quote(str(export_cfg.get("device", "cpu"))),
                "--switch-mode",
                shlex.quote(str(export_cfg.get("switch_mode", "x"))),
                "--switch-to-stair-x",
                str(float(export_cfg.get("switch_to_stair_x", 10.849666533048667))),
                "--switch-to-uphill-x",
                str(float(export_cfg.get("switch_to_uphill_x", 19.34247196446434))),
            ]
            if bool(export_cfg.get("overlay_ignore_fall", False)):
                cmd_parts.append("--ignore-fall")
            bank_value = str(export_cfg.get("bank_path", "") or "")
            if bank_value:
                bank_path = resolve_root_path(bank_value)
                cmd_parts.extend(
                    [
                        "--bank",
                        shlex.quote(str(bank_path.resolve())),
                        "--bank-index",
                        str(int(export_cfg.get("bank_index", 0))),
                    ]
                )
            commands.append(" ".join(cmd_parts))
        commands.append(f"rm -f {shlex.quote(str(lock_file.resolve()))}")
        command = ["bash", "-lc", "\n".join(commands)]
    else:
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
            "--video-camera-azimuth",
            str(float(export_cfg.get("video_camera_azimuth", getattr(args, "video_camera_azimuth", 135.0)))),
            "--video-camera-elevation",
            str(float(export_cfg.get("video_camera_elevation", getattr(args, "video_camera_elevation", -30.0)))),
            "--lock-file",
            str(lock_file.resolve()),
        ]
        for phase in phases if isinstance(phases, list) else []:
            command.extend(["--phase", str(int(phase))])

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

def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)

def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

@torch.no_grad()
def evaluate(
    *,
    agent: Agent,
    obs_normalizer: ObsNormalizer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    config: dict[str, Any],
    reference: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    update: int,
    global_step: int,
) -> dict[str, Any]:
    runner = MJWarpMuscleRunner(
        model=model,
        data=data,
        config=config,
        reference=reference,
        nworld=int(args.eval_worlds),
        nconmax=int(args.nconmax),
        njmax=int(args.njmax),
        seed=int(args.seed) + 10000 + update,
        device=device,
    )
    obs = runner.obs()
    reward_sum = torch.zeros(args.eval_worlds, dtype=torch.float32, device=device)
    done_count = torch.zeros((), dtype=torch.float32, device=device)
    fall_count = torch.zeros((), dtype=torch.float32, device=device)
    qvel_count = torch.zeros((), dtype=torch.float32, device=device)
    for _ in range(int(args.eval_steps)):
        action, _, _, _ = agent.get_action_and_value(obs_normalizer.normalize(obs), deterministic=True)
        obs, reward, _, terms = runner.step(action)
        reward_sum += reward
        done_count += terms["done_count"].sum()
        fall_count += terms["fall_done"].sum()
        qvel_count += terms["qvel_done"].sum()
    return {
        "global_step": global_step,
        "update": update,
        "eval_mean_reward_sum": float(reward_sum.mean().item()),
        "eval_done_rate_per_step": float((done_count / (args.eval_worlds * args.eval_steps)).item()),
        "eval_fall_rate_per_step": float((fall_count / (args.eval_worlds * args.eval_steps)).item()),
        "eval_qvel_done_rate_per_step": float((qvel_count / (args.eval_worlds * args.eval_steps)).item()),
        "eval_mean_pelvis_tx": float(runner.qpos[:, runner.pelvis_tx_qpos].mean().item()),
        "eval_mean_pelvis_height": float(runner.qpos[:, runner.pelvis_ty_qpos].mean().item()),
    }

def set_cpu_reference_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    reference: dict[str, Any],
    phase: int,
    config: dict[str, Any] | None = None,
) -> None:
    full_reset_qpos = reference.get("full_reset_qpos")
    full_reset_qvel = reference.get("full_reset_qvel")
    if full_reset_qpos is None or full_reset_qvel is None:
        raise ValueError("reference must contain complete qpos/qvel reset states")
    data.qpos[:] = full_reset_qpos[phase].detach().cpu().numpy()
    data.qvel[:] = full_reset_qvel[phase].detach().cpu().numpy()
    reset_dq = reference["reset_dq_ref"][phase].detach().cpu().numpy().copy()
    if (
        config is not None
        and str(config.get("reward_mode", "")).lower() == "myoassist_exact"
        and bool(config.get("myoassist_exact", {}).get("scale_reset_qvel_to_target", False))
    ):
        ref_pelvis_vx = max(float(reset_dq[RESET_JOINTS.index("pelvis_tx")]), 1e-6)
        target_vx = float(config.get("myoassist_exact", {}).get("target_velocity", 1.25))
        reset_dq *= target_vx / ref_pelvis_vx
    data.qvel[reference["reset_qvel_indices"].detach().cpu().numpy()] = reset_dq
    if "course_offset" in reference:
        course_offset = float(reference["course_offset"][phase].detach().cpu().item())
        pelvis_tx_ref = float(reference["pelvis_tx_ref"][phase].detach().cpu().item())
        data.qpos[int(reference.get("pelvis_tx_qpos", semantic_qpos_index(model, "pelvis_tx")))] = course_offset + pelvis_tx_ref
    else:
        data.qpos[int(reference.get("pelvis_tx_qpos", semantic_qpos_index(model, "pelvis_tx")))] = 0.0
    initial_activation = float(config.get("reset", {}).get("initial_activation", 0.05)) if config is not None else 0.05
    data.ctrl[:] = 0.5 * (model.actuator_ctrlrange[:, 0] + model.actuator_ctrlrange[:, 1])
    data.ctrl[: int(model.na)] = initial_activation
    if config is not None and bool(config.get("exo_policy", {}).get("enabled", False)):
        data.ctrl[int(model.na) :] = 0.0
    data.act[:] = initial_activation
    mujoco.mj_forward(model, data)

def cpu_policy_obs(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    reference: dict[str, Any],
    config: dict[str, Any],
    phase: int,
    episode_step: int,
    device: torch.device,
) -> torch.Tensor:
    exo_torque: torch.Tensor | None = None
    if any(name in {"exo_torque_r", "exo_torque_l"} for name in policy_task_context_features(config)):
        values: list[float] = []
        for side, joint_name in enumerate(("hip_flexion_r", "hip_flexion_l")):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            dof = int(model.jnt_dofadr[joint_id])
            actuator = int(model.na) + side
            moment = np.asarray(data.actuator_moment)
            if moment.size == int(model.nu) * int(model.nv):
                arm = float(moment.reshape(int(model.nu), int(model.nv))[actuator, dof])
            else:
                row_adr = int(data.moment_rowadr[actuator])
                row_nnz = int(data.moment_rownnz[actuator])
                columns = np.asarray(data.moment_colind)[row_adr : row_adr + row_nnz]
                matches = np.nonzero(columns == dof)[0]
                arm = 0.0 if matches.size == 0 else float(moment[row_adr + int(matches[0])])
            values.append(float(data.actuator_force[actuator]) * arm)
        exo_torque = torch.tensor([values], dtype=torch.float32, device=device)
    return build_policy_obs_tensor(
        qpos=torch.tensor(data.qpos[None, :], dtype=torch.float32, device=device),
        qvel=torch.tensor(data.qvel[None, :], dtype=torch.float32, device=device),
        act=torch.tensor(data.act[None, :], dtype=torch.float32, device=device),
        site_xpos=torch.tensor(data.site_xpos[None, :, :], dtype=torch.float32, device=device),
        sensordata=torch.tensor(data.sensordata[None, :], dtype=torch.float32, device=device) if int(model.nsensordata) > 0 else None,
        foot_sensor_indices=torch.tensor(
            [int(sensor_adr_or_none(model, name)) for name in model_foot_sensor_names(model, config)],
            dtype=torch.long,
            device=device,
        ) if int(model.nsensordata) > 0 and model_foot_sensor_names(model, config) else None,
        model_weight=float(np.sum(model.body_mass) * 9.81),
        phase_idx=torch.tensor([int(phase)], dtype=torch.long, device=device),
        pelvis_tx_qpos=int(reference.get("pelvis_tx_qpos", semantic_qpos_index(model, "pelvis_tx"))),
        foot_site_indices=reference["foot_site_indices"],
        reference=reference,
        config=config,
        non_muscle_ctrl=torch.tensor(
            data.ctrl[None, int(model.na) :], dtype=torch.float32, device=device
        ),
        non_muscle_torque=exo_torque,
    )

def reference_phase_label(reference: dict[str, Any], phase: int) -> str:
    names = reference.get("reference_names")
    ids = reference.get("reference_id")
    if names is None or ids is None:
        return f"phase{int(phase)}"
    ref_id = int(ids[int(phase) % int(reference["length"])].detach().cpu().item())
    name = str(names[ref_id]) if 0 <= ref_id < len(names) else f"ref{ref_id}"
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)
    return f"{safe}_phase{int(phase)}"

@torch.no_grad()
def render_policy_video(
    *,
    agent: Agent,
    obs_normalizer: ObsNormalizer,
    config: dict[str, Any],
    reference: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    update: int,
    global_step: int,
) -> dict[str, Any]:
    model, data = build_muscle_model(config)
    phase = int(args.video_phase) % int(reference["length"])
    set_cpu_reference_state(model, data, reference, phase, config=config)
    frame_skip = int(config["control"]["frame_skip"])
    pelvis_tx_qpos = int(reference.get("pelvis_tx_qpos", semantic_qpos_index(model, "pelvis_tx")))
    pelvis_ty_qpos = int(reference.get("pelvis_ty_qpos", semantic_qpos_index(model, "pelvis_ty")))
    pelvis_tilt_qpos = int(reference.get("pelvis_tilt_qpos", semantic_qpos_index(model, "pelvis_tilt")))
    ctrl_low_np = model.actuator_ctrlrange[:, 0].astype(np.float64)
    ctrl_high_np = model.actuator_ctrlrange[:, 1].astype(np.float64)
    ctrl_low = torch.tensor(ctrl_low_np, dtype=torch.float32, device=device)
    ctrl_high = torch.tensor(ctrl_high_np, dtype=torch.float32, device=device)
    muscle_mapping = muscle_action_mapping_mode(config)

    renderer = mujoco.Renderer(model, height=int(args.video_height), width=int(args.video_width))
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = float(args.video_camera_distance)
    camera.azimuth = float(getattr(args, "video_camera_azimuth", config.get("video_camera_azimuth", 135.0)))
    camera.elevation = float(getattr(args, "video_camera_elevation", config.get("video_camera_elevation", -30.0)))
    camera_forward_axis = terrain_forward_axis(config)
    camera_root_qpos_adr: int | None = None
    root_jid = freejoint_root_id(model)
    if root_jid is not None:
        camera_root_qpos_adr = int(model.jnt_qposadr[root_jid])

    video_reset_cfg = config.get("video_reset", {})
    video_start_label = reference_phase_label(reference, phase)
    if str(video_reset_cfg.get("mode", "") or "").lower() in {"offline_recovery", "bank"}:
        bank_path = str(video_reset_cfg.get("path", "") or config.get("offline_recovery_reset", {}).get("path", "") or "")
        if not bank_path:
            raise ValueError("video_reset.mode=offline_recovery requires video_reset.path or offline_recovery_reset.path")
        bank = Path(bank_path).expanduser()
        if not bank.is_absolute():
            bank = ROOT / bank
        payload = np.load(bank, allow_pickle=True)
        qpos_bank = payload["qpos"]
        qvel_bank = payload["qvel"]
        if qpos_bank.ndim != 2 or qpos_bank.shape[1] != int(model.nq):
            raise ValueError(f"video reset qpos shape {qpos_bank.shape} does not match nq={model.nq}")
        if qvel_bank.ndim != 2 or qvel_bank.shape[1] != int(model.nv):
            raise ValueError(f"video reset qvel shape {qvel_bank.shape} does not match nv={model.nv}")
        row = int(video_reset_cfg.get("row", int(args.video_phase))) % int(qpos_bank.shape[0])
        data.qpos[:] = qpos_bank[row].astype(np.float64)
        data.qvel[:] = qvel_bank[row].astype(np.float64)
        if "act" in payload and int(model.na) > 0:
            act = payload["act"][row].astype(np.float64)
            data.act[:] = act[: int(model.na)]
        if "ctrl" in payload:
            data.ctrl[:] = payload["ctrl"][row].astype(np.float64)
        elif "act" in payload:
            data.ctrl[: int(model.na)] = payload["act"][row].astype(np.float64)[: int(model.na)]
        if "qacc_warmstart" in payload:
            data.qacc_warmstart[:] = payload["qacc_warmstart"][row].astype(np.float64)
        phase = int(payload["phase"][row]) % int(reference["length"])
        if bool(video_reset_cfg.get("align_phase", True)):
            qpos_tensor = torch.tensor(data.qpos[None, :], dtype=torch.float32, device=device)
            phase_tensor = torch.tensor([phase], dtype=torch.long, device=device)
            phase = int(
                reset_reference_phase_from_x(qpos_tensor, phase_tensor, reference, config)[0]
                .detach()
                .cpu()
                .item()
            ) % int(reference["length"])
        data.time = 0.0
        mujoco.mj_forward(model, data)
        video_start_label = f"bankrow{row:04d}_phase{phase}_{reference_phase_label(reference, phase)}"

    frames = []
    rows: list[dict[str, Any]] = []
    fell = False
    try:
        for frame in range(int(args.video_steps)):
            ref_phase = int(
                reference_index(
                    torch.tensor([int(phase)], dtype=torch.long, device=device),
                    reference,
                    config,
                )[0]
                .detach()
                .cpu()
                .item()
            )
            if camera_forward_axis == "y" and camera_root_qpos_adr is not None:
                camera.lookat[:] = [
                    float(data.qpos[camera_root_qpos_adr + 0]),
                    float(data.qpos[camera_root_qpos_adr + 1]),
                    float(args.video_camera_height),
                ]
            else:
                camera.lookat[:] = [float(data.qpos[pelvis_tx_qpos]), 0.0, float(args.video_camera_height)]
            renderer.update_scene(data, camera=camera)
            frames.append(renderer.render())
            obs = cpu_policy_obs(model, data, reference, config, phase, frame, device)
            action, _, _, _ = agent.get_action_and_value(obs_normalizer.normalize(obs), deterministic=True)
            clipped_action = torch.clamp(action[0], -1.0, 1.0)
            action_np = clipped_action.detach().cpu().numpy().astype(np.float64)
            policy_ctrl_tensor = policy_action_to_ctrl(
                clipped_action.unsqueeze(0),
                ctrl_low,
                ctrl_high,
                muscle_count=int(model.na),
                muscle_mapping=muscle_mapping,
            )
            policy_ctrl = (
                apply_non_muscle_ctrl_override(
                    policy_ctrl_tensor,
                    config,
                    muscle_count=int(model.na),
                    ctrl_low=ctrl_low,
                    ctrl_high=ctrl_high,
                )[0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            ctrl = policy_ctrl.copy()
            data.ctrl[:] = ctrl
            for _ in range(frame_skip):
                mujoco.mj_step(model, data)
            logged_phase = int(ref_phase)
            phase = (phase + 1) % int(reference["length"])
            terrain_height = current_terrain_height_np(model, data, reference, config, phase)
            pelvis_height_above_terrain = float(data.qpos[pelvis_ty_qpos]) - terrain_height
            low_height = bool(pelvis_height_above_terrain < float(config["reset"]["safe_pelvis_height"]))
            ref_pelvis_tx = float(reference["reset_q_ref"][ref_phase, RESET_JOINTS.index("pelvis_tx")].detach().cpu().item())
            x_lag = ref_pelvis_tx - float(data.qpos[pelvis_tx_qpos])
            ref_tilt = float(reference["q_ref"][ref_phase, TRACK_JOINTS.index("pelvis_tilt")].detach().cpu().item())
            pelvis_tilt_error = float(data.qpos[pelvis_tilt_qpos] - ref_tilt)
            bad_tilt = bool(
                abs(pelvis_tilt_error)
                > float(config["reset"].get("max_abs_pelvis_tilt_error", config["reset"].get("max_abs_pelvis_tilt", 0.65)))
            )
            fell = low_height or bad_tilt
            rows.append(
                {
                    "video_frame": frame,
                    "phase": logged_phase,
                    "time": float(data.time),
                    "ncon": int(data.ncon),
                    "pelvis_tx": float(data.qpos[pelvis_tx_qpos]),
                    "ref_pelvis_tx": ref_pelvis_tx,
                    "x_lag": x_lag,
                    "pelvis_ty": float(data.qpos[pelvis_ty_qpos]),
                    "terrain_height": terrain_height,
                    "pelvis_height_above_terrain": pelvis_height_above_terrain,
                    "pelvis_tilt": float(data.qpos[pelvis_tilt_qpos]),
                    "ref_pelvis_tilt": ref_tilt,
                    "pelvis_tilt_error": pelvis_tilt_error,
                    "fell": fell,
                    "low_height": low_height,
                    "bad_tilt": bad_tilt,
                    "max_abs_qvel": float(np.max(np.abs(data.qvel))),
                    "mean_ctrl": float(np.mean(ctrl)),
                    "max_ctrl": float(np.max(ctrl)),
                    "mean_policy_ctrl": float(np.mean(policy_ctrl)),
                    "max_policy_ctrl": float(np.max(policy_ctrl)),
                    "mean_normalized_action": float(np.mean(action_np)),
                    "std_normalized_action": float(np.std(action_np)),
                    "max_normalized_action": float(np.max(action_np)),
                    "action_clip_fraction": float(torch.mean((torch.abs(action[0]) > 1.0).float()).item()),
                }
            )
            if fell and not bool(getattr(args, "ignore_fall", False)):
                break
    finally:
        renderer.close()

    video_dir = args.outdir if args.outdir.name == "videos" else args.outdir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    label = video_start_label
    video_path = video_dir / f"policy_update_{update:05d}_step_{global_step:09d}_{label}.mp4"
    diagnostic_path = video_dir / f"policy_update_{update:05d}_step_{global_step:09d}_{label}_diagnostics.csv"
    imageio.mimsave(video_path, frames, fps=30)
    write_csv_rows(diagnostic_path, rows)
    return {
        "global_step": global_step,
        "update": update,
        "video": str(video_path),
        "video_diagnostics": str(diagnostic_path),
        "video_frames": len(frames),
        "video_fell": fell,
        "video_final_pelvis_tx": float(data.qpos[pelvis_tx_qpos]),
        "video_final_pelvis_height": float(data.qpos[pelvis_ty_qpos]),
        "video_max_qvel": max((float(row["max_abs_qvel"]) for row in rows), default=0.0),
    }
