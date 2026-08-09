"""Reference trajectory loading, course alignment, and schedules."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch

from myo_exo_train.env.model import (
    FOOT_SITE_NAMES, RESET_JOINTS, ROOT, TRACK_JOINTS,
    apply_joint_equalities_np, course_height_np, model_foot_site_names, semantic_qpos_index,
    semantic_qvel_index, site_forward_coord_np, site_id, source_terrain_height_np,
    terrain_forward_axis,
)

def scheduled_value(schedule: dict[str, Any] | None, default: float, update: int) -> float:
    if not isinstance(schedule, dict):
        return float(default)
    start = float(schedule.get("start", default))
    final = float(schedule.get("final", start))
    decay_updates = int(schedule.get("decay_updates", 0) or 0)
    if decay_updates <= 0:
        return final
    progress = min(1.0, max(0.0, float(update - 1) / float(decay_updates)))
    return start + progress * (final - start)

def reference_curriculum_for_update(config: dict[str, Any], update: int) -> dict[str, float | int]:
    cfg = config.get("reference_curriculum", {})
    phase_lead = scheduled_value(cfg.get("phase_lead_schedule"), float(cfg.get("phase_lead_steps", 0)), update)
    phase_tolerance = scheduled_value(
        cfg.get("phase_tolerance_schedule"),
        float(cfg.get("phase_tolerance_steps", 0)),
        update,
    )
    swing_exaggeration = scheduled_value(
        cfg.get("swing_exaggeration_schedule"),
        float(cfg.get("swing_exaggeration_scale", 1.0)),
        update,
    )
    return {
        "phase_lead_steps": int(round(phase_lead)),
        "phase_tolerance_steps": max(0, int(round(phase_tolerance))),
        "swing_exaggeration_scale": max(1.0, float(swing_exaggeration)),
    }

def current_reference_curriculum(config: dict[str, Any]) -> dict[str, float | int]:
    cfg = config.get("reference_curriculum", {})
    return {
        "phase_lead_steps": int(cfg.get("current_phase_lead_steps", cfg.get("phase_lead_steps", 0)) or 0),
        "phase_tolerance_steps": int(cfg.get("current_phase_tolerance_steps", cfg.get("phase_tolerance_steps", 0)) or 0),
        "swing_exaggeration_scale": max(
            1.0,
            float(cfg.get("current_swing_exaggeration_scale", cfg.get("swing_exaggeration_scale", 1.0))),
        ),
    }

def load_reference(
    reference_path: Path,
    model: mujoco.MjModel,
    control_hz: float,
    device: torch.device,
    config: dict[str, Any],
) -> dict[str, Any]:
    raw = np.load(reference_path, allow_pickle=True)
    metadata = raw["metadata"].item()
    series = raw["series_data"].item()
    source_hz = float(metadata.get("sample_rate", 500.0))
    length = len(next(iter(series.values())))
    resample_mode = str(config.get("reference_resample", {}).get("mode", "nearest")).lower()
    if resample_mode == "interp":
        resampled_length = int(float(length) * float(control_hz) / float(source_hz))
        sample_x = np.linspace(0.0, float(length - 1), resampled_length)
        original_x = np.linspace(0.0, float(length - 1), length)
        indices = np.arange(resampled_length, dtype=np.int64)

        def sample_series(name: str) -> np.ndarray:
            values = np.asarray(series.get(name, np.zeros(length)), dtype=np.float32)
            return np.interp(sample_x, original_x, values).astype(np.float32)

        def sample_matrix(name: str, width: int) -> np.ndarray | None:
            if name not in series:
                return None
            values = np.asarray(series[name], dtype=np.float32)
            if values.ndim != 2 or values.shape[1] != int(width):
                return None
            cols = [np.interp(sample_x, original_x, values[:, col]).astype(np.float32) for col in range(int(width))]
            return np.stack(cols, axis=1).astype(np.float32)

    else:
        raw_indices = np.round(np.arange(0.0, float(length), source_hz / float(control_hz))).astype(np.int64)
        raw_indices = np.unique(np.clip(raw_indices, 0, length - 1))
        indices = raw_indices

        def sample_series(name: str) -> np.ndarray:
            return np.asarray(series.get(name, np.zeros(length)), dtype=np.float32)[raw_indices]

        def sample_matrix(name: str, width: int) -> np.ndarray | None:
            if name not in series:
                return None
            values = np.asarray(series[name], dtype=np.float32)
            if values.ndim != 2 or values.shape[1] != int(width):
                return None
            return values[raw_indices].astype(np.float32)

    full_reset_qpos_np = sample_matrix("qpos_full", int(model.nq))
    full_reset_qvel_np = sample_matrix("qvel_full", int(model.nv))
    allow_partial_full_state = bool(
        config.get("reference_reset", {}).get("allow_partial_full_state", False)
    )
    if (full_reset_qpos_np is None or full_reset_qvel_np is None) and not allow_partial_full_state:
        raise ValueError("reference must contain qpos_full and qvel_full for complete-state resets")

    def reference_series_for_joint(joint: str, *, kind: str, index: int) -> np.ndarray:
        prefix = "q" if kind == "qpos" else "dq"
        aliases = [f"{prefix}_{joint}"]
        if joint == "pelvis_tx":
            aliases.extend([f"{prefix}_root_y"])
        elif joint == "pelvis_ty":
            aliases.extend([f"{prefix}_root_z"])
        if joint == "pelvis_tilt" and ((kind == "qpos" and full_reset_qpos_np is not None) or (kind == "qvel" and full_reset_qvel_np is not None)):
            # Freejoint models do not have a scalar pelvis_tilt qpos. Use the full-state coordinate
            # as a placeholder; configs should set pelvis_tilt imitation weight to zero for these models.
            return (full_reset_qpos_np if kind == "qpos" else full_reset_qvel_np)[:, int(index)].astype(np.float32)
        for name in aliases:
            if name in series:
                return sample_series(name)
        if kind == "qpos" and full_reset_qpos_np is not None:
            return full_reset_qpos_np[:, int(index)].astype(np.float32)
        if kind == "qvel" and full_reset_qvel_np is not None:
            return full_reset_qvel_np[:, int(index)].astype(np.float32)
        return np.zeros((len(indices),), dtype=np.float32)

    qpos_indices = []
    qvel_indices = []
    q_ref = []
    dq_ref = []
    pose_scales = []
    vel_scales = []
    for joint in TRACK_JOINTS:
        q_index = semantic_qpos_index(model, joint)
        dq_index = semantic_qvel_index(model, joint)
        qpos_indices.append(q_index)
        qvel_indices.append(dq_index)
        q_ref.append(reference_series_for_joint(joint, kind="qpos", index=q_index))
        dq_ref.append(reference_series_for_joint(joint, kind="qvel", index=dq_index))
        pose_scales.append(0.15 if joint.startswith("pelvis") else 0.45)
        vel_scales.append(1.0 if joint.startswith("pelvis") else 4.0)

    reset_qpos_indices = []
    reset_qvel_indices = []
    reset_q_ref = []
    reset_dq_ref = []
    for joint in RESET_JOINTS:
        q_index = semantic_qpos_index(model, joint)
        dq_index = semantic_qvel_index(model, joint)
        reset_qpos_indices.append(q_index)
        reset_qvel_indices.append(dq_index)
        reset_q_ref.append(reference_series_for_joint(joint, kind="qpos", index=q_index))
        reset_dq_ref.append(reference_series_for_joint(joint, kind="qvel", index=dq_index))

    q_ref_np = np.stack(q_ref, axis=1)
    dq_ref_np = np.stack(dq_ref, axis=1)
    reset_q_np = np.stack(reset_q_ref, axis=1)
    reset_dq_np = np.stack(reset_dq_ref, axis=1)
    pelvis_tx_reset_col = RESET_JOINTS.index("pelvis_tx")
    pelvis_tx_ref_np = reset_q_np[:, pelvis_tx_reset_col].copy()
    reset_q_np[:, pelvis_tx_reset_col] = 0.0

    if full_reset_qpos_np is None or full_reset_qvel_np is None:
        full_reset_qpos_np = np.zeros((len(indices), int(model.nq)), dtype=np.float32)
        full_reset_qvel_np = np.zeros((len(indices), int(model.nv)), dtype=np.float32)
        state = mujoco.MjData(model)
        for frame in range(len(indices)):
            mujoco.mj_resetData(model, state)
            state.qpos[reset_qpos_indices] = reset_q_np[frame]
            state.qvel[reset_qvel_indices] = reset_dq_np[frame]
            apply_joint_equalities_np(model, state)
            full_reset_qpos_np[frame] = state.qpos
            full_reset_qvel_np[frame] = state.qvel

    foot_site_names = model_foot_site_names(model, config)
    foot_site_indices = [site_id(model, name) for name in foot_site_names]
    keypoint_cfg = config.get("keypoint_imitation", {})
    keypoint_body_names = [
        str(name)
        for name in keypoint_cfg.get("body_names", [])
        if str(name)
    ]
    keypoint_body_indices = []
    for name in keypoint_body_names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise KeyError(f"Missing keypoint imitation body: {name}")
        keypoint_body_indices.append(int(body_id))
    ref_data = mujoco.MjData(model)
    foot_site_xpos = np.zeros((len(indices), len(foot_site_names), 3), dtype=np.float32)
    keypoint_body_xpos = np.zeros(
        (len(indices), len(keypoint_body_indices), 3),
        dtype=np.float32,
    )
    pelvis_tx_index = semantic_qpos_index(model, "pelvis_tx")
    forward_dim = 1 if terrain_forward_axis(config) == "y" else 0
    for frame in range(len(indices)):
        mujoco.mj_resetData(model, ref_data)
        ref_data.qpos[:] = full_reset_qpos_np[frame]
        ref_data.qvel[:] = full_reset_qvel_np[frame]
        ref_data.qvel[reset_qvel_indices] = reset_dq_np[frame]
        mujoco.mj_forward(model, ref_data)
        foot_frame = ref_data.site_xpos[foot_site_indices].copy()
        foot_frame[:, 0] = site_forward_coord_np(foot_frame, config) - float(ref_data.qpos[pelvis_tx_index])
        foot_site_xpos[frame] = foot_frame
        if keypoint_body_indices:
            keypoint_frame = ref_data.xpos[keypoint_body_indices].copy()
            keypoint_frame[:, forward_dim] -= float(ref_data.qpos[pelvis_tx_index])
            keypoint_body_xpos[frame] = keypoint_frame

    contact_cfg = config.get("reference_contact", {})
    foot_site_world = foot_site_xpos.copy()
    foot_site_world[:, :, 0] += pelvis_tx_ref_np[:, None]
    foot_site_velocity = np.gradient(foot_site_world, 1.0 / float(control_hz), axis=0).astype(np.float32)
    foot_site_speed = np.linalg.norm(foot_site_velocity[:, :, [0, 2]], axis=2).astype(np.float32)
    terrain_height = source_terrain_height_np(metadata, foot_site_world[:, :, 0])
    foot_clearance = foot_site_world[:, :, 2] - terrain_height
    contact_z_threshold = float(contact_cfg.get("z_threshold", 0.025))
    contact_speed_threshold = float(contact_cfg.get("speed_threshold", 0.4))
    foot_contact_ref = (foot_clearance < contact_z_threshold) & (foot_site_speed < contact_speed_threshold)

    return {
        "path": str(reference_path),
        "metadata": metadata,
        "control_hz": float(control_hz),
        "source_indices": indices,
        "length": int(len(indices)),
        "joint_names": list(TRACK_JOINTS),
        "qpos_indices": torch.tensor(qpos_indices, dtype=torch.long, device=device),
        "qvel_indices": torch.tensor(qvel_indices, dtype=torch.long, device=device),
        "q_ref": torch.tensor(q_ref_np, dtype=torch.float32, device=device),
        "dq_ref": torch.tensor(dq_ref_np, dtype=torch.float32, device=device),
        "q_ref_mean": torch.tensor(np.mean(q_ref_np, axis=0), dtype=torch.float32, device=device),
        "dq_ref_mean": torch.tensor(np.mean(dq_ref_np, axis=0), dtype=torch.float32, device=device),
        "reset_qpos_indices": torch.tensor(reset_qpos_indices, dtype=torch.long, device=device),
        "reset_qvel_indices": torch.tensor(reset_qvel_indices, dtype=torch.long, device=device),
        "reset_q_ref": torch.tensor(reset_q_np, dtype=torch.float32, device=device),
        "reset_dq_ref": torch.tensor(reset_dq_np, dtype=torch.float32, device=device),
        "pose_scales": torch.tensor(pose_scales, dtype=torch.float32, device=device),
        "vel_scales": torch.tensor(vel_scales, dtype=torch.float32, device=device),
        "foot_site_names": list(foot_site_names),
        "foot_site_indices": torch.tensor(foot_site_indices, dtype=torch.long, device=device),
        "foot_site_ref": torch.tensor(foot_site_xpos, dtype=torch.float32, device=device),
        "foot_site_min_z": torch.tensor(np.amin(foot_site_xpos[:, :, 2], axis=0), dtype=torch.float32, device=device),
        "foot_contact_ref": torch.tensor(foot_contact_ref, dtype=torch.bool, device=device),
        "foot_speed_ref": torch.tensor(foot_site_speed, dtype=torch.float32, device=device),
        "keypoint_body_names": keypoint_body_names,
        "keypoint_body_indices": torch.tensor(
            keypoint_body_indices,
            dtype=torch.long,
            device=device,
        ),
        "keypoint_body_ref": torch.tensor(
            keypoint_body_xpos,
            dtype=torch.float32,
            device=device,
        ),
        "pelvis_tx_ref": torch.tensor(pelvis_tx_ref_np, dtype=torch.float32, device=device),
        "pelvis_tx_qpos": pelvis_tx_index,
        "pelvis_ty_qpos": semantic_qpos_index(model, "pelvis_ty"),
        "pelvis_tilt_qpos": semantic_qpos_index(model, "pelvis_tilt"),
        "pelvis_tx_qvel": semantic_qvel_index(model, "pelvis_tx"),
        "pelvis_ty_qvel": semantic_qvel_index(model, "pelvis_ty"),
        "full_reset_qpos": torch.tensor(full_reset_qpos_np, dtype=torch.float32, device=device),
        "full_reset_qvel": torch.tensor(full_reset_qvel_np, dtype=torch.float32, device=device),
    }

def parse_terrain_type_and_params(metadata: dict[str, Any]) -> tuple[int, list[float]]:
    terrain_type = str(metadata.get("terrain_type", "flat") or "flat")
    raw = str(metadata.get("terrain_params", "") or "")
    values = [float(item) for item in raw.split()] if raw.strip() else []
    params = [0.0] * 7
    if terrain_type == "slope":
        params[: min(len(values), 3)] = values[:3]
        return 1, params
    if terrain_type == "stairs_box":
        params[: min(len(values), 7)] = values[:7]
        return 2, params
    return 0, params

def smooth_reference_correction_np(values: np.ndarray, window: int, max_step: float) -> np.ndarray:
    out = np.asarray(values, dtype=np.float32).copy()
    if out.size <= 1:
        return out
    window = max(1, int(window))
    if window >= 3:
        if window % 2 == 0:
            window += 1
        radius = window // 2
        padded = np.pad(out, (radius, radius), mode="edge")
        kernel = np.full((window,), 1.0 / float(window), dtype=np.float32)
        out = np.convolve(padded, kernel, mode="valid").astype(np.float32)
    max_step = float(max_step)
    if max_step > 0.0:
        for index in range(1, out.size):
            out[index] = float(np.clip(out[index], out[index - 1] - max_step, out[index - 1] + max_step))
        for index in range(out.size - 2, -1, -1):
            out[index] = float(np.clip(out[index], out[index + 1] - max_step, out[index + 1] + max_step))
    return out

def apply_reference_course_transform(ref: dict[str, Any], config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    course_cfg = config.get("terrain_course", {})
    contact_cfg = config.get("reference_contact", {})
    offset = float(course_cfg.get("reference_x_offset", 0.0))
    local_x = ref["pelvis_tx_ref"].detach().cpu().numpy().astype(np.float64)
    local_height = source_terrain_height_np(ref["metadata"], local_x)
    use_course_x = bool(course_cfg.get("enabled", False))
    if bool(course_cfg.get("enabled", False)):
        world_height = course_height_np(local_x + offset, list(course_cfg.get("segments", [])))
    else:
        world_height = local_height
    delta_np = (world_height - local_height).astype(np.float32)
    offset_tensor = torch.full((ref["length"],), offset, dtype=torch.float32, device=device)

    ref["reset_q_ref"] = ref["reset_q_ref"].clone()
    ref["q_ref"] = ref["q_ref"].clone()
    ref["foot_site_ref"] = ref["foot_site_ref"].clone()
    ref["keypoint_body_ref"] = ref["keypoint_body_ref"].clone()
    full_reset_qpos = ref.get("full_reset_qpos")
    if full_reset_qpos is not None:
        ref["full_reset_qpos"] = full_reset_qpos.clone()
    if use_course_x:
        ref["course_offset"] = offset_tensor
        ref["reset_q_ref"][:, RESET_JOINTS.index("pelvis_tx")] = ref["pelvis_tx_ref"] + offset_tensor
        if ref.get("full_reset_qpos") is not None:
            ref["full_reset_qpos"][:, int(ref["pelvis_tx_qpos"])] = ref["pelvis_tx_ref"] + offset_tensor
    else:
        ref["reset_q_ref"][:, RESET_JOINTS.index("pelvis_tx")] = 0.0
        if ref.get("full_reset_qpos") is not None:
            ref["full_reset_qpos"][:, int(ref["pelvis_tx_qpos"])] = 0.0
        zeros = torch.zeros((ref["length"],), dtype=torch.float32, device=device)
        ref["course_height_delta"] = zeros
        ref["course_vertical_correction"] = zeros
        ref["course_foot_clearance_lift"] = zeros
        ref["course_pelvis_clearance_lift"] = zeros
        return ref

    target_clearance = float(contact_cfg.get("course_clearance_target", 0.0))
    max_vertical_correction = float(contact_cfg.get("max_course_vertical_correction", 0.8))
    foot_ref = ref["foot_site_ref"].detach().cpu().numpy().astype(np.float64)
    foot_ref_shifted = foot_ref.copy()
    foot_ref_shifted[:, :, 2] += delta_np[:, None]
    foot_world_x = foot_ref[:, :, 0] + (local_x + offset)[:, None]
    foot_terrain_z = course_height_np(foot_world_x, list(course_cfg.get("segments", []))) if bool(course_cfg.get("enabled", False)) else np.zeros_like(foot_world_x)
    clearance = foot_ref_shifted[:, :, 2] - foot_terrain_z
    correction_np = np.zeros((ref["length"],), dtype=np.float32)
    for frame in range(int(ref["length"])):
        current = float(np.min(clearance[frame]))
        correction = target_clearance - current
        correction_np[frame] = float(np.clip(correction, -max_vertical_correction, max_vertical_correction))
    total_delta_np = smooth_reference_correction_np(
        (delta_np + correction_np).astype(np.float32),
        int(contact_cfg.get("course_correction_smoothing_window", 1) or 1),
        float(contact_cfg.get("max_course_vertical_correction_step", 0.0) or 0.0),
    )
    min_foot_clearance = max(target_clearance, float(contact_cfg.get("course_min_clearance", 0.0) or 0.0))
    foot_clearance_lift_np = np.zeros((ref["length"],), dtype=np.float32)
    if min_foot_clearance > -1e-6:
        post_clearance = foot_ref[:, :, 2] + total_delta_np[:, None] - foot_terrain_z
        foot_clearance_lift_np = np.maximum(0.0, min_foot_clearance - np.min(post_clearance, axis=1)).astype(np.float32)
        total_delta_np = (total_delta_np + foot_clearance_lift_np).astype(np.float32)
    min_pelvis_clearance = float(contact_cfg.get("min_pelvis_height_above_course", 0.0) or 0.0)
    pelvis_clearance_lift_np = np.zeros((ref["length"],), dtype=np.float32)
    if min_pelvis_clearance > 0.0:
        pelvis_ty_np = ref["reset_q_ref"][:, RESET_JOINTS.index("pelvis_ty")].detach().cpu().numpy().astype(np.float32)
        pelvis_clearance_np = pelvis_ty_np + total_delta_np - world_height.astype(np.float32)
        pelvis_clearance_lift_np = np.maximum(0.0, min_pelvis_clearance - pelvis_clearance_np).astype(np.float32)
        total_delta_np = (total_delta_np + pelvis_clearance_lift_np).astype(np.float32)
    total_delta = torch.tensor(total_delta_np, dtype=torch.float32, device=device)
    ref["course_height_delta"] = total_delta
    ref["course_vertical_correction"] = total_delta - torch.tensor(delta_np, dtype=torch.float32, device=device)
    ref["course_foot_clearance_lift"] = torch.tensor(foot_clearance_lift_np, dtype=torch.float32, device=device)
    ref["course_pelvis_clearance_lift"] = torch.tensor(pelvis_clearance_lift_np, dtype=torch.float32, device=device)
    ref["reset_q_ref"][:, RESET_JOINTS.index("pelvis_ty")] += total_delta
    ref["q_ref"][:, TRACK_JOINTS.index("pelvis_ty")] += total_delta
    if ref.get("full_reset_qpos") is not None:
        ref["full_reset_qpos"][:, int(ref["pelvis_ty_qpos"])] += total_delta
    ref["foot_site_ref"][:, :, 2] += total_delta[:, None]
    ref["keypoint_body_ref"][:, :, 2] += total_delta[:, None]
    ref["foot_site_min_z"] = torch.amin(ref["foot_site_ref"][:, :, 2], dim=0)
    foot_ref_post = ref["foot_site_ref"].detach().cpu().numpy().astype(np.float64)
    pelvis_x_post = ref["reset_q_ref"][:, RESET_JOINTS.index("pelvis_tx")].detach().cpu().numpy().astype(np.float64)
    foot_world_post = foot_ref_post.copy()
    foot_world_post[:, :, 0] += pelvis_x_post[:, None]
    control_hz = float(ref.get("control_hz", config.get("control", {}).get("control_hz", 30.0)) or 30.0)
    foot_velocity_post = np.gradient(foot_world_post, 1.0 / control_hz, axis=0).astype(np.float32)
    foot_speed_post = np.linalg.norm(foot_velocity_post[:, :, [0, 2]], axis=2).astype(np.float32)
    if bool(course_cfg.get("enabled", False)):
        foot_terrain_post = course_height_np(foot_world_post[:, :, 0], list(course_cfg.get("segments", [])))
    else:
        foot_terrain_post = np.zeros_like(foot_world_post[:, :, 0])
    foot_clearance_post = foot_world_post[:, :, 2] - foot_terrain_post
    ref["foot_speed_ref"] = torch.tensor(foot_speed_post, dtype=torch.float32, device=device)
    foot_contact_post = (
        (foot_clearance_post < float(contact_cfg.get("z_threshold", 0.025)))
        & (foot_speed_post < float(contact_cfg.get("speed_threshold", 0.4)))
    )
    if bool(contact_cfg.get("stair_speed_only_enabled", False)):
        speed_threshold = float(
            contact_cfg.get(
                "stair_speed_threshold",
                contact_cfg.get("speed_threshold", 0.4),
            )
        )
        label_ranges = ref["metadata"].get("label_ranges", {})
        labels = contact_cfg.get(
            "stair_speed_only_labels",
            ["stairup", "stairdown"],
        )
        for label in labels:
            bounds = label_ranges.get(str(label), {})
            start = max(0, int(bounds.get("start", 0)))
            end = min(int(ref["length"]), int(bounds.get("end", start)))
            if end <= start:
                continue
            stair_contact = foot_speed_post[start:end] < speed_threshold
            if bool(contact_cfg.get("stair_ensure_support", True)):
                side_split = max(1, stair_contact.shape[1] // 2)
                no_support = ~stair_contact.any(axis=1)
                for local_frame in np.flatnonzero(no_support):
                    site_speed = foot_speed_post[start + local_frame]
                    right_site = int(np.argmin(site_speed[:side_split]))
                    left_site = side_split + int(
                        np.argmin(site_speed[side_split:])
                    )
                    support_site = (
                        right_site
                        if site_speed[right_site] <= site_speed[left_site]
                        else left_site
                    )
                    stair_contact[local_frame, support_site] = True
            foot_contact_post[start:end] = stair_contact
    ref["foot_contact_ref"] = torch.tensor(
        foot_contact_post,
        dtype=torch.bool,
        device=device,
    )
    return ref

def load_reference_from_config(
    reference_path: Path,
    model: mujoco.MjModel,
    control_hz: float,
    device: torch.device,
    config: dict[str, Any],
) -> dict[str, Any]:
    paths = config.get("reference_pool", {}).get("paths", [])
    if isinstance(paths, list) and len(paths) > 1:
        raise ValueError("multiple-reference pools are not supported; use one precomposed course reference")
    path = Path(paths[0]).expanduser() if isinstance(paths, list) and paths else reference_path
    ref = apply_reference_course_transform(load_reference(path, model, control_hz, device, config), config, device)
    terrain_type, terrain_params = parse_terrain_type_and_params(ref["metadata"])
    ref["terrain_type_id"] = torch.full((ref["length"],), terrain_type, dtype=torch.long, device=device)
    ref["terrain_params_tensor"] = torch.tensor([terrain_params] * ref["length"], dtype=torch.float32, device=device)
    ref["reference_id"] = torch.zeros((ref["length"],), dtype=torch.long, device=device)
    label = str(ref["metadata"].get("terrain_id", Path(ref["path"]).stem))
    source_label = str(ref["metadata"].get("source_label", "") or "")
    if source_label and source_label not in label:
        label = f"{label}:{source_label}"
    ref["reference_names"] = [label]
    ref["reference_offsets"] = [{"name": label, "start": 0, "end": int(ref["length"])}]
    return ref

def named_weights(config: dict[str, Any], section: str, names: list[str], default: float = 1.0) -> torch.Tensor:
    values = config.get("imitation", {}).get(section, {})
    return torch.tensor([float(values.get(name, default)) for name in names], dtype=torch.float32)
