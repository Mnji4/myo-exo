"""Canonical 319-dimensional policy observation construction."""
from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
import torch

from myo_exo_train.env.model import (
    FOOT_SITE_NAMES, RESET_JOINTS, TRACK_JOINTS, course_height_np, semantic_qpos_index,
    site_forward_coord_tensor, site_lateral_coord_tensor, stair_box_treads,
    terrain_forward_axis, terrain_forward_site_dim,
)
from myo_exo_train.env.reference import current_reference_curriculum

def reference_q_dq_tensor(
    reference: dict[str, Any],
    phases: torch.Tensor,
    *,
    swing_exaggeration_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    ref_q = reference["q_ref"][phases].clone()
    ref_dq = reference["dq_ref"][phases].clone()
    scale = float(swing_exaggeration_scale)
    if scale <= 1.0:
        return ref_q, ref_dq
    contact = reference["foot_contact_ref"][phases]
    q_mean = reference["q_ref_mean"]
    dq_mean = reference["dq_ref_mean"]
    for mask, cols in (
        (
            ~(contact[:, 0] | contact[:, 1]),
            torch.tensor([TRACK_JOINTS.index("hip_flexion_r"), TRACK_JOINTS.index("knee_angle_r")], dtype=torch.long, device=phases.device),
        ),
        (
            ~(contact[:, 2] | contact[:, 3]),
            torch.tensor([TRACK_JOINTS.index("hip_flexion_l"), TRACK_JOINTS.index("knee_angle_l")], dtype=torch.long, device=phases.device),
        ),
    ):
        if bool(mask.any().item()):
            rows = torch.nonzero(mask, as_tuple=False).flatten()
            ref_q[rows[:, None], cols[None, :]] = q_mean[cols] + scale * (ref_q[rows[:, None], cols[None, :]] - q_mean[cols])
            ref_dq[rows[:, None], cols[None, :]] = dq_mean[cols] + scale * (ref_dq[rows[:, None], cols[None, :]] - dq_mean[cols])
    return ref_q, ref_dq

def reference_foot_tensor(
    reference: dict[str, Any],
    phases: torch.Tensor,
    *,
    swing_exaggeration_scale: float,
) -> torch.Tensor:
    ref_foot = reference["foot_site_ref"][phases].clone()
    scale = float(swing_exaggeration_scale)
    if scale <= 1.0:
        return ref_foot
    contact = reference["foot_contact_ref"][phases]
    min_z = reference["foot_site_min_z"].unsqueeze(0)
    exaggerated_z = min_z + scale * (ref_foot[:, :, 2] - min_z)
    ref_foot[:, :, 2] = torch.where(~contact, exaggerated_z, ref_foot[:, :, 2])
    return ref_foot

def terrain_preview_dim(config: dict[str, Any]) -> int:
    cfg = config.get("terrain_context", {})
    if not bool(cfg.get("include_height_preview", False)):
        return 0
    return max(0, int(cfg.get("num_preview_samples", 0) or 0))

def policy_task_context_dim(config: dict[str, Any]) -> int:
    obs_cfg = config.get("observation", {})
    if not bool(obs_cfg.get("include_task_context", False)):
        return 0
    features = obs_cfg.get("task_context_features")
    if isinstance(features, list):
        return len(features)
    dim = 0
    if bool(obs_cfg.get("include_target_velocity_command", True)):
        dim += 1
    if bool(obs_cfg.get("include_current_terrain_slope", True)):
        dim += 1
    if bool(obs_cfg.get("include_preview_slope_summary", True)):
        dim += 2
    return dim

def policy_task_context_features(config: dict[str, Any]) -> list[str]:
    obs_cfg = config.get("observation", {})
    if not bool(obs_cfg.get("include_task_context", False)):
        return []
    configured = obs_cfg.get("task_context_features")
    if isinstance(configured, list):
        return [str(name) for name in configured]
    features: list[str] = []
    if bool(obs_cfg.get("include_target_velocity_command", True)):
        features.append("target_velocity")
    if bool(obs_cfg.get("include_current_terrain_slope", True)):
        features.append("current_terrain_slope")
    if bool(obs_cfg.get("include_preview_slope_summary", True)):
        features.extend(["preview_slope_mean", "preview_slope_abs_max"])
    return features

def policy_task_context_mirror_perm(config: dict[str, Any]) -> list[int]:
    features = policy_task_context_features(config)
    lookup = {name: idx for idx, name in enumerate(features)}
    paired = {
        "exo_ctrl_r": "exo_ctrl_l",
        "exo_ctrl_l": "exo_ctrl_r",
        "exo_torque_r": "exo_torque_l",
        "exo_torque_l": "exo_torque_r",
    }
    return [lookup.get(paired.get(name, ""), idx) for idx, name in enumerate(features)]

def foot_obs_feature_dim(config: dict[str, Any]) -> int:
    obs_cfg = config.get("observation", {})
    per_foot = 2
    if bool(obs_cfg.get("include_foot_rel_z", False)):
        per_foot += 1
    if bool(obs_cfg.get("include_foot_ground_slope", False)):
        per_foot += 1
    if bool(obs_cfg.get("include_contact_obs", False)):
        per_foot += 2
    return per_foot * len(FOOT_SITE_NAMES)

def footstep_target_dim(config: dict[str, Any]) -> int:
    obs_cfg = config.get("observation", {})
    cfg = config.get("footstep_target", {})
    if not bool(cfg.get("enabled", obs_cfg.get("include_footstep_target", False))):
        return 0
    return 5 * len(FOOT_SITE_NAMES)

def root_lateral_heading_obs_tensor(
    qpos: torch.Tensor,
    reference: dict[str, Any],
    config: dict[str, Any],
    target_phase: torch.Tensor,
) -> torch.Tensor:
    obs_cfg = config.get("observation", {})
    full_ref_qpos = reference.get("full_reset_qpos")
    root_start = int(obs_cfg.get("root_translation_qpos_start", 0) or 0)
    if full_ref_qpos is None or root_start + 7 > int(qpos.shape[1]):
        return torch.zeros((qpos.shape[0], 2), dtype=torch.float32, device=qpos.device)

    full_ref_q = full_ref_qpos[target_phase].to(device=qpos.device, dtype=qpos.dtype)
    forward_axis = terrain_forward_axis(config)
    lateral_offset = 0 if forward_axis == "y" else 1
    lateral_scale = max(float(obs_cfg.get("root_lateral_obs_scale", 0.20)), 1e-6)
    lateral_clip = float(obs_cfg.get("root_lateral_obs_clip", 5.0))
    lateral = (
        qpos[:, root_start + lateral_offset] - full_ref_q[:, root_start + lateral_offset]
    ) / lateral_scale

    root_quat = torch.nn.functional.normalize(qpos[:, root_start + 3 : root_start + 7], dim=1)
    ref_quat = torch.nn.functional.normalize(full_ref_q[:, root_start + 3 : root_start + 7], dim=1)

    def yaw_from_quat(quat: torch.Tensor) -> torch.Tensor:
        w, x, y, z = quat.unbind(dim=1)
        return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    yaw_delta = yaw_from_quat(root_quat) - yaw_from_quat(ref_quat)
    yaw_delta = torch.atan2(torch.sin(yaw_delta), torch.cos(yaw_delta))
    heading_scale = max(float(obs_cfg.get("root_heading_obs_scale_rad", 0.35)), 1e-6)
    heading_clip = float(obs_cfg.get("root_heading_obs_clip", 5.0))
    heading = yaw_delta / heading_scale
    return torch.stack(
        [
            torch.clamp(lateral, min=-lateral_clip, max=lateral_clip),
            torch.clamp(heading, min=-heading_clip, max=heading_clip),
        ],
        dim=1,
    )

def reference_index(phases: torch.Tensor, reference: dict[str, Any], config: dict[str, Any]) -> torch.Tensor:
    del config
    return phases % int(reference["length"])

def reference_phase_from_x(
    qpos: torch.Tensor,
    phase_idx: torch.Tensor,
    reference: dict[str, Any],
    config: dict[str, Any],
    *,
    phase_lead_steps: int = 0,
    x_align_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    base_phase = reference_index(phase_idx + int(phase_lead_steps), reference, config)
    cfg = config.get("x_aligned_reference", {})
    if not bool(cfg.get("enabled", False)):
        return base_phase
    ref_len = int(reference["length"])
    phase_start = int(cfg.get("phase_start", 0))
    phase_end = int(cfg.get("phase_end", ref_len))
    active = (base_phase >= phase_start) & (base_phase < phase_end)
    if x_align_mask is not None:
        active = active & x_align_mask.to(device=base_phase.device, dtype=torch.bool)
    if not bool(active.any().item()):
        return base_phase
    search_start = max(0, int(cfg.get("search_start", phase_start)))
    search_end = min(ref_len - 1, int(cfg.get("search_end", phase_end - 1)))
    if search_end < search_start:
        return base_phase
    candidates = torch.arange(search_start, search_end + 1, dtype=torch.long, device=qpos.device)
    ref_x = reference["reset_q_ref"][candidates, RESET_JOINTS.index("pelvis_tx")]
    pelvis_x = qpos[:, int(reference["pelvis_tx_qpos"])].unsqueeze(1) + float(cfg.get("x_offset", 0.0))
    distance = torch.abs(ref_x.unsqueeze(0) - pelvis_x)
    max_phase_delta = int(cfg.get("max_phase_delta", 0) or 0)
    if max_phase_delta > 0:
        phase_delta = torch.abs(candidates.unsqueeze(0) - base_phase.unsqueeze(1))
        distance = distance.masked_fill(phase_delta > max_phase_delta, torch.inf)
    nearest = candidates[torch.argmin(distance, dim=1)]
    return torch.where(active, nearest, base_phase)

def reset_reference_phase_from_x(
    qpos: torch.Tensor,
    phase_idx: torch.Tensor,
    reference: dict[str, Any],
    config: dict[str, Any],
) -> torch.Tensor:
    cfg = config.get("reset_reference_alignment", {})
    if not bool(cfg.get("enabled", False)):
        return phase_idx
    mode = str(cfg.get("mode", "x_to_time") or "x_to_time").lower()
    if mode not in {"x_to_time", "x_align_then_time", "x_then_t"}:
        return phase_idx
    ref_len = int(reference["length"])
    base_phase = reference_index(phase_idx, reference, config)
    phase_start = int(cfg.get("phase_start", 0))
    phase_end = int(cfg.get("phase_end", ref_len))
    active = (base_phase >= phase_start) & (base_phase < phase_end)
    if not bool(active.any().item()):
        return phase_idx
    search_start = max(0, int(cfg.get("search_start", phase_start)))
    search_end = min(ref_len - 1, int(cfg.get("search_end", phase_end - 1)))
    if search_end < search_start:
        return phase_idx
    candidates = torch.arange(search_start, search_end + 1, dtype=torch.long, device=qpos.device)
    ref_x = reference["reset_q_ref"][candidates, RESET_JOINTS.index("pelvis_tx")]
    pelvis_x = qpos[:, int(reference["pelvis_tx_qpos"])].unsqueeze(1) + float(cfg.get("x_offset", 0.0))
    nearest = candidates[torch.argmin(torch.abs(ref_x.unsqueeze(0) - pelvis_x), dim=1)]
    return torch.where(active, nearest, phase_idx)

def course_height_tensor(x: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    course_cfg = config.get("terrain_course", {})
    segments = course_cfg.get("segments", [])
    if not isinstance(segments, list) or not segments:
        return torch.zeros_like(x)
    height = torch.zeros_like(x)
    for segment in segments:
        x0 = float(segment.get("x0", -1e9))
        x1 = float(segment.get("x1", 1e9))
        mask = (x >= x0) & (x <= x1)
        kind = str(segment.get("type", "flat"))
        if kind in {"flat", "flat_box"}:
            value = torch.full_like(x, float(segment.get("height", 0.0)))
            height = torch.where(mask, value, height)
        elif kind in {"slope", "ramp_box"}:
            height0 = float(segment.get("height0", 0.0))
            if "height1" in segment and x1 != x0:
                slope = (float(segment["height1"]) - height0) / (x1 - x0)
            else:
                slope = float(segment.get("slope", 0.0))
            value = height0 + slope * (x - x0)
            height = torch.where(mask, value, height)
        elif kind == "stairs":
            step_height = max(float(segment.get("step_height", 0.127)), 1e-6)
            step_depth = max(float(segment.get("step_depth", 0.32)), 1e-6)
            direction = 1.0 if float(segment.get("direction", 1.0)) >= 0.0 else -1.0
            steps = max(1, int(segment.get("steps", 4)))
            base_height = float(segment.get("base_height", 0.0))
            progressed = torch.clamp(x - x0, min=0.0)
            step_index = torch.clamp(torch.floor(progressed / step_depth), min=0.0, max=float(steps))
            value = base_height + direction * step_index * step_height
            height = torch.where(mask, value, height)
        elif kind == "stairs_box":
            for tread_x0, tread_x1, tread_height in stair_box_treads(segment):
                tread_mask = (x >= tread_x0) & (x <= tread_x1)
                height = torch.where(tread_mask, torch.full_like(x, tread_height), height)
    return height

def stair_step_index_tensor(x: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    course_cfg = config.get("terrain_course", {})
    segments = course_cfg.get("segments", [])
    if not isinstance(segments, list) or not segments:
        return torch.zeros_like(x)
    target_label = str(
        config.get("reward_stair_progress", {}).get("segment_label", "") or ""
    )
    step_index = torch.zeros_like(x)
    for segment in segments:
        if str(segment.get("type", "flat")) != "stairs_box":
            continue
        if target_label and str(segment.get("semantic_label", "")) != target_label:
            continue
        direction = 1.0 if float(segment.get("direction", 1.0)) >= 0.0 else -1.0
        x0 = float(segment.get("x0", 0.0))
        step_depth = max(float(segment.get("step_depth", 0.32)), 1e-6)
        steps = max(1, int(segment.get("steps", 1)))
        progressed = x - x0
        raw_index = torch.floor(torch.clamp(progressed, min=0.0) / step_depth) + 1.0
        raw_index = torch.clamp(raw_index, min=0.0, max=float(steps))
        step_index = torch.where(x >= x0, torch.maximum(step_index, raw_index), step_index)
        if direction > 0.0:
            x_top = x0 + float(steps) * step_depth
            platform_depth = max(float(segment.get("platform_depth", 0.0)), 0.0)
            top_platform = (x >= x_top) & (x <= x_top + platform_depth)
            top_height = float(segment.get("top_platform_height", float(steps) * float(segment.get("step_height", 0.127))))
            base_height = float(segment.get("base_height", 0.0))
            step_height = max(float(segment.get("step_height", 0.127)), 1e-6)
            top_index = max(float(steps), (top_height - base_height) / step_height)
            step_index = torch.where(top_platform, torch.full_like(step_index, top_index), step_index)
    return step_index

def stair_tread_progress_tensor(x: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    course_cfg = config.get("terrain_course", {})
    segments = course_cfg.get("segments", [])
    if not isinstance(segments, list) or not segments:
        return torch.zeros_like(x)
    target_label = str(
        config.get("reward_stair_progress", {}).get("segment_label", "") or ""
    )
    progress = torch.zeros_like(x)
    for segment in segments:
        if str(segment.get("type", "flat")) != "stairs_box":
            continue
        if target_label and str(segment.get("semantic_label", "")) != target_label:
            continue
        x0 = float(segment.get("x0", 0.0))
        step_depth = max(float(segment.get("step_depth", 0.32)), 1e-6)
        steps = max(1, int(segment.get("steps", 1)))
        x_top = x0 + float(steps) * step_depth
        on_stairs = (x >= x0) & (x < x_top)
        progressed = torch.clamp(x - x0, min=0.0)
        local = (progressed - torch.floor(progressed / step_depth) * step_depth) / step_depth
        progress = torch.where(on_stairs, torch.clamp(local, min=0.0, max=1.0), progress)
    return progress

def course_slope_tensor(x: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    course_cfg = config.get("terrain_course", {})
    segments = course_cfg.get("segments", [])
    if not isinstance(segments, list) or not segments:
        return torch.zeros_like(x)
    slope = torch.zeros_like(x)
    for segment in segments:
        x0 = float(segment.get("x0", -1e9))
        x1 = float(segment.get("x1", 1e9))
        mask = (x >= x0) & (x <= x1)
        kind = str(segment.get("type", "flat"))
        if kind in {"slope", "ramp_box"}:
            value = float(segment.get("slope", 0.0))
            if "height1" in segment and x1 != x0:
                value = (float(segment["height1"]) - float(segment.get("height0", 0.0))) / (x1 - x0)
            slope = torch.where(mask, torch.full_like(x, value), slope)
    return slope

def terrain_height_preview_tensor(qpos: torch.Tensor, phase_idx: torch.Tensor, reference: dict[str, Any], config: dict[str, Any]) -> torch.Tensor:
    count = terrain_preview_dim(config)
    if count <= 0:
        return torch.empty((qpos.shape[0], 0), dtype=torch.float32, device=qpos.device)
    cfg = config.get("terrain_context", {})
    if bool(cfg.get("zero_height_preview", False)):
        return torch.zeros((qpos.shape[0], count), dtype=torch.float32, device=qpos.device)
    start_m = float(cfg.get("preview_start_m", 0.1))
    end_m = float(cfg.get("preview_end_m", 2.4))
    scale = max(float(cfg.get("height_scale", 0.2)), 1e-6)
    offsets = torch.linspace(start_m, end_m, count, dtype=torch.float32, device=qpos.device).unsqueeze(0)
    x0 = qpos[:, int(reference["pelvis_tx_qpos"])].unsqueeze(1)
    h0 = course_height_tensor(x0, config)
    h = course_height_tensor(x0 + offsets, config)
    return torch.clamp((h - h0) / scale, -5.0, 5.0)

def current_terrain_height_tensor(
    qpos: torch.Tensor,
    phase_idx: torch.Tensor,
    reference: dict[str, Any],
    config: dict[str, Any],
) -> torch.Tensor:
    x = qpos[:, int(reference["pelvis_tx_qpos"])].unsqueeze(1)
    return course_height_tensor(x, config).squeeze(1)

def current_terrain_slope_tensor(
    qpos: torch.Tensor,
    phase_idx: torch.Tensor,
    reference: dict[str, Any],
    config: dict[str, Any],
) -> torch.Tensor:
    x = qpos[:, int(reference["pelvis_tx_qpos"])].unsqueeze(1)
    return course_slope_tensor(x, config).squeeze(1)

def policy_task_context_tensor(
    qpos: torch.Tensor,
    phase_idx: torch.Tensor,
    reference: dict[str, Any],
    config: dict[str, Any],
    non_muscle_ctrl: torch.Tensor | None = None,
    non_muscle_torque: torch.Tensor | None = None,
) -> torch.Tensor:
    dim = policy_task_context_dim(config)
    if dim <= 0:
        return torch.empty((qpos.shape[0], 0), dtype=torch.float32, device=qpos.device)
    obs_cfg = config.get("observation", {})
    features = policy_task_context_features(config)
    parts: list[torch.Tensor] = []
    preview_slopes: torch.Tensor | None = None
    for feature in features:
        if feature == "target_velocity":
            cmd_cfg = config.get("reward_tangent_velocity_command", {})
            target = float(cmd_cfg.get("target", obs_cfg.get("target_velocity_command", 0.0)) or 0.0)
            scale = max(float(obs_cfg.get("target_velocity_command_scale", 1.0)), 1e-6)
            parts.append(torch.full((qpos.shape[0], 1), target / scale, dtype=torch.float32, device=qpos.device))
        elif feature == "current_terrain_slope":
            parts.append(current_terrain_slope_tensor(qpos, phase_idx, reference, config).unsqueeze(1))
        elif feature in {"preview_slope_mean", "preview_slope_abs_max"}:
            if preview_slopes is None:
                count = max(2, int(obs_cfg.get("preview_slope_summary_samples", 8) or 8))
                start_m = float(obs_cfg.get("preview_slope_start_m", 0.0))
                end_m = float(obs_cfg.get("preview_slope_end_m", 1.5))
                offsets = torch.linspace(start_m, end_m, count, dtype=torch.float32, device=qpos.device).unsqueeze(0)
                x0 = qpos[:, int(reference["pelvis_tx_qpos"])].unsqueeze(1)
                preview_slopes = terrain_slope_for_world_x_tensor(x0 + offsets, phase_idx, reference, config)
            value = preview_slopes.mean(dim=1, keepdim=True) if feature == "preview_slope_mean" else preview_slopes.abs().amax(dim=1, keepdim=True)
            parts.append(value)
        elif feature in {"exo_ctrl_r", "exo_ctrl_l"}:
            side = 0 if feature.endswith("_r") else 1
            if non_muscle_ctrl is None or int(non_muscle_ctrl.shape[1]) <= side:
                value = torch.zeros((qpos.shape[0], 1), dtype=torch.float32, device=qpos.device)
            else:
                scale = max(float(obs_cfg.get("exo_ctrl_observation_scale", 1.0)), 1e-6)
                value = non_muscle_ctrl[:, side : side + 1] / scale
            parts.append(value)
        elif feature in {"exo_torque_r", "exo_torque_l"}:
            side = 0 if feature.endswith("_r") else 1
            if non_muscle_torque is None or int(non_muscle_torque.shape[1]) <= side:
                value = torch.zeros((qpos.shape[0], 1), dtype=torch.float32, device=qpos.device)
            else:
                scale = max(float(obs_cfg.get("exo_torque_observation_scale_nm", 10.0)), 1e-6)
                value = non_muscle_torque[:, side : side + 1] / scale
            parts.append(value)
        else:
            raise ValueError(f"unsupported observation.task_context_features entry: {feature}")
    if not parts:
        return torch.empty((qpos.shape[0], 0), dtype=torch.float32, device=qpos.device)
    return torch.cat(parts, dim=1)

def terrain_height_for_world_x_tensor(
    x: torch.Tensor,
    phase_idx: torch.Tensor,
    reference: dict[str, Any],
    config: dict[str, Any],
) -> torch.Tensor:
    return course_height_tensor(x, config)

def footstep_target_tensor(
    qpos: torch.Tensor,
    site_xpos: torch.Tensor,
    phase_idx: torch.Tensor,
    reference: dict[str, Any],
    config: dict[str, Any],
    *,
    pelvis_tx_qpos: int,
    foot_site_indices: torch.Tensor,
    target_phase: torch.Tensor | None = None,
) -> torch.Tensor:
    dim = footstep_target_dim(config)
    if dim <= 0:
        return torch.empty((qpos.shape[0], 0), dtype=torch.float32, device=qpos.device)
    cfg = config.get("footstep_target", {})
    horizon = max(1, int(cfg.get("horizon_steps", 48) or 48))
    if target_phase is None:
        target_phase = reference_phase_from_x(qpos, phase_idx, reference, config)

    nworld = int(qpos.shape[0])
    nfoot = len(FOOT_SITE_NAMES)
    contact0 = reference["foot_contact_ref"][target_phase].bool()
    selected = contact0.clone()
    selected_phase = target_phase.unsqueeze(1).expand(-1, nfoot).clone()
    selected_offset = torch.zeros((nworld, nfoot), dtype=torch.float32, device=qpos.device)
    prev_contact = contact0
    for offset in range(1, horizon + 1):
        phase = reference_index(target_phase + offset, reference, config)
        contact = reference["foot_contact_ref"][phase].bool()
        touchdown = contact & (~prev_contact)
        choose = (~selected) & touchdown
        selected_phase = torch.where(choose, phase.unsqueeze(1).expand(-1, nfoot), selected_phase)
        selected_offset = torch.where(choose, torch.full_like(selected_offset, float(offset)), selected_offset)
        selected = selected | choose
        prev_contact = contact

    fallback_phase = reference_index(target_phase + horizon, reference, config)
    selected_phase = torch.where((~selected), fallback_phase.unsqueeze(1).expand(-1, nfoot), selected_phase)
    selected_offset = torch.where((~selected), torch.full_like(selected_offset, float(horizon)), selected_offset)
    foot_diag = torch.arange(nfoot, dtype=torch.long, device=qpos.device)
    selected_contact_all = reference["foot_contact_ref"][selected_phase.reshape(-1)].reshape(nworld, nfoot, nfoot)
    selected_contact = selected_contact_all[:, foot_diag, foot_diag].float()

    selected_flat = selected_phase.reshape(-1)
    curriculum = current_reference_curriculum(config)
    target_foot_all = reference_foot_tensor(
        reference,
        selected_flat,
        swing_exaggeration_scale=float(curriculum["swing_exaggeration_scale"]),
    ).reshape(nworld, nfoot, nfoot, 3)
    target_foot = target_foot_all[:, foot_diag, foot_diag, :]
    ref_pelvis_forward = reference["pelvis_tx_ref"][selected_flat].reshape(nworld, nfoot)
    target_forward = ref_pelvis_forward + target_foot[:, :, 0]
    pelvis_forward = qpos[:, int(pelvis_tx_qpos)].unsqueeze(1)
    foot = site_xpos[:, foot_site_indices, :]
    foot_forward = site_forward_coord_tensor(foot, config)
    foot_z = foot[:, :, 2]

    current_terrain = current_terrain_height_tensor(qpos, phase_idx, reference, config).unsqueeze(1)
    target_terrain = terrain_height_for_world_x_tensor(target_forward, phase_idx, reference, config)
    path_samples = max(2, int(cfg.get("path_samples", 6) or 6))
    alpha = torch.linspace(0.0, 1.0, path_samples, dtype=torch.float32, device=qpos.device).view(1, 1, -1)
    path_x = foot_forward.unsqueeze(2) + (target_forward - foot_forward).unsqueeze(2) * alpha
    path_terrain = terrain_height_for_world_x_tensor(path_x.reshape(nworld, -1), phase_idx, reference, config).reshape(
        nworld,
        nfoot,
        path_samples,
    )
    max_path_terrain = torch.amax(path_terrain, dim=2)

    forward_scale = max(float(cfg.get("forward_scale", 1.0)), 1e-6)
    height_scale = max(float(cfg.get("height_scale", 0.25)), 1e-6)
    clearance_margin = float(cfg.get("clearance_margin", 0.08))
    clearance_scale = max(float(cfg.get("clearance_scale", 0.20)), 1e-6)
    target_forward_offset = torch.clamp((target_forward - pelvis_forward) / forward_scale, -5.0, 5.0)
    terrain_delta = torch.clamp((target_terrain - current_terrain) / height_scale, -5.0, 5.0)
    time_to_contact = torch.clamp(selected_offset / float(horizon), 0.0, 1.0)
    clearance_required = torch.clamp(
        torch.relu(max_path_terrain + clearance_margin - foot_z) / clearance_scale,
        0.0,
        5.0,
    )
    return torch.cat(
        [
            target_forward_offset,
            terrain_delta,
            time_to_contact,
            clearance_required,
            selected_contact,
        ],
        dim=1,
    )

def terrain_slope_for_world_x_tensor(
    x: torch.Tensor,
    phase_idx: torch.Tensor,
    reference: dict[str, Any],
    config: dict[str, Any],
) -> torch.Tensor:
    return course_slope_tensor(x, config)

def localized_qpos_obs_tensor(
    qpos: torch.Tensor,
    phase_idx: torch.Tensor,
    reference: dict[str, Any],
    config: dict[str, Any],
    *,
    pelvis_tx_qpos: int,
    curriculum: dict[str, Any] | None = None,
) -> torch.Tensor:
    obs_cfg = config.get("observation", {})
    qpos_obs = qpos.clone()
    pelvis_ty_qpos = int(reference["qpos_indices"][TRACK_JOINTS.index("pelvis_ty")].detach().cpu().item())
    terrain_height = current_terrain_height_tensor(qpos, phase_idx, reference, config)

    zero_horizontal = bool(
        obs_cfg.get(
            "zero_root_horizontal_translation",
            obs_cfg.get("zero_root_xy_translation", False),
        )
    )
    if zero_horizontal:
        root_start = int(obs_cfg.get("root_translation_qpos_start", max(0, min(int(pelvis_tx_qpos), pelvis_ty_qpos) - 1)))
        for idx in range(root_start, min(int(qpos.shape[1]), root_start + 3)):
            if idx != pelvis_ty_qpos:
                qpos_obs[:, idx] = 0.0
    else:
        root_x_feature = str(obs_cfg.get("root_x_feature", "zero") or "zero")
        if root_x_feature == "nominal_x_lag":
            if curriculum is None:
                curriculum = current_reference_curriculum(config)
            raw_target_phase = phase_idx + int(curriculum["phase_lead_steps"])
            nominal_phase = reference_index(raw_target_phase, reference, config)
            nominal_ref_x = reference["reset_q_ref"][nominal_phase, RESET_JOINTS.index("pelvis_tx")]
            nominal_x_lag = nominal_ref_x - qpos[:, pelvis_tx_qpos]
            nominal_x_lag_scale = max(float(obs_cfg.get("nominal_x_lag_scale", 1.0)), 1e-6)
            nominal_x_lag_clip = float(obs_cfg.get("nominal_x_lag_clip", 3.0))
            qpos_obs[:, pelvis_tx_qpos] = torch.clamp(
                nominal_x_lag / nominal_x_lag_scale,
                min=-nominal_x_lag_clip,
                max=nominal_x_lag_clip,
            )
        else:
            qpos_obs[:, pelvis_tx_qpos] = 0.0
    qpos_obs[:, pelvis_ty_qpos] = qpos[:, pelvis_ty_qpos] - terrain_height
    return qpos_obs

def current_terrain_height_np(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    reference: dict[str, Any],
    config: dict[str, Any],
    phase: int,
) -> float:
    pelvis_tx_qpos = int(reference.get("pelvis_tx_qpos", semantic_qpos_index(model, "pelvis_tx")))
    x = np.array([float(data.qpos[pelvis_tx_qpos])], dtype=np.float64)
    return float(course_height_np(x, list(config.get("terrain_course", {}).get("segments", [])))[0])

def build_policy_obs_tensor(
    *,
    qpos: torch.Tensor,
    qvel: torch.Tensor,
    act: torch.Tensor,
    site_xpos: torch.Tensor,
    sensordata: torch.Tensor | None = None,
    foot_sensor_indices: torch.Tensor | None = None,
    model_weight: float | None = None,
    phase_idx: torch.Tensor,
    pelvis_tx_qpos: int,
    foot_site_indices: torch.Tensor,
    reference: dict[str, Any],
    config: dict[str, Any],
    non_muscle_ctrl: torch.Tensor | None = None,
    non_muscle_torque: torch.Tensor | None = None,
    x_align_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    curriculum = current_reference_curriculum(config)
    target_phase = reference_phase_from_x(
        qpos,
        phase_idx,
        reference,
        config,
        phase_lead_steps=int(curriculum["phase_lead_steps"]),
        x_align_mask=x_align_mask,
    )
    obs_cfg = config.get("observation", {})
    localize_obs = bool(obs_cfg.get("localize_root", False))
    phase_mode = str(obs_cfg.get("phase_obs", "reference") or "reference")
    zero_reference_obs = bool(obs_cfg.get("zero_reference_obs", config.get("amp", {}).get("zero_reference_obs", False)))
    phase_period_steps = max(1.0, float(obs_cfg.get("phase_period_steps", reference["length"])))
    phase = torch.remainder(phase_idx.float(), phase_period_steps) * (
        2.0 * torch.pi / float(phase_period_steps)
    )
    phase_features = torch.stack([torch.sin(phase), torch.cos(phase)], dim=1)
    if phase_mode in {"none", "zero", "disabled"}:
        phase_features = torch.zeros_like(phase_features)
    elif phase_mode in {"root_lateral_heading", "lateral_heading", "root_drift_heading"}:
        phase_features = root_lateral_heading_obs_tensor(qpos, reference, config, target_phase)
    qpos_obs = qpos
    if localize_obs:
        qpos_obs = localized_qpos_obs_tensor(
            qpos,
            phase_idx,
            reference,
            config,
            pelvis_tx_qpos=pelvis_tx_qpos,
            curriculum=curriculum,
        )
    q = qpos[:, reference["qpos_indices"]]
    dq = qvel[:, reference["qvel_indices"]]
    ref_q, ref_dq = reference_q_dq_tensor(
        reference,
        target_phase,
        swing_exaggeration_scale=float(curriculum["swing_exaggeration_scale"]),
    )
    ref_q_delta = ref_q - q
    ref_dq_delta = ref_dq - dq
    if zero_reference_obs:
        ref_q_delta = torch.zeros_like(ref_q_delta)
        ref_dq_delta = torch.zeros_like(ref_dq_delta)
    foot = site_xpos[:, foot_site_indices, :]
    foot_forward = site_forward_coord_tensor(foot, config)
    foot_rel_x = foot_forward - qpos[:, pelvis_tx_qpos].unsqueeze(1)
    pelvis_ty_qpos = int(reference["qpos_indices"][TRACK_JOINTS.index("pelvis_ty")].detach().cpu().item())
    foot_rel_z = foot[:, :, 2] - qpos[:, pelvis_ty_qpos].unsqueeze(1)
    foot_terrain_height = terrain_height_for_world_x_tensor(foot_forward, phase_idx, reference, config)
    foot_clearance = foot[:, :, 2] - foot_terrain_height
    foot_ground_slope = terrain_slope_for_world_x_tensor(foot_forward, phase_idx, reference, config)
    ref_contact_obs = reference["foot_contact_ref"][target_phase].float()
    if zero_reference_obs:
        ref_contact_obs = torch.zeros_like(ref_contact_obs)
    current_contact_obs = (
        foot_clearance < float(config.get("reference_contact", {}).get("z_threshold", 0.025))
    ).float()
    foot_z_feature = foot_clearance if localize_obs else foot[:, :, 2]
    feature_groups = [foot_rel_x]
    if bool(obs_cfg.get("include_foot_rel_z", False)):
        feature_groups.append(foot_rel_z)
    feature_groups.append(foot_z_feature)
    if bool(obs_cfg.get("include_foot_ground_slope", False)):
        feature_groups.append(foot_ground_slope)
    if bool(obs_cfg.get("include_contact_obs", False)):
        feature_groups.extend([current_contact_obs, ref_contact_obs])
    foot_features = torch.cat(feature_groups, dim=1)
    obs_parts = [
        qpos_obs,
        qvel,
        act,
        ref_q_delta,
        ref_dq_delta,
        phase_features,
        foot_features,
    ]
    future_steps = max(0, int(config.get("imitation", {}).get("reference_future_steps", 0) or 0))
    future_stride = max(
        1.0,
        float(config.get("imitation", {}).get("reference_future_stride_steps", 1.0) or 1.0),
    )
    future_dropout_prob = max(0.0, min(1.0, float(config.get("imitation", {}).get("current_future_obs_dropout_prob", 0.0) or 0.0)))
    future_keep_mask = None
    if future_steps > 0 and future_dropout_prob > 0.0:
        future_keep_mask = (torch.rand((qpos.shape[0], 1), dtype=torch.float32, device=qpos.device) >= future_dropout_prob).float()
    for offset in range(1, future_steps + 1):
        future_offset = max(1, int(round(float(offset) * future_stride)))
        future_phase = reference_index(target_phase + future_offset, reference, config)
        future_q, _future_dq = reference_q_dq_tensor(
            reference,
            future_phase,
            swing_exaggeration_scale=float(curriculum["swing_exaggeration_scale"]),
        )
        future_foot = reference_foot_tensor(
            reference,
            future_phase,
            swing_exaggeration_scale=float(curriculum["swing_exaggeration_scale"]),
        )
        future_foot_z = future_foot[:, :, 2]
        future_pelvis_ty = reference["reset_q_ref"][future_phase, RESET_JOINTS.index("pelvis_ty")]
        future_foot_rel_z = future_foot[:, :, 2] - future_pelvis_ty.unsqueeze(1)
        future_ref_contact = reference["foot_contact_ref"][future_phase].float()
        future_current_contact = future_ref_contact
        future_foot_ground_slope = torch.zeros_like(future_foot[:, :, 0])
        if localize_obs:
            future_pelvis_tx = reference["reset_q_ref"][future_phase, RESET_JOINTS.index("pelvis_tx")]
            future_foot_world_x = future_pelvis_tx.unsqueeze(1) + future_foot[:, :, 0]
            future_foot_ground_slope = terrain_slope_for_world_x_tensor(
                future_foot_world_x,
                future_phase,
                reference,
                config,
            )
            future_foot_z = future_foot[:, :, 2] - terrain_height_for_world_x_tensor(
                future_foot_world_x,
                future_phase,
                reference,
                config,
            )
        future_feature_groups = [future_foot[:, :, 0]]
        if bool(obs_cfg.get("include_foot_rel_z", False)):
            future_feature_groups.append(future_foot_rel_z)
        future_feature_groups.append(future_foot_z)
        if bool(obs_cfg.get("include_foot_ground_slope", False)):
            future_feature_groups.append(future_foot_ground_slope)
        if bool(obs_cfg.get("include_contact_obs", False)):
            future_feature_groups.extend([future_current_contact, future_ref_contact])
        future_foot_features = torch.cat(future_feature_groups, dim=1)
        future_q_delta = future_q - q
        future_foot_delta = future_foot_features - foot_features
        if zero_reference_obs:
            future_q_delta = torch.zeros_like(future_q_delta)
            future_foot_delta = torch.zeros_like(future_foot_delta)
        if future_keep_mask is not None:
            future_q_delta = future_q_delta * future_keep_mask
            future_foot_delta = future_foot_delta * future_keep_mask
        obs_parts.append(future_q_delta)
        obs_parts.append(future_foot_delta)
    terrain_preview = terrain_height_preview_tensor(qpos, phase_idx, reference, config)
    if terrain_preview.shape[1] > 0:
        obs_parts.append(terrain_preview)
    footstep_target = footstep_target_tensor(
        qpos,
        site_xpos,
        phase_idx,
        reference,
        config,
        pelvis_tx_qpos=pelvis_tx_qpos,
        foot_site_indices=foot_site_indices,
        target_phase=target_phase,
    )
    if footstep_target.shape[1] > 0:
        obs_parts.append(footstep_target)
    task_context = policy_task_context_tensor(
        qpos,
        phase_idx,
        reference,
        config,
        non_muscle_ctrl,
        non_muscle_torque,
    )
    if task_context.shape[1] > 0:
        obs_parts.append(task_context)
    return torch.cat(obs_parts, dim=1)

class ObsNormalizer:
    def __init__(self, obs_dim: int, device: torch.device, *, enabled: bool = True, clip: float = 10.0, eps: float = 1e-4):
        self.enabled = enabled
        self.clip = float(clip)
        self.eps = float(eps)
        self.mean = torch.zeros(obs_dim, dtype=torch.float32, device=device)
        self.var = torch.ones(obs_dim, dtype=torch.float32, device=device)
        self.count = torch.tensor(float(eps), dtype=torch.float32, device=device)

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        if not self.enabled:
            return
        batch = x.detach()
        finite_rows = torch.isfinite(batch).all(dim=1)
        if not bool(finite_rows.any().item()):
            return
        batch = batch[finite_rows]
        batch_mean = torch.mean(batch, dim=0)
        batch_var = torch.var(batch, dim=0, unbiased=False)
        batch_count = torch.tensor(float(batch.shape[0]), dtype=torch.float32, device=batch.device)
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        correction = torch.square(delta) * self.count * batch_count / total_count
        new_var = (m_a + m_b + correction) / total_count
        self.mean.copy_(new_mean)
        self.var.copy_(torch.clamp(new_var, min=1e-6))
        self.count.copy_(total_count)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return torch.nan_to_num(x, nan=0.0, posinf=self.clip, neginf=-self.clip)
        normalized = (x - self.mean) / torch.sqrt(self.var + self.eps)
        normalized = torch.nan_to_num(normalized, nan=0.0, posinf=self.clip, neginf=-self.clip)
        return torch.clamp(normalized, -self.clip, self.clip)

    def state_dict(self) -> dict[str, torch.Tensor | bool | float]:
        return {
            "enabled": self.enabled,
            "clip": self.clip,
            "eps": self.eps,
            "mean": self.mean.detach().clone(),
            "var": self.var.detach().clone(),
            "count": self.count.detach().clone(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.enabled = bool(state.get("enabled", self.enabled))
        self.clip = float(state.get("clip", self.clip))
        self.eps = float(state.get("eps", self.eps))
        mean = state["mean"].to(self.mean.device)
        var = state["var"].to(self.var.device)
        n = min(int(mean.numel()), int(self.mean.numel()))
        self.mean.zero_()
        self.var.fill_(1.0)
        self.mean[:n].copy_(mean[:n])
        self.var[:n].copy_(var[:n])
        self.count.copy_(state["count"].to(self.count.device))
