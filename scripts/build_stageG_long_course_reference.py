#!/usr/bin/env python3
"""Build one continuous Stage-G ramp course reference from existing short clips."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from cleanrl.ppo_muscle_mjwarp import (  # noqa: E402
    RESET_JOINTS,
    apply_reference_course_transform,
    build_muscle_model,
    course_height_np,
    load_config,
    load_reference,
)


DEFAULT_SOURCES = {
    "level": Path("/home/lzn/exoskeleton_terrain/data/camargo_reference_selected_contact_aligned/camargo_ab06_levelwalking_walk_selected_contact_aligned_myoassist_3d.npz"),
    "level_to_up": ROOT / "results_old/camargo_ramp6_rawmarker_stitched_preview/npz/stitched_level_to_rampascent_rawmarker.npz",
    "up_to_level": ROOT / "results_old/camargo_ramp6_stitched_preview_blend05/npz/stitched_rampascent_to_level.npz",
    "level_to_down": ROOT / "results_old/camargo_ramp6_stitched_preview_blend05/npz/stitched_level_to_rampdescent.npz",
    "down_to_level": ROOT / "results_old/camargo_ramp6_stitched_preview_blend05/npz/stitched_rampdescent_to_level.npz",
}

GAP_M = 0.04
LEAD_LEVEL_CYCLES = 1.5
TAIL_LEVEL_CYCLES = 3
TRANSITION_BLEND_FRAMES = 15
BASE_LEVEL_TO_UP_START_X = 3.826
BASE_HIGH_LEVEL_END_X = 11.75
BASE_LEVEL_TO_DOWN_START_X = 11.80

POSE_MATCH_KEYS = [
    "q_pelvis_ty",
    "q_pelvis_tilt",
    "q_hip_flexion_r",
    "q_knee_angle_r",
    "q_ankle_angle_r",
    "q_mtp_angle_r",
    "q_hip_flexion_l",
    "q_knee_angle_l",
    "q_ankle_angle_l",
    "q_mtp_angle_l",
]
BLEND_KEYS = POSE_MATCH_KEYS


def shifted_course_segments(segments: list[dict[str, Any]], shift: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for segment in segments:
        item = dict(segment)
        for key in ("x0", "x1", "start_x"):
            if key in item:
                item[key] = float(item[key]) + float(shift)
        out.append(item)
    return out


def downhill_start_x(segments: list[dict[str, Any]]) -> float:
    for segment in segments:
        if segment.get("type") == "slope" and float(segment.get("slope", 0.0)) < 0.0:
            return float(segment["x0"])
    raise ValueError("terrain_course has no downhill slope segment")


def downhill_end_x(segments: list[dict[str, Any]]) -> float:
    for segment in segments:
        if segment.get("type") == "slope" and float(segment.get("slope", 0.0)) < 0.0:
            return float(segment["x1"])
    raise ValueError("terrain_course has no downhill slope segment")


def load_relative_q_series(
    path: Path,
    *,
    model: Any,
    control_hz: float,
    config: dict[str, Any],
    course_segments: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    ref = load_reference(path, model, control_hz, torch.device("cpu"), config)
    ref = apply_reference_course_transform(ref, config, torch.device("cpu"))
    reset_q = ref["reset_q_ref"].detach().cpu().numpy().astype(np.float64)
    reset_dq = ref["reset_dq_ref"].detach().cpu().numpy().astype(np.float64)
    out = {f"q_{joint}": reset_q[:, idx].copy() for idx, joint in enumerate(RESET_JOINTS)}
    for idx, joint in enumerate(RESET_JOINTS):
        out[f"dq_{joint}"] = reset_dq[:, idx].copy()
    x = np.asarray(out["q_pelvis_tx"], dtype=np.float64)
    out["q_pelvis_ty"] = np.asarray(out["q_pelvis_ty"], dtype=np.float64) - course_height_np(x, course_segments)
    foot = ref["foot_site_ref"].detach().cpu().numpy().astype(np.float64)
    foot_world_x = foot[:, :, 0] + x[:, None]
    out["foot_rel_x"] = foot[:, :, 0].copy()
    out["foot_rel_z"] = foot[:, :, 2] - course_height_np(foot_world_x, course_segments)
    out["foot_contact"] = ref["foot_contact_ref"].detach().cpu().numpy().astype(bool)
    return dict(ref["metadata"]), out


def add_derivatives(q_series: dict[str, np.ndarray], sample_rate: float, labels: list[str]) -> dict[str, np.ndarray]:
    out = dict(q_series)
    length = len(next(iter(q_series.values())))
    label_arr = np.asarray(labels, dtype=object)
    for key, values in q_series.items():
        if key.startswith("q_"):
            derivative = np.zeros((length,), dtype=np.float64)
            start = 0
            while start < length:
                end = start + 1
                while end < length and label_arr[end] == label_arr[start]:
                    end += 1
                if end - start >= 2:
                    time = np.arange(end - start, dtype=np.float64) / float(sample_rate)
                    derivative[start:end] = np.gradient(values[start:end], time)
                start = end
            out[f"dq_{key[2:]}"] = derivative
    return out


def smoothstep(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def side_contact(contact: np.ndarray) -> np.ndarray:
    contact = np.asarray(contact, dtype=bool)
    return np.array([bool(contact[0] or contact[1]), bool(contact[2] or contact[3])], dtype=bool)


def foot_forward_delta(foot_rel_x: np.ndarray) -> float:
    foot_rel_x = np.asarray(foot_rel_x, dtype=np.float64)
    return float(np.mean(foot_rel_x[0:2]) - np.mean(foot_rel_x[2:4]))


def clip_feature_cost(a: dict[str, np.ndarray], a_idx: int, b: dict[str, np.ndarray], b_idx: int) -> float:
    pose_cost = 0.0
    vel_cost = 0.0
    for key in POSE_MATCH_KEYS:
        scale = 0.18 if key.startswith("q_pelvis") else 0.45
        pose_cost += ((float(a[key][a_idx]) - float(b[key][b_idx])) / scale) ** 2
        dq_key = f"dq_{key[2:]}"
        vel_scale = 1.0 if key.startswith("q_pelvis") else 4.0
        vel_cost += ((float(a[dq_key][a_idx]) - float(b[dq_key][b_idx])) / vel_scale) ** 2
    pose_cost /= float(len(POSE_MATCH_KEYS))
    vel_cost /= float(len(POSE_MATCH_KEYS))
    foot_cost = float(
        np.mean(np.square((a["foot_rel_x"][a_idx] - b["foot_rel_x"][b_idx]) / 0.35))
        + np.mean(np.square((a["foot_rel_z"][a_idx] - b["foot_rel_z"][b_idx]) / 0.18))
    )
    contact_cost = float(np.mean(np.not_equal(side_contact(a["foot_contact"][a_idx]), side_contact(b["foot_contact"][b_idx])).astype(np.float64)))
    a_forward = foot_forward_delta(a["foot_rel_x"][a_idx])
    b_forward = foot_forward_delta(b["foot_rel_x"][b_idx])
    phase_mismatch = abs(a_forward) > 0.04 and abs(b_forward) > 0.04 and np.sign(a_forward) != np.sign(b_forward)
    return pose_cost + 0.35 * vel_cost + 1.25 * foot_cost + 12.0 * contact_cost + 1000.0 * float(phase_mismatch)


def previous_to_clip_cost(
    buffers: dict[str, list[float]],
    aux_buffers: dict[str, list[np.ndarray]],
    rel: dict[str, np.ndarray],
    idx: int,
    course_segments: list[dict[str, Any]],
    sample_rate: float,
) -> float:
    prev_pose = previous_relative_pose(buffers, course_segments)
    prev_vel = previous_relative_velocity(buffers, sample_rate, course_segments)
    prev_foot_x = np.asarray(aux_buffers["foot_rel_x"][-1], dtype=np.float64)
    prev_foot_z = np.asarray(aux_buffers["foot_rel_z"][-1], dtype=np.float64)
    prev_contact = side_contact(np.asarray(aux_buffers["foot_contact"][-1], dtype=bool))
    prev_forward = foot_forward_delta(prev_foot_x)
    pose_cost = 0.0
    vel_cost = 0.0
    length = len(np.asarray(rel["q_pelvis_tx"]))
    for key in POSE_MATCH_KEYS:
        scale = 0.18 if key.startswith("q_pelvis") else 0.45
        pose_cost += ((prev_pose[key] - float(rel[key][idx])) / scale) ** 2
        dq_key = f"dq_{key[2:]}"
        vel_scale = 1.0 if key.startswith("q_pelvis") else 4.0
        vel_cost += ((prev_vel[dq_key] - float(rel.get(dq_key, np.zeros(length))[idx])) / vel_scale) ** 2
    pose_cost /= float(len(POSE_MATCH_KEYS))
    vel_cost /= float(len(POSE_MATCH_KEYS))
    foot_x = np.asarray(rel["foot_rel_x"][idx], dtype=np.float64)
    foot_z = np.asarray(rel["foot_rel_z"][idx], dtype=np.float64)
    foot_cost = float(np.mean(np.square((prev_foot_x - foot_x) / 0.35)) + np.mean(np.square((prev_foot_z - foot_z) / 0.18)))
    contact_cost = float(np.mean(np.not_equal(prev_contact, side_contact(rel["foot_contact"][idx])).astype(np.float64)))
    forward = foot_forward_delta(foot_x)
    phase_mismatch = abs(prev_forward) > 0.04 and abs(forward) > 0.04 and np.sign(prev_forward) != np.sign(forward)
    return pose_cost + 0.35 * vel_cost + 1.25 * foot_cost + 100.0 * contact_cost + 1000.0 * float(phase_mismatch)


def choose_cyclic_level_segment(
    buffers: dict[str, list[float]],
    aux_buffers: dict[str, list[np.ndarray]],
    level: dict[str, np.ndarray],
    next_clip: dict[str, np.ndarray],
    *,
    target_frames: int,
    min_frames: int,
    max_frames: int,
    course_segments: list[dict[str, Any]],
    sample_rate: float,
) -> tuple[int, int]:
    length = len(np.asarray(level["q_pelvis_tx"]))
    low = max(2, int(min_frames))
    high = max(low, min(int(max_frames), length))
    best = (0, int(target_frames))
    best_cost = float("inf")
    for start_idx in range(length):
        start_cost = previous_to_clip_cost(
            buffers,
            aux_buffers,
            level,
            start_idx,
            course_segments,
            sample_rate,
        )
        for frames in range(low, high + 1):
            end_idx = (start_idx + frames - 1) % length
            end_cost = min(
                clip_feature_cost(level, end_idx, next_clip, next_idx)
                for next_idx in range(0, min(4, len(np.asarray(next_clip["q_pelvis_tx"]))))
            )
            cost = start_cost + end_cost + 0.03 * abs(float(frames) - float(target_frames))
            if cost < best_cost:
                best_cost = float(cost)
                best = (int(start_idx), int(frames))
    return best


def choose_high_to_down_join(
    buffers: dict[str, list[float]],
    aux_buffers: dict[str, list[np.ndarray]],
    level: dict[str, np.ndarray],
    level_to_down: dict[str, np.ndarray],
    *,
    high_start_x: float,
    target_downhill_start_x: float,
    base_downhill_start_x: float,
    max_level_frames: int,
    min_level_frames: int,
    max_level_to_down_start: int,
    min_level_to_down_frames: int,
    level_to_down_max_x: float,
    course_segments: list[dict[str, Any]],
    sample_rate: float,
) -> tuple[int, int, int, float]:
    level_length = len(np.asarray(level["q_pelvis_tx"]))
    ltd_x = np.asarray(level_to_down["q_pelvis_tx"], dtype=np.float64)
    best: tuple[float, int, int, int, float] | None = None
    for level_start in range(level_length):
        start_cost = previous_to_clip_cost(
            buffers,
            aux_buffers,
            level,
            level_start,
            course_segments,
            sample_rate,
        )
        for ltd_start in range(0, min(int(max_level_to_down_start), len(ltd_x) - 1) + 1):
            ltd_target_start_x = float(target_downhill_start_x) + (float(ltd_x[ltd_start]) - float(base_downhill_start_x))
            if ltd_target_start_x <= float(high_start_x) + GAP_M:
                continue
            local_rel = ltd_x[ltd_start:] - float(ltd_x[ltd_start])
            ltd_frames = int(np.count_nonzero(ltd_target_start_x + local_rel <= float(level_to_down_max_x)))
            if ltd_frames < int(min_level_to_down_frames):
                continue
            level_frames = max_cyclic_frames_before_x(
                level,
                start_idx=level_start,
                target_start_x=high_start_x,
                max_x=ltd_target_start_x - GAP_M,
                max_frames=int(max_level_frames),
            )
            if level_frames < int(min_level_frames):
                continue
            level_end = (int(level_start) + int(level_frames) - 1) % level_length
            end_cost = clip_feature_cost(level, level_end, level_to_down, ltd_start)
            duration_cost = 0.01 * abs(float(level_frames) - float(max_level_frames) * 0.65)
            total = float(start_cost + end_cost + duration_cost)
            if best is None or total < best[0]:
                best = (total, int(level_start), int(level_frames), int(ltd_start), float(ltd_target_start_x))
    if best is None:
        return 0, max(1, int(min_level_frames)), 0, float(target_downhill_start_x)
    return best[1], best[2], best[3], best[4]


def find_level_loop_segment(level: dict[str, np.ndarray], *, min_frames: int = 45) -> dict[str, Any]:
    length = len(np.asarray(level["q_pelvis_tx"]))
    candidates: list[dict[str, Any]] = []
    for start_idx in range(length):
        for frames in range(int(min_frames), length + 1):
            end_idx = (start_idx + frames - 1) % length
            start_contact = side_contact(level["foot_contact"][start_idx])
            end_contact = side_contact(level["foot_contact"][end_idx])
            if np.any(np.not_equal(start_contact, end_contact)):
                continue
            start_forward = foot_forward_delta(level["foot_rel_x"][start_idx])
            end_forward = foot_forward_delta(level["foot_rel_x"][end_idx])
            phase_mismatch = abs(start_forward) > 0.04 and abs(end_forward) > 0.04 and np.sign(start_forward) != np.sign(end_forward)
            if phase_mismatch:
                continue
            cost = clip_feature_cost(level, end_idx, level, start_idx)
            candidates.append(
                {
                    "start": int(start_idx),
                    "frames": int(frames),
                    "end": int(end_idx),
                    "loop_cost": float(cost),
                    "start_foot_forward_delta": float(start_forward),
                    "end_foot_forward_delta": float(end_forward),
                    "start_side_contact": start_contact.astype(int).tolist(),
                    "end_side_contact": end_contact.astype(int).tolist(),
                }
            )
    if not candidates:
        return {"start": 0, "frames": length, "end": length - 1, "loop_cost": float("inf"), "warning": "no_clean_level_loop"}
    return min(candidates, key=lambda item: float(item["loop_cost"]))


def extract_cyclic_segment(rel: dict[str, np.ndarray], *, start_idx: int, frames: int) -> dict[str, np.ndarray]:
    length = len(np.asarray(rel["q_pelvis_tx"]))
    indices = [(int(start_idx) + pos) % length for pos in range(int(frames))]
    out: dict[str, np.ndarray] = {}
    source_x = np.asarray(rel["q_pelvis_tx"], dtype=np.float64)
    local_rel = cyclic_local_rel(source_x, int(start_idx), int(frames))
    for key, value in rel.items():
        arr = np.asarray(value)
        if key == "q_pelvis_tx":
            out[key] = float(source_x[int(start_idx)]) + local_rel
        else:
            out[key] = arr[indices].copy()
    return out


def max_frames_before_x(rel: dict[str, np.ndarray], target_start_x: float, max_x: float) -> int:
    local_x = np.asarray(rel["q_pelvis_tx"], dtype=np.float64)
    local_rel = local_x - float(local_x[0])
    target_x = float(target_start_x) + local_rel
    valid = np.flatnonzero(target_x <= float(max_x))
    return int(valid[-1] + 1) if valid.size else 0


def max_cyclic_frames_before_x(
    rel: dict[str, np.ndarray],
    *,
    start_idx: int,
    target_start_x: float,
    max_x: float,
    max_frames: int,
) -> int:
    local_x = np.asarray(rel["q_pelvis_tx"], dtype=np.float64)
    local_rel = cyclic_local_rel(local_x, int(start_idx), int(max_frames))
    target_x = float(target_start_x) + local_rel
    valid = np.flatnonzero(target_x <= float(max_x))
    return int(valid[-1] + 1) if valid.size else 0


def cyclic_local_rel(local_x: np.ndarray, start_idx: int, frame_count: int) -> np.ndarray:
    local_x = np.asarray(local_x, dtype=np.float64)
    length = len(local_x)
    if length <= 1 or frame_count <= 0:
        return np.zeros((max(0, frame_count),), dtype=np.float64)
    stride = float(local_x[-1] - local_x[0])
    out = np.zeros((frame_count,), dtype=np.float64)
    start_cycle = 0
    start_x = float(local_x[start_idx])
    for pos in range(frame_count):
        raw = int(start_idx) + int(pos)
        cycle = raw // length
        idx = raw % length
        out[pos] = float(local_x[idx]) + float(cycle) * stride - start_x - float(start_cycle) * stride
    return out


def previous_relative_pose(
    buffers: dict[str, list[float]],
    course_segments: list[dict[str, Any]],
) -> dict[str, float]:
    prev_x = float(buffers["q_pelvis_tx"][-1])
    terrain_h = float(course_height_np(np.array([prev_x], dtype=np.float64), course_segments)[0])
    pose = {key: float(buffers[key][-1]) for key in POSE_MATCH_KEYS}
    pose["q_pelvis_ty"] = float(buffers["q_pelvis_ty"][-1]) - terrain_h
    return pose


def previous_relative_velocity(
    buffers: dict[str, list[float]],
    sample_rate: float,
    course_segments: list[dict[str, Any]],
) -> dict[str, float]:
    if len(buffers["q_pelvis_tx"]) < 2:
        return {f"dq_{key[2:]}": 0.0 for key in POSE_MATCH_KEYS}
    prev_x = float(buffers["q_pelvis_tx"][-1])
    old_x = float(buffers["q_pelvis_tx"][-2])
    prev_h = float(course_height_np(np.array([prev_x], dtype=np.float64), course_segments)[0])
    old_h = float(course_height_np(np.array([old_x], dtype=np.float64), course_segments)[0])
    out: dict[str, float] = {}
    for key in POSE_MATCH_KEYS:
        current = float(buffers[key][-1])
        previous = float(buffers[key][-2])
        if key == "q_pelvis_ty":
            current -= prev_h
            previous -= old_h
        out[f"dq_{key[2:]}"] = (current - previous) * float(sample_rate)
    return out


def find_best_join_frame(
    *,
    buffers: dict[str, list[float]],
    aux_buffers: dict[str, list[np.ndarray]],
    rel: dict[str, np.ndarray],
    course_segments: list[dict[str, Any]],
    sample_rate: float,
    blend_frames: int,
    max_frames: int | None,
    search_max_fraction: float,
    cyclic: bool = False,
) -> dict[str, Any]:
    length = len(np.asarray(rel["q_pelvis_tx"]))
    usable_end = length if cyclic or max_frames is None else max(0, length - int(max_frames) + 1)
    first = 0
    search_limit = int(np.floor(max(0.0, min(1.0, float(search_max_fraction))) * float(length - 1)))
    last = max(first, min(length - int(blend_frames) - 1, usable_end - 1, search_limit))
    if last < first:
        first = 0
        last = max(0, min(length - 1, usable_end - 1))

    prev_pose = previous_relative_pose(buffers, course_segments)
    prev_vel = previous_relative_velocity(buffers, sample_rate, course_segments)
    prev_foot_x = np.asarray(aux_buffers["foot_rel_x"][-1], dtype=np.float64)
    prev_foot_z = np.asarray(aux_buffers["foot_rel_z"][-1], dtype=np.float64)
    prev_contact = np.asarray(aux_buffers["foot_contact"][-1], dtype=bool)
    prev_side_contact = side_contact(prev_contact)
    prev_forward = foot_forward_delta(prev_foot_x)

    candidates: list[dict[str, Any]] = []
    for idx in range(first, last + 1):
        pose_cost = 0.0
        vel_cost = 0.0
        for key in POSE_MATCH_KEYS:
            scale = 0.18 if key.startswith("q_pelvis") else 0.45
            pose_cost += ((prev_pose[key] - float(rel[key][idx])) / scale) ** 2
            dq_key = f"dq_{key[2:]}"
            vel_scale = 1.0 if key.startswith("q_pelvis") else 4.0
            vel_cost += ((prev_vel[dq_key] - float(rel.get(dq_key, np.zeros(length))[idx])) / vel_scale) ** 2
        pose_cost /= float(len(POSE_MATCH_KEYS))
        vel_cost /= float(len(POSE_MATCH_KEYS))

        foot_x = np.asarray(rel["foot_rel_x"][idx], dtype=np.float64)
        foot_z = np.asarray(rel["foot_rel_z"][idx], dtype=np.float64)
        foot_cost = float(np.mean(np.square((prev_foot_x - foot_x) / 0.35)) + np.mean(np.square((prev_foot_z - foot_z) / 0.18)))

        contact = np.asarray(rel["foot_contact"][idx], dtype=bool)
        contact_cost = float(np.mean(np.not_equal(prev_side_contact, side_contact(contact)).astype(np.float64)))
        forward = foot_forward_delta(foot_x)
        phase_mismatch = abs(prev_forward) > 0.04 and abs(forward) > 0.04 and np.sign(prev_forward) != np.sign(forward)
        phase_cost = 1.0 if phase_mismatch else 0.0

        total = pose_cost + 0.35 * vel_cost + 1.25 * foot_cost + 100.0 * contact_cost + 1000.0 * phase_cost
        item = {
            "join_index": int(idx),
            "cost": float(total),
            "pose_cost": float(pose_cost),
            "vel_cost": float(vel_cost),
            "foot_cost": float(foot_cost),
            "contact_cost": float(contact_cost),
            "phase_cost": float(phase_cost),
            "prev_foot_forward_delta": float(prev_forward),
            "new_foot_forward_delta": float(forward),
            "prev_side_contact": prev_side_contact.astype(int).tolist(),
            "new_side_contact": side_contact(contact).astype(int).tolist(),
        }
        candidates.append(item)
    if not candidates:
        return {"join_index": 0, "cost": float("inf"), "warning": "no_join_candidates"}
    same_phase = [item for item in candidates if float(item.get("phase_cost", 0.0)) == 0.0]
    selectable = same_phase if same_phase else candidates
    same_contact = [item for item in selectable if float(item.get("contact_cost", 0.0)) == 0.0]
    if same_contact:
        selectable = same_contact
    best = min(selectable, key=lambda item: float(item["cost"]))
    if float(best.get("phase_cost", 0.0)) > 0.0:
        best["warning"] = "best_candidate_has_left_right_phase_mismatch"
    elif float(best.get("contact_cost", 0.0)) > 0.0:
        best["warning"] = "best_candidate_has_contact_mismatch"
    return best


def apply_join_blend(
    buffers: dict[str, list[float]],
    pending: list[dict[str, float]],
    *,
    course_segments: list[dict[str, Any]],
    blend_frames: int,
) -> None:
    count = min(int(blend_frames), len(pending), len(buffers["q_pelvis_tx"]))
    if count <= 0:
        return
    old_last_x = float(buffers["q_pelvis_tx"][-1])
    new_first_x = float(pending[0]["q_pelvis_tx"])
    old_h = float(course_height_np(np.array([old_last_x], dtype=np.float64), course_segments)[0])
    new_h = float(course_height_np(np.array([new_first_x], dtype=np.float64), course_segments)[0])
    for key in BLEND_KEYS:
        old_last = float(buffers[key][-1])
        new_first = float(pending[0][key])
        if key == "q_pelvis_ty":
            old_last -= old_h
            new_first -= new_h
        delta = old_last - new_first
        for pos in range(count):
            old_weight = -0.5 * delta * smoothstep(float(pos + 1) / float(count))
            old_idx = len(buffers[key]) - count + pos
            if key == "q_pelvis_ty":
                x = float(buffers["q_pelvis_tx"][old_idx])
                terrain_h = float(course_height_np(np.array([x], dtype=np.float64), course_segments)[0])
                old_rel = float(buffers[key][old_idx]) - terrain_h
                buffers[key][old_idx] = old_rel + old_weight + terrain_h
            else:
                buffers[key][old_idx] = float(buffers[key][old_idx]) + old_weight

            t = 0.0 if count == 1 else float(pos) / float(count - 1)
            new_weight = 0.5 * delta * (1.0 - smoothstep(t))
            if key == "q_pelvis_ty":
                x = float(pending[pos]["q_pelvis_tx"])
                terrain_h = float(course_height_np(np.array([x], dtype=np.float64), course_segments)[0])
                new_rel = float(pending[pos][key]) - terrain_h
                pending[pos][key] = new_rel + new_weight + terrain_h
            else:
                pending[pos][key] = float(pending[pos][key]) + new_weight


def append_clip(
    buffers: dict[str, list[float]],
    aux_buffers: dict[str, list[np.ndarray]],
    rel: dict[str, np.ndarray],
    *,
    q_keys: list[str],
    source_name: str,
    target_start_x: float,
    course_segments: list[dict[str, Any]],
    sample_rate: float,
    join_records: list[dict[str, Any]],
    max_x: float | None = None,
    max_frames: int | None = None,
    match_join: bool = True,
    search_max_fraction: float = 0.25,
    cyclic: bool = False,
    forced_join_index: int | None = None,
    min_step: float = 1e-4,
    labels: list[str],
) -> None:
    local_x = np.asarray(rel["q_pelvis_tx"], dtype=np.float64)
    join: dict[str, Any] = {"join_index": 0, "cost": None}
    if forced_join_index is not None:
        join = {"join_index": int(forced_join_index), "cost": None, "forced_join_index": True}
    elif buffers["q_pelvis_tx"] and bool(match_join):
        join = find_best_join_frame(
            buffers=buffers,
            aux_buffers=aux_buffers,
            rel=rel,
            course_segments=course_segments,
            sample_rate=sample_rate,
            blend_frames=int(TRANSITION_BLEND_FRAMES),
            max_frames=max_frames,
            search_max_fraction=search_max_fraction,
            cyclic=cyclic,
        )
    start_idx = int(join.get("join_index", 0))
    frame_limit = int(max_frames) if max_frames is not None else len(local_x)
    if cyclic:
        local_rel = cyclic_local_rel(local_x, start_idx, frame_limit)
        source_indices = [(start_idx + pos) % len(local_x) for pos in range(frame_limit)]
    else:
        local_rel = local_x[start_idx:] - float(local_x[start_idx])
        source_indices = list(range(start_idx, len(local_x)))
    target_x = float(target_start_x) + local_rel
    last_x = buffers["q_pelvis_tx"][-1] if buffers["q_pelvis_tx"] else -np.inf
    pending: list[dict[str, float]] = []
    pending_aux: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for pos, idx in enumerate(source_indices):
        x = float(target_x[pos])
        if x <= last_x + float(min_step):
            continue
        if max_x is not None and x > float(max_x):
            break
        if max_frames is not None and len(pending) >= int(max_frames):
            break
        terrain_h = float(course_height_np(np.array([x], dtype=np.float64), course_segments)[0])
        row: dict[str, float] = {}
        for key in q_keys:
            if key == "q_pelvis_tx":
                value = float(x)
            elif key == "q_pelvis_ty":
                value = float(rel[key][idx]) + terrain_h
            else:
                value = float(rel[key][idx])
            row[key] = value
        pending.append(row)
        pending_aux.append(
            (
                np.asarray(rel["foot_rel_x"][idx], dtype=np.float64).copy(),
                np.asarray(rel["foot_rel_z"][idx], dtype=np.float64).copy(),
                np.asarray(rel["foot_contact"][idx], dtype=bool).copy(),
            )
        )
        last_x = float(x)
    if not pending:
        return
    if buffers["q_pelvis_tx"]:
        boundary = len(labels)
        apply_join_blend(
            buffers,
            pending,
            course_segments=course_segments,
            blend_frames=int(TRANSITION_BLEND_FRAMES),
        )
        join_records.append(
            {
                "source": str(source_name),
                "boundary": int(boundary),
                "boundary_time_sec": float(boundary / float(sample_rate)),
                "target_start_x": float(target_start_x),
                "appended_frames": int(len(pending)),
                **join,
            }
        )
    for row in pending:
        for key in q_keys:
            buffers[key].append(float(row[key]))
        labels.append(str(source_name))
    for foot_x, foot_z, contact in pending_aux:
        aux_buffers["foot_rel_x"].append(foot_x)
        aux_buffers["foot_rel_z"].append(foot_z)
        aux_buffers["foot_contact"].append(contact)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/muscle_2d_mjwarp_stageG0_h32_gated_ref_ramp_short_sac.json")
    parser.add_argument("--outdir", type=Path, default=ROOT / "results/stageG_long_course_reference")
    parser.add_argument("--sample-rate", type=float, default=30.0)
    args = parser.parse_args()

    config = load_config(args.config)
    base_course_segments = list(config.get("terrain_course", {}).get("segments", []))
    model, _ = build_muscle_model(config)
    loaded = {
        name: load_relative_q_series(
            path,
            model=model,
            control_hz=float(args.sample_rate),
            config=config,
            course_segments=base_course_segments,
        )
        for name, path in DEFAULT_SOURCES.items()
    }
    rel = {name: series for name, (_metadata, series) in loaded.items()}
    level_loop_metadata = find_level_loop_segment(rel["level"])
    rel["level_loop"] = extract_cyclic_segment(
        rel["level"],
        start_idx=int(level_loop_metadata["start"]),
        frames=int(level_loop_metadata["frames"]),
    )
    q_keys = [f"q_{joint}" for joint in RESET_JOINTS]
    level_x = np.asarray(rel["level_loop"]["q_pelvis_tx"], dtype=np.float64)
    level_stride = float(level_x[-1] - level_x[0])
    level_frames = len(level_x)
    level_start_x = -8.0
    level_to_up_start_x = level_start_x + LEAD_LEVEL_CYCLES * level_stride + np.ceil(LEAD_LEVEL_CYCLES) * GAP_M
    course_shift = level_to_up_start_x - BASE_LEVEL_TO_UP_START_X
    course_segments = shifted_course_segments(base_course_segments, course_shift)
    base_downhill_start_x = downhill_start_x(base_course_segments)
    base_downhill_end_x = downhill_end_x(base_course_segments)
    target_downhill_start_x = base_downhill_start_x + course_shift
    target_downhill_end_x = base_downhill_end_x + course_shift
    level_to_down_target_start_x = target_downhill_start_x - (base_downhill_start_x - float(rel["level_to_down"]["q_pelvis_tx"][0]))

    buffers = {key: [] for key in q_keys}
    aux_buffers: dict[str, list[np.ndarray]] = {"foot_rel_x": [], "foot_rel_z": [], "foot_contact": []}
    labels: list[str] = []
    join_records: list[dict[str, Any]] = []
    full_lead_cycles = int(np.floor(LEAD_LEVEL_CYCLES))
    partial_lead_fraction = float(LEAD_LEVEL_CYCLES - full_lead_cycles)
    for cycle in range(full_lead_cycles):
        target_start_x = level_start_x if not buffers["q_pelvis_tx"] else float(buffers["q_pelvis_tx"][-1]) + GAP_M
        append_clip(
            buffers,
            aux_buffers,
            rel["level_loop"],
            q_keys=q_keys,
            source_name=f"level_low_{cycle + 1}",
            target_start_x=target_start_x,
            course_segments=course_segments,
            sample_rate=float(args.sample_rate),
            join_records=join_records,
            match_join=bool(buffers["q_pelvis_tx"]),
            cyclic=True,
            max_frames=level_frames,
            labels=labels,
        )
    if partial_lead_fraction > 1e-6:
        target_start_x = level_start_x if not buffers["q_pelvis_tx"] else float(buffers["q_pelvis_tx"][-1]) + GAP_M
        target_partial_frames = max(1, int(round(level_frames * partial_lead_fraction)))
        partial_start, partial_frames = choose_cyclic_level_segment(
            buffers,
            aux_buffers,
            rel["level_loop"],
            rel["level_to_up"],
            target_frames=target_partial_frames,
            min_frames=max(8, int(round(0.25 * level_frames))),
            max_frames=min(level_frames, int(round(0.75 * level_frames))),
            course_segments=course_segments,
            sample_rate=float(args.sample_rate),
        )
        append_clip(
            buffers,
            aux_buffers,
            rel["level_loop"],
            q_keys=q_keys,
            source_name=f"level_low_{full_lead_cycles + 1}_partial",
            target_start_x=target_start_x,
            course_segments=course_segments,
            sample_rate=float(args.sample_rate),
            join_records=join_records,
            max_frames=partial_frames,
            match_join=bool(buffers["q_pelvis_tx"]),
            cyclic=True,
            forced_join_index=partial_start,
            labels=labels,
        )
    append_clip(
        buffers,
        aux_buffers,
        rel["level_to_up"],
        q_keys=q_keys,
        source_name="level_to_up",
        target_start_x=float(buffers["q_pelvis_tx"][-1]) + GAP_M,
        course_segments=course_segments,
        sample_rate=float(args.sample_rate),
        join_records=join_records,
        search_max_fraction=1.0,
        labels=labels,
    )

    last_x = float(buffers["q_pelvis_tx"][-1])
    append_clip(
        buffers,
        aux_buffers,
        rel["up_to_level"],
        q_keys=q_keys,
        source_name="up_to_level",
        target_start_x=last_x + GAP_M,
        course_segments=course_segments,
        sample_rate=float(args.sample_rate),
        join_records=join_records,
        search_max_fraction=1.0,
        labels=labels,
    )
    last_x = float(buffers["q_pelvis_tx"][-1])
    high_start_x = last_x + GAP_M
    high_start, high_frames, level_to_down_start, level_to_down_target_start_x = choose_high_to_down_join(
        buffers,
        aux_buffers,
        rel["level_loop"],
        rel["level_to_down"],
        high_start_x=high_start_x,
        target_downhill_start_x=target_downhill_start_x,
        base_downhill_start_x=base_downhill_start_x,
        max_level_frames=level_frames,
        min_level_frames=max(12, int(round(0.25 * level_frames))),
        max_level_to_down_start=0,
        min_level_to_down_frames=len(np.asarray(rel["level_to_down"]["q_pelvis_tx"])),
        level_to_down_max_x=float("inf"),
        course_segments=course_segments,
        sample_rate=float(args.sample_rate),
    )
    append_clip(
        buffers,
        aux_buffers,
        rel["level_loop"],
        q_keys=q_keys,
        source_name="level_high",
        target_start_x=high_start_x,
        course_segments=course_segments,
        sample_rate=float(args.sample_rate),
        join_records=join_records,
        max_frames=high_frames,
        match_join=True,
        cyclic=True,
        forced_join_index=high_start,
        labels=labels,
    )
    last_x = float(buffers["q_pelvis_tx"][-1])
    append_clip(
        buffers,
        aux_buffers,
        rel["level_to_down"],
        q_keys=q_keys,
        source_name="level_to_down",
        target_start_x=max(level_to_down_target_start_x, last_x + GAP_M),
        course_segments=course_segments,
        sample_rate=float(args.sample_rate),
        join_records=join_records,
        match_join=False,
        forced_join_index=level_to_down_start,
        labels=labels,
    )
    last_x = float(buffers["q_pelvis_tx"][-1])
    append_clip(
        buffers,
        aux_buffers,
        rel["down_to_level"],
        q_keys=q_keys,
        source_name="down_to_level",
        target_start_x=last_x + GAP_M,
        course_segments=course_segments,
        sample_rate=float(args.sample_rate),
        join_records=join_records,
        forced_join_index=0,
        labels=labels,
    )
    for cycle in range(TAIL_LEVEL_CYCLES):
        last_x = float(buffers["q_pelvis_tx"][-1])
        append_clip(
            buffers,
            aux_buffers,
            rel["level_loop"],
            q_keys=q_keys,
            source_name=f"level_low_tail_{cycle + 1}",
            target_start_x=last_x + GAP_M,
            course_segments=course_segments,
            sample_rate=float(args.sample_rate),
            join_records=join_records,
            match_join=True,
            cyclic=True,
            max_frames=level_frames,
            search_max_fraction=1.0,
            labels=labels,
        )

    q_series = {key: np.asarray(values, dtype=np.float32) for key, values in buffers.items()}
    series = {key: value.astype(np.float32) for key, value in add_derivatives(q_series, float(args.sample_rate), labels).items()}
    label_ranges: dict[str, dict[str, int]] = {}
    cursor = 0
    while cursor < len(labels):
        label = labels[cursor]
        end = cursor + 1
        while end < len(labels) and labels[end] == label:
            end += 1
        label_ranges[label] = {"start": cursor, "end": end}
        cursor = end
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    npz_path = outdir / "stageG_long_flat_up_flat_down_flat_myoassist_3d.npz"
    metadata = {
        "source_mode": "stageG_long_course_concat",
        "terrain_id": "stageG_long_course",
        "source_label": "flat-up-flat-down-flat",
        "sample_rate": float(args.sample_rate),
        "data_length": int(len(next(iter(series.values())))),
        "terrain_course_segments": course_segments,
        "segment_labels": labels,
        "label_ranges": label_ranges,
        "lead_level_cycles": int(LEAD_LEVEL_CYCLES),
        "lead_level_cycles_float": float(LEAD_LEVEL_CYCLES),
        "tail_level_cycles": int(TAIL_LEVEL_CYCLES),
        "transition_blend_frames": int(TRANSITION_BLEND_FRAMES),
        "level_loop": level_loop_metadata,
        "join_records": join_records,
        "source_paths": {name: str(path) for name, path in DEFAULT_SOURCES.items()},
    }
    np.savez_compressed(npz_path, metadata=metadata, series_data=series)

    long_config = json.loads(json.dumps(config))
    long_config["experiment_name"] = "muscle_2d_mjwarp_stageG_long_course_gated_ref_sac"
    long_config["reference_pool"] = {"paths": [str(npz_path.resolve())]}
    long_config["reference_pool_schedule"] = []
    long_config["terrain_course"]["segments"] = course_segments
    key_phases = [
        label_ranges[name]["start"]
        for name in (
            "level_low_1",
            "level_low_2",
            "level_low_2_partial",
            "level_to_up",
            "up_to_level",
            "level_high",
            "level_to_down",
            "down_to_level",
            "level_low_tail_1",
        )
        if name in label_ranges
    ]
    long_config["video"] = {"phase_indices": key_phases}
    long_config["checkpoint_video_export"]["every_steps"] = 4096
    long_config["checkpoint_video_export"]["video_height"] = 368
    long_config["checkpoint_video_export"]["video_width"] = 640
    long_config["checkpoint_video_export"]["phase_indices"] = key_phases
    long_config["reset"]["phase_windows"] = [{"start": 0, "end": min(160, int(metadata["data_length"]))}]
    long_config["reset_phase_schedule_mode"] = "absolute"
    long_config["reset_phase_schedule"] = [
        {
            "name": "flat_only_short",
            "after_steps": 0,
            "phase_windows": [{"start": 0, "end": min(160, int(metadata["data_length"]))}],
        },
        {
            "name": "flat_with_ascent_entry",
            "after_steps": 16384,
            "phase_windows": [
                {"start": 0, "end": min(160, int(metadata["data_length"]))},
                {"start": max(0, label_ranges["level_to_up"]["start"] - 32), "end": label_ranges["level_to_up"]["end"]},
            ],
        },
        {
            "name": "ascent_and_high_level",
            "after_steps": 32768,
            "phase_windows": [
                {"start": max(0, label_ranges["level_to_up"]["start"] - 48), "end": label_ranges["level_high"]["end"]},
            ],
        },
        {
            "name": "descent_course",
            "after_steps": 49152,
            "phase_windows": [
                {"start": max(0, label_ranges["level_high"]["start"] - 32), "end": label_ranges["down_to_level"]["end"]},
            ],
        },
        {
            "name": "balanced_full_course",
            "after_steps": 65536,
            "phase_windows": [
                {"start": 0, "end": min(160, int(metadata["data_length"]))},
                {"start": max(0, label_ranges["level_to_up"]["start"] - 48), "end": label_ranges["up_to_level"]["end"]},
                {"start": max(0, label_ranges["level_to_down"]["start"] - 48), "end": label_ranges["down_to_level"]["end"]},
                {"start": label_ranges["level_low_tail_1"]["start"], "end": min(label_ranges["level_low_tail_1"]["start"] + 160, int(metadata["data_length"]))},
            ],
        },
    ]
    long_config["reset"]["episode_steps"] = min(192, int(metadata["data_length"]))
    long_config_path = outdir / "muscle_2d_mjwarp_stageG_long_course_gated_ref_sac.json"
    long_config_path.write_text(json.dumps(long_config, indent=2) + "\n", encoding="utf-8")

    summary = {
        "npz": str(npz_path),
        "config": str(long_config_path),
        "frames": int(metadata["data_length"]),
        "duration_sec": float(metadata["data_length"] / float(args.sample_rate)),
        "x_start": float(series["q_pelvis_tx"][0]),
        "x_end": float(series["q_pelvis_tx"][-1]),
        "label_ranges": label_ranges,
        "level_loop": level_loop_metadata,
        "join_records": join_records,
        "sources": {label: labels.count(label) for label in sorted(set(labels))},
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
