#!/usr/bin/env python3
"""Project the reviewed course80 trajectory onto the planar 22-muscle model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT
    / "reference_exports/course80_3d_balanced_v8/course80_3d_balanced_v8.npz"
)
DEFAULT_XML = (
    ROOT.parent / "myoassist/models/22muscle_2D/myoLeg22_2D_BASELINE.xml"
)
DEFAULT_OUTDIR = ROOT / "reference_exports/course22_v1"
FOOT_SITES = ("r_heel_btm", "r_toe_btm", "l_heel_btm", "l_toe_btm")
SAGITTAL_JOINTS = (
    "hip_flexion_r",
    "knee_angle_r",
    "ankle_angle_r",
    "mtp_angle_r",
    "hip_flexion_l",
    "knee_angle_l",
    "ankle_angle_l",
    "mtp_angle_l",
)
RESET_JOINTS = ("pelvis_tx", "pelvis_ty", "pelvis_tilt", *SAGITTAL_JOINTS)
FRICTION = (1.0, 0.005, 0.0001)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    index = mujoco.mj_name2id(model, kind, name)
    if index < 0:
        raise KeyError(f"missing MuJoCo object {kind}: {name}")
    return int(index)


def qpos_address(model: mujoco.MjModel, joint: str) -> int:
    return int(model.jnt_qposadr[object_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)])


def dof_address(model: mujoco.MjModel, joint: str) -> int:
    return int(model.jnt_dofadr[object_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)])


def apply_joint_equalities(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    for equality in range(model.neq):
        if int(model.eq_type[equality]) != int(mujoco.mjtEq.mjEQ_JOINT):
            continue
        dependent = int(model.eq_obj1id[equality])
        driver = int(model.eq_obj2id[equality])
        qpos1 = int(model.jnt_qposadr[dependent])
        qpos2 = int(model.jnt_qposadr[driver])
        dof1 = int(model.jnt_dofadr[dependent])
        dof2 = int(model.jnt_dofadr[driver])
        coefficients = np.asarray(model.eq_data[equality, :5], dtype=np.float64)
        value = float(data.qpos[qpos2])
        velocity = float(data.qvel[dof2])
        data.qpos[qpos1] = sum(
            float(coefficient) * value**power
            for power, coefficient in enumerate(coefficients)
        )
        derivative = sum(
            power * float(coefficient) * value ** (power - 1)
            for power, coefficient in enumerate(coefficients)
            if power > 0
        )
        data.qvel[dof1] = derivative * velocity


def convert_terrain_segments(
    source_segments: list[dict[str, Any]],
    *,
    box_types: bool,
) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for source in source_segments:
        segment = deepcopy(source)
        kind = str(segment.get("type", ""))
        if kind == "flat_box":
            segment["type"] = "flat_box" if box_types else "flat"
        elif kind == "ramp_box":
            segment["type"] = "ramp_box" if box_types else "slope"
        elif kind == "stairs_box":
            segment["type"] = "stairs_box"
            segment["platform_depth"] = 0.0
            if int(segment.get("direction", 1)) < 0:
                # The 22 trainer defines the first descending tread at base_height,
                # while the course80 terrain defines it one riser below the platform.
                segment["source_base_height"] = float(segment["base_height"])
                segment["base_height"] = float(segment["base_height"]) - float(
                    segment["step_height"]
                )
        else:
            raise ValueError(f"unsupported source terrain type: {kind!r}")
        segments.append(segment)
    return segments


def stair_treads(segment: dict[str, Any]) -> list[tuple[float, float, float]]:
    x0 = float(segment["x0"])
    depth = float(segment["step_depth"])
    height = float(segment["step_height"])
    base = float(segment.get("base_height", 0.0))
    direction = int(segment.get("direction", 1))
    treads = []
    for index in range(int(segment["steps"])):
        top = (
            base + (index + 1) * height
            if direction > 0
            else base - index * height
        )
        treads.append((x0 + index * depth, x0 + (index + 1) * depth, top))
    return treads


def terrain_height(
    positions: np.ndarray, segments: list[dict[str, Any]]
) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float64)
    heights = np.zeros_like(positions)
    for segment in segments:
        x0 = float(segment["x0"])
        x1 = float(segment["x1"])
        mask = (positions >= x0) & (positions <= x1)
        kind = str(segment["type"])
        if kind in {"flat", "flat_box"}:
            heights[mask] = float(segment["height"])
        elif kind in {"slope", "ramp_box"}:
            slope = float(
                segment.get(
                    "slope",
                    (float(segment["height1"]) - float(segment["height0"]))
                    / (x1 - x0),
                )
            )
            heights[mask] = float(segment["height0"]) + slope * (
                positions[mask] - x0
            )
        elif kind == "stairs_box":
            for tread_x0, tread_x1, top in stair_treads(segment):
                tread_mask = (positions >= tread_x0) & (positions <= tread_x1)
                heights[tread_mask] = top
        else:
            raise ValueError(f"unsupported terrain type: {kind!r}")
    return heights


def base_qpos(model: mujoco.MjModel, keyframe: str) -> np.ndarray:
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
    if key_id >= 0:
        return np.asarray(model.key_qpos[key_id], dtype=np.float64).copy()
    return np.asarray(model.qpos0, dtype=np.float64).copy()


def pose_rows(
    model: mujoco.MjModel,
    source: dict[str, Any],
    pelvis_x: np.ndarray,
    pelvis_z: np.ndarray,
    *,
    keyframe: str,
) -> np.ndarray:
    length = len(pelvis_x)
    rows = np.repeat(base_qpos(model, keyframe)[None, :], length, axis=0)
    rows[:, qpos_address(model, "pelvis_tx")] = pelvis_x
    rows[:, qpos_address(model, "pelvis_ty")] = pelvis_z
    rows[:, qpos_address(model, "pelvis_tilt")] = np.asarray(
        source["q_pelvis_tilt"], dtype=np.float64
    )
    for joint in SAGITTAL_JOINTS:
        values = np.asarray(source[f"q_{joint}"], dtype=np.float64)
        if joint.startswith("knee_angle_"):
            values = -values
        rows[:, qpos_address(model, joint)] = values

    data = mujoco.MjData(model)
    for frame in range(length):
        data.qpos[:] = rows[frame]
        data.qvel[:] = 0.0
        apply_joint_equalities(model, data)
        rows[frame] = data.qpos
    return rows


def foot_positions(model: mujoco.MjModel, qpos: np.ndarray) -> np.ndarray:
    data = mujoco.MjData(model)
    site_ids = [
        object_id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in FOOT_SITES
    ]
    positions = np.zeros((len(qpos), len(site_ids), 3), dtype=np.float64)
    for frame, row in enumerate(qpos):
        data.qpos[:] = row
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        positions[frame] = data.site_xpos[site_ids]
    return positions


def smooth_signal(values: np.ndarray) -> np.ndarray:
    kernel = np.asarray([1, 4, 7, 10, 13, 10, 7, 4, 1], dtype=np.float64)
    kernel /= np.sum(kernel)
    radius = len(kernel) // 2
    padded = np.pad(np.asarray(values, dtype=np.float64), (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def infer_support_side(foot: np.ndarray, sample_rate: float) -> np.ndarray:
    horizontal_speed = np.abs(
        np.gradient(foot[:, :, 0], 1.0 / float(sample_rate), axis=0)
    )
    side_speed = np.column_stack(
        (
            np.mean(horizontal_speed[:, :2], axis=1),
            np.mean(horizontal_speed[:, 2:], axis=1),
        )
    )
    smoothed_speed = np.column_stack(
        (smooth_signal(side_speed[:, 0]), smooth_signal(side_speed[:, 1]))
    )
    return np.argmin(smoothed_speed, axis=1)


def stair_context_mask(
    metadata: dict[str, Any], length: int
) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    for label, bounds in metadata.get("label_ranges", {}).items():
        if "stair" not in str(label):
            continue
        mask[int(bounds["start"]) : int(bounds["end"])] = True
    return mask


def retarget_root(
    *,
    foot: np.ndarray,
    source_pelvis_z: np.ndarray,
    sample_rate: float,
    segments: list[dict[str, Any]],
    metadata: dict[str, Any],
    target_clearance: float,
    speed_scale: float,
    shift_limit: float,
    shift_step: float,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    relative_z = foot[:, :, 2] - source_pelvis_z[:, None]
    support_side = infer_support_side(foot, sample_rate)
    context = stair_context_mask(metadata, len(foot))
    if not np.any(context):
        context[:] = True

    candidates = []
    shifts = np.arange(
        -float(shift_limit),
        float(shift_limit) + 0.5 * float(shift_step),
        float(shift_step),
    )
    for shift in shifts:
        desired = (
            terrain_height(foot[:, :, 0] + float(shift), segments)
            + float(target_clearance)
            - relative_z
        )
        side_height = np.column_stack(
            (
                np.max(desired[:, :2], axis=1),
                np.max(desired[:, 2:], axis=1),
            )
        )
        raw_height = side_height[np.arange(len(foot)), support_side]
        height = smooth_signal(raw_height)
        first_step = np.diff(height)
        second_step = np.diff(height, n=2)
        context_first = context[1:]
        context_second = context[2:]
        score = (
            float(np.quantile(np.abs(first_step[context_first]), 0.99))
            + 2.0
            * float(np.quantile(np.abs(second_step[context_second]), 0.99))
            + 0.2 * float(np.max(np.abs(first_step[context_first]), initial=0.0))
        )
        candidates.append(
            {
                "shift": float(shift),
                "score": score,
                "max_frame_delta": float(
                    np.max(np.abs(first_step[context_first]), initial=0.0)
                ),
                "p99_frame_delta": float(
                    np.quantile(np.abs(first_step[context_first]), 0.99)
                ),
                "height": height,
            }
        )
    selected = min(
        candidates,
        key=lambda item: (float(item["score"]), abs(float(item["shift"]))),
    )
    diagnostics = {
        "selected_forward_shift_m": float(selected["shift"]),
        "selected_score": float(selected["score"]),
        "selected_max_frame_delta_m": float(selected["max_frame_delta"]),
        "selected_p99_frame_delta_m": float(selected["p99_frame_delta"]),
        "target_clearance_m": float(target_clearance),
        "horizontal_speed_scale_mps": float(speed_scale),
        "support_inference": (
            "slower foot side after symmetric 9-frame smoothing; "
            "height uses the higher heel/toe contact requirement on that side"
        ),
        "support_side_counts": {
            "right": int(np.sum(support_side == 0)),
            "left": int(np.sum(support_side == 1)),
        },
        "smoothing_kernel": [1, 4, 7, 10, 13, 10, 7, 4, 1],
        "best_candidates": [
            {
                key: float(item[key])
                for key in (
                    "shift",
                    "score",
                    "max_frame_delta",
                    "p99_frame_delta",
                )
            }
            for item in sorted(candidates, key=lambda item: float(item["score"]))[:8]
        ],
    }
    return (
        float(selected["shift"]),
        np.asarray(selected["height"], dtype=np.float64),
        diagnostics,
    )


def differentiate_qpos(
    model: mujoco.MjModel, qpos: np.ndarray, sample_rate: float
) -> np.ndarray:
    if len(qpos) == 1:
        return np.zeros((1, model.nv), dtype=np.float64)
    edge = np.zeros((len(qpos) - 1, model.nv), dtype=np.float64)
    dt = 1.0 / float(sample_rate)
    for frame in range(len(edge)):
        mujoco.mj_differentiatePos(
            model,
            edge[frame],
            dt,
            np.asarray(qpos[frame], dtype=np.float64),
            np.asarray(qpos[frame + 1], dtype=np.float64),
        )
    qvel = np.zeros((len(qpos), model.nv), dtype=np.float64)
    qvel[0] = edge[0]
    qvel[-1] = edge[-1]
    if len(qpos) > 2:
        qvel[1:-1] = 0.5 * (edge[:-1] + edge[1:])
    return qvel


def build_series(
    model: mujoco.MjModel, qpos: np.ndarray, qvel: np.ndarray
) -> dict[str, np.ndarray]:
    series: dict[str, np.ndarray] = {
        "qpos_full": qpos.astype(np.float32),
        "qvel_full": qvel.astype(np.float32),
    }
    for joint in RESET_JOINTS:
        series[f"q_{joint}"] = qpos[:, qpos_address(model, joint)].astype(np.float32)
        series[f"dq_{joint}"] = qvel[:, dof_address(model, joint)].astype(np.float32)
    series["q_root_x"] = series["q_pelvis_tx"].copy()
    series["q_root_z"] = series["q_pelvis_ty"].copy()
    series["dq_root_x"] = series["dq_pelvis_tx"].copy()
    series["dq_root_z"] = series["dq_pelvis_ty"].copy()
    return series


def add_box(
    world: ET.Element,
    *,
    name: str,
    x0: float,
    x1: float,
    top: float,
    half_width: float,
    color: str,
    thickness: float = 0.08,
) -> None:
    thickness = max(float(thickness), min(float(top), 0.08))
    ET.SubElement(
        world,
        "geom",
        {
            "name": name,
            "type": "box",
            "pos": f"{0.5 * (x0 + x1):.10f} 0 {top - 0.5 * thickness:.10f}",
            "size": f"{0.5 * (x1 - x0):.10f} {half_width:.10f} {0.5 * thickness:.10f}",
            "rgba": color,
            "friction": " ".join(str(value) for value in FRICTION),
            "contype": "1",
            "conaffinity": "1",
        },
    )


def add_ramp(
    world: ET.Element,
    *,
    name: str,
    x0: float,
    x1: float,
    height0: float,
    height1: float,
    half_width: float,
    color: str,
    thickness: float = 0.14,
    overlap: float = 0.08,
) -> None:
    x0 -= overlap
    x1 += overlap
    angle = math.atan2(height1 - height0, x1 - x0)
    length = math.hypot(x1 - x0, height1 - height0)
    half_thickness = 0.5 * thickness
    normal_x = -math.sin(angle)
    normal_z = math.cos(angle)
    center_x = 0.5 * (x0 + x1) - half_thickness * normal_x
    center_z = 0.5 * (height0 + height1) - half_thickness * normal_z
    ET.SubElement(
        world,
        "geom",
        {
            "name": name,
            "type": "box",
            "pos": f"{center_x:.10f} 0 {center_z:.10f}",
            "size": f"{0.5 * length:.10f} {half_width:.10f} {half_thickness:.10f}",
            "quat": (
                f"{math.cos(0.5 * angle):.10f} 0 "
                f"{-math.sin(0.5 * angle):.10f} 0"
            ),
            "rgba": color,
            "friction": " ".join(str(value) for value in FRICTION),
            "contype": "1",
            "conaffinity": "1",
        },
    )


def write_terrain_include(
    path: Path, segments: list[dict[str, Any]], half_width: float
) -> int:
    root = ET.Element("mujocoinclude")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(
        world,
        "light",
        {
            "dir": "0 0 -1",
            "directional": "true",
            "diffuse": "0.8 0.8 0.8",
            "specular": "0 0 0",
            "pos": "0 -3 3",
            "mode": "trackcom",
        },
    )
    for name in ("ground-plane", "terrain"):
        ET.SubElement(
            world,
            "geom",
            {
                "name": name,
                "type": "plane",
                "pos": "0 0 -10",
                "size": "50 2.5 0.125",
                "rgba": "1 1 1 0",
                "contype": "0",
                "conaffinity": "0",
            },
        )

    colors = {
        "flat_low": "0.62 0.74 0.88 1",
        "flat_high": "0.12 0.42 0.86 1",
        "ramp": "0.86 0.70 0.32 1",
        "step_a": "0.92 0.56 0.12 1",
        "step_b": "0.22 0.66 0.30 1",
    }
    count = 0
    for segment_index, segment in enumerate(segments):
        kind = str(segment["type"])
        if kind == "flat_box":
            height = float(segment["height"])
            add_box(
                world,
                name=f"course22_flat_{segment_index:02d}",
                x0=float(segment["x0"]),
                x1=float(segment["x1"]),
                top=height,
                half_width=half_width,
                color=colors["flat_high"] if height > 1.0 else colors["flat_low"],
            )
            count += 1
        elif kind == "ramp_box":
            add_ramp(
                world,
                name=f"course22_ramp_{segment_index:02d}",
                x0=float(segment["x0"]),
                x1=float(segment["x1"]),
                height0=float(segment["height0"]),
                height1=float(segment["height1"]),
                half_width=half_width,
                color=colors["ramp"],
            )
            count += 1
        elif kind == "stairs_box":
            for tread_index, (x0, x1, top) in enumerate(stair_treads(segment)):
                add_box(
                    world,
                    name=f"course22_stair_{segment_index:02d}_{tread_index + 1:02d}",
                    x0=x0,
                    x1=x1,
                    top=top,
                    half_width=half_width,
                    color=colors["step_a"] if tread_index % 2 == 0 else colors["step_b"],
                )
                count += 1
    ET.indent(root, space="  ")
    path.write_text(ET.tostring(root, encoding="unicode") + "\n", encoding="utf-8")
    return count


def validation_report(
    *,
    model: mujoco.MjModel,
    qpos: np.ndarray,
    qvel: np.ndarray,
    segments: list[dict[str, Any]],
    metadata: dict[str, Any],
    foot: np.ndarray,
    sample_rate: float,
) -> dict[str, Any]:
    clearance = foot[:, :, 2] - terrain_height(foot[:, :, 0], segments)
    support_side = infer_support_side(foot, sample_rate)
    side_clearance = np.column_stack(
        (
            np.min(clearance[:, :2], axis=1),
            np.min(clearance[:, 2:], axis=1),
        )
    )
    support_clearance = side_clearance[np.arange(len(foot)), support_side]
    ranges = {}
    for label, bounds in metadata.get("label_ranges", {}).items():
        start = int(bounds["start"])
        end = int(bounds["end"])
        values = clearance[start:end]
        support_values = support_clearance[start:end]
        ranges[label] = {
            "all_sites_min_clearance_m": float(np.min(values)),
            "all_sites_p05_clearance_m": float(np.quantile(values, 0.05)),
            "all_sites_median_frame_min_clearance_m": float(
                np.median(np.min(values, axis=1))
            ),
            "all_sites_frames_any_below_minus_3cm": int(
                np.sum(np.any(values < -0.03, axis=1))
            ),
            "inferred_support_min_clearance_m": float(
                np.min(support_values)
            ),
            "inferred_support_p05_clearance_m": float(
                np.quantile(support_values, 0.05)
            ),
            "inferred_support_frames_below_minus_3cm": int(
                np.sum(support_values < -0.03)
            ),
        }
    limit_violations = []
    for joint_index in range(model.njnt):
        if not bool(model.jnt_limited[joint_index]):
            continue
        address = int(model.jnt_qposadr[joint_index])
        low, high = np.asarray(model.jnt_range[joint_index], dtype=np.float64)
        values = qpos[:, address]
        if np.min(values) < low or np.max(values) > high:
            limit_violations.append(
                {
                    "joint": mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_JOINT, joint_index
                    ),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "range": [float(low), float(high)],
                }
            )
    support_contact_failed = any(
        item["inferred_support_frames_below_minus_3cm"]
        for item in ranges.values()
    )
    tracked_limit_violations = [
        item
        for item in limit_violations
        if item["joint"] in SAGITTAL_JOINTS
    ]
    if support_contact_failed:
        status = "review_support_contact"
    elif tracked_limit_violations:
        status = "pass_contact_review_joint_limits"
    else:
        status = "pass"
    return {
        "status": status,
        "frames": int(len(qpos)),
        "qpos_shape": list(qpos.shape),
        "qvel_shape": list(qvel.shape),
        "max_pelvis_height_frame_delta_m": float(
            np.max(
                np.abs(
                    np.diff(qpos[:, qpos_address(model, "pelvis_ty")])
                ),
                initial=0.0,
            )
        ),
        "max_abs_qvel": float(np.max(np.abs(qvel), initial=0.0)),
        "ranges": ranges,
        "joint_limit_violations": limit_violations,
        "tracked_joint_limit_violations": tracked_limit_violations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--keyframe", default="walk_left")
    parser.add_argument("--target-clearance", type=float, default=0.008)
    parser.add_argument("--speed-scale", type=float, default=0.35)
    parser.add_argument("--shift-limit", type=float, default=0.12)
    parser.add_argument("--shift-step", type=float, default=0.0025)
    parser.add_argument("--terrain-half-width", type=float, default=1.25)
    args = parser.parse_args()

    source_path = args.source.resolve()
    xml_path = args.xml.resolve()
    outdir = args.outdir.resolve()
    if not source_path.is_file():
        parser.error(f"source reference not found: {source_path}")
    if not xml_path.is_file():
        parser.error(f"22-muscle XML not found: {xml_path}")

    raw = np.load(source_path, allow_pickle=True)
    source_metadata = dict(raw["metadata"].item())
    source_series = dict(raw["series_data"].item())
    sample_rate = float(source_metadata["sample_rate"])
    length = int(source_metadata["data_length"])
    source_pelvis_x = np.asarray(source_series["q_pelvis_tx"], dtype=np.float64)
    source_pelvis_z = np.asarray(source_series["q_pelvis_ty"], dtype=np.float64)
    if len(source_pelvis_x) != length:
        raise ValueError("source metadata length does not match series")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    if (model.nq, model.nv) != (53, 53):
        raise ValueError(
            f"expected planar 22-muscle nq=nv=53, got nq={model.nq}, nv={model.nv}"
        )
    metadata_segments = convert_terrain_segments(
        list(source_metadata["terrain_course_segments"]), box_types=False
    )
    box_segments = convert_terrain_segments(
        list(source_metadata["terrain_course_segments"]), box_types=True
    )

    provisional_qpos = pose_rows(
        model,
        source_series,
        source_pelvis_x,
        source_pelvis_z,
        keyframe=str(args.keyframe),
    )
    provisional_foot = foot_positions(model, provisional_qpos)
    forward_shift, pelvis_z, retarget = retarget_root(
        foot=provisional_foot,
        source_pelvis_z=source_pelvis_z,
        sample_rate=sample_rate,
        segments=metadata_segments,
        metadata=source_metadata,
        target_clearance=float(args.target_clearance),
        speed_scale=float(args.speed_scale),
        shift_limit=float(args.shift_limit),
        shift_step=float(args.shift_step),
    )
    pelvis_x = source_pelvis_x + forward_shift
    qpos = pose_rows(
        model,
        source_series,
        pelvis_x,
        pelvis_z,
        keyframe=str(args.keyframe),
    )
    qvel = differentiate_qpos(model, qpos, sample_rate)
    series = build_series(model, qpos, qvel)
    foot = foot_positions(model, qpos)

    metadata = {
        "variant": "course22_v1",
        "purpose": "Planar 22-muscle projection of the reviewed course80_3d_balanced_v8 long reference.",
        "source_reference": "../course80_3d_balanced_v8/course80_3d_balanced_v8.npz",
        "source_sha256": sha256(source_path),
        "source_variant": source_metadata.get("variant"),
        "target_model": "myoLeg22_2D_BASELINE",
        "compatible_target_models": [
            "myoLeg22_2D_BASELINE",
            "myoLeg22_2D_HMEDI",
        ],
        "target_xml_repository_path": "models/22muscle_2D/myoLeg22_2D_BASELINE.xml",
        "target_xml_sha256": sha256(xml_path),
        "sample_rate": sample_rate,
        "data_length": length,
        "terrain_type": "course",
        "terrain_forward_axis": "+x",
        "terrain_course_segments": metadata_segments,
        "segment_labels": source_metadata.get("segment_labels"),
        "label_ranges": source_metadata.get("label_ranges"),
        "fine_label_ranges": source_metadata.get("fine_label_ranges"),
        "terrain_event_ranges": source_metadata.get("terrain_event_ranges"),
        "mapping": {
            "forward": "80 world +y / q_pelvis_tx -> 22 pelvis_tx + constant retarget shift",
            "vertical": "stance-speed-weighted 22 foot-to-terrain solve with a 9-frame symmetric smoother",
            "pelvis_tilt": "direct scalar projection",
            "hip_flexion": "direct",
            "knee_angle": "sign flipped from 80 convention to 22 convention",
            "ankle_angle": "direct",
            "mtp_angle": "direct",
            "discarded_3d_dofs": [
                "root lateral translation",
                "pelvis list",
                "pelvis rotation",
                "bilateral hip adduction",
                "bilateral hip rotation",
                "bilateral subtalar",
            ],
        },
        "root_retarget": retarget,
        "qpos_contract": {
            "nq": int(model.nq),
            "nv": int(model.nv),
            "full_state_reset": True,
            "keyframe_base": str(args.keyframe),
        },
    }
    report = validation_report(
        model=model,
        qpos=qpos,
        qvel=qvel,
        segments=metadata_segments,
        metadata=metadata,
        foot=foot,
        sample_rate=sample_rate,
    )

    outdir.mkdir(parents=True, exist_ok=True)
    npz_path = outdir / "course22_v1.npz"
    np.savez_compressed(npz_path, metadata=metadata, series_data=series)
    include_path = outdir / "terrain_course22_include.xml"
    geom_count = write_terrain_include(
        include_path, box_segments, float(args.terrain_half_width)
    )
    terrain_contract = {
        "schema_version": 1,
        "reference": npz_path.name,
        "reference_sha256": sha256(npz_path),
        "terrain_include": include_path.name,
        "terrain_forward_axis": "+x",
        "terrain_half_width_m": float(args.terrain_half_width),
        "terrain_geom_count": int(geom_count),
        "friction": list(FRICTION),
        "segments": box_segments,
        "trainer_settings": {
            "terrain_forward_axis": "x",
            "terrain_box_hide_hfield": True,
            "disable_ground_plane_contact": True,
            "terrain_box_half_width": float(args.terrain_half_width),
            "stair_box_half_width": float(args.terrain_half_width),
            "add_terrain_box_contact_pairs": True,
            "add_stair_box_contact_pairs": True,
        },
    }
    (outdir / "terrain_contract.json").write_text(
        json.dumps(terrain_contract, indent=2) + "\n", encoding="utf-8"
    )
    training_fragment = {
        "terrain_forward_axis": "x",
        "model": {
            "source_xml": "models/22muscle_2D/myoLeg22_2D_BASELINE.xml",
            "disable_multiccd": True,
        },
        "control": {
            "control_hz": int(round(sample_rate)),
        },
        "reference_pool": {
            "paths": [
                "reference_exports/course22_v1/course22_v1.npz",
            ],
        },
        "terrain_course": {
            "enabled": True,
            **terrain_contract["trainer_settings"],
            "terrain_box_thickness": 0.08,
            "segments": box_segments,
        },
    }
    (outdir / "training_config_fragment.json").write_text(
        json.dumps(training_fragment, indent=2) + "\n",
        encoding="utf-8",
    )
    (outdir / "validation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "reference": str(npz_path),
        "terrain_include": str(include_path),
        "terrain_contract": str(outdir / "terrain_contract.json"),
        "frames": length,
        "duration_sec": length / sample_rate,
        "selected_forward_shift_m": forward_shift,
        "pelvis_height_source_delta_mean_m": float(
            np.mean(pelvis_z - source_pelvis_z)
        ),
        "pelvis_height_source_delta_min_m": float(
            np.min(pelvis_z - source_pelvis_z)
        ),
        "pelvis_height_source_delta_max_m": float(
            np.max(pelvis_z - source_pelvis_z)
        ),
        "validation_status": report["status"],
        "tracked_joint_limit_violations": report[
            "tracked_joint_limit_violations"
        ],
    }
    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
