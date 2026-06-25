#!/usr/bin/env python3
"""Build and render small Camargo level<->ramp stitched reference previews."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
TERRAIN_SCRIPTS = Path("/home/lzn/exoskeleton_terrain/scripts")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if str(TERRAIN_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(TERRAIN_SCRIPTS))

from convert_camargo_ab06_to_myoassist_reference import (  # noqa: E402
    best_label_run,
    camargo_ik_to_myoassist_series,
    decode_matlab_table,
    infer_sample_rate,
    save_reference_npz,
    table_segment,
)
from render_camargo_ramp_reference_batch import (  # noqa: E402
    infer_exit_transition_x_from_foot,
    render_one,
)

DEFAULT_CAMARGO_DIR = Path("/mnt/c/Users/liang/Desktop/camargo")
DEFAULT_BASE_CONFIG = ROOT / "configs" / "muscle_2d_mjwarp_stageF_rampascent_only_h192_terrainpreview32_from409k_sac.json"
DEFAULT_SOURCE_MANIFEST = ROOT / "results" / "camargo_ab06_ab08_ramp6_selected4_corrected_terrain" / "manifest.json"
DEFAULT_STEADY_RAMP_MANIFEST = ROOT / "results" / "camargo_ramp6_source_selected4" / "manifest.json"
DEFAULT_OUTDIR = ROOT / "results" / "camargo_ramp6_stitched_preview"
DEFAULT_RAWMARKER_UP_ID = "camargo_ab08_slopeascent_walk-rampascent_ramp_6_r_01_02_rawmarker"
DEFAULT_RAWMARKER_DOWN_ID = "camargo_ab06_slopedescent_walk-rampdescent_ramp_6_r_01_05_rawmarker"
DEFAULT_STEADY_UP_ID = "camargo_ab08_slopeascent_rampascent_ramp_6_l_01_05"
DEFAULT_STEADY_DOWN_ID = "camargo_ab06_slopedescent_rampdescent_ramp_6_l_01_05"

MATCH_KEYS = [
    "q_pelvis_tilt",
    "q_hip_flexion_r",
    "q_knee_angle_r",
    "q_ankle_angle_r",
    "q_mtp_angle_r",
    "q_hip_flexion_l",
    "q_knee_angle_l",
    "q_ankle_angle_l",
    "q_mtp_angle_l",
    "dq_pelvis_tilt",
    "dq_hip_flexion_r",
    "dq_knee_angle_r",
    "dq_ankle_angle_r",
    "dq_hip_flexion_l",
    "dq_knee_angle_l",
    "dq_ankle_angle_l",
]
MATCH_WEIGHTS = np.array(
    [2.0, 2.0, 2.0, 1.0, 0.2, 2.0, 2.0, 1.0, 0.2, 0.3, 0.25, 0.25, 0.15, 0.25, 0.25, 0.15],
    dtype=np.float64,
)


@dataclass
class Reference:
    path: Path
    reference_id: str
    subject: str
    label: str
    metadata: dict[str, Any]
    series: dict[str, np.ndarray]
    sample_rate: float


@dataclass
class FlatSegment:
    subject: str
    trial: str
    metadata: dict[str, Any]
    series: dict[str, np.ndarray]
    sample_rate: float
    features: np.ndarray


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def load_npz_reference(path: Path, reference_id: str, subject: str, label: str) -> Reference:
    with np.load(path, allow_pickle=True) as data:
        metadata = dict(data["metadata"].item())
        series = {key: np.asarray(value, dtype=np.float64) for key, value in data["series_data"].item().items()}
    return Reference(path, reference_id, subject, label, metadata, series, float(metadata["sample_rate"]))


def manifest_by_id(path: Path) -> dict[str, dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return {str(item["reference_id"]): item for item in manifest["records"]}


def reference_from_manifest_record(record: dict[str, Any]) -> Reference:
    npz_path = Path(record["npz_path"])
    if not npz_path.is_absolute():
        npz_path = ROOT / npz_path
    return load_npz_reference(
        npz_path,
        str(record["reference_id"]),
        str(record["subject"]),
        str(record["label"]),
    )


def find_mode_dir(camargo_dir: Path, subject: str, mode: str) -> Path:
    matches = sorted(path for path in (camargo_dir / subject.upper()).glob(f"*/{mode}") if path.is_dir())
    if not matches:
        raise FileNotFoundError(f"Could not find {subject}/{mode} under {camargo_dir}")
    return matches[0]


def load_flat_segments(camargo_dir: Path, subject: str, count: int) -> list[FlatSegment]:
    mode_dir = find_mode_dir(camargo_dir, subject, "levelground")
    trials = sorted((mode_dir / "conditions").glob("levelground_cw_normal_*.mat"))[:count]
    if not trials:
        raise FileNotFoundError(f"No levelground_cw_normal trials under {mode_dir}")
    out: list[FlatSegment] = []
    for condition_path in trials:
        trial = condition_path.name
        ik_path = mode_dir / "ik" / trial
        if not ik_path.exists():
            raise FileNotFoundError(ik_path)
        labels = decode_matlab_table(condition_path.read_bytes())
        ik = decode_matlab_table(ik_path.read_bytes())
        start, end = best_label_run(labels["Label"], "walk")
        ik_segment = table_segment(ik, start, end)
        time = np.asarray(ik_segment["Header"], dtype=np.float64)
        series, transform = camargo_ik_to_myoassist_series(ik_segment)
        sample_rate = infer_sample_rate(time)
        series_np = {key: np.asarray(value, dtype=np.float64) for key, value in series.items()}
        metadata = {
            "sample_rate": float(sample_rate),
            "data_length": int(len(time)),
            "source_dataset": "Camargo",
            "source_subject": subject.upper(),
            "source_mode": "levelground",
            "source_trial": trial,
            "source_label": "walk",
            "terrain_id": "levelwalking",
            "terrain_type": "flat",
            "terrain_params": "",
            "root_transform": transform,
            "source_segment_start_index": int(start),
            "source_segment_end_index": int(end),
            "source_time_start": float(time[0]),
            "source_time_end": float(time[-1]),
        }
        out.append(
            FlatSegment(
                subject=subject.upper(),
                trial=trial,
                metadata=metadata,
                series=series_np,
                sample_rate=float(sample_rate),
                features=feature_matrix(series_np),
            )
        )
    return out


def feature_matrix(series: dict[str, np.ndarray]) -> np.ndarray:
    return np.stack([np.asarray(series[key], dtype=np.float64) for key in MATCH_KEYS], axis=1)


def feature_at(series: dict[str, np.ndarray], index: int) -> np.ndarray:
    return np.array([float(np.asarray(series[key], dtype=np.float64)[index]) for key in MATCH_KEYS], dtype=np.float64)


def terrain_height_from_metadata(x: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
    source_segments = metadata.get("source_terrain_segments")
    if isinstance(source_segments, list) and source_segments:
        return course_height_from_segments(x, source_segments)
    if str(metadata.get("terrain_type", "flat")) != "slope":
        return np.zeros_like(x, dtype=np.float64)
    values = [float(item) for item in str(metadata.get("terrain_params", "") or "").split()]
    slope = values[0] if values else 0.0
    anchor = values[2] if len(values) >= 3 else 0.0
    return anchor + slope * x


def course_height_from_segments(x: np.ndarray, segments: list[dict[str, Any]]) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float64)
    for segment in segments:
        x0 = float(segment.get("x0", -np.inf))
        x1 = float(segment.get("x1", np.inf))
        mask = (x >= x0) & (x <= x1)
        kind = str(segment.get("type", "flat"))
        if kind == "flat":
            out[mask] = float(segment.get("height", 0.0))
        elif kind == "slope":
            out[mask] = float(segment.get("height0", 0.0)) + float(segment.get("slope", 0.0)) * (x[mask] - x0)
    return out


def height_relative_series(series: dict[str, np.ndarray], metadata: dict[str, Any]) -> dict[str, np.ndarray]:
    out = {key: np.asarray(value, dtype=np.float64).copy() for key, value in series.items() if key.startswith("q_")}
    x = np.asarray(series["q_pelvis_tx"], dtype=np.float64)
    out["q_pelvis_ty"] = np.asarray(series["q_pelvis_ty"], dtype=np.float64) - terrain_height_from_metadata(x, metadata)
    return out


def find_flat_match(
    flat_segments: list[FlatSegment],
    subject: str,
    target: np.ndarray,
    *,
    before: int,
    after: int,
) -> tuple[FlatSegment, int, float]:
    best: tuple[FlatSegment, int, float] | None = None
    for segment in flat_segments:
        if segment.subject != subject.upper():
            continue
        length = int(segment.features.shape[0])
        lo = max(int(before), 0)
        hi = length - max(int(after), 0)
        if hi <= lo:
            continue
        diff = (segment.features[lo:hi] - target[None, :]) * MATCH_WEIGHTS[None, :]
        costs = np.mean(diff * diff, axis=1)
        local = int(np.argmin(costs))
        index = lo + local
        rmse = float(math.sqrt(float(costs[local])))
        if best is None or rmse < best[2]:
            best = (segment, index, rmse)
    if best is None:
        raise ValueError(f"No flat match for subject={subject} before={before} after={after}")
    return best


def append_source(
    buffers: dict[str, list[float]],
    source: dict[str, np.ndarray],
    indices: np.ndarray,
    *,
    q_keys: list[str],
    x_values: np.ndarray,
) -> None:
    for out_index, source_index in enumerate(indices):
        for key in q_keys:
            value = float(x_values[out_index]) if key == "q_pelvis_tx" else float(source[key][source_index])
            buffers[key].append(value)


def append_blend(
    buffers: dict[str, list[float]],
    a: dict[str, np.ndarray],
    a_indices: np.ndarray,
    b: dict[str, np.ndarray],
    b_indices: np.ndarray,
    *,
    q_keys: list[str],
    x_values: np.ndarray,
) -> None:
    count = min(len(a_indices), len(b_indices), len(x_values))
    if count <= 0:
        return
    alpha = np.linspace(0.0, 1.0, count, dtype=np.float64)
    for out_index in range(count):
        ai = int(a_indices[out_index])
        bi = int(b_indices[out_index])
        for key in q_keys:
            if key == "q_pelvis_tx":
                value = float(x_values[out_index])
            else:
                value = (1.0 - alpha[out_index]) * float(a[key][ai]) + alpha[out_index] * float(b[key][bi])
            buffers[key].append(value)


def finalize_series(buffers: dict[str, list[float]], sample_rate: float) -> dict[str, np.ndarray]:
    q_series = {key: np.asarray(values, dtype=np.float64) for key, values in buffers.items()}
    if "q_pelvis_tx" in q_series:
        q_series["q_pelvis_tx"] = np.maximum.accumulate(q_series["q_pelvis_tx"])
    time = np.arange(len(next(iter(q_series.values()))), dtype=np.float64) / float(sample_rate)
    out = dict(q_series)
    for key, values in q_series.items():
        if key.startswith("q_"):
            out[f"dq_{key[2:]}"] = np.gradient(values, time)
    return out


def shifted_x(values: np.ndarray, output_start: float) -> np.ndarray:
    return output_start + (values - float(values[0]))


def monotonic_x_from_values(values: np.ndarray, fallback_step: float) -> np.ndarray:
    raw = np.asarray(values, dtype=np.float64)
    if raw.size == 0:
        return raw.copy()
    increments = np.diff(raw)
    positive = increments[increments > 1e-6]
    step = float(np.median(positive)) if positive.size else float(fallback_step)
    increments = np.where(increments > 1e-6, increments, step)
    return np.r_[0.0, np.cumsum(increments)]


def monotonic_x_span(start: float, end: float, count: int) -> np.ndarray:
    if count <= 0:
        return np.zeros((0,), dtype=np.float64)
    if count == 1:
        return np.asarray([start], dtype=np.float64)
    if end <= start:
        end = start + 1e-3 * float(count - 1)
    return np.linspace(float(start), float(end), int(count), dtype=np.float64)


def slope_value(metadata: dict[str, Any]) -> float:
    return math.tan(math.radians(abs(float(metadata.get("ramp_incline_deg", 18.0)))))


def build_level_to_ramp(
    *,
    ramp: Reference,
    flat_segments: list[FlatSegment],
    q_keys: list[str],
    blend_frames: int,
    pre_frames: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, float | str]], dict[str, Any]]:
    ramp_rel = height_relative_series(ramp.series, ramp.metadata)
    target = feature_at(ramp.series, 0)
    flat, match_index, rmse = find_flat_match(flat_segments, ramp.subject, target, before=pre_frames, after=blend_frames + 1)
    flat_rel = height_relative_series(flat.series, flat.metadata)

    start = match_index - pre_frames
    blend_start = match_index - blend_frames
    flat_pre = np.arange(start, blend_start, dtype=np.int64)
    flat_blend = np.arange(blend_start, match_index, dtype=np.int64)
    ramp_blend = np.arange(0, blend_frames, dtype=np.int64)
    ramp_post = np.arange(blend_frames, len(ramp.series["q_pelvis_tx"]), dtype=np.int64)

    buffers = {key: [] for key in q_keys}
    flat_x = np.asarray(flat.series["q_pelvis_tx"], dtype=np.float64)
    ramp_x = np.asarray(ramp.series["q_pelvis_tx"], dtype=np.float64)
    flat_pre_x = monotonic_x_from_values(flat_x[flat_pre], fallback_step=0.004)
    x_blend_start = float(flat_pre_x[-1] + np.median(np.diff(flat_pre_x))) if flat_pre_x.size > 1 else 0.0
    ramp_step = float(np.median(np.diff(ramp_x[ramp_blend]))) if len(ramp_blend) > 1 else 0.004
    blend_x = x_blend_start + np.arange(len(ramp_blend), dtype=np.float64) * max(ramp_step, 1e-4)
    x_offset = float(blend_x[0] - ramp_x[0])
    append_source(buffers, flat_rel, flat_pre, q_keys=q_keys, x_values=flat_pre_x)
    append_blend(buffers, flat_rel, flat_blend, ramp_rel, ramp_blend, q_keys=q_keys, x_values=blend_x)
    append_source(buffers, ramp_rel, ramp_post, q_keys=q_keys, x_values=x_offset + ramp_x[ramp_post])

    output = finalize_series(buffers, ramp.sample_rate)
    x_end = float(output["q_pelvis_tx"][-1])
    slope = slope_value(ramp.metadata)
    slope_start = x_blend_start
    if ramp.label == "rampascent":
        segments = [
            {"type": "flat", "x0": -20.0, "x1": slope_start, "height": 0.0},
            {"type": "slope", "x0": slope_start, "x1": x_end + 1.0, "height0": 0.0, "slope": slope},
        ]
    else:
        height = slope * max(x_end + 1.0 - slope_start, 0.1)
        segments = [
            {"type": "flat", "x0": -20.0, "x1": slope_start, "height": height},
            {"type": "slope", "x0": slope_start, "x1": x_end + 1.0, "height0": height, "slope": -slope},
            {"type": "flat", "x0": x_end + 1.0, "x1": x_end + 20.0, "height": 0.0},
        ]
    diagnostics = {
        "flat_trial": flat.trial,
        "flat_match_index": int(match_index),
        "flat_match_time": float(match_index / flat.sample_rate),
        "match_rmse": rmse,
        "blend_frames": int(blend_frames),
        "pre_frames": int(pre_frames),
        "slope_start_x": float(slope_start),
    }
    return output, segments, diagnostics


def find_ramp_match(
    ramp: Reference,
    target: np.ndarray,
    *,
    before: int,
    after: int,
) -> tuple[int, float]:
    features = feature_matrix(ramp.series)
    length = int(features.shape[0])
    lo = max(int(before), 0)
    hi = length - max(int(after), 0)
    if hi <= lo:
        raise ValueError(f"No valid ramp match range for {ramp.reference_id}: before={before} after={after}")
    diff = (features[lo:hi] - target[None, :]) * MATCH_WEIGHTS[None, :]
    costs = np.mean(diff * diff, axis=1)
    local = int(np.argmin(costs))
    return lo + local, float(math.sqrt(float(costs[local])))


def build_level_to_ramp_from_transition(
    *,
    transition: Reference,
    ramp: Reference,
    q_keys: list[str],
    blend_frames: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, float | str]], dict[str, Any]]:
    trans_rel = height_relative_series(transition.series, transition.metadata)
    ramp_rel = height_relative_series(ramp.series, ramp.metadata)
    target = feature_at(transition.series, len(transition.series["q_pelvis_tx"]) - 1)
    match_index, rmse = find_ramp_match(ramp, target, before=blend_frames, after=blend_frames + 1)

    n_trans = len(transition.series["q_pelvis_tx"])
    trans_pre = np.arange(0, max(n_trans - blend_frames, 1), dtype=np.int64)
    trans_blend = np.arange(max(n_trans - blend_frames, 0), n_trans, dtype=np.int64)
    ramp_blend = np.arange(match_index - len(trans_blend), match_index, dtype=np.int64)
    ramp_post = np.arange(match_index, len(ramp.series["q_pelvis_tx"]), dtype=np.int64)

    buffers = {key: [] for key in q_keys}
    trans_x = np.asarray(transition.series["q_pelvis_tx"], dtype=np.float64)
    ramp_x = np.asarray(ramp.series["q_pelvis_tx"], dtype=np.float64)
    pre_x = monotonic_x_from_values(trans_x[trans_pre], fallback_step=0.004)
    append_source(buffers, trans_rel, trans_pre, q_keys=q_keys, x_values=pre_x)
    pre_positive = np.diff(pre_x)
    pre_positive = pre_positive[pre_positive > 1e-6]
    fallback = float(np.median(pre_positive)) if pre_positive.size else 0.004
    blend_rel = monotonic_x_from_values(trans_x[trans_blend], fallback_step=fallback)
    blend_x = float(pre_x[-1] + fallback) + blend_rel
    append_blend(buffers, trans_rel, trans_blend, ramp_rel, ramp_blend, q_keys=q_keys, x_values=blend_x)
    post_rel = monotonic_x_from_values(ramp_x[ramp_post], fallback_step=fallback)
    post_start = float(blend_x[-1] + fallback)
    append_source(buffers, ramp_rel, ramp_post, q_keys=q_keys, x_values=post_start + post_rel)

    output = finalize_series(buffers, transition.sample_rate)
    slope = slope_value(transition.metadata)
    raw_transition = transition.metadata.get("raw_marker_transition", {})
    slope_start = float(raw_transition.get("transition_x_m", 0.2)) if isinstance(raw_transition, dict) else 0.2
    x_end = float(output["q_pelvis_tx"][-1])
    slope_end = x_end + 1.0
    if transition.label == "walk-rampascent":
        segments = [
            {"type": "flat", "x0": -20.0, "x1": slope_start, "height": 0.0},
            {"type": "slope", "x0": slope_start, "x1": slope_end, "height0": 0.0, "slope": slope},
            {"type": "flat", "x0": slope_end, "x1": slope_end + 20.0, "height": slope * (slope_end - slope_start)},
        ]
    else:
        height = slope * max(slope_end - slope_start, 0.1)
        segments = [
            {"type": "flat", "x0": -20.0, "x1": slope_start, "height": height},
            {"type": "slope", "x0": slope_start, "x1": slope_end, "height0": height, "slope": -slope},
            {"type": "flat", "x0": slope_end, "x1": slope_end + 20.0, "height": 0.0},
        ]
    diagnostics = {
        "transition_reference": transition.reference_id,
        "steady_reference": ramp.reference_id,
        "ramp_match_index": int(match_index),
        "ramp_match_time": float(match_index / ramp.sample_rate),
        "match_rmse": rmse,
        "blend_frames": int(len(trans_blend)),
        "slope_start_x": float(slope_start),
        "slope_end_x": float(slope_end),
        "raw_marker_transition": raw_transition,
    }
    return output, segments, diagnostics


def build_ramp_to_level(
    *,
    transition: Reference,
    flat_segments: list[FlatSegment],
    q_keys: list[str],
    blend_frames: int,
    flat_after_frames: int,
    base_config: dict[str, Any],
    render_args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], list[dict[str, float | str]], dict[str, Any]]:
    trans_rel = height_relative_series(transition.series, transition.metadata)
    target = feature_at(transition.series, len(transition.series["q_pelvis_tx"]) - 1)
    flat, match_index, rmse = find_flat_match(flat_segments, transition.subject, target, before=blend_frames, after=flat_after_frames + 1)
    flat_rel = height_relative_series(flat.series, flat.metadata)

    n_trans = len(transition.series["q_pelvis_tx"])
    trans_pre = np.arange(0, n_trans - blend_frames, dtype=np.int64)
    trans_blend = np.arange(n_trans - blend_frames, n_trans, dtype=np.int64)
    flat_blend = np.arange(match_index - blend_frames, match_index, dtype=np.int64)
    flat_post = np.arange(match_index, match_index + flat_after_frames, dtype=np.int64)

    buffers = {key: [] for key in q_keys}
    trans_x = np.asarray(transition.series["q_pelvis_tx"], dtype=np.float64)
    flat_x = np.asarray(flat.series["q_pelvis_tx"], dtype=np.float64)
    pre_x = monotonic_x_from_values(trans_x[trans_pre], fallback_step=0.004)
    append_source(buffers, trans_rel, trans_pre, q_keys=q_keys, x_values=pre_x)
    pre_positive = np.diff(pre_x)
    pre_positive = pre_positive[pre_positive > 1e-6]
    fallback = float(np.median(pre_positive)) if pre_positive.size else 0.004
    blend_rel = monotonic_x_from_values(trans_x[trans_blend], fallback_step=fallback)
    blend_x = float(pre_x[-1] + fallback) + blend_rel
    append_blend(buffers, trans_rel, trans_blend, flat_rel, flat_blend, q_keys=q_keys, x_values=blend_x)
    post_rel = monotonic_x_from_values(flat_x[flat_post], fallback_step=fallback)
    x_after_start = float(blend_x[-1] + fallback)
    append_source(buffers, flat_rel, flat_post, q_keys=q_keys, x_values=x_after_start + post_rel)

    output = finalize_series(buffers, transition.sample_rate)
    slope = slope_value(transition.metadata)
    inferred = infer_exit_transition_x_from_foot(
        npz_path=transition.path,
        metadata=transition.metadata,
        base_config=base_config,
        args=render_args,
    )
    boundary = float(inferred["transition_x"] + render_args.foot_support_margin_m)
    x_end = float(output["q_pelvis_tx"][-1])
    if transition.label == "rampascent-walk":
        height = slope * boundary
        segments = [
            {"type": "slope", "x0": 0.0, "x1": boundary, "height0": 0.0, "slope": slope},
            {"type": "flat", "x0": boundary, "x1": x_end + 20.0, "height": height},
        ]
    else:
        height = slope * boundary
        segments = [
            {"type": "slope", "x0": 0.0, "x1": boundary, "height0": height, "slope": -slope},
            {"type": "flat", "x0": boundary, "x1": x_end + 20.0, "height": 0.0},
        ]
    diagnostics = {
        "flat_trial": flat.trial,
        "flat_match_index": int(match_index),
        "flat_match_time": float(match_index / flat.sample_rate),
        "match_rmse": rmse,
        "blend_frames": int(blend_frames),
        "flat_after_frames": int(flat_after_frames),
        "inferred_exit": inferred,
    }
    return output, segments, diagnostics


def metadata_for_output(
    *,
    name: str,
    source: Reference,
    series: dict[str, np.ndarray],
    segments: list[dict[str, float | str]],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "sample_rate": float(source.sample_rate),
        "data_length": int(len(next(iter(series.values())))),
        "source_dataset": "Camargo",
        "source_subject": source.subject,
        "source_mode": "stitched_preview",
        "source_trial": source.metadata.get("source_trial", ""),
        "source_label": name,
        "terrain_id": "levelwalking",
        "terrain_type": "flat",
        "terrain_params": "",
        "terrain_course_segments": segments,
        "stitch_diagnostics": diagnostics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camargo-dir", type=Path, default=DEFAULT_CAMARGO_DIR)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--rawmarker-entry-manifest", type=Path, default=None)
    parser.add_argument("--steady-ramp-manifest", type=Path, default=DEFAULT_STEADY_RAMP_MANIFEST)
    parser.add_argument("--rawmarker-rampascent-id", type=str, default=DEFAULT_RAWMARKER_UP_ID)
    parser.add_argument("--rawmarker-rampdescent-id", type=str, default=DEFAULT_RAWMARKER_DOWN_ID)
    parser.add_argument("--steady-rampascent-id", type=str, default=DEFAULT_STEADY_UP_ID)
    parser.add_argument("--steady-rampdescent-id", type=str, default=DEFAULT_STEADY_DOWN_ID)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--flat-trials", type=int, default=5)
    parser.add_argument("--blend-sec", type=float, default=0.2)
    parser.add_argument("--pre-flat-sec", type=float, default=1.0)
    parser.add_argument("--post-flat-sec", type=float, default=2.0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--frames", type=int, default=240)
    args = parser.parse_args()

    base_config = json.loads(args.base_config.read_text(encoding="utf-8"))
    render_args = argparse.Namespace(
        slope_start_x=10.0,
        after_flat_m=30.0,
        foot_support_margin_m=0.45,
        steady_slope_margin_m=1.2,
        infer_exit_platform_from_foot=True,
        foot_platform_tail_fraction=0.45,
        foot_platform_z_percentile=35.0,
        foot_platform_z_margin=0.04,
        foot_platform_speed_threshold=0.7,
        foot_platform_edge_backoff_m=0.2,
        hfield_size_z_min=1.0,
        hfield_size_z_margin=0.2,
        max_course_vertical_correction=1.5,
        course_correction_smoothing_window=1,
        max_course_vertical_correction_step=0.0,
    )

    records = []
    npz_dir = args.outdir / "npz"

    if args.rawmarker_entry_manifest is not None:
        entry_by_id = manifest_by_id(args.rawmarker_entry_manifest)
        steady_by_id = manifest_by_id(args.steady_ramp_manifest)
        refs = {
            "level_to_rampascent": reference_from_manifest_record(entry_by_id[args.rawmarker_rampascent_id]),
            "level_to_rampdescent": reference_from_manifest_record(entry_by_id[args.rawmarker_rampdescent_id]),
            "steady_rampascent": reference_from_manifest_record(steady_by_id[args.steady_rampascent_id]),
            "steady_rampdescent": reference_from_manifest_record(steady_by_id[args.steady_rampdescent_id]),
        }
        q_keys = sorted(set.intersection(*(set(key for key in ref.series if key.startswith("q_")) for ref in refs.values())))
        q_keys = ["q_pelvis_tx", *[key for key in q_keys if key != "q_pelvis_tx"]]
        blend_frames = max(2, int(round(args.blend_sec * refs["level_to_rampascent"].sample_rate)))
        build_specs = [
            (
                "stitched_level_to_rampascent_rawmarker",
                refs["level_to_rampascent"],
                refs["steady_rampascent"],
            ),
            (
                "stitched_level_to_rampdescent_rawmarker",
                refs["level_to_rampdescent"],
                refs["steady_rampdescent"],
            ),
        ]
        for name, transition, steady in build_specs:
            series, segments, diagnostics = build_level_to_ramp_from_transition(
                transition=transition,
                ramp=steady,
                q_keys=q_keys,
                blend_frames=blend_frames,
            )
            metadata = metadata_for_output(name=name, source=transition, series=series, segments=segments, diagnostics=diagnostics)
            out_path = (npz_dir / f"{name}.npz").resolve()
            save_reference_npz(out_path, series, float(transition.sample_rate), metadata)
            records.append(
                {
                    "reference_id": name,
                    "label": name,
                    "subject": transition.subject,
                    "npz_path": str(out_path),
                    "frames": int(metadata["data_length"]),
                    "sample_rate": float(transition.sample_rate),
                    "duration_sec": float(metadata["data_length"] / transition.sample_rate),
                    "diagnostics": diagnostics,
                    "terrain_course_segments": segments,
                }
            )
            print(
                f"[stitch] {name}: {out_path} "
                f"match_rmse={diagnostics['match_rmse']:.4f} slope_end={diagnostics['slope_end_x']:.3f}"
            )
        source_manifest = str(args.rawmarker_entry_manifest)
    else:
        by_id = manifest_by_id(args.source_manifest)
        ids = {
            "level_to_rampascent": "camargo_ab08_slopeascent_rampascent_ramp_6_l_01_05",
            "level_to_rampdescent": "camargo_ab06_slopedescent_rampdescent_ramp_6_l_01_05",
            "rampdescent_to_level": "camargo_ab06_slopedescent_rampdescent-walk_ramp_6_l_01_03",
            "rampascent_to_level": "camargo_ab08_slopeascent_rampascent-walk_ramp_6_l_01_05",
        }
        refs = {name: reference_from_manifest_record(by_id[ref_id]) for name, ref_id in ids.items()}
        flat_segments = load_flat_segments(args.camargo_dir, "AB06", args.flat_trials) + load_flat_segments(
            args.camargo_dir, "AB08", args.flat_trials
        )
        q_keys = sorted(
            set.intersection(*(set(key for key in ref.series if key.startswith("q_")) for ref in refs.values()))
            & set.intersection(*(set(key for key in seg.series if key.startswith("q_")) for seg in flat_segments))
        )
        q_keys = ["q_pelvis_tx", *[key for key in q_keys if key != "q_pelvis_tx"]]

        blend_frames = max(2, int(round(args.blend_sec * refs["level_to_rampascent"].sample_rate)))
        pre_frames = max(blend_frames + 1, int(round(args.pre_flat_sec * refs["level_to_rampascent"].sample_rate)))
        post_frames = max(blend_frames + 1, int(round(args.post_flat_sec * refs["level_to_rampascent"].sample_rate)))
        build_specs = [
            ("stitched_level_to_rampascent", refs["level_to_rampascent"], build_level_to_ramp),
            ("stitched_level_to_rampdescent", refs["level_to_rampdescent"], build_level_to_ramp),
            ("stitched_rampdescent_to_level", refs["rampdescent_to_level"], build_ramp_to_level),
            ("stitched_rampascent_to_level", refs["rampascent_to_level"], build_ramp_to_level),
        ]
        for name, source, builder in build_specs:
            if builder is build_level_to_ramp:
                series, segments, diagnostics = builder(
                    ramp=source,
                    flat_segments=flat_segments,
                    q_keys=q_keys,
                    blend_frames=blend_frames,
                    pre_frames=pre_frames,
                )
            else:
                series, segments, diagnostics = builder(
                    transition=source,
                    flat_segments=flat_segments,
                    q_keys=q_keys,
                    blend_frames=blend_frames,
                    flat_after_frames=post_frames,
                    base_config=base_config,
                    render_args=render_args,
                )
            metadata = metadata_for_output(name=name, source=source, series=series, segments=segments, diagnostics=diagnostics)
            out_path = npz_dir / f"{name}.npz"
            out_path = out_path.resolve()
            save_reference_npz(out_path, series, float(source.sample_rate), metadata)
            records.append(
                {
                    "reference_id": name,
                    "label": name,
                    "subject": source.subject,
                    "npz_path": str(out_path),
                    "frames": int(metadata["data_length"]),
                    "sample_rate": float(source.sample_rate),
                    "duration_sec": float(metadata["data_length"] / source.sample_rate),
                    "diagnostics": diagnostics,
                    "terrain_course_segments": segments,
                }
            )
            print(f"[stitch] {name}: {out_path} match_rmse={diagnostics['match_rmse']:.4f}")
        source_manifest = str(args.source_manifest)

    out_manifest = {
        "source_manifest": source_manifest,
        "steady_ramp_manifest": str(args.steady_ramp_manifest) if args.rawmarker_entry_manifest is not None else "",
        "flat_trials_per_subject": int(args.flat_trials),
        "blend_sec": float(args.blend_sec),
        "pre_flat_sec": float(args.pre_flat_sec),
        "post_flat_sec": float(args.post_flat_sec),
        "record_count": len(records),
        "records": records,
    }
    manifest_path = args.outdir / "manifest.json"
    write_json(manifest_path, out_manifest)

    if args.render:
        video_dir = args.outdir / "videos"
        rendered = []
        render_ns = argparse.Namespace(
            outdir=video_dir,
            frames=int(args.frames),
            width=1280,
            height=720,
            fps=30,
            camera_distance=6.0,
            camera_height=0.9,
            **vars(render_args),
        )
        for index, record in enumerate(records, start=1):
            print(f"[render] {index}/{len(records)} {record['reference_id']}")
            rendered.append(render_one(base_config, record, render_ns))
        write_json(
            video_dir / "video_manifest.json",
            {
                "source_manifest": str(manifest_path),
                "ok_count": len(rendered),
                "error_count": 0,
                "records": rendered,
            },
        )
    print(f"[manifest] {manifest_path}")


if __name__ == "__main__":
    main()
