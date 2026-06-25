#!/usr/bin/env python3
"""Select Camargo ramp transition references using raw foot markers."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TERRAIN_SCRIPTS = Path("/home/lzn/exoskeleton_terrain/scripts")
if str(TERRAIN_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(TERRAIN_SCRIPTS))

from convert_camargo_ab06_to_myoassist_reference import (  # noqa: E402
    SegmentSpec,
    best_label_run,
    camargo_ik_to_myoassist_series,
    decode_matlab_table,
    decode_plain_mat,
    infer_sample_rate,
    save_reference_npz,
    summarize_series,
    table_segment,
    terrain_from_segment,
)

DEFAULT_CAMARGO_DIR = Path("/mnt/c/Users/liang/Desktop/camargo")
DEFAULT_OUTDIR = ROOT / "results" / "camargo_raw_marker_transition_candidates"

FOOT_MARKERS = (
    "R_Heel",
    "R_Toe_Tip",
    "R_Toe_Med",
    "R_Toe_Lat",
    "L_Heel",
    "L_Toe_Tip",
    "L_Toe_Med",
    "L_Toe_Lat",
)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def find_ramp_dir(camargo_dir: Path, subject: str) -> Path:
    matches = sorted((camargo_dir / subject.upper()).glob("*/ramp"))
    if not matches:
        raise FileNotFoundError(f"Could not find {subject}/DATE/ramp under {camargo_dir}")
    return matches[0]


def terrain_id_for_label(label: str) -> str:
    if "ascent" in label:
        return "slopeascent"
    if "descent" in label:
        return "slopedescent"
    raise ValueError(f"Unsupported transition label: {label}")


def signed_slope_for_label(label: str, incline_deg: float) -> float:
    sign = 1.0 if "ascent" in label else -1.0
    return math.tan(math.radians(sign * abs(float(incline_deg))))


def segment_heading(ik_segment: dict[str, np.ndarray]) -> tuple[float, float, float, float]:
    raw_x = np.asarray(ik_segment["pelvis_tx"], dtype=np.float64)
    raw_z = np.asarray(ik_segment["pelvis_tz"], dtype=np.float64)
    x0 = float(raw_x[0])
    z0 = float(raw_z[0])
    dx = float(raw_x[-1] - raw_x[0])
    dz = float(raw_z[-1] - raw_z[0])
    norm = float(math.hypot(dx, dz))
    if norm < 1e-8:
        return x0, z0, 1.0, 0.0
    return x0, z0, dx / norm, dz / norm


def project_forward(raw_x: np.ndarray, raw_z: np.ndarray, x0: float, z0: float, ux: float, uz: float) -> np.ndarray:
    return (np.asarray(raw_x, dtype=np.float64) - x0) * ux + (np.asarray(raw_z, dtype=np.float64) - z0) * uz


def fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]
    y = y[good]
    if x.size < 5 or float(np.ptp(x)) < 1e-8:
        return float("nan"), float("nan"), float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    return float(slope), float(intercept), float(np.sqrt(np.mean(residual * residual))), float(np.max(np.abs(residual)))


def marker_support_series(
    marker_segment: dict[str, np.ndarray],
    ik_segment: dict[str, np.ndarray],
) -> dict[str, np.ndarray | float]:
    n = len(np.asarray(marker_segment["Header"]))
    x0, z0, ux, uz = segment_heading(ik_segment)
    forward: list[np.ndarray] = []
    vertical: list[np.ndarray] = []
    marker_names: list[str] = []
    for marker in FOOT_MARKERS:
        keys = (f"{marker}_x", f"{marker}_y", f"{marker}_z")
        if not all(key in marker_segment for key in keys):
            continue
        forward.append(project_forward(marker_segment[keys[0]], marker_segment[keys[2]], x0, z0, ux, uz))
        vertical.append(np.asarray(marker_segment[keys[1]], dtype=np.float64))
        marker_names.append(marker)
    if not forward:
        raise ValueError("No usable foot markers in marker segment")
    marker_x = np.stack(forward, axis=1)
    marker_y = np.stack(vertical, axis=1)
    support_count = min(3, marker_x.shape[1])
    support_idx = np.argsort(marker_y, axis=1)[:, :support_count]
    rows = np.arange(n)[:, None]
    support_x = marker_x[rows, support_idx]
    support_y = marker_y[rows, support_idx]
    return {
        "support_x": support_x,
        "support_y": support_y,
        "support_x_mean": np.mean(support_x, axis=1),
        "support_y_mean": np.mean(support_y, axis=1),
        "marker_x_span": float(np.nanmax(marker_x) - np.nanmin(marker_x)),
        "marker_y_span": float(np.nanmax(marker_y) - np.nanmin(marker_y)),
    }


def support_fit_metrics(support: dict[str, np.ndarray | float], target_slope: float) -> dict[str, float]:
    sx = np.asarray(support["support_x"], dtype=np.float64).reshape(-1)
    sy = np.asarray(support["support_y"], dtype=np.float64).reshape(-1)
    n = int(np.asarray(support["support_x_mean"]).shape[0])
    k = max(10, n // 3)
    all_fit = fit_line(sx, sy)
    first_fit = fit_line(sx[: 3 * k], sy[: 3 * k])
    last_fit = fit_line(sx[-3 * k :], sy[-3 * k :])

    def ratio(value: float) -> float:
        return float(value / target_slope) if abs(target_slope) > 1e-9 else float("nan")

    return {
        "all_slope": all_fit[0],
        "all_intercept": all_fit[1],
        "all_rmse_raw": all_fit[2],
        "all_maxerr_raw": all_fit[3],
        "first_slope": first_fit[0],
        "first_intercept": first_fit[1],
        "first_rmse_raw": first_fit[2],
        "first_maxerr_raw": first_fit[3],
        "last_slope": last_fit[0],
        "last_intercept": last_fit[1],
        "last_rmse_raw": last_fit[2],
        "last_maxerr_raw": last_fit[3],
        "all_ratio": ratio(all_fit[0]),
        "first_ratio": ratio(first_fit[0]),
        "last_ratio": ratio(last_fit[0]),
        "marker_x_span_raw": float(support["marker_x_span"]),
        "marker_y_span_raw": float(support["marker_y_span"]),
    }


def local_slope_ratios(support: dict[str, np.ndarray | float], target_slope: float, window: int) -> np.ndarray:
    x = np.asarray(support["support_x_mean"], dtype=np.float64)
    y = np.asarray(support["support_y_mean"], dtype=np.float64)
    out = np.full((x.shape[0],), np.nan, dtype=np.float64)
    radius = max(3, int(window) // 2)
    for index in range(x.shape[0]):
        lo = max(0, index - radius)
        hi = min(x.shape[0], index + radius + 1)
        slope = fit_line(x[lo:hi], y[lo:hi])[0]
        if np.isfinite(slope) and abs(target_slope) > 1e-9:
            out[index] = slope / target_slope
    return out


def estimate_transition_frame(
    support: dict[str, np.ndarray | float],
    target_slope: float,
    *,
    window: int,
    threshold: float,
    consecutive: int,
) -> int:
    ratios = local_slope_ratios(support, target_slope, window)
    n = int(ratios.shape[0])
    start = max(0, int(round(0.15 * n)))
    end = max(start + 1, int(round(0.85 * n)))
    active = np.abs(ratios) >= float(threshold)
    run = 0
    for index in range(start, end):
        if bool(active[index]):
            run += 1
            if run >= consecutive:
                return int(index - run + 1)
        else:
            run = 0
    return int(round(0.33 * max(n - 1, 1)))


def marker_to_reference_scale(support: dict[str, np.ndarray | float], forward_distance: float) -> float:
    span = max(float(support["marker_x_span"]), 1e-9)
    if span > 20.0 and forward_distance < 20.0:
        return 0.001
    return 1.0


def source_and_course_segments(
    *,
    label: str,
    slope: float,
    transition_x: float,
    forward_distance: float,
    course_slope_x0: float,
    after_flat_m: float,
) -> tuple[list[dict[str, float | str]], list[dict[str, float | str]], dict[str, float]]:
    local_x0 = max(float(transition_x), 0.05)
    local_x1 = max(float(forward_distance), local_x0 + 0.2)
    slope_len = max(local_x1 - local_x0, 0.2)
    course_x0 = float(course_slope_x0)
    course_x1 = course_x0 + slope_len
    terrain_id = terrain_id_for_label(label)
    shifts = {
        "slopeascent_entry_shift": 0.0,
        "slopeascent_exit_shift": 0.0,
        "slopedescent_entry_shift": 0.0,
        "slopedescent_exit_shift": 0.0,
    }
    if terrain_id == "slopeascent":
        height = float(abs(slope) * slope_len)
        source = [
            {"type": "flat", "x0": -20.0, "x1": local_x0, "height": 0.0},
            {"type": "slope", "x0": local_x0, "x1": local_x1 + 0.5, "height0": 0.0, "slope": abs(slope)},
        ]
        course = [
            {"type": "flat", "x0": -20.0, "x1": course_x0, "height": 0.0},
            {"type": "slope", "x0": course_x0, "x1": course_x1, "height0": 0.0, "slope": abs(slope)},
            {"type": "flat", "x0": course_x1, "x1": course_x1 + after_flat_m, "height": height},
        ]
        shifts["slopeascent_entry_shift"] = -local_x0
        return source, course, shifts

    height = float(abs(slope) * slope_len)
    source = [
        {"type": "flat", "x0": -20.0, "x1": local_x0, "height": height},
        {"type": "slope", "x0": local_x0, "x1": local_x1 + 0.5, "height0": height, "slope": -abs(slope)},
    ]
    course = [
        {"type": "flat", "x0": -20.0, "x1": course_x0, "height": height},
        {"type": "slope", "x0": course_x0, "x1": course_x1, "height0": height, "slope": -abs(slope)},
        {"type": "flat", "x0": course_x1, "x1": course_x1 + after_flat_m, "height": 0.0},
    ]
    shifts["slopedescent_entry_shift"] = -local_x0
    return source, course, shifts


def score_candidate(label: str, metrics: dict[str, float]) -> float:
    first = abs(float(metrics["first_ratio"]))
    all_ratio = float(metrics["all_ratio"])
    last = float(metrics["last_ratio"])
    if "descent" in label:
        return first + abs(last - 1.0) + 0.25 * abs(all_ratio - 0.8)
    uphill_late = max(0.0, 0.35 - abs(last))
    return first + 0.75 * abs(all_ratio - 1.0) + 0.25 * uphill_late


def build_record(
    *,
    subject: str,
    ramp_dir: Path,
    condition_path: Path,
    label: str,
    outdir: Path,
    course_slope_x0: float,
    after_flat_m: float,
    transition_window: int,
    transition_threshold: float,
    transition_consecutive: int,
) -> dict[str, Any]:
    trial = condition_path.name
    ik_path = ramp_dir / "ik" / trial
    marker_path = ramp_dir / "markers" / trial
    conditions_labels = decode_matlab_table(condition_path.read_bytes())
    conditions_plain = decode_plain_mat(condition_path.read_bytes())
    ik = decode_matlab_table(ik_path.read_bytes())
    markers = decode_matlab_table(marker_path.read_bytes())
    start, end = best_label_run(conditions_labels["Label"], label)

    ik_segment = table_segment(ik, start, end)
    marker_segment = table_segment(markers, start, end)
    time = np.asarray(ik_segment["Header"], dtype=np.float64)
    series, transform = camargo_ik_to_myoassist_series(ik_segment)
    sample_rate = infer_sample_rate(time)

    incline = float(conditions_plain["rampIncline"])
    signed_slope = signed_slope_for_label(label, incline)
    support = marker_support_series(marker_segment, ik_segment)
    metrics = support_fit_metrics(support, signed_slope)
    transition_frame = estimate_transition_frame(
        support,
        signed_slope,
        window=transition_window,
        threshold=transition_threshold,
        consecutive=transition_consecutive,
    )
    forward_distance = max(float(transform.get("forward_distance", 0.0)), float(series["q_pelvis_tx"][-1]))
    scale = marker_to_reference_scale(support, forward_distance)
    support_x_mean = np.asarray(support["support_x_mean"], dtype=np.float64)
    transition_x_raw = float((support_x_mean[transition_frame] - support_x_mean[0]) * scale)
    pelvis_transition_x = float(series["q_pelvis_tx"][transition_frame])
    transition_x = float(
        np.clip(max(transition_x_raw, pelvis_transition_x), 0.05, max(forward_distance - 0.05, 0.06))
    )
    source_segments, course_segments, shifts = source_and_course_segments(
        label=label,
        slope=signed_slope,
        transition_x=transition_x,
        forward_distance=forward_distance,
        course_slope_x0=course_slope_x0,
        after_flat_m=after_flat_m,
    )

    terrain_id = terrain_id_for_label(label)
    spec = SegmentSpec(terrain_id=terrain_id, mode="ramp", trial=trial, label=label)
    terrain = terrain_from_segment(spec, conditions_plain, ik_segment, transform)
    trial_stem = condition_path.stem
    reference_id = f"camargo_{subject.lower()}_{terrain_id}_{label}_{trial_stem}_rawmarker"
    output_path = outdir / "npz" / f"{reference_id}_myoassist_3d.npz"
    metadata = save_reference_npz(
        output_path,
        series,
        sample_rate,
        {
            "source_dataset": "Camargo",
            "source_subject": subject.upper(),
            "source_mode": "ramp",
            "source_trial": trial,
            "source_label": label,
            "terrain_id": terrain_id,
            "source_segment_start_index": int(start),
            "source_segment_end_index": int(end),
            "source_time_start": float(time[0]),
            "source_time_end": float(time[-1]),
            "root_transform": transform,
            "ramp_id": int(trial_stem.split("_")[1]),
            "ramp_side": trial_stem.split("_")[2],
            "ramp_trial_stem": trial_stem,
            "raw_marker_transition": {
                **metrics,
                "score": score_candidate(label, metrics),
                "transition_frame": int(transition_frame),
                "transition_time_sec": float(transition_frame / sample_rate),
                "transition_x_m": transition_x,
                "transition_support_x_m": transition_x_raw,
                "transition_pelvis_x_m": pelvis_transition_x,
                "marker_to_reference_scale": float(scale),
            },
            "source_terrain_segments": source_segments,
            "terrain_course_segments": course_segments,
            "terrain_course_shifts": shifts,
            **terrain,
        },
    )
    duration = float(time[-1] - time[0]) if len(time) > 1 else 0.0
    return {
        "reference_id": reference_id,
        "subject": subject.upper(),
        "ramp_id": int(metadata["ramp_id"]),
        "side": str(metadata["ramp_side"]),
        "trial": trial,
        "label": label,
        "terrain_id": terrain_id,
        "ramp_incline_deg": float(metadata.get("ramp_incline_deg", 0.0)),
        "frames": int(metadata["data_length"]),
        "sample_rate": float(sample_rate),
        "duration_sec": duration,
        "forward_distance_m": forward_distance,
        "mean_forward_speed_mps": forward_distance / duration if duration > 0.0 else 0.0,
        "npz_path": str(output_path),
        "metadata": metadata,
        "raw_marker_transition": metadata["raw_marker_transition"],
        "source_terrain_segments": source_segments,
        "terrain_course_segments": course_segments,
        "terrain_course_shifts": shifts,
        "series_summary": summarize_series(series),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camargo-dir", type=Path, default=DEFAULT_CAMARGO_DIR)
    parser.add_argument("--subjects", nargs="+", default=["AB06", "AB08"])
    parser.add_argument("--labels", nargs="+", default=["walk-rampascent", "walk-rampdescent"])
    parser.add_argument("--ramp-ids", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--sides", nargs="+", choices=["l", "r"], default=["l", "r"])
    parser.add_argument("--top-per-label", type=int, default=5)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--all-candidates", type=Path, default=None)
    parser.add_argument("--course-slope-x0", type=float, default=10.0)
    parser.add_argument("--after-flat-m", type=float, default=30.0)
    parser.add_argument("--transition-window", type=int, default=31)
    parser.add_argument("--transition-threshold", type=float, default=0.35)
    parser.add_argument("--transition-consecutive", type=int, default=5)
    args = parser.parse_args()

    records: list[dict[str, Any]] = []
    for subject in args.subjects:
        ramp_dir = find_ramp_dir(args.camargo_dir, subject)
        for condition_path in sorted((ramp_dir / "conditions").glob("ramp_*.mat")):
            parts = condition_path.stem.split("_")
            if len(parts) < 4:
                continue
            if int(parts[1]) not in set(int(item) for item in args.ramp_ids):
                continue
            if parts[2] not in set(args.sides):
                continue
            marker_path = ramp_dir / "markers" / condition_path.name
            ik_path = ramp_dir / "ik" / condition_path.name
            if not marker_path.exists() or not ik_path.exists():
                continue
            for label in args.labels:
                try:
                    print(f"[scan] {subject} {condition_path.name} label={label}")
                    records.append(
                        build_record(
                            subject=subject,
                            ramp_dir=ramp_dir,
                            condition_path=condition_path,
                            label=label,
                            outdir=args.outdir,
                            course_slope_x0=float(args.course_slope_x0),
                            after_flat_m=float(args.after_flat_m),
                            transition_window=int(args.transition_window),
                            transition_threshold=float(args.transition_threshold),
                            transition_consecutive=int(args.transition_consecutive),
                        )
                    )
                except ValueError:
                    continue

    selected: list[dict[str, Any]] = []
    for label in args.labels:
        label_records = [record for record in records if record["label"] == label]
        label_records.sort(key=lambda item: float(item["raw_marker_transition"]["score"]))
        for rank, record in enumerate(label_records[: int(args.top_per_label)], start=1):
            record["selection_rank"] = rank
            selected.append(record)

    all_manifest = {
        "source": {
            "dataset": "Camargo",
            "camargo_dir": str(args.camargo_dir),
            "subjects": [str(item).upper() for item in args.subjects],
        },
        "selection": {
            "labels": list(args.labels),
            "ramp_ids": [int(item) for item in args.ramp_ids],
            "sides": list(args.sides),
            "top_per_label": int(args.top_per_label),
            "score": "raw marker support: flat first third plus later slope consistency",
        },
        "record_count": len(records),
        "records": records,
    }
    selected_manifest = {
        "source": all_manifest["source"],
        "selection": all_manifest["selection"],
        "record_count": len(selected),
        "records": selected,
    }
    all_path = args.all_candidates or args.outdir / "all_candidates_manifest.json"
    selected_path = args.manifest or args.outdir / "manifest.json"
    write_json(all_path, all_manifest)
    write_json(selected_path, selected_manifest)
    print(f"[done] scanned {len(records)} transition candidates")
    print(f"[selected] {len(selected)} records")
    print(f"[manifest] {selected_path}")
    print(f"[all] {all_path}")


if __name__ == "__main__":
    main()
