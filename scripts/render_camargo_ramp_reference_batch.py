#!/usr/bin/env python3
"""Render Camargo ramp references on matching slope terrain courses."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from cleanrl.ppo_muscle_mjwarp import build_muscle_model, load_reference, load_reference_from_config  # noqa: E402
from render_mjwarp_reference_terrain import render_reference_clip  # noqa: E402

DEFAULT_MANIFEST = ROOT / "results" / "camargo_ab06_ab08_ramp6x5_retargeted" / "manifest.json"
DEFAULT_BASE_CONFIG = ROOT / "configs" / "muscle_2d_mjwarp_stageF_rampascent_only_h192_terrainpreview32_from409k_sac.json"
DEFAULT_OUTDIR = ROOT / "results" / "camargo_ab06_ab08_ramp6x5_retargeted" / "videos"


def load_npz_metadata(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        return dict(data["metadata"].item())


def slope_metadata_params(metadata: dict[str, Any]) -> tuple[float, float]:
    values = [float(item) for item in str(metadata.get("terrain_params", "") or "").split()]
    slope = float(values[0]) if values else 0.0
    anchor = float(values[2]) if len(values) >= 3 else 0.0
    return slope, anchor


def last_contiguous_true(mask: np.ndarray, min_len: int = 2) -> np.ndarray:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return indices
    breaks = np.flatnonzero(np.diff(indices) > 1)
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks + 1, indices.size]
    for start, end in zip(starts[::-1], ends[::-1]):
        run = indices[start:end]
        if run.size >= min_len:
            return run
    return indices[-min(min_len, indices.size) :]


def infer_exit_transition_x_from_foot(
    *,
    npz_path: Path,
    metadata: dict[str, Any],
    base_config: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, float]:
    temp_config = json.loads(json.dumps(base_config))
    temp_config["terrain_course"] = {"enabled": False, "segments": []}
    model, _ = build_muscle_model(temp_config)
    ref = load_reference(
        npz_path,
        model,
        float(temp_config["control"]["control_hz"]),
        torch.device("cpu"),
        temp_config,
    )
    foot_rel = ref["foot_site_ref"].detach().cpu().numpy().astype(np.float64)
    pelvis_x = ref["pelvis_tx_ref"].detach().cpu().numpy().astype(np.float64)
    speed = ref["foot_speed_ref"].detach().cpu().numpy().astype(np.float64)
    foot_x = foot_rel[:, :, 0] + pelvis_x[:, None]
    foot_z = foot_rel[:, :, 2]
    min_site = np.argmin(foot_z, axis=1)
    rows = np.arange(foot_z.shape[0])
    min_z = foot_z[rows, min_site]
    min_x = foot_x[rows, min_site]
    min_speed = speed[rows, min_site]

    tail_start = int(round(float(args.foot_platform_tail_fraction) * max(foot_z.shape[0] - 1, 1)))
    tail_mask = rows >= tail_start
    tail_z = min_z[tail_mask]
    if tail_z.size == 0:
        tail_z = min_z
        tail_mask = np.ones_like(rows, dtype=bool)
    z_gate = float(np.percentile(tail_z, float(args.foot_platform_z_percentile))) + float(args.foot_platform_z_margin)
    support_mask = tail_mask & (min_z <= z_gate) & (min_speed <= float(args.foot_platform_speed_threshold))
    support_run = last_contiguous_true(support_mask, min_len=2)
    if support_run.size == 0:
        support_run = np.argsort(min_z[tail_mask])[: max(2, min(6, int(tail_z.size)))]
        support_run = np.flatnonzero(tail_mask)[support_run]

    plateau_z = float(np.median(min_z[support_run]))
    plateau_x = float(np.median(min_x[support_run]))
    signed_slope, anchor = slope_metadata_params(metadata)
    signed_slope = float(signed_slope)
    slope_abs = max(abs(signed_slope), 1e-6)
    forward_distance = max(float(metadata.get("root_transform", {}).get("forward_distance", 0.0)), 0.0)

    early_mask = rows <= int(round(0.60 * max(foot_z.shape[0] - 1, 1)))
    early_z_gate = float(np.percentile(min_z[early_mask], 45.0)) + max(float(args.foot_platform_z_margin), 0.04)
    early_support = early_mask & (min_z <= early_z_gate) & (min_speed <= float(args.foot_platform_speed_threshold))
    if np.count_nonzero(early_support) >= 2 and abs(signed_slope) > 1e-6:
        slope_intercept = float(np.median(min_z[early_support] - signed_slope * min_x[early_support]))
        transition_x = (plateau_z - slope_intercept) / signed_slope
    else:
        terrain_id = str(metadata.get("terrain_id", "") or "")
        if terrain_id == "slopeascent":
            transition_x = plateau_z / slope_abs
        elif terrain_id == "slopedescent":
            transition_x = (anchor - plateau_z) / slope_abs
        else:
            transition_x = plateau_x
        slope_intercept = anchor
    support_edge_x = plateau_x - float(args.foot_platform_edge_backoff_m)
    transition_x = max(float(transition_x), support_edge_x)
    transition_x = float(np.clip(transition_x, 0.2, max(forward_distance + 0.5, 0.5)))
    return {
        "transition_x": transition_x,
        "height_intersection_x": float((plateau_z - slope_intercept) / signed_slope)
        if abs(signed_slope) > 1e-6
        else float(transition_x),
        "support_edge_x": float(support_edge_x),
        "plateau_x": plateau_x,
        "plateau_z": plateau_z,
        "slope_intercept_z": float(slope_intercept),
        "support_start_frame": int(support_run[0]),
        "support_end_frame": int(support_run[-1]),
        "support_frames": int(support_run.size),
        "early_support_frames": int(np.count_nonzero(early_support)),
    }


def slope_course(
    metadata: dict[str, Any],
    *,
    x0: float,
    x_after: float,
    foot_support_margin_m: float,
    steady_slope_margin_m: float,
    inferred_exit: dict[str, float] | None = None,
) -> tuple[list[dict[str, float | str]], dict[str, float]]:
    angle = abs(float(metadata.get("ramp_incline_deg", 0.0)))
    slope = math.tan(math.radians(angle))
    forward_distance = float(metadata.get("root_transform", {}).get("forward_distance", 0.0))
    distance = max(forward_distance, 1.0)
    terrain_id = str(metadata.get("terrain_id", ""))
    label = str(metadata.get("source_label", "") or "")
    if "walk" in label:
        margin = max(float(foot_support_margin_m), 0.0)
    else:
        margin = max(float(steady_slope_margin_m), float(foot_support_margin_m), 0.0)
    shifts = {
        "slopeascent_entry_shift": margin,
        "slopeascent_exit_shift": -(distance + margin),
        "slopedescent_entry_shift": margin,
        "slopedescent_exit_shift": -(distance + margin),
    }
    if inferred_exit is not None and label.endswith("-walk"):
        transition_x = float(inferred_exit["transition_x"])
        slope_len = max(transition_x + margin, 0.2)
        shifts[f"{terrain_id}_exit_shift"] = -transition_x
    else:
        slope_len = distance + 2.0 * margin
    x_slope0 = x0
    x_slope1 = x_slope0 + slope_len
    height = slope * slope_len
    if terrain_id == "slopeascent":
        return (
            [
                {"type": "flat", "x0": -20.0, "x1": x_slope0, "height": 0.0},
                {"type": "slope", "x0": x_slope0, "x1": x_slope1, "height0": 0.0, "slope": slope},
                {"type": "flat", "x0": x_slope1, "x1": x_slope1 + x_after, "height": height},
            ],
            shifts,
        )
    if terrain_id == "slopedescent":
        return (
            [
                {"type": "flat", "x0": -20.0, "x1": x_slope0, "height": height},
                {"type": "slope", "x0": x_slope0, "x1": x_slope1, "height0": height, "slope": -slope},
                {"type": "flat", "x0": x_slope1, "x1": x_slope1 + x_after, "height": 0.0},
            ],
            shifts,
        )
    raise ValueError(f"Unsupported terrain_id for ramp render: {terrain_id}")


def config_for_reference(base_config: dict[str, Any], npz_path: Path, metadata: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = json.loads(json.dumps(base_config))
    config["reference_pool"] = {"paths": [str(npz_path)]}
    config["reference_pool_schedule"] = []
    config["video"] = {"phase_indices": [0]}

    course = dict(config.get("terrain_course", {}))
    label = str(metadata.get("source_label", "") or "")
    inferred_exit = None
    metadata_segments = metadata.get("terrain_course_segments")
    if isinstance(metadata_segments, list) and metadata_segments:
        segments = metadata_segments
        shifts = {
            "slopeascent_entry_shift": 0.0,
            "slopeascent_exit_shift": 0.0,
            "slopedescent_entry_shift": 0.0,
            "slopedescent_exit_shift": 0.0,
        }
        metadata_shifts = metadata.get("terrain_course_shifts")
        if isinstance(metadata_shifts, dict):
            for key in shifts:
                if key in metadata_shifts:
                    shifts[key] = float(metadata_shifts[key])
    elif bool(args.infer_exit_platform_from_foot) and label.endswith("-walk"):
        inferred_exit = infer_exit_transition_x_from_foot(
            npz_path=npz_path,
            metadata=metadata,
            base_config=base_config,
            args=args,
        )
        segments, shifts = slope_course(
            metadata,
            x0=float(args.slope_start_x),
            x_after=float(args.after_flat_m),
            foot_support_margin_m=float(args.foot_support_margin_m),
            steady_slope_margin_m=float(args.steady_slope_margin_m),
            inferred_exit=inferred_exit,
        )
    else:
        segments, shifts = slope_course(
            metadata,
            x0=float(args.slope_start_x),
            x_after=float(args.after_flat_m),
            foot_support_margin_m=float(args.foot_support_margin_m),
            steady_slope_margin_m=float(args.steady_slope_margin_m),
            inferred_exit=inferred_exit,
        )
    course.update({"enabled": True, "segments": segments, **shifts})
    xs = []
    for segment in segments:
        xs.extend([float(segment.get("x0", 0.0)), float(segment.get("x1", 0.0))])
    if xs:
        sample_x = np.linspace(min(xs), max(xs), 512, dtype=np.float64)
        max_height = float(np.max(course_height_for_render(sample_x, segments)))
        course["hfield_size_z"] = max(float(args.hfield_size_z_min), max_height + float(args.hfield_size_z_margin))
    if inferred_exit is not None:
        course["inferred_exit_platform"] = inferred_exit
    config["terrain_course"] = course

    contact = dict(config.get("reference_contact", {}))
    contact.setdefault("course_clearance_target", 0.0)
    contact.setdefault("course_min_clearance", 0.0)
    contact["max_course_vertical_correction"] = float(args.max_course_vertical_correction)
    contact["course_correction_smoothing_window"] = int(args.course_correction_smoothing_window)
    contact["max_course_vertical_correction_step"] = float(args.max_course_vertical_correction_step)
    config["reference_contact"] = contact
    return config


def course_height_for_render(x: np.ndarray, segments: list[dict[str, Any]]) -> np.ndarray:
    height = np.zeros_like(x, dtype=np.float64)
    for segment in segments:
        x0 = float(segment.get("x0", -np.inf))
        x1 = float(segment.get("x1", np.inf))
        mask = (x >= x0) & (x <= x1)
        kind = str(segment.get("type", "flat"))
        if kind == "flat":
            height[mask] = float(segment.get("height", 0.0))
        elif kind == "slope":
            height[mask] = float(segment.get("height0", 0.0)) + float(segment.get("slope", 0.0)) * (x[mask] - x0)
    return height


def move_render_outputs(src_video: Path, final_video: Path) -> dict[str, str]:
    final_video.parent.mkdir(parents=True, exist_ok=True)
    src_clearance = src_video.with_name(src_video.stem + "_clearance.csv")
    src_summary = src_video.with_name(src_video.stem + "_summary.json")
    final_clearance = final_video.with_name(final_video.stem + "_clearance.csv")
    final_summary = final_video.with_name(final_video.stem + "_summary.json")
    for src, dst in ((src_video, final_video), (src_clearance, final_clearance), (src_summary, final_summary)):
        if dst.exists():
            dst.unlink()
        shutil.move(str(src), str(dst))
    return {
        "video_path": str(final_video),
        "clearance_path": str(final_clearance),
        "summary_path": str(final_summary),
    }


def render_one(base_config: dict[str, Any], record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    npz_path = Path(record["npz_path"])
    if not npz_path.is_absolute():
        npz_path = ROOT / npz_path
    metadata = load_npz_metadata(npz_path)
    config = config_for_reference(base_config, npz_path, metadata, args)
    model, _ = build_muscle_model(config)
    reference = load_reference_from_config(
        Path("/home/lzn/myoassist/rl_train/reference_data/short_reference_gait.npz"),
        model,
        float(config["control"]["control_hz"]),
        torch.device("cpu"),
        config,
    )

    slug = str(record["reference_id"])
    tmp_dir = Path(args.outdir) / "_tmp" / slug
    tmp_dir.mkdir(parents=True, exist_ok=True)
    src_video = render_reference_clip(
        config=config,
        reference=reference,
        outdir=tmp_dir,
        start_phase=0,
        max_frames=int(args.frames),
        width=int(args.width),
        height=int(args.height),
        fps=int(args.fps),
        camera_distance=float(args.camera_distance),
        camera_height=float(args.camera_height),
    )
    final_video = Path(args.outdir) / f"reference_{slug}.mp4"
    paths = move_render_outputs(src_video, final_video)
    try:
        tmp_dir.rmdir()
    except OSError:
        pass
    return {**record, **paths, "render_status": "ok"}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--camera-distance", type=float, default=6.0)
    parser.add_argument("--camera-height", type=float, default=0.9)
    parser.add_argument("--slope-start-x", type=float, default=10.0)
    parser.add_argument("--after-flat-m", type=float, default=30.0)
    parser.add_argument("--foot-support-margin-m", type=float, default=0.45)
    parser.add_argument("--steady-slope-margin-m", type=float, default=1.2)
    parser.add_argument("--hfield-size-z-min", type=float, default=1.0)
    parser.add_argument("--hfield-size-z-margin", type=float, default=0.2)
    parser.add_argument("--infer-exit-platform-from-foot", action="store_true")
    parser.add_argument("--foot-platform-tail-fraction", type=float, default=0.45)
    parser.add_argument("--foot-platform-z-percentile", type=float, default=35.0)
    parser.add_argument("--foot-platform-z-margin", type=float, default=0.04)
    parser.add_argument("--foot-platform-speed-threshold", type=float, default=0.7)
    parser.add_argument("--foot-platform-edge-backoff-m", type=float, default=0.2)
    parser.add_argument("--max-course-vertical-correction", type=float, default=1.5)
    parser.add_argument("--course-correction-smoothing-window", type=int, default=1)
    parser.add_argument("--max-course-vertical-correction-step", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    base_config = json.loads(args.base_config.read_text(encoding="utf-8"))
    records = list(manifest.get("records", []))
    if args.limit:
        records = records[: int(args.limit)]

    rendered: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        print(f"[render] {index}/{len(records)} {record['reference_id']}")
        try:
            rendered.append(render_one(base_config, record, args))
        except Exception as exc:  # keep the rest of the preview batch useful
            rendered.append({**record, "render_status": "error", "error": repr(exc)})
            print(f"[error] {record['reference_id']}: {exc!r}")

    ok_count = sum(1 for item in rendered if item.get("render_status") == "ok")
    out_manifest = {
        "source_manifest": str(args.manifest),
        "base_config": str(args.base_config),
        "render_count": len(rendered),
        "ok_count": ok_count,
        "error_count": len(rendered) - ok_count,
        "records": rendered,
    }
    output_path = Path(args.outdir) / "video_manifest.json"
    write_json(output_path, out_manifest)
    print(f"[done] rendered {ok_count}/{len(rendered)} videos")
    print(f"[manifest] {output_path}")


if __name__ == "__main__":
    main()
