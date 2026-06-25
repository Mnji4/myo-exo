#!/usr/bin/env python3
"""Render MJWarp training references on the configured terrain course."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cleanrl.ppo_muscle_mjwarp import (  # noqa: E402
    FOOT_SITE_NAMES,
    build_muscle_model,
    course_height_np,
    joint_id,
    load_reference_from_config,
    reference_phase_label,
    set_cpu_reference_state,
    stair_box_treads,
)


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
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


def containing_reference_end(reference: dict, phase: int) -> int:
    phase = int(phase) % int(reference["length"])
    for item in reference.get("reference_offsets", []):
        if int(item["start"]) <= phase < int(item["end"]):
            return int(item["end"])
    return int(reference["length"])


def merged_stair_surfaces(segments: list[dict]) -> list[tuple[str, float, float, float]]:
    surfaces: list[tuple[str, float, float, float]] = []
    for segment_index, segment in enumerate(segments):
        if str(segment.get("type", "flat")) != "stairs_box":
            continue
        for tread_index, (x0, x1, _height) in enumerate(stair_box_treads(segment)):
            surfaces.append((f"stairs{segment_index}_tread{tread_index}", float(x0), float(x1), float(_height)))
    surfaces.sort(key=lambda item: (item[3], item[1]))
    merged: list[tuple[str, float, float, float]] = []
    for name, x0, x1, height in surfaces:
        if merged and abs(merged[-1][3] - height) < 1e-6 and x0 <= merged[-1][2] + 1e-6:
            prev_name, prev_x0, prev_x1, prev_height = merged[-1]
            merged[-1] = (f"{prev_name}+{name}", prev_x0, max(prev_x1, x1), prev_height)
        else:
            merged.append((name, x0, x1, height))
    return sorted(merged, key=lambda item: item[1])


def stair_surface_at_x(x: float, surfaces: list[tuple[str, float, float, float]]) -> tuple[str, float | str]:
    for name, x0, x1, _height in surfaces:
        if float(x0) <= x <= float(x1):
            return name, float(min(x - x0, x1 - x))
    return "", ""


def clearance_summary(rows: list[dict[str, float | int | str]], z_threshold: float) -> dict[str, object]:
    summary: dict[str, object] = {"support_z_threshold": float(z_threshold), "feet": {}}
    pelvis_ty = [float(row["pelvis_ty"]) for row in rows]
    if len(pelvis_ty) > 1:
        summary["max_pelvis_ty_frame_delta"] = float(np.max(np.abs(np.diff(np.asarray(pelvis_ty, dtype=np.float64)))))
    else:
        summary["max_pelvis_ty_frame_delta"] = 0.0
    feet: dict[str, object] = {}
    for name in FOOT_SITE_NAMES:
        support_rows = [row for row in rows if float(row[f"{name}_clearance"]) < float(z_threshold)]
        margins = [
            float(row[f"{name}_edge_margin"])
            for row in support_rows
            if row.get(f"{name}_edge_margin", "") != ""
        ]
        clearances = [float(row[f"{name}_clearance"]) for row in rows]
        feet[name] = {
            "support_frames": len(support_rows),
            "surfaces": sorted({str(row.get(f"{name}_surface", "")) for row in support_rows if row.get(f"{name}_surface", "")}),
            "min_edge_margin": float(min(margins)) if margins else None,
            "min_clearance": float(min(clearances)) if clearances else None,
            "max_clearance": float(max(clearances)) if clearances else None,
        }
    summary["feet"] = feet
    return summary


def render_reference_clip(
    *,
    config: dict,
    reference: dict,
    outdir: Path,
    start_phase: int,
    max_frames: int,
    width: int,
    height: int,
    fps: int,
    camera_distance: float,
    camera_height: float,
    label_override: str | None = None,
) -> Path:
    model, data = build_muscle_model(config)
    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = float(camera_distance)
    camera.azimuth = 90.0
    camera.elevation = -8.0

    site_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in FOOT_SITE_NAMES]
    pelvis_tx_qpos = int(model.jnt_qposadr[joint_id(model, "pelvis_tx")])
    pelvis_ty_qpos = int(model.jnt_qposadr[joint_id(model, "pelvis_ty")])
    segments = list(config.get("terrain_course", {}).get("segments", []))
    stair_surfaces = merged_stair_surfaces(segments)

    start_phase = int(start_phase) % int(reference["length"])
    end_phase = containing_reference_end(reference, start_phase)
    frame_count = max(1, min(int(max_frames), max(1, end_phase - start_phase)))
    label = label_override or reference_phase_label(reference, start_phase)
    video_path = outdir / f"reference_{label}.mp4"
    diag_path = outdir / f"reference_{label}_clearance.csv"
    summary_path = outdir / f"reference_{label}_summary.json"

    frames = []
    rows: list[dict[str, float | int | str]] = []
    try:
        for frame in range(frame_count):
            phase = (start_phase + frame) % int(reference["length"])
            set_cpu_reference_state(model, data, reference, phase)
            camera.lookat[:] = [float(data.qpos[pelvis_tx_qpos]), 0.0, float(camera_height)]
            renderer.update_scene(data, camera=camera)
            frames.append(renderer.render())

            row: dict[str, float | int | str] = {
                "frame": frame,
                "phase": phase,
                "pelvis_tx": float(data.qpos[pelvis_tx_qpos]),
                "pelvis_ty": float(data.qpos[pelvis_ty_qpos]),
                "ncon": int(data.ncon),
            }
            for name, site_id in zip(FOOT_SITE_NAMES, site_ids):
                x = float(data.site_xpos[site_id, 0])
                z = float(data.site_xpos[site_id, 2])
                terrain_z = float(course_height_np(np.array([x], dtype=np.float64), segments)[0])
                row[f"{name}_x"] = x
                row[f"{name}_z"] = z
                row[f"{name}_terrain_z"] = terrain_z
                row[f"{name}_clearance"] = z - terrain_z
                surface, edge_margin = stair_surface_at_x(x, stair_surfaces)
                row[f"{name}_surface"] = surface
                row[f"{name}_edge_margin"] = edge_margin
            rows.append(row)
    finally:
        renderer.close()

    outdir.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(video_path, frames, fps=fps)
    write_csv(diag_path, rows)
    summary = clearance_summary(rows, float(config.get("reference_contact", {}).get("z_threshold", 0.025)))
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return video_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--phase", type=int, action="append", default=None)
    parser.add_argument("--frames", type=int, default=96)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--camera-distance", type=float, default=8.5)
    parser.add_argument("--camera-height", type=float, default=0.9)
    parser.add_argument("--transitions-from-summary", type=Path, default=None)
    parser.add_argument("--transition-frames", type=int, default=90)
    parser.add_argument("--transition-before-frames", type=int, default=45)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    model, _ = build_muscle_model(config)
    reference = load_reference_from_config(
        Path("/home/lzn/myoassist/rl_train/reference_data/short_reference_gait.npz"),
        model,
        float(config["control"]["control_hz"]),
        torch.device("cpu"),
        config,
    )
    transition_specs = []
    if args.transitions_from_summary is not None:
        summary = json.loads(args.transitions_from_summary.read_text(encoding="utf-8"))
        for record in summary.get("join_records", []):
            boundary = int(record["boundary"])
            source = str(record.get("source", "join"))
            start_phase = max(0, boundary - int(args.transition_before_frames))
            transition_specs.append(
                {
                    "phase": start_phase,
                    "frames": int(args.transition_frames),
                    "label": f"transition_{source}_boundary{boundary:04d}_phase{start_phase:04d}",
                }
            )
    phases = args.phase or list(config.get("video", {}).get("phase_indices", []))
    if not phases:
        phases = [int(item["start"]) for item in reference.get("reference_offsets", [])]

    written = []
    summaries = []
    render_specs = transition_specs or [{"phase": int(phase), "frames": int(args.frames), "label": None} for phase in phases]
    for spec in render_specs:
        video_path = render_reference_clip(
            config=config,
            reference=reference,
            outdir=args.outdir,
            start_phase=int(spec["phase"]),
            max_frames=int(spec["frames"]),
            width=int(args.width),
            height=int(args.height),
            fps=int(args.fps),
            camera_distance=float(args.camera_distance),
            camera_height=float(args.camera_height),
            label_override=spec["label"],
        )
        written.append(str(video_path))
        summaries.append(str(video_path.with_name(video_path.stem + "_summary.json")))
    print(json.dumps({"videos": written, "summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()
