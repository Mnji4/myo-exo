#!/usr/bin/env python3
"""Render reset-state and reference-only probes for a configured MJWarp terrain."""

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
    load_config,
    load_reference_from_config,
    reference_phase_label,
    set_cpu_reference_state,
)


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
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


def diagnostics_row(model: mujoco.MjModel, data: mujoco.MjData, config: dict, reference: dict, phase: int, frame: int) -> dict[str, object]:
    pelvis_tx_qpos = int(model.jnt_qposadr[joint_id(model, "pelvis_tx")])
    pelvis_ty_qpos = int(model.jnt_qposadr[joint_id(model, "pelvis_ty")])
    terrain_height = float(course_height_np(np.array([float(data.qpos[pelvis_tx_qpos])]), list(config.get("terrain_course", {}).get("segments", [])))[0])
    row: dict[str, object] = {
        "frame": int(frame),
        "phase": int(phase),
        "label": reference_phase_label(reference, int(phase)),
        "pelvis_tx": float(data.qpos[pelvis_tx_qpos]),
        "pelvis_ty": float(data.qpos[pelvis_ty_qpos]),
        "terrain_height": terrain_height,
        "pelvis_height_above_terrain": float(data.qpos[pelvis_ty_qpos]) - terrain_height,
        "ncon": int(data.ncon),
    }
    for name in FOOT_SITE_NAMES:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        x = float(data.site_xpos[site_id, 0])
        z = float(data.site_xpos[site_id, 2])
        foot_terrain = float(course_height_np(np.array([x]), list(config.get("terrain_course", {}).get("segments", [])))[0])
        row[f"{name}_x"] = x
        row[f"{name}_z"] = z
        row[f"{name}_terrain_height"] = foot_terrain
        row[f"{name}_clearance"] = z - foot_terrain
    return row


def render_probe(
    *,
    config: dict,
    reference: dict,
    outdir: Path,
    phase: int,
    mode: str,
    frames: int,
    width: int,
    height: int,
    camera_distance: float,
    camera_height: float,
    fps: int,
) -> dict[str, object]:
    model, data = build_muscle_model(config)
    outdir.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, height=int(height), width=int(width))
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = float(camera_distance)
    camera.azimuth = 90.0
    camera.elevation = -8.0
    pelvis_tx_qpos = int(model.jnt_qposadr[joint_id(model, "pelvis_tx")])
    label = reference_phase_label(reference, int(phase))
    video_path = outdir / f"{mode}_{label}.mp4"
    diag_path = outdir / f"{mode}_{label}_diagnostics.csv"
    frame_count = int(frames)
    if mode == "reference":
        frame_count = max(1, min(frame_count, containing_reference_end(reference, int(phase)) - int(phase)))

    rows: list[dict[str, object]] = []
    frames_written = 0
    writer = imageio.get_writer(video_path, fps=int(fps))
    try:
        for frame in range(frame_count):
            render_phase = int(phase) if mode == "reset" else (int(phase) + frame) % int(reference["length"])
            set_cpu_reference_state(model, data, reference, render_phase)
            camera.lookat[:] = [float(data.qpos[pelvis_tx_qpos]), 0.0, float(camera_height)]
            renderer.update_scene(data, camera=camera)
            writer.append_data(renderer.render())
            frames_written += 1
            rows.append(diagnostics_row(model, data, config, reference, render_phase, frame))
    finally:
        writer.close()
        renderer.close()

    write_rows(diag_path, rows)
    return {
        "mode": mode,
        "phase": int(phase),
        "label": label,
        "video": str(video_path),
        "diagnostics": str(diag_path),
        "frames": frames_written,
        "first_pelvis_height_above_terrain": rows[0]["pelvis_height_above_terrain"] if rows else None,
        "min_pelvis_height_above_terrain": min(float(row["pelvis_height_above_terrain"]) for row in rows) if rows else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--phase", type=int, required=True)
    parser.add_argument("--reference", type=Path, default=Path("/home/lzn/myoassist/rl_train/reference_data/short_reference_gait.npz"))
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--reset-frames", type=int, default=60)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--camera-distance", type=float, default=5.0)
    parser.add_argument("--camera-height", type=float, default=0.9)
    parser.add_argument("--mode", choices=["both", "reset", "reference"], default="both")
    args = parser.parse_args()

    config = load_config(args.config)
    model, _ = build_muscle_model(config)
    reference = load_reference_from_config(args.reference, model, float(config["control"]["control_hz"]), torch.device("cpu"), config)
    rows = []
    if args.mode in {"both", "reset"}:
        rows.append(render_probe(
            config=config,
            reference=reference,
            outdir=args.outdir,
            phase=int(args.phase),
            mode="reset",
            frames=int(args.reset_frames),
            width=int(args.width),
            height=int(args.height),
            camera_distance=float(args.camera_distance),
            camera_height=float(args.camera_height),
            fps=int(args.fps),
        ))
    if args.mode in {"both", "reference"}:
        rows.append(render_probe(
            config=config,
            reference=reference,
            outdir=args.outdir,
            phase=int(args.phase),
            mode="reference",
            frames=int(args.frames),
            width=int(args.width),
            height=int(args.height),
            camera_distance=float(args.camera_distance),
            camera_height=float(args.camera_height),
            fps=int(args.fps),
        ))
    print(json.dumps({"probes": rows}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
