#!/usr/bin/env python3
"""Render the projected course22 reference on its exact box terrain."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import mujoco
import numpy as np

from video_ffmpeg import (
    close_rgb_h264_writer,
    open_rgb_h264_writer,
    write_rgb_frame,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = ROOT / "reference_exports/course22_v1/course22_v1.npz"
DEFAULT_TERRAIN = (
    ROOT / "reference_exports/course22_v1/terrain_course22_include.xml"
)
DEFAULT_XML = (
    ROOT.parent / "myoassist/models/22muscle_2D/myoLeg22_2D_BASELINE.xml"
)
DEFAULT_OUTPUT = (
    ROOT / "reference_exports/course22_v1/videos/course22_v1_side.mp4"
)


def model_with_terrain(source_xml: Path, terrain_include: Path) -> mujoco.MjModel:
    xml = source_xml.read_text(encoding="utf-8")
    old_include = '<include file="../terrain_config.xml"/>'
    if old_include not in xml:
        raise ValueError(f"terrain include not found in {source_xml}")
    xml = xml.replace(
        old_include,
        f'<include file="{terrain_include.resolve()}"/>',
        1,
    )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".xml",
            prefix=".course22_render_",
            dir=source_xml.parent,
            encoding="utf-8",
            delete=False,
        ) as handle:
            handle.write(xml)
            temporary_path = Path(handle.name)
        return mujoco.MjModel.from_xml_path(str(temporary_path))
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--terrain", type=Path, default=DEFAULT_TERRAIN)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=272)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--camera-distance", type=float, default=2.8)
    parser.add_argument("--camera-height-offset", type=float, default=-0.05)
    parser.add_argument("--azimuth", type=float, default=90.0)
    parser.add_argument("--elevation", type=float, default=-8.0)
    parser.add_argument("--encoder", default="auto")
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    raw = np.load(args.reference.resolve(), allow_pickle=True)
    metadata = dict(raw["metadata"].item())
    series = dict(raw["series_data"].item())
    qpos = np.asarray(series["qpos_full"], dtype=np.float64)
    frames = len(qpos)
    if int(args.max_frames) > 0:
        frames = min(frames, int(args.max_frames))
    if qpos.shape[1] != 53:
        raise ValueError(f"expected qpos_full width 53, got {qpos.shape}")

    model = model_with_terrain(args.xml.resolve(), args.terrain.resolve())
    if model.nq != qpos.shape[1]:
        raise ValueError(f"model nq={model.nq}, reference width={qpos.shape[1]}")
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(
        model,
        height=int(args.height),
        width=int(args.width),
    )
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = float(args.camera_distance)
    camera.azimuth = float(args.azimuth)
    camera.elevation = float(args.elevation)
    pelvis_tx = int(
        model.jnt_qposadr[
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, "pelvis_tx"
            )
        ]
    )
    pelvis_ty = int(
        model.jnt_qposadr[
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, "pelvis_ty"
            )
        ]
    )

    writer = None
    writer_info = None
    try:
        writer, writer_info = open_rgb_h264_writer(
            args.output.resolve(),
            width=int(args.width),
            height=int(args.height),
            fps=int(args.fps),
            encoder=str(args.encoder),
        )
        for frame in range(frames):
            data.qpos[:] = qpos[frame]
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            camera.lookat[:] = [
                float(data.qpos[pelvis_tx]),
                0.0,
                float(data.qpos[pelvis_ty])
                + float(args.camera_height_offset),
            ]
            renderer.update_scene(data, camera=camera)
            write_rgb_frame(writer, renderer.render())
        close_rgb_h264_writer(writer)
        writer = None
    finally:
        if writer is not None:
            close_rgb_h264_writer(writer)
        renderer.close()

    summary = {
        "video": str(args.output.resolve()),
        "reference": str(args.reference.resolve()),
        "terrain": str(args.terrain.resolve()),
        "model": str(args.xml.resolve()),
        "frames": int(frames),
        "duration_sec": float(frames / int(args.fps)),
        "resolution": [int(args.width), int(args.height)],
        "fps": int(args.fps),
        "camera": {
            "azimuth": float(args.azimuth),
            "elevation": float(args.elevation),
            "distance": float(args.camera_distance),
        },
        "encoder": writer_info,
        "reference_variant": metadata.get("variant"),
    }
    summary_path = args.output.with_suffix(".summary.json").resolve()
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
