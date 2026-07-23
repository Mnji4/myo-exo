#!/usr/bin/env python3
"""Generate unlimited box/ramp/stair MJCF terrain from a course80 reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NPZ = ROOT / "reference_exports/course80_3d_balanced_v8/course80_3d_balanced_v8.npz"
DEFAULT_XML = ROOT.parent / "myoassist/models/80muscle/myoLeg80_HMEDI/myolegs_HMEDI.xml"
FRICTION = [1.0, 0.005, 0.0001]


def add_box(
    specs: list[dict[str, Any]],
    name: str,
    *,
    y0: float,
    y1: float,
    top_z: float,
    half_width: float,
    rgba: str,
    min_thickness: float = 0.06,
) -> None:
    thickness = max(float(top_z), float(min_thickness))
    specs.append(
        {
            "name": name,
            "pos": [0.0, 0.5 * (y0 + y1), top_z - 0.5 * thickness],
            "size": [half_width, 0.5 * abs(y1 - y0), 0.5 * thickness],
            "quat": [1.0, 0.0, 0.0, 0.0],
            "rgba": rgba,
        }
    )


def add_ramp(
    specs: list[dict[str, Any]],
    name: str,
    *,
    y0: float,
    y1: float,
    z0: float,
    z1: float,
    half_width: float,
    rgba: str,
    thickness: float = 0.14,
    overlap: float = 0.08,
) -> None:
    y0 -= overlap
    y1 += overlap
    horizontal = abs(y1 - y0)
    angle = math.atan2(z1 - z0, horizontal)
    length = horizontal / max(math.cos(angle), 1e-9)
    specs.append(
        {
            "name": name,
            "pos": [
                0.0,
                0.5 * (y0 + y1) + 0.5 * thickness * math.sin(angle),
                0.5 * (z0 + z1) - 0.5 * thickness * math.cos(angle),
            ],
            "size": [half_width, 0.5 * length, 0.5 * thickness],
            "quat": [math.cos(0.5 * angle), math.sin(0.5 * angle), 0.0, 0.0],
            "rgba": rgba,
        }
    )


def terrain_specs(
    segments: list[dict[str, Any]], *, half_width: float
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    colors = {
        "low": "0.62 0.74 0.88 1",
        "ramp": "0.86 0.70 0.32 1",
        "high": "0.74 0.84 0.62 1",
        "top": "0.12 0.42 0.86 1",
        "step_a": "0.92 0.56 0.12 1",
        "step_b": "0.22 0.66 0.30 1",
    }
    for segment_index, segment in enumerate(segments):
        kind = str(segment.get("type", ""))
        if kind == "flat_box":
            height = float(segment["height"])
            color = (
                colors["low"]
                if height < 1e-6
                else (colors["top"] if height > 1.0 else colors["high"])
            )
            add_box(
                specs,
                f"terrain_flat_{segment_index:02d}",
                y0=float(segment["x0"]),
                y1=float(segment["x1"]),
                top_z=height,
                half_width=half_width,
                rgba=color,
            )
        elif kind == "ramp_box":
            add_ramp(
                specs,
                f"terrain_ramp_{segment_index:02d}",
                y0=float(segment["x0"]),
                y1=float(segment["x1"]),
                z0=float(segment["height0"]),
                z1=float(segment["height1"]),
                half_width=half_width,
                rgba=colors["ramp"],
            )
        elif kind == "stairs_box":
            start = float(segment["x0"])
            depth = float(segment["step_depth"])
            height = float(segment["step_height"])
            base = float(segment.get("base_height", 0.0))
            direction = int(segment.get("direction", 1))
            for step in range(int(segment["steps"])):
                y0 = start + step * depth
                top_z = (
                    base + (step + 1) * height
                    if direction > 0
                    else base - (step + 1) * height
                )
                add_box(
                    specs,
                    f"terrain_stair_{segment_index:02d}_{step + 1:02d}",
                    y0=y0,
                    y1=y0 + depth,
                    top_z=top_z,
                    half_width=half_width,
                    rgba=colors["step_a"] if step % 2 == 0 else colors["step_b"],
                )
        else:
            raise ValueError(f"unsupported terrain segment type: {kind!r}")
    return specs


def write_terrain_include(path: Path, specs: list[dict[str, Any]]) -> None:
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
            "pos": "0 0 0",
            "mode": "trackcom",
        },
    )
    ET.SubElement(
        world,
        "geom",
        {
            "name": "ground-plane",
            "type": "plane",
            "pos": "0 0 -0.001",
            "size": "3 80 0.01",
            "rgba": "0.35 0.35 0.34 0.10",
            "conaffinity": "0",
            "contype": "0",
        },
    )
    for spec in specs:
        ET.SubElement(
            world,
            "geom",
            {
                "name": str(spec["name"]),
                "type": "box",
                "pos": " ".join(f"{value:.10f}" for value in spec["pos"]),
                "size": " ".join(f"{value:.10f}" for value in spec["size"]),
                "quat": " ".join(f"{value:.10f}" for value in spec["quat"]),
                "rgba": str(spec["rgba"]),
                "friction": " ".join(str(value) for value in FRICTION),
                "conaffinity": "1",
                "contype": "1",
            },
        )
    ET.indent(root, space="  ")
    path.write_text(ET.tostring(root, encoding="unicode") + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_contract(source_xml: Path) -> dict:
    if not source_xml.is_file():
        raise FileNotFoundError(
            f"myoLeg80_HMEDI XML not found: {source_xml}; pass --source-xml explicitly"
        )
    model = mujoco.MjModel.from_xml_path(str(source_xml))
    joint_types = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
    joints = []
    for index in range(model.njnt):
        joints.append(
            {
                "id": index,
                "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index),
                "type": joint_types[int(model.jnt_type[index])],
                "qpos_address": int(model.jnt_qposadr[index]),
                "dof_address": int(model.jnt_dofadr[index]),
            }
        )
    model_root = source_xml.parents[3]
    dependency_paths = {
        "model_xml": source_xml,
        "leg_chain": source_xml.parent / "assets/myolegs_chain_HMEDI.xml",
        "leg_assets": source_xml.parent / "assets/myolegs_assets_HMEDI.xml",
        "torso_chain": model_root / "myosuite/simhive/myo_sim/torso/assets/myotorso_rigid_chain.xml",
        "torso_assets": model_root / "myosuite/simhive/myo_sim/torso/assets/myotorso_rigid_assets.xml",
    }
    return {
        "name": "myoLeg80_HMEDI",
        "expected_repository_path": "models/80muscle/myoLeg80_HMEDI/myolegs_HMEDI.xml",
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "njnt": int(model.njnt),
        "joints": joints,
        "actuator_names": [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
            for index in range(model.nu)
        ],
        "dependency_files": {
            name: {
                "repository_path": str(path.relative_to(model_root)),
                "sha256": sha256(path),
            }
            for name, path in dependency_paths.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate portable MuJoCo box terrain and an integration contract."
    )
    parser.add_argument("--reference", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--source-xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--half-width", type=float, default=1.25)
    args = parser.parse_args()

    if not args.reference.is_file():
        parser.error(f"reference not found: {args.reference}")
    if args.half_width <= 0:
        parser.error("--half-width must be positive")
    reference = args.reference.resolve()
    source_xml = args.source_xml.resolve()
    raw = np.load(reference, allow_pickle=True)
    metadata = raw["metadata"].item()
    series = raw["series_data"].item()
    qpos = np.asarray(series["qpos_full"])
    qvel = np.asarray(series["qvel_full"])
    model_info = model_contract(source_xml)
    if qpos.ndim != 2 or qpos.shape[1] != model_info["nq"]:
        raise ValueError(
            f"qpos_full shape {qpos.shape} does not match model nq={model_info['nq']}"
        )
    if qvel.ndim != 2 or qvel.shape != (qpos.shape[0], model_info["nv"]):
        raise ValueError(
            f"qvel_full shape {qvel.shape} does not match "
            f"({qpos.shape[0]}, model nv={model_info['nv']})"
        )
    if int(metadata["data_length"]) != qpos.shape[0]:
        raise ValueError(
            f"metadata data_length={metadata['data_length']} does not match "
            f"reference frames={qpos.shape[0]}"
        )
    segments = list(metadata["terrain_course_segments"])
    outdir = (args.outdir or reference.parent / "integration").resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    include_path = outdir / "terrain_course_include.xml"
    specs = terrain_specs(segments, half_width=float(args.half_width))
    write_terrain_include(include_path, specs)

    contract = {
        "schema_version": 2,
        "reference": Path(os.path.relpath(reference, start=outdir)).as_posix(),
        "reference_sha256": sha256(reference),
        "terrain_include": include_path.name,
        "coordinate_contract": {
            "model_forward_axis": "+y",
            "metadata_legacy_axis_names": "segment x0/x1 are positions along model world +y",
            "vertical_axis": "+z",
            "lateral_axis": "+x",
        },
        "geometry": {
            "implementation": "compile-time MuJoCo box geoms only; no hfield",
            "half_width_m": float(args.half_width),
            "geom_count": len(specs),
            "friction": FRICTION,
            "segments": segments,
        },
        "integration": {
            "required_reference_arrays": ["series_data.qpos_full", "series_data.qvel_full"],
            "sample_rate_hz": float(metadata["sample_rate"]),
            "frame_count": int(qpos.shape[0]),
            "qpos_shape": list(qpos.shape),
            "qvel_shape": list(qvel.shape),
            "root_qpos_layout": [
                "lateral_x",
                "forward_y",
                "vertical_z",
                "qw",
                "qx",
                "qy",
                "qz",
            ],
            "compile_order": (
                "Generate/include terrain geoms before MjModel compilation; "
                "do not mutate an hfield after compile."
            ),
            "trainer_adapter": (
                "The trainer must load its own local myoLeg80_HMEDI XML, include "
                "equivalent generated geoms, and use root y as course progress."
            ),
        },
        "model_contract": model_info,
    }
    contract_path = outdir / "terrain_contract.json"
    contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "terrain_include": str(include_path),
                "contract": str(contract_path),
                "geom_count": len(specs),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
