#!/usr/bin/env python3
"""Build a continuous flat-uphill-flat reference for Stage-G review/training."""

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

from cleanrl.ppo_muscle_mjwarp import RESET_JOINTS, build_muscle_model, load_config  # noqa: E402
from scripts.build_stageG_long_course_reference import (  # noqa: E402
    GAP_M,
    LEAD_LEVEL_CYCLES,
    TRANSITION_BLEND_FRAMES,
    add_derivatives,
    append_clip,
    extract_cyclic_segment,
    find_level_loop_segment,
    load_relative_q_series,
)


SOURCES = {
    "level": Path(
        "/home/lzn/exoskeleton_terrain/data/camargo_reference_selected_contact_aligned/"
        "camargo_ab06_levelwalking_walk_selected_contact_aligned_myoassist_3d.npz"
    ),
    "level_to_up": ROOT
    / "data/camargo_transition_references/camargo_ab06_slopeascent_walk-rampascent_transition_myoassist_3d.npz",
    "uphill": Path(
        "/home/lzn/exoskeleton_terrain/data/camargo_reference_footlocked/"
        "camargo_ab06_slopeascent_rampascent_footlocked_myoassist_3d.npz"
    ),
    "up_to_level": ROOT
    / "data/camargo_transition_references/camargo_ab06_slopeascent_rampascent-walk_transition_myoassist_3d.npz",
}


def slope_bounds(segments: list[dict[str, Any]]) -> tuple[float, float, float]:
    for segment in segments:
        if str(segment.get("type")) == "slope" and float(segment.get("slope", 0.0)) > 0.0:
            return float(segment["x0"]), float(segment["x1"]), float(segment.get("slope", 0.0))
    raise ValueError("terrain_course must contain a positive uphill slope segment")


def label_ranges(labels: list[str]) -> dict[str, dict[str, int]]:
    ranges: dict[str, dict[str, int]] = {}
    cursor = 0
    while cursor < len(labels):
        label = labels[cursor]
        end = cursor + 1
        while end < len(labels) and labels[end] == label:
            end += 1
        ranges[label] = {"start": int(cursor), "end": int(end)}
        cursor = end
    return ranges


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/stageG_uphill9_footlocked035_clip_staged/muscle_2d_mjwarp_stageG_up9fl035_C_h36_imit_sac.json",
    )
    parser.add_argument("--outdir", type=Path, default=ROOT / "reference_exports/stageG_flat_up_flat_reference")
    parser.add_argument("--sample-rate", type=float, default=30.0)
    parser.add_argument("--uphill-cycles", type=int, default=2)
    parser.add_argument("--tail-level-cycles", type=int, default=2)
    args = parser.parse_args()

    config = load_config(args.config)
    course_segments = list(config.get("terrain_course", {}).get("segments", []))
    up_x0, up_x1, slope = slope_bounds(course_segments)
    model, _ = build_muscle_model(config)
    loaded = {
        name: load_relative_q_series(
            path,
            model=model,
            control_hz=float(args.sample_rate),
            config=config,
            course_segments=course_segments,
        )
        for name, path in SOURCES.items()
    }
    rel = {name: series for name, (_metadata, series) in loaded.items()}

    level_loop_metadata = find_level_loop_segment(rel["level"])
    rel["level_loop"] = extract_cyclic_segment(
        rel["level"],
        start_idx=int(level_loop_metadata["start"]),
        frames=int(level_loop_metadata["frames"]),
    )
    uphill_loop_metadata = find_level_loop_segment(rel["uphill"], min_frames=min(45, len(rel["uphill"]["q_pelvis_tx"])))
    rel["uphill_loop"] = extract_cyclic_segment(
        rel["uphill"],
        start_idx=int(uphill_loop_metadata["start"]),
        frames=int(uphill_loop_metadata["frames"]),
    )

    q_keys = [f"q_{joint}" for joint in RESET_JOINTS]
    buffers = {key: [] for key in q_keys}
    aux_buffers: dict[str, list[np.ndarray]] = {"foot_rel_x": [], "foot_rel_z": [], "foot_contact": []}
    labels: list[str] = []
    join_records: list[dict[str, Any]] = []

    level_x = np.asarray(rel["level_loop"]["q_pelvis_tx"], dtype=np.float64)
    level_stride = float(level_x[-1] - level_x[0])
    level_frames = len(level_x)
    full_lead_cycles = int(np.floor(LEAD_LEVEL_CYCLES))
    partial_lead_fraction = float(LEAD_LEVEL_CYCLES - full_lead_cycles)
    desired_level_to_up_start = float(up_x0 + float(config.get("terrain_course", {}).get("slopeascent_entry_shift", 0.0)))
    lead_span = LEAD_LEVEL_CYCLES * level_stride + np.ceil(LEAD_LEVEL_CYCLES) * GAP_M
    level_start_x = desired_level_to_up_start - lead_span

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
            max_frames=max(1, int(round(level_frames * partial_lead_fraction))),
            match_join=True,
            cyclic=True,
            labels=labels,
        )

    append_clip(
        buffers,
        aux_buffers,
        rel["level_to_up"],
        q_keys=q_keys,
        source_name="level_to_up",
        target_start_x=max(desired_level_to_up_start, float(buffers["q_pelvis_tx"][-1]) + GAP_M),
        course_segments=course_segments,
        sample_rate=float(args.sample_rate),
        join_records=join_records,
        search_max_fraction=1.0,
        labels=labels,
    )
    for cycle in range(max(0, int(args.uphill_cycles))):
        append_clip(
            buffers,
            aux_buffers,
            rel["uphill_loop"],
            q_keys=q_keys,
            source_name=f"uphill_steady_{cycle + 1}",
            target_start_x=float(buffers["q_pelvis_tx"][-1]) + GAP_M,
            course_segments=course_segments,
            sample_rate=float(args.sample_rate),
            join_records=join_records,
            match_join=True,
            cyclic=True,
            max_frames=len(np.asarray(rel["uphill_loop"]["q_pelvis_tx"])),
            search_max_fraction=1.0,
            labels=labels,
        )

    if float(buffers["q_pelvis_tx"][-1]) + 2.0 * GAP_M < float(up_x1):
        append_clip(
            buffers,
            aux_buffers,
            rel["uphill_loop"],
            q_keys=q_keys,
            source_name="uphill_approach_top",
            target_start_x=float(buffers["q_pelvis_tx"][-1]) + GAP_M,
            course_segments=course_segments,
            sample_rate=float(args.sample_rate),
            join_records=join_records,
            match_join=True,
            cyclic=True,
            max_frames=len(np.asarray(rel["uphill_loop"]["q_pelvis_tx"])),
            max_x=float(up_x1) - 0.5 * GAP_M,
            search_max_fraction=1.0,
            labels=labels,
        )

    append_clip(
        buffers,
        aux_buffers,
        rel["up_to_level"],
        q_keys=q_keys,
        source_name="up_to_level",
        target_start_x=max(float(up_x1), float(buffers["q_pelvis_tx"][-1]) + GAP_M),
        course_segments=course_segments,
        sample_rate=float(args.sample_rate),
        join_records=join_records,
        search_max_fraction=1.0,
        labels=labels,
    )
    for cycle in range(max(0, int(args.tail_level_cycles))):
        append_clip(
            buffers,
            aux_buffers,
            rel["level_loop"],
            q_keys=q_keys,
            source_name=f"level_high_{cycle + 1}",
            target_start_x=float(buffers["q_pelvis_tx"][-1]) + GAP_M,
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
    ranges = label_ranges(labels)

    args.outdir.mkdir(parents=True, exist_ok=True)
    npz_path = args.outdir / "stageG_flat_up_flat_9deg_myoassist_3d.npz"
    metadata = {
        "source_mode": "stageG_flat_up_flat_concat",
        "terrain_id": "stageG_flat_up_flat_9deg",
        "source_label": "flat-up-flat",
        "sample_rate": float(args.sample_rate),
        "data_length": int(len(next(iter(series.values())))),
        "terrain_course_segments": course_segments,
        "slope": float(slope),
        "segment_labels": labels,
        "label_ranges": ranges,
        "lead_level_cycles_float": float(LEAD_LEVEL_CYCLES),
        "uphill_cycles": int(args.uphill_cycles),
        "tail_level_cycles": int(args.tail_level_cycles),
        "transition_blend_frames": int(TRANSITION_BLEND_FRAMES),
        "level_loop": level_loop_metadata,
        "uphill_loop": uphill_loop_metadata,
        "join_records": join_records,
        "source_paths": {name: str(path) for name, path in SOURCES.items()},
    }
    np.savez_compressed(npz_path, metadata=metadata, series_data=series)

    review_config = json.loads(json.dumps(config))
    review_config["experiment_name"] = "muscle_2d_mjwarp_stageG_flat_up_flat_9deg_reference_review"
    review_config["reference_pool"] = {"paths": [str(npz_path.resolve())]}
    review_config["reference_pool_schedule"] = []
    key_phases = [ranges[name]["start"] for name in ranges]
    review_config["video"] = {"phase_indices": key_phases, "phase_mode": "round_robin"}
    review_config.setdefault("checkpoint_video_export", {})["enabled"] = False
    config_path = args.outdir / "muscle_2d_mjwarp_stageG_flat_up_flat_9deg_reference_review.json"
    config_path.write_text(json.dumps(review_config, indent=2) + "\n", encoding="utf-8")
    summary_path = args.outdir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "npz": str(npz_path),
                "config": str(config_path),
                "metadata": metadata,
                "key_phases": key_phases,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"npz": str(npz_path), "config": str(config_path), "key_phases": key_phases}, indent=2), flush=True)


if __name__ == "__main__":
    main()
