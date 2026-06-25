#!/usr/bin/env python3
"""Create configs for reviewing 9.207 degree uphill reference candidates."""

from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "results/stageG_long_course_reference/muscle_2d_mjwarp_stageG_long_course_gated_ref_sac.json"
OUTDIR = ROOT / "reference_exports/stageG_uphill9_reference_review"

SLOPE_9 = 0.16209004
SHIFT_VALUES = (0.35, 0.70, 1.05)

REFERENCES = {
    "steady_contact_aligned": Path(
        "/home/lzn/exoskeleton_terrain/data/camargo_reference_selected_contact_aligned/"
        "camargo_ab06_slopeascent_rampascent_selected_contact_aligned_myoassist_3d.npz"
    ),
    "steady_footlocked": Path(
        "/home/lzn/exoskeleton_terrain/data/camargo_reference_footlocked/"
        "camargo_ab06_slopeascent_rampascent_footlocked_myoassist_3d.npz"
    ),
    "steady_raw_200hz": Path(
        "/home/lzn/exoskeleton_terrain/data/camargo_reference/"
        "camargo_ab06_slopeascent_rampascent_myoassist_3d.npz"
    ),
    "walk_to_rampascent": ROOT
    / "data/camargo_transition_references/camargo_ab06_slopeascent_walk-rampascent_transition_myoassist_3d.npz",
}


def terrain_course(shift: float) -> dict:
    return {
        "enabled": True,
        "hfield_name": "terrain",
        "terrain_geom": "terrain",
        "terrain_geom_z": 0.0,
        "terrain_rgba": [1.0, 1.0, 1.0, 1.0],
        "lower_ground_plane": False,
        "ground_geom": "ground-plane",
        "ground_plane_alpha": 0.0,
        "ground_plane_z": -10.0,
        "hfield_size_z": 1.4,
        "levelwalking_reference_x_shift": 0.0,
        "slopeascent_entry_shift": float(shift),
        "segments": [
            {"type": "flat", "x0": -20.0, "x1": 0.0, "height": 0.0},
            {"type": "slope", "x0": 0.0, "x1": 8.0, "height0": 0.0, "slope": SLOPE_9},
            {"type": "flat", "x0": 8.0, "x1": 60.0, "height": 8.0 * SLOPE_9},
        ],
    }


def safe_name(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    base = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    configs: list[dict[str, str | float]] = []

    for ref_name, ref_path in REFERENCES.items():
        for shift in SHIFT_VALUES:
            config = copy.deepcopy(base)
            config["experiment_name"] = f"muscle_2d_mjwarp_stageG_uphill9_{ref_name}_shift_{safe_name(shift)}_review"
            config["reference_pool"] = {"paths": [str(ref_path.resolve())]}
            config["reference_pool_schedule"] = []
            config["video"] = {"phase_indices": [0]}
            config["checkpoint_video_export"]["phase_indices"] = [0]
            config["terrain_course"] = terrain_course(shift)
            config["reset"]["phase_windows"] = [{"start": 0, "end": 160}]

            path = OUTDIR / f"{config['experiment_name']}.json"
            path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            configs.append(
                {
                    "name": ref_name,
                    "shift": float(shift),
                    "config": str(path.resolve()),
                    "reference": str(ref_path.resolve()),
                }
            )

    summary = {
        "slope_deg": 9.207,
        "slope": SLOPE_9,
        "configs": configs,
        "notes": {
            "purpose": "Review 9.207 degree uphill references before training or long-reference stitching.",
            "next_step": "After visual approval, stitch a longer course with pose-matched joins and export 3s join clips.",
        },
    }
    (OUTDIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
