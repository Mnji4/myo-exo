#!/usr/bin/env python3
"""Create configs for reviewing uphill-only reference candidates."""

from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "results/stageG_long_course_reference/muscle_2d_mjwarp_stageG_long_course_gated_ref_sac.json"
OUTDIR = ROOT / "results/stageG_uphill_reference_review"
STEADY_18_REF = (
    ROOT
    / "results_old/camargo_ramp6_source_selected4/npz_ab08/"
    / "camargo_ab08_slopeascent_rampascent_ramp_6_l_01_05_myoassist_3d.npz"
)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    base = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))

    steady = copy.deepcopy(base)
    slope = 0.3249196962329063
    steady["experiment_name"] = "muscle_2d_mjwarp_stageG_uphill_steady18_review"
    steady["reference_pool"] = {"paths": [str(STEADY_18_REF.resolve())]}
    steady["reference_pool_schedule"] = []
    steady["video"] = {"phase_indices": [0]}
    steady["checkpoint_video_export"]["phase_indices"] = [0]
    steady["terrain_course"] = {
        **steady.get("terrain_course", {}),
        "enabled": True,
        "hfield_size_z": 2.2,
        "levelwalking_reference_x_shift": 0.0,
        "segments": [
            {"type": "flat", "x0": -20.0, "x1": 0.0, "height": 0.0},
            {"type": "slope", "x0": 0.0, "x1": 6.0, "height0": 0.0, "slope": slope},
            {"type": "flat", "x0": 6.0, "x1": 60.0, "height": 6.0 * slope},
        ],
    }
    steady_path = OUTDIR / "muscle_2d_mjwarp_stageG_uphill_steady18_review.json"
    steady_path.write_text(json.dumps(steady, indent=2) + "\n", encoding="utf-8")

    long_slice = copy.deepcopy(base)
    long_slice["experiment_name"] = "muscle_2d_mjwarp_stageG_uphill_longcourse_phase89_review"
    long_slice["video"] = {"phase_indices": [89]}
    long_slice["checkpoint_video_export"]["phase_indices"] = [89]
    long_slice_path = OUTDIR / "muscle_2d_mjwarp_stageG_uphill_longcourse_phase89_review.json"
    long_slice_path.write_text(json.dumps(long_slice, indent=2) + "\n", encoding="utf-8")

    summary = {
        "steady18_config": str(steady_path.resolve()),
        "steady18_reference": str(STEADY_18_REF.resolve()),
        "longcourse_phase89_config": str(long_slice_path.resolve()),
        "longcourse_phase89_reference": str(
            (ROOT / "results/stageG_long_course_reference/stageG_long_flat_up_flat_down_flat_myoassist_3d.npz").resolve()
        ),
        "notes": {
            "steady18": "pure rampascent source on a single 18deg uphill slope",
            "longcourse_phase89": "existing long-course uphill segment from level_to_up into up_to_level",
        },
    }
    (OUTDIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
