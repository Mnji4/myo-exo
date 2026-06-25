#!/usr/bin/env python3
"""Build Camargo walk<->terrain transition references for the MJWarp muscle runner."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TERRAIN_ROOT = Path("/home/lzn/exoskeleton_terrain")
CAMARGO_DIR = Path("/mnt/c/Users/liang/Desktop/camargo")
sys.path.insert(0, str(TERRAIN_ROOT / "scripts"))

from convert_camargo_ab06_to_myoassist_reference import (  # noqa: E402
    best_label_run,
    camargo_ik_to_myoassist_series,
    decode_matlab_table,
    decode_plain_mat,
    infer_sample_rate,
    mode_prefix,
    save_reference_npz,
    scalar_str,
    summarize_series,
    table_segment,
    terrain_from_segment,
)


@dataclass(frozen=True)
class TransitionSpec:
    subject: str
    terrain_id: str
    mode: str
    trial: str
    label: str


TRANSITIONS = (
    TransitionSpec("AB06", "slopeascent", "ramp", "ramp_3_l_01_01.mat", "walk-rampascent"),
    TransitionSpec("AB06", "slopeascent", "ramp", "ramp_3_l_01_01.mat", "rampascent-walk"),
    TransitionSpec("AB06", "slopedescent", "ramp", "ramp_3_l_01_01.mat", "walk-rampdescent"),
    TransitionSpec("AB06", "slopedescent", "ramp", "ramp_3_l_01_01.mat", "rampdescent-walk"),
    TransitionSpec("AB08", "stairascent", "stair", "stair_2_l_01_01.mat", "walk-stairascent"),
    TransitionSpec("AB08", "stairascent", "stair", "stair_2_l_01_01.mat", "stairascent-walk"),
    TransitionSpec("AB06", "stairdescent", "stair", "stair_2_l_01_01.mat", "walk-stairdescent"),
    TransitionSpec("AB06", "stairdescent", "stair", "stair_2_l_01_01.mat", "stairdescent-walk"),
)


class TerrainSpecAdapter:
    def __init__(self, item: TransitionSpec):
        self.terrain_id = item.terrain_id
        self.mode = item.mode
        self.trial = item.trial
        self.label = item.label


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def convert_transition(zf: zipfile.ZipFile, spec: TransitionSpec, output_dir: Path) -> dict[str, Any]:
    prefix = mode_prefix(zf, spec.mode, spec.trial)
    ik = decode_matlab_table(zf.read(f"{prefix}/ik/{spec.trial}"))
    conditions_labels = decode_matlab_table(zf.read(f"{prefix}/conditions/{spec.trial}"))
    conditions_plain = decode_plain_mat(zf.read(f"{prefix}/conditions/{spec.trial}"))
    start, end = best_label_run(conditions_labels["Label"], spec.label)
    ik_segment = table_segment(ik, start, end)
    time = np.asarray(ik_segment["Header"], dtype=np.float64)
    series, transform = camargo_ik_to_myoassist_series(ik_segment)
    sample_rate = infer_sample_rate(time)
    terrain = terrain_from_segment(TerrainSpecAdapter(spec), conditions_plain, ik_segment, transform)
    output_path = output_dir / (
        f"camargo_{spec.subject.lower()}_{spec.terrain_id}_{spec.label}_transition_myoassist_3d.npz"
    )
    metadata = save_reference_npz(
        output_path,
        series,
        sample_rate,
        {
            "source_dataset": "Camargo",
            "source_subject": spec.subject.upper(),
            "source_mode": spec.mode,
            "source_trial": spec.trial,
            "source_label": spec.label,
            "terrain_id": spec.terrain_id,
            "is_transition": True,
            "transition_label": spec.label,
            "source_segment_start_index": int(start),
            "source_segment_end_index": int(end),
            "source_time_start": float(time[0]),
            "source_time_end": float(time[-1]),
            "root_transform": transform,
            **terrain,
        },
    )
    return {
        "subject": spec.subject,
        "terrain_id": spec.terrain_id,
        "label": spec.label,
        "path": str(output_path),
        "frames": int(metadata["data_length"]),
        "sample_rate": float(metadata["sample_rate"]),
        "duration_sec": float(metadata["data_length"]) / float(metadata["sample_rate"]),
        "terrain_type": metadata.get("terrain_type", ""),
        "terrain_params": metadata.get("terrain_params", ""),
        "series_summary": summarize_series(series),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camargo-dir", type=Path, default=CAMARGO_DIR)
    parser.add_argument("--outdir", type=Path, default=ROOT / "data" / "camargo_transition_references")
    parser.add_argument("--summary", type=Path, default=None)
    args = parser.parse_args()

    by_subject: dict[str, list[TransitionSpec]] = {}
    for spec in TRANSITIONS:
        by_subject.setdefault(spec.subject.upper(), []).append(spec)

    records = []
    for subject, specs in sorted(by_subject.items()):
        zip_path = args.camargo_dir / f"{subject}.zip"
        if not zip_path.exists():
            raise FileNotFoundError(zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            labels = []
            for spec in specs:
                print(f"[transition] {spec.subject} {spec.terrain_id} {spec.label}", flush=True)
                records.append(convert_transition(zf, spec, args.outdir))
                labels.append(spec.label)
            _ = [scalar_str(label) for label in labels]

    summary = {"transitions": records}
    summary_path = args.summary or args.outdir / "summary.json"
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
