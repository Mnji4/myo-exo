#!/usr/bin/env python3
"""Build a batch of Camargo ramp references for MyoAssist."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TERRAIN_SCRIPTS = Path("/home/lzn/exoskeleton_terrain/scripts")
if str(TERRAIN_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(TERRAIN_SCRIPTS))

from convert_camargo_ab06_to_myoassist_reference import (  # noqa: E402
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
DEFAULT_OUTDIR = ROOT / "results" / "camargo_ab06_ab08_ramp6x5_retargeted" / "npz"


@dataclass(frozen=True)
class SegmentSpec:
    terrain_id: str
    mode: str
    trial: str
    label: str


def terrain_id_for_label(label: str) -> str:
    if "ascent" in label:
        return "slopeascent"
    if "descent" in label:
        return "slopedescent"
    raise ValueError(f"Ramp label does not specify ascent/descent: {label}")


def find_ramp_dir(camargo_dir: Path, subject: str) -> Path:
    subject_root = camargo_dir / subject.upper()
    matches = sorted(path for path in subject_root.glob("*/ramp") if path.is_dir())
    if not matches:
        raise FileNotFoundError(f"Could not find {subject}/DATE/ramp under {camargo_dir}")
    if len(matches) > 1:
        print(f"[warn] multiple ramp dirs for {subject}; using {matches[0]}")
    return matches[0]


def selected_trials(ramp_dir: Path, ramp_id: int, side: str, per_ramp: int) -> list[Path]:
    pattern = f"ramp_{int(ramp_id)}_{side}_*.mat"
    trials = sorted((ramp_dir / "conditions").glob(pattern))
    if len(trials) < per_ramp:
        raise FileNotFoundError(
            f"Only found {len(trials)} trials for {pattern} in {ramp_dir / 'conditions'}; "
            f"need {per_ramp}"
        )
    return trials[:per_ramp]


def reference_id(subject: str, terrain_id: str, label: str, trial_stem: str) -> str:
    return f"camargo_{subject.lower()}_{terrain_id}_{label}_{trial_stem}"


def build_one(
    *,
    subject: str,
    ramp_dir: Path,
    condition_path: Path,
    label: str,
    outdir: Path,
) -> dict[str, Any]:
    trial = condition_path.name
    ik_path = ramp_dir / "ik" / trial
    if not ik_path.exists():
        raise FileNotFoundError(ik_path)

    terrain_id = terrain_id_for_label(label)
    spec = SegmentSpec(terrain_id=terrain_id, mode="ramp", trial=trial, label=label)
    conditions_labels = decode_matlab_table(condition_path.read_bytes())
    conditions_plain = decode_plain_mat(condition_path.read_bytes())
    ik = decode_matlab_table(ik_path.read_bytes())
    start, end = best_label_run(conditions_labels["Label"], label)

    ik_segment = table_segment(ik, start, end)
    time = np.asarray(ik_segment["Header"], dtype=np.float64)
    series, transform = camargo_ik_to_myoassist_series(ik_segment)
    sample_rate = infer_sample_rate(time)
    terrain = terrain_from_segment(spec, conditions_plain, ik_segment, transform)

    trial_stem = condition_path.stem
    ref_id = reference_id(subject, terrain_id, label, trial_stem)
    output_path = outdir / f"{ref_id}_myoassist_3d.npz"
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
            **terrain,
        },
    )
    duration = float(time[-1] - time[0]) if len(time) > 1 else 0.0
    forward_distance = float(transform.get("forward_distance", 0.0))
    return {
        "reference_id": ref_id,
        "subject": subject.upper(),
        "ramp_id": int(metadata["ramp_id"]),
        "side": str(metadata["ramp_side"]),
        "trial": trial,
        "label": label,
        "terrain_id": terrain_id,
        "terrain_label": metadata.get("terrain_label", ""),
        "ramp_incline_deg": float(metadata.get("ramp_incline_deg", 0.0)),
        "frames": int(metadata["data_length"]),
        "sample_rate": float(sample_rate),
        "duration_sec": duration,
        "forward_distance_m": forward_distance,
        "mean_forward_speed_mps": forward_distance / duration if duration > 0.0 else 0.0,
        "npz_path": str(output_path),
        "metadata": metadata,
        "series_summary": summarize_series(series),
    }


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camargo-dir", type=Path, default=DEFAULT_CAMARGO_DIR)
    parser.add_argument("--subjects", nargs="+", default=["AB06", "AB08"])
    parser.add_argument("--side", default="l", choices=["l", "r"])
    parser.add_argument("--per-ramp", type=int, default=5)
    parser.add_argument("--ramp-ids", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--labels", nargs="+", default=["rampascent", "rampdescent"])
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    records: list[dict[str, Any]] = []
    for subject in args.subjects:
        ramp_dir = find_ramp_dir(args.camargo_dir, subject)
        for ramp_id in args.ramp_ids:
            for condition_path in selected_trials(ramp_dir, ramp_id, args.side, args.per_ramp):
                for label in args.labels:
                    print(f"[build] {subject} {condition_path.name} label={label}")
                    records.append(
                        build_one(
                            subject=subject,
                            ramp_dir=ramp_dir,
                            condition_path=condition_path,
                            label=label,
                            outdir=args.outdir,
                        )
                    )
                    if args.limit and len(records) >= args.limit:
                        break
                if args.limit and len(records) >= args.limit:
                    break
            if args.limit and len(records) >= args.limit:
                break
        if args.limit and len(records) >= args.limit:
            break

    manifest = {
        "source": {
            "dataset": "Camargo",
            "camargo_dir": str(args.camargo_dir),
            "subjects": [str(subject).upper() for subject in args.subjects],
        },
        "selection": {
            "side": args.side,
            "per_ramp": int(args.per_ramp),
            "ramp_ids": [int(item) for item in args.ramp_ids],
            "labels": list(args.labels),
        },
        "record_count": len(records),
        "records": records,
    }
    manifest_path = args.manifest or args.outdir.parent / "manifest.json"
    write_manifest(manifest_path, manifest)
    print(f"[done] wrote {len(records)} references")
    print(f"[manifest] {manifest_path}")


if __name__ == "__main__":
    main()
