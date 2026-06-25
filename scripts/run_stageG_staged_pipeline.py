#!/usr/bin/env python3
"""Run Stage-G staged SAC training sequentially from a seed checkpoint."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs/stageG_long_course_staged/manifest.json"


def checkpoint_step(path: Path) -> int:
    checkpoint = torch.load(path, map_location="cpu")
    return int(checkpoint.get("global_step", 0))


def latest_checkpoint(outdir: Path) -> Path:
    latest = outdir / "latest.pt"
    if latest.exists():
        return latest
    candidates = sorted(outdir.glob("agent_step_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"no checkpoints in {outdir}")
    return candidates[-1]


def run_stage(
    stage: dict,
    *,
    resume: Path | None,
    root_outdir: Path,
    device: str,
    video_every: int,
    video_steps: int,
    video_width: int,
    video_height: int,
) -> Path:
    stage_name = str(stage["stage"])
    config = Path(stage["config"])
    start_step = checkpoint_step(resume) if resume is not None else 0
    additional = int(stage["total_timesteps"])
    target_step = start_step + additional
    outdir = root_outdir / f"{stage_name}_{time.strftime('%Y%m%d-%H%M%S')}"
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "train.log"
    cmd = [
        sys.executable,
        str(ROOT / "cleanrl/sac_muscle_mjwarp.py"),
        "--config",
        str(config),
        "--outdir",
        str(outdir),
        "--total-timesteps",
        str(target_step),
        "--device",
        device,
        "--checkpoint-every",
        "8192",
        "--eval-every",
        "8192",
        "--log-every",
        "8192",
        "--video-every",
        str(int(video_every)),
        "--video-steps",
        str(int(video_steps)),
        "--video-width",
        str(int(video_width)),
        "--video-height",
        str(int(video_height)),
    ]
    if resume is not None:
        cmd.extend(["--resume", str(resume)])
    header = {
        "stage": stage_name,
        "config": str(config),
        "resume": str(resume) if resume is not None else None,
        "start_step": start_step,
        "additional_steps": additional,
        "target_step": target_step,
        "outdir": str(outdir),
        "cmd": cmd,
    }
    (outdir / "pipeline_stage.json").write_text(json.dumps(header, indent=2) + "\n", encoding="utf-8")
    with log_path.open("w", encoding="utf-8") as log:
        print(json.dumps({"pipeline_start": header}, ensure_ascii=False), flush=True)
        log.write(json.dumps({"pipeline_start": header}, ensure_ascii=False) + "\n")
        log.flush()
        subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    return latest_checkpoint(outdir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--initial-checkpoint", type=Path, default=None)
    parser.add_argument("--start-stage", default="stageG_A_h24_flat_imit_sac")
    parser.add_argument("--root-outdir", type=Path, default=ROOT / "results/stageG_pipeline")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--video-every", type=int, default=32768)
    parser.add_argument("--video-steps", type=int, default=64)
    parser.add_argument("--video-width", type=int, default=480)
    parser.add_argument("--video-height", type=int, default=288)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    start_index = next(i for i, item in enumerate(manifest) if str(item["stage"]) == str(args.start_stage))
    resume = args.initial_checkpoint.resolve() if args.initial_checkpoint is not None else None
    if start_index > 0 and resume is None:
        raise SystemExit("--initial-checkpoint is required when --start-stage is not stage A")
    args.root_outdir.mkdir(parents=True, exist_ok=True)
    for stage in manifest[start_index:]:
        resume = run_stage(
            stage,
            resume=resume,
            root_outdir=args.root_outdir,
            device=str(args.device),
            video_every=int(args.video_every),
            video_steps=int(args.video_steps),
            video_width=int(args.video_width),
            video_height=int(args.video_height),
        )
    print(json.dumps({"pipeline_done": True, "final_checkpoint": str(resume)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
