"""Command-line entry point for MJWarp muscle SAC training."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from myo_exo_train.rl.trainer import DEFAULT_REFERENCE_PATH, ROOT, run_training

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "muscle_2d_mjwarp_teacher_stage1_swing_hip_sac.json")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE_PATH)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--nworld", type=int, default=None)
    parser.add_argument("--nconmax", type=int, default=128)
    parser.add_argument("--njmax", type=int, default=512)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--episode-steps", type=int, default=None)
    parser.add_argument("--qpos-noise", type=float, default=None)
    parser.add_argument("--qvel-noise", type=float, default=None)
    parser.add_argument("--eval-every", type=int, default=8192)
    parser.add_argument("--eval-worlds", type=int, default=16)
    parser.add_argument("--eval-steps", type=int, default=12)
    parser.add_argument("--video-every", type=int, default=8192)
    parser.add_argument("--video-steps", type=int, default=12)
    parser.add_argument("--video-phase", type=int, default=344)
    parser.add_argument("--video-height", type=int, default=720)
    parser.add_argument("--video-width", type=int, default=1280)
    parser.add_argument("--video-camera-distance", type=float, default=7.0)
    parser.add_argument("--video-camera-height", type=float, default=0.9)
    parser.add_argument("--video-camera-azimuth", type=float, default=135.0)
    parser.add_argument("--video-camera-elevation", type=float, default=-30.0)
    parser.add_argument("--render-only-video", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=8192)
    parser.add_argument("--log-every", type=int, default=8192)
    return parser

def main() -> None:
    run_training(build_parser().parse_args())

if __name__ == "__main__":
    main()
