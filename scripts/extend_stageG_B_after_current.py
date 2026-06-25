#!/usr/bin/env python3
"""Wait for current Stage-G B process to finish, then extend B training."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
B_CONFIG = ROOT / "configs/stageG_long_course_staged/muscle_2d_mjwarp_stageG_B_h48_flat_ascent_imit_sac.json"


def process_alive(pid: int) -> bool:
    return Path(f"/proc/{int(pid)}").exists()


def latest_checkpoint(outdir: Path) -> Path:
    candidates = sorted(outdir.glob("agent_step_*.pt"))
    if not candidates:
        latest = outdir / "latest.pt"
        if latest.exists():
            return latest
        raise FileNotFoundError(f"no checkpoint found in {outdir}")
    return candidates[-1]


def checkpoint_step(path: Path) -> int:
    checkpoint = torch.load(path, map_location="cpu")
    return int(checkpoint.get("global_step", 0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--current-outdir", type=Path, required=True)
    parser.add_argument("--root-outdir", type=Path, required=True)
    parser.add_argument("--additional-steps", type=int, default=524288)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    while process_alive(int(args.wait_pid)):
        time.sleep(30)

    resume = latest_checkpoint(args.current_outdir).resolve()
    start_step = checkpoint_step(resume)
    target_step = start_step + int(args.additional_steps)
    outdir = args.root_outdir / f"stageG_B_h48_flat_ascent_imit_sac_extend_{time.strftime('%Y%m%d-%H%M%S')}"
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "train.log"
    cmd = [
        sys.executable,
        str(ROOT / "cleanrl/sac_muscle_mjwarp.py"),
        "--config",
        str(B_CONFIG),
        "--outdir",
        str(outdir),
        "--resume",
        str(resume),
        "--total-timesteps",
        str(target_step),
        "--device",
        str(args.device),
        "--checkpoint-every",
        "8192",
        "--eval-every",
        "8192",
        "--log-every",
        "8192",
        "--video-every",
        "0",
    ]
    with log_path.open("w", encoding="utf-8") as log:
        log.write(
            f'{{"extend_stage":"B","resume":"{resume}","start_step":{start_step},'
            f'"target_step":{target_step},"additional_steps":{int(args.additional_steps)}}}\n'
        )
        log.flush()
        subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    main()
