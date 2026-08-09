#!/usr/bin/env python3
"""Export full dynamic states from a deterministic MJWarp SAC rollout."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from myo_exo_train.checkpoint import build_sac_actor_for_checkpoint
from myo_exo_train.env.model import build_muscle_model
from myo_exo_train.env.reference import load_reference_from_config
from myo_exo_train.env.runner import MJWarpMuscleRunner
from myo_exo_train.evaluation import load_config


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase", type=int, required=True)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--collect-phase-start", type=int, required=True)
    parser.add_argument("--collect-phase-end", type=int, required=True)
    parser.add_argument("--use-video-reset", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    config = copy.deepcopy(load_config(args.config))
    config["reset"]["episode_steps"] = int(args.steps)
    config["reset"]["phase_indices"] = [int(args.phase)]
    config["reset"]["phase_windows"] = []
    config["reset"]["phase_index_jitter"] = 0
    config.setdefault("recovery_reset", {})["enabled"] = False
    if args.use_video_reset:
        video_reset = config.get("video_reset", {})
        bank_value = str(
            video_reset.get("path", "")
            or config.get("offline_recovery_reset", {}).get("path", "")
            or ""
        )
        if not bank_value:
            raise ValueError("--use-video-reset requires video_reset.path")
        bank_path = Path(bank_value).expanduser()
        if not bank_path.is_absolute():
            bank_path = ROOT / bank_path
        config["offline_recovery_reset"] = {
            "enabled": True,
            "path": str(bank_path),
            "reset_probability": 1.0,
            "min_bank_size": 1,
            "fixed_index": int(video_reset.get("row", 0)),
        }
        config["reset"]["full_state_only"] = True
    else:
        config["reset"]["initial_activation"] = 0.0
        config["reset"]["initial_activation_range"] = []
        config["reset"]["full_state_only"] = False
        config.setdefault("offline_recovery_reset", {})["enabled"] = False

    model, data = build_muscle_model(config)
    reference = load_reference_from_config(
        args.reference,
        model,
        float(config["control"]["control_hz"]),
        device,
        config,
    )
    runner = MJWarpMuscleRunner(
        model=model,
        data=data,
        config=config,
        reference=reference,
        nworld=1,
        nconmax=128,
        njmax=512,
        seed=777,
        device=device,
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    actor, normalizer, _ = build_sac_actor_for_checkpoint(
        checkpoint=checkpoint,
        model=model,
        config=config,
        obs_dim=runner.obs_dim,
        act_dim=runner.act_dim,
        device=device,
    )

    rows: dict[str, list[np.ndarray | int]] = {
        "qpos": [],
        "qvel": [],
        "act": [],
        "ctrl": [],
        "qacc_warmstart": [],
        "prev_activation": [],
        "site_xpos": [],
        "phase": [],
        "x_align_mask": [],
    }
    obs = runner.obs()
    for _ in range(int(args.steps)):
        phase = int(runner.phase_idx[0].item()) % int(reference["length"])
        if int(args.collect_phase_start) <= phase < int(args.collect_phase_end):
            rows["qpos"].append(runner.qpos[0].detach().cpu().numpy().copy())
            rows["qvel"].append(runner.qvel[0].detach().cpu().numpy().copy())
            rows["act"].append(runner.act[0].detach().cpu().numpy().copy())
            rows["ctrl"].append(runner.ctrl[0].detach().cpu().numpy().copy())
            rows["qacc_warmstart"].append(
                runner.qacc_warmstart[0].detach().cpu().numpy().copy()
            )
            rows["prev_activation"].append(
                runner.prev_activation[0].detach().cpu().numpy().copy()
            )
            rows["site_xpos"].append(
                runner.site_xpos[0].detach().cpu().numpy().copy()
            )
            rows["phase"].append(phase)
            rows["x_align_mask"].append(
                bool(runner.x_align_mask[0].item())
            )
        action, _, _, _ = actor.get_action_and_value(
            normalizer.normalize(obs),
            deterministic=True,
        )
        obs, _reward, done, _terms = runner.step(action)
        if bool(done[0].item()):
            break

    if not rows["phase"]:
        raise RuntimeError("rollout produced no states in the requested phase range")
    payload = {
        key: np.asarray(value)
        for key, value in rows.items()
    }
    payload["metadata"] = np.asarray(
        {
            "source_checkpoint": str(args.checkpoint.resolve()),
            "source_config": str(args.config.resolve()),
            "reference": str(args.reference.resolve()),
            "simulator": "MJWarp",
            "start_phase": int(args.phase),
            "collect_phase_start": int(args.collect_phase_start),
            "collect_phase_end": int(args.collect_phase_end),
            "state_count": len(rows["phase"]),
        },
        dtype=object,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **payload)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "states": len(rows["phase"]),
                "phase_min": int(np.min(payload["phase"])),
                "phase_max": int(np.max(payload["phase"])),
            }
        )
    )


if __name__ == "__main__":
    main()
