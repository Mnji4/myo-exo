#!/usr/bin/env python3
"""Build fixed-trajectory active muscle torque maps from a full-state bank."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from myo_exo_train.env.model import build_muscle_model  # noqa: E402
from myo_exo_train.evaluation import load_config  # noqa: E402
from scripts.eval_dynamic_muscle_allocator import (  # noqa: E402
    JOINTS,
    active_torque_map,
    exo_torque_map,
)
from scripts.optimize_course22_horizon_exo_upper_bound import (  # noqa: E402
    configure_direct_exo,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--exo-max-torque-nm", type=float, default=10.0)
    args = parser.parse_args()

    config = load_config(args.config)
    configure_direct_exo(config, args.exo_max_torque_nm, 1.0)
    model, probe_data = build_muscle_model(config)
    if int(model.na) != 22:
        raise ValueError(f"expected 22 muscles, got {model.na}")

    with np.load(args.bank, allow_pickle=True) as bank:
        frames = len(bank["qpos"])
        if args.max_frames > 0:
            frames = min(frames, int(args.max_frames))
        qpos = np.asarray(bank["qpos"][:frames], dtype=np.float64)
        qvel = np.asarray(bank["qvel"][:frames], dtype=np.float64)
        activation = np.asarray(bank["act"][:frames], dtype=np.float64)
        phase = np.asarray(bank["phase"][:frames], dtype=np.int64)
        applied_exo = (
            np.asarray(bank["applied_exo_ctrl"][:frames], dtype=np.float64)
            if "applied_exo_ctrl" in bank
            else np.zeros((frames, 2), dtype=np.float64)
        )

    joint_dofs = np.asarray(
        [
            int(
                model.jnt_dofadr[
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
            )
            for name in JOINTS
        ],
        dtype=np.int64,
    )
    muscle_maps = np.stack(
        [
            active_torque_map(model, probe_data, qpos[i], qvel[i], joint_dofs)
            for i in range(frames)
        ]
    )
    if int(model.nu) > int(model.na):
        exo_maps = np.stack(
            [
                exo_torque_map(model, probe_data, qpos[i], qvel[i], joint_dofs)
                for i in range(frames)
            ]
        )
    else:
        exo_maps = np.zeros((frames, len(JOINTS), 2), dtype=np.float64)
    target_torque = np.einsum("tjm,tm->tj", muscle_maps, activation)
    target_torque += np.einsum("tje,te->tj", exo_maps, applied_exo)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        phase=phase,
        target_torque=target_torque,
        muscle_maps=muscle_maps,
        exo_maps=exo_maps,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "frames": frames,
                "target_torque_rms_nm": float(np.sqrt(np.mean(target_torque**2))),
                "source_exo_abs_mean_nm": float(np.mean(np.abs(applied_exo)) * 10.0),
            }
        )
    )


if __name__ == "__main__":
    main()
