#!/usr/bin/env python3
"""Replay horizon-optimized muscle and Exo controls in full MuJoCo physics."""
from __future__ import annotations

import argparse
import csv
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
from scripts.analyze_flat22_assisted_allocation_upper_bound import (  # noqa: E402
    configure_direct_exo,
)
from scripts.eval_dynamic_muscle_allocator import JOINTS  # noqa: E402


BRANCHES = ("nominal", "muscle_only", "assisted")


def controls_for_branch(
    branch: str,
    frame: int,
    bank_ctrl: np.ndarray,
    solution: dict[str, np.ndarray],
) -> np.ndarray:
    """Return controls for the transition from frame to frame + 1."""
    next_frame = min(frame + 1, bank_ctrl.shape[0] - 1)
    ctrl = np.zeros(bank_ctrl.shape[1], dtype=np.float64)
    if branch == "nominal":
        ctrl[:] = bank_ctrl[next_frame]
        ctrl[22:] = 0.0
    elif branch == "muscle_only":
        ctrl[:22] = solution["muscle_only_excitation"][next_frame]
    elif branch == "assisted":
        ctrl[:22] = solution["assisted_excitation"][next_frame]
        ctrl[22:24] = solution["assisted_exo_control"][frame]
    else:
        raise ValueError(f"unknown branch: {branch}")
    return ctrl


def set_bank_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    bank: dict[str, np.ndarray],
    frame: int,
) -> None:
    data.qpos[:] = bank["qpos"][frame]
    data.qvel[:] = bank["qvel"][frame]
    data.act[:] = bank["act"][frame, : int(model.na)]
    data.ctrl[:] = bank["ctrl"][frame, : int(model.nu)]
    data.ctrl[int(model.na) :] = 0.0
    if "qacc_warmstart" in bank:
        data.qacc_warmstart[:] = bank["qacc_warmstart"][frame, : int(model.nv)]
    mujoco.mj_forward(model, data)


def step_policy_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ctrl: np.ndarray,
    frame_skip: int,
) -> None:
    data.ctrl[:] = np.clip(
        ctrl, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    )
    for _ in range(frame_skip):
        mujoco.mj_step(model, data)


def error_metrics(
    data: mujoco.MjData,
    bank: dict[str, np.ndarray],
    frame: int,
    joint_qpos: np.ndarray,
    pelvis_x_qpos: int,
    pelvis_y_qpos: int,
) -> dict[str, float]:
    qpos_error = np.asarray(data.qpos) - bank["qpos"][frame]
    qvel_error = np.asarray(data.qvel) - bank["qvel"][frame]
    return {
        "qpos_rmse": float(np.sqrt(np.mean(np.square(qpos_error)))),
        "qvel_rmse": float(np.sqrt(np.mean(np.square(qvel_error)))),
        "joint_angle_rmse_rad": float(
            np.sqrt(np.mean(np.square(qpos_error[joint_qpos])))
        ),
        "pelvis_x_error_m": float(qpos_error[pelvis_x_qpos]),
        "pelvis_y_error_m": float(qpos_error[pelvis_y_qpos]),
        "activation_rmse": float(
            np.sqrt(np.mean(np.square(np.asarray(data.act) - bank["act"][frame])))
        ),
    }


def summarize_rows(rows: list[dict[str, float | int | str]]) -> dict[str, float | int]:
    if not rows:
        return {"frames": 0}
    numeric = (
        "qpos_rmse",
        "qvel_rmse",
        "joint_angle_rmse_rad",
        "pelvis_x_error_m",
        "pelvis_y_error_m",
        "activation_rmse",
    )
    result: dict[str, float | int] = {"frames": len(rows)}
    for key in numeric:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        result[f"{key}_mean"] = float(np.mean(np.abs(values)))
        result[f"{key}_final"] = float(values[-1])
        result[f"{key}_max_abs"] = float(np.max(np.abs(values)))
    for threshold in (0.05, 0.10, 0.20, 0.40):
        values = np.asarray(
            [float(row["joint_angle_rmse_rad"]) for row in rows], dtype=np.float64
        )
        crossing = np.flatnonzero(values > threshold)
        result[f"frames_until_joint_rmse_gt_{threshold:.2f}"] = (
            int(crossing[0]) if crossing.size else len(rows)
        )
    return result


def one_step_validation(
    model: mujoco.MjModel,
    bank: dict[str, np.ndarray],
    solution: dict[str, np.ndarray],
    frames: int,
    frame_skip: int,
    joint_qpos: np.ndarray,
    pelvis_x_qpos: int,
    pelvis_y_qpos: int,
) -> tuple[list[dict[str, float | int | str]], dict[str, dict[str, float | int]]]:
    rows: list[dict[str, float | int | str]] = []
    for branch in BRANCHES:
        data = mujoco.MjData(model)
        for frame in range(frames - 1):
            set_bank_state(model, data, bank, frame)
            ctrl = controls_for_branch(branch, frame, bank["ctrl"], solution)
            step_policy_frame(model, data, ctrl, frame_skip)
            row: dict[str, float | int | str] = {
                "mode": "one_step",
                "branch": branch,
                "frame": frame + 1,
                "phase": int(bank["phase"][frame + 1]),
            }
            row.update(
                error_metrics(
                    data,
                    bank,
                    frame + 1,
                    joint_qpos,
                    pelvis_x_qpos,
                    pelvis_y_qpos,
                )
            )
            rows.append(row)
    summary = {
        branch: summarize_rows(
            [row for row in rows if str(row["branch"]) == branch]
        )
        for branch in BRANCHES
    }
    return rows, summary


def continuous_validation(
    model: mujoco.MjModel,
    bank: dict[str, np.ndarray],
    solution: dict[str, np.ndarray],
    frames: int,
    frame_skip: int,
    joint_qpos: np.ndarray,
    pelvis_x_qpos: int,
    pelvis_y_qpos: int,
) -> tuple[list[dict[str, float | int | str]], dict[str, dict[str, float | int]]]:
    rows: list[dict[str, float | int | str]] = []
    for branch in BRANCHES:
        data = mujoco.MjData(model)
        set_bank_state(model, data, bank, 0)
        initial: dict[str, float | int | str] = {
            "mode": "continuous",
            "branch": branch,
            "frame": 0,
            "phase": int(bank["phase"][0]),
        }
        initial.update(
            error_metrics(
                data, bank, 0, joint_qpos, pelvis_x_qpos, pelvis_y_qpos
            )
        )
        rows.append(initial)
        for frame in range(frames - 1):
            ctrl = controls_for_branch(branch, frame, bank["ctrl"], solution)
            step_policy_frame(model, data, ctrl, frame_skip)
            row: dict[str, float | int | str] = {
                "mode": "continuous",
                "branch": branch,
                "frame": frame + 1,
                "phase": int(bank["phase"][frame + 1]),
            }
            row.update(
                error_metrics(
                    data,
                    bank,
                    frame + 1,
                    joint_qpos,
                    pelvis_x_qpos,
                    pelvis_y_qpos,
                )
            )
            rows.append(row)
            if not np.all(np.isfinite(data.qpos)):
                break
    summary = {
        branch: summarize_rows(
            [row for row in rows if str(row["branch"]) == branch]
        )
        for branch in BRANCHES
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--solution", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--exo-max-torque-nm", type=float, default=10.0)
    args = parser.parse_args()

    config = load_config(args.config)
    configure_direct_exo(config, args.exo_max_torque_nm, 1.0)
    model, _ = build_muscle_model(config)
    if int(model.na) != 22 or int(model.nu) != 24:
        raise ValueError(f"expected 22 muscles + 2 Exo controls, got na={model.na}, nu={model.nu}")
    frame_skip = int(config["control"]["frame_skip"])

    with np.load(args.bank, allow_pickle=True) as payload:
        bank = {key: np.asarray(payload[key]) for key in payload.files}
    with np.load(args.solution, allow_pickle=False) as payload:
        solution = {key: np.asarray(payload[key]) for key in payload.files}
    frames = min(len(bank["qpos"]), len(solution["phase"]))
    if args.max_frames > 0:
        frames = min(frames, int(args.max_frames))
    if not np.array_equal(bank["phase"][:frames], solution["phase"][:frames]):
        raise ValueError("bank and solution phases do not match")

    joint_qpos = np.asarray(
        [
            int(
                model.jnt_qposadr[
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
            )
            for name in JOINTS
        ],
        dtype=np.int64,
    )
    pelvis_x_qpos = int(
        model.jnt_qposadr[
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, "pelvis_tx"
            )
        ]
    )
    pelvis_y_qpos = int(
        model.jnt_qposadr[
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, "pelvis_ty"
            )
        ]
    )

    one_step_rows, one_step_summary = one_step_validation(
        model,
        bank,
        solution,
        frames,
        frame_skip,
        joint_qpos,
        pelvis_x_qpos,
        pelvis_y_qpos,
    )
    continuous_rows, continuous_summary = continuous_validation(
        model,
        bank,
        solution,
        frames,
        frame_skip,
        joint_qpos,
        pelvis_x_qpos,
        pelvis_y_qpos,
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = one_step_rows + continuous_rows
    fields = list(rows[0])
    with (args.outdir / "replay_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "frames": frames,
        "duration_seconds": frames / float(config["control"]["control_hz"]),
        "frame_skip": frame_skip,
        "one_step": one_step_summary,
        "continuous": continuous_summary,
        "control_alignment": {
            "muscle_excitation": "solution[t+1] drives bank state t to t+1",
            "exo_control": "solution[t] acts at bank state t",
        },
    }
    (args.outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
