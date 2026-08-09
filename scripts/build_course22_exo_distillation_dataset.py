#!/usr/bin/env python3
"""Build a causal hip-history Exo dataset from a solved trajectory."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from myo_exo_train.checkpoint import build_sac_actor_for_checkpoint  # noqa: E402
from myo_exo_train.env.model import (  # noqa: E402
    build_muscle_model,
    muscle_action_mapping_mode,
)
from myo_exo_train.env.reference import load_reference_from_config  # noqa: E402
from myo_exo_train.env.runner import MJWarpMuscleRunner  # noqa: E402
from myo_exo_train.evaluation import load_config  # noqa: E402
from scripts.evaluate_course22_horizon_residual_mjwarp import (  # noqa: E402
    ctrl_to_policy_action,
)
from scripts.flat22_allocator_distillation_common import (  # noqa: E402
    append_history,
    proprio_indices,
)
from scripts.optimize_course22_horizon_exo_upper_bound import (  # noqa: E402
    configure_direct_exo,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--solution", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history-steps", type=int, default=8)
    parser.add_argument("--exo-max-torque-nm", type=float, default=10.0)
    parser.add_argument("--source-control-hz", type=float, default=30.0)
    parser.add_argument("--output-control-hz", type=float, default=0.0)
    args = parser.parse_args()

    config = copy.deepcopy(load_config(args.config))
    configure_direct_exo(config, args.exo_max_torque_nm, 1.0)
    model, data = build_muscle_model(config)
    qpos_indices, qvel_indices = proprio_indices(model, "hip4_exo6")

    with np.load(args.bank, allow_pickle=True) as source:
        bank = {key: np.asarray(source[key]) for key in source.files}
        qpos = np.asarray(source["qpos"], dtype=np.float32)
        qvel = np.asarray(source["qvel"], dtype=np.float32)
        phase = np.asarray(source["phase"], dtype=np.int64)
    with np.load(args.solution, allow_pickle=True) as solved:
        excitation_key = (
            "excitation" if "excitation" in solved else "assisted_excitation"
        )
        exo_key = "exo_control" if "exo_control" in solved else "assisted_exo_control"
        excitation = np.asarray(solved[excitation_key], dtype=np.float32)
        exo = np.asarray(solved[exo_key], dtype=np.float32)
        solved_phase = np.asarray(solved["phase"], dtype=np.int64)
    if not np.array_equal(phase[: len(solved_phase)], solved_phase):
        raise ValueError("bank and solution phases differ")
    qpos = qpos[: len(exo)]
    qvel = qvel[: len(exo)]
    phase = phase[: len(exo)]
    excitation = excitation[: len(exo)]

    normalized_obs = None
    muscle_action = None
    if (args.checkpoint is None) != (args.reference is None):
        raise ValueError("--checkpoint and --reference must be supplied together")
    if args.checkpoint is not None:
        frame_count = len(exo)
        config["reset"]["episode_steps"] = frame_count + 2
        config.setdefault("recovery_reset", {})["enabled"] = False
        reference = load_reference_from_config(
            args.reference,
            model,
            float(config["control"]["control_hz"]),
            torch.device("cuda"),
            config,
        )
        runner = MJWarpMuscleRunner(
            model=model,
            data=data,
            config=config,
            reference=reference,
            nworld=frame_count,
            nconmax=256,
            njmax=1024,
            seed=20260806,
            device=torch.device("cuda"),
        )
        checkpoint = torch.load(
            args.checkpoint, map_location="cuda", weights_only=False
        )
        _, normalizer, _ = build_sac_actor_for_checkpoint(
            checkpoint=checkpoint,
            model=model,
            config=config,
            obs_dim=runner.obs_dim,
            act_dim=runner.act_dim,
            device=torch.device("cuda"),
        )

        def assign(name: str, fallback: np.ndarray | None = None) -> None:
            value = bank.get(name, fallback)
            if value is not None:
                getattr(runner, name).copy_(
                    torch.as_tensor(
                        np.asarray(value)[:frame_count],
                        dtype=getattr(runner, name).dtype,
                        device="cuda",
                    )
                )

        assign("qpos")
        assign("qvel")
        assign("act")
        assign("ctrl")
        assign("prev_activation", bank.get("act"))
        assign("qacc_warmstart", np.zeros((frame_count, model.nv), np.float32))
        assign("site_xpos")
        runner.phase_idx.copy_(torch.as_tensor(phase, device="cuda"))
        if "x_align_mask" in bank:
            runner.x_align_mask.copy_(
                torch.as_tensor(bank["x_align_mask"][:frame_count], device="cuda")
            )
        if "ctrl" in bank and runner.applied_exo_ctrl.shape[1] >= 2:
            ctrl = np.asarray(bank["ctrl"][:frame_count], dtype=np.float32)
            if ctrl.shape[1] >= int(model.na) + 2:
                runner.applied_exo_ctrl[:, :2].copy_(
                    torch.as_tensor(ctrl[:, int(model.na) : int(model.na) + 2], device="cuda")
                )
        with torch.no_grad():
            normalized_obs = normalizer.normalize(runner.obs()).cpu().numpy().astype(np.float32)

        next_index = np.minimum(np.arange(frame_count) + 1, frame_count - 1)
        muscle_action = ctrl_to_policy_action(
            torch.from_numpy(excitation[next_index]),
            muscle_action_mapping_mode(config),
            int(model.na),
        ).numpy().astype(np.float32)

    hip_state = np.concatenate(
        (qpos[:, qpos_indices], qvel[:, qvel_indices]), axis=1
    ).astype(np.float32)
    output_hz = (
        float(args.output_control_hz)
        if args.output_control_hz > 0.0
        else float(args.source_control_hz)
    )
    if abs(output_hz - float(args.source_control_hz)) > 1.0e-9:
        if normalized_obs is not None:
            raise ValueError(
                "human distillation labels are only valid at the source control rate"
            )
        source_time = np.arange(len(exo), dtype=np.float64) / float(
            args.source_control_hz
        )
        output_frames = int(round(source_time[-1] * output_hz)) + 1
        output_time = np.arange(output_frames, dtype=np.float64) / output_hz
        output_time[-1] = min(output_time[-1], source_time[-1])
        hip_state = np.stack(
            [np.interp(output_time, source_time, hip_state[:, i]) for i in range(4)],
            axis=1,
        ).astype(np.float32)
        exo = np.stack(
            [np.interp(output_time, source_time, exo[:, i]) for i in range(2)],
            axis=1,
        ).astype(np.float32)
        phase = np.rint(np.interp(output_time, source_time, phase)).astype(np.int64)

    history: deque[np.ndarray] = deque(maxlen=max(1, args.history_steps))
    histories = []
    previous_exo = np.zeros(2, dtype=np.float32)
    for frame in range(len(exo)):
        sensor = np.concatenate((hip_state[frame], previous_exo)).astype(np.float32)
        histories.append(append_history(history, sensor, args.history_steps))
        previous_exo = exo[frame]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        proprio_history=np.stack(histories),
        exo_action=exo,
        solver_success=np.ones(len(exo), dtype=np.bool_),
        exo_supervision_mask=np.ones(len(exo), dtype=np.bool_),
        phase=phase,
        history_steps=np.asarray([args.history_steps], dtype=np.int64),
        exo_sensor_mode=np.asarray(["hip4_exo6"]),
        dataset_route=np.asarray(["fixed_trajectory_hard_constrained_solver"]),
        control_hz=np.asarray([output_hz], dtype=np.float32),
    )
    if normalized_obs is not None and muscle_action is not None:
        payload["normalized_obs"] = normalized_obs
        payload["muscle_action"] = muscle_action
    np.savez_compressed(args.output, **payload)
    print(
        json.dumps(
            {"output": str(args.output), "samples": len(exo), "control_hz": output_hz}
        )
    )


if __name__ == "__main__":
    main()
