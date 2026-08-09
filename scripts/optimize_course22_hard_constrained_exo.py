#!/usr/bin/env python3
"""Solve fixed-trajectory muscle/Exo allocation with hard torque and dynamics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import casadi as ca
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from myo_exo_train.env.model import (  # noqa: E402
    build_muscle_model,
    muscle_action_mapping_mode,
    policy_action_to_ctrl,
)
from myo_exo_train.evaluation import load_config  # noqa: E402
from scripts.eval_dynamic_muscle_allocator import (  # noqa: E402
    evolve_activation,
    excitation_for_target_activation,
)
from scripts.optimize_course22_horizon_exo_upper_bound import (  # noqa: E402
    branch_metrics,
    configure_direct_exo,
)


def muscle_control_limits(model, config) -> tuple[np.ndarray, np.ndarray]:
    muscles = int(model.na)
    action_limits = torch.tensor(
        [[-1.0] * muscles, [1.0] * muscles], dtype=torch.float64
    )
    ctrl_low = torch.from_numpy(model.actuator_ctrlrange[:muscles, 0]).double()
    ctrl_high = torch.from_numpy(model.actuator_ctrlrange[:muscles, 1]).double()
    limits = policy_action_to_ctrl(
        action_limits,
        ctrl_low,
        ctrl_high,
        muscle_count=muscles,
        muscle_mapping=muscle_action_mapping_mode(config),
    ).numpy()
    return limits[0], limits[1]


def activation_extreme_step(
    activation,
    *,
    excitation: np.ndarray,
    activation_tau: np.ndarray,
    deactivation_tau: np.ndarray,
    timestep: float,
    frame_skip: int,
    activating: bool,
):
    value = activation
    fixed_excitation = ca.DM(excitation)
    tau_act = ca.DM(activation_tau)
    tau_deact = ca.DM(deactivation_tau)
    for _ in range(frame_skip):
        scale = 0.5 + 1.5 * value
        tau = tau_act * scale if activating else tau_deact / scale
        value = value + float(timestep) * (fixed_excitation - value) / tau
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline-solution", type=Path, required=True)
    warm_group = parser.add_mutually_exclusive_group(required=True)
    warm_group.add_argument("--warm-start", type=Path)
    warm_group.add_argument(
        "--source-bank",
        type=Path,
        help="Use the recorded simulator trajectory as a strictly feasible start.",
    )
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--exo-max-torque-nm", type=float, default=10.0)
    parser.add_argument("--exo-smooth-weight", type=float, default=1.0e-3)
    parser.add_argument("--exo-curvature-weight", type=float, default=1.0e-3)
    parser.add_argument(
        "--exo-max-delta-nm-per-step",
        type=float,
        default=0.0,
        help="Hard adjacent-frame Exo torque change limit; zero disables it.",
    )
    parser.add_argument(
        "--exo-max-curvature-nm-per-step2",
        type=float,
        default=0.0,
        help="Hard second-difference Exo torque limit; zero disables it.",
    )
    parser.add_argument(
        "--exo-zero-initial-torque",
        action="store_true",
        help="Force the first Exo command to zero before applying slew limits.",
    )
    parser.add_argument(
        "--exo-half-cycle-frames",
        type=int,
        default=0,
        help=(
            "Hard bilateral symmetry lag: right[t]=left[t+lag] and vice versa. "
            "Zero disables it."
        ),
    )
    parser.add_argument("--effort-weight", type=float, default=1.0)
    parser.add_argument("--activation-anchor-weight", type=float, default=0.0)
    parser.add_argument("--max-iterations", type=int, default=2000)
    parser.add_argument(
        "--torque-rmse-limit-nm",
        type=float,
        default=0.0,
        help="Allow this trajectory-wide torque RMSE; zero enforces equality.",
    )
    parser.add_argument("--disable-exo", action="store_true")
    parser.add_argument(
        "--fixed-exo-profile",
        type=Path,
        help="Fix Exo control to an exo_control array from an NPZ file.",
    )
    args = parser.parse_args()
    if args.disable_exo and args.fixed_exo_profile is not None:
        parser.error("--disable-exo and --fixed-exo-profile are mutually exclusive")

    config = load_config(args.config)
    # Pure-muscle XMLs intentionally have no Exo_R/Exo_L actuators.
    if not args.disable_exo:
        configure_direct_exo(config, args.exo_max_torque_nm, 1.0)
    model, _ = build_muscle_model(config)
    muscles = int(model.na)
    if muscles != 22:
        raise ValueError(f"expected 22 muscles, got {muscles}")
    frame_skip = int(config["control"]["frame_skip"])
    excitation_low, excitation_high = muscle_control_limits(model, config)
    dynprm = np.asarray(model.actuator_dynprm[:muscles, :3], dtype=np.float64)
    if np.max(np.abs(dynprm[:, 2])) > 1.0e-12:
        raise ValueError("hard-constrained solver requires muscle smoothing=0")

    with np.load(args.baseline_solution, allow_pickle=True) as source:
        phase = np.asarray(source["phase"], dtype=np.int64)
        target_torque = np.asarray(source["target_torque"], dtype=np.float64)
        muscle_maps = np.asarray(source["muscle_maps"], dtype=np.float64)
        exo_maps = np.asarray(source["exo_maps"], dtype=np.float64)
    if args.source_bank is not None:
        with np.load(args.source_bank, allow_pickle=True) as warm:
            warm_phase = np.asarray(warm["phase"], dtype=np.int64)
            if not np.array_equal(warm_phase, phase):
                raise ValueError("source bank phases do not match baseline solution")
            initial_activation = np.asarray(warm["act"], dtype=np.float64)
            initial_excitation = np.asarray(warm["ctrl"][:, :muscles], dtype=np.float64)
            if args.disable_exo:
                initial_exo = np.zeros((len(phase), 2), dtype=np.float64)
            elif "applied_exo_ctrl" in warm:
                initial_exo = np.asarray(
                    warm["applied_exo_ctrl"], dtype=np.float64
                )
            elif "ctrl" in warm and warm["ctrl"].shape[1] >= muscles + 2:
                initial_exo = np.asarray(
                    warm["ctrl"][:, muscles : muscles + 2], dtype=np.float64
                )
            else:
                initial_exo = np.zeros((len(phase), 2), dtype=np.float64)
    else:
        with np.load(args.warm_start, allow_pickle=True) as warm:
            prefix = "muscle_only" if args.disable_exo else "assisted"
            initial_activation = np.asarray(
                warm[f"{prefix}_activation"], dtype=np.float64
            )
            initial_excitation = np.asarray(
                warm[f"{prefix}_excitation"], dtype=np.float64
            )
            initial_exo = (
                np.zeros((len(phase), 2), dtype=np.float64)
                if args.disable_exo
                else np.asarray(warm["assisted_exo_control"], dtype=np.float64)
            )
    frames = len(phase)
    fixed_exo_profile = None
    if args.fixed_exo_profile is not None:
        with np.load(args.fixed_exo_profile, allow_pickle=True) as profile:
            keys = [key for key in profile.files if key == "exo_control"]
            if not keys:
                keys = [key for key in profile.files if key.endswith("__exo_control")]
            if len(keys) != 1:
                raise ValueError(
                    f"{args.fixed_exo_profile}: expected one Exo control array, got {keys}"
                )
            fixed_exo_profile = np.asarray(profile[keys[0]], dtype=np.float64)
        if fixed_exo_profile.shape != (frames, 2):
            raise ValueError(
                f"fixed Exo shape {fixed_exo_profile.shape}, expected {(frames, 2)}"
            )
        initial_exo = fixed_exo_profile.copy()
    expected_activation = (frames, muscles)
    if initial_activation.shape != expected_activation:
        raise ValueError(
            f"warm activation shape {initial_activation.shape}, expected {expected_activation}"
        )

    opti = ca.Opti()
    activation = opti.variable(muscles, frames)
    exo = opti.variable(2, frames)

    opti.subject_to(activation[:, 0] == initial_activation[0])
    opti.subject_to(opti.bounded(0.0, activation, 1.0))
    if args.disable_exo:
        opti.subject_to(exo == 0.0)
    elif fixed_exo_profile is not None:
        opti.subject_to(exo == ca.DM(fixed_exo_profile.T))
    else:
        opti.subject_to(opti.bounded(-1.0, exo, 1.0))
        if args.exo_zero_initial_torque:
            opti.subject_to(exo[:, 0] == 0.0)
        if args.exo_max_delta_nm_per_step > 0.0:
            max_delta = (
                float(args.exo_max_delta_nm_per_step)
                / float(args.exo_max_torque_nm)
            )
            opti.subject_to(
                opti.bounded(
                    -max_delta, exo[:, 1:] - exo[:, :-1], max_delta
                )
            )
        if args.exo_max_curvature_nm_per_step2 > 0.0:
            max_curvature = (
                float(args.exo_max_curvature_nm_per_step2)
                / float(args.exo_max_torque_nm)
            )
            curvature = exo[:, 2:] - 2.0 * exo[:, 1:-1] + exo[:, :-2]
            opti.subject_to(
                opti.bounded(-max_curvature, curvature, max_curvature)
            )
        if args.exo_half_cycle_frames > 0:
            lag = int(args.exo_half_cycle_frames)
            if lag >= frames:
                raise ValueError("Exo half-cycle lag must be shorter than trajectory")
            opti.subject_to(exo[0, :-lag] == exo[1, lag:])
            opti.subject_to(exo[1, :-lag] == exo[0, lag:])

    for frame in range(frames - 1):
        reached_low = activation_extreme_step(
            activation[:, frame],
            excitation=excitation_low,
            activation_tau=dynprm[:, 0],
            deactivation_tau=dynprm[:, 1],
            timestep=float(model.opt.timestep),
            frame_skip=frame_skip,
            activating=False,
        )
        reached_high = activation_extreme_step(
            activation[:, frame],
            excitation=excitation_high,
            activation_tau=dynprm[:, 0],
            deactivation_tau=dynprm[:, 1],
            timestep=float(model.opt.timestep),
            frame_skip=frame_skip,
            activating=True,
        )
        opti.subject_to(activation[:, frame + 1] >= reached_low)
        opti.subject_to(activation[:, frame + 1] <= reached_high)

    torque_scale = np.maximum(np.abs(target_torque), 10.0)
    torque_errors = []
    for frame in range(frames):
        predicted = ca.DM(muscle_maps[frame]) @ activation[:, frame]
        predicted += ca.DM(exo_maps[frame]) @ exo[:, frame]
        error = predicted - ca.DM(target_torque[frame])
        if args.torque_rmse_limit_nm > 0.0 and frame >= args.warmup:
            torque_errors.append(error)
        elif args.torque_rmse_limit_nm <= 0.0:
            opti.subject_to(
                predicted / ca.DM(torque_scale[frame])
                == ca.DM(target_torque[frame] / torque_scale[frame])
            )
    if torque_errors:
        all_torque_errors = ca.horzcat(*torque_errors)
        opti.subject_to(
            ca.sumsqr(all_torque_errors) / all_torque_errors.numel()
            <= float(args.torque_rmse_limit_nm) ** 2
        )

    selected_activation = activation[:, min(args.warmup, frames - 1) :]
    effort = ca.sumsqr(selected_activation) / selected_activation.numel()
    anchor_target = initial_activation[min(args.warmup, frames - 1) :].T
    activation_anchor = (
        ca.sumsqr(selected_activation - ca.DM(anchor_target))
        / selected_activation.numel()
    )
    exo_delta = exo[:, 1:] - exo[:, :-1]
    exo_curvature = exo[:, 2:] - 2.0 * exo[:, 1:-1] + exo[:, :-2]
    objective = (
        args.effort_weight * effort
        + args.activation_anchor_weight * activation_anchor
    )
    if not args.disable_exo:
        objective += args.exo_smooth_weight * ca.sumsqr(exo_delta) / exo_delta.numel()
        objective += (
            args.exo_curvature_weight
            * ca.sumsqr(exo_curvature)
            / exo_curvature.numel()
        )
    opti.minimize(objective)

    opti.set_initial(activation, initial_activation.T)
    opti.set_initial(exo, initial_exo.T)
    opti.solver(
        "ipopt",
        {"expand": True, "print_time": True},
        {
            "max_iter": args.max_iterations,
            "tol": 1.0e-7,
            "acceptable_tol": 1.0e-5,
            "acceptable_iter": 10,
            "print_level": 5,
            "linear_solver": "mumps",
        },
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    try:
        solution = opti.solve()
        status = "success"
        get_value = solution.value
    except RuntimeError:
        status = str(opti.stats().get("return_status", "failed"))
        get_value = opti.debug.value

    solved_activation = np.asarray(get_value(activation), dtype=np.float64).T
    solved_excitation = np.empty_like(solved_activation)
    solved_excitation[0] = initial_excitation[0]
    reached_activation = np.empty_like(solved_activation)
    reached_activation[0] = solved_activation[0]
    for frame in range(1, frames):
        solved_excitation[frame] = excitation_for_target_activation(
            model,
            reached_activation[frame - 1],
            solved_activation[frame],
            frame_skip,
            excitation_low,
            excitation_high,
        )
        reached_activation[frame] = evolve_activation(
            model,
            reached_activation[frame - 1],
            solved_excitation[frame],
            frame_skip,
        )
    solved_exo = np.asarray(get_value(exo), dtype=np.float64).T
    metrics = branch_metrics(
        solved_activation,
        solved_exo,
        muscle_maps,
        exo_maps,
        target_torque,
        args.warmup,
    )
    dynamics_error = np.abs(reached_activation - solved_activation)
    summary = {
        "status": status,
        "solver_stats": {
            key: value
            for key, value in opti.stats().items()
            if key in {"return_status", "iter_count", "success", "t_proc_total"}
        },
        "frames": frames,
        "disable_exo": args.disable_exo,
        "fixed_exo_profile": (
            None if args.fixed_exo_profile is None else str(args.fixed_exo_profile)
        ),
        "exo_smooth_weight": args.exo_smooth_weight,
        "exo_curvature_weight": args.exo_curvature_weight,
        "exo_max_delta_nm_per_step": args.exo_max_delta_nm_per_step,
        "exo_max_curvature_nm_per_step2": args.exo_max_curvature_nm_per_step2,
        "exo_zero_initial_torque": args.exo_zero_initial_torque,
        "exo_half_cycle_frames": args.exo_half_cycle_frames,
        "effort_weight": args.effort_weight,
        "activation_anchor_weight": args.activation_anchor_weight,
        "torque_rmse_limit_nm": args.torque_rmse_limit_nm,
        "metrics": metrics,
        "dynamics_error_max": float(np.max(dynamics_error)),
    }
    np.savez_compressed(
        args.outdir / "solution.npz",
        phase=phase,
        activation=solved_activation,
        excitation=solved_excitation,
        exo_control=solved_exo,
        target_torque=target_torque,
        muscle_maps=muscle_maps,
        exo_maps=exo_maps,
    )
    (args.outdir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
