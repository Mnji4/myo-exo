#!/usr/bin/env python3
"""Optimize a smooth Exo profile over a fixed successful course22 trajectory."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from myo_exo_train.env.model import (  # noqa: E402
    build_muscle_model,
    muscle_action_mapping_mode,
    policy_action_to_ctrl,
)
from myo_exo_train.evaluation import load_config  # noqa: E402
from scripts.analyze_flat22_assisted_allocation_upper_bound import (  # noqa: E402
    configure_direct_exo,
)
from scripts.eval_dynamic_muscle_allocator import (  # noqa: E402
    JOINTS,
    active_torque_map,
    excitation_for_target_activation,
    exo_torque_map,
    evolve_activation,
)


def torch_activation_step(
    activation: torch.Tensor,
    excitation: torch.Tensor,
    *,
    activation_tau: torch.Tensor,
    deactivation_tau: torch.Tensor,
    timestep: float,
    frame_skip: int,
) -> torch.Tensor:
    """Match mju_muscleDynamics for smoothing=0 over one policy frame."""
    value = activation
    for _ in range(frame_skip):
        scale = 0.5 + 1.5 * value
        tau = torch.where(
            excitation > value,
            activation_tau * scale,
            deactivation_tau / scale,
        )
        value = torch.clamp(value + timestep * (excitation - value) / tau, 0.0, 1.0)
    return value


def dynamic_bounds(
    activation: torch.Tensor,
    excitation_low: torch.Tensor,
    excitation_high: torch.Tensor,
    *,
    activation_tau: torch.Tensor,
    deactivation_tau: torch.Tensor,
    timestep: float,
    frame_skip: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    low = torch_activation_step(
        activation,
        excitation_low.expand_as(activation),
        activation_tau=activation_tau,
        deactivation_tau=deactivation_tau,
        timestep=timestep,
        frame_skip=frame_skip,
    )
    high = torch_activation_step(
        activation,
        excitation_high.expand_as(activation),
        activation_tau=activation_tau,
        deactivation_tau=deactivation_tau,
        timestep=timestep,
        frame_skip=frame_skip,
    )
    return low, high


def inverse_sigmoid(value: torch.Tensor, epsilon: float = 1.0e-5) -> torch.Tensor:
    value = value.clamp(epsilon, 1.0 - epsilon)
    return torch.log(value) - torch.log1p(-value)


def activation_from_raw(raw: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(raw)


def interpolate_exo(knots: torch.Tensor, frames: int, max_control: float) -> torch.Tensor:
    values = torch.tanh(knots).transpose(0, 1).unsqueeze(0)
    profile = F.interpolate(values, size=frames, mode="linear", align_corners=True)
    return float(max_control) * profile.squeeze(0).transpose(0, 1)


def optimize_branch(
    *,
    nominal_activation: torch.Tensor,
    muscle_maps: torch.Tensor,
    exo_maps: torch.Tensor,
    target_torque: torch.Tensor,
    torque_scale: torch.Tensor,
    excitation_low: torch.Tensor,
    excitation_high: torch.Tensor,
    activation_tau: torch.Tensor,
    deactivation_tau: torch.Tensor,
    timestep: float,
    frame_skip: int,
    iterations: int,
    learning_rate: float,
    torque_weight: float,
    dynamics_weight: float,
    exo_smooth_weight: float,
    exo_curvature_weight: float,
    exo_l2_weight: float,
    exo_knot_stride: int,
    max_exo_control: float,
    initial_activation: torch.Tensor | None,
    assisted: bool,
    fixed_exo: torch.Tensor | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    frames, muscle_count = nominal_activation.shape
    initial = nominal_activation if initial_activation is None else initial_activation
    activation_raw = torch.nn.Parameter(inverse_sigmoid(initial[1:].detach().clone()))
    parameters: list[torch.nn.Parameter] = [activation_raw]
    exo_knots = None
    if assisted and fixed_exo is None:
        knot_count = max(2, math.ceil((frames - 1) / exo_knot_stride) + 1)
        exo_knots = torch.nn.Parameter(
            torch.zeros(
                (knot_count, exo_maps.shape[2]),
                device=nominal_activation.device,
                dtype=nominal_activation.dtype,
            )
        )
        parameters.append(exo_knots)
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)

    best_loss = float("inf")
    best_activation = None
    best_exo = None
    last_metrics: dict[str, float] = {}
    for iteration in range(iterations):
        activation = torch.cat(
            (nominal_activation[:1], activation_from_raw(activation_raw)), dim=0
        )
        if fixed_exo is not None:
            exo = fixed_exo
        elif exo_knots is None:
            exo = torch.zeros(
                (frames, exo_maps.shape[2]),
                device=nominal_activation.device,
                dtype=nominal_activation.dtype,
            )
        else:
            exo = interpolate_exo(exo_knots, frames, max_exo_control)

        predicted_torque = torch.einsum("tjm,tm->tj", muscle_maps, activation)
        predicted_torque = predicted_torque + torch.einsum(
            "tje,te->tj", exo_maps, exo
        )
        normalized_torque_error = (predicted_torque - target_torque) / torque_scale
        low, high = dynamic_bounds(
            activation[:-1],
            excitation_low,
            excitation_high,
            activation_tau=activation_tau,
            deactivation_tau=deactivation_tau,
            timestep=timestep,
            frame_skip=frame_skip,
        )
        dynamics_violation = torch.relu(low - activation[1:]) + torch.relu(
            activation[1:] - high
        )

        effort = torch.mean(torch.square(activation))
        torque_loss = torch.mean(torch.square(normalized_torque_error))
        dynamics_loss = torch.mean(torch.square(dynamics_violation))
        if frames > 1:
            exo_delta = exo[1:] - exo[:-1]
            smooth_loss = torch.mean(torch.square(exo_delta))
        else:
            smooth_loss = torch.zeros(
                (), device=activation.device, dtype=activation.dtype
            )
        if frames > 2:
            exo_curvature = exo[2:] - 2.0 * exo[1:-1] + exo[:-2]
            curvature_loss = torch.mean(torch.square(exo_curvature))
        else:
            curvature_loss = torch.zeros(
                (), device=activation.device, dtype=activation.dtype
            )
        exo_l2 = torch.mean(torch.square(exo))
        loss = (
            effort
            + torque_weight * torque_loss
            + dynamics_weight * dynamics_loss
            + exo_smooth_weight * smooth_loss
            + exo_curvature_weight * curvature_loss
            + exo_l2_weight * exo_l2
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 10.0)
        optimizer.step()

        loss_value = float(loss.item())
        if loss_value < best_loss:
            best_loss = loss_value
            best_activation = activation.detach().cpu().numpy().copy()
            best_exo = exo.detach().cpu().numpy().copy()
        if iteration == 0 or (iteration + 1) % 250 == 0:
            torque_rmse_nm = torch.sqrt(
                torch.mean(torch.square(predicted_torque - target_torque))
            )
            last_metrics = {
                "iteration": iteration + 1,
                "loss": loss_value,
                "activation_l2": float(effort.item()),
                "torque_rmse_nm": float(torque_rmse_nm.item()),
                "dynamics_violation_max": float(dynamics_violation.max().item()),
                "exo_abs_mean_control": float(exo.abs().mean().item()),
                "exo_delta_rms_control": float(torch.sqrt(smooth_loss).item()),
            }
            print(json.dumps(last_metrics), flush=True)

    if best_activation is None or best_exo is None:
        raise RuntimeError("optimizer produced no solution")
    last_metrics["best_loss"] = best_loss
    return best_activation, best_exo, last_metrics


def validate_activation_sequence(
    *,
    model: mujoco.MjModel,
    target_activation: np.ndarray,
    initial_activation: np.ndarray,
    excitation_low: np.ndarray,
    excitation_high: np.ndarray,
    frame_skip: int,
) -> tuple[np.ndarray, np.ndarray]:
    actual = np.empty_like(target_activation)
    excitation = np.empty_like(target_activation)
    actual[0] = initial_activation
    excitation[0] = initial_activation
    for frame in range(1, target_activation.shape[0]):
        excitation[frame] = excitation_for_target_activation(
            model,
            actual[frame - 1],
            target_activation[frame],
            frame_skip,
            excitation_low,
            excitation_high,
        )
        actual[frame] = evolve_activation(
            model, actual[frame - 1], excitation[frame], frame_skip
        )
    return actual, excitation


def branch_metrics(
    activation: np.ndarray,
    exo: np.ndarray,
    muscle_maps: np.ndarray,
    exo_maps: np.ndarray,
    target_torque: np.ndarray,
    warmup: int,
) -> dict[str, float]:
    predicted = np.einsum("tjm,tm->tj", muscle_maps, activation)
    predicted += np.einsum("tje,te->tj", exo_maps, exo)
    torque_error = predicted - target_torque
    start = min(max(0, warmup), activation.shape[0] - 1)
    selected = slice(start, None)
    muscle_torque = np.einsum(
        "tjm,tm->tj", muscle_maps[selected], activation[selected]
    )
    exo_torque = np.einsum("tje,te->tj", exo_maps[selected], exo[selected])
    target_rms = float(np.sqrt(np.mean(np.square(target_torque[selected]))))
    torque_rmse = float(np.sqrt(np.mean(np.square(torque_error[selected]))))
    opposition = np.where(
        muscle_torque[:, [0, 3]] * exo_torque[:, [0, 3]] < 0.0,
        np.minimum(
            np.abs(muscle_torque[:, [0, 3]]),
            np.abs(exo_torque[:, [0, 3]]),
        ),
        0.0,
    )
    return {
        "activation_l2": float(np.mean(np.square(activation[selected]))),
        "hip_activation_l2": float(
            np.mean(
                np.square(
                    activation[selected][
                        :,
                        np.any(
                            np.abs(muscle_maps[selected][0, [0, 3], :]) > 1.0e-8,
                            axis=0,
                        ),
                    ]
                )
            )
        ),
        "target_torque_rms_nm": target_rms,
        "torque_rmse_nm": torque_rmse,
        "torque_relative_rmse": torque_rmse / max(target_rms, 1.0e-12),
        "torque_error_max_nm": float(np.max(np.abs(torque_error[selected]))),
        "exo_abs_mean_control": float(np.mean(np.abs(exo[selected]))),
        "exo_hip_torque_abs_mean_nm": float(
            np.mean(np.abs(exo_torque[:, [0, 3]]))
        ),
        "exo_delta_rms_control": float(
            np.sqrt(np.mean(np.square(np.diff(exo[selected], axis=0))))
        ),
        "opposed_torque_nm": float(np.mean(opposition)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations-noexo", type=int, default=2500)
    parser.add_argument("--iterations-assisted", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=1.0e-2)
    parser.add_argument("--torque-weight", type=float, default=2.0e4)
    parser.add_argument("--dynamics-weight", type=float, default=2.0e4)
    parser.add_argument("--exo-smooth-weight", type=float, default=2.0e-3)
    parser.add_argument("--exo-curvature-weight", type=float, default=1.0e-2)
    parser.add_argument("--exo-l2-weight", type=float, default=1.0e-6)
    parser.add_argument("--exo-knot-stride", type=int, default=8)
    parser.add_argument("--exo-max-torque-nm", type=float, default=10.0)
    parser.add_argument(
        "--fixed-exo-profile",
        type=Path,
        help="Optimize only muscle activation while applying this fixed Exo command.",
    )
    parser.add_argument(
        "--baseline-solution",
        type=Path,
        help="Reuse the matched muscle-only branch from an existing solution.",
    )
    parser.add_argument(
        "--target-includes-bank-exo",
        action="store_true",
        help=(
            "Match the source trajectory's total active joint torque by adding "
            "its recorded applied_exo_ctrl contribution to the muscle torque."
        ),
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    config = load_config(args.config)
    configure_direct_exo(config, args.exo_max_torque_nm, 1.0)
    model, probe_data = build_muscle_model(config)
    if int(model.na) != 22:
        raise ValueError(f"expected 22 muscles, got {model.na}")
    frame_skip = int(config["control"]["frame_skip"])
    with np.load(args.bank, allow_pickle=True) as bank:
        frames = len(bank["qpos"])
        if args.max_frames > 0:
            frames = min(frames, int(args.max_frames))
        qpos = np.asarray(bank["qpos"][:frames], dtype=np.float64)
        qvel = np.asarray(bank["qvel"][:frames], dtype=np.float64)
        nominal_activation = np.asarray(bank["act"][:frames], dtype=np.float64)
        phase = np.asarray(bank["phase"][:frames], dtype=np.int64)
        if args.target_includes_bank_exo:
            if "applied_exo_ctrl" not in bank:
                raise ValueError(
                    "--target-includes-bank-exo requires applied_exo_ctrl in bank"
                )
            source_exo = np.asarray(
                bank["applied_exo_ctrl"][:frames], dtype=np.float64
            )
        else:
            source_exo = np.zeros((frames, 2), dtype=np.float64)

    fixed_exo = None
    if args.fixed_exo_profile is not None:
        with np.load(args.fixed_exo_profile, allow_pickle=True) as profile:
            key = (
                "exo_control"
                if "exo_control" in profile
                else "assisted_exo_control"
            )
            fixed_exo = np.asarray(profile[key], dtype=np.float64)
        if fixed_exo.shape != (frames, 2):
            raise ValueError(
                f"fixed Exo profile has shape {fixed_exo.shape}, expected {(frames, 2)}"
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
    print(f"building torque maps for {frames} frames", flush=True)
    muscle_maps = np.stack(
        [
            active_torque_map(model, probe_data, qpos[i], qvel[i], joint_dofs)
            for i in range(frames)
        ]
    )
    exo_maps = np.stack(
        [
            exo_torque_map(model, probe_data, qpos[i], qvel[i], joint_dofs)
            for i in range(frames)
        ]
    )
    target_torque = np.einsum("tjm,tm->tj", muscle_maps, nominal_activation)
    target_torque += np.einsum("tje,te->tj", exo_maps, source_exo)
    torque_scale = np.maximum(np.abs(target_torque), 10.0)

    mapping = muscle_action_mapping_mode(config)
    action_limits = torch.tensor(
        [[-1.0] * int(model.na), [1.0] * int(model.na)], dtype=torch.float64
    )
    ctrl_low = torch.from_numpy(model.actuator_ctrlrange[: int(model.na), 0]).double()
    ctrl_high = torch.from_numpy(model.actuator_ctrlrange[: int(model.na), 1]).double()
    ctrl_limits = policy_action_to_ctrl(
        action_limits,
        ctrl_low,
        ctrl_high,
        muscle_count=int(model.na),
        muscle_mapping=mapping,
    ).numpy()
    excitation_low = ctrl_limits[0]
    excitation_high = ctrl_limits[1]

    dynprm = np.asarray(model.actuator_dynprm[: int(model.na), :3], dtype=np.float64)
    if np.max(np.abs(dynprm[:, 2])) > 1.0e-12:
        raise ValueError("torch dynamics currently require muscle smoothing=0")
    common = {
        "nominal_activation": torch.from_numpy(nominal_activation).to(
            device=device, dtype=torch.float64
        ),
        "muscle_maps": torch.from_numpy(muscle_maps).to(
            device=device, dtype=torch.float64
        ),
        "exo_maps": torch.from_numpy(exo_maps).to(device=device, dtype=torch.float64),
        "target_torque": torch.from_numpy(target_torque).to(
            device=device, dtype=torch.float64
        ),
        "torque_scale": torch.from_numpy(torque_scale).to(
            device=device, dtype=torch.float64
        ),
        "excitation_low": torch.from_numpy(excitation_low).to(
            device=device, dtype=torch.float64
        ),
        "excitation_high": torch.from_numpy(excitation_high).to(
            device=device, dtype=torch.float64
        ),
        "activation_tau": torch.from_numpy(dynprm[:, 0]).to(
            device=device, dtype=torch.float64
        ),
        "deactivation_tau": torch.from_numpy(dynprm[:, 1]).to(
            device=device, dtype=torch.float64
        ),
        "timestep": float(model.opt.timestep),
        "frame_skip": frame_skip,
        "learning_rate": float(args.learning_rate),
        "torque_weight": float(args.torque_weight),
        "dynamics_weight": float(args.dynamics_weight),
        "exo_smooth_weight": float(args.exo_smooth_weight),
        "exo_curvature_weight": float(args.exo_curvature_weight),
        "exo_l2_weight": float(args.exo_l2_weight),
        "exo_knot_stride": int(args.exo_knot_stride),
        "max_exo_control": 1.0,
    }

    if args.baseline_solution is not None:
        with np.load(args.baseline_solution, allow_pickle=True) as baseline:
            baseline_phase = np.asarray(baseline["phase"][:frames], dtype=np.int64)
            if not np.array_equal(baseline_phase, phase):
                raise ValueError("baseline solution phases do not match the state bank")
            noexo_activation = np.asarray(
                baseline["muscle_only_activation"][:frames], dtype=np.float64
            )
            noexo_excitation = np.asarray(
                baseline["muscle_only_excitation"][:frames], dtype=np.float64
            )
        noexo_exo = np.zeros((frames, 2), dtype=np.float64)
        noexo_optimizer = {"reused_baseline_solution": 1.0}
        print("reusing matched muscle-only horizon", flush=True)
    else:
        print("optimizing matched muscle-only horizon", flush=True)
        noexo_target, noexo_exo, noexo_optimizer = optimize_branch(
            **common,
            iterations=int(args.iterations_noexo),
            initial_activation=None,
            assisted=False,
        )
        noexo_activation, noexo_excitation = validate_activation_sequence(
            model=model,
            target_activation=noexo_target,
            initial_activation=nominal_activation[0],
            excitation_low=excitation_low,
            excitation_high=excitation_high,
            frame_skip=frame_skip,
        )

    branch_name = "fixed" if fixed_exo is not None else "optimized"
    print(f"optimizing {branch_name} Exo-assisted horizon", flush=True)
    assisted_target, assisted_exo, assisted_optimizer = optimize_branch(
        **common,
        iterations=int(args.iterations_assisted),
        initial_activation=torch.from_numpy(noexo_activation).to(
            device=device, dtype=torch.float64
        ),
        assisted=True,
        fixed_exo=(
            None
            if fixed_exo is None
            else torch.from_numpy(fixed_exo).to(device=device, dtype=torch.float64)
        ),
    )
    assisted_activation, assisted_excitation = validate_activation_sequence(
        model=model,
        target_activation=assisted_target,
        initial_activation=nominal_activation[0],
        excitation_low=excitation_low,
        excitation_high=excitation_high,
        frame_skip=frame_skip,
    )

    nominal_exo = source_exo
    metrics = {
        "frames": frames,
        "phase_start": int(phase[0]),
        "phase_end": int(phase[-1]),
        "warmup": int(args.warmup),
        "target_includes_bank_exo": bool(args.target_includes_bank_exo),
        "fixed_exo_profile": (
            None if args.fixed_exo_profile is None else str(args.fixed_exo_profile)
        ),
        "baseline_solution": (
            None if args.baseline_solution is None else str(args.baseline_solution)
        ),
        "nominal": branch_metrics(
            nominal_activation,
            nominal_exo,
            muscle_maps,
            exo_maps,
            target_torque,
            args.warmup,
        ),
        "muscle_only": branch_metrics(
            noexo_activation,
            noexo_exo,
            muscle_maps,
            exo_maps,
            target_torque,
            args.warmup,
        ),
        "assisted": branch_metrics(
            assisted_activation,
            assisted_exo,
            muscle_maps,
            exo_maps,
            target_torque,
            args.warmup,
        ),
        "optimizer": {
            "muscle_only": noexo_optimizer,
            "assisted": assisted_optimizer,
        },
    }
    metrics["comparison"] = {
        "muscle_only_reduction_vs_nominal": 1.0
        - metrics["muscle_only"]["activation_l2"]
        / metrics["nominal"]["activation_l2"],
        "assisted_reduction_vs_nominal": 1.0
        - metrics["assisted"]["activation_l2"] / metrics["nominal"]["activation_l2"],
        "exo_incremental_reduction_vs_muscle_only": 1.0
        - metrics["assisted"]["activation_l2"]
        / metrics["muscle_only"]["activation_l2"],
        "exo_incremental_hip_reduction_vs_muscle_only": 1.0
        - metrics["assisted"]["hip_activation_l2"]
        / metrics["muscle_only"]["hip_activation_l2"],
    }

    args.outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.outdir / "solution.npz",
        phase=phase,
        nominal_activation=nominal_activation,
        muscle_only_activation=noexo_activation,
        muscle_only_excitation=noexo_excitation,
        assisted_activation=assisted_activation,
        assisted_excitation=assisted_excitation,
        assisted_exo_control=assisted_exo,
        source_exo_control=source_exo,
        target_torque=target_torque,
        muscle_maps=muscle_maps,
        exo_maps=exo_maps,
    )
    with (args.outdir / "timeseries.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = [
            "frame",
            "phase",
            "nominal_activation_l2",
            "muscle_only_activation_l2",
            "assisted_activation_l2",
            "assisted_exo_r_control",
            "assisted_exo_l_control",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for frame in range(frames):
            writer.writerow(
                {
                    "frame": frame,
                    "phase": int(phase[frame]),
                    "nominal_activation_l2": float(
                        np.mean(np.square(nominal_activation[frame]))
                    ),
                    "muscle_only_activation_l2": float(
                        np.mean(np.square(noexo_activation[frame]))
                    ),
                    "assisted_activation_l2": float(
                        np.mean(np.square(assisted_activation[frame]))
                    ),
                    "assisted_exo_r_control": float(assisted_exo[frame, 0]),
                    "assisted_exo_l_control": float(assisted_exo[frame, 1]),
                }
            )
    (args.outdir / "summary.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
