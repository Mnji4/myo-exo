#!/usr/bin/env python3
"""Distill a causal human response while keeping a recurrent Exo policy frozen."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from myo_exo_train.checkpoint import build_sac_actor_for_checkpoint  # noqa: E402
from myo_exo_train.env.model import build_muscle_model  # noqa: E402
from myo_exo_train.env.reference import load_reference_from_config  # noqa: E402
from myo_exo_train.env.runner import MJWarpMuscleRunner  # noqa: E402
from myo_exo_train.evaluation import load_config  # noqa: E402
from scripts.analyze_flat22_assisted_allocation_upper_bound import (  # noqa: E402
    configure_direct_exo,
)
from scripts.flat22_allocator_distillation_common import (  # noqa: E402
    ExoConditionedHumanStudent,
)


@dataclass
class Sequence:
    path: Path
    obs: torch.Tensor
    muscle: torch.Tensor
    exo_context: torch.Tensor


def command_history(action: np.ndarray, history_steps: int) -> np.ndarray:
    """Return causal histories ending in the current command, zero padded."""
    result = np.zeros((len(action), history_steps, 2), dtype=np.float32)
    for frame in range(len(action)):
        first = max(0, frame - history_steps + 1)
        values = action[first : frame + 1]
        result[frame, -len(values) :] = values
    return result.reshape(len(action), history_steps * 2)


def load_sequence(path: Path, history_steps: int) -> Sequence:
    with np.load(path) as payload:
        valid = np.asarray(payload["solver_success"], dtype=bool)
        obs = np.asarray(payload["normalized_obs"], dtype=np.float32)
        muscle = np.asarray(payload["muscle_action"], dtype=np.float32)
        exo = np.asarray(payload["exo_action"], dtype=np.float32)
        proprio = (
            np.asarray(payload["proprio_history"], dtype=np.float32)
            if "proprio_history" in payload
            else None
        )
    if proprio is not None and proprio.shape[1] == history_steps * 6:
        previous_commands = proprio.reshape(-1, history_steps, 6)[:, :, -2:]
        exo_context = np.concatenate(
            (previous_commands[:, 1:], exo[:, None, :]), axis=1
        ).reshape(-1, history_steps * 2)
    else:
        exo_context = command_history(exo, history_steps)
    return Sequence(
        path=path,
        obs=torch.from_numpy(obs[valid]),
        muscle=torch.from_numpy(muscle[valid]),
        exo_context=torch.from_numpy(exo_context[valid]),
    )


def concatenate(sequences: list[Sequence], name: str) -> torch.Tensor:
    return torch.cat([getattr(sequence, name) for sequence in sequences], dim=0)


def rmse(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(torch.square(prediction - target))).item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--datasets", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--muscle-only-student",
        type=Path,
        help="Optional low-activation no-Exo actor used as the human anchor.",
    )
    parser.add_argument("--frozen-exo-model", type=Path, required=True)
    parser.add_argument("--student-init", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history-steps", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument(
        "--absolute-output",
        action="store_true",
        help="Predict the complete muscle action instead of a bounded residual.",
    )
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--command-noise-std", type=float, default=0.015)
    parser.add_argument("--command-dropout", type=float, default=0.05)
    parser.add_argument("--init-anchor-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--train-all",
        action="store_true",
        help="Fit every nominal sample; use closed-loop rollouts for validation.",
    )
    args = parser.parse_args()

    if not args.datasets:
        raise ValueError("at least one trajectory is required")
    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))
    generator = torch.Generator().manual_seed(int(args.seed))

    config = copy.deepcopy(load_config(args.config))
    configure_direct_exo(config, 10.0, 1.0)
    config.setdefault("recovery_reset", {})["enabled"] = False
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
        seed=int(args.seed),
        device=device,
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    base_human, _, _ = build_sac_actor_for_checkpoint(
        checkpoint=checkpoint,
        model=model,
        config=config,
        obs_dim=runner.obs_dim,
        act_dim=runner.act_dim,
        device=device,
    )
    if args.muscle_only_student is not None:
        muscle_payload = torch.load(
            args.muscle_only_student, map_location=device, weights_only=False
        )
        base_human.load_state_dict(
            muscle_payload["human_actor_state_dict"], strict=True
        )
    base_human.requires_grad_(False)
    base_human.eval()

    exo_payload = torch.load(
        args.frozen_exo_model, map_location="cpu", weights_only=False
    )
    if "proprio_exo_state_dict" not in exo_payload:
        raise ValueError("--frozen-exo-model has no proprio Exo policy")
    if int(exo_payload["history_steps"]) != int(args.history_steps):
        raise ValueError("frozen Exo and human use different history lengths")

    sequences = [
        load_sequence(path, int(args.history_steps)) for path in args.datasets
    ]
    if args.train_all:
        train_obs = concatenate(sequences, "obs").to(device)
        train_target = concatenate(sequences, "muscle").to(device)
        train_context = concatenate(sequences, "exo_context").to(device)
        validation_obs = train_obs
        validation_target = train_target
        validation_context = train_context
        validation_path = "#all_training_data"
        training_trajectory_count = len(sequences)
    elif len(sequences) == 1:
        sequence = sequences[0]
        split = max(1, min(len(sequence.obs) - 1, int(0.9 * len(sequence.obs))))
        train_obs = sequence.obs[:split].to(device)
        train_target = sequence.muscle[:split].to(device)
        train_context = sequence.exo_context[:split].to(device)
        validation_obs = sequence.obs[split:].to(device)
        validation_target = sequence.muscle[split:].to(device)
        validation_context = sequence.exo_context[split:].to(device)
        validation_path = f"{sequence.path}#tail10pct"
        training_trajectory_count = 1
    else:
        training_sequences = sequences[:-1]
        validation_sequence = sequences[-1]
        train_obs = concatenate(training_sequences, "obs").to(device)
        train_target = concatenate(training_sequences, "muscle").to(device)
        train_context = concatenate(training_sequences, "exo_context").to(device)
        validation_obs = validation_sequence.obs.to(device)
        validation_target = validation_sequence.muscle.to(device)
        validation_context = validation_sequence.exo_context.to(device)
        validation_path = str(validation_sequence.path)
        training_trajectory_count = len(training_sequences)

    conditioned_human = ExoConditionedHumanStudent(
        runner.obs_dim,
        int(model.na),
        int(args.hidden_dim),
        exo_context_dim=int(args.history_steps) * 2,
        zero_centered=not bool(args.absolute_output),
        absolute_output=bool(args.absolute_output),
    ).to(device)
    if args.student_init is not None:
        initial_payload = torch.load(
            args.student_init, map_location=device, weights_only=False
        )
        if (
            initial_payload.get("model_type")
            != "frozen_recurrent_exo_conditioned_human"
        ):
            raise ValueError("--student-init has an incompatible model type")
        conditioned_human.load_state_dict(
            initial_payload["conditioned_human_state_dict"], strict=True
        )
    optimizer = torch.optim.AdamW(
        conditioned_human.parameters(),
        lr=float(args.learning_rate),
        weight_decay=1.0e-6,
    )
    with torch.no_grad():
        train_base, _, _, _ = base_human.get_action_and_value(
            train_obs, deterministic=True
        )
        validation_base, _, _, _ = base_human.get_action_and_value(
            validation_obs, deterministic=True
        )
        train_base = train_base[:, : int(model.na)]
        validation_base = validation_base[:, : int(model.na)]
        train_anchor = (
            conditioned_human(train_obs, train_base, train_context).detach()
            if args.student_init is not None
            else None
        )
        validation_initial = conditioned_human(
            validation_obs, validation_base, validation_context
        ).detach()

    best_score = rmse(validation_initial, validation_target)
    best_epoch = 0
    best_state = copy.deepcopy(conditioned_human.state_dict())
    last_metrics: dict[str, float] = {}
    for epoch in range(1, int(args.epochs) + 1):
        conditioned_human.train()
        order = torch.randperm(len(train_obs), generator=generator)
        for start in range(0, len(order), int(args.batch_size)):
            index = order[start : start + int(args.batch_size)].to(device)
            context = train_context[index]
            noisy_context = torch.clamp(
                context + float(args.command_noise_std) * torch.randn_like(context),
                -1.0,
                1.0,
            )
            if float(args.command_dropout) > 0.0:
                keep = (
                    torch.rand((len(index), 1), device=device)
                    >= float(args.command_dropout)
                )
                noisy_context = noisy_context * keep
            prediction = conditioned_human(
                train_obs[index], train_base[index], noisy_context
            )
            loss = F.smooth_l1_loss(prediction, train_target[index], beta=0.1)
            if train_anchor is not None and float(args.init_anchor_weight) > 0.0:
                loss = loss + float(args.init_anchor_weight) * F.smooth_l1_loss(
                    prediction, train_anchor[index], beta=0.1
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(conditioned_human.parameters(), 1.0)
            optimizer.step()

        if epoch % 10 == 0 or epoch == int(args.epochs):
            conditioned_human.eval()
            with torch.no_grad():
                validation_prediction = conditioned_human(
                    validation_obs, validation_base, validation_context
                )
                score = rmse(validation_prediction, validation_target)
                base_score = rmse(validation_base, validation_target)
                zero_prediction = conditioned_human(
                    validation_obs,
                    validation_base,
                    torch.zeros_like(validation_context),
                )
                last_metrics = {
                    "validation_rmse": score,
                    "validation_base_rmse": base_score,
                    "validation_zero_anchor_rmse": rmse(
                        zero_prediction, validation_base
                    ),
                }
            if score < best_score:
                best_score = score
                best_epoch = epoch
                best_state = copy.deepcopy(conditioned_human.state_dict())
        if epoch % 100 == 0:
            print(
                json.dumps({"epoch": epoch, "best_epoch": best_epoch, **last_metrics}),
                flush=True,
            )

    if best_state is None:
        raise RuntimeError("training did not produce a validation checkpoint")
    conditioned_human.load_state_dict(best_state)
    conditioned_human.eval()
    with torch.no_grad():
        prediction = conditioned_human(
            validation_obs, validation_base, validation_context
        )
        metrics = {
            "train_trajectories": training_trajectory_count,
            "train_samples": len(train_obs),
            "validation_trajectory": validation_path,
            "validation_samples": len(validation_obs),
            "validation_rmse": rmse(prediction, validation_target),
            "validation_base_rmse": rmse(validation_base, validation_target),
            "best_epoch": best_epoch,
        }

    payload = {
        "model_type": "frozen_recurrent_exo_conditioned_human",
        "human_actor_state_dict": base_human.state_dict(),
        "conditioned_human_state_dict": conditioned_human.state_dict(),
        "obs_dim": int(runner.obs_dim),
        "hidden_dim": int(args.hidden_dim),
        "history_steps": int(args.history_steps),
        "exo_context_dim": int(args.history_steps) * 2,
        "exo_sensor_mode": "hip4_exo6",
        "conditioned_zero_centered": not bool(args.absolute_output),
        "conditioned_absolute_output": bool(args.absolute_output),
        "frozen_exo_model": str(args.frozen_exo_model.resolve()),
        "source_checkpoint": str(args.checkpoint),
        "muscle_only_student": (
            None
            if args.muscle_only_student is None
            else str(args.muscle_only_student)
        ),
        "student_init": (
            None if args.student_init is None else str(args.student_init)
        ),
        "datasets": [str(path) for path in args.datasets],
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
