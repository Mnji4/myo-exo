#!/usr/bin/env python3
"""Train one proprio-history Exo policy from all course22 terrain teachers."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.flat22_allocator_distillation_common import ExoStudent  # noqa: E402


def load_datasets(
    paths: list[Path],
) -> tuple[torch.Tensor, torch.Tensor, int, str, list[int]]:
    proprio_parts: list[np.ndarray] = []
    action_parts: list[np.ndarray] = []
    counts: list[int] = []
    history_steps: set[int] = set()
    sensor_modes: set[str] = set()
    for path in paths:
        with np.load(path) as data:
            valid = np.asarray(data["solver_success"], dtype=bool)
            if "exo_supervision_mask" in data:
                valid &= np.asarray(data["exo_supervision_mask"], dtype=bool)
            proprio = np.asarray(data["proprio_history"], dtype=np.float32)[valid]
            action = np.asarray(data["exo_action"], dtype=np.float32)[valid]
            proprio_parts.append(proprio)
            action_parts.append(action)
            counts.append(len(proprio))
            history_steps.add(int(np.asarray(data["history_steps"]).reshape(-1)[0]))
            sensor_modes.add(str(np.asarray(data["exo_sensor_mode"]).reshape(-1)[0]))
    if len(history_steps) != 1 or len(sensor_modes) != 1:
        raise ValueError(
            f"inconsistent dataset history: {history_steps}, {sensor_modes}"
        )
    return (
        torch.from_numpy(np.concatenate(proprio_parts)),
        torch.from_numpy(np.concatenate(action_parts)),
        history_steps.pop(),
        sensor_modes.pop(),
        counts,
    )


def metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    error = prediction - target
    return {
        "rmse": float(torch.sqrt(torch.mean(torch.square(error))).item()),
        "mae": float(torch.mean(torch.abs(error)).item()),
        "p95_abs": float(torch.quantile(torch.abs(error).flatten(), 0.95).item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--exo-policy-input-mode",
        choices=("kinematic_history", "kinematic_command_history"),
        default="kinematic_history",
    )
    parser.add_argument(
        "--train-all",
        action="store_true",
        help="Fit the complete fixed trajectory; report training-set fit.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    proprio, action, history_steps, sensor_mode, counts = load_datasets(
        args.datasets
    )
    if args.exo_policy_input_mode == "kinematic_history":
        frame_dim = proprio.shape[1] // history_steps
        if frame_dim < 4 or proprio.shape[1] % history_steps != 0:
            raise ValueError(f"invalid proprio history shape: {tuple(proprio.shape)}")
        proprio = proprio.reshape(-1, history_steps, frame_dim)[:, :, :4].reshape(
            -1, history_steps * 4
        )
    generator = torch.Generator().manual_seed(int(args.seed))
    order = torch.randperm(len(proprio), generator=generator)
    if args.train_all:
        train_indices = order
        validation_indices = order
    else:
        split = max(1, int(0.9 * len(order)))
        train_indices = order[:split]
        validation_indices = order[split:]
        if len(validation_indices) == 0:
            validation_indices = train_indices[-1:]
    mean = proprio[train_indices].mean(dim=0)
    std = proprio[train_indices].std(dim=0, unbiased=False).clamp_min(1.0e-4)

    model = ExoStudent(proprio.shape[1], int(args.hidden_dim)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.learning_rate), weight_decay=1.0e-6
    )
    model.train()
    for epoch in range(int(args.epochs)):
        shuffled = train_indices[
            torch.randperm(len(train_indices), generator=generator)
        ]
        for start in range(0, len(shuffled), int(args.batch_size)):
            index = shuffled[start : start + int(args.batch_size)]
            inputs = ((proprio[index] - mean) / std).to(device)
            target = action[index].to(device)
            loss = torch.mean(torch.square(model(inputs) - target))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(
                json.dumps({"epoch": epoch + 1, "loss": float(loss.item())}),
                flush=True,
            )

    model.eval()
    with torch.no_grad():
        prediction = model(
            ((proprio[validation_indices] - mean) / std).to(device)
        ).cpu()
    result = {
        "samples": len(proprio),
        "train_samples": len(train_indices),
        "validation_samples": len(validation_indices),
        "dataset_sample_counts": counts,
        "validation_is_training_data": bool(args.train_all),
        "proprio_history_exo_action": metrics(
            prediction, action[validation_indices]
        ),
    }
    payload = {
        "proprio_exo_state_dict": model.state_dict(),
        "proprio_dim": int(proprio.shape[1]),
        "history_steps": int(history_steps),
        "exo_sensor_mode": str(sensor_mode),
        "exo_policy_input_mode": str(args.exo_policy_input_mode),
        "hidden_dim": int(args.hidden_dim),
        "proprio_mean": mean,
        "proprio_std": std,
        "datasets": [str(path) for path in args.datasets],
        "metrics": result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
