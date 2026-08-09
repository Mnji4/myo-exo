#!/usr/bin/env python3
"""Compare distilled Exo students and fit symmetric target-angle PD policies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


class ExoStudent(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
            nn.Tanh(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


class SharedLegTargetPolicy(nn.Module):
    """One shared network applied to right and mirrored-left sensor histories."""

    def __init__(self, input_dim: int, hidden_dim: int, offset_limit: float) -> None:
        super().__init__()
        self.offset_limit = float(offset_limit)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.offset_limit * self.network(value).squeeze(-1)


def parse_student(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("expected NAME=student.pt")
    return name, Path(path)


def rate_limit_control(desired: np.ndarray, max_delta: float) -> np.ndarray:
    applied = np.empty_like(desired)
    previous = np.zeros(2, dtype=np.float64)
    for frame in range(len(desired)):
        previous = np.clip(desired[frame], previous - max_delta, previous + max_delta)
        applied[frame] = previous
    return applied


def student_torque(
    path: Path, history: np.ndarray, max_delta_control: float
) -> tuple[np.ndarray, np.ndarray]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = ExoStudent(int(payload["proprio_dim"]), int(payload["hidden_dim"]))
    model.load_state_dict(payload["proprio_exo_state_dict"])
    model.eval()
    mean = np.asarray(payload["proprio_mean"], dtype=np.float32)
    std = np.asarray(payload["proprio_std"], dtype=np.float32)
    if payload.get("exo_policy_input_mode") == "kinematic_history":
        flat = history[:, :, :4].reshape(len(history), -1).astype(np.float32)
    else:
        flat = history.reshape(len(history), -1).astype(np.float32)
    with torch.no_grad():
        command = model(torch.from_numpy((flat - mean) / std)).numpy()
    desired = command.astype(np.float64)
    applied = rate_limit_control(desired, max_delta_control)
    return 10.0 * desired, 10.0 * applied


def canonical_leg_inputs(history: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    right = history.copy()
    left = history[:, :, [1, 0, 3, 2, 5, 4]]
    return right.reshape(len(history), -1), left.reshape(len(history), -1)


def fit_target_pd(
    history: np.ndarray,
    teacher_torque: np.ndarray,
    *,
    kp: float,
    kd: float,
    offset_limit: float,
    hidden_dim: int,
    iterations: int,
    learning_rate: float,
    device: torch.device,
) -> tuple[SharedLegTargetPolicy, np.ndarray, dict[str, object]]:
    right, left = canonical_leg_inputs(history)
    features = np.concatenate((right, left), axis=0).astype(np.float32)
    angle = np.concatenate((history[:, -1, 0], history[:, -1, 1])).astype(
        np.float32
    )
    velocity = np.concatenate((history[:, -1, 2], history[:, -1, 3])).astype(
        np.float32
    )
    target_torque = np.concatenate(
        (teacher_torque[:, 0], teacher_torque[:, 1])
    ).astype(np.float32)
    mean = features.mean(axis=0)
    std = np.maximum(features.std(axis=0), 1.0e-4)
    feature_tensor = torch.from_numpy((features - mean) / std).to(device)
    angle_tensor = torch.from_numpy(angle).to(device)
    velocity_tensor = torch.from_numpy(velocity).to(device)
    target_tensor = torch.from_numpy(target_torque).to(device)

    model = SharedLegTargetPolicy(features.shape[1], hidden_dim, offset_limit).to(
        device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for _ in range(iterations):
        offset = model(feature_tensor)
        predicted = torch.clamp(kp * offset + kd * velocity_tensor, -10.0, 10.0)
        loss = torch.mean(torch.square(predicted - target_tensor))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        offset = model(feature_tensor)
        predicted = torch.clamp(kp * offset + kd * velocity_tensor, -10.0, 10.0)
    frames = len(history)
    torque = torch.stack((predicted[:frames], predicted[frames:]), dim=1).cpu().numpy()
    target_angle = torch.stack(
        (angle_tensor[:frames] + offset[:frames], angle_tensor[frames:] + offset[frames:]),
        dim=1,
    ).cpu().numpy()
    error = torque - teacher_torque
    metrics: dict[str, object] = {
        "torque_rmse_nm": float(np.sqrt(np.mean(error**2))),
        "torque_mae_nm": float(np.mean(np.abs(error))),
        "torque_correlation": float(
            np.corrcoef(torque.reshape(-1), teacher_torque.reshape(-1))[0, 1]
        ),
        "teacher_abs_mean_nm": float(np.mean(np.abs(teacher_torque))),
        "pd_abs_mean_nm": float(np.mean(np.abs(torque))),
        "target_offset_abs_mean_deg": float(
            np.degrees(np.mean(np.abs(target_angle - history[:, -1, :2])))
        ),
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "target_angle": target_angle,
    }
    return model.cpu(), torque, metrics


def half_cycle_metrics(torque: np.ndarray, lag: int) -> dict[str, float]:
    right = torque[:-lag, 0]
    left = torque[lag:, 1]
    rmse = float(np.sqrt(np.mean(np.square(right - left))))
    scale = float(np.sqrt(0.5 * np.mean(right**2 + left**2)))
    return {
        "half_cycle_frames": lag,
        "half_cycle_rmse_nm": rmse,
        "half_cycle_normalized_rmse": rmse / max(scale, 1.0e-8),
        "half_cycle_correlation": float(np.corrcoef(right, left)[0, 1]),
        "right_abs_mean_nm": float(np.mean(np.abs(torque[:, 0]))),
        "left_abs_mean_nm": float(np.mean(np.abs(torque[:, 1]))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--student", action="append", type=parse_student, required=True)
    parser.add_argument("--solver-label", action="append", type=parse_student, default=[])
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--control-hz", type=float, default=100.0)
    parser.add_argument("--half-cycle-frames", type=int, default=53)
    parser.add_argument("--kp", type=float, default=20.0)
    parser.add_argument("--kd", type=float, default=0.5)
    parser.add_argument("--target-offset-limit", type=float, default=0.75)
    parser.add_argument("--max-delta-control", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--zoom-seconds", type=float, default=1.2)
    args = parser.parse_args()
    torch.manual_seed(20260805)

    with np.load(args.bank, allow_pickle=True) as bank:
        history_steps = int(np.asarray(bank["history_steps"]).reshape(-1)[0])
        history = np.asarray(bank["proprio_history"], dtype=np.float64).reshape(
            -1, history_steps, 6
        )
    args.outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    results: dict[str, dict[str, object]] = {}
    curves: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    solver_labels = {}
    for name, path in args.solver_label:
        with np.load(path, allow_pickle=True) as dataset:
            solver_labels[name] = 10.0 * np.asarray(
                dataset["exo_action"], dtype=np.float64
            )
    for name, path in args.student:
        desired, teacher = student_torque(
            path, history, float(args.max_delta_control)
        )
        model, pd_torque, fit = fit_target_pd(
            history,
            teacher,
            kp=args.kp,
            kd=args.kd,
            offset_limit=args.target_offset_limit,
            hidden_dim=args.hidden_dim,
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            device=device,
        )
        target_angle = np.asarray(fit.pop("target_angle"))
        teacher_symmetry = half_cycle_metrics(teacher, args.half_cycle_frames)
        pd_symmetry = half_cycle_metrics(pd_torque, args.half_cycle_frames)
        results[name] = {
            "student": str(path),
            "raw_desired_abs_mean_nm": float(np.mean(np.abs(desired))),
            "raw_desired_frame_delta_rms_nm": float(
                np.sqrt(np.mean(np.square(np.diff(desired, axis=0))))
            ),
            "applied_frame_delta_rms_nm": float(
                np.sqrt(np.mean(np.square(np.diff(teacher, axis=0))))
            ),
            "fit": fit,
            "teacher_symmetry": teacher_symmetry,
            "pd_symmetry": pd_symmetry,
        }
        torch.save(
            {
                "model_type": "shared_leg_target_position_pd",
                "state_dict": model.state_dict(),
                "input_mean": fit["feature_mean"],
                "input_std": fit["feature_std"],
                "history_steps": history_steps,
                "kp_nm_per_rad": float(args.kp),
                "kd_nm_s_per_rad": float(args.kd),
                "velocity_sign": "+",
                "target_offset_limit_rad": float(args.target_offset_limit),
                "hidden_dim": int(args.hidden_dim),
                "source_student": str(path),
            },
            args.outdir / f"{name}_target_pd.pt",
        )
        curves[name] = (desired, teacher, pd_torque, target_angle)

    names = list(curves)
    time = np.arange(len(history), dtype=np.float64) / float(args.control_hz)
    figure, axes = plt.subplots(3, len(names), figsize=(7 * len(names), 10), sharex=True)
    if len(names) == 1:
        axes = axes[:, None]
    for column, name in enumerate(names):
        desired, teacher, pd_torque, target_angle = curves[name]
        for side, color in ((0, "#D55E00"), (1, "#0072B2")):
            label = "Right" if side == 0 else "Left"
            axes[0, column].plot(time, teacher[:, side], color=color, label=label)
            axes[0, column].plot(
                time, desired[:, side], color=color, alpha=0.18, linewidth=0.7
            )
            axes[1, column].plot(time, teacher[:, side], color=color, alpha=0.35)
            axes[1, column].plot(time, pd_torque[:, side], color=color, linestyle="--")
            axes[2, column].plot(
                time, np.degrees(history[:, -1, side]), color=color, alpha=0.3
            )
            axes[2, column].plot(
                time, np.degrees(target_angle[:, side]), color=color, linestyle="--"
            )
        axes[0, column].set_title(
            f"{name}: desired (faint) and rate-limited applied torque"
        )
        axes[1, column].set_title(
            f"PD fit, RMSE {results[name]['fit']['torque_rmse_nm']:.2f} Nm"
        )
        axes[2, column].set_title("Hip angle (faint) and target angle (dashed)")
        axes[0, column].legend(ncols=2)
        axes[0, column].set_ylabel("Torque (Nm)")
        axes[1, column].set_ylabel("Torque (Nm)")
        axes[2, column].set_ylabel("Angle (deg)")
        axes[2, column].set_xlabel("Time (s)")
        for axis in axes[:, column]:
            axis.axhline(0.0, color="#777777", linewidth=0.7)
            axis.grid(alpha=0.2)
    figure.suptitle(
        f"Direct Exo students and symmetric target-position PD conversion "
        f"(Kp={args.kp:g}, Kd={args.kd:g})"
    )
    figure.tight_layout()
    figure.savefig(args.outdir / "exo_rounds_and_target_pd.png", dpi=180)
    plt.close(figure)

    zoom_frames = min(
        len(history), max(2, int(round(args.zoom_seconds * args.control_hz)))
    )
    zoom_time = time[:zoom_frames]
    detail, detail_axes = plt.subplots(
        4, len(names), figsize=(7 * len(names), 12), sharex=True
    )
    if len(names) == 1:
        detail_axes = detail_axes[:, None]
    for column, name in enumerate(names):
        desired, applied, pd_torque, target_angle = curves[name]
        label = solver_labels.get(name)
        for side, color in ((0, "#D55E00"), (1, "#0072B2")):
            side_name = "R" if side == 0 else "L"
            if label is not None:
                detail_axes[0, column].plot(
                    zoom_time,
                    label[:zoom_frames, side],
                    color=color,
                    linewidth=2.0,
                    label=f"{side_name} solver label",
                )
            detail_axes[0, column].plot(
                zoom_time,
                desired[:zoom_frames, side],
                color=color,
                linestyle=":",
                label=f"{side_name} network desired",
            )
            detail_axes[1, column].plot(
                zoom_time,
                desired[:zoom_frames, side],
                color=color,
                alpha=0.25,
            )
            detail_axes[1, column].plot(
                zoom_time,
                applied[:zoom_frames, side],
                color=color,
                linewidth=1.8,
                label=f"{side_name} applied",
            )
            detail_axes[2, column].plot(
                zoom_time,
                applied[:zoom_frames, side],
                color=color,
                linewidth=2.0,
                label=f"{side_name} direct",
            )
            detail_axes[2, column].plot(
                zoom_time,
                pd_torque[:zoom_frames, side],
                color=color,
                linestyle="--",
                label=f"{side_name} target-PD",
            )
            detail_axes[3, column].plot(
                zoom_time,
                np.degrees(history[:zoom_frames, -1, side]),
                color=color,
                alpha=0.35,
                label=f"{side_name} hip",
            )
            detail_axes[3, column].plot(
                zoom_time,
                np.degrees(target_angle[:zoom_frames, side]),
                color=color,
                linestyle="--",
                label=f"{side_name} target",
            )
        detail_axes[0, column].set_title(f"{name}: solver label vs raw network")
        detail_axes[1, column].set_title("Raw desired vs rate-limited applied")
        detail_axes[2, column].set_title("Applied direct torque vs target-PD")
        detail_axes[3, column].set_title("Measured hip angle vs PD target angle")
        for row, axis in enumerate(detail_axes[:, column]):
            axis.grid(alpha=0.2)
            axis.axhline(0.0, color="#777777", linewidth=0.7)
            axis.legend(ncols=2, fontsize=8)
            axis.set_ylabel("deg" if row == 3 else "Nm")
        detail_axes[3, column].set_xlabel("Time (s)")
    detail.suptitle("Short-window Exo distillation and target-PD conversion")
    detail.tight_layout()
    detail.savefig(args.outdir / "exo_rounds_and_target_pd_zoom.png", dpi=180)
    plt.close(detail)

    (args.outdir / "summary.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
