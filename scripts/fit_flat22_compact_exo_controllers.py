#!/usr/bin/env python3
"""Fit compact, non-neural Exo controllers to a distilled flat-gait policy."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import lsq_linear


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_flat22_exo_students_target_pd import student_torque  # noqa: E402


def rate_limit(raw: np.ndarray, max_delta_nm: float) -> np.ndarray:
    output = np.empty_like(raw)
    previous = np.zeros(raw.shape[1], dtype=np.float64)
    for frame in range(len(raw)):
        previous = np.clip(raw[frame], previous - max_delta_nm, previous + max_delta_nm)
        previous = np.clip(previous, -10.0, 10.0)
        output[frame] = previous
    return output


def metrics(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    prediction = prediction[mask]
    target = target[mask]
    error = prediction - target
    correlation = float(np.corrcoef(prediction.reshape(-1), target.reshape(-1))[0, 1])
    return {
        "rmse_nm": float(np.sqrt(np.mean(np.square(error)))),
        "mae_nm": float(np.mean(np.abs(error))),
        "correlation": correlation,
        "predicted_abs_mean_nm": float(np.mean(np.abs(prediction))),
        "target_abs_mean_nm": float(np.mean(np.abs(target))),
    }


def stacked_leg_data(history: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    theta = np.concatenate((history[:, -1, 0], history[:, -1, 1]))
    velocity = np.concatenate((history[:, -1, 2], history[:, -1, 3]))
    return theta, velocity


def fit_state_feedback(
    history: np.ndarray,
    target: np.ndarray,
    train_frames: np.ndarray,
    *,
    shared: bool,
    physical_pd: bool,
) -> tuple[np.ndarray, list[float]]:
    theta, velocity = stacked_leg_data(history)
    values = np.concatenate((target[:, 0], target[:, 1]))
    train = np.concatenate((train_frames, train_frames))
    features = np.column_stack((np.ones_like(theta), theta, velocity))
    if shared:
        if physical_pd:
            fit = lsq_linear(
                features[train],
                values[train],
                bounds=([-np.inf, -np.inf, 0.0], [np.inf, 0.0, np.inf]),
            )
            coefficients = fit.x
        else:
            coefficients = np.linalg.lstsq(features[train], values[train], rcond=None)[0]
        raw = (features @ coefficients).reshape(2, len(history)).T
        return raw, coefficients.tolist()

    predictions = []
    coefficients = []
    for side, columns in enumerate(((0, 2), (1, 3))):
        side_features = np.column_stack(
            (
                np.ones(len(history)),
                history[:, -1, columns[0]],
                history[:, -1, columns[1]],
            )
        )
        if physical_pd:
            fit = lsq_linear(
                side_features[train_frames],
                target[train_frames, side],
                bounds=([-np.inf, -np.inf, 0.0], [np.inf, 0.0, np.inf]),
            )
            side_coefficients = fit.x
        else:
            side_coefficients = np.linalg.lstsq(
                side_features[train_frames], target[train_frames, side], rcond=None
            )[0]
        predictions.append(side_features @ side_coefficients)
        coefficients.extend(side_coefficients.tolist())
    return np.stack(predictions, axis=1), coefficients


def fit_autoregressive(
    history: np.ndarray,
    target: np.ndarray,
    train_frames: np.ndarray,
    *,
    shared: bool,
    max_delta_nm: float,
) -> tuple[np.ndarray, list[float]]:
    previous_target = np.vstack((np.zeros((1, 2)), target[:-1]))
    if shared:
        theta, velocity = stacked_leg_data(history)
        previous = np.concatenate((previous_target[:, 0], previous_target[:, 1]))
        values = np.concatenate((target[:, 0], target[:, 1]))
        train = np.concatenate((train_frames, train_frames))
        features = np.column_stack((np.ones_like(theta), theta, velocity, previous))
        coefficients = lsq_linear(
            features[train],
            values[train],
            bounds=([-np.inf, -np.inf, -np.inf, 0.0], [np.inf, np.inf, np.inf, 0.999]),
        ).x
        side_coefficients = [coefficients, coefficients]
        packed = coefficients.tolist()
    else:
        side_coefficients = []
        packed = []
        for side, columns in enumerate(((0, 2), (1, 3))):
            features = np.column_stack(
                (
                    np.ones(len(history)),
                    history[:, -1, columns[0]],
                    history[:, -1, columns[1]],
                    previous_target[:, side],
                )
            )
            coefficients = lsq_linear(
                features[train_frames],
                target[train_frames, side],
                bounds=([-np.inf, -np.inf, -np.inf, 0.0], [np.inf, np.inf, np.inf, 0.999]),
            ).x
            side_coefficients.append(coefficients)
            packed.extend(coefficients.tolist())

    output = np.zeros_like(target)
    for frame in range(len(history)):
        for side, columns in enumerate(((0, 2), (1, 3))):
            b, k_theta, k_velocity, k_previous = side_coefficients[side]
            raw = (
                b
                + k_theta * history[frame, -1, columns[0]]
                + k_velocity * history[frame, -1, columns[1]]
                + k_previous * (output[frame - 1, side] if frame else 0.0)
            )
            previous = output[frame - 1, side] if frame else 0.0
            output[frame, side] = np.clip(
                raw, previous - max_delta_nm, previous + max_delta_nm
            )
            output[frame, side] = np.clip(output[frame, side], -10.0, 10.0)
    return output, packed


def fourier_features(phase: np.ndarray, order: int) -> np.ndarray:
    columns = [np.ones_like(phase)]
    for harmonic in range(1, order + 1):
        columns.extend((np.sin(harmonic * phase), np.cos(harmonic * phase)))
    return np.column_stack(columns)


def fit_fourier(
    phase: np.ndarray,
    target: np.ndarray,
    train_frames: np.ndarray,
    *,
    order: int,
    shared: bool,
    max_delta_nm: float,
) -> tuple[np.ndarray, list[float]]:
    if shared:
        canonical_phase = np.concatenate((phase, phase + np.pi))
        values = np.concatenate((target[:, 0], target[:, 1]))
        train = np.concatenate((train_frames, train_frames))
        features = fourier_features(canonical_phase, order)
        coefficients = np.linalg.lstsq(features[train], values[train], rcond=None)[0]
        raw = (features @ coefficients).reshape(2, len(phase)).T
        return rate_limit(raw, max_delta_nm), coefficients.tolist()

    predictions = []
    packed = []
    features = fourier_features(phase, order)
    for side in range(2):
        coefficients = np.linalg.lstsq(
            features[train_frames], target[train_frames, side], rcond=None
        )[0]
        predictions.append(features @ coefficients)
        packed.extend(coefficients.tolist())
    return rate_limit(np.stack(predictions, axis=1), max_delta_nm), packed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--period-frames", type=float, default=106.0)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--max-delta-nm", type=float, default=0.5)
    parser.add_argument("--control-hz", type=float, default=100.0)
    args = parser.parse_args()

    with np.load(args.dataset, allow_pickle=True) as source:
        history_steps = int(np.asarray(source["history_steps"]).reshape(-1)[0])
        history_flat = np.asarray(source["proprio_history"], dtype=np.float64)
        history = history_flat.reshape(len(history_flat), history_steps, -1)
        phase_index = np.asarray(source["phase"], dtype=np.float64)
    _, target = student_torque(
        args.student, history, float(args.max_delta_nm) / 10.0
    )
    frame_count = len(target)
    split = int(np.clip(round(frame_count * args.train_fraction), 1, frame_count - 1))
    train_frames = np.arange(frame_count) < split
    test_frames = ~train_frames
    phase = 2.0 * np.pi * (phase_index - phase_index[0]) / float(args.period_frames)

    predictions: dict[str, np.ndarray] = {"teacher": target}
    parameters: dict[str, list[float]] = {}
    for shared in (True, False):
        suffix = "shared" if shared else "independent"
        for physical in (True, False):
            prefix = "fixed_pd" if physical else "linear_state"
            raw, coefficients = fit_state_feedback(
                history,
                target,
                train_frames,
                shared=shared,
                physical_pd=physical,
            )
            name = f"{prefix}_{suffix}"
            predictions[name] = rate_limit(raw, args.max_delta_nm)
            parameters[name] = coefficients
        prediction, coefficients = fit_autoregressive(
            history,
            target,
            train_frames,
            shared=shared,
            max_delta_nm=args.max_delta_nm,
        )
        name = f"ar_state_{suffix}"
        predictions[name] = prediction
        parameters[name] = coefficients
        for order in range(1, 5):
            prediction, coefficients = fit_fourier(
                phase,
                target,
                train_frames,
                order=order,
                shared=shared,
                max_delta_nm=args.max_delta_nm,
            )
            name = f"fourier{order}_{suffix}"
            predictions[name] = prediction
            parameters[name] = coefficients

    rows = []
    for name, prediction in predictions.items():
        if name == "teacher":
            continue
        rows.append(
            {
                "controller": name,
                "parameter_count": len(parameters[name]),
                "all": metrics(prediction, target, np.ones(frame_count, dtype=bool)),
                "heldout_tail": metrics(prediction, target, test_frames),
                "parameters": parameters[name],
            }
        )
    rows.sort(key=lambda row: row["heldout_tail"]["rmse_nm"])

    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "metrics.json").write_text(
        json.dumps(
            {
                "dataset": str(args.dataset),
                "student": str(args.student),
                "frames": frame_count,
                "train_frames": split,
                "heldout_frames": frame_count - split,
                "period_frames": args.period_frames,
                "teacher_abs_mean_nm": float(np.mean(np.abs(target))),
                "controllers": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with (args.outdir / "timeseries.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["frame", "time_s", "teacher_r_nm", "teacher_l_nm"]
        for row in rows:
            fieldnames.extend(
                (f"{row['controller']}_r_nm", f"{row['controller']}_l_nm")
            )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for frame in range(frame_count):
            output = {
                "frame": frame,
                "time_s": frame / args.control_hz,
                "teacher_r_nm": target[frame, 0],
                "teacher_l_nm": target[frame, 1],
            }
            for row in rows:
                value = predictions[row["controller"]][frame]
                output[f"{row['controller']}_r_nm"] = value[0]
                output[f"{row['controller']}_l_nm"] = value[1]
            writer.writerow(output)

    selected = [
        "fixed_pd_shared",
        "linear_state_shared",
        "ar_state_shared",
        "fourier2_shared",
        "fourier4_shared",
        "fourier2_independent",
    ]
    time = np.arange(frame_count) / args.control_hz
    figure, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    for side, axis in enumerate(axes):
        axis.plot(time, target[:, side], color="black", linewidth=2.0, label="teacher")
        for name in selected:
            axis.plot(time, predictions[name][:, side], linewidth=1.0, label=name)
        axis.axvline(split / args.control_hz, color="gray", linestyle="--", linewidth=1.0)
        axis.set_ylabel(("Right" if side == 0 else "Left") + " torque (Nm)")
        axis.grid(alpha=0.25)
    axes[0].legend(ncol=4, fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    figure.tight_layout()
    figure.savefig(args.outdir / "torque_comparison.png", dpi=180)
    plt.close(figure)

    print(json.dumps({"output": str(args.outdir), "best": rows[:6]}, indent=2))


if __name__ == "__main__":
    main()
