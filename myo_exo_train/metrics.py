"""Compact CSV/TensorBoard metric presentation."""
from __future__ import annotations

from pathlib import Path
import os
from typing import Any

from torch.utils.tensorboard import SummaryWriter


TRAIN_SCALARS = {
    "episode_reward_per_step_done_mean": "train_diagnostic/completed_episode_reward_per_step",
    "episode_duration_done_mean_s": "train/episode_duration_mean_s",
    "episode_forward_displacement_done_mean_m": "train/forward_displacement_mean_m",
    "window_fall_rate": "train/fall_rate",
    "samples_per_sec_step": "performance/samples_per_sec",
    "q_loss": "optimization/q_loss",
    "actor_loss": "optimization/actor_loss",
    "alpha": "optimization/alpha",
    "activation_mean": "effort/activation_mean",
    "step_mean_human_energy_activation_l2": "effort/activation_l2",
    "step_mean_human_energy_joint_cocontraction_nm": "effort/cocontraction_nm",
    "step_mean_human_energy_hip_opposition": "effort/exo_muscle_opposition_nm",
    "step_mean_lateral_drift_abs": "gait/lateral_drift_abs_m",
    "step_mean_reference_tracking_error": "gait/reference_tracking_error",
    "step_mean_foot_toe_in_angle_r": "gait/toe_in_right",
    "step_mean_foot_toe_in_angle_l": "gait/toe_in_left",
    "step_mean_knee_valgus_r": "gait/knee_valgus_right",
    "step_mean_knee_valgus_l": "gait/knee_valgus_left",
}


class MetricsWriter:
    def __init__(self, outdir: Path):
        configured_root = os.environ.get("MYO_EXO_TENSORBOARD_ROOT")
        if configured_root:
            common_root = Path(configured_root).expanduser().resolve()
        elif outdir.parent.name == "results":
            common_root = outdir.parent.parent / "tensorboard_runs"
        else:
            common_root = outdir / "tensorboard"
        log_dir = (
            common_root / outdir.name
            if common_root != outdir / "tensorboard"
            else common_root
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        legacy_path = outdir / "tensorboard"
        if legacy_path != log_dir and not legacy_path.exists():
            legacy_path.parent.mkdir(parents=True, exist_ok=True)
            legacy_path.symlink_to(log_dir, target_is_directory=True)
        self.writer = SummaryWriter(log_dir=str(log_dir))

    def add_train(self, row: dict[str, Any]) -> None:
        step = int(row["global_step"])
        for key, tag in TRAIN_SCALARS.items():
            value = row.get(key)
            if isinstance(value, (int, float)):
                self.writer.add_scalar(tag, value, step)
        self.writer.flush()

    def add_eval(self, row: dict[str, Any]) -> None:
        step = int(row["global_step"])
        reward_keys = {
            "eval_first_episode_return_mean": "eval_reward/return_mean",
            "eval_first_episode_return_std": "eval_reward/return_std",
            "eval_first_episode_reward_per_step_mean": "eval_reward/reward_per_step_mean",
        }
        performance_keys = {
            "eval_first_episode_completed_rate": "eval_performance/completed_rate",
            "eval_first_episode_fall_rate": "eval_performance/fall_rate",
            "eval_first_episode_qvel_done_rate": "eval_performance/qvel_done_rate",
            "eval_first_episode_mean_pelvis_vx": "eval_performance/mean_pelvis_vx",
            "eval_first_episode_duration_mean_s": "eval_performance/duration_mean_s",
            "eval_first_episode_duration_min_s": "eval_performance/duration_min_s",
            "eval_first_episode_duration_max_s": "eval_performance/duration_max_s",
            "eval_first_episode_forward_displacement_mean_m": (
                "eval_performance/forward_displacement_mean_m"
            ),
            "eval_first_episode_forward_displacement_min_m": (
                "eval_performance/forward_displacement_min_m"
            ),
            "eval_first_episode_forward_displacement_max_m": (
                "eval_performance/forward_displacement_max_m"
            ),
            "eval_gait_landing_count_mean": "eval_gait/landing_count",
            "eval_gait_cycle_interval_mean_steps": (
                "eval_gait/cycle_interval_steps"
            ),
            "eval_gait_half_cycle_interval_mean_steps": (
                "eval_gait/half_cycle_interval_steps"
            ),
            "eval_gait_cycle_interval_mae_steps": (
                "eval_gait/cycle_interval_mae_steps"
            ),
            "eval_gait_half_cycle_interval_mae_steps": (
                "eval_gait/half_cycle_interval_mae_steps"
            ),
            "eval_gait_alternation_rate": "eval_gait/alternation_rate",
            "eval_gait_repeated_side_rate": "eval_gait/repeated_side_rate",
            "eval_gait_dense_half_cycle_pose_rmse": (
                "eval_gait/dense_half_cycle_pose_rmse"
            ),
            "eval_gait_dense_half_cycle_velocity_rmse": (
                "eval_gait/dense_half_cycle_velocity_rmse"
            ),
            "eval_gait_dense_half_cycle_activation_rmse": (
                "eval_gait/dense_half_cycle_activation_rmse"
            ),
            "eval_gait_stance_impulse_relative_error": (
                "eval_gait/stance_impulse_relative_error"
            ),
            "eval_gait_stance_duration_mae_steps": (
                "eval_gait/stance_duration_mae_steps"
            ),
            "eval_gait_stance_peak_force_relative_error": (
                "eval_gait/stance_peak_force_relative_error"
            ),
        }
        for keys in (reward_keys, performance_keys):
            for key, tag in keys.items():
                value = row.get(key)
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(tag, value, step)
        term_returns = row.get("eval_reward_term_returns", {})
        if isinstance(term_returns, dict):
            for name, values in term_returns.items():
                if not isinstance(values, dict):
                    continue
                contribution = values.get("contribution_return_mean")
                if isinstance(contribution, (int, float)):
                    self.writer.add_scalar(
                        f"eval_reward_terms/{name}", contribution, step
                    )
        velocity_fraction = row.get("eval_forward_velocity_target_fraction")
        if isinstance(velocity_fraction, (int, float)):
            self.writer.add_scalar(
                "eval_performance/forward_velocity_target_fraction",
                velocity_fraction,
                step,
            )
        distributions = row.get("eval_reward_distributions", {})
        if isinstance(distributions, dict):
            for name, values in distributions.items():
                if name == "total_reward":
                    continue
                if not isinstance(values, dict):
                    continue
                contribution = values.get("contribution_mean")
                if isinstance(contribution, (int, float)):
                    self.writer.add_scalar(
                        f"eval_reward_per_step/{name}",
                        contribution,
                        step,
                    )
        self.writer.flush()

    def close(self) -> None:
        self.writer.close()
