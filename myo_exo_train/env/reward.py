"""Locomotion, gait, terrain, and human-energy rewards."""
from __future__ import annotations

import torch

from myo_exo_train.env.observation import (
    reference_foot_tensor,
    reference_phase_from_x,
    reference_q_dq_tensor,
)
from myo_exo_train.env.reward_locomotion import DenseLocomotionRewardMixin

class RewardMixin(DenseLocomotionRewardMixin):
    def target_phase_idx(self) -> torch.Tensor:
        return reference_phase_from_x(
            self.qpos,
            self.phase_idx,
            self.reference,
            self.config,
            phase_lead_steps=int(self.reference_phase_lead_steps),
        )

    def reference_q_dq(self, phases: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return reference_q_dq_tensor(
            self.reference,
            phases,
            swing_exaggeration_scale=float(self.reference_swing_exaggeration_scale),
        )

    def reference_foot(self, phases: torch.Tensor) -> torch.Tensor:
        return reference_foot_tensor(
            self.reference,
            phases,
            swing_exaggeration_scale=float(self.reference_swing_exaggeration_scale),
        )

    def reference_match_error(
        self,
        q: torch.Tensor,
        dq: torch.Tensor,
        foot_rel_x: torch.Tensor,
        foot_z: torch.Tensor,
        ref_q: torch.Tensor,
        ref_dq: torch.Tensor,
        ref_foot: torch.Tensor,
    ) -> torch.Tensor:
        pose_sq = torch.square((q - ref_q) / self.reference["pose_scales"])
        vel_sq = torch.square((dq - ref_dq) / self.reference["vel_scales"])
        pose_err = torch.sum(pose_sq * self.pose_weights, dim=1) / torch.clamp(
            torch.sum(self.pose_weights), min=1e-6
        )
        vel_err = torch.sum(vel_sq * self.vel_weights, dim=1) / torch.clamp(
            torch.sum(self.vel_weights), min=1e-6
        )
        foot_z_err = torch.mean(torch.square((foot_z - ref_foot[:, :, 2]) / self.foot_z_scale), dim=1)
        foot_x_err = torch.mean(torch.square((foot_rel_x - ref_foot[:, :, 0]) / self.foot_x_scale), dim=1)
        return pose_err + 0.25 * vel_err + 0.25 * foot_z_err + 0.1 * foot_x_err

    def current_hip_effort_metrics(
        self, current_activation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return moment-arm-weighted hip activation squared and hip cocontraction."""
        activation_l2: list[torch.Tensor] = []
        cocontraction: list[torch.Tensor] = []
        for side in range(2):
            actuator_indices = self.hip_muscle_actuator_indices[side]
            row_offsets = self.hip_muscle_moment_offsets[side]
            moment_indices = self.actuator_moment_rowadr.index_select(1, actuator_indices)
            moment_indices = moment_indices + row_offsets.unsqueeze(0)
            muscle_moments = self.actuator_moment.gather(1, moment_indices)
            moment_weights = torch.abs(muscle_moments)
            side_activation = current_activation.index_select(1, actuator_indices)
            activation_l2.append(
                torch.sum(moment_weights * torch.square(side_activation), dim=1)
                / torch.clamp(torch.sum(moment_weights, dim=1), min=1.0e-6)
            )
            muscle_forces = self.actuator_force.index_select(1, actuator_indices)
            torque_contributions = muscle_forces * muscle_moments
            cocontraction.append(
                torch.sum(torch.abs(torque_contributions), dim=1)
                - torch.abs(torch.sum(torque_contributions, dim=1))
            )
        return (
            torch.stack(activation_l2, dim=1).mean(dim=1),
            torch.stack(cocontraction, dim=1).mean(dim=1),
        )

    def human_energy_reward(
        self,
        current_activation: torch.Tensor,
        tracking_error: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        zero = torch.zeros((self.nworld,), dtype=self.qpos.dtype, device=self.device)
        if not self.human_energy_enabled:
            return zero, {}

        gate = torch.sigmoid(
            (float(self.human_energy_tracking_threshold) - tracking_error)
            / float(self.human_energy_tracking_softness)
        )
        activation_l2 = torch.mean(torch.square(current_activation), dim=1)
        activation_penalty = -self.dt * gate * activation_l2
        reward_delta = self.human_energy_activation_weight * activation_penalty
        terms = {
            "human_energy_tracking_gate": gate,
            "human_energy_activation_l2": activation_l2,
            "human_energy_activation_l2_penalty": activation_penalty,
        }

        if self.hip_torque_measurement_enabled:
            hip_activation_l2, hip_cocontraction = self.current_hip_effort_metrics(current_activation)
            human_torque, exo_torque, _ = self.current_hip_generalized_torques()
            torque_scale = float(self.human_energy_hip_torque_scale)
            hip_activation_penalty = -self.dt * gate * hip_activation_l2
            hip_torque_penalty = (
                -self.dt * gate * torch.mean(torch.abs(human_torque), dim=1) / torque_scale
            )
            hip_cocontraction_penalty = (
                -self.dt
                * gate
                * hip_cocontraction
                / float(self.human_energy_hip_cocontraction_scale)
            )
            opposition = torch.mean(torch.relu(-(human_torque * exo_torque)), dim=1) / (
                torque_scale * torque_scale
            )
            hip_opposition_penalty = -self.dt * gate * opposition
            reward_delta = reward_delta + (
                self.human_energy_hip_activation_weight * hip_activation_penalty
                + self.human_energy_hip_torque_weight * hip_torque_penalty
                + self.human_energy_hip_cocontraction_weight * hip_cocontraction_penalty
                + self.human_energy_hip_opposition_weight * hip_opposition_penalty
            )
            terms.update(
                {
                    "human_energy_hip_activation_l2": hip_activation_l2,
                    "human_energy_hip_activation_l2_penalty": hip_activation_penalty,
                    "human_energy_hip_torque_abs": torch.mean(torch.abs(human_torque), dim=1),
                    "human_energy_hip_torque_l1_penalty": hip_torque_penalty,
                    "human_energy_hip_cocontraction_nm": hip_cocontraction,
                    "human_energy_hip_cocontraction_penalty": hip_cocontraction_penalty,
                    "human_energy_hip_opposition": opposition,
                    "human_energy_hip_opposition_penalty": hip_opposition_penalty,
                }
            )
        return reward_delta, terms

    def reward(
        self,
        action: torch.Tensor,
        activation: torch.Tensor,
        prev_foot: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del prev_foot
        if self.reward_mode != "myoassist_exact":
            raise ValueError(f"unsupported reward_mode: {self.reward_mode}")
        return self.myoassist_exact_reward(action, activation)

def reward_weights_for_step(config: dict[str, Any], global_step: int, run_start_global_step: int) -> dict[str, float]:
    weights = {str(k): float(v) for k, v in config.get("reward", {}).items()}
    schedule = config.get("reward_schedule", [])
    if not isinstance(schedule, list) or not schedule:
        return weights
    schedule_step = int(global_step)
    if str(config.get("reward_schedule_mode", "relative")) == "relative":
        schedule_step = max(0, int(global_step) - int(run_start_global_step))
    for item in sorted(schedule, key=lambda x: int(x.get("after_steps", 0))):
        if schedule_step >= int(item.get("after_steps", 0)):
            weights.update({str(k): float(v) for k, v in item.get("weights", {}).items()})
    return weights

def apply_reward_schedule(config: dict[str, Any], runner: MJWarpMuscleRunner, global_step: int, run_start_global_step: int) -> dict[str, float]:
    weights = reward_weights_for_step(config, global_step, run_start_global_step)
    config["reward"] = dict(weights)
    runner.reward_weights = dict(weights)
    return weights
