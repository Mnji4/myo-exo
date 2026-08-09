"""Locomotion, gait, terrain, and human-energy rewards."""
from __future__ import annotations

import torch

from myo_exo_train.env.observation import (
    reference_foot_tensor,
    reference_phase_from_x,
    reference_q_dq_tensor,
)
from myo_exo_train.env.reward_locomotion import DenseLocomotionRewardMixin


def joint_cocontraction_objective(
    joint_cocontraction: torch.Tensor,
    max_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combine mean and worst-joint cocontraction without changing the Nm scale."""
    mean_nm = torch.mean(joint_cocontraction, dim=1)
    max_nm = torch.max(joint_cocontraction, dim=1).values
    weight = max(0.0, float(max_weight))
    objective_nm = (mean_nm + weight * max_nm) / (1.0 + weight)
    return mean_nm, max_nm, objective_nm


def muscle_passive_force(
    length: torch.Tensor,
    length_range: torch.Tensor,
    acc0: torch.Tensor,
    bias_parameters: torch.Tensor,
) -> torch.Tensor:
    """Vectorized MuJoCo muscle bias force, excluding activation-dependent force."""
    eps = torch.finfo(length.dtype).eps
    range_min = bias_parameters[:, 0]
    range_max = bias_parameters[:, 1]
    force_parameter = bias_parameters[:, 2]
    scale = bias_parameters[:, 3]
    lmax = bias_parameters[:, 5]
    fpmax = bias_parameters[:, 7]
    force = torch.where(
        force_parameter < 0.0,
        scale / torch.clamp(acc0, min=eps),
        force_parameter,
    )
    optimal_length = (length_range[:, 1] - length_range[:, 0]) / torch.clamp(
        range_max - range_min,
        min=eps,
    )
    normalized_length = range_min + (length - length_range[:, 0]) / torch.clamp(
        optimal_length,
        min=eps,
    )
    midpoint = 0.5 * (1.0 + lmax)
    denominator = torch.clamp(midpoint - 1.0, min=eps)
    quadratic_x = (normalized_length - 1.0) / denominator
    linear_x = (normalized_length - midpoint) / denominator
    quadratic = -force * fpmax * 0.5 * torch.square(quadratic_x)
    linear = -force * fpmax * (0.5 + linear_x)
    return torch.where(
        normalized_length <= 1.0,
        torch.zeros_like(length),
        torch.where(normalized_length <= midpoint, quadratic, linear),
    )


def hip_antagonist_target_metrics(
    flexion_torque: torch.Tensor,
    extension_torque: torch.Tensor,
    target_net_torque: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Measure torque above the minimum antagonist-free target and net torque error."""
    target_flexion = torch.relu(target_net_torque)
    target_extension = torch.relu(-target_net_torque)
    antagonist_excess = torch.mean(
        torch.relu(flexion_torque - target_flexion)
        + torch.relu(extension_torque - target_extension),
        dim=1,
    )
    net_torque_error = torch.mean(
        torch.abs(flexion_torque - extension_torque - target_net_torque), dim=1
    )
    return antagonist_excess, net_torque_error


def human_energy_speed_gate(
    forward_speed: torch.Tensor,
    min_forward_speed: float,
    softness: float,
) -> torch.Tensor:
    return torch.sigmoid(
        (forward_speed - float(min_forward_speed)) / max(float(softness), 1.0e-6)
    )


class RewardMixin(DenseLocomotionRewardMixin):
    def target_phase_idx(self) -> torch.Tensor:
        return reference_phase_from_x(
            self.qpos,
            self.phase_idx,
            self.reference,
            self.config,
            phase_lead_steps=int(self.reference_phase_lead_steps),
            x_align_mask=self.x_align_mask,
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
        hip_activation_l2, flexion_torque, extension_torque = self.current_hip_torque_components(
            current_activation
        )
        return hip_activation_l2, torch.minimum(flexion_torque, extension_torque).mean(dim=1)

    def current_hip_torque_components(
        self, current_activation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return hip activation cost and per-side positive flexion/extension torques."""
        activation_l2: list[torch.Tensor] = []
        flexion_torques: list[torch.Tensor] = []
        extension_torques: list[torch.Tensor] = []
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
            flexion_torque = torch.sum(torch.relu(torque_contributions), dim=1)
            extension_torque = torch.sum(torch.relu(-torque_contributions), dim=1)
            flexion_torques.append(flexion_torque)
            extension_torques.append(extension_torque)
        return (
            torch.stack(activation_l2, dim=1).mean(dim=1),
            torch.stack(flexion_torques, dim=1),
            torch.stack(extension_torques, dim=1),
        )

    def current_muscle_force_components(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return activation-dependent, passive, and total muscle actuator forces."""
        total_force = self.actuator_force[:, : int(self.model.na)]
        passive_force = muscle_passive_force(
            self.actuator_length[:, : int(self.model.na)],
            self.muscle_actuator_length_range,
            self.muscle_actuator_acc0,
            self.muscle_actuator_bias_parameters,
        )
        return total_force - passive_force, passive_force, total_force

    def current_joint_cocontraction(self, muscle_forces: torch.Tensor) -> torch.Tensor:
        """Return opposing muscle torque for configured sagittal joints."""
        joint_cocontraction: list[torch.Tensor] = []
        for actuator_indices, row_offsets in zip(
            self.energy_joint_muscle_actuator_indices,
            self.energy_joint_muscle_moment_offsets,
            strict=True,
        ):
            moment_indices = self.actuator_moment_rowadr.index_select(1, actuator_indices)
            moment_indices = moment_indices + row_offsets.unsqueeze(0)
            muscle_moments = self.actuator_moment.gather(1, moment_indices)
            selected_forces = muscle_forces.index_select(1, actuator_indices)
            contributions = selected_forces * muscle_moments
            positive = torch.sum(torch.relu(contributions), dim=1)
            negative = torch.sum(torch.relu(-contributions), dim=1)
            joint_cocontraction.append(torch.minimum(positive, negative))
        return torch.stack(joint_cocontraction, dim=1)

    def human_energy_reward(
        self,
        current_activation: torch.Tensor,
        tracking_error: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        zero = torch.zeros((self.nworld,), dtype=self.qpos.dtype, device=self.device)
        if not self.human_energy_enabled:
            return zero, {}

        tracking_gate = torch.sigmoid(
            (float(self.human_energy_tracking_threshold) - tracking_error)
            / float(self.human_energy_tracking_softness)
        )
        forward_speed = self.qvel[:, self.pelvis_tx_qvel]
        speed_gate = torch.ones_like(tracking_error)
        if self.human_energy_speed_gate_enabled:
            speed_gate = human_energy_speed_gate(
                forward_speed,
                self.human_energy_min_forward_speed,
                self.human_energy_speed_gate_softness,
            )
        lateral_drift = torch.zeros_like(tracking_error)
        lateral_gate = torch.ones_like(tracking_error)
        if self.root_qpos_adr >= 0 and self.human_energy_lateral_threshold > 0.0:
            target_phase = self.target_phase_idx()
            full_ref_q = self.reference["full_reset_qpos"][target_phase].to(
                device=self.device, dtype=self.qpos.dtype
            )
            lateral_axis = 1 if self.forward_axis == "x" else 0
            lateral_qpos = self.root_qpos_adr + lateral_axis
            lateral_drift = torch.abs(self.qpos[:, lateral_qpos] - full_ref_q[:, lateral_qpos])
            lateral_gate = torch.sigmoid(
                (float(self.human_energy_lateral_threshold) - lateral_drift)
                / float(self.human_energy_lateral_softness)
            )
        gate = tracking_gate * lateral_gate * speed_gate
        quality_gate_penalty = -self.dt * (1.0 - gate)
        activation_l2 = torch.mean(torch.square(current_activation), dim=1)
        activation_penalty = -self.dt * gate * activation_l2
        reward_delta = (
            self.human_energy_activation_weight * activation_penalty
            + self.human_energy_quality_gate_penalty_weight * quality_gate_penalty
        )
        terms = {
            "human_energy_tracking_gate": gate,
            "human_energy_pose_gate": tracking_gate,
            "human_energy_speed_gate": speed_gate,
            "human_energy_forward_speed": forward_speed,
            "human_energy_lateral_gate": lateral_gate,
            "human_energy_lateral_drift": lateral_drift,
            "human_energy_quality_gate_penalty": quality_gate_penalty,
            "human_energy_activation_l2": activation_l2,
            "human_energy_activation_l2_penalty": activation_penalty,
        }

        if self.human_energy_joint_cocontraction_measure:
            mode = self.human_energy_joint_cocontraction_force_mode
            detailed = self.human_energy_joint_cocontraction_detailed_measure
            if mode == "total" and not detailed:
                selected_force = self.actuator_force[:, : int(self.model.na)]
                force_by_mode: dict[str, torch.Tensor] = {"total": selected_force}
            else:
                active_force, passive_force, total_force = self.current_muscle_force_components()
                force_by_mode = {
                    "active": active_force,
                    "passive": passive_force,
                    "total": total_force,
                }
                selected_force = force_by_mode[mode]
            joint_cocontraction = self.current_joint_cocontraction(selected_force)
            cocontraction_by_mode = {mode: joint_cocontraction}
            if detailed:
                for detail_mode, detail_force in force_by_mode.items():
                    if detail_mode != mode:
                        cocontraction_by_mode[detail_mode] = self.current_joint_cocontraction(
                            detail_force
                        )
            joint_cocontraction_mean, joint_cocontraction_max, joint_cocontraction_objective_nm = (
                joint_cocontraction_objective(
                    joint_cocontraction,
                    self.human_energy_joint_cocontraction_max_weight,
                )
            )
            cocontraction_gate = (
                gate
                if self.human_energy_joint_cocontraction_use_tracking_gate
                else torch.ones_like(gate)
            )
            joint_cocontraction_penalty = (
                -self.dt
                * cocontraction_gate
                * joint_cocontraction_objective_nm
                / float(self.human_energy_joint_cocontraction_scale)
            )
            reward_delta = reward_delta + (
                self.human_energy_joint_cocontraction_weight * joint_cocontraction_penalty
            )
            terms["human_energy_joint_cocontraction_nm"] = joint_cocontraction_mean
            terms["human_energy_joint_cocontraction_max_nm"] = joint_cocontraction_max
            terms["human_energy_joint_cocontraction_objective_nm"] = (
                joint_cocontraction_objective_nm
            )
            terms["human_energy_joint_cocontraction_gate"] = cocontraction_gate
            terms["human_energy_joint_cocontraction_penalty"] = joint_cocontraction_penalty
            for index, name in enumerate(self.human_energy_joint_cocontraction_names):
                terms[f"human_energy_joint_cocontraction_{name}_nm"] = joint_cocontraction[:, index]
            for detail_mode, detail_cocontraction in cocontraction_by_mode.items():
                terms[f"human_energy_joint_cocontraction_{detail_mode}_nm"] = torch.mean(
                    detail_cocontraction, dim=1
                )
                for index, name in enumerate(self.human_energy_joint_cocontraction_names):
                    terms[
                        f"human_energy_joint_cocontraction_{name}_{detail_mode}_nm"
                    ] = detail_cocontraction[:, index]

        if self.hip_torque_measurement_enabled:
            hip_activation_l2, flexion_torque, extension_torque = self.current_hip_torque_components(
                current_activation
            )
            human_torque = flexion_torque - extension_torque
            _, exo_torque, _ = self.current_hip_generalized_torques()
            hip_cocontraction = torch.minimum(flexion_torque, extension_torque).mean(dim=1)
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
            target_net_torque = torch.zeros_like(human_torque)
            antagonist_excess = torch.zeros_like(hip_cocontraction)
            net_torque_error = torch.zeros_like(hip_cocontraction)
            antagonist_excess_penalty = torch.zeros_like(hip_cocontraction)
            net_torque_error_penalty = torch.zeros_like(hip_cocontraction)
            if self.human_energy_hip_net_torque_target is not None:
                target_phase = self.target_phase_idx() % int(
                    self.human_energy_hip_net_torque_target.shape[0]
                )
                target_net_torque = self.human_energy_hip_net_torque_target[target_phase]
                antagonist_excess, net_torque_error = hip_antagonist_target_metrics(
                    flexion_torque, extension_torque, target_net_torque
                )
                antagonist_excess_penalty = (
                    -self.dt
                    * gate
                    * antagonist_excess
                    / float(self.human_energy_hip_antagonist_excess_scale)
                )
                normalized_net_error = (
                    net_torque_error / float(self.human_energy_hip_net_torque_error_scale)
                )
                net_torque_error_penalty = -self.dt * gate * torch.clamp(
                    torch.square(normalized_net_error), max=4.0
                )
                reward_delta = reward_delta + (
                    self.human_energy_hip_antagonist_excess_weight
                    * antagonist_excess_penalty
                    + self.human_energy_hip_net_torque_error_weight
                    * net_torque_error_penalty
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
                    "human_energy_hip_flexion_nm": flexion_torque.mean(dim=1),
                    "human_energy_hip_extension_nm": extension_torque.mean(dim=1),
                    "human_energy_hip_target_net_torque_abs_nm": torch.mean(
                        torch.abs(target_net_torque), dim=1
                    ),
                    "human_energy_hip_antagonist_excess_nm": antagonist_excess,
                    "human_energy_hip_antagonist_excess_penalty": antagonist_excess_penalty,
                    "human_energy_hip_net_torque_error_nm": net_torque_error,
                    "human_energy_hip_net_torque_error_penalty": net_torque_error_penalty,
                }
            )
        return reward_delta, terms

    def reward(
        self,
        action: torch.Tensor,
        activation: torch.Tensor,
        prev_foot: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.reward_mode != "myoassist_exact":
            raise ValueError(f"unsupported reward_mode: {self.reward_mode}")
        return self.myoassist_exact_reward(action, activation, prev_foot)

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


_HUMAN_ENERGY_WEIGHT_ATTRIBUTES = {
    "activation_l2_weight": "human_energy_activation_weight",
    "hip_activation_l2_weight": "human_energy_hip_activation_weight",
    "hip_torque_l1_weight": "human_energy_hip_torque_weight",
    "hip_cocontraction_weight": "human_energy_hip_cocontraction_weight",
    "hip_opposition_weight": "human_energy_hip_opposition_weight",
}


def human_energy_weights_for_step(
    config: dict[str, Any],
    global_step: int,
    run_start_global_step: int,
) -> dict[str, float]:
    human_energy = config.get("human_energy_objective", {})
    weights = {
        key: float(human_energy.get(key, 0.0))
        for key in _HUMAN_ENERGY_WEIGHT_ATTRIBUTES
    }
    schedule = human_energy.get("weight_schedule", [])
    if not isinstance(schedule, list) or not schedule:
        return weights
    schedule_step = int(global_step)
    if str(human_energy.get("weight_schedule_mode", "relative")) == "relative":
        schedule_step = max(0, int(global_step) - int(run_start_global_step))
    for item in sorted(schedule, key=lambda value: int(value.get("after_steps", 0))):
        if schedule_step < int(item.get("after_steps", 0)):
            continue
        for key in _HUMAN_ENERGY_WEIGHT_ATTRIBUTES:
            if key in item:
                weights[key] = float(item[key])
    return weights

def apply_reward_schedule(config: dict[str, Any], runner: MJWarpMuscleRunner, global_step: int, run_start_global_step: int) -> dict[str, float]:
    weights = reward_weights_for_step(config, global_step, run_start_global_step)
    config["reward"] = dict(weights)
    runner.reward_weights = dict(weights)
    human_energy = config.get("human_energy_objective", {})
    cocontraction_weight = float(human_energy.get("joint_cocontraction_weight", 0.0))
    cocontraction_schedule = human_energy.get("joint_cocontraction_weight_schedule", [])
    if isinstance(cocontraction_schedule, list) and cocontraction_schedule:
        schedule_step = int(global_step)
        if str(human_energy.get("joint_cocontraction_weight_schedule_mode", "relative")) == "relative":
            schedule_step = max(0, int(global_step) - int(run_start_global_step))
        for item in sorted(cocontraction_schedule, key=lambda x: int(x.get("after_steps", 0))):
            if schedule_step >= int(item.get("after_steps", 0)):
                cocontraction_weight = float(item.get("weight", cocontraction_weight))
    runner.human_energy_joint_cocontraction_weight = max(0.0, cocontraction_weight)
    human_energy_weights = human_energy_weights_for_step(
        config, global_step, run_start_global_step
    )
    for key, attribute in _HUMAN_ENERGY_WEIGHT_ATTRIBUTES.items():
        setattr(runner, attribute, max(0.0, human_energy_weights[key]))
    return weights
