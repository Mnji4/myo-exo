"""Batched torque-to-muscle allocation for MJWarp training."""
from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
import torch


DEFAULT_JOINTS = (
    "hip_flexion_r",
    "knee_angle_r",
    "ankle_angle_r",
    "hip_flexion_l",
    "knee_angle_l",
    "ankle_angle_l",
)


def _moment_offsets(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_names: tuple[str, ...],
    actuator_indices: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.zeros((len(joint_names), len(actuator_indices)), dtype=np.int64)
    valid = np.zeros_like(offsets, dtype=np.bool_)
    for joint_index, joint_name in enumerate(joint_names):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise KeyError(f"missing torque-action joint {joint_name}")
        dof = int(model.jnt_dofadr[joint_id])
        for column, actuator in enumerate(actuator_indices):
            rowadr = int(data.moment_rowadr[actuator])
            rownnz = int(data.moment_rownnz[actuator])
            moment_columns = np.asarray(data.moment_colind[rowadr : rowadr + rownnz])
            matches = np.flatnonzero(moment_columns == dof)
            if matches.size:
                offsets[joint_index, column] = int(matches[0])
                valid[joint_index, column] = True
    return offsets, valid


class BatchedTorqueAllocator:
    """Approximate the bounded minimum-activation allocation entirely on GPU."""

    def __init__(
        self,
        *,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        config: dict[str, Any],
        device: torch.device,
    ):
        cfg = config.get("torque_action", {})
        names = tuple(str(name) for name in cfg.get("joint_names", DEFAULT_JOINTS))
        if len(names) != 6:
            raise ValueError("the initial torque allocator requires exactly six sagittal joints")
        self.joint_names = names
        scale = cfg.get("torque_scale_nm", [120.0, 100.0, 80.0, 120.0, 100.0, 80.0])
        if len(scale) != len(names):
            raise ValueError("torque_action.torque_scale_nm must match joint_names")
        self.torque_scale = torch.tensor(scale, dtype=torch.float32, device=device)
        self.iterations = max(1, int(cfg.get("allocator_iterations", 24)))
        self.activation_weight = max(0.0, float(cfg.get("activation_l2_weight", 1.0e-3)))
        self.exo_weight = max(0.0, float(cfg.get("exo_l2_weight", 1.0e-5)))
        self.exo_delta_weight = max(0.0, float(cfg.get("exo_delta_weight", 1.0e-4)))
        self.activation_substeps = max(1, int(cfg.get("activation_prediction_substeps", 4)))
        self.activation_inverse_iterations = max(1, int(cfg.get("activation_inverse_iterations", 8)))
        self.compile_solver = bool(cfg.get("compile_solver", True))
        self.control_dt = 1.0 / float(config["control"].get("control_hz", 30.0))

        muscle_count = int(model.na)
        self.muscle_count = muscle_count
        self.exo_count = min(2, int(model.nu) - muscle_count)
        muscle_actuators = list(range(muscle_count))
        exo_actuators = list(range(muscle_count, muscle_count + self.exo_count))
        muscle_offsets, muscle_valid = _moment_offsets(model, data, names, muscle_actuators)
        exo_offsets, exo_valid = _moment_offsets(model, data, names, exo_actuators)
        self.muscle_offsets = torch.tensor(muscle_offsets, dtype=torch.long, device=device)
        self.muscle_valid = torch.tensor(muscle_valid, dtype=torch.float32, device=device)
        self.exo_offsets = torch.tensor(exo_offsets, dtype=torch.long, device=device)
        self.exo_valid = torch.tensor(exo_valid, dtype=torch.float32, device=device)
        self.exo_actuator_indices = torch.tensor(exo_actuators, dtype=torch.long, device=device)

        self.length_range = torch.tensor(
            model.actuator_lengthrange[:muscle_count].copy(), dtype=torch.float32, device=device
        )
        self.acc0 = torch.tensor(
            model.actuator_acc0[:muscle_count].copy(), dtype=torch.float32, device=device
        )
        self.gain_parameters = torch.tensor(
            model.actuator_gainprm[:muscle_count].copy(), dtype=torch.float32, device=device
        )
        self.dynamics_parameters = torch.tensor(
            model.actuator_dynprm[:muscle_count, :3].copy(), dtype=torch.float32, device=device
        )
        self.exo_unit_force = torch.tensor(
            model.actuator_gainprm[muscle_count : muscle_count + self.exo_count, 0].copy(),
            dtype=torch.float32,
            device=device,
        )
        self._solve_compiled = (
            torch.compile(self._solve, fullgraph=True, mode="reduce-overhead")
            if self.compile_solver
            else self._solve
        )

    @property
    def action_dim(self) -> int:
        return len(self.joint_names)

    def _gather_moments(
        self,
        runner: Any,
        actuator_indices: torch.Tensor,
        offsets: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        if actuator_indices.numel() == 0:
            return torch.zeros(
                (runner.nworld, len(self.joint_names), 0),
                dtype=runner.qpos.dtype,
                device=runner.device,
            )
        rowadr = runner.actuator_moment_rowadr.index_select(1, actuator_indices)
        indices = rowadr[:, None, :] + offsets[None, :, :]
        gathered = runner.actuator_moment.gather(1, indices.flatten(1)).reshape(indices.shape)
        return gathered * valid[None, :, :]

    def _muscle_gain(self, length: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        prm = self.gain_parameters
        eps = torch.finfo(length.dtype).eps
        range_min, range_max = prm[:, 0], prm[:, 1]
        force = torch.where(prm[:, 2] < 0.0, prm[:, 3] / self.acc0.clamp_min(eps), prm[:, 2])
        lmin, lmax, vmax, fvmax = prm[:, 4], prm[:, 5], prm[:, 6], prm[:, 8]
        optimal_length = (self.length_range[:, 1] - self.length_range[:, 0]) / (
            range_max - range_min
        ).clamp_min(eps)
        normalized_length = range_min + (length - self.length_range[:, 0]) / optimal_length.clamp_min(eps)
        normalized_velocity = velocity / (optimal_length * vmax).clamp_min(eps)

        a = 0.5 * (lmin + 1.0)
        b = 0.5 * (1.0 + lmax)
        lower_x = (normalized_length - lmin) / (a - lmin).clamp_min(eps)
        left_x = (1.0 - normalized_length) / (1.0 - a).clamp_min(eps)
        right_x = (normalized_length - 1.0) / (b - 1.0).clamp_min(eps)
        upper_x = (lmax - normalized_length) / (lmax - b).clamp_min(eps)
        length_gain = torch.where(
            (normalized_length < lmin) | (normalized_length > lmax),
            torch.zeros_like(length),
            torch.where(
                normalized_length <= a,
                0.5 * torch.square(lower_x),
                torch.where(
                    normalized_length <= 1.0,
                    1.0 - 0.5 * torch.square(left_x),
                    torch.where(
                        normalized_length <= b,
                        1.0 - 0.5 * torch.square(right_x),
                        0.5 * torch.square(upper_x),
                    ),
                ),
            ),
        )
        y = fvmax - 1.0
        velocity_gain = torch.where(
            normalized_velocity <= -1.0,
            torch.zeros_like(velocity),
            torch.where(
                normalized_velocity <= 0.0,
                torch.square(normalized_velocity + 1.0),
                torch.where(
                    normalized_velocity <= y,
                    fvmax - torch.square(y - normalized_velocity) / y.clamp_min(eps),
                    fvmax,
                ),
            ),
        )
        return -force * length_gain * velocity_gain

    def _activation_after(self, activation: torch.Tensor, excitation: torch.Tensor) -> torch.Tensor:
        value = activation
        substep_dt = self.control_dt / float(self.activation_substeps)
        tau_act_base = self.dynamics_parameters[:, 0]
        tau_deact_base = self.dynamics_parameters[:, 1]
        for _ in range(self.activation_substeps):
            clamped = value.clamp(0.0, 1.0)
            delta = excitation.clamp(0.0, 1.0) - value
            tau_act = tau_act_base * (0.5 + 1.5 * clamped)
            tau_deact = tau_deact_base / (0.5 + 1.5 * clamped)
            tau = torch.where(delta > 0.0, tau_act, tau_deact).clamp_min(1.0e-6)
            value = (value + substep_dt * delta / tau).clamp(0.0, 1.0)
        return value

    def _excitation_for_activation(
        self, current: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        lower = torch.zeros_like(current)
        upper = torch.ones_like(current)
        for _ in range(self.activation_inverse_iterations):
            middle = 0.5 * (lower + upper)
            reached = self._activation_after(current, middle)
            lower = torch.where(reached < target, middle, lower)
            upper = torch.where(reached >= target, middle, upper)
        return 0.5 * (lower + upper)

    def _solve(
        self,
        scaled_map: torch.Tensor,
        requested: torch.Tensor,
        current_activation: torch.Tensor,
        previous_exo: torch.Tensor,
        exo_low: torch.Tensor,
        exo_high: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        activation_lower = self._activation_after(current_activation, torch.zeros_like(current_activation))
        activation_upper = self._activation_after(current_activation, torch.ones_like(current_activation))
        lower = torch.cat((activation_lower, exo_low), dim=1)
        upper = torch.cat((activation_upper, exo_high), dim=1)
        value = torch.cat((current_activation, previous_exo), dim=1).clamp(lower, upper)
        extrapolated = value
        momentum = torch.ones((value.shape[0], 1), dtype=value.dtype, device=value.device)
        lipschitz = torch.sum(torch.square(scaled_map), dim=(1, 2), keepdim=False)
        step_size = 0.95 / (lipschitz + max(self.activation_weight, self.exo_weight, 1.0e-6))
        for _ in range(self.iterations):
            residual = torch.bmm(scaled_map, extrapolated.unsqueeze(2)).squeeze(2) - requested
            gradient = torch.bmm(scaled_map.transpose(1, 2), residual.unsqueeze(2)).squeeze(2)
            gradient[:, : self.muscle_count] += self.activation_weight * extrapolated[:, : self.muscle_count]
            if self.exo_count:
                exo = extrapolated[:, self.muscle_count :]
                gradient[:, self.muscle_count :] += self.exo_weight * exo
                gradient[:, self.muscle_count :] += self.exo_delta_weight * (exo - previous_exo)
            next_value = (extrapolated - step_size[:, None] * gradient).clamp(lower, upper)
            next_momentum = 0.5 * (1.0 + torch.sqrt(1.0 + 4.0 * torch.square(momentum)))
            extrapolated = next_value + ((momentum - 1.0) / next_momentum) * (next_value - value)
            value = next_value
            momentum = next_momentum

        target_activation = value[:, : self.muscle_count]
        excitation = self._excitation_for_activation(current_activation, target_activation)
        achieved_normalized = torch.bmm(scaled_map, value.unsqueeze(2)).squeeze(2)
        return value, excitation, achieved_normalized

    def allocate(self, action: torch.Tensor, runner: Any) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        muscle_indices = torch.arange(self.muscle_count, dtype=torch.long, device=runner.device)
        muscle_moments = self._gather_moments(
            runner, muscle_indices, self.muscle_offsets, self.muscle_valid
        )
        muscle_gain = self._muscle_gain(
            runner.actuator_length[:, : self.muscle_count],
            runner.actuator_velocity[:, : self.muscle_count],
        )
        muscle_map = muscle_moments * muscle_gain[:, None, :]
        exo_moments = self._gather_moments(
            runner, self.exo_actuator_indices, self.exo_offsets, self.exo_valid
        )
        exo_map = exo_moments * self.exo_unit_force[None, None, :]
        combined_map = torch.cat((muscle_map, exo_map), dim=2)
        scaled_map = combined_map / self.torque_scale[None, :, None]
        requested = action.clamp(-1.0, 1.0)
        current_activation = runner.act[:, : self.muscle_count].clamp(0.0, 1.0)
        previous_exo = runner.applied_exo_ctrl[:, : self.exo_count]
        exo_low = torch.full_like(previous_exo, -float(runner.exo_policy_max_ctrl))
        if not runner.exo_policy_bidirectional:
            exo_low.zero_()
        exo_high = torch.full_like(previous_exo, float(runner.exo_policy_max_ctrl))
        if runner.exo_policy_max_delta > 0.0:
            delta = float(runner.exo_policy_max_delta)
            exo_low = torch.maximum(exo_low, previous_exo - delta)
            exo_high = torch.minimum(exo_high, previous_exo + delta)
        value, excitation, achieved_normalized = self._solve_compiled(
            scaled_map, requested, current_activation, previous_exo, exo_low, exo_high
        )
        target_activation = value[:, : self.muscle_count]
        ctrl = torch.zeros((runner.nworld, int(runner.model.nu)), dtype=action.dtype, device=runner.device)
        ctrl[:, : self.muscle_count] = excitation
        if self.exo_count:
            ctrl[:, self.muscle_count : self.muscle_count + self.exo_count] = value[:, self.muscle_count :]
        reset_rows = runner.episode_step == 0
        neutral_excitation = float(runner.config.get("reset", {}).get("initial_activation", 0.0))
        ctrl[:, : self.muscle_count] = torch.where(
            reset_rows[:, None],
            torch.full_like(ctrl[:, : self.muscle_count], neutral_excitation),
            ctrl[:, : self.muscle_count],
        )
        if self.exo_count:
            ctrl[:, self.muscle_count : self.muscle_count + self.exo_count] = torch.where(
                reset_rows[:, None],
                torch.zeros_like(ctrl[:, self.muscle_count : self.muscle_count + self.exo_count]),
                ctrl[:, self.muscle_count : self.muscle_count + self.exo_count],
            )
        terms = {
            "torque_action_projection_error": torch.linalg.norm(achieved_normalized - requested, dim=1),
            "torque_action_requested_abs_mean": torch.mean(torch.abs(requested), dim=1),
            "torque_action_achieved_abs_mean": torch.mean(torch.abs(achieved_normalized), dim=1),
            "torque_allocator_target_activation_l2": torch.mean(torch.square(target_activation), dim=1),
        }
        return ctrl, terms
