"""Shared models and sensor features for flat22 allocator distillation."""
from __future__ import annotations

from collections import deque

import mujoco
import numpy as np
import torch
import torch.nn as nn


LEGACY_PROPRIO_JOINTS = (
    "pelvis_tilt",
    "hip_flexion_r",
    "knee_angle_r",
    "ankle_angle_r",
    "hip_flexion_l",
    "knee_angle_l",
    "ankle_angle_l",
)
HIP_ENCODER_JOINTS = ("hip_flexion_r", "hip_flexion_l")
EXO_SENSOR_MODES = ("legacy16", "hip4", "hip4_exo6")
EXO_POLICY_INPUT_MODES = ("full_history", "kinematic_history")


def exo_policy_features(
    history: torch.Tensor,
    history_steps: int,
    sensor_mode: str,
    input_mode: str = "full_history",
) -> torch.Tensor:
    """Select Exo-policy inputs while retaining command history for the human."""
    if input_mode not in EXO_POLICY_INPUT_MODES:
        raise ValueError(f"unsupported Exo policy input mode: {input_mode}")
    if input_mode == "full_history" or sensor_mode == "hip4":
        return history
    if sensor_mode != "hip4_exo6":
        raise ValueError(
            "kinematic_history requires hip4 or hip4_exo6 sensor history"
        )
    return history.reshape(-1, int(history_steps), 6)[:, :, :4].reshape(
        -1, int(history_steps) * 4
    )


class ExoStudent(nn.Module):
    """Small deterministic policy for the two normalized hip Exo commands."""

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), 2),
            nn.Tanh(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


class ExoPlanStudent(nn.Module):
    """Predict the current and near-future bilateral Exo commands."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        plan_steps: int = 8,
    ):
        super().__init__()
        if plan_steps < 1:
            raise ValueError("plan_steps must be positive")
        self.plan_steps = int(plan_steps)
        self.network = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), self.plan_steps * 2),
            nn.Tanh(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value).reshape(-1, self.plan_steps, 2)


class RecurrentExoPlanStudent(nn.Module):
    """Predict an Exo plan while retaining state across the whole route."""

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 128,
        plan_steps: int = 8,
    ):
        super().__init__()
        if plan_steps < 1:
            raise ValueError("plan_steps must be positive")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.plan_steps = int(plan_steps)
        self.gru = nn.GRU(self.input_dim, self.hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.plan_steps * 2),
            nn.Tanh(),
        )

    def forward_sequence(
        self,
        value: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features, hidden = self.gru(value, hidden)
        plan = self.head(features).reshape(
            value.shape[0], value.shape[1], self.plan_steps, 2
        )
        return plan, hidden

    def step(
        self,
        value: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        plan, hidden = self.forward_sequence(value.unsqueeze(1), hidden)
        return plan[:, 0], hidden


class RecurrentExoStudent(nn.Module):
    """Causal Exo policy with a hard per-frame command-rate limit."""

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 64,
        expert_count: int = 1,
        max_delta: float = 0.12,
        output_mode: str = "delta",
    ):
        super().__init__()
        if expert_count < 1:
            raise ValueError("expert_count must be positive")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.expert_count = int(expert_count)
        self.max_delta = float(max_delta)
        if output_mode not in {"delta", "absolute_slew"}:
            raise ValueError(f"unsupported recurrent Exo output mode: {output_mode}")
        self.output_mode = str(output_mode)
        self.gru = nn.GRU(self.input_dim, self.hidden_dim, batch_first=True)
        self.expert_head = nn.Linear(
            self.hidden_dim, self.expert_count * 2
        )
        self.gate_head = (
            nn.Linear(self.hidden_dim, self.expert_count)
            if self.expert_count > 1
            else None
        )

    def forward_sequence(
        self,
        normalized_input: torch.Tensor,
        previous_exo: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        features, hidden = self.gru(normalized_input, hidden)
        raw_output = self.expert_head(features).reshape(
            *features.shape[:-1], self.expert_count, 2
        )
        if self.output_mode == "delta":
            delta = self.max_delta * torch.tanh(raw_output)
        else:
            target = torch.tanh(raw_output)
            delta = torch.clamp(
                target - previous_exo.unsqueeze(-2),
                -self.max_delta,
                self.max_delta,
            )
        expert_action = torch.clamp(
            previous_exo.unsqueeze(-2) + delta, -1.0, 1.0
        )
        if self.gate_head is None:
            return expert_action.squeeze(-2), hidden, None
        gate_logits = self.gate_head(features)
        gate_weight = torch.softmax(gate_logits, dim=-1)
        action = torch.sum(
            gate_weight.unsqueeze(-1) * expert_action, dim=-2
        )
        return action, hidden, gate_logits

    def step(
        self,
        normalized_input: torch.Tensor,
        previous_exo: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        action, hidden, gate_logits = self.forward_sequence(
            normalized_input.unsqueeze(1),
            previous_exo.unsqueeze(1),
            hidden,
        )
        return (
            action[:, 0],
            hidden,
            None if gate_logits is None else gate_logits[:, 0],
        )


class ExoConditionedHumanStudent(nn.Module):
    """Adjust a human action after the current Exo command is known."""

    def __init__(
        self,
        obs_dim: int,
        muscle_count: int,
        hidden_dim: int = 256,
        residual_indices: list[int] | tuple[int, ...] | None = None,
        exo_context_dim: int = 2,
        zero_centered: bool = False,
        absolute_output: bool = False,
    ):
        super().__init__()
        self.muscle_count = int(muscle_count)
        self.exo_context_dim = int(exo_context_dim)
        self.zero_centered = bool(zero_centered)
        self.absolute_output = bool(absolute_output)
        if self.exo_context_dim < 1:
            raise ValueError("exo_context_dim must be positive")
        indices = (
            tuple(range(self.muscle_count))
            if residual_indices is None
            else tuple(int(index) for index in residual_indices)
        )
        if not indices or len(set(indices)) != len(indices):
            raise ValueError("residual_indices must contain unique muscle indices")
        if min(indices) < 0 or max(indices) >= self.muscle_count:
            raise ValueError("residual_indices contain an out-of-range muscle")
        self.register_buffer(
            "residual_indices",
            torch.tensor(indices, dtype=torch.long),
            persistent=False,
        )
        self.network = nn.Sequential(
            nn.Linear(
                int(obs_dim) + self.muscle_count + self.exo_context_dim,
                int(hidden_dim),
            ),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), len(indices)),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self,
        obs: torch.Tensor,
        base_muscle_action: torch.Tensor,
        exo_context: torch.Tensor,
    ) -> torch.Tensor:
        if exo_context.shape[-1] != self.exo_context_dim:
            raise ValueError(
                "unexpected Exo context width: "
                f"{exo_context.shape[-1]} vs {self.exo_context_dim}"
            )
        inputs = torch.cat([obs, base_muscle_action, exo_context], dim=-1)
        raw_residual = self.network(inputs)
        if self.absolute_output:
            return torch.tanh(raw_residual)
        if self.zero_centered:
            zero_inputs = torch.cat(
                [obs, base_muscle_action, torch.zeros_like(exo_context)], dim=-1
            )
            raw_residual = raw_residual - self.network(zero_inputs)
        residual = torch.tanh(raw_residual)
        if int(self.residual_indices.numel()) == self.muscle_count:
            return torch.clamp(base_muscle_action + residual, -1.0, 1.0)
        result = base_muscle_action.clone()
        selected = base_muscle_action.index_select(-1, self.residual_indices)
        result.index_copy_(
            -1,
            self.residual_indices,
            torch.clamp(selected + residual, -1.0, 1.0),
        )
        return result


def proprio_indices(
    model: mujoco.MjModel,
    sensor_mode: str = "legacy16",
) -> tuple[np.ndarray, np.ndarray]:
    if sensor_mode not in EXO_SENSOR_MODES:
        raise ValueError(f"unsupported Exo sensor mode: {sensor_mode}")
    joint_names = (
        LEGACY_PROPRIO_JOINTS
        if sensor_mode == "legacy16"
        else HIP_ENCODER_JOINTS
    )
    qpos_indices: list[int] = []
    qvel_indices: list[int] = []
    for name in joint_names:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint < 0:
            raise KeyError(f"missing proprio joint {name}")
        qpos_indices.append(int(model.jnt_qposadr[joint]))
        qvel_indices.append(int(model.jnt_dofadr[joint]))
    return np.asarray(qpos_indices), np.asarray(qvel_indices)


def proprio_frame(
    qpos: np.ndarray,
    qvel: np.ndarray,
    previous_exo: np.ndarray,
    qpos_indices: np.ndarray,
    qvel_indices: np.ndarray,
    sensor_mode: str = "legacy16",
) -> np.ndarray:
    """Build one causal Exo sensor frame."""
    if sensor_mode not in EXO_SENSOR_MODES:
        raise ValueError(f"unsupported Exo sensor mode: {sensor_mode}")
    values = [
        np.asarray(qpos, dtype=np.float32)[qpos_indices],
        np.asarray(qvel, dtype=np.float32)[qvel_indices],
    ]
    if sensor_mode in {"legacy16", "hip4_exo6"}:
        values.append(np.asarray(previous_exo, dtype=np.float32))
    return np.concatenate(values).astype(np.float32, copy=False)


def append_history(
    history: deque[np.ndarray], current: np.ndarray, history_steps: int
) -> np.ndarray:
    if not history:
        for _ in range(max(1, int(history_steps))):
            history.append(np.asarray(current, dtype=np.float32).copy())
    else:
        history.append(np.asarray(current, dtype=np.float32).copy())
    while len(history) > max(1, int(history_steps)):
        history.popleft()
    return np.concatenate(tuple(history)).astype(np.float32, copy=False)


def exo_command_context(
    proprio_history: torch.Tensor,
    current_exo: torch.Tensor,
    history_steps: int,
    sensor_mode: str,
) -> torch.Tensor:
    """Return recent applied Exo commands followed by the current command."""
    if sensor_mode not in {"legacy16", "hip4_exo6"}:
        raise ValueError(
            "Exo-conditioned human requires command history in the sensor input; "
            f"got {sensor_mode}"
        )
    steps = int(history_steps)
    if steps < 1 or proprio_history.shape[-1] % steps != 0:
        raise ValueError(
            "invalid proprio history width: "
            f"{proprio_history.shape[-1]} for {steps} steps"
        )
    frame_width = int(proprio_history.shape[-1] // steps)
    history = proprio_history.reshape(-1, steps, frame_width)[..., -2:]
    return torch.cat((history.flatten(1), current_exo), dim=-1)
