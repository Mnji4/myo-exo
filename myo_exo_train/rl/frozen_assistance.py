"""Frozen recurrent Exo and conditioned-human runtime for SAC fine-tuning."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import mujoco
import torch
import torch.nn as nn


HIP_JOINTS = ("hip_flexion_r", "hip_flexion_l")


class RecurrentExoPolicy(nn.Module):
    """Causal bilateral Exo policy used by the deployment checkpoints."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        expert_count: int,
        max_delta: float,
        output_mode: str,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.expert_count = int(expert_count)
        self.max_delta = float(max_delta)
        self.output_mode = str(output_mode)
        self.gru = nn.GRU(int(input_dim), self.hidden_dim, batch_first=True)
        self.expert_head = nn.Linear(self.hidden_dim, self.expert_count * 2)
        self.gate_head = (
            nn.Linear(self.hidden_dim, self.expert_count)
            if self.expert_count > 1
            else None
        )

    def step(
        self,
        normalized_input: torch.Tensor,
        previous_exo: torch.Tensor,
        hidden: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features, hidden = self.gru(normalized_input.unsqueeze(1), hidden)
        raw = self.expert_head(features[:, 0]).reshape(
            len(normalized_input), self.expert_count, 2
        )
        if self.output_mode == "delta":
            delta = self.max_delta * torch.tanh(raw)
        elif self.output_mode == "absolute_slew":
            delta = torch.clamp(
                torch.tanh(raw) - previous_exo.unsqueeze(1),
                -self.max_delta,
                self.max_delta,
            )
        else:
            raise ValueError(f"unsupported recurrent Exo output mode: {self.output_mode}")
        experts = torch.clamp(previous_exo.unsqueeze(1) + delta, -1.0, 1.0)
        if self.gate_head is None:
            return experts[:, 0], hidden
        weights = torch.softmax(self.gate_head(features[:, 0]), dim=-1)
        return torch.sum(weights.unsqueeze(-1) * experts, dim=1), hidden


class FeedforwardExoPolicy(nn.Module):
    """Feed-forward Exo policy over a fixed causal sensor history."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
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


class ExoConditionedHuman(nn.Module):
    """Frozen correction applied after the current Exo command is known."""

    def __init__(
        self,
        obs_dim: int,
        muscle_count: int,
        hidden_dim: int,
        exo_context_dim: int,
        zero_centered: bool,
        absolute_output: bool,
    ) -> None:
        super().__init__()
        self.muscle_count = int(muscle_count)
        self.exo_context_dim = int(exo_context_dim)
        self.zero_centered = bool(zero_centered)
        self.absolute_output = bool(absolute_output)
        self.register_buffer(
            "residual_indices",
            torch.arange(self.muscle_count, dtype=torch.long),
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
            nn.Linear(int(hidden_dim), self.muscle_count),
        )

    def forward(
        self,
        obs: torch.Tensor,
        base_muscle_action: torch.Tensor,
        exo_context: torch.Tensor,
    ) -> torch.Tensor:
        inputs = torch.cat((obs, base_muscle_action, exo_context), dim=-1)
        residual = self.network(inputs)
        if self.absolute_output:
            return torch.tanh(residual)
        if self.zero_centered:
            zero_inputs = torch.cat(
                (obs, base_muscle_action, torch.zeros_like(exo_context)), dim=-1
            )
            residual = residual - self.network(zero_inputs)
        return torch.clamp(base_muscle_action + torch.tanh(residual), -1.0, 1.0)


class FrozenAssistanceRuntime:
    """Runs the deployed Exo and composes its frozen human response."""

    def __init__(self, config: dict[str, Any], runner: Any) -> None:
        cfg = config.get("frozen_assistance", {})
        self.enabled = bool(cfg.get("enabled", False))
        self.train_conditioner = False
        if not self.enabled:
            return
        self.device = runner.device
        exo_path = Path(str(cfg["exo_checkpoint"])).expanduser()
        human_path = Path(str(cfg["conditioned_human_checkpoint"])).expanduser()
        exo_payload = torch.load(exo_path, map_location=self.device, weights_only=False)
        human_payload = torch.load(human_path, map_location=self.device, weights_only=False)
        self.recurrent_exo = exo_payload.get("model_type") == "recurrent_exo"
        if "proprio_exo_state_dict" not in exo_payload:
            raise ValueError("frozen_assistance Exo checkpoint has no policy weights")
        if human_payload.get("model_type") != "frozen_recurrent_exo_conditioned_human":
            raise ValueError("frozen_assistance requires a conditioned-human checkpoint")
        if self.recurrent_exo and int(exo_payload.get("expert_count", 1)) != 1:
            raise ValueError("frozen_assistance currently supports the shared single Exo only")
        self.history_steps = int(exo_payload["history_steps"])
        if self.history_steps != int(human_payload["history_steps"]):
            raise ValueError("Exo and conditioned-human history lengths differ")
        if self.recurrent_exo:
            self.model = RecurrentExoPolicy(
                input_dim=int(exo_payload["proprio_dim"]),
                hidden_dim=int(exo_payload["hidden_dim"]),
                expert_count=int(exo_payload["expert_count"]),
                max_delta=float(exo_payload["max_delta"]),
                output_mode=str(exo_payload.get("output_mode", "delta")),
            ).to(self.device)
        else:
            self.model = FeedforwardExoPolicy(
                input_dim=int(exo_payload["proprio_dim"]),
                hidden_dim=int(exo_payload["hidden_dim"]),
            ).to(self.device)
        self.model.load_state_dict(exo_payload["proprio_exo_state_dict"], strict=True)
        self.model.requires_grad_(False).eval()
        self.mean = exo_payload["proprio_mean"].to(self.device)
        self.std = exo_payload["proprio_std"].to(self.device)
        self.conditioner = ExoConditionedHuman(
            obs_dim=int(human_payload["obs_dim"]),
            muscle_count=int(runner.model.na),
            hidden_dim=int(human_payload["hidden_dim"]),
            exo_context_dim=int(human_payload["exo_context_dim"]),
            zero_centered=bool(human_payload.get("conditioned_zero_centered", False)),
            absolute_output=bool(
                human_payload.get("conditioned_absolute_output", False)
            ),
        ).to(self.device)
        self.conditioner.load_state_dict(
            human_payload["conditioned_human_state_dict"], strict=True
        )
        self.train_conditioner = bool(cfg.get("train_conditioner", False))
        self.conditioner.requires_grad_(self.train_conditioner)
        self.conditioner.train(self.train_conditioner)
        self.conditioner_anchor = copy.deepcopy(self.conditioner).requires_grad_(False).eval()
        self.context_dim = int(human_payload["exo_context_dim"])
        if self.context_dim != self.history_steps * 2:
            raise ValueError("conditioned-human context is not bilateral command history")
        qpos: list[int] = []
        qvel: list[int] = []
        for name in HIP_JOINTS:
            joint = mujoco.mj_name2id(runner.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint < 0:
                raise KeyError(f"missing Exo encoder joint: {name}")
            qpos.append(int(runner.model.jnt_qposadr[joint]))
            qvel.append(int(runner.model.jnt_dofadr[joint]))
        self.qpos_indices = torch.tensor(qpos, dtype=torch.long, device=self.device)
        self.qvel_indices = torch.tensor(qvel, dtype=torch.long, device=self.device)
        self.hidden = (
            torch.zeros(
                1,
                int(runner.nworld),
                int(exo_payload["hidden_dim"]),
                device=self.device,
            )
            if self.recurrent_exo
            else None
        )
        self.exo_policy_input_mode = str(
            exo_payload.get("exo_policy_input_mode", "full_history")
        )
        self.sensor_history = torch.zeros(
            int(runner.nworld), self.history_steps, 6, device=self.device
        )
        self.command_history = torch.zeros(
            int(runner.nworld), self.history_steps, 2, device=self.device
        )
        self.reset_rows(runner, torch.arange(runner.nworld, device=self.device))

    def _current_sensor(self, runner: Any, rows: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (
                runner.qpos[rows].index_select(1, self.qpos_indices),
                runner.qvel[rows].index_select(1, self.qvel_indices),
                runner.applied_exo_ctrl[rows],
            ),
            dim=1,
        )

    @torch.no_grad()
    def reset_rows(self, runner: Any, rows: torch.Tensor) -> None:
        if not self.enabled or int(rows.numel()) == 0:
            return
        rows = rows.long()
        current = self._current_sensor(runner, rows)
        history = current.unsqueeze(1).repeat(1, self.history_steps, 1)
        sample_indices = getattr(runner, "offline_recovery_last_sample_index", None)
        bank_history = getattr(runner, "offline_recovery_bank_proprio_history", None)
        if sample_indices is not None and bank_history is not None:
            selected = sample_indices[rows]
            valid = selected >= 0
            if bool(valid.any().item()):
                stored = bank_history[selected[valid]].reshape(-1, self.history_steps, 6)
                history[valid] = stored
        self.sensor_history[rows] = history
        if self.recurrent_exo:
            hidden = None
            for frame in range(max(0, self.history_steps - 1)):
                sensor = history[:, frame]
                _, hidden = self.model.step(
                    (sensor - self.mean) / self.std,
                    sensor[:, -2:],
                    hidden,
                )
            if hidden is None:
                hidden = torch.zeros(
                    1, len(rows), self.hidden.shape[-1], device=self.device
                )
            self.hidden[:, rows] = hidden
        self.command_history[rows] = history[:, :, -2:]

    @torch.no_grad()
    def action_and_context(self, runner: Any) -> tuple[torch.Tensor, torch.Tensor]:
        rows = torch.arange(runner.nworld, device=self.device)
        sensor = self._current_sensor(runner, rows)
        self.sensor_history = torch.roll(self.sensor_history, shifts=-1, dims=1)
        self.sensor_history[:, -1] = sensor
        if self.recurrent_exo:
            action, hidden = self.model.step(
                (sensor - self.mean) / self.std,
                sensor[:, -2:],
                self.hidden,
            )
            self.hidden = hidden
        else:
            features = self.sensor_history
            if self.exo_policy_input_mode == "kinematic_history":
                features = features[:, :, :4]
            features = features.flatten(1)
            action = self.model((features - self.mean) / self.std)
        self.command_history = torch.roll(self.command_history, shifts=-1, dims=1)
        self.command_history[:, -1] = action
        return action, self.command_history.flatten(1).clone()

    def compose_action(
        self,
        normalized_obs: torch.Tensor,
        base_action: torch.Tensor,
        exo_context: torch.Tensor,
        exo_action: torch.Tensor,
        muscle_count: int,
    ) -> torch.Tensor:
        result = base_action.clone()
        result[:, :muscle_count] = self.conditioner(
            normalized_obs,
            base_action[:, :muscle_count],
            exo_context,
        )
        result[:, muscle_count : muscle_count + 2] = exo_action
        return result

    def conditioner_anchor_loss(
        self,
        normalized_obs: torch.Tensor,
        base_action: torch.Tensor,
        exo_context: torch.Tensor,
        muscle_count: int,
    ) -> torch.Tensor:
        current = self.conditioner(
            normalized_obs,
            base_action[:, :muscle_count],
            exo_context,
        )
        with torch.no_grad():
            anchor = self.conditioner_anchor(
                normalized_obs,
                base_action[:, :muscle_count],
                exo_context,
            )
        return torch.mean(torch.square(current - anchor))

    def load_conditioner_training_state(self, checkpoint: dict[str, Any]) -> None:
        state = checkpoint.get("frozen_conditioner_state_dict")
        if state is not None:
            self.conditioner.load_state_dict(state, strict=True)

    def checkpoint_payload(self) -> dict[str, Any]:
        if not self.train_conditioner:
            return {}
        return {
            "frozen_conditioner_state_dict": self.conditioner.state_dict(),
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "history_steps": int(self.history_steps),
            "context_dim": int(self.context_dim),
            "train_conditioner": bool(self.train_conditioner),
            "recurrent_exo": bool(self.recurrent_exo),
        }
