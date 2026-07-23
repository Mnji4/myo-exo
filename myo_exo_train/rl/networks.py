"""Actor, critic, symmetry, reference-gating, MoE, and Exo networks."""
from __future__ import annotations

from typing import Any

import mujoco
import torch
import torch.nn as nn

from myo_exo_train.env.model import FOOT_SITE_NAMES, TRACK_JOINTS
from myo_exo_train.env.observation import (
    foot_obs_feature_dim,
    footstep_target_dim,
    policy_task_context_dim,
    policy_task_context_mirror_perm,
    terrain_preview_dim,
)

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0

def swap_side_name(name: str) -> str | None:
    """Swap MuJoCo left/right name tokens while leaving sagittal scalar names intact."""
    swapped = str(name)
    if swapped.startswith("r_"):
        swapped = "l_" + swapped[2:]
    elif swapped.startswith("l_"):
        swapped = "r_" + swapped[2:]
    elif swapped.startswith("R_"):
        swapped = "L_" + swapped[2:]
    elif swapped.startswith("L_"):
        swapped = "R_" + swapped[2:]
    marker = "\u0000SIDE_R\u0000"
    swapped = swapped.replace("_r", marker).replace("_l", "_r").replace(marker, "_l")
    marker_upper = "\u0000SIDE_R_UPPER\u0000"
    swapped = swapped.replace("_R", marker_upper).replace("_L", "_R").replace(marker_upper, "_L")
    return swapped if swapped != name else None

def name_mirror_perm(names: list[str]) -> list[int]:
    name_to_idx = {name: idx for idx, name in enumerate(names)}
    perm: list[int] = []
    for idx, name in enumerate(names):
        paired = swap_side_name(name)
        perm.append(name_to_idx.get(paired, idx) if paired is not None else idx)
    for idx, mirrored_idx in enumerate(perm):
        if perm[mirrored_idx] != idx:
            raise ValueError(f"mirror permutation is not involutive at {names[idx]} -> {names[mirrored_idx]}")
    return perm

def mj_names_in_qpos_order(model: mujoco.MjModel) -> list[str]:
    names = [""] * int(model.nq)
    for joint_id in range(int(model.njnt)):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        joint_type = int(model.jnt_type[joint_id])
        qpos_width = 7 if joint_type == int(mujoco.mjtJoint.mjJNT_FREE) else 4 if joint_type == int(mujoco.mjtJoint.mjJNT_BALL) else 1
        start = int(model.jnt_qposadr[joint_id])
        for offset in range(qpos_width):
            names[start + offset] = joint_name if qpos_width == 1 else f"{joint_name}:{offset}"
    if any(name == "" for name in names):
        raise ValueError("could not build complete qpos mirror names")
    return names

def mj_names_in_qvel_order(model: mujoco.MjModel) -> list[str]:
    names = [""] * int(model.nv)
    for joint_id in range(int(model.njnt)):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        joint_type = int(model.jnt_type[joint_id])
        qvel_width = 6 if joint_type == int(mujoco.mjtJoint.mjJNT_FREE) else 3 if joint_type == int(mujoco.mjtJoint.mjJNT_BALL) else 1
        start = int(model.jnt_dofadr[joint_id])
        for offset in range(qvel_width):
            names[start + offset] = joint_name if qvel_width == 1 else f"{joint_name}:{offset}"
    if any(name == "" for name in names):
        raise ValueError("could not build complete qvel mirror names")
    return names

def actuator_names(model: mujoco.MjModel) -> list[str]:
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, idx) for idx in range(int(model.nu))]

def offset_perm(offset: int, perm: list[int]) -> list[int]:
    return [offset + int(index) for index in perm]

def build_sagittal_mirror_spec(
    model: mujoco.MjModel,
    config: dict[str, Any],
    *,
    obs_dim: int,
    future_steps: int,
    device: torch.device,
) -> dict[str, torch.Tensor | dict[str, Any]]:
    qpos_perm = name_mirror_perm(mj_names_in_qpos_order(model))
    qvel_perm = name_mirror_perm(mj_names_in_qvel_order(model))
    action_perm = name_mirror_perm(actuator_names(model))
    obs_act_perm = name_mirror_perm(actuator_names(model)[: int(model.na)])
    track_perm = name_mirror_perm(list(TRACK_JOINTS))
    foot_site_perm = name_mirror_perm(list(FOOT_SITE_NAMES))
    foot_groups = max(1, int(foot_obs_feature_dim(config) // len(FOOT_SITE_NAMES)))
    foot_feature_perm = []
    for group in range(foot_groups):
        foot_feature_perm.extend([group * len(FOOT_SITE_NAMES) + idx for idx in foot_site_perm])
    obs_perm: list[int] = []
    cursor = 0
    obs_perm.extend(offset_perm(cursor, qpos_perm))
    cursor += int(model.nq)
    obs_perm.extend(offset_perm(cursor, qvel_perm))
    cursor += int(model.nv)
    obs_perm.extend(offset_perm(cursor, obs_act_perm))
    cursor += int(model.na)
    obs_perm.extend(offset_perm(cursor, track_perm))
    cursor += len(TRACK_JOINTS)
    obs_perm.extend(offset_perm(cursor, track_perm))
    cursor += len(TRACK_JOINTS)
    phase_obs_start = cursor
    obs_perm.extend([cursor, cursor + 1])  # phase sin/cos: structural mirror only, no half-cycle shift.
    cursor += 2
    obs_perm.extend(offset_perm(cursor, foot_feature_perm))
    cursor += len(foot_feature_perm)
    future_perm = track_perm + [len(TRACK_JOINTS) + idx for idx in foot_feature_perm]
    for _ in range(int(future_steps)):
        obs_perm.extend(offset_perm(cursor, future_perm))
        cursor += len(TRACK_JOINTS) + len(foot_feature_perm)
    terrain_dim = terrain_preview_dim(config)
    obs_perm.extend(range(cursor, cursor + terrain_dim))
    cursor += terrain_dim
    footstep_dim = footstep_target_dim(config)
    footstep_groups = max(0, int(footstep_dim // len(FOOT_SITE_NAMES)))
    footstep_perm: list[int] = []
    for group in range(footstep_groups):
        footstep_perm.extend([group * len(FOOT_SITE_NAMES) + idx for idx in foot_site_perm])
    obs_perm.extend(offset_perm(cursor, footstep_perm))
    cursor += footstep_dim
    task_context_dim = policy_task_context_dim(config)
    obs_perm.extend(offset_perm(cursor, policy_task_context_mirror_perm(config)))
    cursor += task_context_dim
    if cursor != int(obs_dim):
        raise ValueError(f"mirror obs layout mismatch: built {cursor}, runner obs_dim is {obs_dim}")
    for idx, mirrored_idx in enumerate(obs_perm):
        if obs_perm[mirrored_idx] != idx:
            raise ValueError(f"obs mirror permutation is not involutive at dim {idx} -> {mirrored_idx}")

    obs_sign = torch.ones(int(obs_dim), dtype=torch.float32, device=device)
    phase_mode = str(config.get("observation", {}).get("phase_obs", "reference") or "reference").lower()
    if phase_mode in {"root_lateral_heading", "lateral_heading", "root_drift_heading"}:
        obs_sign[phase_obs_start : phase_obs_start + 2] = -1.0
    act_sign = torch.ones(int(model.nu), dtype=torch.float32, device=device)
    return {
        "obs_perm": torch.tensor(obs_perm, dtype=torch.long, device=device),
        "obs_sign": obs_sign,
        "act_perm": torch.tensor(action_perm, dtype=torch.long, device=device),
        "act_sign": torch.ones(int(model.nu), dtype=torch.float32, device=device),
        "metadata": {
            "future_steps": int(future_steps),
            "phase_mirror": "sign_flip" if phase_mode in {"root_lateral_heading", "lateral_heading", "root_drift_heading"} else "unchanged",
            "action_perm_names": actuator_names(model),
            "action_mirror": "permutation",
            "activation_obs_perm_names": actuator_names(model)[: int(model.na)],
            "track_joint_perm": [TRACK_JOINTS[idx] for idx in track_perm],
            "foot_site_perm": [FOOT_SITE_NAMES[idx] for idx in foot_site_perm],
        },
    }

def actor_obs_dim_from_state_dict(state_dict: dict[str, torch.Tensor]) -> int:
    for key in (
        "backbone.0.weight",
        "base_actor.backbone.0.weight",
        "base_encoder.0.weight",
        "base_actor.base_encoder.0.weight",
        "human_trunk.0.weight",
        "base_actor.human_trunk.0.weight",
    ):
        if key in state_dict:
            return int(state_dict[key].shape[1])
    raise KeyError("could not infer actor obs_dim from checkpoint state dict")

def policy_architecture(config: dict[str, Any]) -> str:
    architecture = str(config.get("policy", {}).get("architecture", "gated_ref_sac") or "gated_ref_sac")
    if architecture != "gated_ref_sac":
        raise ValueError(f"unsupported policy architecture: {architecture}; expected gated_ref_sac")
    return architecture

def gated_ref_obs_spec(
    model: mujoco.MjModel,
    config: dict[str, Any],
    *,
    obs_dim: int,
    device: torch.device,
) -> dict[str, torch.Tensor | dict[str, Any]]:
    future_steps = max(0, int(config.get("imitation", {}).get("reference_future_steps", 0) or 0))
    terrain_dim = terrain_preview_dim(config)
    footstep_dim = footstep_target_dim(config)
    task_context_dim = policy_task_context_dim(config)
    foot_dim = foot_obs_feature_dim(config)

    cursor = 0
    qpos = list(range(cursor, cursor + int(model.nq)))
    cursor += int(model.nq)
    qvel = list(range(cursor, cursor + int(model.nv)))
    cursor += int(model.nv)
    act = list(range(cursor, cursor + int(model.na)))
    cursor += int(model.na)
    ref_q = list(range(cursor, cursor + len(TRACK_JOINTS)))
    cursor += len(TRACK_JOINTS)
    ref_dq = list(range(cursor, cursor + len(TRACK_JOINTS)))
    cursor += len(TRACK_JOINTS)
    phase = list(range(cursor, cursor + 2))
    cursor += 2
    foot = list(range(cursor, cursor + foot_dim))
    cursor += foot_dim
    future: list[int] = []
    future_step_dim = len(TRACK_JOINTS) + foot_dim
    for _ in range(future_steps):
        future.extend(range(cursor, cursor + future_step_dim))
        cursor += future_step_dim
    terrain = list(range(cursor, cursor + terrain_dim))
    cursor += terrain_dim
    footstep_target = list(range(cursor, cursor + footstep_dim))
    cursor += footstep_dim
    task_context = list(range(cursor, cursor + task_context_dim))
    cursor += task_context_dim
    if cursor != int(obs_dim):
        raise ValueError(f"gated ref obs layout mismatch: built {cursor}, runner obs_dim is {obs_dim}")

    phase_mode = str(config.get("observation", {}).get("phase_obs", "reference") or "reference").lower()
    phase_is_base = phase_mode in {"root_lateral_heading", "lateral_heading", "root_drift_heading"}
    base_indices = qpos + qvel + act + foot + terrain + footstep_target + task_context
    ref_indices = ref_q + ref_dq + future
    if phase_is_base:
        base_indices += phase
    else:
        ref_indices += phase
    return {
        "base_indices": torch.tensor(base_indices, dtype=torch.long, device=device),
        "ref_indices": torch.tensor(ref_indices, dtype=torch.long, device=device),
        "metadata": {
            "base_dim": len(base_indices),
            "ref_dim": len(ref_indices),
            "qpos": [qpos[0], qpos[-1] + 1] if qpos else [0, 0],
            "qvel": [qvel[0], qvel[-1] + 1] if qvel else [0, 0],
            "act": [act[0], act[-1] + 1] if act else [0, 0],
            "ref_q_error": [ref_q[0], ref_q[-1] + 1] if ref_q else [0, 0],
            "ref_dq_error": [ref_dq[0], ref_dq[-1] + 1] if ref_dq else [0, 0],
            "phase": [phase[0], phase[-1] + 1] if phase else [0, 0],
            "foot": [foot[0], foot[-1] + 1] if foot else [0, 0],
            "future": [future[0], future[-1] + 1] if future else [cursor, cursor],
            "terrain": [terrain[0], terrain[-1] + 1] if terrain else [cursor, cursor],
            "footstep_target": [footstep_target[0], footstep_target[-1] + 1] if footstep_target else [cursor, cursor],
            "task_context": [task_context[0], task_context[-1] + 1] if task_context else [cursor, cursor],
            "future_steps": future_steps,
            "terrain_dim": terrain_dim,
            "footstep_target_dim": footstep_dim,
            "task_context_dim": task_context_dim,
        },
    }

def ref_gate_for_step(config: dict[str, Any], global_step: int, run_start_global_step: int) -> float:
    policy_cfg = config.get("policy", {})
    gate = float(policy_cfg.get("ref_gate", 1.0))
    schedule = policy_cfg.get("ref_gate_schedule", [])
    if not isinstance(schedule, list) or not schedule:
        return max(0.0, min(1.0, gate))
    schedule_step = int(global_step)
    if str(config.get("reward_schedule_mode", "relative")) == "relative":
        schedule_step = max(0, int(global_step) - int(run_start_global_step))
    for item in sorted(schedule, key=lambda x: int(x.get("after_steps", 0))):
        start = int(item.get("after_steps", 0))
        if schedule_step < start:
            continue
        if "gate" in item:
            gate = float(item["gate"])
        duration = int(item.get("duration_steps", 0) or 0)
        if duration > 0 and "end_gate" in item:
            start_gate = float(item.get("gate", gate))
            end_gate = float(item["end_gate"])
            progress = max(0.0, min(1.0, float(schedule_step - start) / float(duration)))
            gate = start_gate + progress * (end_gate - start_gate)
    return max(0.0, min(1.0, float(gate)))

def set_actor_ref_gate(actor: nn.Module, gate: float) -> None:
    if hasattr(actor, "set_ref_gate"):
        actor.set_ref_gate(gate)  # type: ignore[attr-defined]
    elif isinstance(actor, SymmetricSACActor) and hasattr(actor.base_actor, "set_ref_gate"):
        actor.base_actor.set_ref_gate(gate)  # type: ignore[attr-defined]

def gated_ref_base_actor(actor: nn.Module) -> GatedRefSACActor | None:
    base = actor.base_actor if isinstance(actor, SymmetricSACActor) else actor
    return base if isinstance(base, GatedRefSACActor) else None

def actor_optimizer_groups(actor: nn.Module, *, policy_lr: float, exo_lr: float | None) -> list[dict[str, Any]]:
    base = gated_ref_base_actor(actor)
    if base is None:
        return [{"params": [param for param in actor.parameters() if param.requires_grad], "lr": float(policy_lr)}]
    exo_params = (
        [param for param in base.exo_policy_head.parameters() if param.requires_grad]
        if base.exo_head_enabled and exo_lr is not None
        else []
    )
    exo_ids = {id(param) for param in exo_params}
    human_params = [
        param for param in actor.parameters()
        if param.requires_grad and id(param) not in exo_ids
    ]
    groups: list[dict[str, Any]] = []
    if human_params:
        groups.append({"params": human_params, "lr": float(policy_lr), "group_name": "human_base"})
    if exo_params:
        groups.append({"params": exo_params, "lr": float(exo_lr), "group_name": "exo_head"})
    if not groups:
        raise ValueError("actor has no trainable parameters")
    return groups

def mask_ref_obs_for_q(obs: torch.Tensor, ref_indices: torch.Tensor | None, ref_gate: float) -> torch.Tensor:
    if ref_indices is None or int(ref_indices.numel()) == 0 or float(ref_gate) >= 0.999999:
        return obs
    masked = obs.clone()
    masked.index_copy_(
        1,
        ref_indices,
        masked.index_select(1, ref_indices) * float(ref_gate),
    )
    return masked

class TerrainConditionedExoHead(nn.Module):
    def __init__(self, obs_dim: int, exo_dim: int, config: dict[str, Any]):
        super().__init__()
        self.head_type = str(config.get("type", "single") or "single").lower()
        self.num_experts = max(1, int(config.get("num_experts", 3)))
        self.temperature = max(float(config.get("gate_temperature", 1.0)), 1e-3)
        hidden = max(1, int(config.get("hidden_dim", 32)))
        initial_mean = float(config.get("initial_raw_mean", -3.0))
        configured_indices = config.get("observation_indices", [])
        if configured_indices:
            if not isinstance(configured_indices, list):
                raise ValueError("policy.exo_head.observation_indices must be a list")
            indices = [int(index) for index in configured_indices]
            if len(set(indices)) != len(indices):
                raise ValueError("Exo head observation indices must be unique")
            if min(indices) < 0 or max(indices) >= int(obs_dim):
                raise ValueError(f"Exo head observation indices must be within [0, {int(obs_dim)})")
        else:
            indices = list(range(int(obs_dim)))
        self.register_buffer("observation_indices", torch.tensor(indices, dtype=torch.long))
        symmetry_cfg = config.get("bilateral_symmetry", {})
        if not isinstance(symmetry_cfg, dict):
            raise ValueError("policy.exo_head.bilateral_symmetry must be an object")
        self.bilateral_symmetry_enabled = bool(symmetry_cfg.get("enabled", False))
        if self.bilateral_symmetry_enabled:
            input_permutation = [int(index) for index in symmetry_cfg.get("input_permutation", [])]
            input_signs = [float(value) for value in symmetry_cfg.get("input_signs", [])]
            output_permutation = [int(index) for index in symmetry_cfg.get("output_permutation", [])]
            output_signs = [float(value) for value in symmetry_cfg.get("output_signs", [])]
            if len(input_permutation) != len(indices) or sorted(input_permutation) != list(range(len(indices))):
                raise ValueError("bilateral_symmetry.input_permutation must permute the selected Exo observations")
            if len(input_signs) != len(indices) or any(abs(abs(value) - 1.0) > 1e-6 for value in input_signs):
                raise ValueError("bilateral_symmetry.input_signs must contain one +/-1 value per selected observation")
            if len(output_permutation) != int(exo_dim) or sorted(output_permutation) != list(range(int(exo_dim))):
                raise ValueError("bilateral_symmetry.output_permutation must permute the Exo actions")
            if len(output_signs) != int(exo_dim) or any(abs(abs(value) - 1.0) > 1e-6 for value in output_signs):
                raise ValueError("bilateral_symmetry.output_signs must contain one +/-1 value per Exo action")
            input_twice = [input_permutation[index] for index in input_permutation]
            output_twice = [output_permutation[index] for index in output_permutation]
            if input_twice != list(range(len(indices))) or output_twice != list(range(int(exo_dim))):
                raise ValueError("bilateral symmetry permutations must be involutions")
            self.register_buffer("mirror_input_permutation", torch.tensor(input_permutation, dtype=torch.long))
            self.register_buffer("mirror_input_signs", torch.tensor(input_signs, dtype=torch.float32))
            self.register_buffer("mirror_output_permutation", torch.tensor(output_permutation, dtype=torch.long))
            self.register_buffer("mirror_output_signs", torch.tensor(output_signs, dtype=torch.float32))
        self.encoder = nn.Sequential(nn.Linear(len(indices), hidden), nn.ReLU())
        nn.init.normal_(self.encoder[0].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.encoder[0].bias)

        def output(index: int) -> nn.Linear:
            layer = nn.Linear(hidden, int(exo_dim))
            nn.init.zeros_(layer.weight)
            nn.init.constant_(layer.bias, initial_mean)
            return layer

        if self.head_type == "single":
            self.single = output(0)
            self.experts = nn.ModuleList()
            self.gate = None
        elif self.head_type == "moe":
            self.single = output(0)
            self.experts = nn.ModuleList([output(index) for index in range(self.num_experts)])
            self.gate = nn.Linear(hidden, self.num_experts)
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)
            break_std = max(0.0, float(config.get("expert_symmetry_break_std", 1e-3)))
            if self.num_experts > 1 and break_std > 0.0:
                with torch.no_grad():
                    weights = torch.randn(
                        self.num_experts,
                        int(exo_dim),
                        hidden,
                        dtype=self.experts[0].weight.dtype,
                    ) * break_std
                    weights.sub_(weights.mean(dim=0, keepdim=True))
                    biases = torch.randn(
                        self.num_experts,
                        int(exo_dim),
                        dtype=self.experts[0].bias.dtype,
                    ) * break_std
                    biases.sub_(biases.mean(dim=0, keepdim=True))
                    biases.add_(initial_mean)
                    for index, expert in enumerate(self.experts):
                        expert.weight.copy_(weights[index])
                        expert.bias.copy_(biases[index])
        else:
            raise ValueError(f"unsupported policy.exo_head.type: {self.head_type}")
        self.logstd = nn.Parameter(
            torch.full((int(exo_dim),), float(config.get("logstd_init", -3.0)), dtype=torch.float32)
        )
        self.last_gate: torch.Tensor | None = None

    def _mean(self, selected_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        h = self.encoder(selected_obs)
        if self.head_type == "single":
            assert self.single is not None
            mean = self.single(h)
            gate = None
        else:
            assert self.single is not None
            assert self.gate is not None
            expert_mean = torch.stack([head(h) for head in self.experts], dim=1)
            gate = torch.softmax(self.gate(h) / self.temperature, dim=1)
            centered_gate = gate - gate.mean(dim=1, keepdim=True)
            mean = self.single(h) + torch.sum(centered_gate.unsqueeze(-1) * expert_mean, dim=1)
        return mean, gate

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        selected_obs = obs.index_select(-1, self.observation_indices)
        mean, gate = self._mean(selected_obs)
        logstd = self.logstd
        if self.bilateral_symmetry_enabled:
            mirrored_obs = selected_obs.index_select(-1, self.mirror_input_permutation)
            mirrored_obs = mirrored_obs * self.mirror_input_signs.to(dtype=mirrored_obs.dtype)
            mirrored_mean, mirrored_gate = self._mean(mirrored_obs)
            mirrored_mean = mirrored_mean.index_select(-1, self.mirror_output_permutation)
            mirrored_mean = mirrored_mean * self.mirror_output_signs.to(dtype=mirrored_mean.dtype)
            mean = 0.5 * (mean + mirrored_mean)
            mirrored_logstd = logstd.index_select(-1, self.mirror_output_permutation)
            logstd = 0.5 * (logstd + mirrored_logstd)
            if gate is not None and mirrored_gate is not None:
                gate = 0.5 * (gate + mirrored_gate)
        self.last_gate = None if gate is None else gate.detach()
        return mean, logstd.unsqueeze(0).expand(obs.shape[0], -1)

class SharedLatentExoHead(nn.Module):
    """Two-action Exo head that reuses the human actor's post-fusion latent."""

    def __init__(self, latent_dim: int, exo_dim: int, config: dict[str, Any]):
        super().__init__()
        self.head_type = "shared_latent"
        self.mean = nn.Linear(int(latent_dim), int(exo_dim))
        nn.init.zeros_(self.mean.weight)
        nn.init.constant_(self.mean.bias, float(config.get("initial_raw_mean", 0.0)))
        self.logstd = nn.Parameter(
            torch.full((int(exo_dim),), float(config.get("logstd_init", -5.0)), dtype=torch.float32)
        )
        self.last_gate: torch.Tensor | None = None

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean(latent)
        return mean, self.logstd.unsqueeze(0).expand(latent.shape[0], -1)

class GatedRefSACActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        *,
        base_indices: torch.Tensor,
        ref_indices: torch.Tensor,
        logstd_init: float,
        initial_action_mean: float,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        initial_ref_gate: float = 1.0,
        muscle_count: int = 0,
        exo_head_config: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.register_buffer("base_indices", base_indices.detach().clone().long())
        self.register_buffer("ref_indices", ref_indices.detach().clone().long())
        self.register_buffer("ref_gate", torch.tensor(float(initial_ref_gate), dtype=torch.float32))
        self.base_encoder = nn.Sequential(
            nn.Linear(int(self.base_indices.numel()), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(latent_dim)),
            nn.ReLU(),
        )
        self.ref_encoder = nn.Sequential(
            nn.Linear(int(self.ref_indices.numel()), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(latent_dim)),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(int(latent_dim) * 2, int(hidden_dim)),
            nn.ReLU(),
        )
        self.mean = nn.Linear(int(hidden_dim), act_dim)
        self.logstd = nn.Linear(int(hidden_dim), act_dim)
        nn.init.zeros_(self.mean.weight)
        nn.init.constant_(self.mean.bias, float(initial_action_mean))
        nn.init.zeros_(self.logstd.weight)
        nn.init.constant_(self.logstd.bias, float(logstd_init))
        exo_cfg = exo_head_config if isinstance(exo_head_config, dict) else {}
        self.exo_head_enabled = bool(exo_cfg.get("enabled", False))
        self.exo_head_shared_latent = False
        self.muscle_count = max(0, min(int(muscle_count), int(act_dim)))
        self.exo_dim = int(act_dim) - self.muscle_count
        if self.exo_head_enabled:
            if self.exo_dim <= 0:
                raise ValueError("exo policy head enabled but actor has no non-muscle actions")
            self.exo_head_shared_latent = str(exo_cfg.get("type", "single")).lower() == "shared_latent"
            if self.exo_head_shared_latent:
                self.exo_policy_head = SharedLatentExoHead(int(hidden_dim), self.exo_dim, exo_cfg)
            else:
                self.exo_policy_head = TerrainConditionedExoHead(self.obs_dim, self.exo_dim, exo_cfg)

    def set_ref_gate(self, gate: float) -> None:
        self.ref_gate.fill_(max(0.0, min(1.0, float(gate))))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        base_obs = obs.index_select(-1, self.base_indices)
        ref_obs = obs.index_select(-1, self.ref_indices)
        z_base = self.base_encoder(base_obs)
        z_ref = self.ref_encoder(ref_obs) * self.ref_gate.to(dtype=z_base.dtype, device=z_base.device)
        h = self.head(torch.cat([z_base, z_ref], dim=-1))
        mean = self.mean(h)
        logstd = torch.clamp(self.logstd(h), LOG_STD_MIN, LOG_STD_MAX)
        if self.exo_head_enabled:
            exo_input = h if self.exo_head_shared_latent else obs
            exo_mean, exo_logstd = self.exo_policy_head(exo_input)
            mean = torch.cat([mean[:, : self.muscle_count], exo_mean], dim=1)
            logstd = torch.cat([logstd[:, : self.muscle_count], exo_logstd], dim=1)
        return mean, logstd

    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        mean, logstd = self(obs)
        if deterministic:
            action = torch.tanh(mean)
            logprob = torch.zeros(action.shape[0], dtype=torch.float32, device=action.device)
            return action, logprob
        std = torch.exp(logstd)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        logprob = normal.log_prob(x_t)
        logprob -= torch.log(torch.clamp(1.0 - torch.square(action), min=1e-6))
        return action, logprob.sum(dim=1)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if action is None:
            action, logprob = self.get_action(obs, deterministic=deterministic)
        else:
            logprob = torch.zeros(action.shape[0], dtype=torch.float32, device=action.device)
        entropy = torch.zeros(action.shape[0], dtype=torch.float32, device=action.device)
        value = torch.zeros((action.shape[0], 1), dtype=torch.float32, device=action.device)
        return action, logprob, entropy, value

class SymmetricSACActor(nn.Module):
    def __init__(
        self,
        base_actor: nn.Module,
        *,
        obs_perm: torch.Tensor,
        obs_sign: torch.Tensor,
        act_perm: torch.Tensor,
        act_sign: torch.Tensor,
    ):
        super().__init__()
        self.base_actor = base_actor
        self.register_buffer("obs_perm", obs_perm.detach().clone().long())
        self.register_buffer("obs_sign", obs_sign.detach().clone().float())
        self.register_buffer("act_perm", act_perm.detach().clone().long())
        self.register_buffer("act_sign", act_sign.detach().clone().float())

    def mirror_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return obs.index_select(-1, self.obs_perm) * self.obs_sign

    def mirror_action(self, action: torch.Tensor) -> torch.Tensor:
        return action.index_select(-1, self.act_perm) * self.act_sign

    def mirror_logstd(self, logstd: torch.Tensor) -> torch.Tensor:
        return self.mirror_action(logstd)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean0, logstd0 = self.base_actor(obs)
        mean_m, logstd_m = self.base_actor(self.mirror_obs(obs))
        mean = 0.5 * (mean0 + self.mirror_action(mean_m))
        logstd = 0.5 * (logstd0 + self.mirror_logstd(logstd_m))
        return mean, torch.clamp(logstd, LOG_STD_MIN, LOG_STD_MAX)

    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        mean, logstd = self(obs)
        if deterministic:
            action = torch.tanh(mean)
            logprob = torch.zeros(action.shape[0], dtype=torch.float32, device=action.device)
            return action, logprob
        std = torch.exp(logstd)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        logprob = normal.log_prob(x_t)
        logprob -= torch.log(torch.clamp(1.0 - torch.square(action), min=1e-6))
        return action, logprob.sum(dim=1)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if action is None:
            action, logprob = self.get_action(obs, deterministic=deterministic)
        else:
            logprob = torch.zeros(action.shape[0], dtype=torch.float32, device=action.device)
        entropy = torch.zeros(action.shape[0], dtype=torch.float32, device=action.device)
        value = torch.zeros((action.shape[0], 1), dtype=torch.float32, device=action.device)
        return action, logprob, entropy, value

class SoftQNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=1)).squeeze(1)

class SymmetricSoftQNetwork(nn.Module):
    def __init__(
        self,
        base_q: SoftQNetwork,
        *,
        obs_perm: torch.Tensor,
        obs_sign: torch.Tensor,
        act_perm: torch.Tensor,
        act_sign: torch.Tensor,
    ):
        super().__init__()
        self.base_q = base_q
        self.register_buffer("obs_perm", obs_perm.detach().clone().long())
        self.register_buffer("obs_sign", obs_sign.detach().clone().float())
        self.register_buffer("act_perm", act_perm.detach().clone().long())
        self.register_buffer("act_sign", act_sign.detach().clone().float())

    def mirror_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return obs.index_select(-1, self.obs_perm) * self.obs_sign

    def mirror_action(self, action: torch.Tensor) -> torch.Tensor:
        return action.index_select(-1, self.act_perm) * self.act_sign

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q0 = self.base_q(obs, action)
        q1 = self.base_q(self.mirror_obs(obs), self.mirror_action(action))
        return 0.5 * (q0 + q1)

def build_sac_q_modules_for_config(
    *,
    obs_dim: int,
    act_dim: int,
    mirror_spec: dict[str, torch.Tensor | dict[str, Any]] | None,
    device: torch.device,
) -> tuple[nn.Module, nn.Module, nn.Module, nn.Module]:
    if mirror_spec is None:
        qf1: nn.Module = SoftQNetwork(obs_dim, act_dim).to(device)
        qf2: nn.Module = SoftQNetwork(obs_dim, act_dim).to(device)
        qf1_target: nn.Module = SoftQNetwork(obs_dim, act_dim).to(device)
        qf2_target: nn.Module = SoftQNetwork(obs_dim, act_dim).to(device)
    else:
        qf1 = SymmetricSoftQNetwork(
            SoftQNetwork(obs_dim, act_dim).to(device),
            obs_perm=mirror_spec["obs_perm"],  # type: ignore[index]
            obs_sign=mirror_spec["obs_sign"],  # type: ignore[index]
            act_perm=mirror_spec["act_perm"],  # type: ignore[index]
            act_sign=mirror_spec["act_sign"],  # type: ignore[index]
        ).to(device)
        qf2 = SymmetricSoftQNetwork(
            SoftQNetwork(obs_dim, act_dim).to(device),
            obs_perm=mirror_spec["obs_perm"],  # type: ignore[index]
            obs_sign=mirror_spec["obs_sign"],  # type: ignore[index]
            act_perm=mirror_spec["act_perm"],  # type: ignore[index]
            act_sign=mirror_spec["act_sign"],  # type: ignore[index]
        ).to(device)
        qf1_target = SymmetricSoftQNetwork(
            SoftQNetwork(obs_dim, act_dim).to(device),
            obs_perm=mirror_spec["obs_perm"],  # type: ignore[index]
            obs_sign=mirror_spec["obs_sign"],  # type: ignore[index]
            act_perm=mirror_spec["act_perm"],  # type: ignore[index]
            act_sign=mirror_spec["act_sign"],  # type: ignore[index]
        ).to(device)
        qf2_target = SymmetricSoftQNetwork(
            SoftQNetwork(obs_dim, act_dim).to(device),
            obs_perm=mirror_spec["obs_perm"],  # type: ignore[index]
            obs_sign=mirror_spec["obs_sign"],  # type: ignore[index]
            act_perm=mirror_spec["act_perm"],  # type: ignore[index]
            act_sign=mirror_spec["act_sign"],  # type: ignore[index]
        ).to(device)
    return qf1, qf2, qf1_target, qf2_target

@torch.no_grad()
def symmetric_module_self_test(
    actor: SymmetricSACActor,
    qf: SymmetricSoftQNetwork,
    *,
    obs_dim: int,
    act_dim: int,
    device: torch.device,
) -> dict[str, float]:
    obs = torch.randn((1024, int(obs_dim)), dtype=torch.float32, device=device)
    action = torch.randn((1024, int(act_dim)), dtype=torch.float32, device=device)
    obs_roundtrip = actor.mirror_obs(actor.mirror_obs(obs))
    action_roundtrip = actor.mirror_action(actor.mirror_action(action))
    mean, logstd = actor(obs)
    mean_from_mirror, logstd_from_mirror = actor(actor.mirror_obs(obs))
    q = qf(obs, action)
    q_from_mirror = qf(actor.mirror_obs(obs), actor.mirror_action(action))
    return {
        "obs_roundtrip_max_abs": float((obs_roundtrip - obs).abs().max().item()),
        "action_roundtrip_max_abs": float((action_roundtrip - action).abs().max().item()),
        "actor_mean_equivariance_max_abs": float((actor.mirror_action(mean_from_mirror) - mean).abs().max().item()),
        "actor_logstd_equivariance_max_abs": float((actor.mirror_logstd(logstd_from_mirror) - logstd).abs().max().item()),
        "q_invariance_max_abs": float((q_from_mirror - q).abs().max().item()),
    }
