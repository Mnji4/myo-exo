"""Four-expert hard routing for a precomposed terrain course."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from myo_exo_train.checkpoint import (
    build_sac_actor_for_checkpoint,
    load_shape_compatible_q_state_dict,
)
from myo_exo_train.env.observation import ObsNormalizer
from myo_exo_train.evaluation import resolve_root_path
from myo_exo_train.rl.networks import (
    actor_obs_dim_from_state_dict,
    actor_optimizer_groups,
    build_sac_q_modules_for_config,
)
from myo_exo_train.rl.replay_buffer import ReplayBuffer
from myo_exo_train.rl.sac import parse_x_window, sample_x_thresholds


@dataclass
class RouteExpert:
    name: str
    actor: nn.Module
    normalizer: ObsNormalizer
    qf1: nn.Module
    qf2: nn.Module
    qf1_target: nn.Module
    qf2_target: nn.Module
    actor_optimizer: optim.Optimizer
    q_optimizer: optim.Optimizer
    alpha_optimizer: optim.Optimizer
    log_alpha: torch.Tensor
    replay: ReplayBuffer
    human_anchor_actor: nn.Module | None = None
    checkpoint: str = ""
    trainable: bool = True
    train_windows: list[tuple[float, float]] = field(default_factory=list)
    rollout_deterministic: bool = False
    train_stats: dict[str, float] = field(default_factory=dict)
    learned_gradient_steps: int = 0


@dataclass
class RouteHandoffTarget:
    sender_index: int
    receiver_index: int
    center_x: float
    scale_x: float
    max_bonus: float
    qpos_target: torch.Tensor
    qvel_target: torch.Tensor
    activation_target: torch.Tensor
    qpos_scale: float
    qvel_scale: float
    activation_scale: float
    qpos_weight: float
    qvel_weight: float
    activation_weight: float


@dataclass
class RouteMoEState:
    enabled: bool = False
    experts: list[RouteExpert] = field(default_factory=list)
    boundary_x: list[float] = field(default_factory=list)
    boundary_windows: list[tuple[float, float] | None] = field(default_factory=list)
    env_boundaries: torch.Tensor | None = None
    boundary_done: bool = True
    use_policy_warmup: bool = True
    rollout_deterministic: bool = False
    pelvis_tx_qpos: int = 0
    handoff_targets: list[RouteHandoffTarget] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=lambda: {"enabled": False})

    def route_index(self, pelvis_forward: torch.Tensor) -> torch.Tensor:
        if self.env_boundaries is None:
            boundaries = torch.tensor(
                self.boundary_x,
                dtype=pelvis_forward.dtype,
                device=pelvis_forward.device,
            ).unsqueeze(0)
        else:
            boundaries = self.env_boundaries
        return torch.sum(
            pelvis_forward.unsqueeze(1) >= boundaries, dim=1
        ).long()

    def resample_switches(
        self, rows: torch.Tensor, *, generator: torch.Generator
    ) -> None:
        if self.env_boundaries is None or not bool(rows.any().item()):
            return
        count = int(rows.sum().item())
        for index, (base, window) in enumerate(
            zip(self.boundary_x, self.boundary_windows, strict=True)
        ):
            self.env_boundaries[rows, index] = sample_x_thresholds(
                base=base,
                window=window,
                count=count,
                device=self.env_boundaries.device,
                generator=generator,
            )

    def handoff_bonus(
        self,
        qpos: torch.Tensor,
        qvel: torch.Tensor,
        activation: torch.Tensor,
        route_index: torch.Tensor,
    ) -> torch.Tensor:
        bonus = torch.zeros(
            qpos.shape[0], dtype=qpos.dtype, device=qpos.device
        )
        for target in self.handoff_targets:
            rows = route_index == target.sender_index
            if not bool(rows.any().item()):
                continue
            sender_qpos = qpos[rows]
            qpos_error = sender_qpos - target.qpos_target.unsqueeze(0)
            qpos_error[:, self.pelvis_tx_qpos] = 0.0
            qpos_rms = torch.sqrt(torch.mean(qpos_error.square(), dim=1))
            qvel_rms = torch.sqrt(
                torch.mean(
                    (qvel[rows] - target.qvel_target.unsqueeze(0)).square(),
                    dim=1,
                )
            )
            activation_rms = torch.sqrt(
                torch.mean(
                    (
                        activation[rows]
                        - target.activation_target.unsqueeze(0)
                    ).square(),
                    dim=1,
                )
            )
            score = (
                target.qpos_weight
                * torch.exp(-0.5 * (qpos_rms / target.qpos_scale).square())
                + target.qvel_weight
                * torch.exp(-0.5 * (qvel_rms / target.qvel_scale).square())
                + target.activation_weight
                * torch.exp(
                    -0.5
                    * (activation_rms / target.activation_scale).square()
                )
            )
            x_gate = torch.exp(
                -0.5
                * (
                    (qpos[rows, self.pelvis_tx_qpos] - target.center_x)
                    / target.scale_x
                ).square()
            )
            bonus[rows] += target.max_bonus * x_gate * score
        return bonus


def _empty_stats(alpha: float) -> dict[str, float]:
    return {
        "q_loss": 0.0,
        "actor_loss": 0.0,
        "sac_actor_loss": 0.0,
        "human_anchor_loss": 0.0,
        "alpha_loss": 0.0,
        "alpha": float(alpha),
        "sample_logprob": 0.0,
        "q_batch_q1_mean": 0.0,
        "q_batch_q2_mean": 0.0,
    }


def build_route_moe_state(
    *,
    config: dict[str, Any],
    policy_cfg: dict[str, Any],
    sac_cfg: dict[str, Any],
    runner: Any,
    model: Any,
    mirror_spec: dict[str, Any] | None,
    device: torch.device,
    nworld: int,
    primary: dict[str, Any],
    human_anchor_weight: float,
    reset_critic: bool,
    reset_optimizers: bool,
    reset_alpha: bool,
) -> RouteMoEState:
    cfg = config.get("route_moe", {})
    state = RouteMoEState(enabled=bool(cfg.get("enabled", False)))
    if not state.enabled:
        return state

    specs = list(cfg.get("experts", []))
    if len(specs) < 2:
        raise ValueError("route_moe requires at least two experts")
    boundaries = list(cfg.get("boundaries", []))
    if len(boundaries) != len(specs) - 1:
        raise ValueError("route_moe boundaries must have len(experts)-1 entries")
    state.boundary_x = [float(item["x"]) for item in boundaries]
    state.boundary_windows = [
        parse_x_window(item.get("window")) for item in boundaries
    ]
    if state.boundary_x != sorted(state.boundary_x):
        raise ValueError("route_moe boundary x values must be increasing")
    for left, right in zip(
        state.boundary_windows[:-1], state.boundary_windows[1:], strict=True
    ):
        if left is not None and right is not None and right[0] <= left[1]:
            raise ValueError("route_moe boundary windows must not overlap")
    state.boundary_done = bool(cfg.get("treat_switch_as_done", True))
    state.use_policy_warmup = bool(cfg.get("use_policy_during_warmup", True))
    state.rollout_deterministic = bool(cfg.get("rollout_deterministic", False))
    state.pelvis_tx_qpos = int(runner.pelvis_tx_qpos)

    def expert_options(spec: dict[str, Any]) -> dict[str, Any]:
        windows = [
            parsed
            for value in spec.get("train_windows", [])
            if (parsed := parse_x_window(value)) is not None
        ]
        return {
            "trainable": bool(spec.get("trainable", True)),
            "train_windows": windows,
            "rollout_deterministic": bool(
                spec.get(
                    "rollout_deterministic",
                    state.rollout_deterministic,
                )
            ),
        }

    primary_name = str(specs[0].get("name", "expert0"))
    primary_checkpoint = str(specs[0].get("checkpoint", ""))
    state.experts.append(
        RouteExpert(
            name=primary_name,
            checkpoint=primary_checkpoint,
            train_stats=_empty_stats(
                float(primary["log_alpha"].exp().detach().item())
            ),
            **expert_options(specs[0]),
            **primary,
        )
    )

    learning_rate = float(sac_cfg.get("learning_rate", 3e-4))
    policy_lr = float(sac_cfg.get("policy_lr", learning_rate))
    q_lr = float(sac_cfg.get("q_lr", learning_rate))
    alpha_lr = float(sac_cfg.get("alpha_lr", learning_rate))
    exo_lr = policy_cfg.get("exo_head", {}).get("learning_rate")
    buffer_size = int(sac_cfg.get("buffer_size", 250000))

    expert_meta: list[dict[str, Any]] = [
        {
            "name": primary_name,
            "checkpoint": primary_checkpoint,
            "primary": True,
            **expert_options(specs[0]),
        }
    ]
    for spec in specs[1:]:
        name = str(spec["name"])
        checkpoint_path = resolve_root_path(spec["checkpoint"])
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        actor, normalizer, actor_meta = build_sac_actor_for_checkpoint(
            checkpoint=checkpoint,
            model=model,
            config=config,
            obs_dim=runner.obs_dim,
            act_dim=runner.act_dim,
            device=device,
        )
        qf1, qf2, qf1_target, qf2_target = build_sac_q_modules_for_config(
            obs_dim=runner.obs_dim,
            act_dim=runner.act_dim,
            mirror_spec=mirror_spec,
            device=device,
        )
        old_obs_dim = int(
            checkpoint.get("run_config", {}).get("obs_dim", 0) or 0
        )
        if old_obs_dim <= 0:
            old_obs_dim = actor_obs_dim_from_state_dict(
                checkpoint["actor_state_dict"]
            )
        exact_q = [True, True, True, True]
        if not reset_critic:
            exact_q = [
                load_shape_compatible_q_state_dict(
                    module,
                    checkpoint[key],
                    old_obs_dim=old_obs_dim,
                    new_obs_dim=runner.obs_dim,
                    act_dim=runner.act_dim,
                )
                for module, key in (
                    (qf1, "qf1_state_dict"),
                    (qf2, "qf2_state_dict"),
                    (qf1_target, "qf1_target_state_dict"),
                    (qf2_target, "qf2_target_state_dict"),
                )
            ]
        actor_optimizer = optim.Adam(
            actor_optimizer_groups(
                actor,
                policy_lr=policy_lr,
                exo_lr=None if exo_lr is None else float(exo_lr),
            ),
            eps=1e-5,
        )
        q_optimizer = optim.Adam(
            list(qf1.parameters()) + list(qf2.parameters()),
            lr=q_lr,
            eps=1e-5,
        )
        log_alpha = torch.tensor(
            np.log(float(sac_cfg.get("alpha", 0.2))),
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        if not reset_alpha:
            with torch.no_grad():
                log_alpha.copy_(checkpoint["log_alpha"].to(device))
        alpha_optimizer = optim.Adam([log_alpha], lr=alpha_lr, eps=1e-5)
        if (
            not reset_optimizers
            and bool(actor_meta.get("exact_actor", False))
            and all(exact_q)
        ):
            actor_optimizer.load_state_dict(
                checkpoint["actor_optimizer_state_dict"]
            )
            q_optimizer.load_state_dict(checkpoint["q_optimizer_state_dict"])
            alpha_optimizer.load_state_dict(
                checkpoint["alpha_optimizer_state_dict"]
            )
            for group in actor_optimizer.param_groups:
                group["lr"] = (
                    float(exo_lr)
                    if group.get("group_name") == "exo_head"
                    and exo_lr is not None
                    else policy_lr
                )
            for group in q_optimizer.param_groups:
                group["lr"] = q_lr
            for group in alpha_optimizer.param_groups:
                group["lr"] = alpha_lr
        anchor = None
        if human_anchor_weight > 0.0:
            anchor = copy.deepcopy(actor).eval()
            for parameter in anchor.parameters():
                parameter.requires_grad_(False)
        state.experts.append(
            RouteExpert(
                name=name,
                actor=actor,
                normalizer=normalizer,
                qf1=qf1,
                qf2=qf2,
                qf1_target=qf1_target,
                qf2_target=qf2_target,
                actor_optimizer=actor_optimizer,
                q_optimizer=q_optimizer,
                alpha_optimizer=alpha_optimizer,
                log_alpha=log_alpha,
                replay=ReplayBuffer(
                    buffer_size, runner.obs_dim, runner.act_dim, device
                ),
                human_anchor_actor=anchor,
                checkpoint=str(checkpoint_path),
                **expert_options(spec),
                train_stats=_empty_stats(
                    float(log_alpha.exp().detach().item())
                ),
            )
        )
        expert_meta.append(
            {
                "name": name,
                "checkpoint": str(checkpoint_path),
                "primary": False,
                "actor": actor_meta,
                "exact_q": all(exact_q),
                **expert_options(spec),
            }
        )

    name_to_index = {
        expert.name: index for index, expert in enumerate(state.experts)
    }
    handoff_meta: list[dict[str, Any]] = []
    for target_cfg in cfg.get("handoff_targets", []):
        sender = str(target_cfg["sender"])
        receiver = str(target_cfg["receiver"])
        if sender not in name_to_index or receiver not in name_to_index:
            raise ValueError(
                f"unknown route handoff {sender!r} -> {receiver!r}"
            )
        bank_path = resolve_root_path(target_cfg["bank"])
        row = int(target_cfg.get("row", 0))
        with np.load(bank_path, allow_pickle=False) as bank:
            qpos_target = torch.as_tensor(
                bank["qpos"][row], dtype=torch.float32, device=device
            )
            qvel_target = torch.as_tensor(
                bank["qvel"][row], dtype=torch.float32, device=device
            )
            activation_target = torch.as_tensor(
                bank["act"][row], dtype=torch.float32, device=device
            )
        weights = target_cfg.get("weights", {})
        target = RouteHandoffTarget(
            sender_index=name_to_index[sender],
            receiver_index=name_to_index[receiver],
            center_x=float(target_cfg["center_x"]),
            scale_x=max(float(target_cfg.get("scale_x", 0.35)), 1e-6),
            max_bonus=float(target_cfg.get("max_bonus", 1.0)),
            qpos_target=qpos_target,
            qvel_target=qvel_target,
            activation_target=activation_target,
            qpos_scale=max(float(target_cfg.get("qpos_scale", 0.18)), 1e-6),
            qvel_scale=max(float(target_cfg.get("qvel_scale", 1.5)), 1e-6),
            activation_scale=max(
                float(target_cfg.get("activation_scale", 0.2)), 1e-6
            ),
            qpos_weight=float(weights.get("qpos", 0.55)),
            qvel_weight=float(weights.get("qvel", 0.25)),
            activation_weight=float(weights.get("activation", 0.2)),
        )
        state.handoff_targets.append(target)
        handoff_meta.append(
            {
                "sender": sender,
                "receiver": receiver,
                "bank": str(bank_path),
                "row": row,
                "center_x": target.center_x,
                "scale_x": target.scale_x,
                "max_bonus": target.max_bonus,
            }
        )

    state.env_boundaries = torch.empty(
        (nworld, len(state.boundary_x)),
        dtype=torch.float32,
        device=device,
    )
    state.resample_switches(
        torch.ones(nworld, dtype=torch.bool, device=device),
        generator=runner.rng,
    )
    state.metadata = {
        "enabled": True,
        "mode": "x",
        "experts": expert_meta,
        "handoff_targets": handoff_meta,
        "boundaries": [
            {"x": x, "window": window}
            for x, window in zip(
                state.boundary_x, state.boundary_windows, strict=True
            )
        ],
        "treat_switch_as_done": state.boundary_done,
        "rollout_deterministic": state.rollout_deterministic,
    }
    return state
