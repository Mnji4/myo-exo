"""Hard-switch expert construction and routing state."""
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
    gated_ref_base_actor,
)
from myo_exo_train.rl.replay_buffer import ReplayBuffer
from myo_exo_train.rl.sac import parse_x_window, parse_x_windows, sample_x_thresholds

@dataclass
class HardSwitchState:
    enabled: bool = False
    mode: str = "x"
    train_expert: str = "stair_B"
    to_stair_x: float = 10.849666533048667
    to_uphill_x: float = 19.34247196446434
    to_stair_x_window: tuple[float, float] | None = None
    to_uphill_x_window: tuple[float, float] | None = None
    boundary_done: bool = True
    use_policy_warmup: bool = True
    train_both: bool = False
    train_stair: bool = True
    train_uphill: bool = False
    uphill_rollout_deterministic: bool = True
    route_replay_only: bool = True
    replay_overlap_enabled: bool = False
    stair_train_windows: list[tuple[float, float]] = field(default_factory=list)
    uphill_train_windows: list[tuple[float, float]] = field(default_factory=list)
    handoff_value_enabled: bool = False
    u_to_s_value_weight: float = 0.0
    s_to_u_value_weight: float = 0.0
    handoff_value_scale: float = 10.0
    stair_train_mask: torch.Tensor | None = None
    uphill_train_mask: torch.Tensor | None = None
    env_to_stair_x: torch.Tensor | None = None
    env_to_uphill_x: torch.Tensor | None = None
    uphill_actor: nn.Module | None = None
    uphill_human_anchor_actor: nn.Module | None = None
    uphill_normalizer: ObsNormalizer | None = None
    uphill_qf1: nn.Module | None = None
    uphill_qf2: nn.Module | None = None
    uphill_qf1_target: nn.Module | None = None
    uphill_qf2_target: nn.Module | None = None
    uphill_actor_optimizer: optim.Optimizer | None = None
    uphill_q_optimizer: optim.Optimizer | None = None
    uphill_alpha_optimizer: optim.Optimizer | None = None
    uphill_log_alpha: torch.Tensor | None = None
    uphill_replay: ReplayBuffer | None = None
    metadata: dict[str, Any] = field(default_factory=lambda: {"enabled": False})

    def route_mask(self, pelvis_forward: torch.Tensor) -> torch.Tensor:
        if self.env_to_stair_x is None or self.env_to_uphill_x is None:
            return (pelvis_forward >= self.to_stair_x) & (pelvis_forward < self.to_uphill_x)
        return (pelvis_forward >= self.env_to_stair_x) & (pelvis_forward < self.env_to_uphill_x)

    def resample_switches(self, rows: torch.Tensor, *, generator: torch.Generator) -> None:
        if self.env_to_stair_x is None or self.env_to_uphill_x is None or not bool(rows.any().item()):
            return
        count = int(rows.sum().item())
        self.env_to_stair_x[rows] = sample_x_thresholds(
            base=self.to_stair_x,
            window=self.to_stair_x_window,
            count=count,
            device=self.env_to_stair_x.device,
            generator=generator,
        )
        self.env_to_uphill_x[rows] = sample_x_thresholds(
            base=self.to_uphill_x,
            window=self.to_uphill_x_window,
            count=count,
            device=self.env_to_uphill_x.device,
            generator=generator,
        )

def build_hard_switch_state(
    *,
    config: dict[str, Any],
    policy_cfg: dict[str, Any],
    sac_cfg: dict[str, Any],
    runner: Any,
    model: Any,
    actor: nn.Module,
    mirror_spec: dict[str, Any] | None,
    device: torch.device,
    nworld: int,
    human_anchor_weight: float,
    reset_critic: bool,
    reset_optimizers: bool,
    reset_alpha: bool,
) -> HardSwitchState:
    cfg = config.get("hard_switch_experts", {})
    state = HardSwitchState(
        enabled=bool(cfg.get("enabled", False)),
        mode=str(cfg.get("mode", "x")).lower(),
        train_expert=str(cfg.get("train_expert", "stair_B")),
        to_stair_x=float(cfg.get("switch_to_stair_x", 10.849666533048667)),
        to_uphill_x=float(cfg.get("switch_to_uphill_x", 19.34247196446434)),
        to_stair_x_window=parse_x_window(cfg.get("switch_to_stair_x_window", None)),
        to_uphill_x_window=parse_x_window(cfg.get("switch_to_uphill_x_window", None)),
        boundary_done=bool(cfg.get("treat_switch_as_done", True)),
        use_policy_warmup=bool(cfg.get("use_policy_during_warmup", True)),
        train_both=bool(cfg.get("train_both", False)),
        train_stair=bool(cfg.get("train_stair", True)),
        uphill_rollout_deterministic=bool(cfg.get("uphill_rollout_deterministic", True)),
        route_replay_only=bool(cfg.get("route_replay_only", True)),
        replay_overlap_enabled=bool(cfg.get("replay_overlap_enabled", False)),
        stair_train_windows=parse_x_windows(cfg.get("stair_train_x_windows", [])),
        uphill_train_windows=parse_x_windows(cfg.get("uphill_train_x_windows", [])),
        handoff_value_enabled=bool(cfg.get("handoff_value_bonus_enabled", False)),
        u_to_s_value_weight=float(cfg.get("u_to_s_value_bonus_weight", 0.0)),
        s_to_u_value_weight=float(cfg.get("s_to_u_value_bonus_weight", 0.0)),
        handoff_value_scale=max(1.0e-6, float(cfg.get("handoff_value_scale", 10.0))),
    )
    state.train_uphill = bool(cfg.get("train_uphill", state.train_both))
    if state.replay_overlap_enabled and not state.stair_train_windows:
        state.stair_train_windows = [(7.905666533048667, 22.351285550476067)]
    if state.replay_overlap_enabled and not state.uphill_train_windows:
        state.uphill_train_windows = [(-1.0e9, 13.793666533048668), (16.333666533048667, 1.0e9)]
    if not state.enabled:
        return state

    if state.mode != "x":
        raise ValueError("hard_switch_experts currently supports mode='x' only")
    if state.to_uphill_x <= state.to_stair_x:
        raise ValueError(
            "hard_switch switch_to_uphill_x must be > switch_to_stair_x, got "
            f"{state.to_stair_x} -> {state.to_uphill_x}"
        )
    if state.to_stair_x_window is not None and state.to_uphill_x_window is not None:
        if state.to_uphill_x_window[0] <= state.to_stair_x_window[1]:
            raise ValueError(
                "hard_switch switch x windows must not overlap: "
                f"{state.to_stair_x_window} -> {state.to_uphill_x_window}"
            )
    if state.train_expert != "stair_B" and not state.train_both:
        raise ValueError("hard_switch_experts currently trains the current actor as train_expert='stair_B'")

    checkpoint_value = cfg.get("uphill_checkpoint", "")
    if not checkpoint_value:
        raise ValueError("hard_switch_experts.enabled=true requires uphill_checkpoint")
    checkpoint_path = resolve_root_path(checkpoint_value)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state.uphill_actor, state.uphill_normalizer, uphill_meta = build_sac_actor_for_checkpoint(
        checkpoint=checkpoint,
        model=model,
        config=config,
        obs_dim=runner.obs_dim,
        act_dim=runner.act_dim,
        device=device,
    )
    if human_anchor_weight > 0.0:
        state.uphill_human_anchor_actor = copy.deepcopy(state.uphill_actor).eval()
        for param in state.uphill_human_anchor_actor.parameters():
            param.requires_grad_(False)

    uphill_base_for_resume = gated_ref_base_actor(state.uphill_actor)
    if (
        uphill_base_for_resume is not None
        and uphill_base_for_resume.exo_head_enabled
        and not any("exo_policy_head" in key for key in checkpoint["actor_state_dict"])
    ):
        uphill_meta["exact_actor"] = False
    if bool(policy_cfg.get("exo_head", {}).get("share_across_hard_switch", False)):
        stair_base = gated_ref_base_actor(actor)
        uphill_base = gated_ref_base_actor(state.uphill_actor)
        if stair_base is None or uphill_base is None or not stair_base.exo_head_enabled or not uphill_base.exo_head_enabled:
            raise ValueError("shared hard-switch exo head requires enabled gated_ref_sac exo heads on both actors")
        uphill_base.exo_policy_head = stair_base.exo_policy_head
        uphill_meta["shared_exo_head"] = True

    actor_lr_scale = float(cfg.get("uphill_actor_lr_scale", 1.0))
    q_lr_scale = float(cfg.get("uphill_q_lr_scale", 1.0))
    alpha_lr_scale = float(cfg.get("uphill_alpha_lr_scale", 1.0))
    if state.train_both:
        state.uphill_qf1, state.uphill_qf2, state.uphill_qf1_target, state.uphill_qf2_target = build_sac_q_modules_for_config(
            obs_dim=runner.obs_dim,
            act_dim=runner.act_dim,
            mirror_spec=mirror_spec,
            device=device,
        )
        old_obs_dim = int(checkpoint.get("run_config", {}).get("obs_dim", 0) or 0)
        if old_obs_dim <= 0:
            old_obs_dim = actor_obs_dim_from_state_dict(checkpoint["actor_state_dict"])
        if reset_critic:
            exact_q1 = exact_q2 = exact_q1_target = exact_q2_target = True
        else:
            exact_q1 = load_shape_compatible_q_state_dict(
                state.uphill_qf1, checkpoint["qf1_state_dict"], old_obs_dim=old_obs_dim,
                new_obs_dim=runner.obs_dim, act_dim=runner.act_dim,
            )
            exact_q2 = load_shape_compatible_q_state_dict(
                state.uphill_qf2, checkpoint["qf2_state_dict"], old_obs_dim=old_obs_dim,
                new_obs_dim=runner.obs_dim, act_dim=runner.act_dim,
            )
            exact_q1_target = load_shape_compatible_q_state_dict(
                state.uphill_qf1_target, checkpoint["qf1_target_state_dict"], old_obs_dim=old_obs_dim,
                new_obs_dim=runner.obs_dim, act_dim=runner.act_dim,
            )
            exact_q2_target = load_shape_compatible_q_state_dict(
                state.uphill_qf2_target, checkpoint["qf2_target_state_dict"], old_obs_dim=old_obs_dim,
                new_obs_dim=runner.obs_dim, act_dim=runner.act_dim,
            )

        learning_rate = float(sac_cfg.get("learning_rate", 3e-4))
        policy_lr = float(sac_cfg.get("policy_lr", learning_rate)) * actor_lr_scale
        q_lr = float(sac_cfg.get("q_lr", learning_rate)) * q_lr_scale
        alpha_lr = float(sac_cfg.get("alpha_lr", learning_rate)) * alpha_lr_scale
        exo_lr = policy_cfg.get("exo_head", {}).get("learning_rate", None)
        state.uphill_actor_optimizer = optim.Adam(
            actor_optimizer_groups(
                state.uphill_actor,
                policy_lr=policy_lr,
                exo_lr=None if exo_lr is None else float(exo_lr),
            ),
            eps=1e-5,
        )
        state.uphill_q_optimizer = optim.Adam(
            list(state.uphill_qf1.parameters()) + list(state.uphill_qf2.parameters()), lr=q_lr, eps=1e-5,
        )
        state.uphill_log_alpha = torch.tensor(
            np.log(float(sac_cfg.get("alpha", 0.2))), dtype=torch.float32, device=device, requires_grad=True,
        )
        if not reset_alpha:
            with torch.no_grad():
                state.uphill_log_alpha.copy_(checkpoint["log_alpha"].to(device))
        state.uphill_alpha_optimizer = optim.Adam([state.uphill_log_alpha], lr=alpha_lr, eps=1e-5)
        if (
            not reset_optimizers
            and bool(uphill_meta.get("exact_actor", False))
            and all([exact_q1, exact_q2, exact_q1_target, exact_q2_target])
        ):
            if "actor_optimizer_state_dict" in checkpoint:
                state.uphill_actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
            if "q_optimizer_state_dict" in checkpoint:
                state.uphill_q_optimizer.load_state_dict(checkpoint["q_optimizer_state_dict"])
            if "alpha_optimizer_state_dict" in checkpoint:
                state.uphill_alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer_state_dict"])
            for group in state.uphill_actor_optimizer.param_groups:
                group["lr"] = float(exo_lr) if group.get("group_name") == "exo_head" and exo_lr is not None else policy_lr
            for group in state.uphill_q_optimizer.param_groups:
                group["lr"] = q_lr
            for group in state.uphill_alpha_optimizer.param_groups:
                group["lr"] = alpha_lr
        state.uphill_replay = ReplayBuffer(
            int(sac_cfg.get("buffer_size", 250000)), runner.obs_dim, runner.act_dim, device,
        )
        uphill_meta.update(
            {
                "exact_qf1": bool(exact_q1),
                "exact_qf2": bool(exact_q2),
                "exact_qf1_target": bool(exact_q1_target),
                "exact_qf2_target": bool(exact_q2_target),
                "trainable": True,
            }
        )
    else:
        for param in state.uphill_actor.parameters():
            param.requires_grad_(False)
        uphill_meta["trainable"] = False

    state.env_to_stair_x = sample_x_thresholds(
        base=state.to_stair_x, window=state.to_stair_x_window, count=nworld, device=device, generator=runner.rng,
    )
    state.env_to_uphill_x = sample_x_thresholds(
        base=state.to_uphill_x, window=state.to_uphill_x_window, count=nworld, device=device, generator=runner.rng,
    )
    state.metadata = {
        "enabled": True,
        "mode": state.mode,
        "train_expert": state.train_expert,
        "train_both": state.train_both,
        "train_stair": state.train_stair,
        "train_uphill": state.train_uphill,
        "frozen_uphill_checkpoint": str(checkpoint_path),
        "switch_to_stair_x": state.to_stair_x,
        "switch_to_uphill_x": state.to_uphill_x,
        "switch_to_stair_x_window": state.to_stair_x_window,
        "switch_to_uphill_x_window": state.to_uphill_x_window,
        "treat_switch_as_done": state.boundary_done,
        "use_policy_during_warmup": state.use_policy_warmup,
        "uphill_rollout_deterministic": state.uphill_rollout_deterministic,
        "uphill_actor_lr_scale": actor_lr_scale,
        "uphill_q_lr_scale": q_lr_scale,
        "uphill_alpha_lr_scale": alpha_lr_scale,
        "route_replay_only": state.route_replay_only,
        "replay_overlap_enabled": state.replay_overlap_enabled,
        "stair_train_x_windows": state.stair_train_windows,
        "uphill_train_x_windows": state.uphill_train_windows,
        "handoff_value_bonus_enabled": state.handoff_value_enabled,
        "u_to_s_value_bonus_weight": state.u_to_s_value_weight,
        "s_to_u_value_bonus_weight": state.s_to_u_value_weight,
        "handoff_value_scale": state.handoff_value_scale,
        "uphill": uphill_meta,
    }
    return state
