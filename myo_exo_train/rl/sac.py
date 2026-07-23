"""SAC schedules, optimization steps, and rollout-side helpers."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from myo_exo_train.env.observation import ObsNormalizer
from myo_exo_train.rl.networks import mask_ref_obs_for_q
from myo_exo_train.rl.replay_buffer import ReplayBuffer

def x_windows_mask(x: torch.Tensor, windows: list[tuple[float, float]]) -> torch.Tensor:
    mask = torch.zeros_like(x, dtype=torch.bool)
    for start, end in windows:
        mask = mask | ((x >= float(start)) & (x < float(end)))
    return mask

def parse_x_windows(raw: Any) -> list[tuple[float, float]]:
    windows: list[tuple[float, float]] = []
    if not isinstance(raw, list):
        return windows
    for item in raw:
        if isinstance(item, dict):
            start = float(item.get("start", item.get("x_start", 0.0)))
            end = float(item.get("end", item.get("x_end", start)))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            start = float(item[0])
            end = float(item[1])
        else:
            continue
        if end > start:
            windows.append((start, end))
    return windows

def parse_x_window(raw: Any) -> tuple[float, float] | None:
    if raw is None:
        return None
    if isinstance(raw, dict):
        start = float(raw.get("start", raw.get("x_start", 0.0)))
        end = float(raw.get("end", raw.get("x_end", start)))
    elif isinstance(raw, (list, tuple)) and len(raw) >= 2:
        start = float(raw[0])
        end = float(raw[1])
    else:
        return None
    if end <= start:
        raise ValueError(f"invalid x window: {raw}")
    return start, end

def sample_x_thresholds(
    *,
    base: float,
    window: tuple[float, float] | None,
    count: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    if window is None:
        return torch.full((count,), float(base), dtype=torch.float32, device=device)
    low, high = window
    return torch.empty((count,), dtype=torch.float32, device=device).uniform_(float(low), float(high), generator=generator)

def action_anchor_weight_for_step(config: dict[str, Any], global_step: int, finetune_start_global_step: int) -> float:
    anchor = config.get("sac", config.get("ppo", {})).get("action_anchor", {})
    if not isinstance(anchor, dict) or not bool(anchor.get("enabled", False)):
        return max(0.0, float(config.get("exo_policy", {}).get("human_anchor_weight", 0.0)))
    initial = max(0.0, float(anchor.get("initial_weight", anchor.get("weight", 0.0))))
    final = max(0.0, float(anchor.get("final_weight", initial)))
    decay_steps = max(0, int(anchor.get("decay_steps", 0)))
    start_after_steps = max(0, int(anchor.get("start_after_steps", 0)))
    if decay_steps == 0:
        return final
    elapsed = max(0, int(global_step) - int(finetune_start_global_step) - start_after_steps)
    fraction = min(1.0, float(elapsed) / float(decay_steps))
    return initial + fraction * (final - initial)

def reset_phase_schedule_for_step(config: dict[str, Any], global_step: int, run_start_global_step: int) -> dict[str, Any] | None:
    schedule = config.get("reset_phase_schedule", [])
    if not isinstance(schedule, list) or not schedule:
        return None
    schedule_step = int(global_step)
    if str(config.get("reset_phase_schedule_mode", "absolute")) == "relative":
        schedule_step = max(0, int(global_step) - int(run_start_global_step))
    current = schedule[0]
    for item in sorted(schedule, key=lambda x: int(x.get("after_steps", 0))):
        if schedule_step >= int(item.get("after_steps", 0)):
            current = item
    return {
        "name": str(current.get("name", "")),
        "after_steps": int(current.get("after_steps", 0)),
        "phase_windows": list(current.get("phase_windows", [])),
        "phase_indices": list(current.get("phase_indices", [])),
    }

def episode_steps_for_step(config: dict[str, Any], global_step: int, run_start_global_step: int) -> int:
    steps = int(config.get("reset", {}).get("episode_steps", 320))
    schedule = config.get("episode_steps_schedule", [])
    if not isinstance(schedule, list) or not schedule:
        return steps
    schedule_step = int(global_step)
    if str(config.get("episode_steps_schedule_mode", "relative")) == "relative":
        schedule_step = max(0, int(global_step) - int(run_start_global_step))
    for item in sorted(schedule, key=lambda x: int(x.get("after_steps", 0))):
        if schedule_step >= int(item.get("after_steps", 0)):
            steps = int(item.get("episode_steps", steps))
    return max(1, int(steps))

def apply_episode_steps(runner: MJWarpMuscleRunner, config: dict[str, Any], episode_steps: int) -> None:
    steps = max(1, int(episode_steps))
    config.setdefault("reset", {})["episode_steps"] = steps
    runner.config.setdefault("reset", {})["episode_steps"] = steps
    runner.episode_steps = steps

def apply_reset_phase_stage(runner: MJWarpMuscleRunner, stage: dict[str, Any] | None) -> None:
    if stage is None:
        runner.phase_choices = runner.build_phase_choices(
            runner.config["reset"].get("phase_windows", []),
            runner.config["reset"].get("phase_indices", []),
            int(runner.config["reset"].get("phase_index_jitter", 0) or 0),
            int(runner.reference["length"]),
        )
        return
    runner.phase_choices = runner.build_phase_choices(
        stage.get("phase_windows", []),
        stage.get("phase_indices", []),
        int(runner.config["reset"].get("phase_index_jitter", 0) or 0),
        int(runner.reference["length"]),
    )

def future_obs_dropout_prob_for_step(config: dict[str, Any], global_step: int, run_start_global_step: int) -> float:
    imitation = config.get("imitation", {})
    prob = float(imitation.get("future_obs_dropout_prob", 0.0) or 0.0)
    schedule = imitation.get("future_obs_dropout_schedule", [])
    if isinstance(schedule, list):
        schedule_step = int(global_step)
        if str(config.get("reward_schedule_mode", "relative")) == "relative":
            schedule_step = max(0, int(global_step) - int(run_start_global_step))
        for item in sorted(schedule, key=lambda x: int(x.get("after_steps", 0))):
            if schedule_step >= int(item.get("after_steps", 0)):
                prob = float(item.get("prob", prob))
    return max(0.0, min(1.0, prob))

def set_future_obs_dropout_prob(config: dict[str, Any], prob: float) -> None:
    config.setdefault("imitation", {})["current_future_obs_dropout_prob"] = max(0.0, min(1.0, float(prob)))

def update_sac_expert_once(
    *,
    replay: ReplayBuffer,
    batch_size: int,
    actor: nn.Module,
    qf1: nn.Module,
    qf2: nn.Module,
    qf1_target: nn.Module,
    qf2_target: nn.Module,
    actor_optimizer: optim.Optimizer,
    q_optimizer: optim.Optimizer,
    alpha_optimizer: optim.Optimizer,
    log_alpha: torch.Tensor,
    obs_normalizer: ObsNormalizer,
    gated_spec: dict[str, torch.Tensor | dict[str, Any]] | None,
    current_ref_gate: float,
    gamma: float,
    tau: float,
    target_entropy: float,
    max_grad_norm: float,
    human_anchor_actor: nn.Module | None = None,
    human_anchor_weight: float = 0.0,
    muscle_count: int = 0,
    actor_updates_enabled: bool = True,
) -> dict[str, float]:
    b_obs_raw, b_action, b_reward, b_next_obs_raw, b_done = replay.sample(batch_size)
    b_obs = obs_normalizer.normalize(b_obs_raw)
    b_next_obs = obs_normalizer.normalize(b_next_obs_raw)
    q_b_obs = mask_ref_obs_for_q(
        b_obs,
        gated_spec["ref_indices"] if gated_spec is not None else None,  # type: ignore[index]
        current_ref_gate,
    )
    q_b_next_obs = mask_ref_obs_for_q(
        b_next_obs,
        gated_spec["ref_indices"] if gated_spec is not None else None,  # type: ignore[index]
        current_ref_gate,
    )
    with torch.no_grad():
        next_action, next_logprob = actor.get_action(b_next_obs)  # type: ignore[attr-defined]
        target_q = torch.min(qf1_target(q_b_next_obs, next_action), qf2_target(q_b_next_obs, next_action))
        alpha = log_alpha.exp()
        next_q_value = b_reward + (1.0 - b_done) * gamma * (target_q - alpha * next_logprob)
    q1 = qf1(q_b_obs, b_action)
    q2 = qf2(q_b_obs, b_action)
    q_loss = F.mse_loss(q1, next_q_value) + F.mse_loss(q2, next_q_value)
    q_optimizer.zero_grad()
    q_loss.backward()
    nn.utils.clip_grad_norm_(list(qf1.parameters()) + list(qf2.parameters()), float(max_grad_norm))
    q_optimizer.step()

    pi, pi_logprob = actor.get_action(b_obs)  # type: ignore[attr-defined]
    min_q_pi = torch.min(qf1(q_b_obs, pi), qf2(q_b_obs, pi))
    alpha = log_alpha.exp().detach()
    sac_actor_loss = (alpha * pi_logprob - min_q_pi).mean()
    human_anchor_loss = torch.zeros((), dtype=torch.float32, device=b_obs.device)
    if human_anchor_actor is not None and human_anchor_weight > 0.0 and muscle_count > 0:
        student_action, _ = actor.get_action(b_obs, deterministic=True)  # type: ignore[attr-defined]
        with torch.no_grad():
            anchor_action, _ = human_anchor_actor.get_action(b_obs, deterministic=True)  # type: ignore[attr-defined]
        human_anchor_loss = F.mse_loss(student_action[:, :muscle_count], anchor_action[:, :muscle_count])
    actor_loss = sac_actor_loss + float(human_anchor_weight) * human_anchor_loss
    if actor_updates_enabled:
        actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), float(max_grad_norm))
        actor_optimizer.step()

    alpha_loss = -(log_alpha * (pi_logprob + target_entropy).detach()).mean()
    if actor_updates_enabled:
        alpha_optimizer.zero_grad()
        alpha_loss.backward()
        alpha_optimizer.step()

    soft_update(qf1, qf1_target, tau)
    soft_update(qf2, qf2_target, tau)
    return {
        "q_loss": float(q_loss.detach().item()),
        "actor_loss": float(actor_loss.detach().item()),
        "sac_actor_loss": float(sac_actor_loss.detach().item()),
        "human_anchor_loss": float(human_anchor_loss.detach().item()),
        "alpha_loss": float(alpha_loss.detach().item()),
        "alpha": float(log_alpha.exp().detach().item()),
        "sample_logprob": float(pi_logprob.detach().mean().item()),
        "q_batch_q1_mean": float(q1.detach().mean().item()),
        "q_batch_q2_mean": float(q2.detach().mean().item()),
    }

def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for src_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.mul_(1.0 - tau)
            target_param.data.add_(tau * src_param.data)
