"""Checkpoint compatibility, loading, and persistence."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco
import torch
import torch.nn as nn
import torch.optim as optim

from myo_exo_train.env.observation import ObsNormalizer
from myo_exo_train.rl.networks import (
    GatedRefSACActor,
    SymmetricSACActor,
    build_sagittal_mirror_spec,
    gated_ref_base_actor,
    gated_ref_obs_spec,
    policy_architecture,
)

def save_checkpoint(
    path: Path,
    *,
    actor: nn.Module,
    qf1: nn.Module,
    qf2: nn.Module,
    qf1_target: nn.Module,
    qf2_target: nn.Module,
    actor_optimizer: optim.Optimizer,
    q_optimizer: optim.Optimizer,
    alpha_optimizer: optim.Optimizer,
    log_alpha: torch.Tensor,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor_state_dict": actor.state_dict(),
            "qf1_state_dict": qf1.state_dict(),
            "qf2_state_dict": qf2.state_dict(),
            "qf1_target_state_dict": qf1_target.state_dict(),
            "qf2_target_state_dict": qf2_target.state_dict(),
            "actor_optimizer_state_dict": actor_optimizer.state_dict(),
            "q_optimizer_state_dict": q_optimizer.state_dict(),
            "alpha_optimizer_state_dict": alpha_optimizer.state_dict(),
            "log_alpha": log_alpha.detach().clone(),
            **payload,
        },
        path,
    )

def save_expert_checkpoint_set(
    *,
    outdir: Path,
    global_step: int,
    payload: dict[str, Any],
    primary: dict[str, Any],
    hard_switch: Any,
) -> Path:
    """Persist numbered/latest checkpoints for the primary and optional U expert."""
    base_path = outdir / f"agent_step_{global_step}.pt"

    def save(path: Path, modules: dict[str, Any], extra: dict[str, Any]) -> None:
        save_checkpoint(path, payload={**payload, **extra}, **modules)

    if not hard_switch.train_both:
        save(base_path, primary, {})
        save(outdir / "latest.pt", primary, {})
    save(outdir / f"agent_step_{global_step}_S.pt", primary, {"expert_name": "S"})
    save(outdir / "latest_S.pt", primary, {"expert_name": "S"})

    if hard_switch.train_both:
        uphill_names = (
            "uphill_actor",
            "uphill_qf1",
            "uphill_qf2",
            "uphill_qf1_target",
            "uphill_qf2_target",
            "uphill_actor_optimizer",
            "uphill_q_optimizer",
            "uphill_alpha_optimizer",
            "uphill_log_alpha",
            "uphill_normalizer",
        )
        missing = [name for name in uphill_names if getattr(hard_switch, name) is None]
        if missing:
            raise RuntimeError(f"trainable U expert is incomplete: {missing}")
        uphill = {
            "actor": hard_switch.uphill_actor,
            "qf1": hard_switch.uphill_qf1,
            "qf2": hard_switch.uphill_qf2,
            "qf1_target": hard_switch.uphill_qf1_target,
            "qf2_target": hard_switch.uphill_qf2_target,
            "actor_optimizer": hard_switch.uphill_actor_optimizer,
            "q_optimizer": hard_switch.uphill_q_optimizer,
            "alpha_optimizer": hard_switch.uphill_alpha_optimizer,
            "log_alpha": hard_switch.uphill_log_alpha,
        }
        extra = {
            "expert_name": "U",
            "obs_normalizer": hard_switch.uphill_normalizer.state_dict(),
        }
        save(outdir / f"agent_step_{global_step}_U.pt", uphill, extra)
        save(outdir / "latest_U.pt", uphill, extra)
    return base_path

def build_sac_actor_for_checkpoint(
    *,
    checkpoint: dict[str, Any],
    model: mujoco.MjModel,
    config: dict[str, Any],
    obs_dim: int,
    act_dim: int,
    device: torch.device,
) -> tuple[nn.Module, ObsNormalizer, dict[str, Any]]:
    run_cfg = checkpoint.get("run_config", {})
    sac_cfg = config.get("sac", config.get("ppo", {}))
    policy_cfg = config.get("policy", {})
    architecture = str(run_cfg.get("policy_architecture", policy_architecture(config)))
    if architecture != "gated_ref_sac":
        raise ValueError(f"unsupported checkpoint architecture: {architecture}")
    normalizer_spec: dict[str, Any] | None = None
    gated_spec = gated_ref_obs_spec(model, config, obs_dim=obs_dim, device=device)
    base_actor = GatedRefSACActor(
        obs_dim,
        act_dim,
        base_indices=gated_spec["base_indices"],
        ref_indices=gated_spec["ref_indices"],
        logstd_init=float(sac_cfg.get("actor_logstd_init", -0.5)),
        initial_action_mean=float(sac_cfg.get("initial_actor_action_mean", -0.2)),
        hidden_dim=int(policy_cfg.get("hidden_dim", 256)),
        latent_dim=int(policy_cfg.get("latent_dim", 128)),
        initial_ref_gate=float(policy_cfg.get("current_ref_gate", policy_cfg.get("ref_gate", 1.0))),
        muscle_count=int(model.na),
        exo_head_config=policy_cfg.get("exo_head", {}),
    ).to(device)
    normalizer_spec = gated_spec["metadata"]
    if bool(run_cfg.get("symmetric_policy", sac_cfg.get("symmetric_policy", False))):
        mirror_spec = build_sagittal_mirror_spec(
            model,
            config,
            obs_dim=obs_dim,
            future_steps=int(config.get("imitation", {}).get("reference_future_steps", 0)),
            device=device,
        )
        actor: nn.Module = SymmetricSACActor(
            base_actor,
            obs_perm=mirror_spec["obs_perm"],
            obs_sign=mirror_spec["obs_sign"],
            act_perm=mirror_spec["act_perm"],
            act_sign=mirror_spec["act_sign"],
        ).to(device)
    else:
        actor = base_actor
    exact_actor = load_shape_compatible_state_dict(actor, checkpoint["actor_state_dict"])
    actor.eval()
    normalizer = ObsNormalizer(
        obs_dim,
        device,
        enabled=bool(sac_cfg.get("normalize_observations", True)),
        clip=float(sac_cfg.get("obs_norm_clip", 10.0)),
    )
    normalizer_resume = {"mode": "missing"}
    if "obs_normalizer" in checkpoint:
        normalizer_resume = load_resume_obs_normalizer(
            normalizer,
            checkpoint["obs_normalizer"],
            old_run_config=run_cfg,
            new_gated_spec=normalizer_spec,
            config=config,
        )
    return actor, normalizer, {
        "architecture": architecture,
        "exact_actor": bool(exact_actor),
        "obs_normalizer_resume": normalizer_resume,
    }

def load_shape_compatible_state_dict(module: nn.Module, state_dict: dict[str, torch.Tensor]) -> bool:
    filtered = {
        key: value for key, value in state_dict.items()
        if not (key.endswith("act_matrix") and int(value.numel()) == 0)
    }
    result = module.load_state_dict(filtered, strict=False)
    return not result.missing_keys and not result.unexpected_keys

def _range_from_obs_spec(spec: dict[str, Any] | None, name: str) -> tuple[int, int] | None:
    if not isinstance(spec, dict):
        return None
    value = spec.get(name)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    return int(value[0]), int(value[1])

def load_resume_obs_normalizer(
    obs_normalizer: ObsNormalizer,
    state: dict[str, Any],
    *,
    old_run_config: dict[str, Any],
    new_gated_spec: dict[str, Any] | None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del old_run_config
    old_dim = int(state["mean"].numel())
    new_dim = int(obs_normalizer.mean.numel())
    if old_dim != new_dim:
        raise ValueError(f"observation normalizer dimension mismatch: checkpoint={old_dim}, current={new_dim}")
    obs_normalizer.load_state_dict(state)
    reset_groups = [str(name) for name in (config or {}).get("resume_reset_obs_normalizer_groups", [])]
    reset_indices = sorted({
        int(index) for index in (config or {}).get("resume_reset_obs_normalizer_indices", [])
        if 0 <= int(index) < new_dim
    })
    reset: list[str] = []
    for name in reset_groups:
        bounds = _range_from_obs_spec(new_gated_spec, name)
        if bounds is None:
            continue
        start, end = bounds
        obs_normalizer.mean[start:end].zero_()
        obs_normalizer.var[start:end].fill_(1.0)
        reset.append(name)
    if reset_indices:
        indices = torch.tensor(reset_indices, dtype=torch.long, device=obs_normalizer.mean.device)
        obs_normalizer.mean.index_fill_(0, indices, 0.0)
        obs_normalizer.var.index_fill_(0, indices, 1.0)
    mode = "exact_reset" if reset or reset_indices else "exact"
    return {"mode": mode, "old_dim": old_dim, "new_dim": new_dim, "reset": reset, "reset_indices": reset_indices}

def load_shape_compatible_q_state_dict(
    module: nn.Module,
    state_dict: dict[str, torch.Tensor],
    *,
    old_obs_dim: int,
    new_obs_dim: int,
    act_dim: int,
) -> bool:
    if int(old_obs_dim) != int(new_obs_dim):
        raise ValueError(f"Q observation dimension mismatch: checkpoint={old_obs_dim}, current={new_obs_dim}")
    del act_dim
    filtered = {
        key: value for key, value in state_dict.items()
        if not (key.endswith("act_matrix") and int(value.numel()) == 0)
    }
    result = module.load_state_dict(filtered, strict=False)
    return not result.missing_keys and not result.unexpected_keys
