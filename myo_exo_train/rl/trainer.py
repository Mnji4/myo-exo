"""Training orchestration for 80-muscle MJWarp SAC policies."""
from __future__ import annotations

import argparse
import copy
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE_PATH = ROOT / "reference_exports/80/course/course80_3d_balanced_v8.npz"

from myo_exo_train.checkpoint import (
    load_resume_obs_normalizer,
    load_shape_compatible_q_state_dict,
    load_shape_compatible_state_dict,
    save_expert_checkpoint_set,
)
from myo_exo_train.env.model import build_muscle_model, muscle_action_mapping_mode
from myo_exo_train.env.observation import ObsNormalizer, build_policy_obs_tensor
from myo_exo_train.env.reference import (
    load_reference_from_config,
    reference_curriculum_for_update,
)
from myo_exo_train.env.reward import apply_reward_schedule
from myo_exo_train.env.runner import MJWarpMuscleRunner
from myo_exo_train.evaluation import (
    append_csv,
    configured_video_phases,
    evaluate,
    load_config,
    maybe_launch_checkpoint_video_export,
    render_policy_video,
    write_json,
)
from myo_exo_train.rl.hard_switch import build_hard_switch_state
from myo_exo_train.rl.networks import (
    GatedRefSACActor,
    SoftQNetwork,
    SymmetricSACActor,
    SymmetricSoftQNetwork,
    actor_optimizer_groups,
    build_sagittal_mirror_spec,
    gated_ref_base_actor,
    gated_ref_obs_spec,
    mask_ref_obs_for_q,
    policy_architecture,
    ref_gate_for_step,
    set_actor_ref_gate,
    symmetric_module_self_test,
)
from myo_exo_train.rl.replay_buffer import ReplayBuffer
from myo_exo_train.rl.sac import (
    action_anchor_weight_for_step,
    apply_episode_steps,
    apply_reset_phase_stage,
    episode_steps_for_step,
    future_obs_dropout_prob_for_step,
    reset_phase_schedule_for_step,
    set_future_obs_dropout_prob,
    soft_update,
    update_sac_expert_once,
    x_windows_mask,
)

def _validate_supported_training_path(config: dict[str, Any], args: argparse.Namespace) -> None:
    """Reject legacy branches intentionally excluded from this package."""
    if bool(config.get("amp", {}).get("enabled", False)):
        raise ValueError("AMP training is not supported")
    policy_architecture(config)
    if bool(config.get("sar_action", {}).get("enabled", False)):
        raise ValueError("SAR actions are not supported")
    if bool(config.get("exo_assistance_adaptation", {}).get("enabled", False)):
        raise ValueError("legacy Exo assistance adaptation is not supported")
    if bool(config.get("counterfactual_human_energy", {}).get("enabled", False)):
        raise ValueError("counterfactual shadow training is not supported")
    if bool(config.get("separate_exo_agent", {}).get("enabled", False)):
        raise ValueError("separate Exo SAC is not supported")
    if bool(config.get("policy", {}).get("human_adapter", {}).get("enabled", False)):
        raise ValueError("human assistance adapters are not supported")
    teacher_bc = config.get("sac", config.get("ppo", {})).get("teacher_bc", {})
    if isinstance(teacher_bc, dict) and bool(teacher_bc.get("enabled", False)):
        raise ValueError("SB3 teacher BC is not part of myo_exo_train")
    sac_cfg = config.get("sac", {})
    if bool(sac_cfg.get("dynamic_env_schedule", {}).get("enabled", False)):
        raise ValueError("dynamic environment counts are not supported")
    if bool(sac_cfg.get("dynamic_episode_schedule", {}).get("enabled", False)):
        raise ValueError("dynamic episode horizons are not supported")
    if config.get("reference_pool_schedule"):
        raise ValueError("reference pool schedules are not supported; use one precomposed course reference")
    obs_cfg = config.get("observation", {})
    if str(obs_cfg.get("mode", "default")).lower() != "default":
        raise ValueError("only the canonical observation mode is supported")
    if int(obs_cfg.get("frame_stack_prev_steps", 0) or 0) != 0:
        raise ValueError("observation frame stacking is not supported")
    if bool(config.get("post_reference", {}).get("enabled", False)):
        raise ValueError("post-reference rollout mode is not supported")
    if not bool(config.get("terrain_course", {}).get("enabled", False)):
        raise ValueError("myo_exo_train requires one explicit terrain_course")

def run_training(args: argparse.Namespace) -> None:

    if args.device != "cuda":
        raise SystemExit("MJWarp SAC is intended for --device cuda")
    device = torch.device(args.device)
    config = load_config(args.config)
    _validate_supported_training_path(config, args)
    if args.episode_steps is not None:
        config["reset"]["episode_steps"] = int(args.episode_steps)
    if args.qpos_noise is not None:
        config["reset"]["qpos_noise"] = float(args.qpos_noise)
    if args.qvel_noise is not None:
        config["reset"]["qvel_noise"] = float(args.qvel_noise)

    sac_cfg = config.get("sac", config.get("ppo", {}))
    seed = int(args.seed if args.seed is not None else config["seed"])
    args.seed = seed
    total_timesteps = int(args.total_timesteps if args.total_timesteps is not None else sac_cfg["total_timesteps"])
    nworld = int(args.nworld if args.nworld is not None else sac_cfg["num_envs"])
    batch_size = int(sac_cfg.get("batch_size", 1024))
    learning_starts = int(sac_cfg.get("learning_starts", 1024))
    train_freq = int(sac_cfg.get("train_freq", 1))
    gradient_steps = int(sac_cfg.get("gradient_steps", 2))
    gamma = float(sac_cfg.get("gamma", 0.99))
    tau = float(sac_cfg.get("tau", 0.005))
    target_entropy = float(sac_cfg.get("target_entropy", -float("nan")))
    symmetric_policy = bool(sac_cfg.get("symmetric_policy", False))
    architecture = policy_architecture(config)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if args.outdir is None:
        args.outdir = ROOT / "results" / f"mjwarp_muscle_sac_{time.strftime('%Y%m%d-%H%M%S')}"
    args.outdir.mkdir(parents=True, exist_ok=True)

    model, data = build_muscle_model(config)
    reference = load_reference_from_config(args.reference, model, float(config["control"]["control_hz"]), device, config)
    runner = MJWarpMuscleRunner(
        model=model,
        data=data,
        config=config,
        reference=reference,
        nworld=nworld,
        nconmax=int(args.nconmax),
        njmax=int(args.njmax),
        seed=seed,
        device=device,
    )

    obs_normalizer = ObsNormalizer(
        runner.obs_dim,
        device,
        enabled=bool(sac_cfg.get("normalize_observations", True)),
        clip=float(sac_cfg.get("obs_norm_clip", 10.0)),
    )
    freeze_obs_normalizer_updates = bool(sac_cfg.get("freeze_obs_normalizer_updates", False))
    mirror_spec: dict[str, torch.Tensor | dict[str, Any]] | None = None
    if symmetric_policy:
        mirror_spec = build_sagittal_mirror_spec(
            model,
            config,
            obs_dim=runner.obs_dim,
            future_steps=int(config.get("imitation", {}).get("reference_future_steps", 0)),
            device=device,
        )
    gated_spec: dict[str, torch.Tensor | dict[str, Any]] | None = None
    current_ref_gate = ref_gate_for_step(config, 0, 0)
    policy_cfg = config.get("policy", {})
    action_anchor_cfg = sac_cfg.get("action_anchor", {})
    if not isinstance(action_anchor_cfg, dict):
        action_anchor_cfg = {}
    human_anchor_weight = max(
        0.0,
        float(
            action_anchor_cfg.get(
                "initial_weight",
                action_anchor_cfg.get("weight", config.get("exo_policy", {}).get("human_anchor_weight", 0.0)),
            )
        ),
    )
    critic_warmup_steps = max(0, int(sac_cfg.get("critic_warmup_steps", 0)))
    actor_update_interval = max(1, int(sac_cfg.get("actor_update_interval", 1)))
    resume_reset_critic = bool(sac_cfg.get("resume_reset_critic", False))
    resume_reset_optimizers = bool(sac_cfg.get("resume_reset_optimizers", resume_reset_critic))
    resume_reset_alpha = bool(sac_cfg.get("resume_reset_alpha", resume_reset_critic))
    gated_spec = gated_ref_obs_spec(model, config, obs_dim=runner.obs_dim, device=device)
    base_actor = GatedRefSACActor(
        runner.obs_dim,
        runner.act_dim,
        base_indices=gated_spec["base_indices"],
        ref_indices=gated_spec["ref_indices"],
        logstd_init=float(sac_cfg.get("actor_logstd_init", -0.5)),
        initial_action_mean=float(sac_cfg.get("initial_actor_action_mean", -0.2)),
        hidden_dim=int(policy_cfg.get("hidden_dim", 256)),
        latent_dim=int(policy_cfg.get("latent_dim", 128)),
        initial_ref_gate=current_ref_gate,
        muscle_count=int(model.na),
        exo_head_config=policy_cfg.get("exo_head", {}),
    ).to(device)
    if mirror_spec is None:
        actor: nn.Module = base_actor
        qf1: nn.Module = SoftQNetwork(runner.obs_dim, runner.act_dim).to(device)
        qf2: nn.Module = SoftQNetwork(runner.obs_dim, runner.act_dim).to(device)
        qf1_target: nn.Module = SoftQNetwork(runner.obs_dim, runner.act_dim).to(device)
        qf2_target: nn.Module = SoftQNetwork(runner.obs_dim, runner.act_dim).to(device)
    else:
        actor = SymmetricSACActor(
            base_actor,
            obs_perm=mirror_spec["obs_perm"],
            obs_sign=mirror_spec["obs_sign"],
            act_perm=mirror_spec["act_perm"],
            act_sign=mirror_spec["act_sign"],
        ).to(device)
        qf1 = SymmetricSoftQNetwork(
            SoftQNetwork(runner.obs_dim, runner.act_dim).to(device),
            obs_perm=mirror_spec["obs_perm"],
            obs_sign=mirror_spec["obs_sign"],
            act_perm=mirror_spec["act_perm"],
            act_sign=mirror_spec["act_sign"],
        ).to(device)
        qf2 = SymmetricSoftQNetwork(
            SoftQNetwork(runner.obs_dim, runner.act_dim).to(device),
            obs_perm=mirror_spec["obs_perm"],
            obs_sign=mirror_spec["obs_sign"],
            act_perm=mirror_spec["act_perm"],
            act_sign=mirror_spec["act_sign"],
        ).to(device)
        qf1_target = SymmetricSoftQNetwork(
            SoftQNetwork(runner.obs_dim, runner.act_dim).to(device),
            obs_perm=mirror_spec["obs_perm"],
            obs_sign=mirror_spec["obs_sign"],
            act_perm=mirror_spec["act_perm"],
            act_sign=mirror_spec["act_sign"],
        ).to(device)
        qf2_target = SymmetricSoftQNetwork(
            SoftQNetwork(runner.obs_dim, runner.act_dim).to(device),
            obs_perm=mirror_spec["obs_perm"],
            obs_sign=mirror_spec["obs_sign"],
            act_perm=mirror_spec["act_perm"],
            act_sign=mirror_spec["act_sign"],
        ).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    symmetry_test: dict[str, float] | None = None
    if isinstance(actor, SymmetricSACActor) and isinstance(qf1, SymmetricSoftQNetwork):
        symmetry_test = symmetric_module_self_test(
            actor,
            qf1,
            obs_dim=runner.obs_dim,
            act_dim=runner.act_dim,
            device=device,
        )
        print(json.dumps({"symmetry_test": symmetry_test}, ensure_ascii=False), flush=True)
    policy_lr = float(sac_cfg.get("policy_lr", sac_cfg.get("learning_rate", 3e-4)))
    exo_lr_value = policy_cfg.get("exo_head", {}).get("learning_rate", None)
    actor_optimizer = optim.Adam(
        actor_optimizer_groups(
            actor,
            policy_lr=policy_lr,
            exo_lr=None if exo_lr_value is None else float(exo_lr_value),
        ),
        eps=1e-5,
    )
    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()),
        lr=float(sac_cfg.get("q_lr", sac_cfg.get("learning_rate", 3e-4))),
        eps=1e-5,
    )
    log_alpha = torch.tensor(np.log(float(sac_cfg.get("alpha", 0.2))), dtype=torch.float32, device=device, requires_grad=True)
    alpha_optimizer = optim.Adam([log_alpha], lr=float(sac_cfg.get("alpha_lr", sac_cfg.get("learning_rate", 3e-4))), eps=1e-5)
    if not np.isfinite(target_entropy):
        target_entropy = -float(runner.act_dim)

    replay = ReplayBuffer(
        int(sac_cfg.get("buffer_size", 250000)),
        runner.obs_dim,
        runner.act_dim,
        device,
    )

    global_step = 0
    start_env_step = 1
    resumed_from: str | None = None
    partial_resume = False
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        old_obs_dim = int(checkpoint.get("run_config", {}).get("obs_dim", 0) or 0)
        if old_obs_dim <= 0:
            old_obs_dim = actor_obs_dim_from_state_dict(checkpoint["actor_state_dict"])
        new_obs_dim = int(runner.obs_dim)
        exact_actor = load_shape_compatible_state_dict(actor, checkpoint["actor_state_dict"])
        actor_base_for_resume = gated_ref_base_actor(actor)
        if (
            actor_base_for_resume is not None
            and actor_base_for_resume.exo_head_enabled
            and not any("exo_policy_head" in key for key in checkpoint["actor_state_dict"])
        ):
            exact_actor = False
        if resume_reset_critic:
            exact_qf1 = exact_qf2 = exact_qf1_target = exact_qf2_target = True
            print(json.dumps({"resume_reset_critic": True, "critic_warmup_steps": critic_warmup_steps}), flush=True)
        else:
            exact_qf1 = load_shape_compatible_q_state_dict(
                qf1,
                checkpoint["qf1_state_dict"],
                old_obs_dim=old_obs_dim,
                new_obs_dim=new_obs_dim,
                act_dim=runner.act_dim,
            )
            exact_qf2 = load_shape_compatible_q_state_dict(
                qf2,
                checkpoint["qf2_state_dict"],
                old_obs_dim=old_obs_dim,
                new_obs_dim=new_obs_dim,
                act_dim=runner.act_dim,
            )
            exact_qf1_target = load_shape_compatible_q_state_dict(
                qf1_target,
                checkpoint["qf1_target_state_dict"],
                old_obs_dim=old_obs_dim,
                new_obs_dim=new_obs_dim,
                act_dim=runner.act_dim,
            )
            exact_qf2_target = load_shape_compatible_q_state_dict(
                qf2_target,
                checkpoint["qf2_target_state_dict"],
                old_obs_dim=old_obs_dim,
                new_obs_dim=new_obs_dim,
                act_dim=runner.act_dim,
            )
        partial_resume = not all([exact_actor, exact_qf1, exact_qf2, exact_qf1_target, exact_qf2_target])
        if not partial_resume:
            if not resume_reset_optimizers:
                actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
                q_optimizer.load_state_dict(checkpoint["q_optimizer_state_dict"])
                alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer_state_dict"])
            for group in actor_optimizer.param_groups:
                if group.get("group_name") == "exo_head" and exo_lr_value is not None:
                    group["lr"] = float(exo_lr_value)
                else:
                    group["lr"] = policy_lr
            for group in q_optimizer.param_groups:
                group["lr"] = float(sac_cfg.get("q_lr", sac_cfg.get("learning_rate", 3e-4)))
            for group in alpha_optimizer.param_groups:
                group["lr"] = float(sac_cfg.get("alpha_lr", sac_cfg.get("learning_rate", 3e-4)))
        if not resume_reset_alpha:
            with torch.no_grad():
                log_alpha.copy_(checkpoint["log_alpha"].to(device))
        if "obs_normalizer" in checkpoint:
            normalizer_resume = load_resume_obs_normalizer(
                obs_normalizer,
                checkpoint["obs_normalizer"],
                old_run_config=checkpoint.get("run_config", {}),
                new_gated_spec=gated_spec["metadata"] if gated_spec is not None else None,
                config=config,
            )
            print(json.dumps({"obs_normalizer_resume": normalizer_resume}), flush=True)
        global_step = int(checkpoint.get("global_step", 0))
        start_env_step = int(checkpoint.get("env_step", 0)) + 1
        resumed_from = str(args.resume)
    finetune_start_global_step = int(global_step)
    run_start_global_step = int(global_step)
    if args.resume is not None:
        if bool(config.get("resume_schedule_from_checkpoint", True)):
            run_start_global_step = int(checkpoint.get("run_start_global_step", checkpoint.get("run_config", {}).get("run_start_global_step", global_step)))
    human_anchor_actor: nn.Module | None = None
    if human_anchor_weight > 0.0:
        human_anchor_actor = copy.deepcopy(actor).eval()
        for param in human_anchor_actor.parameters():
            param.requires_grad_(False)
        print(
            json.dumps(
                {
                    "human_action_anchor": {
                        "enabled": True,
                        "weight": float(human_anchor_weight),
                        "action_dims": int(model.na),
                    }
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    current_ref_gate = ref_gate_for_step(config, global_step, run_start_global_step)
    config.setdefault("policy", {})["current_ref_gate"] = float(current_ref_gate)
    set_actor_ref_gate(actor, current_ref_gate)
    apply_reward_schedule(config, runner, global_step, run_start_global_step)

    hard_switch = build_hard_switch_state(
        config=config,
        policy_cfg=policy_cfg,
        sac_cfg=sac_cfg,
        runner=runner,
        model=model,
        actor=actor,
        mirror_spec=mirror_spec,
        device=device,
        nworld=nworld,
        human_anchor_weight=human_anchor_weight,
        reset_critic=resume_reset_critic,
        reset_optimizers=resume_reset_optimizers,
        reset_alpha=resume_reset_alpha,
    )
    hard_switch_enabled = hard_switch.enabled
    hard_switch_train_both = hard_switch.train_both
    hard_switch_train_stair = hard_switch.train_stair
    hard_switch_train_uphill = hard_switch.train_uphill
    hard_switch_use_policy_warmup = hard_switch.use_policy_warmup
    hard_switch_uphill_rollout_deterministic = hard_switch.uphill_rollout_deterministic
    hard_switch_route_replay_only = hard_switch.route_replay_only
    hard_switch_replay_overlap_enabled = hard_switch.replay_overlap_enabled
    hard_switch_stair_train_windows = hard_switch.stair_train_windows
    hard_switch_uphill_train_windows = hard_switch.uphill_train_windows
    hard_switch_handoff_value_enabled = hard_switch.handoff_value_enabled
    hard_switch_u_to_s_value_weight = hard_switch.u_to_s_value_weight
    hard_switch_s_to_u_value_weight = hard_switch.s_to_u_value_weight
    hard_switch_handoff_value_scale = hard_switch.handoff_value_scale
    hard_switch_boundary_done = hard_switch.boundary_done
    hard_switch_uphill_actor = hard_switch.uphill_actor
    hard_switch_uphill_human_anchor_actor = hard_switch.uphill_human_anchor_actor
    hard_switch_uphill_normalizer = hard_switch.uphill_normalizer
    hard_switch_uphill_qf1 = hard_switch.uphill_qf1
    hard_switch_uphill_qf2 = hard_switch.uphill_qf2
    hard_switch_uphill_qf1_target = hard_switch.uphill_qf1_target
    hard_switch_uphill_qf2_target = hard_switch.uphill_qf2_target
    hard_switch_uphill_actor_optimizer = hard_switch.uphill_actor_optimizer
    hard_switch_uphill_q_optimizer = hard_switch.uphill_q_optimizer
    hard_switch_uphill_alpha_optimizer = hard_switch.uphill_alpha_optimizer
    hard_switch_uphill_log_alpha = hard_switch.uphill_log_alpha
    hard_switch_uphill_replay = hard_switch.uphill_replay
    hard_switch_stair_train_mask: torch.Tensor | None = None
    hard_switch_uphill_train_mask: torch.Tensor | None = None
    hard_switch_meta = hard_switch.metadata
    if hard_switch_enabled:
        print(json.dumps({"hard_switch_experts": hard_switch_meta}, ensure_ascii=False), flush=True)

    run_config = {
        "config": config,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "reference": {
            "path": reference["path"],
            "length": reference["length"],
            "metadata": reference["metadata"],
            "source_index_count": int(len(reference["source_indices"])),
            "reference_names": reference.get("reference_names", []),
            "reference_offsets": reference.get("reference_offsets", []),
        },
        "obs_dim": runner.obs_dim,
        "act_dim": runner.act_dim,
        "action_mapping": f"muscle={muscle_action_mapping_mode(config)}, non_muscle=linear_ctrlrange",
        "normalize_observations": obs_normalizer.enabled,
        "freeze_obs_normalizer_updates": bool(freeze_obs_normalizer_updates),
        "target_entropy": target_entropy,

        "human_action_anchor_weight": float(human_anchor_weight),
        "critic_warmup_steps": int(critic_warmup_steps),
        "resume_reset_critic": bool(resume_reset_critic),
        "symmetric_policy": symmetric_policy,
        "policy_architecture": architecture,
        "gated_ref_obs_spec": gated_spec["metadata"] if gated_spec is not None else None,
        "mirror_spec": mirror_spec["metadata"] if mirror_spec is not None else None,
        "symmetry_test": symmetry_test,
        "hard_switch_experts": hard_switch_meta,
        "resumed_from": resumed_from,
        "partial_resume": partial_resume,
        "run_start_global_step": run_start_global_step,
    }
    write_json(args.outdir / "run_config.json", run_config)

    current_reset_phase_stage = reset_phase_schedule_for_step(config, global_step, run_start_global_step)
    if current_reset_phase_stage is not None:
        apply_reset_phase_stage(runner, current_reset_phase_stage)
        runner.reset(torch.ones(nworld, dtype=torch.bool, device=device))
    current_episode_steps = episode_steps_for_step(config, global_step, run_start_global_step)
    apply_episode_steps(runner, config, current_episode_steps)

    current_future_obs_dropout_prob = future_obs_dropout_prob_for_step(config, global_step, run_start_global_step)
    set_future_obs_dropout_prob(config, current_future_obs_dropout_prob)
    if args.render_only_video:
        set_future_obs_dropout_prob(config, 0.0)
        original_video_phase = int(args.video_phase)
        video_rows = []
        for video_phase in configured_video_phases(
            config,
            reference,
            original_video_phase,
            global_step=global_step,
            run_start_global_step=run_start_global_step,
            video_every=int(args.video_every),
        ):
            args.video_phase = int(video_phase)
            video_row = render_policy_video(
                agent=actor,
                obs_normalizer=obs_normalizer,
                config=config,
                reference=reference,
                args=args,
                device=device,
                update=start_env_step,
                global_step=global_step,
            )
            append_csv(args.outdir / "video_metrics.csv", video_row)
            print(json.dumps({"video": video_row}, ensure_ascii=False), flush=True)
            video_rows.append(video_row)
        args.video_phase = original_video_phase
        print(json.dumps({"render_only_video_done": video_rows}, ensure_ascii=False), flush=True)
        return
    env_step = start_env_step

    next_obs_raw = runner.obs()
    next_done = torch.zeros(nworld, dtype=torch.float32, device=device)
    train_stats: dict[str, float] = {
        "q_loss": 0.0,
        "actor_loss": 0.0,
        "sac_actor_loss": 0.0,
        "human_anchor_loss": 0.0,
        "alpha_loss": 0.0,
        "alpha": float(log_alpha.exp().detach().item()),
        "sample_logprob": 0.0,
        "q_batch_q1_mean": 0.0,
        "q_batch_q2_mean": 0.0,
    }
    hard_switch_uphill_train_stats: dict[str, float] = {
        "q_loss": 0.0,
        "actor_loss": 0.0,
        "sac_actor_loss": 0.0,
        "human_anchor_loss": 0.0,
        "alpha_loss": 0.0,
        "alpha": 0.0,
        "sample_logprob": 0.0,
        "q_batch_q1_mean": 0.0,
        "q_batch_q2_mean": 0.0,
    }

    tracked_train_terms = sorted(set(runner.myoassist_dense_weights) | {
        "myoassist_dense",
        "normal_reward",
        "recovery_reward",
        "forward_speed_shortfall",
        "foot_contact_phase_mismatch",
        "foot_toe_in_penalty_r",
        "foot_toe_in_penalty_l",
        "foot_toe_in_angle_r",
        "foot_toe_in_angle_l",
        "knee_valgus_penalty_r",
        "knee_valgus_penalty_l",
        "knee_valgus_r",
        "knee_valgus_l",
        "foot_lateral_gap",
        "stair_contact_step_index",
        "stair_pelvis_step_index",
        "stair_forward_step_delta",
        "nearest_trajectory_best_error",
        "root_xy_drift",
        "root_xy_drift_done",
        "lateral_drift_abs",
        "lateral_drift_done",
        "exo_assistance_ctrl_r",
        "exo_assistance_ctrl_l",
        "exo_assistance_ctrl_mean",
        "human_energy_tracking_gate",
        "human_energy_activation_l2",
        "human_energy_hip_torque_abs",
        "human_energy_hip_cocontraction_nm",
        "human_energy_hip_opposition",
        "pelvis_height_above_terrain",
        "activation_mean",
        "activation_max",
        "myoassist_foot_force_r",
        "myoassist_foot_force_l",
    })
    log_reward_sum = torch.zeros((), dtype=torch.float32, device=device)
    log_sample_count = 0
    log_u_to_s_value_bonus_sum = torch.zeros((), dtype=torch.float32, device=device)
    log_s_to_u_value_bonus_sum = torch.zeros((), dtype=torch.float32, device=device)
    log_u_to_s_value_bonus_count = 0
    log_s_to_u_value_bonus_count = 0
    log_done_count = torch.zeros((), dtype=torch.float32, device=device)
    log_fall_count = torch.zeros((), dtype=torch.float32, device=device)
    log_done_return_sum = torch.zeros((), dtype=torch.float32, device=device)
    log_done_length_sum = torch.zeros((), dtype=torch.float32, device=device)
    log_term_step_sum = {key: torch.zeros((), dtype=torch.float32, device=device) for key in tracked_train_terms}
    log_term_done_sum = {key: torch.zeros((), dtype=torch.float32, device=device) for key in tracked_train_terms}
    episode_term_running = {
        key: torch.zeros(nworld, dtype=torch.float32, device=device) for key in tracked_train_terms
    }
    active_video_export: subprocess.Popen | None = None
    while global_step < total_timesteps:
        update_start = time.perf_counter()
        apply_reward_schedule(config, runner, global_step, run_start_global_step)
        current_human_anchor_weight = action_anchor_weight_for_step(
            config, global_step, finetune_start_global_step
        )
        actor_updates_enabled = (
            int(global_step) - int(finetune_start_global_step) >= int(critic_warmup_steps)
            and int(env_step) % int(actor_update_interval) == 0
        )
        next_episode_steps = episode_steps_for_step(config, global_step, run_start_global_step)
        if next_episode_steps != current_episode_steps:
            current_episode_steps = next_episode_steps
            apply_episode_steps(runner, config, current_episode_steps)
        next_reset_phase_stage = reset_phase_schedule_for_step(config, global_step, run_start_global_step)
        if next_reset_phase_stage != current_reset_phase_stage:
            current_reset_phase_stage = next_reset_phase_stage
            apply_reset_phase_stage(runner, current_reset_phase_stage)
            runner.reset(torch.ones(nworld, dtype=torch.bool, device=device))
        current_future_obs_dropout_prob = future_obs_dropout_prob_for_step(config, global_step, run_start_global_step)
        set_future_obs_dropout_prob(config, current_future_obs_dropout_prob)
        current_ref_gate = ref_gate_for_step(config, global_step, run_start_global_step)
        config.setdefault("policy", {})["current_ref_gate"] = float(current_ref_gate)
        set_actor_ref_gate(actor, current_ref_gate)
        reference_curriculum = reference_curriculum_for_update(config, max(1, env_step))
        config.setdefault("reference_curriculum", {})["current_phase_lead_steps"] = int(reference_curriculum["phase_lead_steps"])
        config.setdefault("reference_curriculum", {})["current_phase_tolerance_steps"] = int(reference_curriculum["phase_tolerance_steps"])
        config.setdefault("reference_curriculum", {})["current_swing_exaggeration_scale"] = float(
            reference_curriculum["swing_exaggeration_scale"]
        )
        runner.set_reference_curriculum(
            phase_lead_steps=int(reference_curriculum["phase_lead_steps"]),
            phase_tolerance_steps=int(reference_curriculum["phase_tolerance_steps"]),
            swing_exaggeration_scale=float(reference_curriculum["swing_exaggeration_scale"]),
        )
        hard_switch_stair_mask: torch.Tensor | None = None
        if hard_switch_enabled:
            pelvis_forward = runner.qpos[:, runner.pelvis_tx_qpos].detach()
            hard_switch_stair_mask = hard_switch.route_mask(pelvis_forward)
        if not freeze_obs_normalizer_updates:
            if hard_switch_enabled and hard_switch_stair_mask is not None:
                if bool(hard_switch_stair_mask.any().item()):
                    obs_normalizer.update(next_obs_raw[hard_switch_stair_mask])
            else:
                obs_normalizer.update(next_obs_raw)
        obs = obs_normalizer.normalize(next_obs_raw)
        if global_step < learning_starts and not (hard_switch_enabled and hard_switch_use_policy_warmup):
            action = torch.empty((nworld, runner.act_dim), dtype=torch.float32, device=device).uniform_(-1.0, 1.0)
            logprob = torch.zeros(nworld, dtype=torch.float32, device=device)
        else:
            with torch.no_grad():
                action, logprob = actor.get_action(obs, deterministic=False)
        if hard_switch_enabled:
            if hard_switch_uphill_actor is None or hard_switch_uphill_normalizer is None:
                raise RuntimeError("hard_switch_experts enabled but frozen uphill expert was not loaded")
            if hard_switch_stair_mask is None:
                pelvis_forward = runner.qpos[:, runner.pelvis_tx_qpos].detach()
                hard_switch_stair_mask = hard_switch.route_mask(pelvis_forward)
            with torch.no_grad():
                uphill_obs = hard_switch_uphill_normalizer.normalize(next_obs_raw)
                uphill_action, _uphill_logprob = hard_switch_uphill_actor.get_action(  # type: ignore[attr-defined]
                    uphill_obs,
                    deterministic=bool(hard_switch_uphill_rollout_deterministic),
                )
            route_mask = hard_switch_stair_mask.unsqueeze(1)
            action = torch.where(route_mask, action, uphill_action)
            logprob = torch.where(hard_switch_stair_mask, logprob, torch.zeros_like(logprob))
        with torch.no_grad():
            obs_before = next_obs_raw
            next_obs_raw, reward, done, terms = runner.step(action)

        if hard_switch_enabled:
            if hard_switch_stair_mask is None:
                raise RuntimeError("hard_switch_experts enabled but route mask is missing")
            next_pelvis_forward = runner.qpos[:, runner.pelvis_tx_qpos].detach()
            next_stair_mask = hard_switch.route_mask(next_pelvis_forward)
            if hard_switch_replay_overlap_enabled and not hard_switch_route_replay_only:
                hard_switch_stair_train_mask = x_windows_mask(pelvis_forward, hard_switch_stair_train_windows)
                next_stair_train_mask = x_windows_mask(next_pelvis_forward, hard_switch_stair_train_windows)
                hard_switch_uphill_train_mask = x_windows_mask(pelvis_forward, hard_switch_uphill_train_windows)
                next_uphill_train_mask = x_windows_mask(next_pelvis_forward, hard_switch_uphill_train_windows)
            else:
                hard_switch_stair_train_mask = hard_switch_stair_mask
                next_stair_train_mask = next_stair_mask
                hard_switch_uphill_train_mask = ~hard_switch_stair_mask
                next_uphill_train_mask = ~next_stair_mask
            stair_replay_reward = reward
            uphill_replay_reward = reward
            u_to_s_value_bonus = torch.zeros_like(reward)
            s_to_u_value_bonus = torch.zeros_like(reward)
            if hard_switch_handoff_value_enabled:
                u_to_s_boundary = (~hard_switch_stair_mask) & next_stair_mask
                s_to_u_boundary = hard_switch_stair_mask & (~next_stair_mask)
                with torch.no_grad():
                    if bool(u_to_s_boundary.any().item()) and hard_switch_u_to_s_value_weight != 0.0:
                        receiver_obs = obs_normalizer.normalize(next_obs_raw[u_to_s_boundary])
                        receiver_q_obs = mask_ref_obs_for_q(
                            receiver_obs,
                            gated_spec["ref_indices"] if gated_spec is not None else None,
                            current_ref_gate,
                        )
                        receiver_action, _ = actor.get_action(receiver_obs, deterministic=True)
                        receiver_value = torch.min(
                            qf1(receiver_q_obs, receiver_action),
                            qf2(receiver_q_obs, receiver_action),
                        )
                        u_to_s_value_bonus[u_to_s_boundary] = float(hard_switch_u_to_s_value_weight) * torch.tanh(
                            receiver_value / float(hard_switch_handoff_value_scale)
                        )
                        uphill_replay_reward = reward + u_to_s_value_bonus
                    if (
                        bool(s_to_u_boundary.any().item())
                        and hard_switch_s_to_u_value_weight != 0.0
                        and hard_switch_uphill_actor is not None
                        and hard_switch_uphill_normalizer is not None
                        and hard_switch_uphill_qf1 is not None
                        and hard_switch_uphill_qf2 is not None
                    ):
                        receiver_obs = hard_switch_uphill_normalizer.normalize(next_obs_raw[s_to_u_boundary])
                        receiver_q_obs = mask_ref_obs_for_q(
                            receiver_obs,
                            gated_spec["ref_indices"] if gated_spec is not None else None,
                            current_ref_gate,
                        )
                        receiver_action, _ = hard_switch_uphill_actor.get_action(receiver_obs, deterministic=True)
                        receiver_value = torch.min(
                            hard_switch_uphill_qf1(receiver_q_obs, receiver_action),
                            hard_switch_uphill_qf2(receiver_q_obs, receiver_action),
                        )
                        s_to_u_value_bonus[s_to_u_boundary] = float(hard_switch_s_to_u_value_weight) * torch.tanh(
                            receiver_value / float(hard_switch_handoff_value_scale)
                        )
                        stair_replay_reward = reward + s_to_u_value_bonus
            replay_mask = hard_switch_stair_train_mask
            if bool(replay_mask.any().item()):
                replay_done = done
                if hard_switch_boundary_done:
                    replay_done = done | (hard_switch_stair_train_mask != next_stair_train_mask)
                replay.add(
                    obs_before[replay_mask],
                    action[replay_mask],
                    stair_replay_reward[replay_mask],
                    next_obs_raw[replay_mask],
                    replay_done[replay_mask],
                )
            if hard_switch_train_both:
                if hard_switch_uphill_replay is None:
                    raise RuntimeError("hard_switch train_both enabled but U replay is missing")
                uphill_replay_mask = hard_switch_uphill_train_mask
                if bool(uphill_replay_mask.any().item()):
                    uphill_done = done
                    if hard_switch_boundary_done:
                        uphill_done = done | (hard_switch_uphill_train_mask != next_uphill_train_mask)
                    hard_switch_uphill_replay.add(
                        obs_before[uphill_replay_mask],
                        action[uphill_replay_mask],
                        uphill_replay_reward[uphill_replay_mask],
                        next_obs_raw[uphill_replay_mask],
                        uphill_done[uphill_replay_mask],
                    )
        else:
            u_to_s_value_bonus = torch.zeros_like(reward)
            s_to_u_value_bonus = torch.zeros_like(reward)
            replay.add(
                obs_before,
                action,
                reward,
                next_obs_raw,
                done,
            )
        if hard_switch_enabled:
            hard_switch.resample_switches(done, generator=runner.rng)
        next_done = done.float()
        global_step += nworld
        log_reward_sum += reward.detach().sum()
        log_sample_count += int(nworld)
        log_u_to_s_value_bonus_sum += u_to_s_value_bonus.detach().sum()
        log_s_to_u_value_bonus_sum += s_to_u_value_bonus.detach().sum()
        log_u_to_s_value_bonus_count += int((u_to_s_value_bonus != 0.0).sum().item())
        log_s_to_u_value_bonus_count += int((s_to_u_value_bonus != 0.0).sum().item())
        log_done_count += terms["done_count"].detach().sum()
        log_fall_count += terms["fall_done"].detach().sum()
        log_done_return_sum += terms["episode_return_done_sum"].detach().sum()
        log_done_length_sum += terms["episode_length_done_sum"].detach().sum()
        done_f = done.detach().float()
        keep_f = 1.0 - done_f
        for key in tracked_train_terms:
            value = terms.get(key)
            if value is None:
                episode_term_running[key].mul_(keep_f)
                continue
            value = value.detach().to(dtype=torch.float32)
            log_term_step_sum[key] += value.sum()
            running = episode_term_running[key]
            running.add_(value)
            log_term_done_sum[key] += (running * done_f).sum()
            running.mul_(keep_f)

        learned = 0
        learned_uphill = 0
        if (
            replay.size >= max(batch_size, learning_starts)
            and env_step % train_freq == 0
            and (not hard_switch_enabled or hard_switch_train_stair)
        ):
            for _ in range(gradient_steps):
                train_stats = update_sac_expert_once(
                    replay=replay,
                    batch_size=batch_size,
                    actor=actor,
                    qf1=qf1,
                    qf2=qf2,
                    qf1_target=qf1_target,
                    qf2_target=qf2_target,
                    actor_optimizer=actor_optimizer,
                    q_optimizer=q_optimizer,
                    alpha_optimizer=alpha_optimizer,
                    log_alpha=log_alpha,
                    obs_normalizer=obs_normalizer,
                    gated_spec=gated_spec,
                    current_ref_gate=current_ref_gate,
                    gamma=gamma,
                    tau=tau,
                    target_entropy=target_entropy,
                    max_grad_norm=float(sac_cfg.get("max_grad_norm", 10.0)),
                    human_anchor_actor=human_anchor_actor,
                    human_anchor_weight=current_human_anchor_weight,
                    muscle_count=int(model.na),
                    actor_updates_enabled=actor_updates_enabled,
                )
                learned += 1
        if (
            hard_switch_train_both
            and hard_switch_train_uphill
            and hard_switch_uphill_replay is not None
            and hard_switch_uphill_actor is not None
            and hard_switch_uphill_qf1 is not None
            and hard_switch_uphill_qf2 is not None
            and hard_switch_uphill_qf1_target is not None
            and hard_switch_uphill_qf2_target is not None
            and hard_switch_uphill_actor_optimizer is not None
            and hard_switch_uphill_q_optimizer is not None
            and hard_switch_uphill_alpha_optimizer is not None
            and hard_switch_uphill_log_alpha is not None
            and hard_switch_uphill_normalizer is not None
            and hard_switch_uphill_replay.size >= max(batch_size, learning_starts)
            and env_step % train_freq == 0
        ):
            for _ in range(gradient_steps):
                hard_switch_uphill_train_stats = update_sac_expert_once(
                    replay=hard_switch_uphill_replay,
                    batch_size=batch_size,
                    actor=hard_switch_uphill_actor,
                    qf1=hard_switch_uphill_qf1,
                    qf2=hard_switch_uphill_qf2,
                    qf1_target=hard_switch_uphill_qf1_target,
                    qf2_target=hard_switch_uphill_qf2_target,
                    actor_optimizer=hard_switch_uphill_actor_optimizer,
                    q_optimizer=hard_switch_uphill_q_optimizer,
                    alpha_optimizer=hard_switch_uphill_alpha_optimizer,
                    log_alpha=hard_switch_uphill_log_alpha,
                    obs_normalizer=hard_switch_uphill_normalizer,
                    gated_spec=gated_spec,
                    current_ref_gate=current_ref_gate,
                    gamma=gamma,
                    tau=tau,
                    target_entropy=target_entropy,
                    max_grad_norm=float(sac_cfg.get("max_grad_norm", 10.0)),
                    human_anchor_actor=hard_switch_uphill_human_anchor_actor,
                    human_anchor_weight=current_human_anchor_weight,
                    muscle_count=int(model.na),
                    actor_updates_enabled=actor_updates_enabled,
                )
                learned_uphill += 1

        should_log = args.log_every > 0 and (global_step % int(args.log_every) < nworld or global_step >= total_timesteps)
        if should_log:
            with torch.no_grad():
                log_obs = obs_normalizer.normalize(obs_before)
                q_log_obs = mask_ref_obs_for_q(
                    log_obs,
                    gated_spec["ref_indices"] if gated_spec is not None else None,
                    current_ref_gate,
                )
                q_rollout = torch.min(qf1(q_log_obs, action), qf2(q_log_obs, action))
                deterministic_action, _ = actor.get_action(log_obs, deterministic=True)
                q_policy = torch.min(qf1(q_log_obs, deterministic_action), qf2(q_log_obs, deterministic_action))
            done_count_for_mean = torch.clamp(log_done_count, min=1.0)
            sample_count_for_mean = max(int(log_sample_count), 1)
            compact_train_stats = dict(train_stats)
            row: dict[str, Any] = {
                "global_step": global_step,
                "env_step": env_step,
                "nworld": int(nworld),
                "gradient_steps": int(gradient_steps),
                "replay_size": int(replay.size),
                "learned_gradient_steps": int(learned),
                "uphill_replay_size": 0 if hard_switch_uphill_replay is None else int(hard_switch_uphill_replay.size),
                "uphill_learned_gradient_steps": int(learned_uphill),
                "seconds_step": time.perf_counter() - update_start,
                "samples_per_sec_step": float(nworld / max(time.perf_counter() - update_start, 1e-9)),
                "step_reward_mean": float(reward.mean().item()),
                "window_reward_mean": float((log_reward_sum / float(sample_count_for_mean)).item()),
                "step_done_rate": float(next_done.mean().item()),
                "window_done_rate": float((log_done_count / float(sample_count_for_mean)).item()),
                "window_fall_rate": float((log_fall_count / float(sample_count_for_mean)).item()),
                "done_count": float(log_done_count.item()),
                "episode_return_done_mean": float((log_done_return_sum / done_count_for_mean).item()),
                "episode_len_done_mean": float((log_done_length_sum / done_count_for_mean).item()),
                "current_episode_return_mean": float(runner.episode_return.mean().item()),
                "current_episode_return_max": float(runner.episode_return.max().item()),
                "current_episode_len_mean": float(runner.episode_length.mean().item()),
                "q_rollout_action_mean": float(q_rollout.mean().item()),
                "q_rollout_action_min": float(q_rollout.min().item()),
                "q_rollout_action_max": float(q_rollout.max().item()),
                "q_policy_action_mean": float(q_policy.mean().item()),
                "q_policy_action_min": float(q_policy.min().item()),
                "q_policy_action_max": float(q_policy.max().item()),
                "step_fall_rate": float(terms["fall_done"].mean().item()),
                "step_qvel_done_rate": float(terms["qvel_done"].mean().item()),
                "policy_logprob": float(logprob.mean().item()),
                "activation_mean": float(
                    terms.get("activation_mean", torch.zeros_like(reward)).mean().item()
                ),
                "action_clip_fraction": float((torch.abs(action) > 0.999).float().mean().item()),
                "reference_phase_lead_steps": int(reference_curriculum["phase_lead_steps"]),
                "reference_phase_tolerance_steps": int(reference_curriculum["phase_tolerance_steps"]),
                "reference_swing_exaggeration_scale": float(reference_curriculum["swing_exaggeration_scale"]),
                "x_aligned_reference_rate": float(terms.get("x_aligned_reference", torch.zeros_like(reward)).mean().item()),
                "recovery_mode_rate": float(terms.get("recovery_mode", torch.zeros_like(reward)).mean().item()),
                "reset_phase_stage": "" if current_reset_phase_stage is None else str(current_reset_phase_stage["name"]),
                "reset_phase_choice_count": 0 if runner.phase_choices is None else int(runner.phase_choices.numel()),
                "hard_switch_enabled": bool(hard_switch_enabled),
                "hard_switch_stair_route_rate": 0.0
                if hard_switch_stair_mask is None
                else float(hard_switch_stair_mask.float().mean().item()),
                "hard_switch_stair_train_rate": 0.0
                if hard_switch_stair_train_mask is None
                else float(hard_switch_stair_train_mask.float().mean().item()),
                "hard_switch_uphill_train_rate": 0.0
                if hard_switch_uphill_train_mask is None
                else float(hard_switch_uphill_train_mask.float().mean().item()),
                "hard_switch_replay_overlap_rate": 0.0
                if hard_switch_stair_train_mask is None or hard_switch_uphill_train_mask is None
                else float((hard_switch_stair_train_mask & hard_switch_uphill_train_mask).float().mean().item()),
                "hard_switch_u_to_s_value_bonus_mean": float(
                    (log_u_to_s_value_bonus_sum / float(max(log_u_to_s_value_bonus_count, 1))).item()
                ),
                "hard_switch_s_to_u_value_bonus_mean": float(
                    (log_s_to_u_value_bonus_sum / float(max(log_s_to_u_value_bonus_count, 1))).item()
                ),
                "hard_switch_u_to_s_value_bonus_count": int(log_u_to_s_value_bonus_count),
                "hard_switch_s_to_u_value_bonus_count": int(log_s_to_u_value_bonus_count),
                "hard_switch_switch_to_stair_x": float(hard_switch.to_stair_x) if hard_switch_enabled else 0.0,
                "hard_switch_switch_to_uphill_x": float(hard_switch.to_uphill_x) if hard_switch_enabled else 0.0,
                "hard_switch_switch_to_stair_x_mean": 0.0
                if hard_switch.env_to_stair_x is None
                else float(hard_switch.env_to_stair_x.mean().item()),
                "hard_switch_switch_to_uphill_x_mean": 0.0
                if hard_switch.env_to_uphill_x is None
                else float(hard_switch.env_to_uphill_x.mean().item()),
                "recovery_bank_size": int(getattr(runner, "recovery_bank_size", 0)),
                "recovery_last_stage_count": int(getattr(runner, "recovery_last_stage_count", 0)),
                "recovery_last_commit_count": int(getattr(runner, "recovery_last_commit_count", 0)),
                "recovery_last_restore_count": int(getattr(runner, "recovery_last_restore_count", 0)),
                "offline_recovery_bank_size": int(getattr(runner, "offline_recovery_bank_size", 0)),
                "offline_recovery_last_restore_count": int(getattr(runner, "offline_recovery_last_restore_count", 0)),
                "future_obs_dropout_prob": float(current_future_obs_dropout_prob),
                "ref_gate": float(current_ref_gate),
                "episode_steps": int(current_episode_steps),
                "action_anchor_weight": float(current_human_anchor_weight),
                "actor_updates_enabled": bool(actor_updates_enabled),
                **compact_train_stats,
            }
            if bool(hard_switch_train_both):
                for key, value in hard_switch_uphill_train_stats.items():
                    row[f"uphill_{key}"] = float(value)
            exo_base = gated_ref_base_actor(actor)
            if exo_base is not None and exo_base.exo_head_enabled:
                row["exo_head_type"] = exo_base.exo_policy_head.head_type
                gate = exo_base.exo_policy_head.last_gate
                if gate is not None:
                    gate_mean = gate.mean(dim=0)
                    gate_entropy = -(gate * torch.log(torch.clamp(gate, min=1e-8))).sum(dim=1).mean()
                    row["exo_gate_entropy"] = float(gate_entropy.item())
                    for index, value in enumerate(gate_mean):
                        row[f"exo_gate_usage_{index}"] = float(value.item())
            for key in tracked_train_terms:
                row[f"step_mean_{key}"] = float((log_term_step_sum[key] / float(sample_count_for_mean)).item())
                row[f"episode_done_mean_{key}"] = float((log_term_done_sum[key] / done_count_for_mean).item())
            append_csv(args.outdir / "train_metrics.csv", row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            log_reward_sum.zero_()
            log_sample_count = 0
            log_u_to_s_value_bonus_sum.zero_()
            log_s_to_u_value_bonus_sum.zero_()
            log_u_to_s_value_bonus_count = 0
            log_s_to_u_value_bonus_count = 0
            log_done_count.zero_()
            log_fall_count.zero_()
            log_done_return_sum.zero_()
            log_done_length_sum.zero_()
            for key in tracked_train_terms:
                log_term_step_sum[key].zero_()
                log_term_done_sum[key].zero_()

        if args.eval_every > 0 and (global_step % int(args.eval_every) < nworld or global_step >= total_timesteps):
            set_future_obs_dropout_prob(config, 0.0)
            eval_row = evaluate(
                agent=actor,
                obs_normalizer=obs_normalizer,
                model=model,
                data=data,
                config=config,
                reference=reference,
                args=args,
                device=device,
                update=env_step,
                global_step=global_step,
            )
            set_future_obs_dropout_prob(config, current_future_obs_dropout_prob)
            append_csv(args.outdir / "eval_metrics.csv", eval_row)
            print(json.dumps({"eval": eval_row}, ensure_ascii=False), flush=True)

        if args.video_every > 0 and (global_step % int(args.video_every) < nworld or global_step >= total_timesteps):
            set_future_obs_dropout_prob(config, 0.0)
            original_video_phase = int(args.video_phase)
            for video_phase in configured_video_phases(
                config,
                reference,
                original_video_phase,
                global_step=global_step,
                run_start_global_step=run_start_global_step,
                video_every=int(args.video_every),
            ):
                args.video_phase = int(video_phase)
                video_row = render_policy_video(
                    agent=actor,
                    obs_normalizer=obs_normalizer,
                    config=config,
                    reference=reference,
                    args=args,
                    device=device,
                    update=env_step,
                    global_step=global_step,
                )
                append_csv(args.outdir / "video_metrics.csv", video_row)
                print(json.dumps({"video": video_row}, ensure_ascii=False), flush=True)
            args.video_phase = original_video_phase
            set_future_obs_dropout_prob(config, current_future_obs_dropout_prob)

        if args.checkpoint_every > 0 and (global_step % int(args.checkpoint_every) < nworld or global_step >= total_timesteps):
            payload = {
                "global_step": global_step,
                "env_step": env_step,
                "run_start_global_step": run_start_global_step,
                "run_config": run_config,
                "obs_normalizer": obs_normalizer.state_dict(),
            }
            primary_modules = {
                "actor": actor,
                "qf1": qf1,
                "qf2": qf2,
                "qf1_target": qf1_target,
                "qf2_target": qf2_target,
                "actor_optimizer": actor_optimizer,
                "q_optimizer": q_optimizer,
                "alpha_optimizer": alpha_optimizer,
                "log_alpha": log_alpha,
            }
            checkpoint_path = save_expert_checkpoint_set(
                outdir=args.outdir,
                global_step=global_step,
                payload=payload,
                primary=primary_modules,
                hard_switch=hard_switch,
            )
            active_video_export, export_row = maybe_launch_checkpoint_video_export(
                config=config,
                args=args,
                checkpoint_path=checkpoint_path,
                global_step=global_step,
                nworld=nworld,
                active_process=active_video_export,
            )
            append_csv(args.outdir / "video_export_metrics.csv", export_row)
            print(json.dumps({"video_export": export_row}, ensure_ascii=False), flush=True)

        env_step += 1
