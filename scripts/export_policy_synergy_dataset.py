#!/usr/bin/env python3
"""Export policy activations and muscle length traces for synergy analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cleanrl.ppo_muscle_mjwarp import (  # noqa: E402
    DEFAULT_REFERENCE_PATH,
    ObsNormalizer,
    build_muscle_model,
    cpu_policy_obs,
    current_terrain_height_np,
    joint_id,
    load_reference_from_config,
    muscle_action_to_activation,
    post_reference_enabled,
    post_reference_valid_steps,
    reference_index,
    set_cpu_reference_state,
)
from cleanrl.sac_muscle_mjwarp import (  # noqa: E402
    GatedRefSACActor,
    SACActor,
    SymmetricSACActor,
    build_sagittal_mirror_spec,
    gated_ref_obs_spec,
    policy_architecture,
)


def actuator_names(model: mujoco.MjModel) -> list[str]:
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, idx) or str(idx) for idx in range(int(model.nu))]


def load_actor(
    checkpoint: dict[str, Any],
    *,
    model: mujoco.MjModel,
    config: dict[str, Any],
    obs_dim: int,
    act_dim: int,
    device: torch.device,
) -> torch.nn.Module:
    sac_cfg = config.get("sac", config.get("ppo", {}))
    architecture = policy_architecture(config)
    if architecture == "gated_ref_sac":
        spec = gated_ref_obs_spec(model, config, obs_dim=int(obs_dim), device=device)
        base_actor = GatedRefSACActor(
            int(obs_dim),
            int(act_dim),
            base_indices=spec["base_indices"],
            ref_indices=spec["ref_indices"],
            logstd_init=float(sac_cfg.get("actor_logstd_init", -0.5)),
            initial_action_mean=float(sac_cfg.get("initial_actor_action_mean", -0.2)),
            hidden_dim=int(config.get("policy", {}).get("hidden_dim", 256)),
            latent_dim=int(config.get("policy", {}).get("latent_dim", 128)),
            initial_ref_gate=float(config.get("policy", {}).get("current_ref_gate", config.get("policy", {}).get("ref_gate", 1.0))),
        ).to(device)
    else:
        base_actor = SACActor(
            int(obs_dim),
            int(act_dim),
            logstd_init=float(sac_cfg.get("actor_logstd_init", -0.5)),
            initial_action_mean=float(sac_cfg.get("initial_actor_action_mean", -0.2)),
        ).to(device)

    if bool(sac_cfg.get("symmetric_policy", False)):
        mirror = build_sagittal_mirror_spec(
            model,
            config,
            obs_dim=int(obs_dim),
            future_steps=int(config.get("imitation", {}).get("reference_future_steps", 0)),
            device=device,
        )
        actor: torch.nn.Module = SymmetricSACActor(
            base_actor,
            obs_perm=mirror["obs_perm"],
            obs_sign=mirror["obs_sign"],
            act_perm=mirror["act_perm"],
            act_sign=mirror["act_sign"],
        ).to(device)
    else:
        actor = base_actor
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()
    return actor


def selected_phases(
    reference_length: int,
    *,
    phase_list: str,
    phase_start: int,
    phase_end: int | None,
    phase_stride: int,
    max_phases: int | None,
) -> list[int]:
    if str(phase_list).strip():
        phases = [int(item) for item in str(phase_list).replace(",", " ").split()]
        phases = [phase for phase in phases if 0 <= int(phase) < int(reference_length)]
    else:
        end = int(reference_length) if phase_end is None else min(int(reference_length), int(phase_end))
        phases = list(range(max(0, int(phase_start)), end, max(1, int(phase_stride))))
    if max_phases is not None:
        phases = phases[: max(0, int(max_phases))]
    if not phases:
        raise ValueError("empty phase selection")
    return phases


@torch.no_grad()
def export_dataset(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    run_config = checkpoint.get("run_config", {})
    if "config" not in run_config:
        raise ValueError(f"checkpoint has no embedded run_config.config: {args.checkpoint}")
    config = json.loads(json.dumps(run_config["config"]))
    reference_path = args.reference
    if reference_path is not None:
        config["reference_pool"] = {"paths": [str(reference_path.resolve())]}
        config["reference_pool_schedule"] = []
    elif not config.get("reference_pool", {}).get("paths"):
        raw_reference = run_config.get("args", {}).get("reference", None)
        reference_path = Path(raw_reference) if raw_reference else DEFAULT_REFERENCE_PATH

    model, data = build_muscle_model(config)
    reference = load_reference_from_config(reference_path, model, float(config["control"]["control_hz"]), device, config)
    obs_dim = int(run_config.get("obs_dim", 0) or 0)
    act_dim = int(run_config.get("act_dim", int(model.nu)) or int(model.nu))
    if obs_dim <= 0:
        probe_phase = max(0, min(int(args.phase_start), int(reference["length"]) - 1))
        set_cpu_reference_state(model, data, reference, probe_phase)
        obs_dim = int(cpu_policy_obs(model, data, reference, config, probe_phase, 0, device).shape[-1])

    actor = load_actor(checkpoint, model=model, config=config, obs_dim=obs_dim, act_dim=act_dim, device=device)
    obs_normalizer = ObsNormalizer(
        obs_dim,
        device,
        enabled=bool(config.get("sac", {}).get("normalize_observations", True)),
        clip=float(config.get("sac", {}).get("obs_norm_clip", 10.0)),
    )
    if "obs_normalizer" in checkpoint:
        obs_normalizer.load_state_dict(checkpoint["obs_normalizer"])

    phases = selected_phases(
        int(reference["length"]),
        phase_list=str(args.phase_list),
        phase_start=int(args.phase_start),
        phase_end=args.phase_end,
        phase_stride=int(args.phase_stride),
        max_phases=args.max_phases,
    )
    frame_skip = int(config["control"]["frame_skip"])
    pelvis_tx_qpos = int(model.jnt_qposadr[joint_id(model, "pelvis_tx")])
    pelvis_ty_qpos = int(model.jnt_qposadr[joint_id(model, "pelvis_ty")])
    pelvis_tilt_qpos = int(model.jnt_qposadr[joint_id(model, "pelvis_tilt")])
    names = actuator_names(model)

    rows: list[dict[str, Any]] = []
    action_rows: list[np.ndarray] = []
    activation_rows: list[np.ndarray] = []
    actuator_length_rows: list[np.ndarray] = []
    actuator_velocity_rows: list[np.ndarray] = []
    qpos_rows: list[np.ndarray] = []
    qvel_rows: list[np.ndarray] = []
    act_rows: list[np.ndarray] = []

    for start_phase in phases:
        phase = int(start_phase)
        set_cpu_reference_state(model, data, reference, phase)
        fell = False
        for frame in range(int(args.frames_per_phase)):
            ref_phase = int(reference_index(torch.tensor([phase], dtype=torch.long, device=device), reference, config)[0].detach().cpu().item())
            obs = cpu_policy_obs(model, data, reference, config, phase, frame, device)
            norm_obs = obs_normalizer.normalize(obs)
            action, _logprob = actor.get_action(norm_obs, deterministic=not bool(args.stochastic))
            action0 = torch.clamp(action[0], -1.0, 1.0)
            activation = muscle_action_to_activation(action0).detach().cpu().numpy().astype(np.float32)
            action_np = action0.detach().cpu().numpy().astype(np.float32)

            terrain_height = current_terrain_height_np(model, data, reference, config, phase)
            pelvis_height = float(data.qpos[pelvis_ty_qpos]) - float(terrain_height)
            reference_valid = (not post_reference_enabled(config)) or frame < int(post_reference_valid_steps(reference, config))
            rows.append(
                {
                    "sample": len(rows),
                    "start_phase": int(start_phase),
                    "frame": int(frame),
                    "phase": int(ref_phase),
                    "reference_valid": bool(reference_valid),
                    "pelvis_tx": float(data.qpos[pelvis_tx_qpos]),
                    "pelvis_height_above_terrain": float(pelvis_height),
                    "pelvis_tilt": float(data.qpos[pelvis_tilt_qpos]),
                    "terrain_height": float(terrain_height),
                    "fell": bool(fell),
                }
            )
            action_rows.append(action_np)
            activation_rows.append(activation)
            actuator_length_rows.append(np.asarray(data.actuator_length, dtype=np.float32).copy())
            actuator_velocity_rows.append(np.asarray(data.actuator_velocity, dtype=np.float32).copy())
            qpos_rows.append(np.asarray(data.qpos, dtype=np.float32).copy())
            qvel_rows.append(np.asarray(data.qvel, dtype=np.float32).copy())
            act_rows.append(np.asarray(data.act, dtype=np.float32).copy())

            data.ctrl[:] = activation
            for _ in range(frame_skip):
                mujoco.mj_step(model, data)
            if bool(args.stop_on_fall):
                terrain_after = current_terrain_height_np(model, data, reference, config, phase)
                height_after = float(data.qpos[pelvis_ty_qpos]) - float(terrain_after)
                if height_after < float(config["reset"].get("safe_pelvis_height", 0.5)):
                    fell = True
                    break
            phase = phase + 1 if post_reference_enabled(config) else (phase + 1) % int(reference["length"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "checkpoint": str(args.checkpoint.resolve()),
        "global_step": int(checkpoint.get("global_step", -1)),
        "label": str(args.label),
        "reference_path": str(reference.get("path", "")),
        "reference_length": int(reference["length"]),
        "phase_start": int(args.phase_start),
        "phase_end": args.phase_end,
        "phase_list": str(args.phase_list),
        "phase_stride": int(args.phase_stride),
        "phases": phases,
        "frames_per_phase": int(args.frames_per_phase),
        "stochastic": bool(args.stochastic),
        "stop_on_fall": bool(args.stop_on_fall),
        "actuator_names": names,
        "obs_dim": int(obs_dim),
        "act_dim": int(act_dim),
        "config_experiment_name": str(config.get("experiment_name", "")),
    }
    np.savez_compressed(
        args.out,
        metadata=metadata,
        rows=np.asarray(rows, dtype=object),
        actions=np.stack(action_rows).astype(np.float32),
        activations=np.stack(activation_rows).astype(np.float32),
        actuator_length=np.stack(actuator_length_rows).astype(np.float32),
        actuator_velocity=np.stack(actuator_velocity_rows).astype(np.float32),
        qpos=np.stack(qpos_rows).astype(np.float32),
        qvel=np.stack(qvel_rows).astype(np.float32),
        act=np.stack(act_rows).astype(np.float32),
    )
    summary = {
        "out": str(args.out),
        "samples": len(rows),
        "phases": len(phases),
        "activation_mean": float(np.mean(np.stack(activation_rows))),
        "activation_max": float(np.max(np.stack(activation_rows))),
        "actuator_names": names,
    }
    args.out.with_suffix(".summary.json").write_text(json.dumps({**summary, "metadata": metadata}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--phase-start", type=int, default=0)
    parser.add_argument("--phase-end", type=int, default=None)
    parser.add_argument("--phase-list", default="")
    parser.add_argument("--phase-stride", type=int, default=4)
    parser.add_argument("--max-phases", type=int, default=None)
    parser.add_argument("--frames-per-phase", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--stop-on-fall", action="store_true")
    args = parser.parse_args()
    export_dataset(args)


if __name__ == "__main__":
    main()
