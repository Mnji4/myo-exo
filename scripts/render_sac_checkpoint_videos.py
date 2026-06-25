#!/usr/bin/env python3
"""Render deterministic policy videos from a saved MJWarp SAC checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cleanrl.ppo_muscle_mjwarp import (  # noqa: E402
    ObsNormalizer,
    append_csv,
    build_muscle_model,
    load_config,
    load_reference_from_config,
    render_policy_video,
)
from cleanrl.sac_muscle_mjwarp import (  # noqa: E402
    GatedRefSACActor,
    SACActor,
    SymmetricSACActor,
    build_sagittal_mirror_spec,
    configured_video_phases,
    gated_ref_obs_spec,
    policy_architecture,
)


def pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def acquire_lock(lock_file: Path | None) -> bool:
    if lock_file is None:
        return True
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    if lock_file.exists():
        try:
            pid = int(lock_file.read_text(encoding="utf-8").strip())
        except ValueError:
            pid = -1
        if pid_is_running(pid):
            print(json.dumps({"status": "skipped", "reason": "lock_active", "pid": pid}), flush=True)
            return False
    lock_file.write_text(str(os.getpid()), encoding="utf-8")
    return True


def release_lock(lock_file: Path | None) -> None:
    if lock_file is None or not lock_file.exists():
        return
    try:
        pid = int(lock_file.read_text(encoding="utf-8").strip())
    except ValueError:
        pid = -1
    if pid == os.getpid():
        lock_file.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=Path("/home/lzn/myoassist/rl_train/reference_data/short_reference_gait.npz"))
    parser.add_argument("--phase", type=int, action="append", default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--video-steps", type=int, default=96)
    parser.add_argument("--video-height", type=int, default=720)
    parser.add_argument("--video-width", type=int, default=1280)
    parser.add_argument("--video-camera-distance", type=float, default=6.0)
    parser.add_argument("--video-camera-height", type=float, default=0.9)
    parser.add_argument("--video-activation-prior-execution-mix", type=float, default=None)
    parser.add_argument("--ignore-fall", action="store_true")
    parser.add_argument("--lock-file", type=Path, default=None)
    args = parser.parse_args()

    if not acquire_lock(args.lock_file):
        return

    try:
        run(args)
    finally:
        release_lock(args.lock_file)


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    config = load_config(args.config)
    model, _data = build_muscle_model(config)
    reference = load_reference_from_config(args.reference, model, float(config["control"]["control_hz"]), device, config)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    obs_dim = int(checkpoint.get("run_config", {}).get("obs_dim", 0) or 0)
    act_dim = int(checkpoint.get("run_config", {}).get("act_dim", model.nu) or model.nu)
    if obs_dim <= 0:
        # Fall back to the current config layout.
        future_steps = int(config.get("imitation", {}).get("reference_future_steps", 0))
        obs_dim = int(model.nq + model.nv + model.na + 2 * len(reference["joint_names"]) + 2 + 2 * len(reference["foot_site_names"]))
        obs_dim += future_steps * int(len(reference["joint_names"]) + 2 * len(reference["foot_site_names"]))

    sac_cfg = config.get("sac", config.get("ppo", {}))
    run_cfg = checkpoint.get("run_config", {})
    architecture = str(run_cfg.get("policy_architecture", policy_architecture(config)))
    if architecture == "gated_ref_sac":
        gated_spec = gated_ref_obs_spec(model, config, obs_dim=obs_dim, device=device)
        base_actor = GatedRefSACActor(
            obs_dim,
            act_dim,
            base_indices=gated_spec["base_indices"],
            ref_indices=gated_spec["ref_indices"],
            logstd_init=float(sac_cfg.get("actor_logstd_init", -0.5)),
            initial_action_mean=float(sac_cfg.get("initial_actor_action_mean", -0.2)),
            hidden_dim=int(config.get("policy", {}).get("hidden_dim", 256)),
            latent_dim=int(config.get("policy", {}).get("latent_dim", 128)),
            initial_ref_gate=float(config.get("policy", {}).get("current_ref_gate", config.get("policy", {}).get("ref_gate", 1.0))),
        ).to(device)
    else:
        base_actor = SACActor(
            obs_dim,
            act_dim,
            logstd_init=float(sac_cfg.get("actor_logstd_init", -0.5)),
            initial_action_mean=float(sac_cfg.get("initial_actor_action_mean", -0.2)),
        ).to(device)
    if bool(checkpoint.get("run_config", {}).get("symmetric_policy", sac_cfg.get("symmetric_policy", False))):
        mirror_spec = build_sagittal_mirror_spec(
            model,
            config,
            obs_dim=obs_dim,
            future_steps=int(config.get("imitation", {}).get("reference_future_steps", 0)),
            device=device,
        )
        actor = SymmetricSACActor(
            base_actor,
            obs_perm=mirror_spec["obs_perm"],
            obs_sign=mirror_spec["obs_sign"],
            act_perm=mirror_spec["act_perm"],
            act_sign=mirror_spec["act_sign"],
        ).to(device)
    else:
        actor = base_actor
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()

    obs_normalizer = ObsNormalizer(
        obs_dim,
        device,
        enabled=bool(sac_cfg.get("normalize_observations", True)),
        clip=float(sac_cfg.get("obs_norm_clip", 10.0)),
    )
    if "obs_normalizer" in checkpoint:
        obs_normalizer.load_state_dict(checkpoint["obs_normalizer"])

    args.outdir.mkdir(parents=True, exist_ok=True)
    phases = args.phase if args.phase is not None else configured_video_phases(config, reference, 0)
    rows = []
    for phase in phases:
        args.video_phase = int(phase)
        row = render_policy_video(
            agent=actor,
            obs_normalizer=obs_normalizer,
            config=config,
            reference=reference,
            args=args,
            device=device,
            update=int(checkpoint.get("env_step", 0)),
            global_step=int(checkpoint.get("global_step", 0)),
        )
        append_csv(args.outdir / "video_metrics.csv", row)
        rows.append(row)
    print(json.dumps({"videos": rows}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
