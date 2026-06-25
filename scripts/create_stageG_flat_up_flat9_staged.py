#!/usr/bin/env python3
"""Create staged SAC configs for the 9-degree flat-up-flat reference."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = (
    ROOT
    / "reference_exports/stageG_flat_up_flat_9deg_topfilled_20260624-232908/stageG_flat_up_flat_9deg_myoassist_3d.npz"
)
DEFAULT_BASE_CONFIG = (
    ROOT
    / "reference_exports/stageG_flat_up_flat_9deg_topfilled_20260624-232908/"
    "muscle_2d_mjwarp_stageG_flat_up_flat_9deg_reference_review.json"
)
DEFAULT_OUTDIR = ROOT / "configs/stageG_flat_up_flat9_staged"


def clamp_window(start: int, end: int, length: int) -> dict[str, int]:
    start = max(0, min(int(start), int(length)))
    end = max(start, min(int(end), int(length)))
    return {"start": start, "end": end}


def reward_weights(
    *,
    slip: float,
    fall: float,
    height: float,
    upright: float,
    tangent: float,
    normal: float,
) -> dict[str, float]:
    return {
        "tracking_qpos_penalty": 18.0,
        "tracking_qvel_penalty": 0.55,
        "tracking_foot_site_penalty": 20.0,
        "tracking_swing_foot_site_penalty": 24.0,
        "tracking_swing_hip_penalty": 1.0,
        "tracking_swing_limb_penalty": 8.0,
        "tracking_activation_symmetry_penalty": 0.0,
        "tracking_future_foot_site_penalty": 8.0,
        "terminal_swing_landing_penalty": 4.0,
        "tracking_pelvis_penalty": 20.0,
        "pelvis_tx_vel_ref": 0.0,
        "pelvis_ty_vel_ref": 0.0,
        "pelvis_tangent_vel_ref": float(tangent),
        "pelvis_normal_vel_ref": float(normal),
        "foot_slip": float(slip),
        "tracking_energy_penalty": 0.02,
        "activation_smooth": 0.75,
        "upright": 10.0 * float(upright),
        "height": float(height),
        "alive": 0.0,
        "fall": float(fall),
    }


def set_common(
    config: dict[str, Any],
    *,
    name: str,
    reference_path: Path,
    episode_steps: int,
    phase_windows: list[dict[str, int]],
    video_phases: list[int],
) -> None:
    config["experiment_name"] = name
    config["reference_pool"] = {"paths": [str(reference_path.resolve())]}
    config["reference_pool_schedule"] = []
    config["resume_schedule_from_checkpoint"] = False
    config["reset"]["episode_steps"] = int(episode_steps)
    config["reset"]["phase_windows"] = phase_windows
    config["reset"]["phase_indices"] = []
    config["reset"]["phase_index_jitter"] = 0
    config["reset"]["qpos_noise"] = 0.0015 if int(episode_steps) <= 96 else 0.002
    config["reset"]["qvel_noise"] = 0.002 if int(episode_steps) <= 96 else 0.003
    config.pop("reset_phase_schedule", None)
    config.pop("reset_phase_schedule_mode", None)
    config["observation"] = {
        "localize_root": True,
        "phase_obs": "none",
        "include_foot_rel_z": True,
        "include_foot_ground_slope": True,
        "include_contact_obs": True,
        "frame_stack_prev_steps": 2,
    }
    config.setdefault("policy", {})["ref_gate"] = 1.0
    config.setdefault("policy", {})["ref_gate_schedule"] = [{"after_steps": 0, "gate": 1.0}]
    config["video"] = {"phase_indices": list(video_phases), "phase_mode": "round_robin"}
    export = config.setdefault("checkpoint_video_export", {})
    export["enabled"] = False
    export["phase_indices"] = list(video_phases)


def set_reward_schedule(config: dict[str, Any], initial: dict[str, float], later: dict[str, float]) -> None:
    config["reward"] = dict(initial)
    config["reward_schedule_mode"] = "relative"
    config["reward_schedule"] = [
        {"after_steps": 0, "weights": dict(initial)},
        {"after_steps": 32768, "weights": dict(later)},
    ]


def set_sac(config: dict[str, Any], *, total_timesteps: int, alpha: float, entropy: float, logstd: float) -> None:
    sac = config.setdefault("sac", {})
    sac["total_timesteps"] = int(total_timesteps)
    sac["num_envs"] = 64
    sac["buffer_size"] = 250000
    sac["learning_starts"] = 1024
    sac["batch_size"] = 1024
    sac["gradient_steps"] = 2
    sac["alpha"] = float(alpha)
    sac["target_entropy"] = float(entropy)
    sac["actor_logstd_init"] = float(logstd)
    sac["initial_actor_action_mean"] = -0.6
    sac["symmetric_policy"] = True


def main() -> None:
    reference_path = DEFAULT_REFERENCE
    base_config_path = DEFAULT_BASE_CONFIG
    outdir = DEFAULT_OUTDIR
    raw = np.load(reference_path, allow_pickle=True)
    metadata = raw["metadata"].item()
    length = int(metadata["data_length"])
    ranges = metadata["label_ranges"]
    base = json.loads(base_config_path.read_text(encoding="utf-8"))

    key_phases = []
    for name in [
        "level_low_1",
        "level_low_2_partial",
        "level_to_up",
        "uphill_steady_1",
        "uphill_steady_2",
        "uphill_approach_top",
        "up_to_level",
        "level_high_1",
        "level_high_2",
    ]:
        if name not in ranges:
            continue
        phase = int(ranges[name]["start"])
        if name in {"uphill_steady_1", "uphill_steady_2"}:
            phase += 8
        key_phases.append(min(phase, length - 1))

    stages = [
        {
            "stage": "stageG_fuf9_A_h48_islands_imit_sac",
            "episode_steps": 48,
            "phase_windows": [
                clamp_window(0, 48, length),
                clamp_window(140, 188, length),
                clamp_window(333, 381, length),
            ],
            "total_timesteps": 196608,
            "reward0": reward_weights(slip=0.0, fall=0.5, height=0.25, upright=0.25, tangent=1.5, normal=0.5),
            "reward1": reward_weights(slip=0.0, fall=1.0, height=0.5, upright=0.5, tangent=2.0, normal=0.75),
            "alpha": 0.10,
            "entropy": -12.0,
            "logstd": -0.9,
        },
        {
            "stage": "stageG_fuf9_B_h48_transition_short_imit_sac",
            "episode_steps": 48,
            "phase_windows": [
                clamp_window(64, 112, length),
                clamp_window(246, 309, length),
                clamp_window(124, 172, length),
                clamp_window(333, 381, length),
            ],
            "total_timesteps": 262144,
            "reward0": reward_weights(slip=0.0, fall=1.0, height=0.5, upright=0.5, tangent=2.0, normal=0.75),
            "reward1": reward_weights(slip=0.01, fall=2.0, height=0.75, upright=0.75, tangent=2.5, normal=1.0),
            "alpha": 0.095,
            "entropy": -11.0,
            "logstd": -1.0,
        },
        {
            "stage": "stageG_fuf9_C_h72_transition_imit_sac",
            "episode_steps": 72,
            "phase_windows": [
                clamp_window(48, 124, length),
                clamp_window(224, 318, length),
                clamp_window(132, 193, length),
                clamp_window(201, 261, length),
                clamp_window(333, 387, length),
            ],
            "total_timesteps": 393216,
            "reward0": reward_weights(slip=0.01, fall=2.0, height=0.75, upright=0.75, tangent=2.5, normal=1.0),
            "reward1": reward_weights(slip=0.02, fall=3.0, height=1.0, upright=1.0, tangent=3.0, normal=1.25),
            "alpha": 0.09,
            "entropy": -10.0,
            "logstd": -1.05,
        },
        {
            "stage": "stageG_fuf9_D_h96_mixed_imit_sac",
            "episode_steps": 96,
            "phase_windows": [
                clamp_window(0, 124, length),
                clamp_window(96, 124, length),
                clamp_window(132, 193, length),
                clamp_window(201, 294, length),
                clamp_window(240, 363, length),
                clamp_window(333, 363, length),
            ],
            "total_timesteps": 393216,
            "reward0": reward_weights(slip=0.02, fall=3.0, height=1.0, upright=1.0, tangent=3.0, normal=1.25),
            "reward1": reward_weights(slip=0.03, fall=4.0, height=1.0, upright=1.0, tangent=3.5, normal=1.5),
            "alpha": 0.085,
            "entropy": -9.5,
            "logstd": -1.1,
        },
        {
            "stage": "stageG_fuf9_E_h144_long_imit_sac",
            "episode_steps": 144,
            "phase_windows": [
                clamp_window(0, 124, length),
                clamp_window(96, 124, length),
                clamp_window(132, 261, length),
                clamp_window(216, 333, length),
            ],
            "total_timesteps": 524288,
            "reward0": reward_weights(slip=0.03, fall=4.0, height=1.0, upright=1.0, tangent=3.5, normal=1.5),
            "reward1": reward_weights(slip=0.04, fall=5.0, height=1.0, upright=1.0, tangent=4.0, normal=1.75),
            "alpha": 0.08,
            "entropy": -9.0,
            "logstd": -1.15,
        },
        {
            "stage": "stageG_fuf9_F_h192_full_imit_sac",
            "episode_steps": 192,
            "phase_windows": [
                clamp_window(0, 96, length),
                clamp_window(96, 267, length),
                clamp_window(240, 267, length),
            ],
            "total_timesteps": 524288,
            "reward0": reward_weights(slip=0.04, fall=5.0, height=1.0, upright=1.0, tangent=4.0, normal=1.75),
            "reward1": reward_weights(slip=0.05, fall=6.0, height=1.0, upright=1.0, tangent=4.0, normal=2.0),
            "alpha": 0.075,
            "entropy": -8.5,
            "logstd": -1.2,
        },
    ]

    outdir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for stage in stages:
        config = copy.deepcopy(base)
        set_common(
            config,
            name=str(stage["stage"]),
            reference_path=reference_path,
            episode_steps=int(stage["episode_steps"]),
            phase_windows=stage["phase_windows"],
            video_phases=key_phases,
        )
        set_reward_schedule(config, stage["reward0"], stage["reward1"])
        set_sac(
            config,
            total_timesteps=int(stage["total_timesteps"]),
            alpha=float(stage["alpha"]),
            entropy=float(stage["entropy"]),
            logstd=float(stage["logstd"]),
        )
        path = outdir / f"muscle_2d_mjwarp_{stage['stage']}.json"
        path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        manifest.append(
            {
                "stage": str(stage["stage"]),
                "config": str(path.resolve()),
                "episode_steps": int(stage["episode_steps"]),
                "phase_windows": stage["phase_windows"],
                "total_timesteps": int(stage["total_timesteps"]),
                "initial_foot_slip_weight": float(stage["reward0"]["foot_slip"]),
                "later_foot_slip_weight": float(stage["reward1"]["foot_slip"]),
            }
        )
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"reference": str(reference_path.resolve()), "manifest": manifest}, indent=2), flush=True)


if __name__ == "__main__":
    main()
