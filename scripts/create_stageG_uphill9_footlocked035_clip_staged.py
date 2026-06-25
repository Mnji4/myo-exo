#!/usr/bin/env python3
"""Create 9.207 degree uphill-only short-horizon configs for one footlocked clip."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs/muscle_2d_mjwarp_stageG0_h32_gated_ref_ramp_short_sac.json"
OUTDIR = ROOT / "configs/stageG_uphill9_footlocked035_clip_staged"
REFERENCE = Path(
    "/home/lzn/exoskeleton_terrain/data/camargo_reference_footlocked/"
    "camargo_ab06_slopeascent_rampascent_footlocked_myoassist_3d.npz"
)
SLOPE_9 = 0.16209004
ENTRY_SHIFT = 0.35


def reference_length_at_control_hz(path: Path, control_hz: float) -> int:
    raw = np.load(path, allow_pickle=True)
    metadata = raw["metadata"].item()
    series = raw["series_data"].item()
    source_hz = float(metadata.get("sample_rate", control_hz))
    length = len(next(iter(series.values())))
    indices = np.unique(
        np.clip(
            np.round(np.arange(0.0, float(length), source_hz / float(control_hz))).astype(np.int64),
            0,
            length - 1,
        )
    )
    return int(len(indices))


def window_for_horizon(length: int, horizon: int) -> dict[str, int]:
    end = max(1, int(length) - int(horizon) + 1)
    return {"start": 0, "end": end}


def configure_base(config: dict[str, Any]) -> None:
    config["reference_pool"] = {"paths": [str(REFERENCE.resolve())]}
    config["reference_pool_schedule"] = []
    config["terrain_course"] = {
        **config.get("terrain_course", {}),
        "enabled": True,
        "hfield_size_z": 1.4,
        "levelwalking_reference_x_shift": 0.0,
        "slopeascent_entry_shift": ENTRY_SHIFT,
        "segments": [
            {"type": "flat", "x0": -20.0, "x1": 0.0, "height": 0.0},
            {"type": "slope", "x0": 0.0, "x1": 8.0, "height0": 0.0, "slope": SLOPE_9},
            {"type": "flat", "x0": 8.0, "x1": 60.0, "height": 8.0 * SLOPE_9},
        ],
    }
    config["observation"] = {
        "localize_root": True,
        "phase_obs": "none",
        "include_foot_rel_z": True,
        "include_foot_ground_slope": True,
        "include_contact_obs": True,
        "frame_stack_prev_steps": 2,
    }
    config.setdefault("imitation", {})["pelvis_ty_vel_scale"] = 0.25
    config.setdefault("imitation", {})["pelvis_tangent_vel_scale"] = 0.5
    config.setdefault("imitation", {})["pelvis_normal_vel_scale"] = 0.18
    config.setdefault("policy", {})["ref_gate"] = 1.0
    config.setdefault("policy", {})["ref_gate_schedule"] = [{"after_steps": 0, "gate": 1.0}]
    config["resume_schedule_from_checkpoint"] = False


def configure_common(
    config: dict[str, Any], *, name: str, episode_steps: int, phase_windows: list[dict[str, int]], video_phases: list[int]
) -> None:
    config["experiment_name"] = name
    config["reset"]["episode_steps"] = int(episode_steps)
    config["reset"]["phase_windows"] = phase_windows
    config["reset"]["phase_indices"] = []
    config["reset"]["phase_index_jitter"] = 0
    config["reset"]["qpos_noise"] = 0.001
    config["reset"]["qvel_noise"] = 0.0015
    config.pop("reset_phase_schedule", None)
    config.pop("reset_phase_schedule_mode", None)
    export = config.setdefault("checkpoint_video_export", {})
    export["enabled"] = False
    export["every_steps"] = 8192
    export["video_height"] = 368
    export["video_width"] = 640
    export["video_camera_distance"] = 7.5
    export["video_camera_height"] = 1.2
    export["phase_indices"] = list(video_phases)
    config["video"] = {"phase_indices": list(video_phases)}


def imitation_reward(*, slip: float, fall: float, height: float, upright: float, vx: float) -> dict[str, float]:
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
        "pelvis_tx_vel_ref": 0.25 * float(vx),
        "pelvis_ty_vel_ref": 0.25 * float(vx),
        "pelvis_tangent_vel_ref": float(vx),
        "pelvis_normal_vel_ref": max(0.35, 0.5 * float(vx)),
        "foot_slip": float(slip),
        "tracking_energy_penalty": 0.02,
        "activation_smooth": 0.75,
        "upright": float(upright),
        "height": float(height),
        "alive": 0.0,
        "fall": float(fall),
    }


def set_stage_rewards(config: dict[str, Any], *, initial: dict[str, float], later: dict[str, float] | None = None) -> None:
    config["reward"] = dict(initial)
    config["reward_schedule_mode"] = "relative"
    config["reward_schedule"] = [{"after_steps": 0, "weights": dict(initial)}]
    if later is not None:
        config["reward_schedule"].append({"after_steps": 32768, "weights": dict(later)})


def set_sac(config: dict[str, Any], *, total_timesteps: int, alpha: float, target_entropy: float, logstd: float) -> None:
    sac = config.setdefault("sac", {})
    sac["total_timesteps"] = int(total_timesteps)
    sac["num_envs"] = 64
    sac["buffer_size"] = 220000
    sac["learning_starts"] = 1024
    sac["batch_size"] = 1024
    sac["gradient_steps"] = 2
    sac["alpha"] = float(alpha)
    sac["target_entropy"] = float(target_entropy)
    sac["actor_logstd_init"] = float(logstd)
    sac["initial_actor_action_mean"] = -0.6
    sac["symmetric_policy"] = True


def main() -> None:
    base = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    configure_base(base)
    length = reference_length_at_control_hz(REFERENCE, float(base["control"]["control_hz"]))
    video_phases = [0, min(12, length - 1), min(24, length - 1)]
    stages = [
        {
            "suffix": "stageG_up9fl035_A_h12_imit_sac",
            "episode_steps": 12,
            "total": 131072,
            "reward0": imitation_reward(slip=0.0, fall=0.0, height=0.0, upright=0.0, vx=1.0),
            "reward1": imitation_reward(slip=0.0, fall=0.5, height=0.15, upright=0.15, vx=1.25),
            "alpha": 0.12,
            "entropy": -14.0,
            "logstd": -0.8,
        },
        {
            "suffix": "stageG_up9fl035_B_h24_imit_sac",
            "episode_steps": 24,
            "total": 262144,
            "reward0": imitation_reward(slip=0.0, fall=0.5, height=0.15, upright=0.15, vx=1.25),
            "reward1": imitation_reward(slip=0.0, fall=1.0, height=0.25, upright=0.25, vx=1.5),
            "alpha": 0.1,
            "entropy": -12.0,
            "logstd": -0.9,
        },
        {
            "suffix": "stageG_up9fl035_C_h36_imit_sac",
            "episode_steps": 36,
            "total": 393216,
            "reward0": imitation_reward(slip=0.0, fall=1.0, height=0.25, upright=0.25, vx=1.5),
            "reward1": imitation_reward(slip=0.01, fall=2.0, height=0.5, upright=0.5, vx=2.0),
            "alpha": 0.09,
            "entropy": -10.0,
            "logstd": -1.0,
        },
        {
            "suffix": "stageG_up9fl035_D_h48_imit_sac",
            "episode_steps": 48,
            "total": 393216,
            "reward0": imitation_reward(slip=0.01, fall=2.0, height=0.5, upright=0.5, vx=2.0),
            "reward1": imitation_reward(slip=0.02, fall=3.0, height=0.75, upright=0.75, vx=2.5),
            "alpha": 0.085,
            "entropy": -9.0,
            "logstd": -1.1,
        },
        {
            "suffix": "stageG_up9fl035_E_h60_imit_sac",
            "episode_steps": 60,
            "total": 393216,
            "reward0": imitation_reward(slip=0.02, fall=3.0, height=0.75, upright=0.75, vx=2.5),
            "reward1": imitation_reward(slip=0.03, fall=4.0, height=1.0, upright=1.0, vx=3.0),
            "alpha": 0.08,
            "entropy": -8.5,
            "logstd": -1.15,
        },
    ]

    OUTDIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for stage in stages:
        horizon = int(stage["episode_steps"])
        phase_windows = [window_for_horizon(length, horizon)]
        config = copy.deepcopy(base)
        configure_common(config, name=stage["suffix"], episode_steps=horizon, phase_windows=phase_windows, video_phases=video_phases)
        set_stage_rewards(config, initial=stage["reward0"], later=stage["reward1"])
        set_sac(
            config,
            total_timesteps=stage["total"],
            alpha=stage["alpha"],
            target_entropy=stage["entropy"],
            logstd=stage["logstd"],
        )
        path = OUTDIR / f"muscle_2d_mjwarp_{stage['suffix']}.json"
        path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        manifest.append(
            {
                "stage": stage["suffix"],
                "config": str(path.resolve()),
                "episode_steps": horizon,
                "phase_windows": phase_windows,
                "total_timesteps": stage["total"],
                "reference_length": length,
                "initial_foot_slip_weight": stage["reward0"]["foot_slip"],
                "later_foot_slip_weight": stage["reward1"]["foot_slip"],
            }
        )
    (OUTDIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"reference_length": length, "manifest": manifest}, indent=2), flush=True)


if __name__ == "__main__":
    main()
