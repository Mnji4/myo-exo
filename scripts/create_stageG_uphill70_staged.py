#!/usr/bin/env python3
"""Create uphill-only Stage-G SAC configs using the 0.70m shifted rampascent reference."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE_REF = (
    ROOT
    / "results_old/camargo_ramp6_source_selected4/npz_ab08/"
    / "camargo_ab08_slopeascent_rampascent_ramp_6_l_01_05_myoassist_3d.npz"
)
BASE_CONFIG = ROOT / "results/stageG_uphill_reference_review/muscle_2d_mjwarp_stageG_uphill_steady18_entry_shift_0p7_review.json"
OUTDIR = ROOT / "results/stageG_uphill70_reference"
CONFIG_OUTDIR = ROOT / "configs/stageG_uphill70_staged"

SLOPE = 0.3249196962329063
CYCLES = 4
ENTRY_SHIFT = 0.70


def write_tiled_reference() -> tuple[Path, dict[str, Any]]:
    raw = np.load(SOURCE_REF, allow_pickle=True)
    metadata = copy.deepcopy(raw["metadata"].item())
    series = raw["series_data"].item()
    source_length = len(next(iter(series.values())))
    x = np.asarray(series["q_pelvis_tx"], dtype=np.float32)
    x_stride = float(x[-1] - x[0])
    z_stride = float(SLOPE * x_stride)

    tiled: dict[str, np.ndarray] = {}
    for key, value in series.items():
        arr = np.asarray(value)
        parts = []
        for cycle in range(CYCLES):
            shifted = arr.copy()
            if key == "q_pelvis_tx":
                shifted = shifted + np.asarray(cycle * x_stride, dtype=shifted.dtype)
            elif key == "q_pelvis_ty":
                shifted = shifted + np.asarray(cycle * z_stride, dtype=shifted.dtype)
            parts.append(shifted)
        tiled[key] = np.concatenate(parts, axis=0)

    cycle_ranges = {
        f"uphill_cycle_{cycle + 1}": {
            "start": int(cycle * source_length),
            "end": int((cycle + 1) * source_length),
        }
        for cycle in range(CYCLES)
    }
    metadata.update(
        {
            "source_mode": "stageG_uphill70_tiled",
            "terrain_id": "slopeascent",
            "source_label": "rampascent_uphill70_tiled",
            "data_length": int(source_length * CYCLES),
            "tile_cycles": int(CYCLES),
            "tile_source_length": int(source_length),
            "tile_x_stride": float(x_stride),
            "tile_z_stride": float(z_stride),
            "label_ranges": cycle_ranges,
            "segment_labels": list(cycle_ranges.keys()),
            "source_reference": str(SOURCE_REF.resolve()),
        }
    )

    OUTDIR.mkdir(parents=True, exist_ok=True)
    path = OUTDIR / "stageG_uphill70_tiled_myoassist_3d.npz"
    np.savez(path, metadata=metadata, series_data=tiled)

    control_hz = 30.0
    source_hz = float(metadata.get("sample_rate", 200.0))
    resampled_length = int(
        len(np.unique(np.clip(np.round(np.arange(0.0, source_length * CYCLES, source_hz / control_hz)).astype(np.int64), 0, source_length * CYCLES - 1)))
    )
    summary = {
        "npz": str(path.resolve()),
        "source_reference": str(SOURCE_REF.resolve()),
        "source_frames": int(source_length),
        "cycles": int(CYCLES),
        "raw_frames": int(source_length * CYCLES),
        "resampled_frames_at_30hz": int(resampled_length),
        "duration_sec_at_30hz": float(resampled_length) / control_hz,
        "x_start": float(tiled["q_pelvis_tx"][0]),
        "x_end": float(tiled["q_pelvis_tx"][-1]),
        "tile_x_stride": float(x_stride),
        "tile_z_stride": float(z_stride),
        "entry_shift": float(ENTRY_SHIFT),
        "slope": float(SLOPE),
    }
    (OUTDIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return path, summary


def window(start: int, end: int) -> dict[str, int]:
    return {"start": int(start), "end": int(end)}


def configure_common(
    config: dict[str, Any],
    *,
    name: str,
    episode_steps: int,
    phase_windows: list[dict[str, int]],
    video_phases: list[int],
) -> None:
    config["experiment_name"] = name
    config["reset"]["episode_steps"] = int(episode_steps)
    config["reset"]["phase_windows"] = phase_windows
    config["reset"]["phase_indices"] = []
    config["reset"]["phase_index_jitter"] = 0
    config["reset"]["qpos_noise"] = 0.0015 if episode_steps <= 48 else 0.002
    config["reset"]["qvel_noise"] = 0.002 if episode_steps <= 48 else 0.003
    config.pop("reset_phase_schedule", None)
    config.pop("reset_phase_schedule_mode", None)
    config["resume_schedule_from_checkpoint"] = False
    config["observation"] = {
        "localize_root": True,
        "phase_obs": "none",
    }
    config.setdefault("policy", {})["ref_gate"] = 1.0
    config.setdefault("policy", {})["ref_gate_schedule"] = [{"after_steps": 0, "gate": 1.0}]
    export = config.setdefault("checkpoint_video_export", {})
    export["enabled"] = True
    export["every_steps"] = 8192
    export["video_height"] = 368
    export["video_width"] = 640
    export["video_camera_distance"] = 7.5
    export["video_camera_height"] = 1.2
    export["phase_indices"] = video_phases
    config["video"] = {"phase_indices": video_phases}


def imitation_reward(*, slip: float, fall: float, height: float, upright: float, vx: float) -> dict[str, float]:
    return {
        "tracking_qpos_penalty": 16.0,
        "tracking_qvel_penalty": 0.45,
        "tracking_foot_site_penalty": 18.0,
        "tracking_swing_foot_site_penalty": 22.0,
        "tracking_swing_hip_penalty": 1.0,
        "tracking_swing_limb_penalty": 8.0,
        "tracking_activation_symmetry_penalty": 0.0,
        "tracking_future_foot_site_penalty": 8.0,
        "terminal_swing_landing_penalty": 4.0,
        "tracking_pelvis_penalty": 18.0,
        "pelvis_tx_vel_ref": float(vx),
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
    sac["buffer_size"] = 250000
    sac["learning_starts"] = 1024
    sac["batch_size"] = 1024
    sac["gradient_steps"] = 2
    sac["alpha"] = float(alpha)
    sac["target_entropy"] = float(target_entropy)
    sac["actor_logstd_init"] = float(logstd)
    sac["initial_actor_action_mean"] = -0.6
    sac["symmetric_policy"] = True


def write_configs(reference_path: Path, summary: dict[str, Any]) -> None:
    base = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    base["reference_pool"] = {"paths": [str(reference_path.resolve())]}
    base["reference_pool_schedule"] = []
    base["experiment_name"] = "muscle_2d_mjwarp_stageG_uphill70_gated_ref_sac"
    base["terrain_course"] = {
        **base.get("terrain_course", {}),
        "enabled": True,
        "hfield_size_z": 9.0,
        "slopeascent_entry_shift": float(ENTRY_SHIFT),
        "segments": [
            {"type": "flat", "x0": -20.0, "x1": 0.0, "height": 0.0},
            {"type": "slope", "x0": 0.0, "x1": 25.0, "height0": 0.0, "slope": float(SLOPE)},
            {"type": "flat", "x0": 25.0, "x1": 80.0, "height": float(25.0 * SLOPE)},
        ],
    }
    (OUTDIR / "muscle_2d_mjwarp_stageG_uphill70_gated_ref_sac.json").write_text(
        json.dumps(base, indent=2) + "\n",
        encoding="utf-8",
    )

    length = int(summary["resampled_frames_at_30hz"])
    video_phases = sorted({0, min(63, length - 1), min(126, length - 1), min(189, length - 1)})
    stage_specs = [
        {
            "suffix": "stageG_up70_A_h24_imit_sac",
            "episode_steps": 24,
            "total": 131072,
            "reward0": imitation_reward(slip=0.0, fall=0.0, height=0.0, upright=0.0, vx=1.0),
            "reward1": imitation_reward(slip=0.0, fall=1.0, height=0.25, upright=0.25, vx=1.5),
            "alpha": 0.12,
            "entropy": -14.0,
            "logstd": -0.8,
        },
        {
            "suffix": "stageG_up70_B_h48_imit_sac",
            "episode_steps": 48,
            "total": 196608,
            "reward0": imitation_reward(slip=0.0, fall=1.0, height=0.25, upright=0.25, vx=1.5),
            "reward1": imitation_reward(slip=0.01, fall=2.0, height=0.5, upright=0.5, vx=2.0),
            "alpha": 0.1,
            "entropy": -12.0,
            "logstd": -0.9,
        },
        {
            "suffix": "stageG_up70_C_h96_imit_sac",
            "episode_steps": 96,
            "total": 262144,
            "reward0": imitation_reward(slip=0.01, fall=2.0, height=0.5, upright=0.5, vx=2.0),
            "reward1": imitation_reward(slip=0.03, fall=3.0, height=0.75, upright=0.75, vx=2.5),
            "alpha": 0.09,
            "entropy": -10.0,
            "logstd": -1.0,
        },
        {
            "suffix": "stageG_up70_D_h144_imit_sac",
            "episode_steps": 144,
            "total": 262144,
            "reward0": imitation_reward(slip=0.03, fall=3.0, height=0.75, upright=0.75, vx=2.5),
            "reward1": imitation_reward(slip=0.05, fall=4.0, height=1.0, upright=1.0, vx=3.0),
            "alpha": 0.085,
            "entropy": -9.0,
            "logstd": -1.1,
        },
        {
            "suffix": "stageG_up70_E_h192_imit_sac",
            "episode_steps": 192,
            "total": 393216,
            "reward0": imitation_reward(slip=0.05, fall=4.0, height=1.0, upright=1.0, vx=3.0),
            "reward1": imitation_reward(slip=0.08, fall=5.0, height=1.0, upright=1.0, vx=4.0),
            "alpha": 0.08,
            "entropy": -8.0,
            "logstd": -1.2,
        },
    ]

    CONFIG_OUTDIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for spec in stage_specs:
        horizon = int(spec["episode_steps"])
        max_start_exclusive = max(1, length - horizon + 1)
        windows = [window(0, max_start_exclusive)]
        config = copy.deepcopy(base)
        configure_common(
            config,
            name=spec["suffix"],
            episode_steps=horizon,
            phase_windows=windows,
            video_phases=video_phases,
        )
        set_stage_rewards(config, initial=spec["reward0"], later=spec["reward1"])
        set_sac(
            config,
            total_timesteps=spec["total"],
            alpha=spec["alpha"],
            target_entropy=spec["entropy"],
            logstd=spec["logstd"],
        )
        path = CONFIG_OUTDIR / f"muscle_2d_mjwarp_{spec['suffix']}.json"
        path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        manifest.append(
            {
                "stage": spec["suffix"],
                "config": str(path.resolve()),
                "episode_steps": horizon,
                "phase_windows": windows,
                "total_timesteps": spec["total"],
                "initial_foot_slip_weight": spec["reward0"]["foot_slip"],
                "later_foot_slip_weight": spec["reward1"]["foot_slip"],
            }
        )
    (CONFIG_OUTDIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"reference": summary, "manifest": manifest}, indent=2), flush=True)


def main() -> None:
    reference_path, summary = write_tiled_reference()
    write_configs(reference_path, summary)


if __name__ == "__main__":
    main()
