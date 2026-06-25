#!/usr/bin/env python3
"""Create flat-to-uphill specialist reference and staged SAC configs."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE_REFERENCE = ROOT / "results/stageG_long_course_reference/stageG_long_flat_up_flat_down_flat_myoassist_3d.npz"
BASE_CONFIG = ROOT / "results/stageG_long_course_reference/muscle_2d_mjwarp_stageG_long_course_gated_ref_sac.json"
OUTDIR = ROOT / "results/stageG_flat_to_up_reference"
CONFIG_OUTDIR = ROOT / "configs/stageG_flat_to_up_staged"

SLICE_START = 0
SLICE_END = 248


def slice_ranges(ranges: dict[str, Any], start: int, end: int) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for name, value in ranges.items():
        item_start = int(value["start"])
        item_end = int(value["end"])
        clipped_start = max(item_start, start)
        clipped_end = min(item_end, end)
        if clipped_end <= clipped_start:
            continue
        out[str(name)] = {"start": clipped_start - start, "end": clipped_end - start}
    return out


def write_reference() -> tuple[Path, dict[str, Any]]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    raw = np.load(BASE_REFERENCE, allow_pickle=True)
    metadata = copy.deepcopy(raw["metadata"].item())
    series = raw["series_data"].item()
    start = int(SLICE_START)
    end = int(SLICE_END)
    sliced = {key: np.asarray(value)[start:end].copy() for key, value in series.items()}
    old_ranges = metadata.get("label_ranges", {})
    label_ranges = slice_ranges(old_ranges, start, end)
    metadata.update(
        {
            "source_mode": "stageG_flat_to_up_slice",
            "terrain_id": "stageG_flat_to_up",
            "source_label": "flat-to-up",
            "data_length": int(end - start),
            "slice_start": start,
            "slice_end": end,
            "label_ranges": label_ranges,
            "segment_labels": list(label_ranges.keys()),
            "source_reference": str(BASE_REFERENCE.resolve()),
        }
    )
    path = OUTDIR / "stageG_flat_to_up_myoassist_3d.npz"
    np.savez(path, metadata=metadata, series_data=sliced)
    summary = {
        "npz": str(path.resolve()),
        "config": str((OUTDIR / "muscle_2d_mjwarp_stageG_flat_to_up_gated_ref_sac.json").resolve()),
        "source_reference": str(BASE_REFERENCE.resolve()),
        "slice_start": start,
        "slice_end": end,
        "frames": int(end - start),
        "duration_sec": float(end - start) / float(metadata.get("sample_rate", 30.0)),
        "label_ranges": label_ranges,
        "x_start": float(np.asarray(sliced["q_pelvis_tx"])[0]),
        "x_end": float(np.asarray(sliced["q_pelvis_tx"])[-1]),
    }
    (OUTDIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return path, summary


def window(start: int, end: int, *, horizon: int, length: int) -> dict[str, int]:
    start = max(0, min(int(start), int(length)))
    end = max(start, min(int(end), int(length) - int(horizon) + 1))
    return {"start": start, "end": end}


def label_start(ranges: dict[str, dict[str, int]], name: str) -> int:
    return int(ranges[name]["start"])


def label_end(ranges: dict[str, dict[str, int]], name: str) -> int:
    return int(ranges[name]["end"])


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
    base = json.loads(BASE_CONFIG.read_text())
    base["reference_pool"] = {"paths": [str(reference_path.resolve())]}
    base["reference_pool_schedule"] = []
    base["experiment_name"] = "muscle_2d_mjwarp_stageG_flat_to_up_gated_ref_sac"
    (OUTDIR / "muscle_2d_mjwarp_stageG_flat_to_up_gated_ref_sac.json").write_text(
        json.dumps(base, indent=2) + "\n",
        encoding="utf-8",
    )

    ranges = summary["label_ranges"]
    length = int(summary["frames"])
    flat_end = label_end(ranges, "level_low_2_partial")
    ascent_start = label_start(ranges, "level_to_up")
    ascent_end = label_end(ranges, "level_to_up")
    up_end = label_end(ranges, "up_to_level")
    video_phases = sorted({0, 30, label_start(ranges, "level_low_2_partial"), ascent_start, ascent_end})
    stages = [
        {
            "suffix": "stageG_ftu_A_h24_flat_imit_sac",
            "episode_steps": 24,
            "windows": [window(0, flat_end, horizon=24, length=length)],
            "total": 131072,
            "reward0": imitation_reward(slip=0.0, fall=0.0, height=0.0, upright=0.0, vx=1.0),
            "reward1": imitation_reward(slip=0.0, fall=1.0, height=0.25, upright=0.25, vx=1.5),
            "alpha": 0.12,
            "entropy": -14.0,
            "logstd": -0.8,
        },
        {
            "suffix": "stageG_ftu_B_h48_flat_ascent_imit_sac",
            "episode_steps": 48,
            "windows": [
                window(0, min(flat_end, length), horizon=48, length=length),
                window(ascent_start - 32, ascent_end, horizon=48, length=length),
            ],
            "total": 196608,
            "reward0": imitation_reward(slip=0.0, fall=1.0, height=0.25, upright=0.25, vx=1.5),
            "reward1": imitation_reward(slip=0.01, fall=2.0, height=0.5, upright=0.5, vx=2.0),
            "alpha": 0.1,
            "entropy": -12.0,
            "logstd": -0.9,
        },
        {
            "suffix": "stageG_ftu_C_h96_ascent_high_imit_sac",
            "episode_steps": 96,
            "windows": [window(ascent_start - 48, up_end, horizon=96, length=length)],
            "total": 262144,
            "reward0": imitation_reward(slip=0.01, fall=2.0, height=0.5, upright=0.5, vx=2.0),
            "reward1": imitation_reward(slip=0.03, fall=3.0, height=0.75, upright=0.75, vx=2.5),
            "alpha": 0.09,
            "entropy": -10.0,
            "logstd": -1.0,
        },
        {
            "suffix": "stageG_ftu_D_h144_flat_to_up_imit_sac",
            "episode_steps": 144,
            "windows": [window(ascent_start - 64, up_end, horizon=144, length=length)],
            "total": 262144,
            "reward0": imitation_reward(slip=0.03, fall=3.0, height=0.75, upright=0.75, vx=2.5),
            "reward1": imitation_reward(slip=0.05, fall=4.0, height=1.0, upright=1.0, vx=3.0),
            "alpha": 0.085,
            "entropy": -9.0,
            "logstd": -1.1,
        },
        {
            "suffix": "stageG_ftu_E_h192_full_imit_sac",
            "episode_steps": 192,
            "windows": [window(0, up_end, horizon=192, length=length)],
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
    for stage in stages:
        config = copy.deepcopy(base)
        configure_common(
            config,
            name=stage["suffix"],
            episode_steps=stage["episode_steps"],
            phase_windows=stage["windows"],
            video_phases=video_phases,
        )
        set_stage_rewards(config, initial=stage["reward0"], later=stage["reward1"])
        set_sac(
            config,
            total_timesteps=stage["total"],
            alpha=stage["alpha"],
            target_entropy=stage["entropy"],
            logstd=stage["logstd"],
        )
        path = CONFIG_OUTDIR / f"muscle_2d_mjwarp_{stage['suffix']}.json"
        path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        manifest.append(
            {
                "stage": stage["suffix"],
                "config": str(path.resolve()),
                "episode_steps": stage["episode_steps"],
                "phase_windows": stage["windows"],
                "total_timesteps": stage["total"],
                "initial_foot_slip_weight": stage["reward0"]["foot_slip"],
                "later_foot_slip_weight": stage["reward1"]["foot_slip"],
            }
        )
    (CONFIG_OUTDIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"reference": summary, "manifest": manifest}, indent=2), flush=True)


def main() -> None:
    reference_path, summary = write_reference()
    write_configs(reference_path, summary)


if __name__ == "__main__":
    main()
