#!/usr/bin/env python3
"""Create staged Stage-G long-course SAC configs with imitation-heavy early stages."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "results/stageG_long_course_reference/muscle_2d_mjwarp_stageG_long_course_gated_ref_sac.json"
OUTDIR = ROOT / "configs/stageG_long_course_staged"
SUMMARY_PATH = ROOT / "results/stageG_long_course_reference/summary.json"


DEFAULT_RANGES = {
    "level_low_1": {"start": 0, "end": 117},
    "level_low_2": {"start": 117, "end": 234},
    "level_low_3": {"start": 234, "end": 351},
    "level_to_up": {"start": 351, "end": 415},
    "up_to_level": {"start": 415, "end": 510},
    "level_high": {"start": 510, "end": 558},
    "level_to_down": {"start": 558, "end": 642},
    "down_to_level": {"start": 642, "end": 748},
    "level_low_tail_1": {"start": 748, "end": 865},
}


def load_ranges() -> dict[str, dict[str, int]]:
    if SUMMARY_PATH.exists():
        summary = json.loads(SUMMARY_PATH.read_text())
        ranges = summary.get("label_ranges")
        if isinstance(ranges, dict) and ranges:
            return {
                str(name): {"start": int(value["start"]), "end": int(value["end"])}
                for name, value in ranges.items()
            }
    return copy.deepcopy(DEFAULT_RANGES)


RANGES = load_ranges()


def window(start: int, end: int) -> dict[str, int]:
    return {"start": int(start), "end": int(end)}


def label_start(name: str) -> int:
    return int(RANGES[name]["start"])


def label_end(name: str) -> int:
    return int(RANGES[name]["end"])


def data_length() -> int:
    return max(int(value["end"]) for value in RANGES.values())


def configure_common(config: dict[str, Any], *, name: str, episode_steps: int, phase_windows: list[dict[str, int]]) -> None:
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
    export["phase_indices"] = [
        label_start("level_low_1"),
        label_start("level_to_up"),
        label_start("up_to_level"),
        label_start("level_to_down"),
        label_start("down_to_level"),
        label_start("level_low_tail_1"),
    ]
    config["video"] = {"phase_indices": export["phase_indices"]}


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


def main() -> None:
    base = json.loads(BASE_CONFIG.read_text())
    OUTDIR.mkdir(parents=True, exist_ok=True)
    stages = [
        {
            "suffix": "stageG_A_h24_flat_imit_sac",
            "episode_steps": 24,
            "windows": [window(0, 160)],
            "total": 131072,
            "reward0": imitation_reward(slip=0.0, fall=0.0, height=0.0, upright=0.0, vx=1.0),
            "reward1": imitation_reward(slip=0.0, fall=1.0, height=0.25, upright=0.25, vx=1.5),
            "alpha": 0.12,
            "entropy": -14.0,
            "logstd": -0.8,
        },
        {
            "suffix": "stageG_B_h48_flat_ascent_imit_sac",
            "episode_steps": 48,
            "windows": [window(0, min(180, data_length())), window(label_start("level_to_up") - 32, label_end("level_to_up"))],
            "total": 196608,
            "reward0": imitation_reward(slip=0.0, fall=1.0, height=0.25, upright=0.25, vx=1.5),
            "reward1": imitation_reward(slip=0.01, fall=2.0, height=0.5, upright=0.5, vx=2.0),
            "alpha": 0.1,
            "entropy": -12.0,
            "logstd": -0.9,
        },
        {
            "suffix": "stageG_C_h96_ascent_high_imit_sac",
            "episode_steps": 96,
            "windows": [window(label_start("level_to_up") - 48, label_end("level_high"))],
            "total": 262144,
            "reward0": imitation_reward(slip=0.01, fall=2.0, height=0.5, upright=0.5, vx=2.0),
            "reward1": imitation_reward(slip=0.03, fall=3.0, height=0.75, upright=0.75, vx=2.5),
            "alpha": 0.09,
            "entropy": -10.0,
            "logstd": -1.0,
        },
        {
            "suffix": "stageG_D_h144_descent_imit_sac",
            "episode_steps": 144,
            "windows": [window(label_start("level_high") - 32, label_end("down_to_level"))],
            "total": 262144,
            "reward0": imitation_reward(slip=0.03, fall=3.0, height=0.75, upright=0.75, vx=2.5),
            "reward1": imitation_reward(slip=0.05, fall=4.0, height=1.0, upright=1.0, vx=3.0),
            "alpha": 0.085,
            "entropy": -9.0,
            "logstd": -1.1,
        },
        {
            "suffix": "stageG_E_h192_full_imit_sac",
            "episode_steps": 192,
            "windows": [
                window(0, min(180, data_length())),
                window(label_start("level_to_up") - 48, label_end("up_to_level")),
                window(label_start("level_to_down") - 48, label_end("down_to_level")),
                window(label_start("level_low_tail_1"), min(label_start("level_low_tail_1") + 180, data_length())),
            ],
            "total": 393216,
            "reward0": imitation_reward(slip=0.05, fall=4.0, height=1.0, upright=1.0, vx=3.0),
            "reward1": imitation_reward(slip=0.08, fall=5.0, height=1.0, upright=1.0, vx=4.0),
            "alpha": 0.08,
            "entropy": -8.0,
            "logstd": -1.2,
        },
    ]

    manifest = []
    for stage in stages:
        config = copy.deepcopy(base)
        configure_common(
            config,
            name=stage["suffix"],
            episode_steps=stage["episode_steps"],
            phase_windows=stage["windows"],
        )
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
                "config": str(path),
                "episode_steps": stage["episode_steps"],
                "phase_windows": stage["windows"],
                "total_timesteps": stage["total"],
                "initial_foot_slip_weight": stage["reward0"]["foot_slip"],
                "later_foot_slip_weight": stage["reward1"]["foot_slip"],
            }
        )
    (OUTDIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
