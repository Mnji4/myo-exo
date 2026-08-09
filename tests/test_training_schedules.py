import torch

from myo_exo_train.env.reward import (
    human_energy_speed_gate,
    human_energy_weights_for_step,
)
from myo_exo_train.rl.sac import out_of_trajectory_threshold_for_step


def test_out_of_trajectory_threshold_schedule_relative_to_run_start() -> None:
    config = {
        "myoassist_exact": {
            "out_of_trajectory_threshold": 0.8,
            "out_of_trajectory_threshold_schedule": [
                {"after_steps": 0, "threshold": 0.8},
                {"after_steps": 100, "threshold": 0.5},
                {"after_steps": 200, "threshold": 0.2},
            ],
        },
        "out_of_trajectory_threshold_schedule_mode": "relative",
    }

    assert out_of_trajectory_threshold_for_step(config, 999, 999) == 0.8
    assert out_of_trajectory_threshold_for_step(config, 1099, 999) == 0.5
    assert out_of_trajectory_threshold_for_step(config, 1199, 999) == 0.2


def test_out_of_trajectory_threshold_without_schedule_uses_base_value() -> None:
    config = {"myoassist_exact": {"out_of_trajectory_threshold": 0.35}}

    assert out_of_trajectory_threshold_for_step(config, 1000, 0) == 0.35


def test_human_energy_weight_schedule_absolute() -> None:
    config = {
        "human_energy_objective": {
            "hip_opposition_weight": 2.0,
            "hip_torque_l1_weight": 0.0,
            "weight_schedule_mode": "absolute",
            "weight_schedule": [
                {"after_steps": 100, "hip_opposition_weight": 5.0},
                {
                    "after_steps": 200,
                    "hip_opposition_weight": 10.0,
                    "hip_torque_l1_weight": 0.25,
                },
            ],
        }
    }

    assert human_energy_weights_for_step(config, 99, 0)["hip_opposition_weight"] == 2.0
    assert human_energy_weights_for_step(config, 100, 0)["hip_opposition_weight"] == 5.0
    final = human_energy_weights_for_step(config, 200, 0)
    assert final["hip_opposition_weight"] == 10.0
    assert final["hip_torque_l1_weight"] == 0.25


def test_human_energy_speed_gate_blocks_slow_gait() -> None:
    speed = torch.tensor([0.55, 0.9, 1.3])
    gate = human_energy_speed_gate(speed, min_forward_speed=0.9, softness=0.1)

    torch.testing.assert_close(gate[0], torch.tensor(0.0293122), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(gate[1], torch.tensor(0.5))
    assert float(gate[2]) > 0.98
