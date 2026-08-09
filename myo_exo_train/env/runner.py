"""Batched MJWarp environment for the 80-muscle locomotion model."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from myo_exo_train.env.model import (
    FOOT_SITE_NAMES, RESET_JOINTS, ROOT, TRACK_JOINTS, freejoint_root_id,
    apply_non_muscle_ctrl_override, joint_equality_specs_np, joint_id_or_none,
    model_foot_sensor_names, model_ground_force_sensor_names,
    muscle_action_mapping_mode, policy_action_to_ctrl, semantic_qpos_index, semantic_qvel_index,
    sensor_adr_or_none, site_forward_coord_tensor, site_id_or_none, terrain_forward_axis,
)
from myo_exo_train.env.observation import (
    build_policy_obs_tensor, current_terrain_height_tensor,
    current_terrain_slope_tensor, foot_obs_feature_dim, footstep_target_dim,
    policy_task_context_dim, policy_task_context_features, reference_index,
    terrain_height_for_world_x_tensor, terrain_preview_dim, terrain_slope_for_world_x_tensor,
)
from myo_exo_train.env.reference import named_weights
from myo_exo_train.env.reset import ResetMixin
from myo_exo_train.env.reward import RewardMixin
from myo_exo_train.env.torque_allocator import BatchedTorqueAllocator, DEFAULT_JOINTS

class MJWarpMuscleRunner(ResetMixin, RewardMixin):
    def __init__(
        self,
        *,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        config: dict[str, Any],
        reference: dict[str, Any],
        nworld: int,
        nconmax: int,
        njmax: int,
        seed: int,
        device: torch.device,
    ):
        self.model = model
        self.config = config
        self.reference = reference
        self.nworld = nworld
        self.device = device
        self.rng = torch.Generator(device=device)
        self.rng.manual_seed(seed)
        self.frame_skip = int(config["control"]["frame_skip"])
        mjwarp_cfg = config.get("mujoco_warp", {})
        if not isinstance(mjwarp_cfg, dict):
            mjwarp_cfg = {}
        self.capture_step_graph = bool(mjwarp_cfg.get("capture_step_graph", False))
        self.step_graph = None
        self.dt = 1.0 / float(config["control"].get("control_hz", 30.0))
        self.episode_steps = int(config["reset"]["episode_steps"])
        x_align_cfg = config.get("x_aligned_reference", {})
        x_align_sampling_cfg = x_align_cfg.get("episode_sampling", {})
        self.x_align_enabled = bool(x_align_cfg.get("enabled", False))
        self.x_align_episode_sampling_enabled = bool(
            x_align_sampling_cfg.get("enabled", False)
        )
        self.x_align_episode_probability = max(
            0.0,
            min(1.0, float(x_align_sampling_cfg.get("x_probability", 0.5))),
        )
        self.x_align_forced_mode = str(
            x_align_sampling_cfg.get("forced_mode", "") or ""
        ).lower()
        if self.x_align_forced_mode not in {"", "x", "t"}:
            raise ValueError(
                "x_aligned_reference.episode_sampling.forced_mode must be '', 'x', or 't'"
            )
        self.muscle_action_mapping = muscle_action_mapping_mode(config)
        torque_action_cfg = config.get("torque_action", {})
        self.torque_action_enabled = bool(torque_action_cfg.get("enabled", False))
        self.direct_torque_action_enabled = (
            self.torque_action_enabled
            and str(torque_action_cfg.get("mode", "allocator")).lower() == "direct"
        )
        direct_joint_names = tuple(
            str(name) for name in torque_action_cfg.get("joint_names", DEFAULT_JOINTS)
        )
        direct_torque_scale = torque_action_cfg.get(
            "torque_scale_nm", [160.0, 100.0, 180.0, 160.0, 100.0, 180.0]
        )
        if self.direct_torque_action_enabled and len(direct_joint_names) != 6:
            raise ValueError("direct torque action requires exactly six joints")
        if self.direct_torque_action_enabled and len(direct_torque_scale) != 6:
            raise ValueError("direct torque action requires six torque scales")
        self.direct_torque_dof_indices = torch.tensor(
            [semantic_qvel_index(model, name) for name in direct_joint_names],
            dtype=torch.long,
            device=device,
        )
        self.direct_torque_scale = torch.tensor(
            direct_torque_scale, dtype=torch.float32, device=device
        )
        self.direct_torque_filter_alpha = min(
            1.0, max(0.0, float(torque_action_cfg.get("filter_alpha", 1.0)))
        )
        direct_max_delta = torque_action_cfg.get(
            "max_delta_nm_per_step", [float("inf")] * 6
        )
        direct_delta_scale = torque_action_cfg.get(
            "delta_penalty_scale_nm", direct_max_delta
        )
        if len(direct_max_delta) != 6 or len(direct_delta_scale) != 6:
            raise ValueError("direct torque smoothing settings require six entries")
        self.direct_torque_max_delta = torch.tensor(
            direct_max_delta, dtype=torch.float32, device=device
        )
        self.direct_torque_delta_penalty_scale = torch.tensor(
            direct_delta_scale, dtype=torch.float32, device=device
        ).clamp_min(1.0e-6)
        self.direct_torque_delta_penalty_weight = max(
            0.0, float(torque_action_cfg.get("delta_penalty_weight", 0.0))
        )
        self.direct_torque_in_activation_obs = bool(
            torque_action_cfg.get("previous_torque_in_activation_obs", False)
        )
        self.torque_projection_penalty_weight = max(
            0.0,
            float(config.get("torque_action", {}).get("projection_error_penalty_weight", 0.0)),
        )
        self.hip_torque_measurement_enabled = any(
            name in {"exo_torque_r", "exo_torque_l"}
            for name in policy_task_context_features(config)
        )

        exo_policy_cfg = config.get("exo_policy", {})
        self.exo_policy_enabled = bool(exo_policy_cfg.get("enabled", False))
        self.exo_policy_bidirectional = bool(exo_policy_cfg.get("bidirectional", False))
        self.exo_policy_max_ctrl = max(0.0, float(exo_policy_cfg.get("max_ctrl", 0.10)))
        self.exo_policy_max_delta = max(0.0, float(exo_policy_cfg.get("max_delta_per_step", 0.02)))
        self.exo_policy_ctrl_l2_penalty = max(0.0, float(exo_policy_cfg.get("ctrl_l2_penalty", 0.0)))
        self.exo_policy_ctrl_smooth_penalty = max(0.0, float(exo_policy_cfg.get("ctrl_smooth_penalty", 0.0)))
        self.exo_policy_ctrl_balance_penalty = max(0.0, float(exo_policy_cfg.get("ctrl_balance_penalty", 0.0)))
        self.exo_policy_ctrl_balance_ema_alpha = max(
            0.0,
            min(1.0, float(exo_policy_cfg.get("ctrl_balance_ema_alpha", 0.02))),
        )
        self.exo_policy_human_effort_l2_penalty = max(0.0, float(exo_policy_cfg.get("human_effort_l2_penalty", 0.0)))
        self.exo_policy_hip_activation_l2_penalty = max(
            0.0, float(exo_policy_cfg.get("hip_activation_l2_penalty", 0.0))
        )
        self.exo_policy_hip_cocontraction_penalty = max(
            0.0, float(exo_policy_cfg.get("hip_cocontraction_penalty", 0.0))
        )
        self.exo_policy_hip_cocontraction_scale = max(
            1.0e-6, float(exo_policy_cfg.get("hip_cocontraction_scale_nm", 60.0))
        )
        human_energy_cfg = config.get("human_energy_objective", {})
        if not isinstance(human_energy_cfg, dict):
            human_energy_cfg = {}
        self.human_energy_enabled = bool(human_energy_cfg.get("enabled", False))
        self.human_energy_activation_weight = max(
            0.0, float(human_energy_cfg.get("activation_l2_weight", 0.0))
        )
        self.human_energy_hip_activation_weight = max(
            0.0, float(human_energy_cfg.get("hip_activation_l2_weight", 0.0))
        )
        self.human_energy_hip_torque_weight = max(
            0.0, float(human_energy_cfg.get("hip_torque_l1_weight", 0.0))
        )
        self.human_energy_hip_cocontraction_weight = max(
            0.0, float(human_energy_cfg.get("hip_cocontraction_weight", 0.0))
        )
        self.human_energy_hip_opposition_weight = max(
            0.0, float(human_energy_cfg.get("hip_opposition_weight", 0.0))
        )
        self.human_energy_hip_torque_scale = max(
            1.0e-6, float(human_energy_cfg.get("hip_torque_scale_nm", 60.0))
        )
        self.human_energy_hip_cocontraction_scale = max(
            1.0e-6, float(human_energy_cfg.get("hip_cocontraction_scale_nm", 60.0))
        )
        self.human_energy_tracking_threshold = float(
            human_energy_cfg.get("tracking_error_threshold", 0.8)
        )
        self.human_energy_tracking_softness = max(
            1.0e-6, float(human_energy_cfg.get("tracking_error_softness", 0.15))
        )
        self.human_energy_speed_gate_enabled = bool(
            human_energy_cfg.get("speed_gate_enabled", False)
        )
        self.human_energy_min_forward_speed = float(
            human_energy_cfg.get("min_forward_speed_mps", 0.0)
        )
        self.human_energy_speed_gate_softness = max(
            1.0e-6, float(human_energy_cfg.get("speed_gate_softness_mps", 0.1))
        )
        self.human_energy_lateral_threshold = float(
            human_energy_cfg.get("lateral_drift_threshold_m", 0.0)
        )
        self.human_energy_lateral_softness = max(
            1.0e-6, float(human_energy_cfg.get("lateral_drift_softness_m", 0.03))
        )
        self.human_energy_quality_gate_penalty_weight = max(
            0.0, float(human_energy_cfg.get("quality_gate_penalty_weight", 0.0))
        )
        self.human_energy_hip_antagonist_excess_weight = max(
            0.0, float(human_energy_cfg.get("hip_antagonist_excess_weight", 0.0))
        )
        self.human_energy_hip_antagonist_excess_scale = max(
            1.0e-6, float(human_energy_cfg.get("hip_antagonist_excess_scale_nm", 120.0))
        )
        self.human_energy_hip_net_torque_error_weight = max(
            0.0, float(human_energy_cfg.get("hip_net_torque_error_weight", 0.0))
        )
        self.human_energy_hip_net_torque_error_scale = max(
            1.0e-6, float(human_energy_cfg.get("hip_net_torque_error_scale_nm", 30.0))
        )
        self.human_energy_hip_net_torque_target: torch.Tensor | None = None
        joint_names = human_energy_cfg.get(
            "joint_cocontraction_names",
            [
                "hip_flexion_r", "knee_angle_r", "ankle_angle_r",
                "hip_flexion_l", "knee_angle_l", "ankle_angle_l",
            ],
        )
        self.human_energy_joint_cocontraction_names = [str(name) for name in joint_names]
        self.human_energy_joint_cocontraction_weight = max(
            0.0, float(human_energy_cfg.get("joint_cocontraction_weight", 0.0))
        )
        self.human_energy_joint_cocontraction_measure = bool(
            human_energy_cfg.get(
                "joint_cocontraction_measure",
                self.human_energy_joint_cocontraction_weight > 0.0,
            )
        )
        self.human_energy_joint_cocontraction_force_mode = str(
            human_energy_cfg.get("joint_cocontraction_force_mode", "total")
        ).lower()
        if self.human_energy_joint_cocontraction_force_mode not in {"active", "passive", "total"}:
            raise ValueError(
                "joint_cocontraction_force_mode must be active, passive, or total"
            )
        self.human_energy_joint_cocontraction_detailed_measure = bool(
            human_energy_cfg.get("joint_cocontraction_detailed_measure", False)
        )
        self.human_energy_joint_cocontraction_use_tracking_gate = bool(
            human_energy_cfg.get("joint_cocontraction_use_tracking_gate", False)
        )
        self.human_energy_joint_cocontraction_max_weight = max(
            0.0, float(human_energy_cfg.get("joint_cocontraction_max_weight", 0.0))
        )
        self.human_energy_joint_cocontraction_scale = max(
            1.0e-6, float(human_energy_cfg.get("joint_cocontraction_scale_nm", 60.0))
        )
        self.human_energy_constraint_failure_penalty = max(
            0.0, float(human_energy_cfg.get("constraint_failure_penalty", 0.0))
        )
        torque_target_path = str(human_energy_cfg.get("hip_net_torque_target_path", "") or "")
        if torque_target_path:
            target_path = Path(torque_target_path).expanduser()
            if not target_path.is_absolute():
                target_path = Path.cwd() / target_path
            target_key = str(human_energy_cfg.get("hip_net_torque_target_key", "hip_net_torque_nm"))
            with np.load(target_path, allow_pickle=False) as target_data:
                if target_key not in target_data:
                    raise KeyError(f"missing {target_key!r} in hip torque target {target_path}")
                target_values = np.asarray(target_data[target_key], dtype=np.float32)
            if target_values.ndim != 2 or target_values.shape[1] != 2:
                raise ValueError(
                    f"hip torque target must have shape [phase, 2], got {target_values.shape}"
                )
            self.human_energy_hip_net_torque_target = torch.tensor(
                target_values, dtype=torch.float32, device=device
            )
        human_energy_needs_hip_measurement = self.human_energy_enabled and any(
            weight > 0.0
            for weight in (
                self.human_energy_hip_activation_weight,
                self.human_energy_hip_torque_weight,
                self.human_energy_hip_cocontraction_weight,
                self.human_energy_hip_opposition_weight,
                self.human_energy_hip_antagonist_excess_weight,
                self.human_energy_hip_net_torque_error_weight,
            )
        )
        self.hip_torque_measurement_enabled = bool(
            self.hip_torque_measurement_enabled
            or self.exo_policy_hip_activation_l2_penalty > 0.0
            or self.exo_policy_hip_cocontraction_penalty > 0.0
            or human_energy_needs_hip_measurement
        )
        self.safe_pelvis_height = float(config["reset"]["safe_pelvis_height"])
        self.full_state_reset_only = bool(config["reset"].get("full_state_only", False))
        self.initial_activation = float(config["reset"].get("initial_activation", 0.05))
        initial_activation_range = config["reset"].get("initial_activation_range", [])
        if isinstance(initial_activation_range, list) and len(initial_activation_range) >= 2:
            self.initial_activation_low = float(initial_activation_range[0])
            self.initial_activation_high = float(initial_activation_range[1])
        else:
            self.initial_activation_low = self.initial_activation
            self.initial_activation_high = self.initial_activation
        self.reset_qpos_noise = float(config["reset"].get("qpos_noise", 0.0))
        self.reset_qvel_noise = float(config["reset"].get("qvel_noise", 0.0))
        recovery_cfg = config.get("recovery_reset", {})
        self.recovery_reset_enabled = bool(recovery_cfg.get("enabled", False))
        self.recovery_reset_probability = float(recovery_cfg.get("reset_probability", 0.0))
        self.recovery_collect_probability = float(recovery_cfg.get("collect_probability", 1.0))
        self.recovery_bank_capacity = max(0, int(recovery_cfg.get("capacity", 0) or 0))
        segmented_cfg = recovery_cfg.get("segmented_retention", {})
        self.recovery_segmented_retention_enabled = bool(
            segmented_cfg.get("enabled", False)
        )
        self.recovery_segment_sampling_mode = str(
            segmented_cfg.get("sampling_mode", "uniform")
        ).lower()
        self.recovery_dedicated_phase0_worlds = max(
            0, int(segmented_cfg.get("dedicated_phase0_worlds", 0) or 0)
        )
        if self.recovery_dedicated_phase0_worlds >= int(nworld):
            raise ValueError(
                "dedicated_phase0_worlds must be smaller than nworld"
            )
        if self.recovery_segment_sampling_mode not in {
            "uniform",
            "world_partition",
        }:
            raise ValueError(
                "recovery segmented sampling_mode must be uniform or "
                "world_partition"
            )
        self.recovery_segment_boundaries = [
            float(value) for value in segmented_cfg.get("boundaries", [])
        ]
        segment_count = len(self.recovery_segment_boundaries) + 1
        configured_capacities = [
            int(value) for value in segmented_cfg.get("capacities", [])
        ]
        if self.recovery_segmented_retention_enabled:
            if configured_capacities:
                if len(configured_capacities) != segment_count:
                    raise ValueError(
                        "recovery_reset.segmented_retention.capacities must "
                        "have len(boundaries)+1 entries"
                    )
                if any(value <= 0 for value in configured_capacities):
                    raise ValueError(
                        "segmented recovery capacities must all be positive"
                    )
                if sum(configured_capacities) != self.recovery_bank_capacity:
                    raise ValueError(
                        "segmented recovery capacities must sum to capacity"
                    )
                self.recovery_segment_capacities = configured_capacities
            else:
                base = self.recovery_bank_capacity // segment_count
                remainder = self.recovery_bank_capacity % segment_count
                self.recovery_segment_capacities = [
                    base + (1 if index < remainder else 0)
                    for index in range(segment_count)
                ]
        else:
            self.recovery_segment_capacities = [self.recovery_bank_capacity]
        self.recovery_bootstrap_banks = list(
            recovery_cfg.get("bootstrap_banks", [])
        )
        self.recovery_min_bank_size = max(1, int(recovery_cfg.get("min_bank_size", 1) or 1))
        self.recovery_min_height = float(recovery_cfg.get("min_height_above_terrain", self.safe_pelvis_height))
        self.recovery_max_height = float(recovery_cfg.get("max_height_above_terrain", 0.8))
        self.recovery_phase_start = int(recovery_cfg.get("phase_start", 0))
        self.recovery_phase_end = int(recovery_cfg.get("phase_end", int(reference["length"])))
        self.recovery_min_episode_steps = max(0, int(recovery_cfg.get("min_episode_steps", 0) or 0))
        self.recovery_survival_delay_steps = max(0, int(recovery_cfg.get("survival_delay_steps", 0) or 0))
        self.recovery_max_abs_lateral_drift = float(recovery_cfg.get("max_abs_lateral_drift", 0.0) or 0.0)
        self.recovery_min_forward_position = float(
            recovery_cfg.get("min_forward_position", -float("inf"))
        )
        self.recovery_max_forward_position = float(
            recovery_cfg.get("max_forward_position", float("inf"))
        )
        self.recovery_min_forward_velocity = float(
            recovery_cfg.get("min_forward_velocity", -float("inf"))
        )
        self.recovery_phase_windows = self.parse_phase_windows(recovery_cfg.get("phase_windows", []), int(reference["length"]))
        prioritized_cfg = recovery_cfg.get("prioritized_sampling", {})
        self.recovery_priority_enabled = bool(prioritized_cfg.get("enabled", False))
        self.recovery_priority_x_min = float(prioritized_cfg.get("x_min", -5.0))
        self.recovery_priority_x_max = float(prioritized_cfg.get("x_max", 36.0))
        self.recovery_priority_bin_width = max(
            1.0e-3, float(prioritized_cfg.get("bin_width", 0.75))
        )
        self.recovery_priority_fraction = min(
            1.0, max(0.0, float(prioritized_cfg.get("priority_fraction", 0.70)))
        )
        self.recovery_priority_min_progress = max(
            0.0,
            float(
                prioritized_cfg.get(
                    "min_progress_m", self.recovery_priority_bin_width
                )
            ),
        )
        self.recovery_priority_prior_attempts = max(
            1.0, float(prioritized_cfg.get("prior_attempts", 4.0))
        )
        self.recovery_priority_prior_successes = min(
            self.recovery_priority_prior_attempts,
            max(0.0, float(prioritized_cfg.get("prior_successes", 2.0))),
        )
        self.recovery_priority_bin_count = max(
            1,
            int(
                np.ceil(
                    (self.recovery_priority_x_max - self.recovery_priority_x_min)
                    / self.recovery_priority_bin_width
                )
            ),
        )
        self.recovery_priority_attempts = torch.zeros(
            self.recovery_priority_bin_count, dtype=torch.float32, device=device
        )
        self.recovery_priority_successes = torch.zeros_like(
            self.recovery_priority_attempts
        )
        self.recovery_priority_reset_counts = torch.zeros(
            self.recovery_priority_bin_count, dtype=torch.long, device=device
        )
        self.recovery_priority_last_sample_count = 0
        offline_recovery_cfg = config.get("offline_recovery_reset", {})
        self.offline_recovery_enabled = bool(offline_recovery_cfg.get("enabled", False))
        self.offline_recovery_path = str(offline_recovery_cfg.get("path", "") or "")
        self.offline_recovery_probability = float(offline_recovery_cfg.get("reset_probability", 0.0))
        self.offline_recovery_min_bank_size = max(1, int(offline_recovery_cfg.get("min_bank_size", 1) or 1))
        self.offline_recovery_fixed_index = int(
            offline_recovery_cfg.get("fixed_index", -1)
        )
        recovery_reward_cfg = config.get("recovery_reward", {})
        self.recovery_reward_enabled = bool(recovery_reward_cfg.get("enabled", False))
        self.recovery_reward_horizon_steps = max(0, int(recovery_reward_cfg.get("horizon_steps", 0) or 0))
        self.recovery_reward_weights = {
            str(k): float(v) for k, v in recovery_reward_cfg.get("weights", {}).items()
        }
        self.phase_start = int(config["reset"].get("phase_start", 0))
        self.phase_end = int(config["reset"].get("phase_end", 0) or reference["length"])
        self.phase_choices = self.build_phase_choices(
            config["reset"].get("phase_windows", []),
            config["reset"].get("phase_indices", []),
            int(config["reset"].get("phase_index_jitter", 0) or 0),
            int(reference["length"]),
        )
        self.reward_weights = {k: float(v) for k, v in config["reward"].items()}
        route_reward_cfg = config.get("route_reward_profiles", {})
        self.route_reward_profiles = [
            {
                "name": str(item.get("name", f"expert{index}")),
                "weights": {
                    str(key): float(value)
                    for key, value in item.get("weights", {}).items()
                },
            }
            for index, item in enumerate(route_reward_cfg.get("profiles", []))
        ]
        self.route_reward_boundaries = [
            float(value) for value in route_reward_cfg.get("boundaries", [])
        ]
        self.route_reward_profiles_enabled = bool(
            route_reward_cfg.get("enabled", False)
        )
        if self.route_reward_profiles_enabled:
            if len(self.route_reward_profiles) < 2:
                raise ValueError(
                    "route_reward_profiles requires at least two profiles"
                )
            if len(self.route_reward_boundaries) != (
                len(self.route_reward_profiles) - 1
            ):
                raise ValueError(
                    "route_reward_profiles boundaries must have "
                    "len(profiles)-1 entries"
                )
        self.route_reward_env_boundaries: torch.Tensor | None = None
        self.reward_mode = str(config.get("reward_mode", "default") or "default").lower()
        self.pose_weights = named_weights(config, "joint_pose_weights", TRACK_JOINTS).to(device)
        self.vel_weights = named_weights(config, "joint_vel_weights", TRACK_JOINTS).to(device)
        exact_cfg = config.get("myoassist_exact", {})
        gait_cycle_cfg = config.get("reward_gait_cycle", {})
        self.gait_cycle_enabled = bool(gait_cycle_cfg.get("enabled", False))
        self.gait_cycle_phase_start = int(
            gait_cycle_cfg.get("phase_start", 0)
        )
        self.gait_cycle_phase_end = int(
            gait_cycle_cfg.get("phase_end", int(reference["length"]))
        )
        self.gait_cycle_target_steps = max(
            2,
            int(gait_cycle_cfg.get("target_cycle_steps", 34)),
        )
        self.gait_half_cycle_target_steps = max(
            1,
            int(
                gait_cycle_cfg.get(
                    "target_half_cycle_steps",
                    self.gait_cycle_target_steps // 2,
                )
            ),
        )
        self.gait_cycle_tolerance_steps = max(
            1.0,
            float(gait_cycle_cfg.get("cycle_tolerance_steps", 6.0)),
        )
        self.gait_half_cycle_tolerance_steps = max(
            1.0,
            float(gait_cycle_cfg.get("half_cycle_tolerance_steps", 5.0)),
        )
        self.gait_half_cycle_balance_tolerance_steps = max(
            1.0e-6,
            float(
                gait_cycle_cfg.get(
                    "half_cycle_balance_tolerance_steps",
                    2.0,
                )
            ),
        )
        self.gait_cycle_min_landing_interval_steps = max(
            1,
            int(gait_cycle_cfg.get("min_landing_interval_steps", 8)),
        )
        self.gait_cycle_max_half_cycle_steps = max(
            self.gait_cycle_min_landing_interval_steps + 1,
            int(gait_cycle_cfg.get("max_half_cycle_steps", 26)),
        )
        self.gait_cycle_pose_scale = max(
            1.0e-6,
            float(gait_cycle_cfg.get("pose_scale", 0.30)),
        )
        self.gait_cycle_velocity_scale = max(
            1.0e-6,
            float(gait_cycle_cfg.get("velocity_scale", 3.0)),
        )
        self.gait_cycle_activation_scale = max(
            1.0e-6,
            float(gait_cycle_cfg.get("activation_scale", 0.30)),
        )
        self.gait_cycle_event_hold_steps = max(
            1,
            int(gait_cycle_cfg.get("event_hold_steps", 1)),
        )
        self.gait_cycle_event_hold_normalize = bool(
            gait_cycle_cfg.get("event_hold_normalize", True)
        )
        self.gait_dense_half_cycle_enabled = bool(
            gait_cycle_cfg.get("dense_half_cycle_enabled", False)
        )
        self.gait_dense_half_cycle_steps = max(
            1,
            int(
                gait_cycle_cfg.get(
                    "dense_half_cycle_steps",
                    self.gait_half_cycle_target_steps,
                )
            ),
        )
        self.gait_dense_half_cycle_force_scale_n = max(
            1.0e-6,
            float(gait_cycle_cfg.get("dense_half_cycle_force_scale_n", 300.0)),
        )
        self.gait_sequence_half_cycle_enabled = bool(
            gait_cycle_cfg.get("sequence_half_cycle_enabled", False)
        )
        self.gait_sequence_half_cycle_points = max(
            3, int(gait_cycle_cfg.get("sequence_half_cycle_points", 8))
        )
        self.gait_sequence_half_cycle_capacity = max(
            self.gait_cycle_max_half_cycle_steps + 2,
            int(
                gait_cycle_cfg.get(
                    "sequence_half_cycle_capacity",
                    self.gait_cycle_max_half_cycle_steps + 2,
                )
            ),
        )
        self.gait_sequence_half_cycle_force_scale_n = max(
            1.0e-6,
            float(gait_cycle_cfg.get("sequence_half_cycle_force_scale_n", 200.0)),
        )
        phase_force_cfg = gait_cycle_cfg.get("phase_force_target", {})
        if not isinstance(phase_force_cfg, dict):
            raise TypeError("reward_gait_cycle.phase_force_target must be an object")
        self.gait_phase_force_target_enabled = bool(
            phase_force_cfg.get("enabled", False)
        )
        self.gait_phase_force_target_scale_n = max(
            1.0e-6,
            float(phase_force_cfg.get("scale_n", 200.0)),
        )
        self.gait_phase_force_target = torch.zeros(
            (1, 2), dtype=torch.float32, device=device
        )
        if self.gait_phase_force_target_enabled:
            target_path = Path(str(phase_force_cfg.get("path", ""))).expanduser()
            if not target_path.is_absolute():
                target_path = ROOT / target_path
            if not target_path.is_file():
                raise FileNotFoundError(f"missing gait phase force target: {target_path}")
            with np.load(target_path, allow_pickle=False) as payload:
                target = np.asarray(
                    payload["symmetric_target_side_force_n"], dtype=np.float32
                )
            if target.ndim != 2 or target.shape[1] != 2 or target.shape[0] < 2:
                raise ValueError(
                    "gait phase force target must have shape [period_steps, 2]"
                )
            self.gait_phase_force_target = torch.tensor(
                target, dtype=torch.float32, device=device
            )
        self.gait_stance_balance_enabled = bool(
            gait_cycle_cfg.get("stance_balance_enabled", False)
        )
        self.gait_stance_min_steps = max(
            1,
            int(gait_cycle_cfg.get("stance_min_steps", 3)),
        )
        self.gait_stance_impulse_relative_scale = max(
            1.0e-6,
            float(gait_cycle_cfg.get("stance_impulse_relative_scale", 0.15)),
        )
        self.gait_stance_duration_scale_steps = max(
            1.0e-6,
            float(gait_cycle_cfg.get("stance_duration_scale_steps", 4.0)),
        )
        self.gait_stance_peak_force_relative_scale = max(
            1.0e-6,
            float(gait_cycle_cfg.get("stance_peak_force_relative_scale", 0.20)),
        )
        self.gait_stance_event_hold_steps = max(
            1,
            int(gait_cycle_cfg.get("stance_event_hold_steps", 8)),
        )
        exact_qpos_weights = exact_cfg.get(
            "qpos_imitation_rewards",
            {
                "pelvis_ty": 0.1,
                "pelvis_tilt": 1.0,
                "hip_flexion_r": 0.2,
                "knee_angle_r": 1.0,
                "ankle_angle_r": 0.2,
                "hip_flexion_l": 0.2,
                "knee_angle_l": 1.0,
                "ankle_angle_l": 0.2,
            },
        )
        exact_qvel_weights = exact_cfg.get(
            "qvel_imitation_rewards",
            {
                "pelvis_ty": 0.1,
                "pelvis_tilt": 0.1,
                "hip_flexion_r": 0.2,
                "knee_angle_r": 0.1,
                "ankle_angle_r": 0.1,
                "hip_flexion_l": 0.2,
                "knee_angle_l": 0.1,
                "ankle_angle_l": 0.1,
            },
        )
        self.myoassist_qpos_weights = torch.tensor(
            [float(exact_qpos_weights.get(name, 0.0)) for name in TRACK_JOINTS],
            dtype=torch.float32,
            device=device,
        )
        self.myoassist_qvel_weights = torch.tensor(
            [float(exact_qvel_weights.get(name, 0.0)) for name in TRACK_JOINTS],
            dtype=torch.float32,
            device=device,
        )
        reference_error_weights = exact_cfg.get(
            "reference_joint_error_weights",
            {},
        )
        reference_velocity_error_weights = exact_cfg.get(
            "reference_joint_velocity_error_weights",
            reference_error_weights,
        )
        self.reference_joint_error_weights = torch.tensor(
            [
                float(reference_error_weights.get(name, 0.0))
                for name in TRACK_JOINTS
            ],
            dtype=torch.float32,
            device=device,
        )
        self.reference_joint_velocity_error_weights = torch.tensor(
            [
                float(reference_velocity_error_weights.get(name, 0.0))
                for name in TRACK_JOINTS
            ],
            dtype=torch.float32,
            device=device,
        )
        self.reference_joint_error_huber_delta = max(
            float(exact_cfg.get("reference_joint_error_huber_delta", 0.15)),
            1.0e-6,
        )
        self.reference_joint_velocity_error_huber_delta = max(
            float(
                exact_cfg.get(
                    "reference_joint_velocity_error_huber_delta",
                    2.0,
                )
            ),
            1.0e-6,
        )
        self.myoassist_dense_weights = {
            "forward_reward": float(exact_cfg.get("forward_reward", 1.0)),
            "muscle_activation_penalty": float(exact_cfg.get("muscle_activation_penalty", 0.1)),
            "muscle_activation_diff_penalty": float(exact_cfg.get("muscle_activation_diff_penalty", 0.1)),
            "joint_constraint_force_penalty": float(exact_cfg.get("joint_constraint_force_penalty", 1.0)),
            "foot_force_penalty": float(exact_cfg.get("foot_force_penalty", 0.5)),
            "qpos_imitation_rewards": float(sum(float(v) for v in exact_qpos_weights.values())),
            "qvel_imitation_rewards": float(sum(float(v) for v in exact_qvel_weights.values())),
            "end_effector_imitation_reward": float(exact_cfg.get("end_effector_imitation_reward", 0.0)),
        }
        self.full_state_imitation_enabled = bool(exact_cfg.get("full_state_imitation_enabled", False))
        self.full_qpos_scale = max(float(exact_cfg.get("full_qpos_scale", 0.35)), 1e-6)
        self.full_qvel_scale = max(float(exact_cfg.get("full_qvel_scale", 6.0)), 1e-6)
        self.full_qpos_mask = torch.ones(int(model.nq), dtype=torch.bool, device=device)
        self.full_qvel_mask = torch.ones(int(model.nv), dtype=torch.bool, device=device)
        root_jid = freejoint_root_id(model)
        self.root_qpos_adr = -1 if root_jid is None else int(model.jnt_qposadr[root_jid])
        self.root_dof_adr = -1 if root_jid is None else int(model.jnt_dofadr[root_jid])
        self.forward_axis = terrain_forward_axis(config)
        if bool(exact_cfg.get("full_state_exclude_root_translation", True)):
            if root_jid is not None:
                qadr = int(model.jnt_qposadr[root_jid])
                dadr = int(model.jnt_dofadr[root_jid])
                self.full_qpos_mask[qadr : qadr + 3] = False
                self.full_qvel_mask[dadr : dadr + 3] = False
            else:
                # Planar models represent root translation as scalar slide joints,
                # rather than a freejoint. Exclude those coordinates as well.
                for joint_name in ("pelvis_tx", "pelvis_ty", "pelvis_tz"):
                    joint_id = joint_id_or_none(model, joint_name)
                    if joint_id is None:
                        continue
                    self.full_qpos_mask[int(model.jnt_qposadr[joint_id])] = False
                    self.full_qvel_mask[int(model.jnt_dofadr[joint_id])] = False
        if bool(exact_cfg.get("full_state_exclude_root_quat", False)):
            if root_jid is not None:
                qadr = int(model.jnt_qposadr[root_jid])
                self.full_qpos_mask[qadr + 3 : qadr + 7] = False
                self.full_qvel_mask[int(model.jnt_dofadr[root_jid]) + 3 : int(model.jnt_dofadr[root_jid]) + 6] = False
        full_qpos_weight = float(exact_cfg.get("full_qpos_imitation_rewards", 0.0))
        full_qvel_weight = float(exact_cfg.get("full_qvel_imitation_rewards", 0.0))
        if self.full_state_imitation_enabled and full_qpos_weight != 0.0:
            self.myoassist_dense_weights["full_qpos_imitation_rewards"] = full_qpos_weight
        if self.full_state_imitation_enabled and full_qvel_weight != 0.0:
            self.myoassist_dense_weights["full_qvel_imitation_rewards"] = full_qvel_weight
        self.root_orientation_scale = max(float(exact_cfg.get("root_orientation_scale", 0.25)), 1e-6)
        self.root_angvel_scale = max(float(exact_cfg.get("root_angvel_scale", 4.0)), 1e-6)
        self.root_xy_position_scale = max(float(exact_cfg.get("root_xy_position_scale", 0.25)), 1e-6)
        self.lateral_velocity_scale = max(float(exact_cfg.get("lateral_velocity_scale", 0.5)), 1e-6)
        self.lateral_drift_scale = max(float(exact_cfg.get("lateral_drift_scale", 0.5)), 1e-6)
        self.foot_site_local_mimic_scale = max(float(exact_cfg.get("foot_site_local_mimic_scale", 0.15)), 1e-6)
        flat_approach_cfg = config.get("reward_flat_approach", {})
        self.flat_approach_enabled = bool(flat_approach_cfg.get("enabled", False))
        stair_entries = [
            float(segment.get("x0", 0.0))
            for segment in list(config.get("terrain_course", {}).get("segments", []))
            if str(segment.get("type", "flat")) == "stairs_box" and float(segment.get("direction", 1.0)) >= 0.0
        ]
        default_entry_x = min(stair_entries) if stair_entries else 0.0
        self.flat_approach_entry_x = float(flat_approach_cfg.get("entry_x", default_entry_x))
        self.flat_approach_start_x = float(flat_approach_cfg.get("start_x", 0.0))
        self.flat_approach_active_back = float(flat_approach_cfg.get("active_back", 0.05))
        self.flat_approach_active_ahead = float(flat_approach_cfg.get("active_ahead", 0.10))
        self.flat_approach_target_velocity = max(
            float(flat_approach_cfg.get("target_velocity", exact_cfg.get("target_velocity", 1.25))),
            1e-6,
        )
        default_approach_len = max(self.flat_approach_entry_x - self.flat_approach_start_x, 1e-6)
        self.flat_approach_distance_scale = max(
            float(flat_approach_cfg.get("distance_scale", default_approach_len)),
            1e-6,
        )
        self.root_xy_drift_done_threshold = float(
            exact_cfg.get(
                "root_xy_drift_done_threshold",
                exact_cfg.get("lateral_drift_done_threshold", 0.0),
            )
            or 0.0
        )
        for term_name in [
            "root_xy_position_reward",
            "root_orientation_reward",
            "root_angvel_penalty",
            "lateral_vel_penalty",
            "lateral_drift_penalty",
            "foot_site_local_mimic_reward",
            "future_foot_site_local_mimic_reward",
            "keypoint_position_imitation_reward",
            "root_forward_velocity_imitation_reward",
            "root_forward_velocity_shortfall_penalty",
            "root_forward_velocity_overspeed_penalty",
            "reference_joint_error_penalty",
            "reference_joint_velocity_error_penalty",
            "footstep_target_reward",
            "footstep_landing_reward",
            "footstep_clearance_reward",
            "foot_contact_phase_reward",
            "foot_rollover_toe_first_reward",
            "foot_rollover_heel_follow_reward",
            "foot_rollover_heel_loading_penalty",
            "foot_rollover_heel_first_penalty",
            "foot_rollover_missing_heel_penalty",
            "foot_lateral_target_penalty",
            "foot_toe_in_penalty",
            "knee_valgus_penalty",
            "foot_lateral_gap_penalty",
            "flat_approach_progress_reward",
            "flat_approach_velocity_reward",
            "flat_approach_shortfall_penalty",
            "stair_contact_step_progress_reward",
            "stair_step_ahead_reward",
            "stair_contact_presence_reward",
            "stair_pelvis_step_progress_reward",
            "stair_step_gap_penalty",
            "stair_support_height_reward",
            "stair_support_height_penalty",
            "stair_foot_tread_target_reward",
            "stair_same_step_contact_penalty",
            "stair_step_separation_reward",
            "stair_pelvis_contact_lag_penalty",
            "stair_pelvis_drop_penalty",
            "stair_foot_tread_overshoot_penalty",
            "stair_top_platform_pelvis_reward",
            "stair_top_platform_contact_reward",
            "stair_top_platform_height_reward",
            "stair_top_platform_height_penalty",
            "stair_top_platform_forward_reward",
            "stair_top_platform_shortfall_penalty",
            "stair_trailing_foot_forward_reward",
            "stair_trailing_foot_clearance_reward",
            "stair_trailing_foot_land_ready_reward",
            "stair_trailing_foot_contact_reward",
            "stair_trailing_foot_lag_penalty",
            "nearest_trajectory_reward",
            "forward_velocity_reward",
            "forward_shortfall_penalty",
            "ramp_progress_reward",
            "ramp_progress_tracking_reward",
            "ramp_progress_lag_penalty",
            "ramp_velocity_reward",
            "ramp_step_reward",
            "xalign_phase_lag_penalty",
            "gait_cycle_cadence_reward",
            "gait_half_cycle_cadence_reward",
            "gait_half_cycle_balance_reward",
            "gait_cycle_pose_reward",
            "gait_half_cycle_pose_reward",
            "gait_cycle_velocity_reward",
            "gait_half_cycle_velocity_reward",
            "gait_cycle_activation_reward",
            "gait_half_cycle_activation_reward",
            "gait_dense_half_cycle_pose_reward",
            "gait_dense_half_cycle_velocity_reward",
            "gait_dense_half_cycle_activation_reward",
            "gait_dense_half_cycle_force_balance_penalty",
            "gait_sequence_half_cycle_pose_reward",
            "gait_sequence_half_cycle_velocity_reward",
            "gait_sequence_half_cycle_activation_reward",
            "gait_sequence_half_cycle_force_reward",
            "gait_phase_force_target_penalty",
            "gait_stance_impulse_balance_reward",
            "gait_stance_duration_balance_reward",
            "gait_stance_peak_force_balance_reward",
            "gait_stance_impulse_balance_penalty",
            "gait_stance_peak_force_balance_penalty",
            "gait_alternation_reward",
            "gait_missing_landing_penalty",
        ]:
            term_weight = float(exact_cfg.get(term_name, 0.0))
            if term_weight != 0.0:
                self.myoassist_dense_weights[term_name] = term_weight
        self.myoassist_target_velocity = float(exact_cfg.get("target_velocity", config.get("reward_tangent_velocity_command", {}).get("target", 1.25)))
        self.scale_reference_velocity_to_target = bool(
            exact_cfg.get("scale_reference_velocity_to_target", True)
        )
        self.root_forward_velocity_imitation_scale = max(
            float(exact_cfg.get("root_forward_velocity_imitation_scale", 0.35)),
            1.0e-6,
        )
        self.root_forward_velocity_overspeed_margin = max(
            0.0,
            float(exact_cfg.get("root_forward_velocity_overspeed_margin", 0.1)),
        )
        self.root_forward_velocity_overspeed_scale = max(
            float(exact_cfg.get("root_forward_velocity_overspeed_scale", 0.2)),
            1.0e-6,
        )
        self.root_forward_velocity_overspeed_max_ratio_sq = max(
            0.0,
            float(
                exact_cfg.get(
                    "root_forward_velocity_overspeed_max_ratio_sq",
                    9.0,
                )
            ),
        )
        self.forward_velocity_error_mode = str(exact_cfg.get("forward_velocity_error_mode", "symmetric")).lower()
        self.myoassist_out_of_trajectory_threshold = float(exact_cfg.get("out_of_trajectory_threshold", 0.6))
        self.foot_x_scale = float(config.get("imitation", {}).get("foot_x_scale", 0.18))
        self.foot_z_scale = float(config.get("imitation", {}).get("foot_z_scale", 0.05))
        self.reference_future_steps = max(0, int(config.get("imitation", {}).get("reference_future_steps", 0) or 0))
        reference_curriculum = config.get("reference_curriculum", {})
        self.reference_phase_lead_steps = int(reference_curriculum.get("current_phase_lead_steps", reference_curriculum.get("phase_lead_steps", 0)) or 0)
        self.reference_phase_tolerance_steps = int(
            reference_curriculum.get("current_phase_tolerance_steps", reference_curriculum.get("phase_tolerance_steps", 0)) or 0
        )
        self.reference_swing_exaggeration_scale = float(
            reference_curriculum.get(
                "current_swing_exaggeration_scale",
                reference_curriculum.get("swing_exaggeration_scale", 1.0),
            )
        )
        wp.init()
        self.warp_model = mjw.put_model(model)
        self.warp_data = mjw.put_data(model, data, nworld=nworld, nconmax=nconmax, njmax=njmax)
        self.qpos = wp.to_torch(self.warp_data.qpos)
        self.qvel = wp.to_torch(self.warp_data.qvel)
        self.act = wp.to_torch(self.warp_data.act)
        self.ctrl = wp.to_torch(self.warp_data.ctrl)
        self.actuator_length = wp.to_torch(self.warp_data.actuator_length)
        self.actuator_velocity = wp.to_torch(self.warp_data.actuator_velocity)
        self.actuator_force = wp.to_torch(self.warp_data.actuator_force)
        self.actuator_moment = wp.to_torch(self.warp_data.actuator_moment)
        self.actuator_moment_rowadr = wp.to_torch(self.warp_data.moment_rowadr)
        self.qfrc_applied = wp.to_torch(self.warp_data.qfrc_applied)
        self.sensordata = wp.to_torch(self.warp_data.sensordata) if int(model.nsensordata) > 0 else None
        self.site_xpos = wp.to_torch(self.warp_data.site_xpos)
        self.body_xpos = wp.to_torch(self.warp_data.xpos)
        self.qacc_warmstart = wp.to_torch(self.warp_data.qacc_warmstart)
        self.time = wp.to_torch(self.warp_data.time)
        self.torque_allocator = (
            BatchedTorqueAllocator(model=model, data=data, config=config, device=device)
            if self.torque_action_enabled and not self.direct_torque_action_enabled
            else None
        )
        self.last_torque_allocator_terms: dict[str, torch.Tensor] = {}
        if self.capture_step_graph:
            wp.capture_begin(device=str(device))
            for _ in range(self.frame_skip):
                mjw.step(self.warp_model, self.warp_data)
            self.step_graph = wp.capture_end(device=str(device))
            wp.synchronize()
        self.ctrl_low = torch.tensor(model.actuator_ctrlrange[:, 0].copy(), dtype=torch.float32, device=device)
        self.ctrl_high = torch.tensor(model.actuator_ctrlrange[:, 1].copy(), dtype=torch.float32, device=device)
        muscle_slice = slice(0, int(model.na))
        self.muscle_actuator_length_range = torch.tensor(
            model.actuator_lengthrange[muscle_slice].copy(), dtype=torch.float32, device=device
        )
        self.muscle_actuator_acc0 = torch.tensor(
            model.actuator_acc0[muscle_slice].copy(), dtype=torch.float32, device=device
        )
        self.muscle_actuator_bias_parameters = torch.tensor(
            model.actuator_biasprm[muscle_slice].copy(), dtype=torch.float32, device=device
        )
        self.hip_muscle_actuator_indices: list[torch.Tensor] = []
        self.hip_muscle_moment_offsets: list[torch.Tensor] = []
        self.hip_exo_actuator_indices: list[int] = []
        self.hip_exo_moment_offsets: list[int] = []
        self.hip_flexion_dof_indices: list[int] = []
        self.hip_exo_unit_force: list[float] = []
        self.energy_joint_muscle_actuator_indices: list[torch.Tensor] = []
        self.energy_joint_muscle_moment_offsets: list[torch.Tensor] = []
        if self.hip_torque_measurement_enabled:
            for side, joint_name in enumerate(("hip_flexion_r", "hip_flexion_l")):
                joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if joint_id < 0:
                    raise KeyError(f"missing hip joint {joint_name}")
                dof = int(model.jnt_dofadr[joint_id])
                self.hip_flexion_dof_indices.append(dof)
                muscle_actuators: list[int] = []
                muscle_offsets: list[int] = []
                exo_actuator = int(model.na) + side
                exo_offset: int | None = None
                for actuator in range(int(model.nu)):
                    rowadr = int(data.moment_rowadr[actuator])
                    rownnz = int(data.moment_rownnz[actuator])
                    columns = data.moment_colind[rowadr : rowadr + rownnz]
                    matches = np.flatnonzero(np.asarray(columns) == dof)
                    if matches.size == 0:
                        continue
                    offset = int(matches[0])
                    if actuator < int(model.na):
                        muscle_actuators.append(actuator)
                        muscle_offsets.append(offset)
                    elif actuator == exo_actuator:
                        exo_offset = offset
                if not muscle_actuators or exo_offset is None:
                    raise ValueError(f"could not resolve muscle/Exo moment rows for {joint_name}")
                self.hip_muscle_actuator_indices.append(
                    torch.tensor(muscle_actuators, dtype=torch.long, device=device)
                )
                self.hip_muscle_moment_offsets.append(
                    torch.tensor(muscle_offsets, dtype=torch.long, device=device)
                )
                self.hip_exo_actuator_indices.append(exo_actuator)
                self.hip_exo_moment_offsets.append(exo_offset)
                self.hip_exo_unit_force.append(float(model.actuator_gainprm[exo_actuator, 0]))
        if self.human_energy_enabled and self.human_energy_joint_cocontraction_measure:
            for joint_name in self.human_energy_joint_cocontraction_names:
                joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if joint_id < 0:
                    raise KeyError(f"missing joint cocontraction joint {joint_name}")
                dof = int(model.jnt_dofadr[joint_id])
                actuator_indices: list[int] = []
                moment_offsets: list[int] = []
                for actuator in range(int(model.na)):
                    rowadr = int(data.moment_rowadr[actuator])
                    rownnz = int(data.moment_rownnz[actuator])
                    columns = data.moment_colind[rowadr : rowadr + rownnz]
                    matches = np.flatnonzero(np.asarray(columns) == dof)
                    if matches.size > 0:
                        actuator_indices.append(actuator)
                        moment_offsets.append(int(matches[0]))
                if not actuator_indices:
                    raise ValueError(f"no muscles cross joint cocontraction joint {joint_name}")
                self.energy_joint_muscle_actuator_indices.append(
                    torch.tensor(actuator_indices, dtype=torch.long, device=device)
                )
                self.energy_joint_muscle_moment_offsets.append(
                    torch.tensor(moment_offsets, dtype=torch.long, device=device)
                )
        self.model_weight = float(np.sum(model.body_mass) * 9.81)
        self.foot_sensor_names = model_foot_sensor_names(model, config)
        self.foot_sensor_indices = torch.tensor(
            [int(sensor_adr_or_none(model, name)) for name in self.foot_sensor_names],
            dtype=torch.long,
            device=device,
        )
        self.ground_force_sensor_names = model_ground_force_sensor_names(model, config)
        self.ground_force_sensor_indices = torch.tensor(
            [
                int(sensor_adr_or_none(model, name))
                for name in self.ground_force_sensor_names
            ],
            dtype=torch.long,
            device=device,
        )
        joint_sensor_indices: list[int] = []
        for name in ("r_knee_sensor", "l_knee_sensor", "r_hip_sensor", "l_hip_sensor", "r_ankle_sensor", "l_ankle_sensor", "r_mtp_sensor", "l_mtp_sensor"):
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            if sid >= 0:
                joint_sensor_indices.append(int(model.sensor_adr[sid]))
        self.joint_limit_sensor_indices = torch.tensor(joint_sensor_indices, dtype=torch.long, device=device)

        self.pelvis_tx_qpos = int(reference.get("pelvis_tx_qpos", semantic_qpos_index(model, "pelvis_tx")))
        self.pelvis_tx_qvel = int(reference.get("pelvis_tx_qvel", semantic_qvel_index(model, "pelvis_tx")))
        self.pelvis_ty_qpos = int(reference.get("pelvis_ty_qpos", semantic_qpos_index(model, "pelvis_ty")))
        self.pelvis_ty_qvel = int(reference.get("pelvis_ty_qvel", semantic_qvel_index(model, "pelvis_ty")))
        self.foot_site_indices = reference["foot_site_indices"]
        self.keypoint_body_indices = reference["keypoint_body_indices"]
        gait_track_positions = [
            index
            for index, name in enumerate(TRACK_JOINTS)
            if name != "pelvis_ty"
        ]
        gait_track_names = [TRACK_JOINTS[index] for index in gait_track_positions]
        gait_name_to_index = {
            name: index for index, name in enumerate(gait_track_names)
        }

        def mirrored_name(name: str) -> str:
            if name.endswith("_r"):
                return name[:-2] + "_l"
            if name.endswith("_l"):
                return name[:-2] + "_r"
            return name

        self.gait_cycle_qpos_indices = reference["qpos_indices"].index_select(
            0,
            torch.tensor(gait_track_positions, dtype=torch.long, device=device),
        )
        self.gait_cycle_qvel_indices = reference["qvel_indices"].index_select(
            0,
            torch.tensor(gait_track_positions, dtype=torch.long, device=device),
        )
        self.gait_cycle_state_mirror_perm = torch.tensor(
            [
                gait_name_to_index[mirrored_name(name)]
                for name in gait_track_names
            ],
            dtype=torch.long,
            device=device,
        )
        actuator_names = [
            mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                index,
            )
            for index in range(int(model.na))
        ]
        actuator_name_to_index = {
            name: index for index, name in enumerate(actuator_names)
        }
        self.gait_cycle_activation_mirror_perm = torch.tensor(
            [
                actuator_name_to_index.get(mirrored_name(name), index)
                for index, name in enumerate(actuator_names)
            ],
            dtype=torch.long,
            device=device,
        )
        keypoint_cfg = config.get("keypoint_imitation", {})
        self.keypoint_imitation_scale = max(
            float(keypoint_cfg.get("scale_m", 0.12)),
            1.0e-6,
        )
        limb_alignment_site_names = ["hip_r", "knee_r", "hip_l", "knee_l"]
        limb_alignment_site_ids = [site_id_or_none(model, name) for name in limb_alignment_site_names]
        if all(site_index is not None for site_index in limb_alignment_site_ids):
            self.limb_alignment_site_indices = torch.tensor(
                [int(site_index) for site_index in limb_alignment_site_ids],
                dtype=torch.long,
                device=device,
            )
        else:
            self.limb_alignment_site_indices = torch.empty((0,), dtype=torch.long, device=device)

        equality_specs = joint_equality_specs_np(model)
        self.eq_qpos1 = torch.tensor([v[0] for v in equality_specs], dtype=torch.long, device=device)
        self.eq_qpos2 = torch.tensor([v[1] for v in equality_specs], dtype=torch.long, device=device)
        self.eq_qvel1 = torch.tensor([v[2] for v in equality_specs], dtype=torch.long, device=device)
        self.eq_qvel2 = torch.tensor([v[3] for v in equality_specs], dtype=torch.long, device=device)
        eq_poly = np.stack([v[4] for v in equality_specs], axis=0) if equality_specs else np.zeros((0, 5), dtype=np.float32)
        self.eq_poly = torch.tensor(eq_poly, dtype=torch.float32, device=device)

        self.phase_idx = torch.zeros(nworld, dtype=torch.long, device=device)
        self.x_align_mask = torch.full(
            (nworld,),
            self.x_align_enabled,
            dtype=torch.bool,
            device=device,
        )
        self.episode_step = torch.zeros(nworld, dtype=torch.long, device=device)
        gait_state_dim = int(self.gait_cycle_qpos_indices.numel())
        self.gait_cycle_last_landing_step = torch.full(
            (nworld, 2),
            -1,
            dtype=torch.long,
            device=device,
        )
        self.gait_cycle_last_landing_valid = torch.zeros(
            (nworld, 2),
            dtype=torch.bool,
            device=device,
        )
        self.gait_cycle_last_landing_qpos = torch.zeros(
            (nworld, 2, gait_state_dim),
            dtype=torch.float32,
            device=device,
        )
        self.gait_cycle_last_landing_qvel = torch.zeros_like(
            self.gait_cycle_last_landing_qpos
        )
        self.gait_cycle_last_landing_activation = torch.zeros(
            (nworld, 2, int(model.na)),
            dtype=torch.float32,
            device=device,
        )
        self.gait_dense_half_cycle_qpos = torch.zeros(
            (nworld, self.gait_dense_half_cycle_steps, gait_state_dim),
            dtype=torch.float32,
            device=device,
        )
        self.gait_dense_half_cycle_qvel = torch.zeros_like(
            self.gait_dense_half_cycle_qpos
        )
        self.gait_dense_half_cycle_activation = torch.zeros(
            (nworld, self.gait_dense_half_cycle_steps, int(model.na)),
            dtype=torch.float32,
            device=device,
        )
        self.gait_dense_half_cycle_side_force = torch.zeros(
            (nworld, self.gait_dense_half_cycle_steps, 2),
            dtype=torch.float32,
            device=device,
        )
        sequence_shape = (
            nworld,
            self.gait_sequence_half_cycle_capacity,
            gait_state_dim,
        )
        self.gait_sequence_current_qpos = torch.zeros(
            sequence_shape, dtype=torch.float32, device=device
        )
        self.gait_sequence_current_qvel = torch.zeros_like(
            self.gait_sequence_current_qpos
        )
        self.gait_sequence_current_activation = torch.zeros(
            (nworld, self.gait_sequence_half_cycle_capacity, int(model.na)),
            dtype=torch.float32,
            device=device,
        )
        self.gait_sequence_current_side_force = torch.zeros(
            (nworld, self.gait_sequence_half_cycle_capacity, 2),
            dtype=torch.float32,
            device=device,
        )
        self.gait_sequence_previous_qpos = torch.zeros_like(
            self.gait_sequence_current_qpos
        )
        self.gait_sequence_previous_qvel = torch.zeros_like(
            self.gait_sequence_current_qvel
        )
        self.gait_sequence_previous_activation = torch.zeros_like(
            self.gait_sequence_current_activation
        )
        self.gait_sequence_previous_side_force = torch.zeros_like(
            self.gait_sequence_current_side_force
        )
        self.gait_sequence_current_length = torch.zeros(
            nworld, dtype=torch.long, device=device
        )
        self.gait_sequence_previous_length = torch.zeros_like(
            self.gait_sequence_current_length
        )
        self.gait_sequence_current_start_side = torch.full(
            (nworld,), -1, dtype=torch.long, device=device
        )
        self.gait_sequence_previous_start_side = torch.full_like(
            self.gait_sequence_current_start_side, -1
        )
        self.gait_sequence_started = torch.zeros(
            nworld, dtype=torch.bool, device=device
        )
        self.gait_sequence_previous_valid = torch.zeros_like(
            self.gait_sequence_started
        )
        self.gait_sequence_current_overflow = torch.zeros_like(
            self.gait_sequence_started
        )
        self.gait_stance_started = torch.zeros(
            (nworld, 2), dtype=torch.bool, device=device
        )
        self.gait_stance_impulse = torch.zeros(
            (nworld, 2), dtype=torch.float32, device=device
        )
        self.gait_stance_duration_steps = torch.zeros(
            (nworld, 2), dtype=torch.long, device=device
        )
        self.gait_stance_peak_force = torch.zeros_like(self.gait_stance_impulse)
        self.gait_last_stance_impulse = torch.zeros_like(self.gait_stance_impulse)
        self.gait_last_stance_duration_steps = torch.zeros_like(
            self.gait_stance_duration_steps
        )
        self.gait_last_stance_peak_force = torch.zeros_like(
            self.gait_stance_impulse
        )
        self.gait_last_stance_valid = torch.zeros_like(self.gait_stance_started)
        self.gait_stance_event_hold_remaining = torch.zeros(
            nworld, dtype=torch.long, device=device
        )
        self.gait_stance_held_terms = {
            name: torch.zeros(nworld, dtype=torch.float32, device=device)
            for name in (
                "gait_stance_impulse_balance_reward",
                "gait_stance_duration_balance_reward",
                "gait_stance_peak_force_balance_reward",
                "gait_stance_impulse_balance_penalty",
                "gait_stance_peak_force_balance_penalty",
            )
        }
        self.gait_cycle_last_event_step = torch.full(
            (nworld,),
            -1,
            dtype=torch.long,
            device=device,
        )
        self.gait_cycle_last_event_side = torch.full(
            (nworld,),
            -1,
            dtype=torch.long,
            device=device,
        )
        self.gait_cycle_last_half_interval = torch.zeros(
            nworld,
            dtype=torch.float32,
            device=device,
        )
        self.gait_cycle_last_half_interval_valid = torch.zeros(
            nworld,
            dtype=torch.bool,
            device=device,
        )
        self.gait_cycle_event_hold_remaining = torch.zeros(
            nworld,
            dtype=torch.long,
            device=device,
        )
        self.gait_cycle_held_terms = {
            name: torch.zeros(
                nworld,
                dtype=torch.float32,
                device=device,
            )
            for name in (
                "gait_cycle_cadence_reward",
                "gait_half_cycle_cadence_reward",
                "gait_half_cycle_balance_reward",
                "gait_cycle_pose_reward",
                "gait_half_cycle_pose_reward",
                "gait_cycle_velocity_reward",
                "gait_half_cycle_velocity_reward",
                "gait_cycle_activation_reward",
                "gait_half_cycle_activation_reward",
                "gait_sequence_half_cycle_pose_reward",
                "gait_sequence_half_cycle_velocity_reward",
                "gait_sequence_half_cycle_activation_reward",
                "gait_sequence_half_cycle_force_reward",
                "gait_alternation_reward",
            )
        }
        self.stair_contact_progress_max = torch.zeros(
            nworld, dtype=torch.float32, device=device
        )
        rollover_cfg = config.get("reward_foot_rollover", {})
        self.foot_rollover_enabled = bool(rollover_cfg.get("enabled", False))
        self.foot_rollover_contact_force_threshold_n = max(
            0.0, float(rollover_cfg.get("contact_force_threshold_n", 5.0))
        )
        self.foot_rollover_min_heel_delay_steps = max(
            1,
            int(round(float(rollover_cfg.get("min_heel_delay_s", 0.03)) / self.dt)),
        )
        self.foot_rollover_max_heel_delay_steps = max(
            self.foot_rollover_min_heel_delay_steps,
            int(round(float(rollover_cfg.get("max_heel_delay_s", 0.15)) / self.dt)),
        )
        self.foot_rollover_release_steps = max(
            1, int(round(float(rollover_cfg.get("release_time_s", 0.03)) / self.dt))
        )
        self.foot_rollover_required_heel_stable_steps = max(
            1,
            int(round(float(rollover_cfg.get("heel_stable_time_s", 0.03)) / self.dt)),
        )
        self.foot_rollover_max_heel_force_delta_n = max(
            1.0,
            float(
                rollover_cfg.get(
                    "max_heel_force_delta_n",
                    0.30 * self.model_weight,
                )
            ),
        )
        descending_stairs = [
            segment
            for segment in config.get("terrain_course", {}).get("segments", [])
            if str(segment.get("type", "")) == "stairs_box"
            and float(segment.get("direction", 1.0)) < 0.0
        ]
        self.foot_rollover_x_start = float(
            rollover_cfg.get(
                "x_start",
                min((float(segment["x0"]) for segment in descending_stairs), default=-float("inf")),
            )
        )
        self.foot_rollover_x_end = float(
            rollover_cfg.get(
                "x_end",
                max((float(segment["x1"]) for segment in descending_stairs), default=float("inf")),
            )
        )
        self.foot_rollover_previous_contact = torch.zeros(
            (nworld, 4), dtype=torch.bool, device=device
        )
        self.foot_rollover_previous_force = torch.zeros(
            (nworld, 4), dtype=torch.float32, device=device
        )
        self.foot_rollover_state = torch.zeros(
            (nworld, 2), dtype=torch.long, device=device
        )
        self.foot_rollover_elapsed_steps = torch.zeros_like(
            self.foot_rollover_state
        )
        self.foot_rollover_airborne_steps = torch.zeros_like(
            self.foot_rollover_state
        )
        self.foot_rollover_heel_stable_steps = torch.zeros_like(
            self.foot_rollover_state
        )
        self.foot_rollover_heel_loading_excess = torch.zeros(
            (nworld, 2), dtype=torch.float32, device=device
        )
        self.ramp_previous_landing_x = torch.zeros(
            (nworld, 2), dtype=torch.float32, device=device
        )
        self.ramp_previous_landing_side = torch.full(
            (nworld,), -1, dtype=torch.long, device=device
        )
        self.applied_exo_ctrl = torch.zeros((nworld, 2), dtype=torch.float32, device=device)
        self.applied_direct_torque = torch.zeros(
            (nworld, 6), dtype=torch.float32, device=device
        )
        self.exo_ctrl_sq_ema = torch.zeros((nworld, 2), dtype=torch.float32, device=device)
        self.last_step_activation = torch.zeros((nworld, model.na), dtype=torch.float32, device=device)
        self.last_step_hip_human_torque = torch.zeros((nworld, 2), dtype=torch.float32, device=device)
        self.last_step_hip_exo_torque = torch.zeros((nworld, 2), dtype=torch.float32, device=device)
        self.prev_activation = torch.full((nworld, model.na), self.initial_activation, dtype=torch.float32, device=device)
        self.prev_activation_valid = torch.zeros(nworld, dtype=torch.bool, device=device)
        self.episode_return = torch.zeros(nworld, dtype=torch.float32, device=device)
        self.episode_length = torch.zeros(nworld, dtype=torch.float32, device=device)
        self.episode_start_pelvis_tx = torch.zeros(nworld, dtype=torch.float32, device=device)
        self.episode_start_phase = torch.zeros(
            nworld, dtype=torch.long, device=device
        )
        self.episode_reset_source = torch.zeros(
            nworld, dtype=torch.long, device=device
        )
        self.reset_source_counts = torch.zeros(
            3, dtype=torch.long, device=device
        )
        self.reset_source_start_x_sum = torch.zeros(
            3, dtype=torch.float64, device=device
        )
        self.recovery_mode_steps = torch.zeros(nworld, dtype=torch.long, device=device)
        self.recovery_bank_size = 0
        self.recovery_bank_write = 0
        self.recovery_segment_starts: list[int] = []
        segment_start = 0
        for segment_capacity in self.recovery_segment_capacities:
            self.recovery_segment_starts.append(segment_start)
            segment_start += int(segment_capacity)
        self.recovery_segment_sizes = [
            0 for _ in self.recovery_segment_capacities
        ]
        self.recovery_segment_writes = [
            0 for _ in self.recovery_segment_capacities
        ]
        self.recovery_last_collect_count = 0
        self.recovery_last_stage_count = 0
        self.recovery_last_commit_count = 0
        self.recovery_last_restore_count = 0
        self.offline_recovery_bank_size = 0
        self.offline_recovery_last_restore_count = 0
        self.offline_recovery_last_sample_index = torch.full(
            (nworld,), -1, dtype=torch.long, device=device
        )
        self.offline_recovery_bank_proprio_history = None
        if self.recovery_reset_enabled and self.recovery_bank_capacity > 0:
            self.recovery_bank_qpos = torch.zeros((self.recovery_bank_capacity, model.nq), dtype=torch.float32, device=device)
            self.recovery_bank_qvel = torch.zeros((self.recovery_bank_capacity, model.nv), dtype=torch.float32, device=device)
            self.recovery_bank_act = torch.zeros((self.recovery_bank_capacity, model.na), dtype=torch.float32, device=device)
            self.recovery_bank_ctrl = torch.zeros((self.recovery_bank_capacity, model.nu), dtype=torch.float32, device=device)
            self.recovery_bank_qacc_warmstart = torch.zeros(
                (self.recovery_bank_capacity, model.nv), dtype=torch.float32, device=device
            )
            self.recovery_bank_site_xpos = torch.zeros(
                (self.recovery_bank_capacity, *self.site_xpos.shape[1:]),
                dtype=torch.float32,
                device=device,
            )
            self.recovery_bank_phase = torch.zeros(self.recovery_bank_capacity, dtype=torch.long, device=device)
            self.recovery_bank_x_align_mask = torch.zeros(
                self.recovery_bank_capacity, dtype=torch.bool, device=device
            )
            self.recovery_bank_prev_activation = torch.zeros((self.recovery_bank_capacity, model.na), dtype=torch.float32, device=device)
            self.load_recovery_bootstrap_banks()
        if (
            self.recovery_reset_enabled
            and self.recovery_bank_capacity > 0
            and self.recovery_survival_delay_steps > 0
        ):
            delay = int(self.recovery_survival_delay_steps)
            self.recovery_pending_write = 0
            self.recovery_pending_valid = torch.zeros((delay, nworld), dtype=torch.bool, device=device)
            self.recovery_pending_qpos = torch.zeros((delay, nworld, model.nq), dtype=torch.float32, device=device)
            self.recovery_pending_qvel = torch.zeros((delay, nworld, model.nv), dtype=torch.float32, device=device)
            self.recovery_pending_act = torch.zeros((delay, nworld, model.na), dtype=torch.float32, device=device)
            self.recovery_pending_ctrl = torch.zeros((delay, nworld, model.nu), dtype=torch.float32, device=device)
            self.recovery_pending_qacc_warmstart = torch.zeros(
                (delay, nworld, model.nv), dtype=torch.float32, device=device
            )
            self.recovery_pending_site_xpos = torch.zeros(
                (delay, nworld, *self.site_xpos.shape[1:]),
                dtype=torch.float32,
                device=device,
            )
            self.recovery_pending_phase = torch.zeros((delay, nworld), dtype=torch.long, device=device)
            self.recovery_pending_x_align_mask = torch.zeros(
                (delay, nworld), dtype=torch.bool, device=device
            )
            self.recovery_pending_prev_activation = torch.zeros(
                (delay, nworld, model.na),
                dtype=torch.float32,
                device=device,
            )
        self.load_offline_recovery_bank()
        self.reset(torch.ones(nworld, dtype=torch.bool, device=device))

    def set_reference_curriculum(self, *, phase_lead_steps: int, phase_tolerance_steps: int, swing_exaggeration_scale: float) -> None:
        self.reference_phase_lead_steps = int(phase_lead_steps)
        self.reference_phase_tolerance_steps = max(0, int(phase_tolerance_steps))
        self.reference_swing_exaggeration_scale = max(1.0, float(swing_exaggeration_scale))

    def reset_exo_control(self, rows: torch.Tensor) -> None:
        self.applied_exo_ctrl[rows] = 0.0
        self.exo_ctrl_sq_ema[rows] = 0.0

    def current_hip_generalized_torques(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return current human, applied Exo, and ctrl=1 Exo hip torques in Nm."""
        if not self.hip_torque_measurement_enabled:
            zero = torch.zeros((self.nworld, 2), dtype=self.qpos.dtype, device=self.device)
            return zero, zero, zero
        human_torques: list[torch.Tensor] = []
        exo_torques: list[torch.Tensor] = []
        exo_unit_torques: list[torch.Tensor] = []
        for side in range(2):
            actuator_indices = self.hip_muscle_actuator_indices[side]
            row_offsets = self.hip_muscle_moment_offsets[side]
            moment_indices = self.actuator_moment_rowadr.index_select(1, actuator_indices)
            moment_indices = moment_indices + row_offsets.unsqueeze(0)
            muscle_moments = self.actuator_moment.gather(1, moment_indices)
            muscle_forces = self.actuator_force.index_select(1, actuator_indices)
            human_torques.append(torch.sum(muscle_forces * muscle_moments, dim=1))

            exo_actuator = self.hip_exo_actuator_indices[side]
            exo_moment_index = self.actuator_moment_rowadr[:, exo_actuator] + int(
                self.hip_exo_moment_offsets[side]
            )
            exo_moment = self.actuator_moment.gather(1, exo_moment_index.unsqueeze(1)).squeeze(1)
            exo_torques.append(self.actuator_force[:, exo_actuator] * exo_moment)
            exo_unit_torques.append(exo_moment * float(self.hip_exo_unit_force[side]))
        return (
            torch.stack(human_torques, dim=1),
            torch.stack(exo_torques, dim=1),
            torch.stack(exo_unit_torques, dim=1),
        )

    def finite_state_mask(self) -> torch.Tensor:
        return (
            torch.isfinite(self.qpos).all(dim=1)
            & torch.isfinite(self.qvel).all(dim=1)
            & torch.isfinite(self.act).all(dim=1)
            & torch.isfinite(self.ctrl).all(dim=1)
            & torch.isfinite(self.site_xpos).all(dim=(1, 2))
        )

    @property
    def obs_dim(self) -> int:
        foot_dim = foot_obs_feature_dim(self.config)
        future_dim = self.reference_future_steps * (len(TRACK_JOINTS) + foot_dim)
        return (
            self.model.nq
            + self.model.nv
            + self.model.na
            + 2 * len(TRACK_JOINTS)
            + 2
            + foot_dim
            + future_dim
            + terrain_preview_dim(self.config)
            + footstep_target_dim(self.config)
            + policy_task_context_dim(self.config)
        )

    @property
    def act_dim(self) -> int:
        if self.direct_torque_action_enabled:
            return 6
        if self.torque_allocator is not None:
            return int(self.torque_allocator.action_dim)
        return int(self.model.nu)

    def apply_joint_equalities(self, rows: torch.Tensor) -> None:
        if self.eq_poly.numel() == 0:
            return
        q = self.qpos[rows[:, None], self.eq_qpos2[None, :]]
        dq = self.qvel[rows[:, None], self.eq_qvel2[None, :]]
        q2 = torch.square(q)
        q3 = q2 * q
        q4 = q3 * q
        poly = self.eq_poly
        value = poly[:, 0] + poly[:, 1] * q + poly[:, 2] * q2 + poly[:, 3] * q3 + poly[:, 4] * q4
        derivative = poly[:, 1] + 2.0 * poly[:, 2] * q + 3.0 * poly[:, 3] * q2 + 4.0 * poly[:, 4] * q3
        self.qpos[rows[:, None], self.eq_qpos1[None, :]] = value
        self.qvel[rows[:, None], self.eq_qvel1[None, :]] = derivative * dq

    def obs(self) -> torch.Tensor:
        exo_torque = self.current_hip_generalized_torques()[1] if self.hip_torque_measurement_enabled else None
        activation_obs = self.act
        if self.direct_torque_action_enabled and self.direct_torque_in_activation_obs:
            activation_obs = self.act.clone()
            activation_obs[:, :6] = self.applied_direct_torque / self.direct_torque_scale.unsqueeze(0)
        return build_policy_obs_tensor(
            qpos=self.qpos,
            qvel=self.qvel,
            act=activation_obs,
            site_xpos=self.site_xpos,
            sensordata=self.sensordata,
            foot_sensor_indices=self.foot_sensor_indices,
            model_weight=self.model_weight,
            phase_idx=self.phase_idx,
            pelvis_tx_qpos=self.pelvis_tx_qpos,
            foot_site_indices=self.foot_site_indices,
            reference=self.reference,
            config=self.config,
            non_muscle_ctrl=self.ctrl[:, int(self.model.na) :],
            non_muscle_torque=exo_torque,
            x_align_mask=self.x_align_mask,
        )

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if self.direct_torque_action_enabled:
            normalized_torque = action.clamp(-1.0, 1.0)
            target_torque = normalized_torque * self.direct_torque_scale.unsqueeze(0)
            filtered_torque = self.applied_direct_torque + self.direct_torque_filter_alpha * (
                target_torque - self.applied_direct_torque
            )
            torque_delta = torch.clamp(
                filtered_torque - self.applied_direct_torque,
                min=-self.direct_torque_max_delta.unsqueeze(0),
                max=self.direct_torque_max_delta.unsqueeze(0),
            )
            self.applied_direct_torque.add_(torque_delta)
            applied_torque = self.applied_direct_torque
            self.qfrc_applied.zero_()
            self.qfrc_applied[:, self.direct_torque_dof_indices] = applied_torque
            policy_ctrl = torch.zeros(
                (self.nworld, int(self.model.nu)), dtype=action.dtype, device=self.device
            )
            self.act.zero_()
            zeros = torch.zeros(self.nworld, dtype=action.dtype, device=self.device)
            self.last_torque_allocator_terms = {
                "torque_action_projection_error": torch.linalg.norm(
                    applied_torque / self.direct_torque_scale.unsqueeze(0) - normalized_torque,
                    dim=1,
                ),
                "torque_action_requested_abs_mean": torch.mean(torch.abs(normalized_torque), dim=1),
                "torque_action_achieved_abs_mean": torch.mean(
                    torch.abs(applied_torque / self.direct_torque_scale.unsqueeze(0)), dim=1
                ),
                "torque_allocator_target_activation_l2": zeros,
                "direct_joint_torque_abs_mean_nm": torch.mean(torch.abs(applied_torque), dim=1),
                "direct_joint_torque_delta_abs_mean_nm": torch.mean(torch.abs(torque_delta), dim=1),
                "direct_joint_torque_delta_penalty": -self.dt * torch.mean(
                    torch.square(
                        torque_delta / self.direct_torque_delta_penalty_scale.unsqueeze(0)
                    ),
                    dim=1,
                ),
            }
        elif self.torque_allocator is not None:
            policy_ctrl, self.last_torque_allocator_terms = self.torque_allocator.allocate(action, self)
        else:
            policy_ctrl = policy_action_to_ctrl(
                action,
                self.ctrl_low,
                self.ctrl_high,
                muscle_count=int(self.model.na),
                muscle_mapping=self.muscle_action_mapping,
            )
            policy_ctrl = apply_non_muscle_ctrl_override(
                policy_ctrl,
                self.config,
                muscle_count=int(self.model.na),
                ctrl_low=self.ctrl_low,
                ctrl_high=self.ctrl_high,
            )
            self.last_torque_allocator_terms = {}
        activation = policy_ctrl[:, : int(self.model.na)]
        ctrl = policy_ctrl.clone()
        ctrl[:, : int(self.model.na)] = activation
        prev_exo_ctrl = self.applied_exo_ctrl.clone()
        if self.exo_policy_enabled and self.torque_allocator is None and not self.direct_torque_action_enabled:
            exo_action = torch.clamp(
                action[:, int(self.model.na) : int(self.model.na) + 2], -1.0, 1.0
            )
            if not self.exo_policy_bidirectional:
                exo_action = torch.relu(exo_action)
            exo = self.exo_policy_max_ctrl * exo_action
            if self.exo_policy_max_delta > 0.0:
                exo = self.applied_exo_ctrl + torch.clamp(
                    exo - self.applied_exo_ctrl,
                    min=-self.exo_policy_max_delta,
                    max=self.exo_policy_max_delta,
                )
            ctrl[:, int(self.model.na) : int(self.model.na) + 2] = exo
        self.applied_exo_ctrl.zero_()
        non_muscle_ctrl = ctrl[:, int(self.model.na) : int(self.model.na) + 2]
        if int(non_muscle_ctrl.shape[1]) > 0:
            self.applied_exo_ctrl[:, : int(non_muscle_ctrl.shape[1])].copy_(non_muscle_ctrl)
        prev_foot = self.site_xpos[:, self.foot_site_indices, :].clone()
        prev_foot_forward = site_forward_coord_tensor(prev_foot, self.config)
        prev_terrain_height = terrain_height_for_world_x_tensor(prev_foot_forward, self.phase_idx, self.reference, self.config)
        prev_foot_contact = (
            prev_foot[:, :, 2] - prev_terrain_height
        ) < float(self.config.get("reference_contact", {}).get("z_threshold", 0.025))
        self.ctrl.copy_(ctrl)
        if self.step_graph is not None:
            wp.capture_launch(self.step_graph)
        else:
            for _ in range(self.frame_skip):
                mjw.step(self.warp_model, self.warp_data)
        wp.synchronize()
        self.phase_idx = (self.phase_idx + 1) % int(self.reference["length"])
        self.episode_step += 1
        nonfinite_state = ~self.finite_state_mask()
        if bool(nonfinite_state.any().item()):
            self.reset(nonfinite_state)
        current_activation = self.act.clone()
        self.last_step_activation.copy_(current_activation)
        if self.hip_torque_measurement_enabled:
            human_torque, exo_torque, _ = self.current_hip_generalized_torques()
            self.last_step_hip_human_torque.copy_(human_torque)
            self.last_step_hip_exo_torque.copy_(exo_torque)
        else:
            self.last_step_hip_human_torque.zero_()
            self.last_step_hip_exo_torque.zero_()
        reward, terms = self.reward(action, current_activation, prev_foot)
        terms.update(self.last_torque_allocator_terms)
        if self.direct_torque_action_enabled:
            reward = reward + self.direct_torque_delta_penalty_weight * terms[
                "direct_joint_torque_delta_penalty"
            ]
        if self.last_torque_allocator_terms:
            projection_penalty = -self.dt * self.last_torque_allocator_terms[
                "torque_action_projection_error"
            ]
            reward = reward + self.torque_projection_penalty_weight * projection_penalty
            terms["torque_action_projection_penalty"] = projection_penalty
        if self.human_energy_enabled:
            energy_reward, energy_terms = self.human_energy_reward(
                current_activation=current_activation,
                tracking_error=terms["reference_tracking_error"],
            )
            reward = reward + energy_reward
            terms.update(energy_terms)
        effort_l2 = -self.dt * torch.mean(torch.square(current_activation), dim=1)
        if self.exo_policy_enabled and self.exo_policy_human_effort_l2_penalty > 0.0:
            reward = reward + self.exo_policy_human_effort_l2_penalty * effort_l2
        terms["exo_effort_l2_penalty"] = effort_l2
        if self.exo_policy_enabled and (
            self.exo_policy_hip_activation_l2_penalty > 0.0
            or self.exo_policy_hip_cocontraction_penalty > 0.0
        ):
            hip_activation_l2, hip_cocontraction = self.current_hip_effort_metrics(current_activation)
            hip_activation_penalty = -self.dt * hip_activation_l2
            hip_cocontraction_penalty = (
                -self.dt * hip_cocontraction / self.exo_policy_hip_cocontraction_scale
            )
            reward = reward + self.exo_policy_hip_activation_l2_penalty * hip_activation_penalty
            reward = reward + self.exo_policy_hip_cocontraction_penalty * hip_cocontraction_penalty
            terms["hip_activation_l2_penalty"] = hip_activation_penalty
            terms["hip_cocontraction_penalty"] = hip_cocontraction_penalty
            terms["hip_activation_l2"] = hip_activation_l2
            terms["hip_cocontraction_nm"] = hip_cocontraction
        exo_ctrl_l2 = -self.dt * torch.mean(torch.square(self.applied_exo_ctrl), dim=1)
        exo_ctrl_smooth = -self.dt * torch.mean(
            torch.square(self.applied_exo_ctrl - prev_exo_ctrl), dim=1
        )
        balance_alpha = self.exo_policy_ctrl_balance_ema_alpha
        self.exo_ctrl_sq_ema.lerp_(torch.square(self.applied_exo_ctrl), balance_alpha)
        exo_ctrl_balance = -self.dt * torch.abs(
            self.exo_ctrl_sq_ema[:, 0] - self.exo_ctrl_sq_ema[:, 1]
        )
        if self.exo_policy_enabled:
            reward = reward + self.exo_policy_ctrl_l2_penalty * exo_ctrl_l2
            reward = reward + self.exo_policy_ctrl_smooth_penalty * exo_ctrl_smooth
            reward = reward + self.exo_policy_ctrl_balance_penalty * exo_ctrl_balance
        terms["exo_ctrl_l2_penalty"] = exo_ctrl_l2
        terms["exo_ctrl_smooth_penalty"] = exo_ctrl_smooth
        terms["exo_ctrl_balance_penalty"] = exo_ctrl_balance
        terms["exo_ctrl_sq_ema_r"] = self.exo_ctrl_sq_ema[:, 0]
        terms["exo_ctrl_sq_ema_l"] = self.exo_ctrl_sq_ema[:, 1]
        terms["exo_assistance_ctrl_r"] = self.applied_exo_ctrl[:, 0]
        terms["exo_assistance_ctrl_l"] = self.applied_exo_ctrl[:, 1]
        terms["exo_assistance_ctrl_mean"] = self.applied_exo_ctrl.mean(dim=1)
        reward = torch.nan_to_num(reward, nan=-float(self.reward_weights.get("fall", 20.0)), posinf=0.0, neginf=-1e6)
        for key, value in list(terms.items()):
            if torch.is_tensor(value) and value.is_floating_point():
                terms[key] = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        terrain_height = current_terrain_height_tensor(self.qpos, self.phase_idx, self.reference, self.config)
        pelvis_height_above_terrain = self.qpos[:, self.pelvis_ty_qpos] - terrain_height
        low_height = pelvis_height_above_terrain < self.safe_pelvis_height
        ref_q = self.reference_q_dq(self.target_phase_idx())[0]
        q = self.qpos[:, self.reference["qpos_indices"]]
        qpos_mask = self.pose_weights > 0.0
        tracked_qpos_abs_error = torch.abs(
            q[:, qpos_mask] - ref_q[:, qpos_mask]
        )
        out_of_trajectory = torch.any(
            tracked_qpos_abs_error > self.myoassist_out_of_trajectory_threshold,
            dim=1,
        )
        terms["out_of_trajectory_max_abs_error"] = torch.max(
            tracked_qpos_abs_error, dim=1
        ).values
        terms["out_of_trajectory_max_error_index"] = torch.argmax(
            tracked_qpos_abs_error, dim=1
        ).float()
        root_xy_drift_done = torch.zeros_like(low_height)
        if self.root_xy_drift_done_threshold > 0.0 and self.root_qpos_adr >= 0:
            full_ref_q = self.reference["full_reset_qpos"][self.target_phase_idx()].to(
                device=self.device, dtype=self.qpos.dtype
            )
            root_xy_delta = self.qpos[:, self.root_qpos_adr : self.root_qpos_adr + 2] - full_ref_q[
                :, self.root_qpos_adr : self.root_qpos_adr + 2
            ]
            root_xy_drift = torch.linalg.norm(root_xy_delta, dim=1)
            root_xy_drift_done = root_xy_drift > self.root_xy_drift_done_threshold
            terms["root_xy_drift"] = root_xy_drift
            lateral = self.root_qpos_adr + (1 if self.forward_axis == "x" else 0)
            terms["lateral_drift_abs"] = torch.abs(self.qpos[:, lateral] - full_ref_q[:, lateral])
        else:
            terms["root_xy_drift"] = torch.zeros_like(low_height, dtype=self.qpos.dtype)
            terms["lateral_drift_abs"] = torch.zeros_like(low_height, dtype=self.qpos.dtype)
        bad_tilt = torch.zeros_like(low_height)
        fallen = low_height | out_of_trajectory | root_xy_drift_done
        qvel_bad = torch.zeros_like(low_height)
        terms["out_of_trajectory_done"] = out_of_trajectory.float()
        terms["root_xy_drift_done"] = root_xy_drift_done.float()
        terms["lateral_drift_done"] = root_xy_drift_done.float()
        truncated = self.episode_step >= self.episode_steps
        done = fallen | qvel_bad | truncated | nonfinite_state
        constraint_failure = fallen | qvel_bad | nonfinite_state
        if self.human_energy_enabled and self.human_energy_constraint_failure_penalty > 0.0:
            constraint_failure_cost = (
                float(self.human_energy_constraint_failure_penalty)
                * constraint_failure.float()
            )
            reward = reward - constraint_failure_cost
            terms["human_energy_constraint_failure_penalty"] = -constraint_failure_cost
        self.episode_return += reward
        self.episode_length += 1.0
        terms["done"] = done.float()
        terms["pelvis_tx_velocity"] = self.qvel[:, self.pelvis_tx_qvel].clone()
        terms["fall_done"] = fallen.float()
        terms["truncated_done"] = truncated.float()
        terms["horizon_completed"] = (truncated & ~constraint_failure).float()
        terms["low_height_done"] = low_height.float()
        terms["tilt_done"] = bad_tilt.float()
        terms["qvel_done"] = qvel_bad.float()
        terms["nonfinite_done"] = nonfinite_state.float()
        terms["done_count"] = done.float()
        terms["episode_return_done_sum"] = torch.where(done, self.episode_return, torch.zeros_like(reward))
        terms["episode_length_done_sum"] = torch.where(done, self.episode_length, torch.zeros_like(reward))
        episode_forward_displacement = self.qpos[:, self.pelvis_tx_qpos] - self.episode_start_pelvis_tx
        terms["episode_forward_displacement_done"] = torch.where(
            done,
            episode_forward_displacement,
            torch.zeros_like(episode_forward_displacement),
        )
        terms["episode_reset_source"] = self.episode_reset_source.float().clone()
        terms["episode_start_x"] = self.episode_start_pelvis_tx.clone()
        terms["episode_start_phase"] = self.episode_start_phase.float().clone()
        terms["episode_end_x_done"] = torch.where(
            done,
            self.qpos[:, self.pelvis_tx_qpos],
            torch.zeros_like(episode_forward_displacement),
        )
        self.update_recovery_priority_stats(done, constraint_failure)
        self.prev_activation.copy_(current_activation)
        self.prev_activation_valid[:] = True
        self.collect_recovery_states(done, pelvis_height_above_terrain)
        self.recovery_mode_steps = torch.clamp(self.recovery_mode_steps - 1, min=0)
        if bool(done.any().item()):
            self.reset(done)
        return self.obs(), reward, done, terms
