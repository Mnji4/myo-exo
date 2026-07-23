"""Dense locomotion, gait, terrain, and stair reward terms."""
from __future__ import annotations

import torch

from myo_exo_train.env.model import FOOT_SITE_NAMES, RESET_JOINTS, site_forward_coord_tensor, site_lateral_coord_tensor
from myo_exo_train.env.observation import (
    current_terrain_height_tensor,
    footstep_target_tensor,
    reference_index,
    stair_step_index_tensor,
    stair_tread_progress_tensor,
    terrain_height_for_world_x_tensor,
)

class DenseLocomotionRewardMixin:
    def myoassist_exact_reward(
        self,
        action: torch.Tensor,
        muscle_activation: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        target_phase = self.target_phase_idx()
        nominal_phase = reference_index(
            self.phase_idx + int(self.reference_phase_lead_steps),
            self.reference,
            self.config,
        )
        ref_len = int(self.reference["length"])
        x_aligned_phase_offset = (
            (target_phase.to(torch.long) - nominal_phase.to(torch.long) + ref_len // 2) % ref_len - ref_len // 2
        ).float()
        x_aligned_reference = (target_phase.to(torch.long) != nominal_phase.to(torch.long)).float()
        ref_q, ref_dq = self.reference_q_dq(target_phase)
        q = self.qpos[:, self.reference["qpos_indices"]]
        dq = self.qvel[:, self.reference["qvel_indices"]]
        dt = float(self.dt)

        forward_vel_error = self.qvel[:, self.pelvis_tx_qvel] - float(self.myoassist_target_velocity)
        if self.forward_velocity_error_mode in {"under_only", "no_fast_penalty", "min_speed"}:
            forward_vel_error = torch.relu(-forward_vel_error)
        forward_reward = dt * torch.exp(-5.0 * torch.square(forward_vel_error))
        muscle_activation_penalty = -dt * torch.mean(muscle_activation, dim=1)
        activation_diff_raw = dt * torch.mean(torch.exp(-4.0 * torch.square(self.prev_activation - muscle_activation)), dim=1)
        muscle_activation_diff_penalty = torch.where(
            self.prev_activation_valid,
            activation_diff_raw,
            torch.zeros_like(activation_diff_raw),
        )

        if self.sensordata is not None and self.foot_sensor_indices.numel() == 4:
            foot_force = self.sensordata.index_select(1, self.foot_sensor_indices)
            right_force = foot_force[:, 0] + foot_force[:, 1]
            left_force = foot_force[:, 2] + foot_force[:, 3]
            normalized_foot_force_sum = (torch.abs(right_force) + torch.abs(left_force)) / max(float(self.model_weight), 1e-6)
            foot_force_penalty = -dt * torch.relu(normalized_foot_force_sum - 1.2)
        else:
            foot_force = torch.zeros((self.nworld, 4), dtype=torch.float32, device=self.device)
            foot_force_penalty = torch.zeros((self.nworld,), dtype=torch.float32, device=self.device)

        if self.sensordata is not None and self.joint_limit_sensor_indices.numel() > 0:
            joint_force = self.sensordata.index_select(1, self.joint_limit_sensor_indices)
            max_joint_force = torch.amax(torch.abs(joint_force), dim=1)
            joint_constraint_force_penalty = -dt * max_joint_force / max(float(self.model_weight), 1e-6)
        else:
            joint_constraint_force_penalty = torch.zeros((self.nworld,), dtype=torch.float32, device=self.device)

        qpos_reward_per_joint = dt * torch.exp(-8.0 * torch.square(q - ref_q))
        qpos_imitation_rewards = torch.sum(qpos_reward_per_joint * self.myoassist_qpos_weights, dim=1)
        ref_pelvis_vx = self.reference["reset_dq_ref"][target_phase, RESET_JOINTS.index("pelvis_tx")]
        speed_ratio = float(self.myoassist_target_velocity) / torch.clamp(ref_pelvis_vx, min=1e-6)
        qvel_reward_per_joint = dt * torch.exp(-8.0 * torch.square(dq - ref_dq * speed_ratio.unsqueeze(1)))
        qvel_imitation_rewards = torch.sum(qvel_reward_per_joint * self.myoassist_qvel_weights, dim=1)
        speed_cfg = self.config.get("reward_forward_shortfall", {})
        speed_target = float(speed_cfg.get("target_velocity", self.myoassist_target_velocity))
        speed_margin = float(speed_cfg.get("margin", 0.0) or 0.0)
        speed_scale = max(float(speed_cfg.get("scale", max(speed_target, 1e-3)) or max(speed_target, 1e-3)), 1e-6)
        speed_shortfall = torch.relu(speed_target - speed_margin - self.qvel[:, self.pelvis_tx_qvel])
        forward_velocity_reward = dt * torch.clamp(self.qvel[:, self.pelvis_tx_qvel], min=0.0, max=speed_target) / max(speed_target, 1e-6)
        forward_shortfall_penalty = -dt * torch.square(speed_shortfall / speed_scale)
        phase_lag_cfg = self.config.get("reward_xalign_phase_lag", {})
        phase_lag_margin = float(phase_lag_cfg.get("margin_steps", 4.0))
        phase_lag_scale = max(float(phase_lag_cfg.get("scale_steps", 24.0)), 1e-6)
        phase_lag_steps = torch.relu(-x_aligned_phase_offset - phase_lag_margin)
        xalign_phase_lag_penalty = -dt * torch.square(phase_lag_steps / phase_lag_scale)
        full_qpos_imitation_rewards = torch.zeros_like(forward_reward)
        full_qvel_imitation_rewards = torch.zeros_like(forward_reward)
        if self.full_state_imitation_enabled and self.reference.get("full_reset_qpos") is not None:
            full_ref_q = self.reference["full_reset_qpos"][target_phase].to(device=self.device, dtype=self.qpos.dtype)
            full_qpos_reward = dt * torch.exp(-torch.square((self.qpos - full_ref_q) / self.full_qpos_scale))
            if bool(self.full_qpos_mask.any().item()):
                full_qpos_imitation_rewards = full_qpos_reward[:, self.full_qpos_mask].mean(dim=1)
        if self.full_state_imitation_enabled and self.reference.get("full_reset_qvel") is not None:
            full_ref_dq = self.reference["full_reset_qvel"][target_phase].to(device=self.device, dtype=self.qvel.dtype)
            full_ref_dq = full_ref_dq.clone()
            full_ref_dq[:, self.pelvis_tx_qvel] = full_ref_dq[:, self.pelvis_tx_qvel] * speed_ratio
            full_qvel_reward = dt * torch.exp(-torch.square((self.qvel - full_ref_dq) / self.full_qvel_scale))
            if bool(self.full_qvel_mask.any().item()):
                full_qvel_imitation_rewards = full_qvel_reward[:, self.full_qvel_mask].mean(dim=1)
        root_orientation_reward = torch.zeros_like(forward_reward)
        root_xy_position_reward = torch.zeros_like(forward_reward)
        root_angvel_penalty = torch.zeros_like(forward_reward)
        lateral_vel_penalty = torch.zeros_like(forward_reward)
        lateral_drift_penalty = torch.zeros_like(forward_reward)
        foot_site_local_mimic_reward = torch.zeros_like(forward_reward)
        future_foot_site_local_mimic_reward = torch.zeros_like(forward_reward)
        footstep_target_reward = torch.zeros_like(forward_reward)
        footstep_landing_reward = torch.zeros_like(forward_reward)
        footstep_clearance_reward = torch.zeros_like(forward_reward)
        foot_contact_phase_reward = torch.zeros_like(forward_reward)
        foot_contact_phase_mismatch = torch.zeros_like(forward_reward)
        foot_lateral_target_penalty = torch.zeros_like(forward_reward)
        foot_toe_in_penalty = torch.zeros_like(forward_reward)
        foot_toe_in_penalty_r = torch.zeros_like(forward_reward)
        foot_toe_in_penalty_l = torch.zeros_like(forward_reward)
        foot_toe_in_angle_r = torch.zeros_like(forward_reward)
        foot_toe_in_angle_l = torch.zeros_like(forward_reward)
        knee_valgus_penalty = torch.zeros_like(forward_reward)
        knee_valgus_penalty_r = torch.zeros_like(forward_reward)
        knee_valgus_penalty_l = torch.zeros_like(forward_reward)
        knee_valgus_r = torch.zeros_like(forward_reward)
        knee_valgus_l = torch.zeros_like(forward_reward)
        foot_lateral_gap_penalty = torch.zeros_like(forward_reward)
        foot_lateral_gap = torch.zeros_like(forward_reward)
        foot_progression_imitation_reward = torch.zeros_like(forward_reward)
        foot_lateral_gap_imitation_reward = torch.zeros_like(forward_reward)
        handoff_state_imitation_reward = torch.zeros_like(forward_reward)
        flat_approach_progress_reward = torch.zeros_like(forward_reward)
        flat_approach_velocity_reward = torch.zeros_like(forward_reward)
        flat_approach_shortfall_penalty = torch.zeros_like(forward_reward)
        flat_approach_entry_distance = torch.zeros_like(forward_reward)
        if self.root_qpos_adr >= 0 and self.root_dof_adr >= 0:
            if self.reference.get("full_reset_qpos") is not None:
                full_ref_q = self.reference["full_reset_qpos"][target_phase].to(device=self.device, dtype=self.qpos.dtype)
                root_xy_delta = self.qpos[:, self.root_qpos_adr : self.root_qpos_adr + 2] - full_ref_q[
                    :, self.root_qpos_adr : self.root_qpos_adr + 2
                ]
                root_xy_position_reward = dt * torch.exp(
                    -torch.sum(torch.square(root_xy_delta / self.root_xy_position_scale), dim=1)
                )
                root_quat = torch.nn.functional.normalize(self.qpos[:, self.root_qpos_adr + 3 : self.root_qpos_adr + 7], dim=1)
                ref_quat = torch.nn.functional.normalize(full_ref_q[:, self.root_qpos_adr + 3 : self.root_qpos_adr + 7], dim=1)
                quat_dot = torch.abs(torch.sum(root_quat * ref_quat, dim=1)).clamp(max=1.0)
                quat_angle = 2.0 * torch.acos(quat_dot)
                root_orientation_reward = dt * torch.exp(-torch.square(quat_angle / self.root_orientation_scale))
                lateral_qpos_offset = 1 if self.forward_axis == "x" else 0
                lateral_drift = self.qpos[:, self.root_qpos_adr + lateral_qpos_offset] - full_ref_q[:, self.root_qpos_adr + lateral_qpos_offset]
                lateral_drift_penalty = -dt * torch.clamp(torch.square(lateral_drift / self.lateral_drift_scale), max=4.0)
                ref_foot = self.reference["foot_site_ref"][target_phase].to(device=self.device, dtype=self.site_xpos.dtype)
                cur_root_xyz = self.qpos[:, self.root_qpos_adr : self.root_qpos_adr + 3]
                ref_root_xyz = full_ref_q[:, self.root_qpos_adr : self.root_qpos_adr + 3].to(dtype=self.site_xpos.dtype)
                cur_foot_local = self.site_xpos[:, self.foot_site_indices, :] - cur_root_xyz[:, None, :].to(dtype=self.site_xpos.dtype)
                ref_foot_local = ref_foot - ref_root_xyz[:, None, :]
                foot_local_sq = torch.sum(torch.square((cur_foot_local - ref_foot_local) / self.foot_site_local_mimic_scale), dim=2)
                foot_site_local_mimic_reward = dt * torch.mean(torch.exp(-foot_local_sq), dim=1)
                foot_lateral_dim = 0 if self.forward_axis == "y" else 1
                foot_lateral_scale = max(float(self.config.get("footstep_target", {}).get("lateral_scale", 0.10)), 1e-6)
                foot_lateral_error = (
                    self.site_xpos[:, self.foot_site_indices, foot_lateral_dim] - ref_foot[:, :, foot_lateral_dim]
                )
                foot_lateral_target_penalty = -dt * torch.mean(
                    torch.clamp(torch.square(foot_lateral_error / foot_lateral_scale), max=4.0),
                    dim=1,
                )
                future_steps = max(0, int(self.config.get("imitation", {}).get("reference_reward_future_steps", 0) or 0))
                if future_steps > 0:
                    future_sum = torch.zeros_like(forward_reward)
                    for offset in range(1, future_steps + 1):
                        future_phase = reference_index(target_phase + offset, self.reference, self.config)
                        future_ref_q = self.reference["full_reset_qpos"][future_phase].to(device=self.device, dtype=self.qpos.dtype)
                        future_ref_foot = self.reference["foot_site_ref"][future_phase].to(
                            device=self.device,
                            dtype=self.site_xpos.dtype,
                        )
                        future_ref_root_xyz = future_ref_q[:, self.root_qpos_adr : self.root_qpos_adr + 3].to(dtype=self.site_xpos.dtype)
                        future_ref_foot_local = future_ref_foot - future_ref_root_xyz[:, None, :]
                        future_foot_local_sq = torch.sum(
                            torch.square((cur_foot_local - future_ref_foot_local) / self.foot_site_local_mimic_scale),
                            dim=2,
                        )
                        future_sum = future_sum + torch.mean(torch.exp(-future_foot_local_sq), dim=1)
                    future_foot_site_local_mimic_reward = dt * future_sum / float(future_steps)
            lateral_dof_offset = 1 if self.forward_axis == "x" else 0
            lateral_vel = self.qvel[:, self.root_dof_adr + lateral_dof_offset]
            lateral_vel_penalty = -dt * torch.clamp(torch.square(lateral_vel / self.lateral_velocity_scale), max=4.0)
            root_angvel = self.qvel[:, self.root_dof_adr + 3 : self.root_dof_adr + 6]
            root_angvel_penalty = -dt * torch.clamp(
                torch.mean(torch.square(root_angvel / self.root_angvel_scale), dim=1),
                max=4.0,
            )
            if self.flat_approach_enabled:
                pelvis_forward = self.qpos[:, self.pelvis_tx_qpos]
                forward_vel = self.qvel[:, self.pelvis_tx_qvel]
                active = (
                    (pelvis_forward >= (self.flat_approach_start_x - self.flat_approach_active_back))
                    & (pelvis_forward <= (self.flat_approach_entry_x + self.flat_approach_active_ahead))
                ).float()
                approach_len = max(self.flat_approach_entry_x - self.flat_approach_start_x, 1e-6)
                approach_progress = torch.clamp(
                    (pelvis_forward - self.flat_approach_start_x) / approach_len,
                    min=0.0,
                    max=1.0,
                )
                flat_approach_entry_distance = torch.relu(self.flat_approach_entry_x - pelvis_forward)
                flat_approach_progress_reward = dt * approach_progress * active
                flat_approach_velocity_reward = (
                    dt
                    * torch.clamp(forward_vel, min=0.0, max=self.flat_approach_target_velocity)
                    / self.flat_approach_target_velocity
                    * active
                )
                flat_approach_shortfall_penalty = (
                    -dt
                    * torch.clamp(
                        torch.square(flat_approach_entry_distance / self.flat_approach_distance_scale),
                        max=4.0,
                    )
                    * active
                )
        contact_phase_cfg = self.config.get("reward_contact_phase", {})
        if bool(contact_phase_cfg.get("enabled", False)) and "foot_contact_ref" in self.reference:
            foot = self.site_xpos[:, self.foot_site_indices, :]
            foot_forward = site_forward_coord_tensor(foot, self.config)
            foot_terrain = terrain_height_for_world_x_tensor(foot_forward, self.phase_idx, self.reference, self.config)
            foot_clearance = foot[:, :, 2] - foot_terrain
            contact_threshold = float(
                contact_phase_cfg.get(
                    "contact_z_threshold",
                    self.config.get("reference_contact", {}).get("z_threshold", 0.025),
                )
            )
            current_contact = (foot_clearance < contact_threshold).float()
            ref_contact = self.reference["foot_contact_ref"][target_phase].to(
                device=self.device,
                dtype=current_contact.dtype,
            )
            contact_mismatch = torch.abs(current_contact - ref_contact)
            phase_start = int(contact_phase_cfg.get("phase_start", 0))
            phase_end = int(contact_phase_cfg.get("phase_end", int(self.reference["length"])))
            active = ((target_phase >= phase_start) & (target_phase < phase_end)).float()
            if bool(contact_phase_cfg.get("stair_only", False)):
                pelvis_step = stair_step_index_tensor(
                    self.qpos[:, self.pelvis_tx_qpos].unsqueeze(1),
                    self.config,
                ).squeeze(1)
                active = active * (pelvis_step > 0.0).float()
            foot_contact_phase_mismatch = torch.mean(contact_mismatch, dim=1) * active
            foot_contact_phase_reward = dt * (1.0 - foot_contact_phase_mismatch) * active
        footstep_features = footstep_target_tensor(
            self.qpos,
            self.site_xpos,
            self.phase_idx,
            self.reference,
            self.config,
            pelvis_tx_qpos=self.pelvis_tx_qpos,
            foot_site_indices=self.foot_site_indices,
            target_phase=target_phase,
        )
        if footstep_features.shape[1] > 0:
            footstep_cfg = self.config.get("footstep_target", {})
            nfoot = len(FOOT_SITE_NAMES)
            target_forward_offset = footstep_features[:, 0:nfoot]
            time_to_contact = footstep_features[:, 2 * nfoot : 3 * nfoot]
            clearance_required = footstep_features[:, 3 * nfoot : 4 * nfoot]
            target_contact = footstep_features[:, 4 * nfoot : 5 * nfoot]
            target_forward_scale = max(float(footstep_cfg.get("forward_scale", 1.0)), 1e-6)
            reward_forward_scale = max(float(footstep_cfg.get("reward_forward_scale", 0.20)), 1e-6)
            landing_forward_scale = max(float(footstep_cfg.get("landing_forward_scale", reward_forward_scale)), 1e-6)
            clearance_reward_scale = max(float(footstep_cfg.get("reward_clearance_scale", 1.0)), 1e-6)
            foot = self.site_xpos[:, self.foot_site_indices, :]
            foot_forward = site_forward_coord_tensor(foot, self.config)
            foot_terrain = terrain_height_for_world_x_tensor(foot_forward, self.phase_idx, self.reference, self.config)
            foot_clearance = foot[:, :, 2] - foot_terrain
            contact_threshold = float(self.config.get("reference_contact", {}).get("z_threshold", 0.025))
            current_contact = (foot_clearance < contact_threshold).float()
            swing_mask = (time_to_contact > 0.0).float()
            target_forward = self.qpos[:, self.pelvis_tx_qpos].unsqueeze(1) + target_forward_offset * target_forward_scale
            forward_err = torch.abs(foot_forward - target_forward)
            target_score = torch.exp(-torch.square(forward_err / reward_forward_scale)) * target_contact
            landing_score = torch.exp(-torch.square(forward_err / landing_forward_scale)) * current_contact * target_contact
            clearance_score = torch.exp(-torch.square(clearance_required / clearance_reward_scale)) * swing_mask
            normalizer = torch.clamp(target_contact.sum(dim=1), min=1.0)
            swing_normalizer = torch.clamp(swing_mask.sum(dim=1), min=1.0)
            footstep_target_reward = dt * (target_score * swing_mask).sum(dim=1) / swing_normalizer
            footstep_landing_reward = dt * landing_score.sum(dim=1) / normalizer
            footstep_clearance_reward = dt * clearance_score.sum(dim=1) / swing_normalizer
        foot_shape_cfg = self.config.get("reward_foot_shape", {})
        if bool(foot_shape_cfg.get("enabled", False)) and int(self.foot_site_indices.numel()) >= 4:
            foot = self.site_xpos[:, self.foot_site_indices, :]
            foot_forward = site_forward_coord_tensor(foot, self.config)
            foot_lateral = site_lateral_coord_tensor(foot, self.config)
            foot_terrain = terrain_height_for_world_x_tensor(foot_forward, self.phase_idx, self.reference, self.config)
            foot_clearance = foot[:, :, 2] - foot_terrain
            contact_threshold = float(
                foot_shape_cfg.get(
                    "contact_z_threshold",
                    self.config.get("reference_contact", {}).get("z_threshold", 0.025),
                )
            )
            contact = foot_clearance < contact_threshold
            right_heel = foot[:, 0, :]
            right_toe = foot[:, 1, :]
            left_heel = foot[:, 2, :]
            left_toe = foot[:, 3, :]
            right_forward_delta = site_forward_coord_tensor(
                torch.stack([right_heel, right_toe], dim=1),
                self.config,
            )[:, 1] - site_forward_coord_tensor(torch.stack([right_heel, right_toe], dim=1), self.config)[:, 0]
            left_forward_delta = site_forward_coord_tensor(
                torch.stack([left_heel, left_toe], dim=1),
                self.config,
            )[:, 1] - site_forward_coord_tensor(torch.stack([left_heel, left_toe], dim=1), self.config)[:, 0]
            right_lateral_delta = foot_lateral[:, 1] - foot_lateral[:, 0]
            left_lateral_delta = foot_lateral[:, 3] - foot_lateral[:, 2]
            right_center_lateral = 0.5 * (foot_lateral[:, 0] + foot_lateral[:, 1])
            left_center_lateral = 0.5 * (foot_lateral[:, 2] + foot_lateral[:, 3])
            right_to_midline_sign = torch.sign(left_center_lateral - right_center_lateral)
            left_to_midline_sign = torch.sign(right_center_lateral - left_center_lateral)
            right_to_midline_sign = torch.where(
                right_to_midline_sign == 0.0,
                torch.ones_like(right_to_midline_sign),
                right_to_midline_sign,
            )
            left_to_midline_sign = torch.where(
                left_to_midline_sign == 0.0,
                -torch.ones_like(left_to_midline_sign),
                left_to_midline_sign,
            )
            right_toe_in_lateral = right_lateral_delta * right_to_midline_sign
            left_toe_in_lateral = left_lateral_delta * left_to_midline_sign
            forward_eps = max(float(foot_shape_cfg.get("forward_epsilon", 1e-4)), 1e-6)
            foot_toe_in_angle_r = torch.atan2(right_toe_in_lateral, torch.abs(right_forward_delta) + forward_eps)
            foot_toe_in_angle_l = torch.atan2(left_toe_in_lateral, torch.abs(left_forward_delta) + forward_eps)
            threshold_rad = float(foot_shape_cfg.get("toe_in_threshold_rad", 0.12))
            scale_rad = max(float(foot_shape_cfg.get("toe_in_scale_rad", 0.12)), 1e-6)
            right_toe_in_weight = float(foot_shape_cfg.get("toe_in_right_weight", 1.0))
            left_toe_in_weight = float(foot_shape_cfg.get("toe_in_left_weight", 1.0))
            right_stance = (contact[:, 0] | contact[:, 1]).float()
            left_stance = (contact[:, 2] | contact[:, 3]).float()
            right_excess = torch.relu(foot_toe_in_angle_r - threshold_rad)
            left_excess = torch.relu(foot_toe_in_angle_l - threshold_rad)
            toe_in_sq_r = torch.clamp(torch.square(right_excess / scale_rad), max=4.0) * right_stance
            toe_in_sq_l = torch.clamp(torch.square(left_excess / scale_rad), max=4.0) * left_stance
            toe_in_sq = right_toe_in_weight * toe_in_sq_r + left_toe_in_weight * toe_in_sq_l
            stance_count = torch.clamp(right_stance + left_stance, min=1.0)
            foot_toe_in_penalty = -dt * toe_in_sq / stance_count
            foot_toe_in_penalty_r = -dt * toe_in_sq_r
            foot_toe_in_penalty_l = -dt * toe_in_sq_l
            if bool(foot_shape_cfg.get("knee_valgus_enabled", False)) and int(self.limb_alignment_site_indices.numel()) == 4:
                limb_sites = self.site_xpos[:, self.limb_alignment_site_indices, :]
                limb_lateral = site_lateral_coord_tensor(limb_sites, self.config)
                right_hip_lateral = limb_lateral[:, 0]
                right_knee_lateral = limb_lateral[:, 1]
                left_hip_lateral = limb_lateral[:, 2]
                left_knee_lateral = limb_lateral[:, 3]
                right_inward_sign = torch.where(
                    right_to_midline_sign == 0.0,
                    torch.ones_like(right_to_midline_sign),
                    right_to_midline_sign,
                )
                left_inward_sign = torch.where(
                    left_to_midline_sign == 0.0,
                    -torch.ones_like(left_to_midline_sign),
                    left_to_midline_sign,
                )
                right_knee_line = 0.5 * (right_hip_lateral + right_center_lateral)
                left_knee_line = 0.5 * (left_hip_lateral + left_center_lateral)
                knee_valgus_r = (right_knee_lateral - right_knee_line) * right_inward_sign
                knee_valgus_l = (left_knee_lateral - left_knee_line) * left_inward_sign
                knee_valgus_threshold = float(foot_shape_cfg.get("knee_valgus_threshold_m", 0.02))
                knee_valgus_scale = max(float(foot_shape_cfg.get("knee_valgus_scale_m", 0.04)), 1e-6)
                knee_valgus_sq_r = (
                    torch.clamp(torch.square(torch.relu(knee_valgus_r - knee_valgus_threshold) / knee_valgus_scale), max=4.0)
                    * right_stance
                )
                knee_valgus_sq_l = (
                    torch.clamp(torch.square(torch.relu(knee_valgus_l - knee_valgus_threshold) / knee_valgus_scale), max=4.0)
                    * left_stance
                )
                knee_valgus_penalty_r = -dt * knee_valgus_sq_r
                knee_valgus_penalty_l = -dt * knee_valgus_sq_l
                knee_valgus_penalty = (knee_valgus_penalty_r + knee_valgus_penalty_l) / stance_count
            foot_lateral_gap = torch.abs(left_center_lateral - right_center_lateral)
            min_gap = float(foot_shape_cfg.get("min_lateral_gap", 0.10))
            gap_scale = max(float(foot_shape_cfg.get("lateral_gap_scale", 0.04)), 1e-6)
            gap_active = torch.clamp(right_stance + left_stance, min=0.0, max=1.0)
            gap_shortfall = torch.relu(min_gap - foot_lateral_gap)
            foot_lateral_gap_penalty = -dt * torch.clamp(torch.square(gap_shortfall / gap_scale), max=4.0) * gap_active
            ref_foot = self.reference["foot_site_ref"][target_phase].to(device=self.device, dtype=foot.dtype)
            ref_forward = site_forward_coord_tensor(ref_foot, self.config)
            ref_lateral = site_lateral_coord_tensor(ref_foot, self.config)
            ref_center_r = 0.5 * (ref_lateral[:, 0] + ref_lateral[:, 1])
            ref_center_l = 0.5 * (ref_lateral[:, 2] + ref_lateral[:, 3])
            ref_inward_r = torch.sign(ref_center_l - ref_center_r)
            ref_inward_l = torch.sign(ref_center_r - ref_center_l)
            ref_inward_r = torch.where(ref_inward_r == 0.0, torch.ones_like(ref_inward_r), ref_inward_r)
            ref_inward_l = torch.where(ref_inward_l == 0.0, -torch.ones_like(ref_inward_l), ref_inward_l)
            ref_toe_r = torch.atan2(
                (ref_lateral[:, 1] - ref_lateral[:, 0]) * ref_inward_r,
                torch.abs(ref_forward[:, 1] - ref_forward[:, 0]) + forward_eps,
            )
            ref_toe_l = torch.atan2(
                (ref_lateral[:, 3] - ref_lateral[:, 2]) * ref_inward_l,
                torch.abs(ref_forward[:, 3] - ref_forward[:, 2]) + forward_eps,
            )
            progression_scale = max(float(foot_shape_cfg.get("progression_imitation_scale_rad", 0.12)), 1e-6)
            toe_err_r = torch.atan2(
                torch.sin(foot_toe_in_angle_r - ref_toe_r),
                torch.cos(foot_toe_in_angle_r - ref_toe_r),
            )
            toe_err_l = torch.atan2(
                torch.sin(foot_toe_in_angle_l - ref_toe_l),
                torch.cos(foot_toe_in_angle_l - ref_toe_l),
            )
            progression_score = (
                torch.exp(-torch.square(toe_err_r / progression_scale)) * right_stance
                + torch.exp(-torch.square(toe_err_l / progression_scale)) * left_stance
            ) / stance_count
            foot_progression_imitation_reward = dt * progression_score
            ref_gap = torch.abs(ref_center_l - ref_center_r)
            gap_imitation_scale = max(float(foot_shape_cfg.get("gap_imitation_scale_m", 0.04)), 1e-6)
            foot_lateral_gap_imitation_reward = dt * torch.exp(
                -torch.square((foot_lateral_gap - ref_gap) / gap_imitation_scale)
            )
        handoff_cfg = self.config.get("reward_handoff_state", {})
        if bool(handoff_cfg.get("enabled", False)):
            center_x = float(handoff_cfg.get("center_x", 17.3))
            scale_x = max(float(handoff_cfg.get("scale_x", 0.45)), 1e-6)
            pelvis_forward = self.qpos[:, self.pelvis_tx_qpos]
            handoff_mask = torch.exp(-torch.square((pelvis_forward - center_x) / scale_x))
            handoff_state_imitation_reward = handoff_mask * (
                float(handoff_cfg.get("qpos_mix", 0.35)) * full_qpos_imitation_rewards
                + float(handoff_cfg.get("qvel_mix", 0.25)) * full_qvel_imitation_rewards
                + float(handoff_cfg.get("root_mix", 0.20)) * root_orientation_reward
                + float(handoff_cfg.get("foot_mix", 0.20)) * foot_site_local_mimic_reward
            )
        end_effector_imitation_reward = dt * torch.ones((self.nworld,), dtype=torch.float32, device=self.device)
        stair_contact_step_progress_reward = torch.zeros_like(forward_reward)
        stair_step_ahead_reward = torch.zeros_like(forward_reward)
        stair_contact_presence_reward = torch.zeros_like(forward_reward)
        stair_pelvis_step_progress_reward = torch.zeros_like(forward_reward)
        stair_step_gap_penalty = torch.zeros_like(forward_reward)
        stair_support_height_reward = torch.zeros_like(forward_reward)
        stair_support_height_penalty = torch.zeros_like(forward_reward)
        stair_foot_tread_target_reward = torch.zeros_like(forward_reward)
        stair_foot_tread_position_error = torch.zeros_like(forward_reward)
        stair_same_step_contact_penalty = torch.zeros_like(forward_reward)
        stair_step_separation_reward = torch.zeros_like(forward_reward)
        stair_pelvis_contact_lag_penalty = torch.zeros_like(forward_reward)
        stair_pelvis_drop_penalty = torch.zeros_like(forward_reward)
        stair_foot_tread_overshoot_penalty = torch.zeros_like(forward_reward)
        stair_top_platform_pelvis_reward = torch.zeros_like(forward_reward)
        stair_top_platform_contact_reward = torch.zeros_like(forward_reward)
        stair_top_platform_height_reward = torch.zeros_like(forward_reward)
        stair_top_platform_height_penalty = torch.zeros_like(forward_reward)
        stair_top_platform_forward_reward = torch.zeros_like(forward_reward)
        stair_top_platform_shortfall_penalty = torch.zeros_like(forward_reward)
        stair_top_platform_virtual_step_reward = torch.zeros_like(forward_reward)
        stair_top_platform_both_feet_contact_reward = torch.zeros_like(forward_reward)
        stair_top_platform_stable_reward = torch.zeros_like(forward_reward)
        stair_trailing_foot_forward_reward = torch.zeros_like(forward_reward)
        stair_trailing_foot_clearance_reward = torch.zeros_like(forward_reward)
        stair_trailing_foot_land_ready_reward = torch.zeros_like(forward_reward)
        stair_trailing_foot_contact_reward = torch.zeros_like(forward_reward)
        stair_trailing_foot_lag_penalty = torch.zeros_like(forward_reward)
        stair_trailing_foot_whole_forward_reward = torch.zeros_like(forward_reward)
        stair_trailing_foot_center_target_reward = torch.zeros_like(forward_reward)
        stair_trailing_foot_hover_penalty = torch.zeros_like(forward_reward)
        stair_contact_step_index = torch.zeros_like(forward_reward)
        stair_pelvis_step_index = torch.zeros_like(forward_reward)
        stair_forward_step_delta = torch.zeros_like(forward_reward)
        stair_cfg = self.config.get("reward_stair_progress", {})
        if bool(stair_cfg.get("enabled", False)):
            foot = self.site_xpos[:, self.foot_site_indices, :]
            foot_forward = site_forward_coord_tensor(foot, self.config)
            foot_height = terrain_height_for_world_x_tensor(foot_forward, self.phase_idx, self.reference, self.config)
            foot_clearance = foot[:, :, 2] - foot_height
            contact_threshold = float(stair_cfg.get("contact_z_threshold", self.config.get("reference_contact", {}).get("z_threshold", 0.025)))
            contact = foot_clearance < contact_threshold
            foot_step = stair_step_index_tensor(foot_forward, self.config)
            contact_step = torch.where(contact, foot_step, torch.zeros_like(foot_step))
            stair_contact_step_index = torch.amax(contact_step, dim=1)
            stair_pelvis_step_index = stair_step_index_tensor(
                self.qpos[:, self.pelvis_tx_qpos].unsqueeze(1),
                self.config,
            ).squeeze(1)
            stair_forward_step_delta = stair_contact_step_index - stair_pelvis_step_index
            step_count = max(float(stair_cfg.get("step_count", 8.0)), 1.0)
            has_contact = contact.any(dim=1).float()
            stair_contact_step_progress_reward = dt * torch.clamp(stair_contact_step_index / step_count, min=0.0, max=1.0)
            min_ahead = float(stair_cfg.get("min_ahead_steps", 0.25))
            ahead_scale = max(float(stair_cfg.get("ahead_scale", 0.35)), 1e-6)
            stair_step_ahead_reward = dt * torch.sigmoid((stair_forward_step_delta - min_ahead) / ahead_scale) * has_contact
            stair_contact_presence_reward = dt * has_contact
            stair_pelvis_step_progress_reward = dt * torch.clamp(stair_pelvis_step_index / step_count, min=0.0, max=1.0)
            max_gap = float(stair_cfg.get("max_foot_pelvis_gap_steps", 0.75))
            gap_scale = max(float(stair_cfg.get("foot_pelvis_gap_scale", 0.35)), 1e-6)
            excessive_gap = torch.relu(stair_forward_step_delta - max_gap)
            stair_step_gap_penalty = -dt * torch.clamp(torch.square(excessive_gap / gap_scale), max=4.0) * has_contact
            support_mask = ((stair_pelvis_step_index > 0.0) | (stair_contact_step_index > 0.0)).float()
            pelvis_height_above_terrain = self.qpos[:, self.pelvis_ty_qpos] - current_terrain_height_tensor(
                self.qpos,
                self.phase_idx,
                self.reference,
                self.config,
            )
            support_margin = float(stair_cfg.get("support_height_margin", 0.04))
            support_scale = max(float(stair_cfg.get("support_height_scale", 0.08)), 1e-6)
            support_target = float(self.safe_pelvis_height) + support_margin
            support_shortfall = torch.relu(support_target - pelvis_height_above_terrain)
            stair_support_height_reward = (
                dt
                * torch.sigmoid((pelvis_height_above_terrain - support_target) / support_scale)
                * support_mask
            )
            stair_support_height_penalty = -dt * torch.clamp(torch.square(support_shortfall / support_scale), max=4.0) * support_mask
            drop_threshold = float(stair_cfg.get("pelvis_drop_velocity_threshold", 0.35))
            drop_scale = max(float(stair_cfg.get("pelvis_drop_velocity_scale", 0.45)), 1e-6)
            downward_speed = torch.relu(-self.qvel[:, self.pelvis_ty_qvel] - drop_threshold)
            stair_pelvis_drop_penalty = -dt * torch.clamp(torch.square(downward_speed / drop_scale), max=4.0) * support_mask
            tread_target = float(stair_cfg.get("foot_tread_target", 0.62))
            tread_scale = max(float(stair_cfg.get("foot_tread_scale", 0.16)), 1e-6)
            stair_foot_contact = contact & (foot_step > 0.0)
            tread_progress = stair_tread_progress_tensor(foot_forward, self.config)
            tread_error = torch.abs(tread_progress - tread_target)
            tread_reward = torch.exp(-torch.square(tread_error / tread_scale))
            contact_count = torch.clamp(stair_foot_contact.float().sum(dim=1), min=1.0)
            stair_foot_tread_target_reward = dt * (tread_reward * stair_foot_contact.float()).sum(dim=1) / contact_count
            stair_foot_tread_position_error = (tread_error * stair_foot_contact.float()).sum(dim=1) / contact_count
            if int(stair_foot_contact.shape[1]) >= 4:
                side_split = int(stair_foot_contact.shape[1]) // 2
                right_contact = stair_foot_contact[:, :side_split]
                left_contact = stair_foot_contact[:, side_split:]
                right_has = right_contact.any(dim=1)
                left_has = left_contact.any(dim=1)
                right_step = torch.amax(torch.where(right_contact, foot_step[:, :side_split], torch.zeros_like(foot_step[:, :side_split])), dim=1)
                left_step = torch.amax(torch.where(left_contact, foot_step[:, side_split:], torch.zeros_like(foot_step[:, side_split:])), dim=1)
                same_step_tolerance = float(stair_cfg.get("same_step_contact_tolerance", 0.25))
                same_step_contact = (
                    right_has
                    & left_has
                    & (right_step > 0.0)
                    & (left_step > 0.0)
                    & (torch.abs(right_step - left_step) <= same_step_tolerance)
                ).float()
                same_step_scale = dt
                if bool(stair_cfg.get("same_step_contact_raw_penalty", False)):
                    same_step_scale = 1.0
                stair_same_step_contact_penalty = -same_step_scale * same_step_contact
                step_sep_target = float(stair_cfg.get("step_separation_target", 1.0))
                step_sep_scale = max(float(stair_cfg.get("step_separation_scale", 0.35)), 1e-6)
                step_sep = torch.abs(right_step - left_step)
                both_stair_contact = (right_has & left_has & (right_step > 0.0) & (left_step > 0.0)).float()
                stair_step_separation_reward = (
                    dt
                    * torch.sigmoid((step_sep - step_sep_target) / step_sep_scale)
                    * both_stair_contact
                )
            max_pelvis_lag = float(stair_cfg.get("max_pelvis_contact_lag_steps", 0.45))
            pelvis_lag_scale = max(float(stair_cfg.get("pelvis_contact_lag_scale", 0.35)), 1e-6)
            pelvis_contact_lag = torch.relu((stair_pelvis_step_index - stair_contact_step_index) - max_pelvis_lag)
            stair_pelvis_contact_lag_penalty = (
                -dt
                * torch.clamp(torch.square(pelvis_contact_lag / pelvis_lag_scale), max=4.0)
                * has_contact
                * support_mask
            )
            tread_max = float(stair_cfg.get("foot_tread_overshoot_max", 0.86))
            tread_overshoot_scale = max(float(stair_cfg.get("foot_tread_overshoot_scale", 0.10)), 1e-6)
            tread_overshoot = torch.relu(tread_progress - tread_max)
            tread_overshoot_sq = torch.clamp(torch.square(tread_overshoot / tread_overshoot_scale), max=4.0)
            stair_foot_tread_overshoot_penalty = -dt * (tread_overshoot_sq * stair_foot_contact.float()).sum(dim=1) / contact_count
            top_bounds: list[tuple[float, float]] = []
            for segment in list(self.config.get("terrain_course", {}).get("segments", [])):
                if str(segment.get("type", "flat")) != "stairs_box":
                    continue
                if float(segment.get("direction", 1.0)) < 0.0:
                    continue
                platform_depth = max(float(segment.get("platform_depth", 0.0)), 0.0)
                if platform_depth <= 0.0:
                    continue
                seg_x0 = float(segment.get("x0", 0.0))
                step_depth = max(float(segment.get("step_depth", 0.32)), 1e-6)
                steps = max(1, int(segment.get("steps", 1)))
                top_x0 = seg_x0 + float(steps) * step_depth
                top_bounds.append((top_x0, top_x0 + platform_depth))
            if top_bounds:
                top_x0 = min(item[0] for item in top_bounds)
                top_x1 = max(item[1] for item in top_bounds)
                top_margin = float(stair_cfg.get("top_platform_margin", 0.05))
                top_scale = max(float(stair_cfg.get("top_platform_progress_scale", 0.25)), 1e-6)
                top_contact_margin = float(stair_cfg.get("top_platform_contact_margin", 0.05))
                top_active_back = float(stair_cfg.get("top_platform_active_back", 0.20))
                pelvis_forward = self.qpos[:, self.pelvis_tx_qpos]
                pelvis_top_active = (pelvis_forward >= (top_x0 - top_active_back)) & (pelvis_forward <= top_x1)
                pelvis_on_platform = (pelvis_forward >= (top_x0 + top_margin)) & (pelvis_forward <= top_x1)
                platform_progress = torch.sigmoid((pelvis_forward - (top_x0 + top_margin)) / top_scale)
                stair_top_platform_pelvis_reward = dt * platform_progress * pelvis_top_active.float()
                foot_on_platform = contact & (foot_forward >= (top_x0 + top_contact_margin)) & (foot_forward <= top_x1)
                platform_contact = foot_on_platform.any(dim=1).float()
                stair_top_platform_contact_reward = dt * platform_contact
                top_height_margin = float(stair_cfg.get("top_platform_height_margin", stair_cfg.get("support_height_margin", 0.08)))
                top_height_scale = max(float(stair_cfg.get("top_platform_height_scale", stair_cfg.get("support_height_scale", 0.08))), 1e-6)
                top_height_target = float(self.safe_pelvis_height) + top_height_margin
                top_height_shortfall = torch.relu(top_height_target - pelvis_height_above_terrain)
                top_height_active = (pelvis_top_active | (platform_contact > 0.0)).float()
                stair_top_platform_height_reward = (
                    dt
                    * torch.sigmoid((pelvis_height_above_terrain - top_height_target) / top_height_scale)
                    * top_height_active
                )
                stair_top_platform_height_penalty = (
                    -dt
                    * torch.clamp(torch.square(top_height_shortfall / top_height_scale), max=4.0)
                    * top_height_active
                )
                top_speed_target = float(stair_cfg.get("top_platform_target_velocity", self.myoassist_target_velocity))
                top_speed_margin = float(stair_cfg.get("top_platform_velocity_margin", 0.05))
                top_speed_scale = max(float(stair_cfg.get("top_platform_velocity_scale", 0.5)), 1e-6)
                top_forward_vel = self.qvel[:, self.pelvis_tx_qvel]
                stair_top_platform_forward_reward = (
                    dt
                    * torch.clamp(top_forward_vel, min=0.0, max=top_speed_target)
                    / max(top_speed_target, 1e-6)
                    * top_height_active
                )
                top_speed_shortfall = torch.relu(top_speed_target - top_speed_margin - top_forward_vel)
                stair_top_platform_shortfall_penalty = (
                    -dt
                    * torch.clamp(torch.square(top_speed_shortfall / top_speed_scale), max=4.0)
                    * top_height_active
                )
                if int(foot_on_platform.shape[1]) >= 4:
                    side_split = int(foot_on_platform.shape[1]) // 2
                    right_platform = foot_on_platform[:, :side_split].any(dim=1)
                    left_platform = foot_on_platform[:, side_split:].any(dim=1)
                    right_platform_all = foot_on_platform[:, :side_split].all(dim=1)
                    left_platform_all = foot_on_platform[:, side_split:].all(dim=1)
                    one_side_platform = right_platform ^ left_platform
                    right_forward = torch.amax(foot_forward[:, :side_split], dim=1)
                    left_forward = torch.amax(foot_forward[:, side_split:], dim=1)
                    right_min_forward = torch.amin(foot_forward[:, :side_split], dim=1)
                    left_min_forward = torch.amin(foot_forward[:, side_split:], dim=1)
                    right_center_forward = torch.mean(foot_forward[:, :side_split], dim=1)
                    left_center_forward = torch.mean(foot_forward[:, side_split:], dim=1)
                    right_clearance = torch.amax(foot_clearance[:, :side_split], dim=1)
                    left_clearance = torch.amax(foot_clearance[:, side_split:], dim=1)
                    right_is_trailing = torch.where(
                        right_platform & ~left_platform,
                        torch.zeros_like(right_platform),
                        torch.where(
                            left_platform & ~right_platform,
                            torch.ones_like(right_platform),
                            right_center_forward <= left_center_forward,
                        ),
                    )
                    trailing_forward = torch.where(right_is_trailing, right_forward, left_forward)
                    trailing_min_forward = torch.where(right_is_trailing, right_min_forward, left_min_forward)
                    trailing_center_forward = torch.where(right_is_trailing, right_center_forward, left_center_forward)
                    trailing_clearance = torch.where(right_is_trailing, right_clearance, left_clearance)
                    trailing_side_all_contact = torch.where(right_is_trailing, right_platform_all, left_platform_all)
                    trailing_platform_contact = (right_platform & left_platform).float()
                    both_feet_contact = trailing_platform_contact
                    lagging_center_forward = torch.minimum(right_center_forward, left_center_forward)
                    trailing_active = (one_side_platform | pelvis_on_platform | pelvis_top_active).float()
                    trailing_target = top_x0 + float(stair_cfg.get("trailing_foot_platform_margin", top_contact_margin))
                    trailing_forward_scale = max(float(stair_cfg.get("trailing_foot_forward_scale", 0.18)), 1e-6)
                    trailing_clearance_target = float(stair_cfg.get("trailing_foot_clearance_target", 0.08))
                    trailing_clearance_scale = max(float(stair_cfg.get("trailing_foot_clearance_scale", 0.04)), 1e-6)
                    trailing_land_clearance_target = float(stair_cfg.get("trailing_foot_land_clearance_target", 0.018))
                    trailing_land_clearance_scale = max(float(stair_cfg.get("trailing_foot_land_clearance_scale", 0.035)), 1e-6)
                    trailing_lag_scale = max(float(stair_cfg.get("trailing_foot_lag_scale", 0.22)), 1e-6)
                    trailing_hover_clearance = float(stair_cfg.get("trailing_foot_hover_clearance", 0.045))
                    trailing_hover_scale = max(float(stair_cfg.get("trailing_foot_hover_scale", 0.05)), 1e-6)
                    trailing_whole_target = top_x0 + float(
                        stair_cfg.get("trailing_foot_whole_platform_margin", top_contact_margin)
                    )
                    trailing_whole_scale = max(float(stair_cfg.get("trailing_foot_whole_forward_scale", 0.14)), 1e-6)
                    trailing_center_target = top_x0 + float(stair_cfg.get("trailing_foot_center_target", 0.20))
                    trailing_center_scale = max(float(stair_cfg.get("trailing_foot_center_scale", 0.16)), 1e-6)
                    virtual_scale = max(float(stair_cfg.get("top_platform_virtual_step_scale", 0.18)), 1e-6)
                    trailing_progress = torch.sigmoid((trailing_forward - trailing_target) / trailing_forward_scale)
                    trailing_whole_progress = torch.sigmoid(
                        (trailing_min_forward - trailing_whole_target) / trailing_whole_scale
                    )
                    trailing_center_score = torch.exp(
                        -torch.square((trailing_center_forward - trailing_center_target) / trailing_center_scale)
                    )
                    virtual_step_progress = torch.sigmoid(
                        (lagging_center_forward - trailing_center_target) / virtual_scale
                    )
                    trailing_clearance_score = torch.sigmoid((trailing_clearance - trailing_clearance_target) / trailing_clearance_scale)
                    trailing_over_platform = (trailing_min_forward >= trailing_whole_target).float()
                    trailing_land_ready = torch.exp(
                        -torch.square((trailing_clearance - trailing_land_clearance_target) / trailing_land_clearance_scale)
                    )
                    trailing_hover = torch.relu(trailing_clearance - trailing_hover_clearance)
                    trailing_lag = torch.relu(trailing_target - trailing_forward)
                    stable_height_score = torch.sigmoid((pelvis_height_above_terrain - top_height_target) / top_height_scale)
                    stable_forward_score = torch.clamp(top_forward_vel, min=0.0, max=top_speed_target) / max(top_speed_target, 1e-6)
                    stair_trailing_foot_forward_reward = dt * trailing_progress * trailing_active
                    stair_trailing_foot_clearance_reward = dt * trailing_clearance_score * trailing_active * (1.0 - trailing_over_platform)
                    stair_trailing_foot_land_ready_reward = dt * trailing_land_ready * trailing_active * trailing_over_platform
                    stair_trailing_foot_contact_reward = dt * trailing_platform_contact * trailing_active
                    stair_trailing_foot_whole_forward_reward = dt * trailing_whole_progress * trailing_active
                    stair_trailing_foot_center_target_reward = dt * trailing_center_score * trailing_active
                    stair_top_platform_virtual_step_reward = dt * virtual_step_progress * trailing_active
                    stair_top_platform_both_feet_contact_reward = dt * both_feet_contact
                    stair_top_platform_stable_reward = (
                        dt
                        * both_feet_contact
                        * stable_height_score
                        * (0.5 + 0.5 * stable_forward_score)
                    )
                    stair_trailing_foot_lag_penalty = (
                        -dt
                        * torch.clamp(torch.square(trailing_lag / trailing_lag_scale), max=4.0)
                        * trailing_active
                    )
                    stair_trailing_foot_hover_penalty = (
                        -dt
                        * torch.clamp(torch.square(trailing_hover / trailing_hover_scale), max=4.0)
                        * trailing_active
                        * trailing_over_platform
                        * (1.0 - trailing_side_all_contact.float())
                    )
        nearest_trajectory_reward = torch.zeros_like(forward_reward)
        nearest_trajectory_pose_reward = torch.zeros_like(forward_reward)
        nearest_trajectory_direction_reward = torch.zeros_like(forward_reward)
        nearest_trajectory_amplitude_reward = torch.zeros_like(forward_reward)
        nearest_trajectory_phase_offset = torch.zeros_like(forward_reward)
        nearest_trajectory_best_error = torch.zeros_like(forward_reward)
        nearest_trajectory_agent_amp = torch.zeros_like(forward_reward)
        nearest_trajectory_ref_amp = torch.zeros_like(forward_reward)
        nearest_cfg = self.config.get("reward_nearest_trajectory", {})
        if bool(nearest_cfg.get("enabled", False)):
            foot = self.site_xpos[:, self.foot_site_indices, :]
            foot_forward = site_forward_coord_tensor(foot, self.config)
            foot_rel_x = foot_forward - self.qpos[:, self.pelvis_tx_qpos].unsqueeze(1)
            foot_z = foot[:, :, 2]
            before = max(0, int(nearest_cfg.get("search_before", 8) or 0))
            after = max(0, int(nearest_cfg.get("search_after", 24) or 0))
            best_phase = target_phase
            best_ref_q = ref_q
            best_ref_dq = ref_dq
            best_ref_foot = self.reference_foot(target_phase)
            best_err = self.reference_match_error(q, dq, foot_rel_x, foot_z, best_ref_q, best_ref_dq, best_ref_foot)
            for offset in range(-before, after + 1):
                if offset == 0:
                    continue
                phase = reference_index(target_phase + offset, self.reference, self.config)
                cand_q, cand_dq = self.reference_q_dq(phase)
                cand_foot = self.reference_foot(phase)
                err = self.reference_match_error(q, dq, foot_rel_x, foot_z, cand_q, cand_dq, cand_foot)
                better = err < best_err
                best_err = torch.where(better, err, best_err)
                best_phase = torch.where(better, phase, best_phase)
                best_ref_q = torch.where(better[:, None], cand_q, best_ref_q)
                best_ref_dq = torch.where(better[:, None], cand_dq, best_ref_dq)
                best_ref_foot = torch.where(better[:, None, None], cand_foot, best_ref_foot)

            lead = max(1, int(nearest_cfg.get("lead_steps", 2) or 2))
            lead_phase = reference_index(best_phase + lead, self.reference, self.config)
            lead_ref_q, _lead_ref_dq = self.reference_q_dq(lead_phase)
            ref_delta = lead_ref_q - best_ref_q
            agent_delta = dq * (dt * float(lead))
            weights = torch.clamp(self.myoassist_qpos_weights, min=0.0)
            if not bool(torch.any(weights > 0.0).item()):
                weights = torch.ones_like(weights)
            weights = weights / torch.clamp(torch.mean(weights), min=1e-6)
            weighted_ref_delta = ref_delta * weights.unsqueeze(0)
            weighted_agent_delta = agent_delta * weights.unsqueeze(0)
            ref_amp = torch.linalg.norm(weighted_ref_delta, dim=1)
            agent_amp = torch.linalg.norm(weighted_agent_delta, dim=1)
            dot = torch.sum(weighted_ref_delta * weighted_agent_delta, dim=1)
            denom = torch.clamp(ref_amp * agent_amp, min=1e-6)
            cosine = torch.clamp(dot / denom, min=-1.0, max=1.0)
            direction_floor = float(nearest_cfg.get("direction_floor", 0.0))
            nearest_trajectory_direction_reward = torch.clamp((cosine - direction_floor) / max(1.0 - direction_floor, 1e-6), 0.0, 1.0)
            amp_ratio = float(nearest_cfg.get("amp_ratio", 0.7))
            amp_scale = max(float(nearest_cfg.get("amp_scale", 0.08)), 1e-6)
            nearest_trajectory_amplitude_reward = torch.sigmoid((agent_amp - amp_ratio * ref_amp) / amp_scale)
            pose_scale = max(float(nearest_cfg.get("pose_scale", 1.5)), 1e-6)
            nearest_trajectory_pose_reward = torch.exp(-best_err / pose_scale)
            nearest_trajectory_reward = (
                dt
                * nearest_trajectory_pose_reward
                * nearest_trajectory_direction_reward
                * nearest_trajectory_amplitude_reward
            )
            nearest_trajectory_best_error = best_err
            nearest_trajectory_agent_amp = agent_amp
            nearest_trajectory_ref_amp = ref_amp
            ref_len = int(self.reference["length"])
            phase_delta = best_phase.to(torch.long) - target_phase.to(torch.long)
            nearest_trajectory_phase_offset = ((phase_delta + ref_len // 2) % ref_len - ref_len // 2).float()

        terms = {
            "forward_reward": forward_reward,
            "forward_velocity_reward": forward_velocity_reward,
            "forward_shortfall_penalty": forward_shortfall_penalty,
            "forward_speed_shortfall": speed_shortfall,
            "xalign_phase_lag_penalty": xalign_phase_lag_penalty,
            "xalign_phase_lag_steps": phase_lag_steps,
            "muscle_activation_penalty": muscle_activation_penalty,
            "muscle_activation_diff_penalty": muscle_activation_diff_penalty,
            "foot_force_penalty": foot_force_penalty,
            "joint_constraint_force_penalty": joint_constraint_force_penalty,
            "qpos_imitation_rewards": qpos_imitation_rewards,
            "qvel_imitation_rewards": qvel_imitation_rewards,
            "full_qpos_imitation_rewards": full_qpos_imitation_rewards,
            "full_qvel_imitation_rewards": full_qvel_imitation_rewards,
            "root_xy_position_reward": root_xy_position_reward,
            "root_orientation_reward": root_orientation_reward,
            "root_angvel_penalty": root_angvel_penalty,
            "lateral_vel_penalty": lateral_vel_penalty,
            "lateral_drift_penalty": lateral_drift_penalty,
            "foot_site_local_mimic_reward": foot_site_local_mimic_reward,
            "future_foot_site_local_mimic_reward": future_foot_site_local_mimic_reward,
            "footstep_target_reward": footstep_target_reward,
            "footstep_landing_reward": footstep_landing_reward,
            "footstep_clearance_reward": footstep_clearance_reward,
            "foot_contact_phase_reward": foot_contact_phase_reward,
            "foot_contact_phase_mismatch": foot_contact_phase_mismatch,
            "foot_lateral_target_penalty": foot_lateral_target_penalty,
            "foot_toe_in_penalty": foot_toe_in_penalty,
            "foot_toe_in_penalty_r": foot_toe_in_penalty_r,
            "foot_toe_in_penalty_l": foot_toe_in_penalty_l,
            "foot_toe_in_angle_r": foot_toe_in_angle_r,
            "foot_toe_in_angle_l": foot_toe_in_angle_l,
            "knee_valgus_penalty": knee_valgus_penalty,
            "knee_valgus_penalty_r": knee_valgus_penalty_r,
            "knee_valgus_penalty_l": knee_valgus_penalty_l,
            "knee_valgus_r": knee_valgus_r,
            "knee_valgus_l": knee_valgus_l,
            "foot_lateral_gap_penalty": foot_lateral_gap_penalty,
            "foot_lateral_gap": foot_lateral_gap,
            "foot_progression_imitation_reward": foot_progression_imitation_reward,
            "foot_lateral_gap_imitation_reward": foot_lateral_gap_imitation_reward,
            "handoff_state_imitation_reward": handoff_state_imitation_reward,
            "flat_approach_progress_reward": flat_approach_progress_reward,
            "flat_approach_velocity_reward": flat_approach_velocity_reward,
            "flat_approach_shortfall_penalty": flat_approach_shortfall_penalty,
            "flat_approach_entry_distance": flat_approach_entry_distance,
            "end_effector_imitation_reward": end_effector_imitation_reward,
            "stair_contact_step_progress_reward": stair_contact_step_progress_reward,
            "stair_step_ahead_reward": stair_step_ahead_reward,
            "stair_contact_presence_reward": stair_contact_presence_reward,
            "stair_pelvis_step_progress_reward": stair_pelvis_step_progress_reward,
            "stair_step_gap_penalty": stair_step_gap_penalty,
            "stair_support_height_reward": stair_support_height_reward,
            "stair_support_height_penalty": stair_support_height_penalty,
            "stair_foot_tread_target_reward": stair_foot_tread_target_reward,
            "stair_foot_tread_position_error": stair_foot_tread_position_error,
            "stair_same_step_contact_penalty": stair_same_step_contact_penalty,
            "stair_step_separation_reward": stair_step_separation_reward,
            "stair_pelvis_contact_lag_penalty": stair_pelvis_contact_lag_penalty,
            "stair_pelvis_drop_penalty": stair_pelvis_drop_penalty,
            "stair_foot_tread_overshoot_penalty": stair_foot_tread_overshoot_penalty,
            "stair_top_platform_pelvis_reward": stair_top_platform_pelvis_reward,
            "stair_top_platform_contact_reward": stair_top_platform_contact_reward,
            "stair_top_platform_height_reward": stair_top_platform_height_reward,
            "stair_top_platform_height_penalty": stair_top_platform_height_penalty,
            "stair_top_platform_forward_reward": stair_top_platform_forward_reward,
            "stair_top_platform_shortfall_penalty": stair_top_platform_shortfall_penalty,
            "stair_top_platform_virtual_step_reward": stair_top_platform_virtual_step_reward,
            "stair_top_platform_both_feet_contact_reward": stair_top_platform_both_feet_contact_reward,
            "stair_top_platform_stable_reward": stair_top_platform_stable_reward,
            "stair_trailing_foot_forward_reward": stair_trailing_foot_forward_reward,
            "stair_trailing_foot_clearance_reward": stair_trailing_foot_clearance_reward,
            "stair_trailing_foot_land_ready_reward": stair_trailing_foot_land_ready_reward,
            "stair_trailing_foot_contact_reward": stair_trailing_foot_contact_reward,
            "stair_trailing_foot_lag_penalty": stair_trailing_foot_lag_penalty,
            "stair_trailing_foot_whole_forward_reward": stair_trailing_foot_whole_forward_reward,
            "stair_trailing_foot_center_target_reward": stair_trailing_foot_center_target_reward,
            "stair_trailing_foot_hover_penalty": stair_trailing_foot_hover_penalty,
            "stair_contact_step_index": stair_contact_step_index,
            "stair_pelvis_step_index": stair_pelvis_step_index,
            "stair_forward_step_delta": stair_forward_step_delta,
            "x_aligned_reference": x_aligned_reference,
            "x_aligned_phase_offset": x_aligned_phase_offset,
            "x_aligned_target_phase": target_phase.float(),
            "nearest_trajectory_reward": nearest_trajectory_reward,
            "nearest_trajectory_pose_reward": nearest_trajectory_pose_reward,
            "nearest_trajectory_direction_reward": nearest_trajectory_direction_reward,
            "nearest_trajectory_amplitude_reward": nearest_trajectory_amplitude_reward,
            "nearest_trajectory_phase_offset": nearest_trajectory_phase_offset,
            "nearest_trajectory_best_error": nearest_trajectory_best_error,
            "nearest_trajectory_agent_amp": nearest_trajectory_agent_amp,
            "nearest_trajectory_ref_amp": nearest_trajectory_ref_amp,
            "myoassist_foot_force_r": foot_force[:, 0] if foot_force.ndim == 2 else torch.zeros_like(forward_reward),
            "myoassist_foot_force_l": foot_force[:, 1] if foot_force.ndim == 2 else torch.zeros_like(forward_reward),
            "recovery_mode": (self.recovery_mode_steps > 0).float(),
            "activation_mean": torch.mean(muscle_activation, dim=1),
            "activation_max": torch.amax(muscle_activation, dim=1),
            "normalized_action_mean": torch.mean(torch.clamp(action, -1.0, 1.0), dim=1),
            "normalized_action_std": torch.std(torch.clamp(action, -1.0, 1.0), dim=1, unbiased=False),
            "action_clip_fraction": torch.mean((torch.abs(action) > 1.0).float(), dim=1),
            "pelvis_height_above_terrain": self.qpos[:, self.pelvis_ty_qpos] - current_terrain_height_tensor(
                self.qpos,
                self.phase_idx,
                self.reference,
                self.config,
            ),
            "pelvis_tx_vel_abs_err": -torch.abs(self.qvel[:, self.pelvis_tx_qvel] - float(self.myoassist_target_velocity)),
            "pelvis_tx_vel_under_err": -torch.relu(float(self.myoassist_target_velocity) - self.qvel[:, self.pelvis_tx_qvel]),
        }
        reward = torch.zeros_like(forward_reward)
        for key, weight in self.myoassist_dense_weights.items():
            reward = reward + float(weight) * terms[key]
        normal_reward = reward
        if self.recovery_reward_enabled and self.recovery_reward_horizon_steps > 0 and self.recovery_reward_weights:
            recovery_reward = torch.zeros_like(forward_reward)
            for key, value in terms.items():
                recovery_reward = recovery_reward + self.recovery_reward_weights.get(key, 0.0) * value
            recovery_active = self.recovery_mode_steps > 0
            reward = torch.where(recovery_active, recovery_reward, normal_reward)
            terms["normal_reward"] = normal_reward
            terms["recovery_reward"] = recovery_reward
        terms["myoassist_dense"] = reward
        return reward, terms
