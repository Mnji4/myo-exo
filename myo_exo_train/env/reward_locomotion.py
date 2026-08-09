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


def ramp_alternating_step_terms(
    *,
    current_contact: torch.Tensor,
    previous_contact: torch.Tensor,
    foot_forward: torch.Tensor,
    previous_landing_x: torch.Tensor,
    previous_landing_side: torch.Tensor,
    episode_start: torch.Tensor,
    active: torch.Tensor,
    direction: float,
    min_advance_m: float,
    target_advance_m: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reward supported, alternating foot landings that advance along a ramp."""
    if current_contact.shape[1] != 2:
        raise ValueError("ramp step reward expects right/left contact columns")

    previous_landing_x[episode_start] = foot_forward[episode_start]
    previous_landing_side[episode_start] = -1

    landing = current_contact & (~previous_contact)
    supported_landing = torch.stack(
        (
            landing[:, 0] & previous_contact[:, 1],
            landing[:, 1] & previous_contact[:, 0],
        ),
        dim=1,
    )
    single_landing = supported_landing.sum(dim=1) == 1
    landing_side = torch.argmax(supported_landing.to(torch.long), dim=1)
    rows = torch.arange(current_contact.shape[0], device=current_contact.device)
    landing_x = foot_forward[rows, landing_side]
    previous_x = previous_landing_x[rows, landing_side]
    advance_m = float(direction) * (landing_x - previous_x)
    landing_active = active[rows, landing_side]
    alternating = (previous_landing_side < 0) | (
        landing_side != previous_landing_side
    )
    valid = (
        single_landing
        & landing_active
        & alternating
        & (~episode_start)
        & (advance_m >= float(min_advance_m))
    )
    reward = torch.clamp(
        advance_m / max(float(target_advance_m), 1.0e-6),
        min=0.0,
        max=1.0,
    ) * valid.to(foot_forward.dtype)

    update = single_landing & landing_active & (~episode_start)
    update_rows = rows[update]
    update_sides = landing_side[update]
    previous_landing_x[update_rows, update_sides] = landing_x[update]
    previous_landing_side[update_rows] = update_sides
    return (
        reward,
        valid.to(foot_forward.dtype),
        torch.where(valid, advance_m, torch.zeros_like(advance_m)),
    )


def foot_rollover_sequence_terms(
    *,
    current_contact: torch.Tensor,
    previous_contact: torch.Tensor,
    current_force: torch.Tensor,
    previous_force: torch.Tensor,
    state: torch.Tensor,
    elapsed_steps: torch.Tensor,
    airborne_steps: torch.Tensor,
    heel_stable_steps: torch.Tensor,
    heel_loading_excess: torch.Tensor,
    active: torch.Tensor,
    episode_start: torch.Tensor,
    min_heel_delay_steps: int,
    max_heel_delay_steps: int,
    release_steps: int,
    required_heel_stable_steps: int,
    max_heel_force_delta_n: float,
) -> dict[str, torch.Tensor]:
    """Score a supported toe-first touchdown completed by smooth heel loading.

    Contact columns are right heel, right toe, left heel, left toe. State is
    kept per foot: 0 ready, 1 waiting for heel, 2 completed, 3 invalid, and
    4 confirming stable heel contact. Toe contact alone never earns reward.
    """
    if current_contact.shape[1] != 4 or previous_contact.shape != current_contact.shape:
        raise ValueError("foot rollover expects four heel/toe contact columns")
    if current_force.shape != current_contact.shape or previous_force.shape != current_force.shape:
        raise ValueError("foot rollover force tensors must match contact shape")
    if state.shape != (current_contact.shape[0], 2):
        raise ValueError("foot rollover state must have shape [world, 2]")

    heel = current_contact[:, (0, 2)]
    toe = current_contact[:, (1, 3)]
    previous_heel = previous_contact[:, (0, 2)]
    previous_toe = previous_contact[:, (1, 3)]
    heel_force = current_force[:, (0, 2)]
    previous_heel_force = previous_force[:, (0, 2)]
    any_contact = heel | toe
    no_contact = ~any_contact
    opposite_support = torch.stack((any_contact[:, 1], any_contact[:, 0]), dim=1)
    starting = episode_start.unsqueeze(1).expand_as(state)
    previous_state = state.clone()

    airborne_steps.copy_(
        torch.where(no_contact, airborne_steps + 1, torch.zeros_like(airborne_steps))
    )
    released = airborne_steps >= max(1, int(release_steps))
    abandoned_before_heel = (previous_state == 1) & released & (~starting)
    unstable_heel = (previous_state == 4) & (~heel) & (~starting)
    state[released] = 0
    elapsed_steps[released] = 0
    heel_stable_steps[released] = 0
    heel_loading_excess[released] = 0.0

    state[starting & any_contact] = 2
    state[starting & no_contact] = 0
    elapsed_steps[starting] = 0
    heel_stable_steps[starting] = 0
    heel_loading_excess[starting] = 0.0

    toe_onset = toe & (~previous_toe)
    heel_onset = heel & (~previous_heel)
    waiting = (previous_state == 1) & (~released) & (~starting)
    elapsed_steps.copy_(
        torch.where(waiting, elapsed_steps + 1, elapsed_steps)
    )
    heel_after_toe = waiting & heel_onset
    valid_heel_delay = (
        heel_after_toe
        & (elapsed_steps >= max(1, int(min_heel_delay_steps)))
        & (elapsed_steps <= max(1, int(max_heel_delay_steps)))
    )
    invalid_heel_delay = heel_after_toe & (~valid_heel_delay)
    missing_heel_timeout = (
        waiting
        & (~heel)
        & (elapsed_steps > max(1, int(max_heel_delay_steps)))
    )
    force_delta = torch.relu(heel_force - previous_heel_force)
    force_delta_limit = max(float(max_heel_force_delta_n), 1.0e-6)
    loading_excess_step = torch.relu(force_delta - force_delta_limit) / force_delta_limit
    heel_loading_excess[valid_heel_delay] = loading_excess_step[valid_heel_delay]
    heel_stable_steps[valid_heel_delay] = 1
    state[valid_heel_delay] = 4
    state[invalid_heel_delay | missing_heel_timeout] = 3

    confirming = (previous_state == 4) & (~released) & (~starting)
    confirming_contact = confirming & heel
    heel_loading_excess[confirming_contact] += loading_excess_step[confirming_contact]
    heel_stable_steps[confirming_contact] += 1
    completed = confirming_contact & (
        heel_stable_steps >= max(1, int(required_heel_stable_steps))
    )
    state[completed] = 2
    state[unstable_heel] = 3

    ready = (state == 0) & (~starting)
    toe_first_raw = ready & toe_onset & (~heel) & active
    toe_first = toe_first_raw & opposite_support
    unsupported_toe = toe_first_raw & (~opposite_support)
    heel_first = ready & heel_onset & (~toe) & active
    simultaneous = ready & toe_onset & heel_onset & active
    inactive_landing = ready & (toe_onset | heel_onset) & (~active)
    state[toe_first] = 1
    elapsed_steps[toe_first] = 0
    heel_stable_steps[toe_first] = 0
    heel_loading_excess[toe_first] = 0.0
    state[unsupported_toe | heel_first] = 3
    state[simultaneous | inactive_landing] = 2

    missing_heel = (
        missing_heel_timeout
        | abandoned_before_heel
        | unstable_heel
        | unsupported_toe
    )
    completion_score = torch.exp(-heel_loading_excess) * completed.to(
        heel_loading_excess.dtype
    )
    loading_active = valid_heel_delay | confirming_contact
    loading_penalty = -(
        loading_excess_step * loading_active.to(loading_excess_step.dtype)
    ).sum(dim=1)
    dtype = current_force.dtype
    return {
        "foot_rollover_toe_first_reward": torch.zeros(
            current_contact.shape[0], dtype=dtype, device=current_contact.device
        ),
        "foot_rollover_heel_follow_reward": completion_score.sum(dim=1),
        "foot_rollover_heel_loading_penalty": loading_penalty,
        "foot_rollover_heel_first_penalty": -heel_first.to(dtype).sum(dim=1),
        "foot_rollover_missing_heel_penalty": -missing_heel.to(dtype).sum(dim=1),
        "foot_rollover_toe_first_event": toe_first.to(dtype).sum(dim=1),
        "foot_rollover_heel_follow_event": completed.to(dtype).sum(dim=1),
        "foot_rollover_heel_first_event": heel_first.to(dtype).sum(dim=1),
        "foot_rollover_missing_heel_event": missing_heel.to(dtype).sum(dim=1),
        "foot_rollover_unsupported_toe_event": unsupported_toe.to(dtype).sum(dim=1),
    }


class DenseLocomotionRewardMixin:
    def elastic_gait_cycle_terms(
        self,
        muscle_activation: torch.Tensor,
        prev_foot: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Score periodicity at observed foot-landings without fixing wall-clock phase."""
        zero = torch.zeros(
            self.nworld,
            dtype=self.qpos.dtype,
            device=self.device,
        )
        names = (
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
            "gait_sequence_half_cycle_valid",
            "gait_sequence_half_cycle_pose_rmse",
            "gait_sequence_half_cycle_velocity_rmse",
            "gait_sequence_half_cycle_activation_rmse",
            "gait_sequence_half_cycle_force_rmse_n",
            "gait_phase_force_target_penalty",
            "gait_dense_half_cycle_valid",
            "gait_dense_half_cycle_pose_rmse",
            "gait_dense_half_cycle_velocity_rmse",
            "gait_dense_half_cycle_activation_rmse",
            "gait_dense_half_cycle_force_rmse_n",
            "gait_phase_force_target_valid",
            "gait_phase_force_target_rmse_n",
            "gait_stance_impulse_balance_reward",
            "gait_stance_duration_balance_reward",
            "gait_stance_peak_force_balance_reward",
            "gait_stance_impulse_balance_penalty",
            "gait_stance_peak_force_balance_penalty",
            "gait_stance_balance_event",
            "gait_stance_impulse_relative_error",
            "gait_stance_duration_abs_error_steps",
            "gait_stance_peak_force_relative_error",
            "gait_alternation_reward",
            "gait_missing_landing_penalty",
            "gait_landing_event",
            "gait_cycle_event",
            "gait_half_cycle_event",
            "gait_alternating_event",
            "gait_repeated_side_event",
            "gait_cycle_interval_steps",
            "gait_half_cycle_interval_steps",
            "gait_cycle_interval_abs_error_steps",
            "gait_half_cycle_interval_abs_error_steps",
            "gait_half_cycle_balance_event",
            "gait_half_cycle_balance_abs_error_steps",
        )
        terms = {name: zero.clone() for name in names}
        if not self.gait_cycle_enabled:
            return terms

        foot = self.site_xpos[:, self.foot_site_indices, :]
        foot_forward = site_forward_coord_tensor(foot, self.config)
        previous_forward = site_forward_coord_tensor(prev_foot, self.config)
        foot_height = terrain_height_for_world_x_tensor(
            foot_forward,
            self.phase_idx,
            self.reference,
            self.config,
        )
        previous_height = terrain_height_for_world_x_tensor(
            previous_forward,
            self.phase_idx,
            self.reference,
            self.config,
        )
        threshold = float(
            self.config.get("reference_contact", {}).get(
                "z_threshold",
                0.025,
            )
        )
        site_contact = (foot[:, :, 2] - foot_height) < threshold
        previous_site_contact = (
            prev_foot[:, :, 2] - previous_height
        ) < threshold
        split = int(site_contact.shape[1]) // 2
        side_contact = torch.stack(
            (
                site_contact[:, :split].any(dim=1),
                site_contact[:, split:].any(dim=1),
            ),
            dim=1,
        )
        previous_side_contact = torch.stack(
            (
                previous_site_contact[:, :split].any(dim=1),
                previous_site_contact[:, split:].any(dim=1),
            ),
            dim=1,
        )
        active = (
            (self.phase_idx >= self.gait_cycle_phase_start)
            & (self.phase_idx < self.gait_cycle_phase_end)
        )
        foot_force = torch.abs(
            self.sensordata.index_select(1, self.ground_force_sensor_indices)
        )
        force_split = int(foot_force.shape[1]) // 2
        side_force = torch.stack(
            (
                foot_force[:, :force_split].sum(dim=1),
                foot_force[:, force_split:].sum(dim=1),
            ),
            dim=1,
        )
        if self.gait_phase_force_target_enabled:
            target_index = torch.remainder(
                self.phase_idx,
                int(self.gait_phase_force_target.shape[0]),
            )
            target_side_force = self.gait_phase_force_target.index_select(
                0, target_index
            )
            phase_force_rmse = torch.sqrt(
                torch.mean(torch.square(side_force - target_side_force), dim=1)
            )
            phase_force_valid = active.to(self.qpos.dtype)
            terms["gait_phase_force_target_penalty"] = (
                -self.dt
                * torch.clamp(
                    phase_force_rmse / self.gait_phase_force_target_scale_n,
                    max=2.0,
                )
                * phase_force_valid
            )
            terms["gait_phase_force_target_valid"] = phase_force_valid
            terms["gait_phase_force_target_rmse_n"] = (
                phase_force_rmse * phase_force_valid
            )
        landing = side_contact & (~previous_side_contact)
        single_landing = landing.sum(dim=1) == 1
        landing_side = torch.argmax(landing.to(torch.long), dim=1)
        since_last_event = self.episode_step - self.gait_cycle_last_event_step
        refractory_ready = (
            (self.gait_cycle_last_event_step < 0)
            | (
                since_last_event
                >= self.gait_cycle_min_landing_interval_steps
            )
        )
        accepted = single_landing & active & refractory_ready
        rows = torch.arange(self.nworld, device=self.device)
        current_qpos = self.qpos.index_select(
            1,
            self.gait_cycle_qpos_indices,
        )
        current_qvel = self.qvel.index_select(
            1,
            self.gait_cycle_qvel_indices,
        )

        if self.gait_sequence_half_cycle_enabled:
            collecting = active & self.gait_sequence_started
            collecting_rows = rows[collecting]
            write_index = torch.clamp(
                self.gait_sequence_current_length,
                min=0,
                max=self.gait_sequence_half_cycle_capacity - 1,
            )
            collecting_index = write_index[collecting]
            self.gait_sequence_current_qpos[
                collecting_rows, collecting_index
            ] = current_qpos[collecting]
            self.gait_sequence_current_qvel[
                collecting_rows, collecting_index
            ] = current_qvel[collecting]
            self.gait_sequence_current_activation[
                collecting_rows, collecting_index
            ] = muscle_activation[collecting]
            self.gait_sequence_current_side_force[
                collecting_rows, collecting_index
            ] = side_force[collecting]
            self.gait_sequence_current_overflow |= collecting & (
                self.gait_sequence_current_length
                >= self.gait_sequence_half_cycle_capacity
            )
            self.gait_sequence_current_length.copy_(
                torch.where(
                    collecting,
                    torch.clamp(
                        self.gait_sequence_current_length + 1,
                        max=self.gait_sequence_half_cycle_capacity + 1,
                    ),
                    self.gait_sequence_current_length,
                )
            )

            sequence_valid = (
                accepted
                & self.gait_sequence_started
                & self.gait_sequence_previous_valid
                & (~self.gait_sequence_current_overflow)
                & (self.gait_sequence_current_length >= 2)
                & (self.gait_sequence_previous_length >= 2)
                & (
                    self.gait_sequence_current_start_side
                    != self.gait_sequence_previous_start_side
                )
            )

            def normalized_sequence(
                values: torch.Tensor,
                lengths: torch.Tensor,
            ) -> torch.Tensor:
                sample_phase = torch.linspace(
                    0.0,
                    1.0,
                    self.gait_sequence_half_cycle_points,
                    dtype=values.dtype,
                    device=self.device,
                ).unsqueeze(0)
                position = sample_phase * torch.clamp(
                    lengths - 1, min=0
                ).to(values.dtype).unsqueeze(1)
                lower = torch.floor(position).long()
                upper = torch.minimum(
                    lower + 1,
                    torch.clamp(lengths - 1, min=0).unsqueeze(1),
                )
                fraction = (position - lower.to(position.dtype)).unsqueeze(2)
                lower_values = values.gather(
                    1,
                    lower.unsqueeze(2).expand(-1, -1, values.shape[2]),
                )
                upper_values = values.gather(
                    1,
                    upper.unsqueeze(2).expand(-1, -1, values.shape[2]),
                )
                return lower_values + fraction * (upper_values - lower_values)

            current_length = torch.clamp(
                self.gait_sequence_current_length,
                min=1,
                max=self.gait_sequence_half_cycle_capacity,
            )
            previous_length = torch.clamp(
                self.gait_sequence_previous_length,
                min=1,
                max=self.gait_sequence_half_cycle_capacity,
            )
            current_sequence_qpos = normalized_sequence(
                self.gait_sequence_current_qpos, current_length
            )
            current_sequence_qvel = normalized_sequence(
                self.gait_sequence_current_qvel, current_length
            )
            current_sequence_activation = normalized_sequence(
                self.gait_sequence_current_activation, current_length
            )
            current_sequence_force = normalized_sequence(
                self.gait_sequence_current_side_force, current_length
            )
            previous_sequence_qpos = normalized_sequence(
                self.gait_sequence_previous_qpos, previous_length
            ).index_select(2, self.gait_cycle_state_mirror_perm)
            previous_sequence_qvel = normalized_sequence(
                self.gait_sequence_previous_qvel, previous_length
            ).index_select(2, self.gait_cycle_state_mirror_perm)
            previous_sequence_activation = normalized_sequence(
                self.gait_sequence_previous_activation, previous_length
            ).index_select(2, self.gait_cycle_activation_mirror_perm)
            previous_sequence_force = normalized_sequence(
                self.gait_sequence_previous_side_force, previous_length
            ).flip(dims=(2,))

            sequence_pose_rmse = torch.sqrt(
                torch.mean(
                    torch.square(
                        current_sequence_qpos - previous_sequence_qpos
                    ),
                    dim=(1, 2),
                )
            )
            sequence_velocity_rmse = torch.sqrt(
                torch.mean(
                    torch.square(
                        current_sequence_qvel - previous_sequence_qvel
                    ),
                    dim=(1, 2),
                )
            )
            sequence_activation_rmse = torch.sqrt(
                torch.mean(
                    torch.square(
                        current_sequence_activation
                        - previous_sequence_activation
                    ),
                    dim=(1, 2),
                )
            )
            sequence_force_rmse = torch.sqrt(
                torch.mean(
                    torch.square(
                        current_sequence_force - previous_sequence_force
                    ),
                    dim=(1, 2),
                )
            )
            sequence_valid_float = sequence_valid.to(self.qpos.dtype)
            terms["gait_sequence_half_cycle_pose_reward"] = (
                torch.exp(
                    -torch.square(
                        sequence_pose_rmse / self.gait_cycle_pose_scale
                    )
                )
                * sequence_valid_float
            )
            terms["gait_sequence_half_cycle_velocity_reward"] = (
                torch.exp(
                    -torch.square(
                        sequence_velocity_rmse
                        / self.gait_cycle_velocity_scale
                    )
                )
                * sequence_valid_float
            )
            terms["gait_sequence_half_cycle_activation_reward"] = (
                torch.exp(
                    -torch.square(
                        sequence_activation_rmse
                        / self.gait_cycle_activation_scale
                    )
                )
                * sequence_valid_float
            )
            terms["gait_sequence_half_cycle_force_reward"] = (
                torch.exp(
                    -torch.square(
                        sequence_force_rmse
                        / self.gait_sequence_half_cycle_force_scale_n
                    )
                )
                * sequence_valid_float
            )
            terms["gait_sequence_half_cycle_valid"] = sequence_valid_float
            terms["gait_sequence_half_cycle_pose_rmse"] = (
                sequence_pose_rmse * sequence_valid_float
            )
            terms["gait_sequence_half_cycle_velocity_rmse"] = (
                sequence_velocity_rmse * sequence_valid_float
            )
            terms["gait_sequence_half_cycle_activation_rmse"] = (
                sequence_activation_rmse * sequence_valid_float
            )
            terms["gait_sequence_half_cycle_force_rmse_n"] = (
                sequence_force_rmse * sequence_valid_float
            )

            completed = (
                accepted
                & self.gait_sequence_started
                & (~self.gait_sequence_current_overflow)
                & (self.gait_sequence_current_length >= 2)
            )
            completed_rows = rows[completed]
            self.gait_sequence_previous_qpos[completed_rows] = (
                self.gait_sequence_current_qpos[completed_rows]
            )
            self.gait_sequence_previous_qvel[completed_rows] = (
                self.gait_sequence_current_qvel[completed_rows]
            )
            self.gait_sequence_previous_activation[completed_rows] = (
                self.gait_sequence_current_activation[completed_rows]
            )
            self.gait_sequence_previous_side_force[completed_rows] = (
                self.gait_sequence_current_side_force[completed_rows]
            )
            self.gait_sequence_previous_length[completed_rows] = (
                self.gait_sequence_current_length[completed_rows]
            )
            self.gait_sequence_previous_start_side[completed_rows] = (
                self.gait_sequence_current_start_side[completed_rows]
            )
            self.gait_sequence_previous_valid[completed_rows] = True

            accepted_rows = rows[accepted]
            self.gait_sequence_current_length[accepted_rows] = 1
            self.gait_sequence_current_start_side[accepted_rows] = (
                landing_side[accepted_rows]
            )
            self.gait_sequence_started[accepted_rows] = True
            self.gait_sequence_current_overflow[accepted_rows] = False
            self.gait_sequence_current_qpos[accepted_rows, 0] = (
                current_qpos[accepted_rows]
            )
            self.gait_sequence_current_qvel[accepted_rows, 0] = (
                current_qvel[accepted_rows]
            )
            self.gait_sequence_current_activation[accepted_rows, 0] = (
                muscle_activation[accepted_rows]
            )
            self.gait_sequence_current_side_force[accepted_rows, 0] = (
                side_force[accepted_rows]
            )

        if self.gait_dense_half_cycle_enabled:
            history_index = torch.remainder(
                self.episode_step,
                self.gait_dense_half_cycle_steps,
            )
            history_rows = torch.arange(self.nworld, device=self.device)
            past_qpos = self.gait_dense_half_cycle_qpos[
                history_rows, history_index
            ].index_select(1, self.gait_cycle_state_mirror_perm)
            past_qvel = self.gait_dense_half_cycle_qvel[
                history_rows, history_index
            ].index_select(1, self.gait_cycle_state_mirror_perm)
            past_activation = self.gait_dense_half_cycle_activation[
                history_rows, history_index
            ].index_select(1, self.gait_cycle_activation_mirror_perm)
            past_side_force = self.gait_dense_half_cycle_side_force[
                history_rows, history_index
            ].flip(dims=(1,))
            dense_valid = active & (
                self.episode_step >= self.gait_dense_half_cycle_steps
            )
            dense_pose_rmse = torch.sqrt(
                torch.mean(torch.square(current_qpos - past_qpos), dim=1)
            )
            dense_velocity_rmse = torch.sqrt(
                torch.mean(torch.square(current_qvel - past_qvel), dim=1)
            )
            dense_activation_rmse = torch.sqrt(
                torch.mean(
                    torch.square(muscle_activation - past_activation),
                    dim=1,
                )
            )
            dense_force_rmse = torch.sqrt(
                torch.mean(torch.square(side_force - past_side_force), dim=1)
            )
            valid_float = dense_valid.to(self.qpos.dtype)
            terms["gait_dense_half_cycle_pose_reward"] = (
                self.dt
                * torch.exp(
                    -torch.square(
                        dense_pose_rmse / self.gait_cycle_pose_scale
                    )
                )
                * valid_float
            )
            terms["gait_dense_half_cycle_velocity_reward"] = (
                self.dt
                * torch.exp(
                    -torch.square(
                        dense_velocity_rmse
                        / self.gait_cycle_velocity_scale
                    )
                )
                * valid_float
            )
            terms["gait_dense_half_cycle_activation_reward"] = (
                self.dt
                * torch.exp(
                    -torch.square(
                        dense_activation_rmse
                        / self.gait_cycle_activation_scale
                    )
                )
                * valid_float
            )
            terms["gait_dense_half_cycle_force_balance_penalty"] = (
                -self.dt
                * torch.clamp(
                    dense_force_rmse / self.gait_dense_half_cycle_force_scale_n,
                    max=2.0,
                )
                * valid_float
            )
            terms["gait_dense_half_cycle_valid"] = valid_float
            terms["gait_dense_half_cycle_pose_rmse"] = (
                dense_pose_rmse * valid_float
            )
            terms["gait_dense_half_cycle_velocity_rmse"] = (
                dense_velocity_rmse * valid_float
            )
            terms["gait_dense_half_cycle_activation_rmse"] = (
                dense_activation_rmse * valid_float
            )
            terms["gait_dense_half_cycle_force_rmse_n"] = (
                dense_force_rmse * valid_float
            )
            self.gait_dense_half_cycle_qpos[
                history_rows, history_index
            ] = current_qpos
            self.gait_dense_half_cycle_qvel[
                history_rows, history_index
            ] = current_qvel
            self.gait_dense_half_cycle_activation[
                history_rows, history_index
            ] = muscle_activation
            self.gait_dense_half_cycle_side_force[
                history_rows, history_index
            ] = side_force

        if self.gait_stance_balance_enabled:
            active_contact = side_contact & active[:, None]
            self.gait_stance_started |= active_contact
            self.gait_stance_impulse += (
                side_force * self.dt * active_contact.to(side_force.dtype)
            )
            self.gait_stance_duration_steps += active_contact.to(torch.long)
            self.gait_stance_peak_force.copy_(
                torch.where(
                    active_contact,
                    torch.maximum(self.gait_stance_peak_force, side_force),
                    self.gait_stance_peak_force,
                )
            )

            liftoff = (
                previous_side_contact
                & (~side_contact)
                & active[:, None]
                & self.gait_stance_started
            )
            previous_impulse = self.gait_last_stance_impulse.clone()
            previous_duration = self.gait_last_stance_duration_steps.clone()
            previous_peak = self.gait_last_stance_peak_force.clone()
            previous_valid = self.gait_last_stance_valid.clone()
            stance_balance_event = torch.zeros_like(active)
            for side in range(2):
                other = 1 - side
                completed = (
                    liftoff[:, side]
                    & (
                        self.gait_stance_duration_steps[:, side]
                        >= self.gait_stance_min_steps
                    )
                )
                compare = completed & previous_valid[:, other]
                current_impulse = self.gait_stance_impulse[:, side]
                current_duration = self.gait_stance_duration_steps[:, side].float()
                current_peak = self.gait_stance_peak_force[:, side]
                impulse_denominator = torch.clamp(
                    0.5 * (current_impulse + previous_impulse[:, other]),
                    min=1.0e-6,
                )
                peak_denominator = torch.clamp(
                    0.5 * (current_peak + previous_peak[:, other]),
                    min=1.0e-6,
                )
                impulse_error = torch.abs(
                    current_impulse - previous_impulse[:, other]
                ) / impulse_denominator
                duration_error = torch.abs(
                    current_duration - previous_duration[:, other].float()
                )
                peak_error = torch.abs(
                    current_peak - previous_peak[:, other]
                ) / peak_denominator
                compare_float = compare.to(self.qpos.dtype)
                terms["gait_stance_impulse_balance_reward"] += (
                    1.0
                    / (
                        1.0
                        + torch.square(
                            impulse_error / self.gait_stance_impulse_relative_scale
                        )
                    )
                    * compare_float
                )
                terms["gait_stance_duration_balance_reward"] += (
                    1.0
                    / (
                        1.0
                        + torch.square(
                            duration_error / self.gait_stance_duration_scale_steps
                        )
                    )
                    * compare_float
                )
                terms["gait_stance_peak_force_balance_reward"] += (
                    1.0
                    / (
                        1.0
                        + torch.square(
                            peak_error / self.gait_stance_peak_force_relative_scale
                        )
                    )
                    * compare_float
                )
                terms["gait_stance_impulse_balance_penalty"] += (
                    -torch.clamp(impulse_error, max=2.0) * compare_float
                )
                terms["gait_stance_peak_force_balance_penalty"] += (
                    -torch.clamp(peak_error, max=2.0) * compare_float
                )
                terms["gait_stance_impulse_relative_error"] += (
                    impulse_error * compare_float
                )
                terms["gait_stance_duration_abs_error_steps"] += (
                    duration_error * compare_float
                )
                terms["gait_stance_peak_force_relative_error"] += (
                    peak_error * compare_float
                )
                stance_balance_event |= compare

                completed_rows = rows[completed]
                self.gait_last_stance_impulse[completed_rows, side] = (
                    current_impulse[completed]
                )
                self.gait_last_stance_duration_steps[completed_rows, side] = (
                    self.gait_stance_duration_steps[completed_rows, side]
                )
                self.gait_last_stance_peak_force[completed_rows, side] = (
                    current_peak[completed]
                )
                self.gait_last_stance_valid[completed_rows, side] = True

                liftoff_rows = rows[liftoff[:, side]]
                self.gait_stance_started[liftoff_rows, side] = False
                self.gait_stance_impulse[liftoff_rows, side] = 0.0
                self.gait_stance_duration_steps[liftoff_rows, side] = 0
                self.gait_stance_peak_force[liftoff_rows, side] = 0.0

            terms["gait_stance_balance_event"] = stance_balance_event.float()
            if self.gait_stance_event_hold_steps > 1:
                previous_hold = self.gait_stance_event_hold_remaining > 0
                hold_scale = 1.0 / float(self.gait_stance_event_hold_steps)
                for name, held_value in self.gait_stance_held_terms.items():
                    event_value = terms[name]
                    held_value.copy_(
                        torch.where(stance_balance_event, event_value, held_value)
                    )
                    terms[name] = (
                        torch.where(
                            stance_balance_event,
                            event_value,
                            torch.where(
                                previous_hold,
                                held_value,
                                torch.zeros_like(event_value),
                            ),
                        )
                        * hold_scale
                    )
                self.gait_stance_event_hold_remaining.copy_(
                    torch.where(
                        stance_balance_event,
                        torch.full_like(
                            self.gait_stance_event_hold_remaining,
                            self.gait_stance_event_hold_steps - 1,
                        ),
                        torch.clamp(
                            self.gait_stance_event_hold_remaining - 1,
                            min=0,
                        ),
                    )
                )

        for side in range(2):
            event = accepted & (landing_side == side)
            other = 1 - side
            same_valid = event & self.gait_cycle_last_landing_valid[:, side]
            half_valid = event & self.gait_cycle_last_landing_valid[:, other]
            cycle_interval = (
                self.episode_step
                - self.gait_cycle_last_landing_step[:, side]
            )
            half_interval = (
                self.episode_step
                - self.gait_cycle_last_landing_step[:, other]
            )
            cycle_score = torch.exp(
                -torch.square(
                    (
                        cycle_interval.float()
                        - float(self.gait_cycle_target_steps)
                    )
                    / self.gait_cycle_tolerance_steps
                )
            )
            half_score = torch.exp(
                -torch.square(
                    (
                        half_interval.float()
                        - float(self.gait_half_cycle_target_steps)
                    )
                    / self.gait_half_cycle_tolerance_steps
                )
            )
            balance_valid = (
                half_valid & self.gait_cycle_last_half_interval_valid
            )
            balance_error = torch.abs(
                half_interval.float()
                - self.gait_cycle_last_half_interval
            )
            balance_score = torch.exp(
                -torch.square(
                    balance_error
                    / self.gait_half_cycle_balance_tolerance_steps
                )
            )
            terms["gait_cycle_cadence_reward"] += (
                cycle_score * same_valid.float()
            )
            terms["gait_half_cycle_cadence_reward"] += (
                half_score * half_valid.float()
            )
            terms["gait_half_cycle_balance_reward"] += (
                balance_score * balance_valid.float()
            )
            terms["gait_cycle_interval_steps"] += torch.where(
                same_valid,
                cycle_interval.float(),
                torch.zeros_like(cycle_interval, dtype=self.qpos.dtype),
            )
            terms["gait_half_cycle_interval_steps"] += torch.where(
                half_valid,
                half_interval.float(),
                torch.zeros_like(half_interval, dtype=self.qpos.dtype),
            )
            terms["gait_cycle_event"] += same_valid.float()
            terms["gait_half_cycle_event"] += half_valid.float()
            terms["gait_cycle_interval_abs_error_steps"] += torch.where(
                same_valid,
                torch.abs(
                    cycle_interval.float()
                    - float(self.gait_cycle_target_steps)
                ),
                torch.zeros_like(cycle_interval, dtype=self.qpos.dtype),
            )
            terms["gait_half_cycle_interval_abs_error_steps"] += torch.where(
                half_valid,
                torch.abs(
                    half_interval.float()
                    - float(self.gait_half_cycle_target_steps)
                ),
                torch.zeros_like(half_interval, dtype=self.qpos.dtype),
            )
            terms["gait_half_cycle_balance_event"] += balance_valid.float()
            terms["gait_half_cycle_balance_abs_error_steps"] += torch.where(
                balance_valid,
                balance_error,
                torch.zeros_like(balance_error),
            )

            previous_qpos = self.gait_cycle_last_landing_qpos[:, side]
            previous_qvel = self.gait_cycle_last_landing_qvel[:, side]
            previous_activation = (
                self.gait_cycle_last_landing_activation[:, side]
            )
            pose_score = torch.exp(
                -torch.mean(
                    torch.square(
                        (current_qpos - previous_qpos)
                        / self.gait_cycle_pose_scale
                    ),
                    dim=1,
                )
            )
            velocity_score = torch.exp(
                -torch.mean(
                    torch.square(
                        (current_qvel - previous_qvel)
                        / self.gait_cycle_velocity_scale
                    ),
                    dim=1,
                )
            )
            activation_score = torch.exp(
                -torch.mean(
                    torch.square(
                        (muscle_activation - previous_activation)
                        / self.gait_cycle_activation_scale
                    ),
                    dim=1,
                )
            )
            terms["gait_cycle_pose_reward"] += (
                pose_score * same_valid.float()
            )
            terms["gait_cycle_velocity_reward"] += (
                velocity_score * same_valid.float()
            )
            terms["gait_cycle_activation_reward"] += (
                activation_score * same_valid.float()
            )

            mirrored_qpos = self.gait_cycle_last_landing_qpos[
                :, other
            ].index_select(1, self.gait_cycle_state_mirror_perm)
            mirrored_qvel = self.gait_cycle_last_landing_qvel[
                :, other
            ].index_select(1, self.gait_cycle_state_mirror_perm)
            mirrored_activation = self.gait_cycle_last_landing_activation[
                :, other
            ].index_select(1, self.gait_cycle_activation_mirror_perm)
            half_pose_score = torch.exp(
                -torch.mean(
                    torch.square(
                        (current_qpos - mirrored_qpos)
                        / self.gait_cycle_pose_scale
                    ),
                    dim=1,
                )
            )
            half_velocity_score = torch.exp(
                -torch.mean(
                    torch.square(
                        (current_qvel - mirrored_qvel)
                        / self.gait_cycle_velocity_scale
                    ),
                    dim=1,
                )
            )
            half_activation_score = torch.exp(
                -torch.mean(
                    torch.square(
                        (muscle_activation - mirrored_activation)
                        / self.gait_cycle_activation_scale
                    ),
                    dim=1,
                )
            )
            terms["gait_half_cycle_pose_reward"] += (
                half_pose_score * half_valid.float()
            )
            terms["gait_half_cycle_velocity_reward"] += (
                half_velocity_score * half_valid.float()
            )
            terms["gait_half_cycle_activation_reward"] += (
                half_activation_score * half_valid.float()
            )
            alternating = (
                (self.gait_cycle_last_event_side < 0)
                | (self.gait_cycle_last_event_side != side)
            )
            terms["gait_alternation_reward"] += (
                event & alternating
            ).float()
            valid_transition = (
                event & (self.gait_cycle_last_event_side >= 0)
            )
            terms["gait_alternating_event"] += (
                valid_transition & alternating
            ).float()
            terms["gait_repeated_side_event"] += (
                valid_transition & (~alternating)
            ).float()

            event_rows = rows[event]
            self.gait_cycle_last_landing_step[event_rows, side] = (
                self.episode_step[event_rows]
            )
            self.gait_cycle_last_landing_valid[event_rows, side] = True
            self.gait_cycle_last_landing_qpos[event_rows, side] = (
                current_qpos[event_rows]
            )
            self.gait_cycle_last_landing_qvel[event_rows, side] = (
                current_qvel[event_rows]
            )
            self.gait_cycle_last_landing_activation[event_rows, side] = (
                muscle_activation[event_rows]
            )
            half_event_rows = rows[half_valid]
            self.gait_cycle_last_half_interval[half_event_rows] = (
                half_interval[half_valid].float()
            )
            self.gait_cycle_last_half_interval_valid[half_event_rows] = True

        terms["gait_landing_event"] = accepted.float()
        accepted_rows = rows[accepted]
        self.gait_cycle_last_event_step[accepted_rows] = self.episode_step[
            accepted_rows
        ]
        self.gait_cycle_last_event_side[accepted_rows] = landing_side[
            accepted_rows
        ]
        missing = (
            active
            & (self.gait_cycle_last_event_step >= 0)
            & (
                (self.episode_step - self.gait_cycle_last_event_step)
                > self.gait_cycle_max_half_cycle_steps
            )
        )
        missing_excess = torch.relu(
            (
                self.episode_step
                - self.gait_cycle_last_event_step
                - self.gait_cycle_max_half_cycle_steps
            ).float()
        )
        terms["gait_missing_landing_penalty"] = (
            -self.dt
            * torch.clamp(
                missing_excess
                / float(self.gait_cycle_max_half_cycle_steps),
                max=1.0,
            )
            * missing.float()
        )
        if self.gait_cycle_event_hold_steps > 1:
            previous_hold = self.gait_cycle_event_hold_remaining > 0
            hold_scale = (
                1.0 / float(self.gait_cycle_event_hold_steps)
                if self.gait_cycle_event_hold_normalize
                else 1.0
            )
            for name, held_value in self.gait_cycle_held_terms.items():
                event_value = terms[name]
                held_value.copy_(
                    torch.where(accepted, event_value, held_value)
                )
                terms[name] = (
                    torch.where(
                        accepted,
                        event_value,
                        torch.where(
                            previous_hold,
                            held_value,
                            torch.zeros_like(event_value),
                        ),
                    )
                    * hold_scale
                )
            self.gait_cycle_event_hold_remaining.copy_(
                torch.where(
                    accepted,
                    torch.full_like(
                        self.gait_cycle_event_hold_remaining,
                        self.gait_cycle_event_hold_steps - 1,
                    ),
                    torch.clamp(
                        self.gait_cycle_event_hold_remaining - 1,
                        min=0,
                    ),
                )
            )
        return terms

    def myoassist_exact_reward(
        self,
        action: torch.Tensor,
        muscle_activation: torch.Tensor,
        prev_foot: torch.Tensor,
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

        if self.sensordata is not None and self.ground_force_sensor_indices.numel() >= 4:
            foot_force = self.sensordata.index_select(
                1, self.ground_force_sensor_indices
            )
            force_split = int(foot_force.shape[1]) // 2
            right_force = foot_force[:, :force_split].sum(dim=1)
            left_force = foot_force[:, force_split:].sum(dim=1)
            normalized_foot_force_sum = (torch.abs(right_force) + torch.abs(left_force)) / max(float(self.model_weight), 1e-6)
            foot_force_penalty = -dt * torch.relu(normalized_foot_force_sum - 1.2)
        else:
            foot_force = torch.zeros((self.nworld, 4), dtype=torch.float32, device=self.device)
            right_force = torch.zeros_like(forward_reward)
            left_force = torch.zeros_like(forward_reward)
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
        if self.scale_reference_velocity_to_target:
            speed_ratio = float(self.myoassist_target_velocity) / torch.clamp(
                ref_pelvis_vx,
                min=1e-6,
            )
        else:
            speed_ratio = torch.ones_like(ref_pelvis_vx)
        foot = self.site_xpos[:, self.foot_site_indices, :]
        foot_forward = site_forward_coord_tensor(foot, self.config)
        reference_tracking_error = self.reference_match_error(
            q,
            dq,
            foot_forward - self.qpos[:, self.pelvis_tx_qpos].unsqueeze(1),
            foot[:, :, 2],
            ref_q,
            ref_dq * speed_ratio.unsqueeze(1),
            self.reference_foot(target_phase),
        )
        qvel_reward_per_joint = dt * torch.exp(-8.0 * torch.square(dq - ref_dq * speed_ratio.unsqueeze(1)))
        qvel_imitation_rewards = torch.sum(qvel_reward_per_joint * self.myoassist_qvel_weights, dim=1)
        reference_qvel = ref_dq * speed_ratio.unsqueeze(1)
        joint_error_ratio = torch.abs(q - ref_q) / self.reference_joint_error_huber_delta
        joint_error_huber = torch.where(
            joint_error_ratio <= 1.0,
            0.5 * torch.square(joint_error_ratio),
            joint_error_ratio - 0.5,
        )
        joint_error_weight_sum = torch.clamp(
            self.reference_joint_error_weights.sum(),
            min=1.0e-6,
        )
        reference_joint_error_penalty = -dt * torch.sum(
            joint_error_huber * self.reference_joint_error_weights,
            dim=1,
        ) / joint_error_weight_sum
        joint_velocity_error_ratio = (
            torch.abs(dq - reference_qvel)
            / self.reference_joint_velocity_error_huber_delta
        )
        joint_velocity_error_huber = torch.where(
            joint_velocity_error_ratio <= 1.0,
            0.5 * torch.square(joint_velocity_error_ratio),
            joint_velocity_error_ratio - 0.5,
        )
        joint_velocity_error_weight_sum = torch.clamp(
            self.reference_joint_velocity_error_weights.sum(),
            min=1.0e-6,
        )
        reference_joint_velocity_error_penalty = -dt * torch.sum(
            joint_velocity_error_huber
            * self.reference_joint_velocity_error_weights,
            dim=1,
        ) / joint_velocity_error_weight_sum
        root_forward_velocity_imitation_reward = dt * torch.exp(
            -torch.square(
                (
                    self.qvel[:, self.pelvis_tx_qvel]
                    - ref_pelvis_vx * speed_ratio
                )
                / self.root_forward_velocity_imitation_scale
            )
        )
        root_forward_velocity_shortfall = torch.relu(
            ref_pelvis_vx * speed_ratio - self.qvel[:, self.pelvis_tx_qvel]
        )
        root_forward_velocity_shortfall_penalty = -dt * torch.clamp(
            torch.square(
                root_forward_velocity_shortfall
                / self.root_forward_velocity_imitation_scale
            ),
            max=4.0,
        )
        reference_forward_velocity = torch.clamp(
            ref_pelvis_vx * speed_ratio,
            min=0.0,
        )
        root_forward_velocity_overspeed = torch.relu(
            self.qvel[:, self.pelvis_tx_qvel]
            - reference_forward_velocity
            - self.root_forward_velocity_overspeed_margin
        )
        root_forward_velocity_overspeed_penalty = -dt * torch.clamp(
            torch.square(
                root_forward_velocity_overspeed
                / self.root_forward_velocity_overspeed_scale
            ),
            max=self.root_forward_velocity_overspeed_max_ratio_sq,
        )
        speed_cfg = self.config.get("reward_forward_shortfall", {})
        speed_target = float(speed_cfg.get("target_velocity", self.myoassist_target_velocity))
        speed_margin = float(speed_cfg.get("margin", 0.0) or 0.0)
        speed_scale = max(float(speed_cfg.get("scale", max(speed_target, 1e-3)) or max(speed_target, 1e-3)), 1e-6)
        speed_shortfall = torch.relu(speed_target - speed_margin - self.qvel[:, self.pelvis_tx_qvel])
        forward_velocity_reward = dt * torch.clamp(self.qvel[:, self.pelvis_tx_qvel], min=0.0, max=speed_target) / max(speed_target, 1e-6)
        forward_shortfall_penalty = -dt * torch.square(speed_shortfall / speed_scale)
        ramp_progress_reward = torch.zeros_like(forward_reward)
        ramp_progress_tracking_reward = torch.zeros_like(forward_reward)
        ramp_progress_lag_penalty = torch.zeros_like(forward_reward)
        ramp_velocity_reward = torch.zeros_like(forward_reward)
        ramp_step_reward = torch.zeros_like(forward_reward)
        ramp_step_event = torch.zeros_like(forward_reward)
        ramp_step_advance_m = torch.zeros_like(forward_reward)
        ramp_progress_fraction = torch.zeros_like(forward_reward)
        ramp_reference_progress_fraction = torch.zeros_like(forward_reward)
        ramp_progress_lag_m = torch.zeros_like(forward_reward)
        ramp_cfg = self.config.get("reward_ramp_progress", {})
        if bool(ramp_cfg.get("enabled", False)) and self.reference.get("full_reset_qpos") is not None:
            ramp_phase_start = int(ramp_cfg.get("phase_start", 0))
            ramp_phase_end = int(ramp_cfg.get("phase_end", ref_len))
            ramp_schedule_phase = reference_index(
                self.phase_idx,
                self.reference,
                self.config,
            )
            ramp_mask = (
                (ramp_schedule_phase >= ramp_phase_start)
                & (ramp_schedule_phase < ramp_phase_end)
            ).to(dtype=forward_reward.dtype)
            ramp_x_start = float(ramp_cfg.get("x_start", 0.0))
            ramp_x_end = float(ramp_cfg.get("x_end", ramp_x_start + 1.0))
            ramp_direction = 1.0 if ramp_x_end >= ramp_x_start else -1.0
            ramp_length = max(abs(ramp_x_end - ramp_x_start), 1.0e-6)
            agent_x = self.qpos[:, self.pelvis_tx_qpos]
            full_ref_q = self.reference["full_reset_qpos"][ramp_schedule_phase].to(
                device=self.device, dtype=self.qpos.dtype
            )
            ref_x = full_ref_q[:, self.pelvis_tx_qpos]
            ramp_progress_fraction = torch.clamp(
                ramp_direction * (agent_x - ramp_x_start) / ramp_length,
                min=0.0,
                max=1.0,
            )
            ramp_reference_progress_fraction = torch.clamp(
                ramp_direction * (ref_x - ramp_x_start) / ramp_length,
                min=0.0,
                max=1.0,
            )
            progress_error = (
                ramp_progress_fraction - ramp_reference_progress_fraction
            )
            tracking_scale = max(
                float(ramp_cfg.get("tracking_scale_fraction", 0.15)), 1.0e-6
            )
            lag_margin = float(ramp_cfg.get("lag_margin_m", 0.15))
            lag_scale = max(float(ramp_cfg.get("lag_scale_m", 0.75)), 1.0e-6)
            lag_limit = max(float(ramp_cfg.get("lag_penalty_limit", 9.0)), 0.0)
            ramp_progress_lag_m = torch.relu(
                ramp_direction * (ref_x - agent_x) - lag_margin
            )
            ramp_progress_reward = dt * ramp_progress_fraction * ramp_mask
            ramp_progress_tracking_reward = (
                dt
                * torch.exp(-torch.square(progress_error / tracking_scale))
                * ramp_mask
            )
            ramp_progress_lag_penalty = (
                -dt
                * torch.clamp(
                    torch.square(ramp_progress_lag_m / lag_scale),
                    max=lag_limit,
                )
                * ramp_mask
            )
            ramp_speed_target = max(
                float(ramp_cfg.get("target_velocity", speed_target)), 1.0e-6
            )
            ramp_velocity = ramp_direction * self.qvel[:, self.pelvis_tx_qvel]
            ramp_velocity_reward = (
                dt
                * torch.clamp(ramp_velocity, min=0.0, max=ramp_speed_target)
                / ramp_speed_target
                * ramp_mask
            )
        ramp_step_cfg = self.config.get("reward_ramp_step", {})
        if bool(ramp_step_cfg.get("enabled", False)):
            foot = self.site_xpos[:, self.foot_site_indices, :]
            foot_forward = site_forward_coord_tensor(foot, self.config)
            previous_foot_forward = site_forward_coord_tensor(prev_foot, self.config)
            foot_height = terrain_height_for_world_x_tensor(
                foot_forward, self.phase_idx, self.reference, self.config
            )
            previous_foot_height = terrain_height_for_world_x_tensor(
                previous_foot_forward, self.phase_idx, self.reference, self.config
            )
            contact_threshold = float(
                ramp_step_cfg.get(
                    "contact_z_threshold",
                    self.config.get("reference_contact", {}).get(
                        "z_threshold", 0.025
                    ),
                )
            )
            site_contact = (foot[:, :, 2] - foot_height) < contact_threshold
            previous_site_contact = (
                prev_foot[:, :, 2] - previous_foot_height
            ) < contact_threshold
            side_split = int(site_contact.shape[1]) // 2
            current_contact = torch.stack(
                (
                    site_contact[:, :side_split].any(dim=1),
                    site_contact[:, side_split:].any(dim=1),
                ),
                dim=1,
            )
            previous_contact = torch.stack(
                (
                    previous_site_contact[:, :side_split].any(dim=1),
                    previous_site_contact[:, side_split:].any(dim=1),
                ),
                dim=1,
            )
            side_forward = torch.stack(
                (
                    foot_forward[:, :side_split].mean(dim=1),
                    foot_forward[:, side_split:].mean(dim=1),
                ),
                dim=1,
            )
            ramp_x_start = float(ramp_step_cfg.get("x_start", 0.0))
            ramp_x_end = float(ramp_step_cfg.get("x_end", ramp_x_start + 1.0))
            ramp_direction = 1.0 if ramp_x_end >= ramp_x_start else -1.0
            x_low = min(ramp_x_start, ramp_x_end)
            x_high = max(ramp_x_start, ramp_x_end)
            schedule_phase = reference_index(
                self.phase_idx, self.reference, self.config
            )
            phase_start = int(ramp_step_cfg.get("phase_start", 0))
            phase_end = int(ramp_step_cfg.get("phase_end", ref_len))
            phase_active = (
                (schedule_phase >= phase_start) & (schedule_phase < phase_end)
            ).unsqueeze(1)
            active = (
                (side_forward >= x_low)
                & (side_forward <= x_high)
                & phase_active
            )
            (
                ramp_step_reward,
                ramp_step_event,
                ramp_step_advance_m,
            ) = ramp_alternating_step_terms(
                current_contact=current_contact,
                previous_contact=previous_contact,
                foot_forward=side_forward,
                previous_landing_x=self.ramp_previous_landing_x,
                previous_landing_side=self.ramp_previous_landing_side,
                episode_start=self.episode_step <= 1,
                active=active,
                direction=ramp_direction,
                min_advance_m=float(ramp_step_cfg.get("min_advance_m", 0.12)),
                target_advance_m=float(
                    ramp_step_cfg.get("target_advance_m", 0.45)
                ),
            )
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
        keypoint_position_imitation_reward = torch.zeros_like(forward_reward)
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
        if self.reference.get("full_reset_qpos") is not None:
            full_ref_q = self.reference["full_reset_qpos"][target_phase].to(
                device=self.device,
                dtype=self.qpos.dtype,
            )
            if self.root_qpos_adr >= 0:
                root_xy_delta = self.qpos[:, self.root_qpos_adr : self.root_qpos_adr + 2] - full_ref_q[
                    :, self.root_qpos_adr : self.root_qpos_adr + 2
                ]
                cur_root_xyz = self.qpos[
                    :, self.root_qpos_adr : self.root_qpos_adr + 3
                ]
                ref_root_xyz = full_ref_q[
                    :, self.root_qpos_adr : self.root_qpos_adr + 3
                ]
            else:
                # The planar 22-muscle model uses scalar slide joints instead of
                # a freejoint. pelvis_ty maps to world height after the root body's
                # fixed rotation.
                root_xy_delta = torch.stack(
                    (
                        self.qpos[:, self.pelvis_tx_qpos]
                        - full_ref_q[:, self.pelvis_tx_qpos],
                        self.qpos[:, self.pelvis_ty_qpos]
                        - full_ref_q[:, self.pelvis_ty_qpos],
                    ),
                    dim=1,
                )
                cur_root_xyz = torch.zeros(
                    (self.nworld, 3),
                    dtype=self.site_xpos.dtype,
                    device=self.device,
                )
                ref_root_xyz = torch.zeros_like(cur_root_xyz)
                cur_root_xyz[:, 0] = self.qpos[:, self.pelvis_tx_qpos]
                cur_root_xyz[:, 2] = self.qpos[:, self.pelvis_ty_qpos]
                ref_root_xyz[:, 0] = full_ref_q[:, self.pelvis_tx_qpos]
                ref_root_xyz[:, 2] = full_ref_q[:, self.pelvis_ty_qpos]

            root_xy_position_reward = dt * torch.exp(
                -torch.sum(
                    torch.square(root_xy_delta / self.root_xy_position_scale),
                    dim=1,
                )
            )
            if self.root_qpos_adr >= 0:
                root_quat = torch.nn.functional.normalize(self.qpos[:, self.root_qpos_adr + 3 : self.root_qpos_adr + 7], dim=1)
                ref_quat = torch.nn.functional.normalize(full_ref_q[:, self.root_qpos_adr + 3 : self.root_qpos_adr + 7], dim=1)
                quat_dot = torch.abs(torch.sum(root_quat * ref_quat, dim=1)).clamp(max=1.0)
                quat_angle = 2.0 * torch.acos(quat_dot)
                root_orientation_reward = dt * torch.exp(-torch.square(quat_angle / self.root_orientation_scale))
                lateral_qpos_offset = 1 if self.forward_axis == "x" else 0
                lateral_drift = self.qpos[:, self.root_qpos_adr + lateral_qpos_offset] - full_ref_q[:, self.root_qpos_adr + lateral_qpos_offset]
                lateral_drift_penalty = -dt * torch.clamp(torch.square(lateral_drift / self.lateral_drift_scale), max=4.0)

            ref_foot = self.reference["foot_site_ref"][target_phase].to(
                device=self.device,
                dtype=self.site_xpos.dtype,
            )
            cur_foot_local = (
                self.site_xpos[:, self.foot_site_indices, :]
                - cur_root_xyz[:, None, :].to(dtype=self.site_xpos.dtype)
            )
            ref_foot_local = ref_foot - ref_root_xyz[:, None, :].to(
                dtype=self.site_xpos.dtype
            )
            foot_local_sq = torch.sum(
                torch.square(
                    (cur_foot_local - ref_foot_local)
                    / self.foot_site_local_mimic_scale
                ),
                dim=2,
            )
            foot_site_local_mimic_reward = dt * torch.mean(
                torch.exp(-foot_local_sq),
                dim=1,
            )
            if self.root_qpos_adr >= 0:
                foot_lateral_dim = 0 if self.forward_axis == "y" else 1
                foot_lateral_scale = max(
                    float(
                        self.config.get("footstep_target", {}).get(
                            "lateral_scale",
                            0.10,
                        )
                    ),
                    1e-6,
                )
                foot_lateral_error = (
                    self.site_xpos[
                        :, self.foot_site_indices, foot_lateral_dim
                    ]
                    - ref_foot[:, :, foot_lateral_dim]
                )
                foot_lateral_target_penalty = -dt * torch.mean(
                    torch.clamp(
                        torch.square(
                            foot_lateral_error / foot_lateral_scale
                        ),
                        max=4.0,
                    ),
                    dim=1,
                )
            future_steps = max(
                0,
                int(
                    self.config.get("imitation", {}).get(
                        "reference_reward_future_steps",
                        0,
                    )
                    or 0
                ),
            )
            if future_steps > 0:
                future_sum = torch.zeros_like(forward_reward)
                for offset in range(1, future_steps + 1):
                    future_phase = reference_index(
                        target_phase + offset,
                        self.reference,
                        self.config,
                    )
                    future_ref_q = self.reference["full_reset_qpos"][
                        future_phase
                    ].to(device=self.device, dtype=self.qpos.dtype)
                    future_ref_foot = self.reference["foot_site_ref"][
                        future_phase
                    ].to(device=self.device, dtype=self.site_xpos.dtype)
                    if self.root_qpos_adr >= 0:
                        future_ref_root_xyz = future_ref_q[
                            :,
                            self.root_qpos_adr : self.root_qpos_adr + 3,
                        ].to(dtype=self.site_xpos.dtype)
                    else:
                        future_ref_root_xyz = torch.zeros_like(ref_root_xyz)
                        future_ref_root_xyz[:, 0] = future_ref_q[
                            :, self.pelvis_tx_qpos
                        ]
                        future_ref_root_xyz[:, 2] = future_ref_q[
                            :, self.pelvis_ty_qpos
                        ]
                    future_ref_foot_local = (
                        future_ref_foot
                        - future_ref_root_xyz[:, None, :]
                    )
                    future_foot_local_sq = torch.sum(
                        torch.square(
                            (cur_foot_local - future_ref_foot_local)
                            / self.foot_site_local_mimic_scale
                        ),
                        dim=2,
                    )
                    future_sum = future_sum + torch.mean(
                        torch.exp(-future_foot_local_sq),
                        dim=1,
                    )
                future_foot_site_local_mimic_reward = (
                    dt * future_sum / float(future_steps)
                )

        if self.root_qpos_adr >= 0 and self.root_dof_adr >= 0:
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
        elif self.reference.get("full_reset_qpos") is not None:
            # Planar models use scalar pelvis slide joints instead of a freejoint.
            ref_foot = self.reference["foot_site_ref"][target_phase].to(
                device=self.device, dtype=self.site_xpos.dtype
            )
            forward_dim = 0 if self.forward_axis == "x" else 1
            cur_foot_local = self.site_xpos[
                :, self.foot_site_indices, :
            ].clone()
            cur_foot_local[:, :, forward_dim] -= self.qpos[
                :, self.pelvis_tx_qpos
            ].unsqueeze(1).to(dtype=self.site_xpos.dtype)
            ref_foot_local = ref_foot
            foot_local_sq = torch.sum(
                torch.square(
                    (cur_foot_local - ref_foot_local)
                    / self.foot_site_local_mimic_scale
                ),
                dim=2,
            )
            foot_site_local_mimic_reward = dt * torch.mean(
                torch.exp(-foot_local_sq), dim=1
            )
            future_steps = max(
                0,
                int(
                    self.config.get("imitation", {}).get(
                        "reference_reward_future_steps", 0
                    )
                    or 0
                ),
            )
            if future_steps > 0:
                future_sum = torch.zeros_like(forward_reward)
                for offset in range(1, future_steps + 1):
                    future_phase = reference_index(
                        target_phase + offset, self.reference, self.config
                    )
                    future_ref_foot = self.reference["foot_site_ref"][
                        future_phase
                    ].to(device=self.device, dtype=self.site_xpos.dtype)
                    future_ref_foot_local = future_ref_foot
                    future_foot_local_sq = torch.sum(
                        torch.square(
                            (cur_foot_local - future_ref_foot_local)
                            / self.foot_site_local_mimic_scale
                        ),
                        dim=2,
                    )
                    future_sum = future_sum + torch.mean(
                        torch.exp(-future_foot_local_sq), dim=1
                    )
                future_foot_site_local_mimic_reward = (
                    dt * future_sum / float(future_steps)
                )
        if (
            self.keypoint_body_indices.numel() > 0
            and self.reference.get("keypoint_body_ref") is not None
        ):
            current_keypoints = self.body_xpos[
                :, self.keypoint_body_indices, :
            ].clone()
            forward_dim = 1 if self.forward_axis == "y" else 0
            current_keypoints[:, :, forward_dim] -= self.qpos[
                :, self.pelvis_tx_qpos
            ].unsqueeze(1).to(dtype=current_keypoints.dtype)
            reference_keypoints = self.reference["keypoint_body_ref"][
                target_phase
            ].to(device=self.device, dtype=current_keypoints.dtype)
            keypoint_sq_error = torch.sum(
                torch.square(
                    (current_keypoints - reference_keypoints)
                    / self.keypoint_imitation_scale
                ),
                dim=2,
            )
            keypoint_position_imitation_reward = dt * torch.mean(
                torch.exp(-keypoint_sq_error),
                dim=1,
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
        rollover_terms = {
            name: torch.zeros_like(forward_reward)
            for name in (
                "foot_rollover_toe_first_reward",
                "foot_rollover_heel_follow_reward",
                "foot_rollover_heel_loading_penalty",
                "foot_rollover_heel_first_penalty",
                "foot_rollover_missing_heel_penalty",
                "foot_rollover_toe_first_event",
                "foot_rollover_heel_follow_event",
                "foot_rollover_heel_first_event",
                "foot_rollover_missing_heel_event",
                "foot_rollover_unsupported_toe_event",
            )
        }
        if self.foot_rollover_enabled and self.foot_sensor_indices.numel() == 4:
            contact_force = torch.abs(
                self.sensordata.index_select(1, self.foot_sensor_indices)
            )
            rollover_contact = (
                contact_force >= self.foot_rollover_contact_force_threshold_n
            )
            toe_forward = foot_forward[:, (1, 3)]
            rollover_active = (
                (toe_forward >= self.foot_rollover_x_start)
                & (toe_forward <= self.foot_rollover_x_end)
            )
            rollover_terms = foot_rollover_sequence_terms(
                current_contact=rollover_contact,
                previous_contact=self.foot_rollover_previous_contact,
                current_force=contact_force,
                previous_force=self.foot_rollover_previous_force,
                state=self.foot_rollover_state,
                elapsed_steps=self.foot_rollover_elapsed_steps,
                airborne_steps=self.foot_rollover_airborne_steps,
                heel_stable_steps=self.foot_rollover_heel_stable_steps,
                heel_loading_excess=self.foot_rollover_heel_loading_excess,
                active=rollover_active,
                episode_start=self.episode_step <= 1,
                min_heel_delay_steps=self.foot_rollover_min_heel_delay_steps,
                max_heel_delay_steps=self.foot_rollover_max_heel_delay_steps,
                release_steps=self.foot_rollover_release_steps,
                required_heel_stable_steps=self.foot_rollover_required_heel_stable_steps,
                max_heel_force_delta_n=self.foot_rollover_max_heel_force_delta_n,
            )
            self.foot_rollover_previous_contact.copy_(rollover_contact)
            self.foot_rollover_previous_force.copy_(contact_force)
        gait_cycle_terms = self.elastic_gait_cycle_terms(
            muscle_activation,
            prev_foot,
        )
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
            progress_mode = str(
                stair_cfg.get("contact_progress_reward_mode", "absolute")
            ).lower()
            if progress_mode == "new_step":
                episode_start = self.episode_step <= 1
                previous_max = torch.where(
                    episode_start,
                    stair_contact_step_index,
                    self.stair_contact_progress_max,
                )
                stair_contact_step_progress_reward = torch.clamp(
                    stair_contact_step_index - previous_max,
                    min=0.0,
                    max=1.0,
                )
                self.stair_contact_progress_max.copy_(
                    torch.maximum(previous_max, stair_contact_step_index)
                )
            elif progress_mode == "absolute":
                stair_contact_step_progress_reward = dt * torch.clamp(
                    stair_contact_step_index / step_count,
                    min=0.0,
                    max=1.0,
                )
            else:
                raise ValueError(
                    "reward_stair_progress.contact_progress_reward_mode must be "
                    f"'absolute' or 'new_step', got {progress_mode!r}"
                )
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
            "ramp_progress_reward": ramp_progress_reward,
            "ramp_progress_tracking_reward": ramp_progress_tracking_reward,
            "ramp_progress_lag_penalty": ramp_progress_lag_penalty,
            "ramp_velocity_reward": ramp_velocity_reward,
            "ramp_step_reward": ramp_step_reward,
            "ramp_step_event": ramp_step_event,
            "ramp_step_advance_m": ramp_step_advance_m,
            "ramp_progress_fraction": ramp_progress_fraction,
            "ramp_reference_progress_fraction": ramp_reference_progress_fraction,
            "ramp_progress_lag_m": ramp_progress_lag_m,
            "xalign_phase_lag_penalty": xalign_phase_lag_penalty,
            "xalign_phase_lag_steps": phase_lag_steps,
            "muscle_activation_penalty": muscle_activation_penalty,
            "muscle_activation_diff_penalty": muscle_activation_diff_penalty,
            "foot_force_penalty": foot_force_penalty,
            "joint_constraint_force_penalty": joint_constraint_force_penalty,
            "qpos_imitation_rewards": qpos_imitation_rewards,
            "qvel_imitation_rewards": qvel_imitation_rewards,
            "reference_joint_error_penalty": reference_joint_error_penalty,
            "reference_joint_velocity_error_penalty": (
                reference_joint_velocity_error_penalty
            ),
            "root_forward_velocity_imitation_reward": root_forward_velocity_imitation_reward,
            "root_forward_velocity_shortfall_penalty": root_forward_velocity_shortfall_penalty,
            "root_forward_velocity_overspeed_penalty": root_forward_velocity_overspeed_penalty,
            "root_forward_velocity_overspeed_mps": root_forward_velocity_overspeed,
            "reference_forward_velocity_target_mps": reference_forward_velocity,
            "full_qpos_imitation_rewards": full_qpos_imitation_rewards,
            "full_qvel_imitation_rewards": full_qvel_imitation_rewards,
            "root_xy_position_reward": root_xy_position_reward,
            "root_orientation_reward": root_orientation_reward,
            "root_angvel_penalty": root_angvel_penalty,
            "lateral_vel_penalty": lateral_vel_penalty,
            "lateral_drift_penalty": lateral_drift_penalty,
            "foot_site_local_mimic_reward": foot_site_local_mimic_reward,
            "future_foot_site_local_mimic_reward": future_foot_site_local_mimic_reward,
            "keypoint_position_imitation_reward": keypoint_position_imitation_reward,
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
            "reference_tracking_error": reference_tracking_error,
            "nearest_trajectory_agent_amp": nearest_trajectory_agent_amp,
            "nearest_trajectory_ref_amp": nearest_trajectory_ref_amp,
            "myoassist_foot_force_r": right_force,
            "myoassist_foot_force_l": left_force,
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
        terms.update(rollover_terms)
        terms.update(gait_cycle_terms)
        reward = torch.zeros_like(forward_reward)
        for key, weight in self.myoassist_dense_weights.items():
            reward = reward + float(weight) * terms[key]
        if self.route_reward_profiles_enabled:
            if self.route_reward_env_boundaries is None:
                boundaries = torch.tensor(
                    self.route_reward_boundaries,
                    dtype=self.qpos.dtype,
                    device=self.device,
                ).unsqueeze(0)
            else:
                boundaries = self.route_reward_env_boundaries
            route_index = torch.sum(
                self.qpos[:, self.pelvis_tx_qpos].unsqueeze(1) >= boundaries,
                dim=1,
            ).long()
            route_reward = torch.zeros_like(reward)
            for profile_index, profile in enumerate(self.route_reward_profiles):
                profile_reward = torch.zeros_like(reward)
                for key, value in terms.items():
                    weight = profile["weights"].get(
                        key, self.myoassist_dense_weights.get(key, 0.0)
                    )
                    profile_reward = profile_reward + float(weight) * value
                route_reward = torch.where(
                    route_index == profile_index,
                    profile_reward,
                    route_reward,
                )
            reward = route_reward
            terms["route_reward_profile_index"] = route_index.float()
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
