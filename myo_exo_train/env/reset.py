"""Episode reset and complete-state recovery-bank behavior."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from myo_exo_train.env.model import RESET_JOINTS, ROOT, apply_non_muscle_ctrl_override
from myo_exo_train.env.observation import reference_index, reset_reference_phase_from_x

class ResetMixin:
    @staticmethod
    def parse_phase_windows(phase_windows: Any, reference_length: int) -> list[tuple[int, int]]:
        windows: list[tuple[int, int]] = []
        if not isinstance(phase_windows, list):
            return windows
        for item in phase_windows:
            if isinstance(item, dict):
                start = int(item.get("start", item.get("phase_start", 0)))
                end = int(item.get("end", item.get("phase_end", reference_length)))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                start = int(item[0])
                end = int(item[1])
            else:
                continue
            start = max(0, min(start, reference_length - 1))
            end = max(start + 1, min(end, reference_length))
            windows.append((start, end))
        return windows

    def build_phase_choices(
        self,
        phase_windows: list[Any],
        phase_indices: list[Any],
        phase_index_jitter: int,
        reference_length: int,
    ) -> torch.Tensor | None:
        choices: list[int] = []
        jitter = max(0, int(phase_index_jitter))
        for item in phase_indices or []:
            phase = int(item)
            for offset in range(-jitter, jitter + 1):
                jittered_phase = phase + offset
                if 0 <= jittered_phase < reference_length:
                    choices.append(jittered_phase)
        for item in phase_windows or []:
            if isinstance(item, dict):
                start = int(item.get("start", 0))
                end = int(item.get("end", start))
            else:
                start = int(item[0])
                end = int(item[1])
            start = max(0, min(start, reference_length))
            end = max(start, min(end, reference_length))
            choices.extend(range(start, end))
        if not choices:
            return None
        return torch.tensor(sorted(set(choices)), dtype=torch.long, device=self.device)

    def recovery_bank_ready(self) -> bool:
        return (
            self.recovery_reset_enabled
            and self.recovery_bank_capacity > 0
            and self.recovery_bank_size >= self.recovery_min_bank_size
        )

    def load_offline_recovery_bank(self) -> None:
        if not self.offline_recovery_enabled or not self.offline_recovery_path:
            return
        path = Path(self.offline_recovery_path).expanduser()
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"offline_recovery_reset.path does not exist: {path}")
        payload = np.load(path, allow_pickle=True)
        qpos = torch.tensor(payload["qpos"], dtype=torch.float32, device=self.device)
        qvel = torch.tensor(payload["qvel"], dtype=torch.float32, device=self.device)
        act = torch.tensor(payload["act"], dtype=torch.float32, device=self.device)
        phase = torch.tensor(payload["phase"], dtype=torch.long, device=self.device)
        if qpos.ndim != 2 or qpos.shape[1] != int(self.model.nq):
            raise ValueError(f"offline recovery qpos shape {tuple(qpos.shape)} does not match nq={self.model.nq}")
        if qvel.ndim != 2 or qvel.shape[1] != int(self.model.nv):
            raise ValueError(f"offline recovery qvel shape {tuple(qvel.shape)} does not match nv={self.model.nv}")
        if act.ndim != 2 or act.shape[1] != int(self.model.na):
            raise ValueError(f"offline recovery act shape {tuple(act.shape)} does not match na={self.model.na}")
        if phase.ndim != 1 or phase.shape[0] != qpos.shape[0]:
            raise ValueError("offline recovery phase must be a vector with one entry per qpos row")
        self.offline_recovery_bank_qpos = qpos
        self.offline_recovery_bank_qvel = qvel
        self.offline_recovery_bank_act = act
        self.offline_recovery_bank_phase = phase
        if "ctrl" in payload:
            ctrl = torch.tensor(payload["ctrl"], dtype=torch.float32, device=self.device)
            if ctrl.ndim != 2 or ctrl.shape[1] != int(self.model.nu) or ctrl.shape[0] != qpos.shape[0]:
                raise ValueError(f"offline recovery ctrl shape {tuple(ctrl.shape)} does not match ({qpos.shape[0]}, {self.model.nu})")
            self.offline_recovery_bank_ctrl = apply_non_muscle_ctrl_override(
                ctrl,
                self.config,
                muscle_count=int(self.model.na),
                ctrl_low=self.ctrl_low,
                ctrl_high=self.ctrl_high,
            )
        else:
            ctrl = torch.zeros((qpos.shape[0], int(self.model.nu)), dtype=torch.float32, device=self.device)
            ctrl[:, : int(self.model.na)] = act
            if int(self.model.nu) > int(self.model.na):
                ctrl[:, int(self.model.na) :] = 0.5 * (
                    self.ctrl_low[int(self.model.na) :] + self.ctrl_high[int(self.model.na) :]
                ).unsqueeze(0)
            self.offline_recovery_bank_ctrl = apply_non_muscle_ctrl_override(
                ctrl,
                self.config,
                muscle_count=int(self.model.na),
                ctrl_low=self.ctrl_low,
                ctrl_high=self.ctrl_high,
            )
        if "prev_activation" in payload:
            prev_activation = torch.tensor(payload["prev_activation"], dtype=torch.float32, device=self.device)
            if prev_activation.ndim != 2 or prev_activation.shape[1] != int(self.model.na) or prev_activation.shape[0] != qpos.shape[0]:
                raise ValueError(
                    f"offline recovery prev_activation shape {tuple(prev_activation.shape)} does not match ({qpos.shape[0]}, {self.model.na})"
                )
            self.offline_recovery_bank_prev_activation = prev_activation
        else:
            self.offline_recovery_bank_prev_activation = act
        if "qacc_warmstart" in payload:
            qacc_warmstart = torch.tensor(payload["qacc_warmstart"], dtype=torch.float32, device=self.device)
            if qacc_warmstart.ndim != 2 or qacc_warmstart.shape[1] != int(self.model.nv) or qacc_warmstart.shape[0] != qpos.shape[0]:
                raise ValueError(
                    f"offline recovery qacc_warmstart shape {tuple(qacc_warmstart.shape)} does not match ({qpos.shape[0]}, {self.model.nv})"
                )
            self.offline_recovery_bank_qacc_warmstart = qacc_warmstart
        else:
            self.offline_recovery_bank_qacc_warmstart = None
        if "site_xpos" in payload:
            site_xpos = torch.tensor(payload["site_xpos"], dtype=torch.float32, device=self.device)
            if tuple(site_xpos.shape[1:]) != tuple(self.site_xpos.shape[1:]) or site_xpos.shape[0] != qpos.shape[0]:
                raise ValueError(
                    f"offline recovery site_xpos shape {tuple(site_xpos.shape)} does not match expected (*, {tuple(self.site_xpos.shape[1:])})"
                )
            self.offline_recovery_bank_site_xpos = site_xpos
        else:
            self.offline_recovery_bank_site_xpos = None
        self.offline_recovery_bank_size = int(qpos.shape[0])

    def offline_recovery_bank_ready(self) -> bool:
        return (
            self.offline_recovery_enabled
            and self.offline_recovery_bank_size >= self.offline_recovery_min_bank_size
            and hasattr(self, "offline_recovery_bank_qpos")
        )

    def restore_offline_recovery_states(self, rows: torch.Tensor) -> None:
        count = int(rows.numel())
        if count <= 0:
            return
        self.offline_recovery_last_restore_count += count
        bank_size = int(self.offline_recovery_bank_size)
        sample_idx = torch.randint(0, bank_size, (count,), generator=self.rng, device=self.device)
        self.qpos[rows] = self.offline_recovery_bank_qpos[sample_idx]
        self.qvel[rows] = self.offline_recovery_bank_qvel[sample_idx]
        self.act[rows] = self.offline_recovery_bank_act[sample_idx]
        self.ctrl[rows] = self.offline_recovery_bank_ctrl[sample_idx]
        if self.offline_recovery_bank_site_xpos is not None:
            self.site_xpos[rows] = self.offline_recovery_bank_site_xpos[sample_idx]
        self.phase_idx[rows] = self.offline_recovery_bank_phase[sample_idx]
        self.phase_idx[rows] = reset_reference_phase_from_x(
            self.qpos[rows],
            self.phase_idx[rows],
            self.reference,
            self.config,
        )
        self.prev_activation[rows] = self.offline_recovery_bank_prev_activation[sample_idx]
        self.prev_activation_valid[rows] = True
        if self.offline_recovery_bank_qacc_warmstart is not None:
            self.qacc_warmstart[rows] = self.offline_recovery_bank_qacc_warmstart[sample_idx]
        else:
            self.qacc_warmstart[rows] = 0.0
        self.time[rows] = 0.0
        self.episode_step[rows] = 0
        self.episode_return[rows] = 0.0
        self.episode_length[rows] = 0.0
        self.recovery_mode_steps[rows] = int(self.recovery_reward_horizon_steps)
        self.apply_joint_equalities(rows)

    def restore_recovery_states(self, rows: torch.Tensor) -> None:
        count = int(rows.numel())
        if count <= 0:
            return
        self.recovery_last_restore_count += count
        bank_size = int(self.recovery_bank_size)
        sample_idx = torch.randint(0, bank_size, (count,), generator=self.rng, device=self.device)
        self.qpos[rows] = self.recovery_bank_qpos[sample_idx]
        self.qvel[rows] = self.recovery_bank_qvel[sample_idx]
        self.act[rows] = self.recovery_bank_act[sample_idx]
        self.ctrl[rows] = self.recovery_bank_ctrl[sample_idx]
        self.ctrl[rows] = apply_non_muscle_ctrl_override(
            self.ctrl[rows],
            self.config,
            muscle_count=int(self.model.na),
            ctrl_low=self.ctrl_low,
            ctrl_high=self.ctrl_high,
        )
        self.site_xpos[rows] = self.recovery_bank_site_xpos[sample_idx]
        self.phase_idx[rows] = self.recovery_bank_phase[sample_idx]
        self.phase_idx[rows] = reset_reference_phase_from_x(
            self.qpos[rows],
            self.phase_idx[rows],
            self.reference,
            self.config,
        )
        self.prev_activation[rows] = self.recovery_bank_prev_activation[sample_idx]
        self.prev_activation_valid[rows] = True
        self.qacc_warmstart[rows] = self.recovery_bank_qacc_warmstart[sample_idx]
        if int(self.model.nu) >= int(self.model.na) + 2:
            restored_exo_ctrl = self.ctrl[rows, int(self.model.na) : int(self.model.na) + 2]
            self.applied_exo_ctrl[rows] = restored_exo_ctrl
        self.time[rows] = 0.0
        self.episode_step[rows] = 0
        self.episode_return[rows] = 0.0
        self.episode_length[rows] = 0.0
        self.recovery_mode_steps[rows] = int(self.recovery_reward_horizon_steps)
        self.apply_joint_equalities(rows)

    def collect_recovery_states(self, done: torch.Tensor, pelvis_height_above_terrain: torch.Tensor) -> None:
        self.recovery_last_collect_count = 0
        self.recovery_last_stage_count = 0
        self.recovery_last_commit_count = 0
        if not self.recovery_reset_enabled or self.recovery_bank_capacity <= 0:
            return
        if self.recovery_survival_delay_steps > 0 and hasattr(self, "recovery_pending_valid"):
            self.commit_survived_recovery_pending(done)
            if bool(done.any().item()):
                self.recovery_pending_valid[:, done] = False
        phase = reference_index(self.phase_idx, self.reference, self.config)
        if self.recovery_phase_windows:
            phase_ok = torch.zeros_like(done, dtype=torch.bool)
            for start, end in self.recovery_phase_windows:
                phase_ok = phase_ok | ((phase >= start) & (phase < end))
        else:
            phase_ok = (phase >= self.recovery_phase_start) & (phase < self.recovery_phase_end)
        candidate = (
            (~done)
            & (pelvis_height_above_terrain >= self.recovery_min_height)
            & (pelvis_height_above_terrain <= self.recovery_max_height)
            & (self.episode_step >= self.recovery_min_episode_steps)
            & phase_ok
        )
        if self.recovery_max_abs_lateral_drift > 0.0 and self.root_qpos_adr >= 0:
            full_ref_qpos = self.reference.get("full_reset_qpos")
            if full_ref_qpos is not None:
                lateral_offset = 1 if self.forward_axis == "x" else 0
                lateral_col = int(self.root_qpos_adr + lateral_offset)
                ref_q = full_ref_qpos[self.target_phase_idx()].to(device=self.device, dtype=self.qpos.dtype)
                lateral_drift = torch.abs(self.qpos[:, lateral_col] - ref_q[:, lateral_col])
                candidate = candidate & (lateral_drift <= self.recovery_max_abs_lateral_drift)
        if self.recovery_collect_probability < 1.0:
            keep = torch.rand((self.nworld,), generator=self.rng, device=self.device) < max(0.0, self.recovery_collect_probability)
            candidate = candidate & keep
        if not bool(candidate.any().item()):
            if self.recovery_survival_delay_steps > 0 and hasattr(self, "recovery_pending_valid"):
                self.recovery_pending_write = (int(self.recovery_pending_write) + 1) % int(self.recovery_survival_delay_steps)
            return
        rows = torch.nonzero(candidate, as_tuple=False).flatten()
        if self.recovery_survival_delay_steps > 0 and hasattr(self, "recovery_pending_valid"):
            self.stage_recovery_pending(rows)
        else:
            self.write_recovery_bank_rows(rows)

    def write_recovery_bank_rows(self, rows: torch.Tensor) -> None:
        count = int(rows.numel())
        if count <= 0:
            return
        start = int(self.recovery_bank_write)
        slots = (torch.arange(count, device=self.device, dtype=torch.long) + start) % int(self.recovery_bank_capacity)
        self.recovery_bank_qpos[slots] = self.qpos[rows].detach()
        self.recovery_bank_qvel[slots] = self.qvel[rows].detach()
        self.recovery_bank_act[slots] = self.act[rows].detach()
        self.recovery_bank_ctrl[slots] = self.ctrl[rows].detach()
        self.recovery_bank_qacc_warmstart[slots] = self.qacc_warmstart[rows].detach()
        self.recovery_bank_site_xpos[slots] = self.site_xpos[rows].detach()
        self.recovery_bank_phase[slots] = self.phase_idx[rows].detach()
        self.recovery_bank_prev_activation[slots] = self.prev_activation[rows].detach()
        self.recovery_bank_write = int((start + count) % int(self.recovery_bank_capacity))
        self.recovery_bank_size = min(int(self.recovery_bank_capacity), int(self.recovery_bank_size) + count)
        self.recovery_last_collect_count = count
        self.recovery_last_commit_count = count

    def stage_recovery_pending(self, rows: torch.Tensor) -> None:
        count = int(rows.numel())
        if count <= 0:
            return
        slot = int(self.recovery_pending_write)
        self.recovery_pending_valid[slot] = False
        self.recovery_pending_valid[slot, rows] = True
        self.recovery_pending_qpos[slot, rows] = self.qpos[rows].detach()
        self.recovery_pending_qvel[slot, rows] = self.qvel[rows].detach()
        self.recovery_pending_act[slot, rows] = self.act[rows].detach()
        self.recovery_pending_ctrl[slot, rows] = self.ctrl[rows].detach()
        self.recovery_pending_qacc_warmstart[slot, rows] = self.qacc_warmstart[rows].detach()
        self.recovery_pending_site_xpos[slot, rows] = self.site_xpos[rows].detach()
        self.recovery_pending_phase[slot, rows] = self.phase_idx[rows].detach()
        self.recovery_pending_prev_activation[slot, rows] = self.prev_activation[rows].detach()
        self.recovery_pending_write = (slot + 1) % int(self.recovery_survival_delay_steps)
        self.recovery_last_stage_count = count

    def commit_survived_recovery_pending(self, done: torch.Tensor) -> None:
        slot = int(self.recovery_pending_write)
        rows = torch.nonzero(self.recovery_pending_valid[slot] & (~done), as_tuple=False).flatten()
        if int(rows.numel()) <= 0:
            self.recovery_pending_valid[slot] = False
            return
        count = int(rows.numel())
        start = int(self.recovery_bank_write)
        dst = (torch.arange(count, device=self.device, dtype=torch.long) + start) % int(self.recovery_bank_capacity)
        self.recovery_bank_qpos[dst] = self.recovery_pending_qpos[slot, rows]
        self.recovery_bank_qvel[dst] = self.recovery_pending_qvel[slot, rows]
        self.recovery_bank_act[dst] = self.recovery_pending_act[slot, rows]
        self.recovery_bank_ctrl[dst] = self.recovery_pending_ctrl[slot, rows]
        self.recovery_bank_qacc_warmstart[dst] = self.recovery_pending_qacc_warmstart[slot, rows]
        self.recovery_bank_site_xpos[dst] = self.recovery_pending_site_xpos[slot, rows]
        self.recovery_bank_phase[dst] = self.recovery_pending_phase[slot, rows]
        self.recovery_bank_prev_activation[dst] = self.recovery_pending_prev_activation[slot, rows]
        self.recovery_bank_write = int((start + count) % int(self.recovery_bank_capacity))
        self.recovery_bank_size = min(int(self.recovery_bank_capacity), int(self.recovery_bank_size) + count)
        self.recovery_pending_valid[slot] = False
        self.recovery_last_collect_count = count
        self.recovery_last_commit_count = count

    def reset(self, mask: torch.Tensor) -> None:
        if not bool(mask.any().item()):
            return
        rows = torch.nonzero(mask, as_tuple=False).flatten()
        self.reset_exo_control(rows)
        count = int(rows.numel())
        self.recovery_last_restore_count = 0
        self.offline_recovery_last_restore_count = 0
        if self.offline_recovery_bank_ready() and self.offline_recovery_probability > 0.0:
            use_offline = torch.rand((count,), generator=self.rng, device=self.device) < min(1.0, self.offline_recovery_probability)
        else:
            use_offline = torch.zeros(count, dtype=torch.bool, device=self.device)
        if bool(use_offline.any().item()):
            offline_rows = rows[use_offline]
            self.restore_offline_recovery_states(offline_rows)
            if self.exo_policy_enabled:
                self.ctrl[offline_rows, int(self.model.na) :] = 0.0
        rows = rows[~use_offline]
        count = int(rows.numel())
        if count <= 0:
            return
        if self.recovery_bank_ready() and self.recovery_reset_probability > 0.0:
            use_recovery = torch.rand((count,), generator=self.rng, device=self.device) < min(1.0, self.recovery_reset_probability)
        else:
            use_recovery = torch.zeros(count, dtype=torch.bool, device=self.device)
        if bool(use_recovery.any().item()):
            recovery_rows = rows[use_recovery]
            self.restore_recovery_states(recovery_rows)
        rows = rows[~use_recovery]
        count = int(rows.numel())
        if count <= 0:
            return
        if self.full_state_reset_only and self.offline_recovery_bank_ready():
            self.restore_offline_recovery_states(rows)
            if self.exo_policy_enabled:
                self.ctrl[rows, int(self.model.na) :] = 0.0
            return
        if self.full_state_reset_only:
            raise RuntimeError(
                "reset.full_state_only=true but no offline/online full-state bank row was selected; "
                "set offline_recovery_reset.reset_probability=1.0 and provide a valid bank"
            )
        if self.phase_choices is not None and int(self.phase_choices.numel()) > 0:
            choice_idx = torch.randint(0, int(self.phase_choices.numel()), (count,), generator=self.rng, device=self.device)
            phase = self.phase_choices[choice_idx]
        else:
            phase_low = max(0, min(self.phase_start, int(self.reference["length"]) - 1))
            phase_high = max(phase_low + 1, min(self.phase_end, int(self.reference["length"])))
            phase = torch.randint(phase_low, phase_high, (count,), generator=self.rng, device=self.device)
        self.phase_idx[rows] = phase
        self.qpos[rows] = self.reference["full_reset_qpos"][phase].to(
            device=self.device, dtype=self.qpos.dtype
        )
        self.qvel[rows] = self.reference["full_reset_qvel"][phase].to(
            device=self.device, dtype=self.qvel.dtype
        )
        reset_dq = self.reference["reset_dq_ref"][phase]
        if bool(self.config.get("myoassist_exact", {}).get("scale_reset_qvel_to_target", False)):
            ref_pelvis_vx = reset_dq[:, RESET_JOINTS.index("pelvis_tx")]
            speed_ratio = float(self.myoassist_target_velocity) / torch.clamp(ref_pelvis_vx, min=1e-6)
            reset_dq = reset_dq * speed_ratio.unsqueeze(1)
        self.qvel[rows[:, None], self.reference["reset_qvel_indices"][None, :]] = reset_dq
        if self.reset_qpos_noise > 0.0:
            self.qpos[rows[:, None], self.reference["reset_qpos_indices"][None, :]] += (
                torch.randn((count, len(RESET_JOINTS)), generator=self.rng, device=self.device)
                * self.reset_qpos_noise
            )
        if self.reset_qvel_noise > 0.0:
            self.qvel[rows[:, None], self.reference["reset_qvel_indices"][None, :]] += (
                torch.randn((count, len(RESET_JOINTS)), generator=self.rng, device=self.device)
                * self.reset_qvel_noise
            )
        if "course_offset" in self.reference:
            self.qpos[rows, self.pelvis_tx_qpos] = self.reference["course_offset"][phase] + self.reference["pelvis_tx_ref"][phase]
        else:
            self.qpos[rows, self.pelvis_tx_qpos] = 0.0
        self.phase_idx[rows] = reset_reference_phase_from_x(
            self.qpos[rows],
            self.phase_idx[rows],
            self.reference,
            self.config,
        )
        initial_ctrl = 0.5 * (self.ctrl_low + self.ctrl_high).unsqueeze(0).expand(count, -1).clone()
        if self.initial_activation_high > self.initial_activation_low:
            initial_activation = self.initial_activation_low + (
                torch.rand((count, self.model.na), generator=self.rng, device=self.device)
                * (self.initial_activation_high - self.initial_activation_low)
            )
        else:
            initial_activation = torch.full((count, self.model.na), self.initial_activation, dtype=torch.float32, device=self.device)
        initial_ctrl[:, : int(self.model.na)] = initial_activation
        initial_ctrl = apply_non_muscle_ctrl_override(
            initial_ctrl,
            self.config,
            muscle_count=int(self.model.na),
            ctrl_low=self.ctrl_low,
            ctrl_high=self.ctrl_high,
        )
        if self.exo_policy_enabled:
            initial_ctrl[:, int(self.model.na) :] = 0.0
        self.ctrl[rows] = initial_ctrl
        self.act[rows] = initial_activation
        self.qacc_warmstart[rows] = 0.0
        self.time[rows] = 0.0
        self.prev_activation[rows] = initial_activation
        self.prev_activation_valid[rows] = False
        self.episode_step[rows] = 0
        self.episode_return[rows] = 0.0
        self.episode_length[rows] = 0.0
        self.recovery_mode_steps[rows] = 0
        self.apply_joint_equalities(rows)
