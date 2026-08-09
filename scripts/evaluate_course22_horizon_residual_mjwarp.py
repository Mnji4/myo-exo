#!/usr/bin/env python3
"""Evaluate horizon-optimized control residuals on a closed-loop MJWarp expert."""
from __future__ import annotations

import argparse
import csv
import copy
import json
import sys
from collections import deque
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from myo_exo_train.checkpoint import build_sac_actor_for_checkpoint  # noqa: E402
from myo_exo_train.env.model import (  # noqa: E402
    build_muscle_model,
    muscle_action_mapping_mode,
    policy_action_to_ctrl,
    semantic_qpos_index,
)
from myo_exo_train.env.reference import load_reference_from_config  # noqa: E402
from myo_exo_train.env.runner import MJWarpMuscleRunner  # noqa: E402
from myo_exo_train.evaluation import load_config, set_cpu_reference_state  # noqa: E402
from scripts.analyze_flat22_assisted_allocation_upper_bound import (  # noqa: E402
    configure_direct_exo,
)
from scripts.eval_dynamic_muscle_allocator import (  # noqa: E402
    add_label,
    tint_ghost_model,
)
from scripts.flat22_allocator_distillation_common import (  # noqa: E402
    ExoConditionedHumanStudent,
    ExoPlanStudent,
    ExoStudent,
    RecurrentExoPlanStudent,
    RecurrentExoStudent,
    append_history,
    exo_command_context,
    exo_policy_features,
    proprio_frame,
    proprio_indices,
)
from scripts.compare_flat22_exo_students_target_pd import (  # noqa: E402
    SharedLegTargetPolicy,
)


def ctrl_to_policy_action(ctrl: torch.Tensor, mapping: str, muscle_count: int) -> torch.Tensor:
    action = ctrl.clone()
    muscle_ctrl = ctrl[:, :muscle_count].clamp(1.0e-5, 1.0 - 1.0e-5)
    if mapping in {"myosuite", "myosuite_sigmoid", "myosuite_muscle_sigmoid"}:
        action[:, :muscle_count] = (
            torch.log(muscle_ctrl) - torch.log1p(-muscle_ctrl)
        ) / 5.0 + 0.5
    else:
        action[:, :muscle_count] = 2.0 * muscle_ctrl - 1.0
    return action.clamp(-1.0, 1.0)


def render_snapshot(
    *,
    qpos: np.ndarray,
    phase: int,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    ghost_model: mujoco.MjModel,
    ghost_data: mujoco.MjData,
    ghost_renderer: mujoco.Renderer,
    camera: mujoco.MjvCamera,
    reference: dict,
    config: dict,
    origin_x: float,
    origin_reference_x: float,
) -> np.ndarray:
    """Render a cached state without touching the live MJWarp runner."""
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    data.act[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
    pelvis_tx_qpos = int(
        reference.get("pelvis_tx_qpos", semantic_qpos_index(model, "pelvis_tx"))
    )
    free_joints = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if free_joints.size:
        root_qpos = int(model.jnt_qposadr[int(free_joints[0])])
        camera.lookat[:] = data.qpos[root_qpos : root_qpos + 3]
    else:
        camera.lookat[:] = [float(data.qpos[pelvis_tx_qpos]), 0.0, 0.9]
    renderer.update_scene(data, camera=camera)
    base = renderer.render()

    effective_phase = int(phase) % int(reference["length"])
    set_cpu_reference_state(
        ghost_model,
        ghost_data,
        reference,
        effective_phase,
        config=config,
    )
    ref_x = float(reference["pelvis_tx_ref"][effective_phase].item())
    ghost_data.qpos[pelvis_tx_qpos] = origin_x + ref_x - origin_reference_x
    mujoco.mj_forward(ghost_model, ghost_data)
    ghost_renderer.update_scene(ghost_data, camera=camera)
    ghost = ghost_renderer.render()

    result = base.astype(np.float32)
    ghost_float = ghost.astype(np.float32)
    red, green, blue = (ghost_float[:, :, index] for index in range(3))
    mask = (blue > 35.0) & (blue > red + 22.0) & (blue > green + 8.0)
    result[mask] = 0.52 * result[mask] + 0.48 * ghost_float[mask]
    return np.clip(result, 0, 255).astype(np.uint8)


def load_exo_policy(
    path: Path,
    device: torch.device,
    *,
    use_full_obs: bool = False,
) -> tuple[
    ExoStudent | RecurrentExoStudent,
    torch.Tensor,
    torch.Tensor,
    bool,
    dict[str, object],
]:
    payload = torch.load(path, map_location=device, weights_only=False)
    recurrent = payload.get("model_type") == "recurrent_exo" and not use_full_obs
    if use_full_obs:
        if "full_obs_exo_state_dict" not in payload:
            raise ValueError(f"student has no full-observation Exo head: {path}")
        model = ExoStudent(
            int(payload["obs_dim"]),
            int(payload["hidden_dim"]),
        ).to(device)
        model.load_state_dict(payload["full_obs_exo_state_dict"])
        model.eval()
        return (
            model,
            payload["proprio_mean"].to(device),
            payload["proprio_std"].to(device),
            False,
            payload,
        )
    if recurrent:
        model = RecurrentExoStudent(
            input_dim=int(payload["proprio_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
            expert_count=int(payload["expert_count"]),
            max_delta=float(payload["max_delta"]),
            output_mode=str(payload.get("output_mode", "delta")),
        ).to(device)
    else:
        model = ExoStudent(
            int(payload["proprio_dim"]),
            int(payload["hidden_dim"]),
        ).to(device)
    model.load_state_dict(payload["proprio_exo_state_dict"])
    model.eval()
    return (
        model,
        payload["proprio_mean"].to(device),
        payload["proprio_std"].to(device),
        recurrent,
        payload,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--solution", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--mixes", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0]
    )
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument(
        "--human-solver-residual-scale",
        type=float,
        default=1.0,
        help="Scale solved muscle correction independently from Exo output.",
    )
    parser.add_argument("--bank-index", type=int, default=0)
    parser.add_argument("--exo-max-torque-nm", type=float, default=10.0)
    parser.add_argument(
        "--exo-max-delta-nm-per-step",
        type=float,
        default=0.0,
        help=(
            "Hard executed Exo torque slew limit. Zero keeps the legacy "
            "full-range-per-step limiter."
        ),
    )
    parser.add_argument("--dataset-output", type=Path)
    parser.add_argument(
        "--state-bank-output",
        type=Path,
        help=(
            "Save the dataset route's visited full states, normalized "
            "observations, sensor history, and applied Exo commands."
        ),
    )
    parser.add_argument(
        "--all-state-banks-output-dir",
        type=Path,
        help=(
            "Save one full-state bank per mix from the same batched rollout. "
            "Files are named mix_<value>.npz."
        ),
    )
    parser.add_argument("--dataset-mix", type=float, default=1.0)
    parser.add_argument(
        "--dataset-label-source",
        choices=("executed", "solver"),
        default="executed",
        help=(
            "Store executed student actions or query the horizon solver at "
            "student-visited states for DAgger."
        ),
    )
    parser.add_argument(
        "--dataset-solver-branch",
        choices=("assisted", "muscle_only"),
        default="assisted",
        help="Horizon branch used when --dataset-label-source=solver.",
    )
    parser.add_argument(
        "--dataset-absolute-solver-action",
        action="store_true",
        help="Label the complete solved muscle action instead of a source-policy residual.",
    )
    parser.add_argument(
        "--teacher-branch",
        choices=("assisted", "muscle_only", "paired"),
        default="assisted",
        help=(
            "Horizon solution branch. Paired runs matched muscle-only for "
            "non-positive mixes and assisted for positive mixes."
        ),
    )
    parser.add_argument("--history-steps", type=int, default=8)
    parser.add_argument(
        "--exo-sensor-mode",
        choices=("hip4", "hip4_exo6"),
        default="hip4_exo6",
    )
    parser.add_argument(
        "--exo-use-full-obs",
        action="store_true",
        help="Drive the saved full-observation Exo head instead of the proprio-history head.",
    )
    parser.add_argument(
        "--student",
        type=Path,
        help="Evaluate a distilled human and proprio-history Exo policy instead of the residual teacher.",
    )
    parser.add_argument(
        "--exo-student",
        type=Path,
        help="Optional Exo-only student override; keeps the human from --student fixed.",
    )
    parser.add_argument(
        "--muscle-only-student",
        type=Path,
        help=(
            "Matched low-activation baseline. Non-positive mixes use this "
            "human with Exo forced to zero."
        ),
    )
    parser.add_argument(
        "--teacher-exo-student",
        type=Path,
        help=(
            "Replace the assisted solver's Exo profile with a sensor-limited "
            "Exo student while retaining solver muscle control."
        ),
    )
    parser.add_argument(
        "--teacher-target-pd",
        type=Path,
        help="Use a packaged target-angle PD policy as the assisted Exo teacher.",
    )
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--nconmax", type=int, default=256)
    parser.add_argument("--njmax", type=int, default=1024)
    parser.add_argument("--video-output", type=Path)
    parser.add_argument("--video-width", type=int, default=480)
    parser.add_argument("--video-height", type=int, default=270)
    args = parser.parse_args()
    if args.teacher_exo_student is not None and args.teacher_target_pd is not None:
        parser.error("--teacher-exo-student and --teacher-target-pd are exclusive")

    device = torch.device(args.device)
    config = copy.deepcopy(load_config(args.config))
    with np.load(args.bank, allow_pickle=True) as payload:
        bank = {key: np.asarray(payload[key]) for key in payload.files}
    with np.load(args.solution, allow_pickle=False) as payload:
        solution = {key: np.asarray(payload[key]) for key in payload.files}
    if not np.array_equal(bank["phase"][: len(solution["phase"])], solution["phase"]):
        raise ValueError("bank and solution phases do not match")

    steps = int(args.steps) if args.steps > 0 else len(solution["phase"]) - 1
    config["reset"]["episode_steps"] = steps + 2
    config["reset"]["phase_indices"] = [int(bank["phase"][args.bank_index])]
    config["reset"]["phase_windows"] = []
    config["reset"]["phase_index_jitter"] = 0
    config["reset"]["full_state_only"] = True
    config.setdefault("recovery_reset", {})["enabled"] = False
    config["offline_recovery_reset"] = {
        "enabled": True,
        "path": str(args.bank.resolve()),
        "reset_probability": 1.0,
        "min_bank_size": 1,
        "fixed_index": int(args.bank_index),
    }
    config.setdefault("myoassist_exact", {})["out_of_trajectory_threshold"] = 10.0
    exo_max_delta_control = (
        float(args.exo_max_delta_nm_per_step) / float(args.exo_max_torque_nm)
        if args.exo_max_delta_nm_per_step > 0.0
        else 1.0
    )
    configure_direct_exo(config, args.exo_max_torque_nm, exo_max_delta_control)

    model, data = build_muscle_model(config)
    reference = load_reference_from_config(
        args.reference,
        model,
        float(config["control"]["control_hz"]),
        device,
        config,
    )
    mixes = torch.tensor(args.mixes, dtype=torch.float32, device=device)
    runner = MJWarpMuscleRunner(
        model=model,
        data=data,
        config=config,
        reference=reference,
        nworld=len(args.mixes),
        nconmax=int(args.nconmax),
        njmax=int(args.njmax),
        seed=int(args.seed),
        device=device,
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    actor, normalizer, _ = build_sac_actor_for_checkpoint(
        checkpoint=checkpoint,
        model=model,
        config=config,
        obs_dim=runner.obs_dim,
        act_dim=runner.act_dim,
        device=device,
    )
    actor.eval()
    student_human = None
    muscle_only_human = None
    student_exo = None
    student_proprio_mean = None
    student_proprio_std = None
    recurrent_exo = False
    recurrent_hidden = None
    exo_plan_student = None
    conditioned_human = None
    exo_plan_steps = 0
    recurrent_exo_plan = False
    recurrent_exo_plan_hidden = None
    unified_student = False
    frozen_exo_conditioned = False
    history_conditioned = False
    exo_policy_input_mode = "full_history"
    target_pd_model = None
    target_pd_mean = None
    target_pd_std = None
    target_pd_payload = None
    if args.student is not None:
        if (
            args.dataset_output is not None
            and args.dataset_label_source != "solver"
        ):
            raise ValueError(
                "student dataset collection requires "
                "--dataset-label-source solver"
            )
        student_payload = torch.load(
            args.student, map_location=device, weights_only=False
        )
        unified_student = student_payload.get("student_action_mode") == "unified"
        student_config = config
        if unified_student and student_payload.get("unified_separate_exo_head"):
            student_config = copy.deepcopy(config)
            student_config.setdefault("policy", {})["exo_head"] = copy.deepcopy(
                student_payload["unified_exo_head_config"]
            )
        student_human, _, _ = build_sac_actor_for_checkpoint(
            checkpoint=checkpoint,
            model=model,
            config=student_config,
            obs_dim=runner.obs_dim,
            act_dim=runner.act_dim,
            device=device,
        )
        student_human.load_state_dict(
            student_payload["human_actor_state_dict"], strict=True
        )
        student_human.eval()
        if args.muscle_only_student is not None:
            muscle_only_payload = torch.load(
                args.muscle_only_student,
                map_location=device,
                weights_only=False,
            )
            muscle_only_config = config
            if muscle_only_payload.get("unified_separate_exo_head"):
                muscle_only_config = copy.deepcopy(config)
                muscle_only_config.setdefault("policy", {})[
                    "exo_head"
                ] = copy.deepcopy(
                    muscle_only_payload["unified_exo_head_config"]
                )
            muscle_only_human, _, _ = build_sac_actor_for_checkpoint(
                checkpoint=checkpoint,
                model=model,
                config=muscle_only_config,
                obs_dim=runner.obs_dim,
                act_dim=runner.act_dim,
                device=device,
            )
            muscle_only_human.load_state_dict(
                muscle_only_payload["human_actor_state_dict"], strict=True
            )
            muscle_only_human.eval()
        plan_conditioned = (
            student_payload.get("model_type")
            in {
                "exo_plan_conditioned_human",
                "recurrent_exo_plan_conditioned_human",
            }
            and args.exo_student is None
        )
        frozen_exo_conditioned = (
            student_payload.get("model_type")
            == "frozen_recurrent_exo_conditioned_human"
        )
        history_conditioned = (
            student_payload.get("model_type") == "exo_history_conditioned_human"
            or bool(student_payload.get("conditioned_human_active", False))
        )
        if (
            history_conditioned
            and student_payload.get("conditioner_exo_head") == "full_obs"
        ):
            args.exo_use_full_obs = True
        if unified_student:
            exo_payload = None
        elif plan_conditioned:
            exo_plan_steps = int(student_payload["plan_steps"])
            recurrent_exo_plan = (
                student_payload.get("model_type")
                == "recurrent_exo_plan_conditioned_human"
            )
            if recurrent_exo_plan:
                exo_plan_student = RecurrentExoPlanStudent(
                    int(student_payload["proprio_dim"]),
                    int(student_payload["plan_hidden_dim"]),
                    exo_plan_steps,
                ).to(device)
            else:
                exo_plan_student = ExoPlanStudent(
                    int(student_payload["proprio_dim"]),
                    int(student_payload["hidden_dim"]),
                    exo_plan_steps,
                ).to(device)
            exo_plan_student.load_state_dict(
                student_payload["exo_plan_state_dict"], strict=True
            )
            exo_plan_student.eval()
            conditioned_human = ExoConditionedHumanStudent(
                int(student_payload["obs_dim"]),
                int(model.na),
                int(student_payload["hidden_dim"]),
                exo_context_dim=int(student_payload["exo_context_dim"]),
                zero_centered=bool(
                    student_payload.get("conditioned_zero_centered", False)
                ),
                absolute_output=bool(
                    student_payload.get("conditioned_absolute_output", False)
                ),
            ).to(device)
            conditioned_human.load_state_dict(
                student_payload["conditioned_human_state_dict"], strict=True
            )
            conditioned_human.eval()
            student_proprio_mean = student_payload["proprio_mean"].to(device)
            student_proprio_std = student_payload["proprio_std"].to(device)
            exo_payload = student_payload
        else:
            exo_path = (
                args.exo_student
                if args.exo_student is not None
                else (
                    Path(student_payload["frozen_exo_model"])
                    if frozen_exo_conditioned
                    else args.student
                )
            )
            (
                student_exo,
                student_proprio_mean,
                student_proprio_std,
                recurrent_exo,
                exo_payload,
            ) = load_exo_policy(
                exo_path,
                device,
                use_full_obs=bool(args.exo_use_full_obs),
            )
            exo_policy_input_mode = str(
                exo_payload.get("exo_policy_input_mode", "full_history")
            )
            if history_conditioned:
                conditioned_human = ExoConditionedHumanStudent(
                    int(student_payload["obs_dim"]),
                    int(model.na),
                    int(student_payload["hidden_dim"]),
                    exo_context_dim=int(student_payload["exo_context_dim"]),
                    zero_centered=bool(
                        student_payload.get("conditioned_zero_centered", True)
                    ),
                    absolute_output=bool(
                        student_payload.get("conditioned_absolute_output", False)
                    ),
                ).to(device)
                conditioned_human.load_state_dict(
                    student_payload["conditioned_human_state_dict"], strict=True
                )
                conditioned_human.eval()
            if frozen_exo_conditioned:
                conditioned_human = ExoConditionedHumanStudent(
                    int(student_payload["obs_dim"]),
                    int(model.na),
                    int(student_payload["hidden_dim"]),
                    exo_context_dim=int(student_payload["exo_context_dim"]),
                    zero_centered=bool(
                        student_payload.get("conditioned_zero_centered", True)
                    ),
                    absolute_output=bool(
                        student_payload.get("conditioned_absolute_output", False)
                    ),
                ).to(device)
                conditioned_human.load_state_dict(
                    student_payload["conditioned_human_state_dict"], strict=True
                )
                conditioned_human.eval()
        if not unified_student:
            if int(exo_payload["history_steps"]) != int(args.history_steps):
                raise ValueError("student history_steps does not match --history-steps")
            if str(exo_payload["exo_sensor_mode"]) != str(args.exo_sensor_mode):
                raise ValueError("student Exo sensor mode does not match --exo-sensor-mode")
    elif args.teacher_exo_student is not None:
        (
            student_exo,
            student_proprio_mean,
            student_proprio_std,
            recurrent_exo,
            exo_payload,
        ) = load_exo_policy(
            args.teacher_exo_student,
            device,
            use_full_obs=bool(args.exo_use_full_obs),
        )
        exo_policy_input_mode = str(
            exo_payload.get("exo_policy_input_mode", "full_history")
        )
        if int(exo_payload["history_steps"]) != int(args.history_steps):
            raise ValueError("teacher Exo history_steps does not match")
        if str(exo_payload["exo_sensor_mode"]) != str(args.exo_sensor_mode):
            raise ValueError("teacher Exo sensor mode does not match")
    elif args.teacher_target_pd is not None:
        target_pd_payload = torch.load(
            args.teacher_target_pd, map_location=device, weights_only=False
        )
        target_pd_mean = torch.as_tensor(
            target_pd_payload["input_mean"], dtype=torch.float32, device=device
        )
        target_pd_std = torch.as_tensor(
            target_pd_payload["input_std"], dtype=torch.float32, device=device
        )
        target_pd_model = SharedLegTargetPolicy(
            int(target_pd_mean.numel()),
            int(target_pd_payload["hidden_dim"]),
            float(target_pd_payload["target_offset_limit_rad"]),
        ).to(device)
        target_pd_model.load_state_dict(target_pd_payload["state_dict"], strict=True)
        target_pd_model.eval()
        if int(target_pd_payload["history_steps"]) != int(args.history_steps):
            raise ValueError("target PD history_steps does not match")
    if args.exo_student is not None and args.student is None:
        raise ValueError("--exo-student requires --student")
    if args.teacher_branch == "paired" and args.dataset_output is not None:
        raise ValueError("paired teacher mode cannot collect one student dataset")
    mapping = muscle_action_mapping_mode(config)
    muscle_count = int(model.na)
    ctrl_low = runner.ctrl_low
    ctrl_high = runner.ctrl_high

    phase_to_index = {
        int(phase): index for index, phase in enumerate(solution["phase"].tolist())
    }
    bank_ctrl = torch.tensor(
        bank["ctrl"][:, : int(model.nu)], dtype=torch.float32, device=device
    )
    muscle_only_excitation = torch.tensor(
        solution["muscle_only_excitation"], dtype=torch.float32, device=device
    )
    assisted_excitation = torch.tensor(
        solution["assisted_excitation"], dtype=torch.float32, device=device
    )
    assisted_exo = torch.tensor(
        solution["assisted_exo_control"], dtype=torch.float32, device=device
    )
    proprio_qpos, proprio_qvel = proprio_indices(model, args.exo_sensor_mode)
    histories = [
        deque(maxlen=max(1, int(args.history_steps))) for _ in args.mixes
    ]
    conditioned_command_histories = [
        deque(maxlen=max(1, int(args.history_steps))) for _ in args.mixes
    ]
    dataset_world = min(
        range(len(args.mixes)),
        key=lambda index: abs(float(args.mixes[index]) - float(args.dataset_mix)),
    )
    if abs(float(args.mixes[dataset_world]) - float(args.dataset_mix)) > 1.0e-6:
        raise ValueError("--dataset-mix must be one of --mixes")
    dataset_obs: list[np.ndarray] = []
    dataset_proprio: list[np.ndarray] = []
    dataset_muscle_action: list[np.ndarray] = []
    dataset_exo_action: list[np.ndarray] = []
    dataset_phase: list[int] = []
    state_qpos: list[np.ndarray] = []
    state_qvel: list[np.ndarray] = []
    state_act: list[np.ndarray] = []
    state_ctrl: list[np.ndarray] = []
    state_prev_activation: list[np.ndarray] = []
    state_qacc_warmstart: list[np.ndarray] = []
    state_site_xpos: list[np.ndarray] = []
    state_phase: list[int] = []
    state_x_align_mask: list[bool] = []
    state_applied_exo_ctrl: list[np.ndarray] = []
    state_normalized_obs: list[np.ndarray] = []
    state_proprio_history: list[np.ndarray] = []
    all_state_banks: list[dict[str, list]] = []
    if args.all_state_banks_output_dir is not None:
        all_state_banks = [
            {
                key: []
                for key in (
                    "qpos",
                    "qvel",
                    "act",
                    "ctrl",
                    "prev_activation",
                    "qacc_warmstart",
                    "site_xpos",
                    "phase",
                    "x_align_mask",
                    "applied_exo_ctrl",
                    "normalized_obs",
                    "proprio_history",
                )
            }
            for _ in args.mixes
        ]

    obs = runner.obs()
    active = torch.ones(len(args.mixes), dtype=torch.bool, device=device)
    origin_x = runner.qpos[:, runner.pelvis_tx_qpos].clone()
    rows: list[dict[str, float | int | bool]] = []
    completion: list[dict[str, float | int | bool] | None] = [None] * len(args.mixes)
    render_qpos: list[np.ndarray] = []
    render_phase: list[np.ndarray] = []

    for frame in range(steps):
        phase_before = runner.phase_idx.clone()
        with torch.no_grad():
            normalized_obs = normalizer.normalize(obs)
            proprio_histories: list[np.ndarray] = []
            for world in range(len(args.mixes)):
                proprio = proprio_frame(
                    runner.qpos[world].detach().cpu().numpy(),
                    runner.qvel[world].detach().cpu().numpy(),
                    runner.applied_exo_ctrl[world].detach().cpu().numpy(),
                    proprio_qpos,
                    proprio_qvel,
                    args.exo_sensor_mode,
                )
                proprio_histories.append(
                    append_history(
                        histories[world], proprio, int(args.history_steps)
                    ).copy()
                )
            base_action, _, _, _ = actor.get_action_and_value(
                normalized_obs, deterministic=True
            )
            if all_state_banks:
                for world, world_state in enumerate(all_state_banks):
                    if not bool(active[world].item()):
                        continue
                    for key in (
                        "qpos",
                        "qvel",
                        "act",
                        "ctrl",
                        "prev_activation",
                        "qacc_warmstart",
                        "site_xpos",
                        "applied_exo_ctrl",
                    ):
                        world_state[key].append(
                            getattr(runner, key)[world]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32, copy=True)
                        )
                    world_state["phase"].append(int(phase_before[world].item()))
                    world_state["x_align_mask"].append(
                        bool(runner.x_align_mask[world].item())
                    )
                    world_state["normalized_obs"].append(
                        normalized_obs[world]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32, copy=True)
                    )
                    world_state["proprio_history"].append(
                        proprio_histories[world].astype(np.float32, copy=True)
                    )
            if (
                args.state_bank_output is not None
                and bool(active[dataset_world].item())
            ):
                state_qpos.append(
                    runner.qpos[dataset_world]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=True)
                )
                state_qvel.append(
                    runner.qvel[dataset_world]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=True)
                )
                state_act.append(
                    runner.act[dataset_world]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=True)
                )
                state_ctrl.append(
                    runner.ctrl[dataset_world]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=True)
                )
                state_prev_activation.append(
                    runner.prev_activation[dataset_world]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=True)
                )
                state_qacc_warmstart.append(
                    runner.qacc_warmstart[dataset_world]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=True)
                )
                state_site_xpos.append(
                    runner.site_xpos[dataset_world]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=True)
                )
                state_phase.append(int(phase_before[dataset_world].item()))
                state_x_align_mask.append(
                    bool(runner.x_align_mask[dataset_world].item())
                )
                state_applied_exo_ctrl.append(
                    runner.applied_exo_ctrl[dataset_world]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=True)
                )
                state_normalized_obs.append(
                    normalized_obs[dataset_world]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=True)
                )
                state_proprio_history.append(
                    proprio_histories[dataset_world].astype(
                        np.float32, copy=True
                    )
                )
            base_ctrl = policy_action_to_ctrl(
                base_action,
                ctrl_low,
                ctrl_high,
                muscle_count=muscle_count,
                muscle_mapping=mapping,
            )
            corrected_ctrl = base_ctrl.clone()
            residual_abs = torch.zeros(len(args.mixes), device=device)
            profile_found = torch.zeros(len(args.mixes), dtype=torch.bool, device=device)
            gate_expert = torch.full(
                (len(args.mixes),), -1, dtype=torch.long, device=device
            )
            gate_confidence = torch.zeros(len(args.mixes), device=device)
            predicted_exo = None
            exo_context = None
            if exo_plan_student is not None:
                proprio_tensor = torch.from_numpy(
                    np.stack(proprio_histories)
                ).to(device)
                if recurrent_exo_plan:
                    current_proprio = proprio_tensor[:, -6:]
                    normalized_proprio = (
                        current_proprio - student_proprio_mean
                    ) / student_proprio_std
                    predicted_plan, recurrent_exo_plan_hidden = (
                        exo_plan_student.step(
                            normalized_proprio,
                            recurrent_exo_plan_hidden,
                        )
                    )
                else:
                    normalized_proprio = (
                        proprio_tensor - student_proprio_mean
                    ) / student_proprio_std
                    predicted_plan = exo_plan_student(normalized_proprio)
                predicted_exo = predicted_plan[:, 0]
                command_history = (
                    proprio_tensor.reshape(
                        len(args.mixes), int(args.history_steps), 6
                    )[:, :, -2:]
                    .reshape(len(args.mixes), int(args.history_steps) * 2)
                )
                exo_context = torch.cat(
                    (command_history, predicted_plan.flatten(1)), dim=-1
                )
            elif target_pd_model is not None:
                proprio_tensor = torch.from_numpy(
                    np.stack(proprio_histories)
                ).to(device)
                history = proprio_tensor.reshape(
                    len(args.mixes), int(args.history_steps), 6
                )
                right = history.reshape(len(args.mixes), -1)
                left = history[:, :, [1, 0, 3, 2, 5, 4]].reshape(
                    len(args.mixes), -1
                )
                features = torch.cat((right, left), dim=0)
                normalized_features = (
                    features - target_pd_mean
                ) / target_pd_std
                offsets = target_pd_model(normalized_features)
                world_count = len(args.mixes)
                velocity = history[:, -1, 2:4]
                torque_r = (
                    float(target_pd_payload["kp_nm_per_rad"])
                    * offsets[:world_count]
                    + float(target_pd_payload["kd_nm_s_per_rad"])
                    * velocity[:, 0]
                )
                torque_l = (
                    float(target_pd_payload["kp_nm_per_rad"])
                    * offsets[world_count:]
                    + float(target_pd_payload["kd_nm_s_per_rad"])
                    * velocity[:, 1]
                )
                torque_limit = float(target_pd_payload.get("torque_limit_nm", 10.0))
                torque_scale = float(target_pd_payload.get("torque_scale_nm", 10.0))
                predicted_exo = torch.stack((torque_r, torque_l), dim=1)
                predicted_exo = torch.clamp(
                    predicted_exo, -torque_limit, torque_limit
                ) / torque_scale
            elif student_exo is not None:
                proprio_tensor = torch.from_numpy(
                    np.stack(proprio_histories)
                ).to(device)
                if args.exo_use_full_obs:
                    predicted_exo = student_exo(normalized_obs)
                if recurrent_exo:
                    current_proprio = proprio_tensor[:, -6:]
                    normalized_proprio = (
                        current_proprio - student_proprio_mean
                    ) / student_proprio_std
                    predicted_exo, recurrent_hidden, gate_logits = student_exo.step(
                        normalized_proprio,
                        current_proprio[:, -2:],
                        recurrent_hidden,
                    )
                    if gate_logits is not None:
                        gate_probability = torch.softmax(gate_logits, dim=-1)
                        gate_confidence, gate_expert = torch.max(
                            gate_probability, dim=-1
                        )
                elif not args.exo_use_full_obs:
                    policy_features = exo_policy_features(
                        proprio_tensor,
                        int(args.history_steps),
                        str(args.exo_sensor_mode),
                        exo_policy_input_mode,
                    )
                    normalized_proprio = (
                        policy_features - student_proprio_mean
                    ) / student_proprio_std
                    predicted_exo = student_exo(normalized_proprio)
                if history_conditioned:
                    exo_context = exo_command_context(
                        proprio_tensor,
                        predicted_exo,
                        int(args.history_steps),
                        str(args.exo_sensor_mode),
                    )
                if frozen_exo_conditioned:
                    command_rows = []
                    for world in range(len(args.mixes)):
                        history = conditioned_command_histories[world]
                        history.append(
                            predicted_exo[world]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32, copy=True)
                        )
                        padded = [np.zeros(2, dtype=np.float32)] * (
                            int(args.history_steps) - len(history)
                        )
                        command_rows.append(
                            np.concatenate([*padded, *list(history)])
                        )
                    exo_context = torch.from_numpy(
                        np.stack(command_rows)
                    ).to(device)
            if student_human is None:
                for world in range(len(args.mixes)):
                    phase = int(phase_before[world].item()) % int(reference["length"])
                    profile_index = phase_to_index.get(phase)
                    if profile_index is None:
                        continue
                    profile_found[world] = True
                    next_index = min(profile_index + 1, len(solution["phase"]) - 1)
                    if args.teacher_branch == "paired":
                        assisted_world = float(mixes[world]) > 0.0
                        target_excitation = (
                            assisted_excitation
                            if assisted_world
                            else muscle_only_excitation
                        )
                        target_exo = (
                            assisted_exo[profile_index]
                            if assisted_world
                            else torch.zeros(2, device=device)
                        )
                        if assisted_world and predicted_exo is not None:
                            target_exo = predicted_exo[world]
                        residual_scale = float(args.human_solver_residual_scale)
                        exo_scale = 1.0 if assisted_world else 0.0
                    else:
                        target_excitation = (
                            assisted_excitation
                            if args.teacher_branch == "assisted"
                            else muscle_only_excitation
                        )
                        target_exo = (
                            assisted_exo[profile_index]
                            if args.teacher_branch == "assisted"
                            else torch.zeros(2, device=device)
                        )
                        if (
                            args.teacher_branch == "assisted"
                            and predicted_exo is not None
                        ):
                            target_exo = predicted_exo[world]
                        residual_scale = float(mixes[world]) * float(
                            args.human_solver_residual_scale
                        )
                        exo_scale = float(mixes[world])
                    delta = (
                        target_excitation[next_index]
                        - bank_ctrl[next_index, :muscle_count]
                    )
                    scaled_delta = residual_scale * delta
                    corrected_ctrl[world, :muscle_count] = torch.clamp(
                        base_ctrl[world, :muscle_count] + scaled_delta,
                        ctrl_low[:muscle_count],
                        ctrl_high[:muscle_count],
                    )
                    corrected_ctrl[world, muscle_count : muscle_count + 2] = (
                        exo_scale * target_exo
                    )
                    residual_abs[world] = torch.mean(torch.abs(scaled_delta))
                corrected_action = ctrl_to_policy_action(
                    corrected_ctrl, mapping, muscle_count
                )
            else:
                student_action, _, _, _ = student_human.get_action_and_value(
                    normalized_obs, deterministic=True
                )
                if unified_student:
                    predicted_exo = student_action[
                        :, muscle_count : muscle_count + 2
                    ]
                muscle_only_action = None
                if muscle_only_human is not None:
                    muscle_only_action, _, _, _ = (
                        muscle_only_human.get_action_and_value(
                            normalized_obs, deterministic=True
                        )
                    )
                if predicted_exo is None:
                    raise RuntimeError("student evaluation requires an Exo policy")
                conditioned_action = None
                if conditioned_human is not None:
                    if exo_context is None:
                        raise RuntimeError("conditioned human requires Exo context")
                    conditioned_action = conditioned_human(
                        normalized_obs,
                        student_action[:, :muscle_count],
                        exo_context,
                    )
                corrected_action = base_action.clone()
                corrected_action[:, muscle_count : muscle_count + 2] = 0.0
                for world in range(len(args.mixes)):
                    if float(args.mixes[world]) <= 0.0:
                        if muscle_only_action is not None:
                            profile_found[world] = True
                            corrected_action[world, :muscle_count] = (
                                muscle_only_action[world, :muscle_count]
                            )
                        continue
                    profile_found[world] = True
                    source_muscle = (
                        conditioned_action
                        if conditioned_action is not None
                        else student_action[:, :muscle_count]
                    )
                    corrected_action[world, :muscle_count] = source_muscle[world]
                    corrected_action[
                        world, muscle_count : muscle_count + 2
                    ] = predicted_exo[world]
                    residual_abs[world] = torch.mean(
                        torch.abs(
                            source_muscle[world]
                            - base_action[world, :muscle_count]
                        )
                    )
            if (
                args.dataset_output is not None
                and bool(active[dataset_world].item())
                and bool(profile_found[dataset_world].item())
            ):
                labelled_action = corrected_action[dataset_world]
                if args.dataset_label_source == "solver":
                    phase = (
                        int(phase_before[dataset_world].item())
                        % int(reference["length"])
                    )
                    profile_index = phase_to_index.get(phase)
                    if profile_index is None:
                        continue
                    next_index = min(
                        profile_index + 1, len(solution["phase"]) - 1
                    )
                    labelled_ctrl = base_ctrl[dataset_world].clone()
                    solver_excitation = (
                        assisted_excitation
                        if args.dataset_solver_branch == "assisted"
                        else muscle_only_excitation
                    )
                    if args.dataset_absolute_solver_action:
                        labelled_ctrl[:muscle_count] = solver_excitation[next_index]
                    else:
                        labelled_ctrl[:muscle_count] = torch.clamp(
                            base_ctrl[dataset_world, :muscle_count]
                            + solver_excitation[next_index]
                            - bank_ctrl[next_index, :muscle_count],
                            ctrl_low[:muscle_count],
                            ctrl_high[:muscle_count],
                        )
                    labelled_ctrl[muscle_count : muscle_count + 2] = (
                        assisted_exo[profile_index]
                        if args.dataset_solver_branch == "assisted"
                        else 0.0
                    )
                    labelled_action = ctrl_to_policy_action(
                        labelled_ctrl.unsqueeze(0), mapping, muscle_count
                    )[0]
                dataset_obs.append(
                    normalized_obs[dataset_world]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=True)
                )
                dataset_proprio.append(
                    proprio_histories[dataset_world]
                )
                dataset_muscle_action.append(
                    labelled_action[:muscle_count]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=True)
                )
                dataset_exo_action.append(
                    labelled_action[muscle_count : muscle_count + 2]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=True)
                )
                dataset_phase.append(int(phase_before[dataset_world].item()))
            obs, _reward, done, terms = runner.step(corrected_action)

        if args.video_output is not None:
            qpos_snapshot = (
                runner.qpos.detach().cpu().numpy().astype(np.float32, copy=True)
            )
            phase_snapshot = (
                runner.target_phase_idx()
                .detach()
                .cpu()
                .numpy()
                .astype(np.int64, copy=True)
            )
            if render_qpos:
                inactive = ~active.detach().cpu().numpy()
                qpos_snapshot[inactive] = render_qpos[-1][inactive]
                phase_snapshot[inactive] = render_phase[-1][inactive]
            render_qpos.append(qpos_snapshot)
            render_phase.append(phase_snapshot)

        for world, mix in enumerate(args.mixes):
            if not bool(active[world].item()):
                continue
            displacement = float(
                runner.qpos[world, runner.pelvis_tx_qpos].item()
                - origin_x[world].item()
            )
            if bool(done[world].item()):
                displacement = float(
                    terms["episode_forward_displacement_done"][world].item()
                )
            row = {
                "frame": frame + 1,
                "phase": int(phase_before[world].item()),
                "mix": float(mix),
                "profile_found": bool(profile_found[world].item()),
                "activation_l2": float(
                    torch.mean(torch.square(runner.last_step_activation[world])).item()
                ),
                "hip_exo_abs_nm": float(
                    args.exo_max_torque_nm
                    * torch.mean(torch.abs(runner.applied_exo_ctrl[world])).item()
                ),
                "hip_exo_r_nm": float(
                    args.exo_max_torque_nm
                    * runner.applied_exo_ctrl[world, 0].item()
                ),
                "hip_exo_l_nm": float(
                    args.exo_max_torque_nm
                    * runner.applied_exo_ctrl[world, 1].item()
                ),
                "muscle_residual_abs_mean": float(residual_abs[world].item()),
                "exo_gate_expert": int(gate_expert[world].item()),
                "exo_gate_confidence": float(gate_confidence[world].item()),
                "forward_displacement_m": displacement,
                "pelvis_forward_velocity": float(
                    runner.qvel[world, runner.pelvis_tx_qvel].item()
                ),
                "reference_tracking_error": float(
                    terms["reference_tracking_error"][world].item()
                ),
                "done": bool(done[world].item()),
                "fall": bool(terms["fall_done"][world].item()),
                "low_height": bool(terms["low_height_done"][world].item()),
            }
            rows.append(row)
            if row["done"]:
                completion[world] = row
                active[world] = False
        if not bool(active.any().item()):
            break

    summary: dict[str, dict[str, float | int | bool | None]] = {}
    for world, mix in enumerate(args.mixes):
        selected = [row for row in rows if float(row["mix"]) == float(mix)]
        steady = selected[min(int(args.warmup), len(selected)) :]
        terminal = completion[world]
        summary[f"{float(mix):.3f}"] = {
            "frames": len(selected),
            "duration_seconds": len(selected) / float(config["control"]["control_hz"]),
            "completed_horizon": len(selected) >= steps,
            "fell": bool(terminal["fall"]) if terminal is not None else False,
            "forward_displacement_m": (
                float(selected[-1]["forward_displacement_m"]) if selected else 0.0
            ),
            "activation_l2": (
                float(np.mean([float(row["activation_l2"]) for row in steady]))
                if steady
                else None
            ),
            "hip_exo_abs_nm": (
                float(np.mean([float(row["hip_exo_abs_nm"]) for row in steady]))
                if steady
                else None
            ),
            "tracking_error": (
                float(
                    np.mean([float(row["reference_tracking_error"]) for row in steady])
                )
                if steady
                else None
            ),
            "exo_gate_usage": {
                str(expert): sum(
                    int(row["exo_gate_expert"]) == expert for row in steady
                )
                / len(steady)
                for expert in sorted(
                    {
                        int(row["exo_gate_expert"])
                        for row in steady
                        if int(row["exo_gate_expert"]) >= 0
                    }
                )
            },
        }
    baseline = summary.get("0.000", {}).get("activation_l2")
    if isinstance(baseline, float) and baseline > 0.0:
        for result in summary.values():
            value = result["activation_l2"]
            result["activation_reduction_vs_base"] = (
                1.0 - float(value) / baseline if isinstance(value, float) else None
            )

    args.outdir.mkdir(parents=True, exist_ok=True)
    with (args.outdir / "rollout.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if args.dataset_output is not None:
        if not dataset_obs:
            raise RuntimeError("dataset collection produced no samples")
        args.dataset_output.parent.mkdir(parents=True, exist_ok=True)
        sample_count = len(dataset_obs)
        np.savez_compressed(
            args.dataset_output,
            normalized_obs=np.stack(dataset_obs),
            proprio_history=np.stack(dataset_proprio),
            muscle_action=np.stack(dataset_muscle_action),
            exo_action=np.stack(dataset_exo_action),
            solver_success=np.ones(sample_count, dtype=np.bool_),
            exo_supervision_mask=np.ones(sample_count, dtype=np.bool_),
            counterfactual_fraction=np.ones(sample_count, dtype=np.float32),
            phase=np.asarray(dataset_phase, dtype=np.int64),
            history_steps=np.asarray([int(args.history_steps)], dtype=np.int64),
            exo_sensor_mode=np.asarray([str(args.exo_sensor_mode)]),
            dataset_route=np.asarray(["horizon_residual_closed_loop"]),
            teacher_branch=np.asarray([str(args.teacher_branch)]),
        )
    if args.state_bank_output is not None:
        if not state_qpos:
            raise RuntimeError("state bank collection produced no states")
        args.state_bank_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.state_bank_output,
            qpos=np.stack(state_qpos),
            qvel=np.stack(state_qvel),
            act=np.stack(state_act),
            ctrl=np.stack(state_ctrl),
            prev_activation=np.stack(state_prev_activation),
            qacc_warmstart=np.stack(state_qacc_warmstart),
            site_xpos=np.stack(state_site_xpos),
            phase=np.asarray(state_phase, dtype=np.int64),
            x_align_mask=np.asarray(state_x_align_mask, dtype=np.bool_),
            applied_exo_ctrl=np.stack(state_applied_exo_ctrl),
            normalized_obs=np.stack(state_normalized_obs),
            proprio_history=np.stack(state_proprio_history),
            history_steps=np.asarray([int(args.history_steps)], dtype=np.int64),
            exo_sensor_mode=np.asarray([str(args.exo_sensor_mode)]),
            metadata=np.asarray(
                {
                    "source_checkpoint": str(args.checkpoint),
                    "student": (
                        None if args.student is None else str(args.student)
                    ),
                    "seed": int(args.seed),
                    "mix": float(args.mixes[dataset_world]),
                    "steps": int(steps),
                },
                dtype=object,
            ),
        )
    if all_state_banks:
        args.all_state_banks_output_dir.mkdir(parents=True, exist_ok=True)
        for world, (mix, state) in enumerate(zip(args.mixes, all_state_banks)):
            if not state["qpos"]:
                continue
            np.savez_compressed(
                args.all_state_banks_output_dir / f"mix_{float(mix):.3f}.npz",
                qpos=np.stack(state["qpos"]),
                qvel=np.stack(state["qvel"]),
                act=np.stack(state["act"]),
                ctrl=np.stack(state["ctrl"]),
                prev_activation=np.stack(state["prev_activation"]),
                qacc_warmstart=np.stack(state["qacc_warmstart"]),
                site_xpos=np.stack(state["site_xpos"]),
                phase=np.asarray(state["phase"], dtype=np.int64),
                x_align_mask=np.asarray(state["x_align_mask"], dtype=np.bool_),
                applied_exo_ctrl=np.stack(state["applied_exo_ctrl"]),
                normalized_obs=np.stack(state["normalized_obs"]),
                proprio_history=np.stack(state["proprio_history"]),
                history_steps=np.asarray([int(args.history_steps)], dtype=np.int64),
                exo_sensor_mode=np.asarray([str(args.exo_sensor_mode)]),
                metadata=np.asarray(
                    {
                        "source_checkpoint": str(args.checkpoint),
                        "student": None if args.student is None else str(args.student),
                        "seed": int(args.seed),
                        "mix": float(mix),
                        "world": int(world),
                        "steps": int(steps),
                    },
                    dtype=object,
                ),
            )
    if args.video_output is not None:
        initial_phase = int(bank["phase"][args.bank_index]) % int(
            reference["length"]
        )
        origin_reference_x = float(
            reference["pelvis_tx_ref"][initial_phase].item()
        )
        origin_x_cpu = origin_x.detach().cpu().numpy()
        render_assets = []
        for _ in args.mixes:
            render_data = mujoco.MjData(model)
            renderer = mujoco.Renderer(
                model,
                height=int(args.video_height),
                width=int(args.video_width),
            )
            ghost_model = copy.deepcopy(model)
            tint_ghost_model(ghost_model)
            ghost_data = mujoco.MjData(ghost_model)
            ghost_renderer = mujoco.Renderer(
                ghost_model,
                height=int(args.video_height),
                width=int(args.video_width),
            )
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.distance = 7.0
            camera.azimuth = 135.0
            camera.elevation = -30.0
            render_assets.append(
                (
                    render_data,
                    renderer,
                    ghost_model,
                    ghost_data,
                    ghost_renderer,
                    camera,
                )
            )

        frames: list[np.ndarray] = []
        for qpos_snapshot, phase_snapshot in zip(render_qpos, render_phase):
            images = []
            for world, mix in enumerate(args.mixes):
                (
                    render_data,
                    renderer,
                    ghost_model,
                    ghost_data,
                    ghost_renderer,
                    camera,
                ) = render_assets[world]
                label = (
                    "Muscle-only (Exo off)"
                    if float(mix) <= 0.0
                    else "Exo-assisted"
                )
                images.append(
                    add_label(
                        render_snapshot(
                            qpos=qpos_snapshot[world],
                            phase=int(phase_snapshot[world]),
                            model=model,
                            data=render_data,
                            renderer=renderer,
                            ghost_model=ghost_model,
                            ghost_data=ghost_data,
                            ghost_renderer=ghost_renderer,
                            camera=camera,
                            reference=reference,
                            config=config,
                            origin_x=float(origin_x_cpu[world]),
                            origin_reference_x=origin_reference_x,
                        ),
                        label,
                    )
                )
            frames.append(np.concatenate(images, axis=1))
        args.video_output.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(
            args.video_output,
            frames,
            fps=int(config["control"]["control_hz"]),
            quality=8,
        )
        for asset in render_assets:
            asset[1].close()
            asset[4].close()
    output = {
        "checkpoint": str(args.checkpoint),
        "bank": str(args.bank),
        "solution": str(args.solution),
        "student": None if args.student is None else str(args.student),
        "muscle_only_student": (
            None
            if args.muscle_only_student is None
            else str(args.muscle_only_student)
        ),
        "exo_student": (
            None if args.exo_student is None else str(args.exo_student)
        ),
        "teacher_exo_student": (
            None
            if args.teacher_exo_student is None
            else str(args.teacher_exo_student)
        ),
        "teacher_target_pd": (
            None if args.teacher_target_pd is None else str(args.teacher_target_pd)
        ),
        "steps": steps,
        "teacher_branch": str(args.teacher_branch),
        "baseline": (
            "matched_low_activation_muscle_only"
            if args.muscle_only_student is not None
            or args.teacher_branch == "paired"
            else "original_source_policy"
        ),
        "warmup": int(args.warmup),
        "exo_max_delta_nm_per_step": float(
            args.exo_max_delta_nm_per_step
        ),
        "human_solver_residual_scale": float(args.human_solver_residual_scale),
        "summary": summary,
    }
    (args.outdir / "summary.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
