#!/usr/bin/env python3
"""Compare policy, muscle-only allocation, and assisted allocation on flat22."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
from collections import deque
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from myo_exo_train.checkpoint import build_sac_actor_for_checkpoint  # noqa: E402
from myo_exo_train.env.model import (  # noqa: E402
    TRACK_JOINTS,
    build_muscle_model,
    muscle_action_mapping_mode,
    policy_action_to_ctrl,
    semantic_qpos_index,
)
from myo_exo_train.env.observation import reference_index  # noqa: E402
from myo_exo_train.env.reference import load_reference_from_config  # noqa: E402
from myo_exo_train.env.runner import MJWarpMuscleRunner  # noqa: E402
from myo_exo_train.evaluation import load_config  # noqa: E402
from scripts.eval_dynamic_muscle_allocator import (  # noqa: E402
    JOINTS,
    active_torque_map,
    add_label,
    allocate_activation,
    allocate_activation_and_exo,
    allocate_hip_pair_impulse_replacement,
    allocate_hip_pair_replacement,
    cocontraction_nm,
    evolve_activation,
    excitation_for_target_activation,
    excitation_to_action,
    exo_torque_map,
    render_world,
    tint_ghost_model,
)
from scripts.flat22_allocator_distillation_common import (  # noqa: E402
    EXO_SENSOR_MODES,
    ExoConditionedHumanStudent,
    ExoStudent,
    RecurrentExoPlanStudent,
    append_history,
    proprio_frame,
    proprio_indices,
)
from scripts.compare_flat22_exo_students_target_pd import (  # noqa: E402
    SharedLegTargetPolicy,
)


ROUTE_NAMES = ("policy", "muscle_allocator", "assisted_allocator")


def csv_name(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p50": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "max": float(array.max()),
    }


def set_non_muscle_override(config: dict, enabled: bool) -> None:
    override = {"enabled": bool(enabled), "value": 0.0}
    config["non_muscle_ctrl_override"] = copy.deepcopy(override)
    config.setdefault("control", {})["non_muscle_ctrl_override"] = copy.deepcopy(override)


def configure_direct_exo(config: dict, max_torque_nm: float, max_delta: float) -> None:
    config.setdefault("model", {})["exo_direct_hip_motor"] = {
        "enabled": True,
        "max_torque_nm": float(max_torque_nm),
    }
    set_non_muscle_override(config, False)
    config["exo_policy"] = {
        "enabled": True,
        "bidirectional": True,
        "max_ctrl": 1.0,
        "max_delta_per_step": float(max_delta),
    }


def build_render_assets(model: mujoco.MjModel, height: int, width: int):
    render_data = [mujoco.MjData(model) for _ in ROUTE_NAMES]
    renderers = [mujoco.Renderer(model, height=height, width=width) for _ in ROUTE_NAMES]
    ghost_models = [copy.deepcopy(model) for _ in ROUTE_NAMES]
    for ghost_model in ghost_models:
        tint_ghost_model(ghost_model)
    ghost_data = [mujoco.MjData(ghost_model) for ghost_model in ghost_models]
    ghost_renderers = [
        mujoco.Renderer(ghost_models[index], height=height, width=width)
        for index in range(len(ROUTE_NAMES))
    ]
    cameras = []
    for _ in ROUTE_NAMES:
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.distance = 7.0
        camera.azimuth = 135.0
        camera.elevation = -30.0
        cameras.append(camera)
    return render_data, renderers, ghost_models, ghost_data, ghost_renderers, cameras


def close_render_assets(assets) -> None:
    if assets is None:
        return
    _render_data, renderers, _ghost_models, _ghost_data, ghost_renderers, _cameras = assets
    for renderer in [*renderers, *ghost_renderers]:
        renderer.close()


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--phase", type=int, default=597)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--steady-start", type=int, default=30)
    parser.add_argument("--exo-max-torque-nm", type=float, default=10.0)
    parser.add_argument("--exo-max-delta", type=float, default=0.25)
    parser.add_argument("--exo-l2-weight", type=float, default=1.0e-3)
    parser.add_argument("--exo-delta-weight", type=float, default=1.0e-2)
    parser.add_argument(
        "--assisted-allocation-mode",
        choices=("full", "hip_pair", "hip_pair_impulse"),
        default="full",
        help=(
            "Use the unrestricted allocator or only replace bilateral "
            "glutmax/iliopsoas torque with Exo."
        ),
    )
    parser.add_argument(
        "--muscle-allocation-mode",
        choices=("full", "hip_pair"),
        default="full",
        help=(
            "Use the unrestricted muscle-only allocator or reallocate only "
            "bilateral glutmax/iliopsoas for a matched Exo comparison."
        ),
    )
    parser.add_argument(
        "--allocator-execution-mix",
        type=float,
        default=1.0,
        help=(
            "Interpolate next-step activation and Exo torque from the original "
            "policy target (0) to the low-activation allocator solution (1)."
        ),
    )
    parser.add_argument(
        "--prevent-exo-opposition",
        action="store_true",
        help="Constrain each Exo hip torque to share the target torque's sign without overshoot.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--out-of-trajectory-threshold", type=float)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=270)
    parser.add_argument("--dataset-output", type=Path)
    parser.add_argument(
        "--dataset-counterfactual-fractions",
        type=float,
        nargs="+",
        help=(
            "For each solver state, also label human actions at these fractions "
            "of the configured allocator mix. Exo is supervised only at fraction 1."
        ),
    )
    parser.add_argument("--state-bank-output", type=Path)
    parser.add_argument("--state-bank-start-frame", type=int, default=1)
    parser.add_argument("--state-bank-end-phase", type=int)
    parser.add_argument("--state-bank-stride", type=int, default=1)
    parser.add_argument("--initial-state-bank", type=Path)
    parser.add_argument(
        "--initial-state-bank-index",
        type=int,
        help="Use one fixed full-state bank row for every comparison route.",
    )
    parser.add_argument(
        "--stop-dataset-route-on-done",
        action="store_true",
        help="End collection when the route supplying DAgger states first terminates.",
    )
    parser.add_argument(
        "--stop-on-any-done",
        action="store_true",
        help="End a synchronized comparison when any route first terminates.",
    )
    parser.add_argument(
        "--stop-on-route",
        choices=ROUTE_NAMES,
        help="End when this route first terminates, independent of dataset collection.",
    )
    parser.add_argument(
        "--video-route",
        choices=("all", *ROUTE_NAMES),
        default="all",
        help="Render all comparison routes or only one selected route.",
    )
    parser.add_argument(
        "--stop-forward-x",
        type=float,
        help="End when the dataset route reaches this global forward position.",
    )
    parser.add_argument(
        "--dataset-route",
        choices=("assisted_allocator", "muscle_allocator"),
        default="assisted_allocator",
    )
    parser.add_argument("--history-steps", type=int, default=8)
    parser.add_argument(
        "--exo-sensor-mode",
        choices=EXO_SENSOR_MODES,
        default="legacy16",
        help="Per-frame sensors available to the causal Exo student.",
    )
    parser.add_argument("--rollout-student", type=Path)
    parser.add_argument(
        "--rollout-target-pd",
        type=Path,
        help=(
            "Replace the rollout student's Exo network with a packaged "
            "shared-leg target-position PD policy. The conditioned human "
            "network is still loaded from --rollout-student."
        ),
    )
    parser.add_argument(
        "--student-exo-mode",
        choices=(
            "unified",
            "full_obs",
            "proprio_history",
            "coupled_proprio_history",
            "plan_conditioned",
        ),
        default="proprio_history",
    )
    parser.add_argument(
        "--student-execution-mix",
        type=float,
        default=1.0,
        help="Fraction of the student action executed on its DAgger route.",
    )
    args = parser.parse_args()
    if args.rollout_target_pd is not None and args.rollout_student is None:
        parser.error("--rollout-target-pd requires --rollout-student")
    if (
        args.rollout_target_pd is not None
        and args.student_exo_mode != "coupled_proprio_history"
    ):
        parser.error(
            "--rollout-target-pd requires --student-exo-mode "
            "coupled_proprio_history"
        )

    device = torch.device(args.device)
    config = copy.deepcopy(load_config(args.config))
    config["reset"]["episode_steps"] = int(args.steps) + 1
    config["reset"]["phase_indices"] = [int(args.phase)]
    config["reset"]["phase_windows"] = []
    config["reset"]["phase_index_jitter"] = 0
    config["reset"]["initial_activation"] = 0.0
    config["reset"]["initial_activation_range"] = []
    config["reset"]["full_state_only"] = args.initial_state_bank is not None
    config.setdefault("recovery_reset", {})["enabled"] = False
    if args.initial_state_bank is not None:
        config["offline_recovery_reset"] = {
            "enabled": True,
            "path": str(args.initial_state_bank.resolve()),
            "reset_probability": 1.0,
            "min_bank_size": 1,
        }
        if args.initial_state_bank_index is not None:
            if args.initial_state_bank_index < 0:
                raise ValueError("--initial-state-bank-index must be non-negative")
            config["offline_recovery_reset"]["fixed_index"] = int(
                args.initial_state_bank_index
            )
    elif args.initial_state_bank_index is not None:
        raise ValueError(
            "--initial-state-bank-index requires --initial-state-bank"
        )
    else:
        config.setdefault("offline_recovery_reset", {})["enabled"] = False
    if args.out_of_trajectory_threshold is not None:
        config.setdefault("myoassist_exact", {})["out_of_trajectory_threshold"] = float(
            args.out_of_trajectory_threshold
        )
    configure_direct_exo(config, args.exo_max_torque_nm, args.exo_max_delta)

    model, data = build_muscle_model(config)
    if int(model.na) != 22:
        raise ValueError(f"This analysis is intentionally limited to flat22; model.na={model.na}")
    probe_data = mujoco.MjData(model)
    reference = load_reference_from_config(
        args.reference, model, float(config["control"]["control_hz"]), device, config
    )
    runner = MJWarpMuscleRunner(
        model=model,
        data=data,
        config=config,
        reference=reference,
        nworld=len(ROUTE_NAMES),
        nconmax=128,
        njmax=512,
        seed=int(args.seed),
        device=device,
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    actor, normalizer, _ = build_sac_actor_for_checkpoint(
        checkpoint=checkpoint,
        model=model,
        config=config,
        obs_dim=runner.obs_dim,
        act_dim=runner.act_dim,
        device=device,
    )
    actor.eval()
    student_payload = None
    human_student = None
    exo_student = None
    proprio_mean = None
    proprio_std = None
    conditioned_human = None
    target_pd_model = None
    target_pd_payload = None
    target_pd_mean = None
    target_pd_std = None
    recurrent_plan_hidden = None
    recurrent_plan_steps = 0
    if args.rollout_student is not None:
        student_payload = torch.load(args.rollout_student, map_location=device)
        student_sensor_mode = str(
            student_payload.get("exo_sensor_mode", "legacy16")
        )
        if (
            args.student_exo_mode in {"proprio_history", "coupled_proprio_history"}
            and student_sensor_mode != str(args.exo_sensor_mode)
        ):
            raise ValueError(
                "student Exo sensor mode mismatch: "
                f"checkpoint={student_sensor_mode}, requested={args.exo_sensor_mode}"
            )
        human_student, _student_normalizer, _ = build_sac_actor_for_checkpoint(
            checkpoint=checkpoint,
            model=model,
            config=config,
            obs_dim=runner.obs_dim,
            act_dim=runner.act_dim,
            device=device,
        )
        human_student.load_state_dict(student_payload["human_actor_state_dict"], strict=True)
        if args.dataset_route == "muscle_allocator":
            exo_student = None
        elif args.student_exo_mode == "unified":
            exo_student = None
        elif args.student_exo_mode == "full_obs":
            exo_student = ExoStudent(
                int(student_payload["obs_dim"]), int(student_payload["hidden_dim"])
            ).to(device)
            exo_student.load_state_dict(student_payload["full_obs_exo_state_dict"])
        elif args.student_exo_mode == "plan_conditioned":
            if (
                student_payload.get("model_type")
                != "recurrent_exo_plan_conditioned_human"
            ):
                raise ValueError(
                    "plan_conditioned requires a recurrent-plan checkpoint"
                )
            recurrent_plan_steps = int(student_payload["plan_steps"])
            exo_student = RecurrentExoPlanStudent(
                int(student_payload["proprio_dim"]),
                int(student_payload["plan_hidden_dim"]),
                recurrent_plan_steps,
            ).to(device)
            exo_student.load_state_dict(
                student_payload["exo_plan_state_dict"]
            )
            proprio_mean = student_payload["proprio_mean"].to(device)
            proprio_std = student_payload["proprio_std"].to(device)
        else:
            exo_student = ExoStudent(
                int(student_payload["proprio_dim"]), int(student_payload["hidden_dim"])
            ).to(device)
            exo_student.load_state_dict(student_payload["proprio_exo_state_dict"])
            proprio_mean = student_payload["proprio_mean"].to(device)
            proprio_std = student_payload["proprio_std"].to(device)
        if args.student_exo_mode in {
            "coupled_proprio_history",
            "plan_conditioned",
        }:
            conditioned_human = ExoConditionedHumanStudent(
                int(student_payload["obs_dim"]),
                int(model.na),
                int(student_payload["hidden_dim"]),
                residual_indices=student_payload.get("human_residual_indices"),
                exo_context_dim=int(
                    student_payload.get("exo_context_dim", 2)
                ),
                zero_centered=bool(
                    student_payload.get("conditioned_zero_centered", False)
                ),
            ).to(device)
            conditioned_human.load_state_dict(
                student_payload["conditioned_human_state_dict"]
            )
            conditioned_human.eval()
        human_student.eval()
        if exo_student is not None:
            exo_student.eval()
        if args.rollout_target_pd is not None:
            target_pd_payload = torch.load(
                args.rollout_target_pd, map_location=device, weights_only=False
            )
            target_pd_mean = torch.as_tensor(
                target_pd_payload["input_mean"],
                dtype=torch.float32,
                device=device,
            )
            target_pd_std = torch.as_tensor(
                target_pd_payload["input_std"],
                dtype=torch.float32,
                device=device,
            )
            target_pd_model = SharedLegTargetPolicy(
                int(target_pd_mean.numel()),
                int(target_pd_payload["hidden_dim"]),
                float(target_pd_payload["target_offset_limit_rad"]),
            ).to(device)
            target_pd_model.load_state_dict(
                target_pd_payload["state_dict"], strict=True
            )
            target_pd_model.eval()
            if int(target_pd_payload["history_steps"]) != int(args.history_steps):
                raise ValueError("target-PD history length does not match evaluator")

    joint_dofs = np.asarray(
        [
            int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
            for name in JOINTS
        ],
        dtype=np.int64,
    )
    muscle_count = int(model.na)
    actuator_names = [
        csv_name(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
            or f"muscle_{index}"
        )
        for index in range(muscle_count)
    ]
    hip_pair_indices = np.asarray(
        [
            actuator_names.index(name)
            for name in (
                "glutmax_r",
                "iliopsoas_r",
                "glutmax_l",
                "iliopsoas_l",
            )
        ],
        dtype=np.int64,
    )
    mapping = muscle_action_mapping_mode(config)
    action_limits = torch.tensor(
        [[-1.0] * muscle_count, [1.0] * muscle_count], device=device
    )
    ctrl_limits = policy_action_to_ctrl(
        action_limits,
        runner.ctrl_low[:muscle_count],
        runner.ctrl_high[:muscle_count],
        muscle_count=muscle_count,
        muscle_mapping=mapping,
    ).detach().cpu().numpy()
    excitation_low, excitation_high = ctrl_limits[0], ctrl_limits[1]
    proprio_qpos, proprio_qvel = proprio_indices(model, args.exo_sensor_mode)
    dataset_world = 1 if args.dataset_route == "muscle_allocator" else 2
    assisted_history: deque[np.ndarray] = deque(maxlen=max(1, int(args.history_steps)))
    dataset_obs: list[np.ndarray] = []
    dataset_proprio: list[np.ndarray] = []
    dataset_muscle_action: list[np.ndarray] = []
    dataset_exo_action: list[np.ndarray] = []
    dataset_solver_success: list[bool] = []
    dataset_torque_error: list[float] = []
    dataset_exo_supervision_mask: list[bool] = []
    dataset_counterfactual_fraction: list[float] = []
    dataset_phase: list[int] = []
    bank_qpos: list[np.ndarray] = []
    bank_qvel: list[np.ndarray] = []
    bank_act: list[np.ndarray] = []
    bank_ctrl: list[np.ndarray] = []
    bank_prev_activation: list[np.ndarray] = []
    bank_qacc_warmstart: list[np.ndarray] = []
    bank_site_xpos: list[np.ndarray] = []
    bank_phase: list[int] = []
    bank_x_align_mask: list[bool] = []

    pelvis_tx_qpos = int(
        reference.get("pelvis_tx_qpos", semantic_qpos_index(model, "pelvis_tx"))
    )
    tracked_qpos = reference["qpos_indices"].detach().cpu().numpy().astype(np.int64)
    angular_track = np.arange(1, len(TRACK_JOINTS), dtype=np.int64)
    pelvis_tilt_track = TRACK_JOINTS.index("pelvis_tilt")
    origin_x = [float(runner.qpos[index, pelvis_tx_qpos].item()) for index in range(3)]
    initial_effective_phase = int(runner.target_phase_idx()[0].item()) % int(
        reference["length"]
    )
    origin_reference_x = float(reference["pelvis_tx_ref"][initial_effective_phase].item())

    render_assets = (
        build_render_assets(model, int(args.height), int(args.width)) if args.video else None
    )
    frames: list[np.ndarray] = []
    rows: list[dict[str, float | int | bool]] = []
    alive = [True] * len(ROUTE_NAMES)
    obs = runner.obs()

    try:
        for frame in range(int(args.steps)):
            if render_assets is not None:
                render_data, renderers, ghost_models, ghost_data, ghost_renderers, cameras = (
                    render_assets
                )
                images = []
                render_worlds = (
                    range(len(ROUTE_NAMES))
                    if args.video_route == "all"
                    else (ROUTE_NAMES.index(args.video_route),)
                )
                for world in render_worlds:
                    label = ROUTE_NAMES[world]
                    image = render_world(
                        runner=runner,
                        world=world,
                        model=model,
                        data=render_data[world],
                        renderer=renderers[world],
                        ghost_model=ghost_models[world],
                        ghost_data=ghost_data[world],
                        ghost_renderer=ghost_renderers[world],
                        camera=cameras[world],
                        reference=reference,
                        config=config,
                        origin_x=origin_x[world],
                        origin_reference_x=origin_reference_x,
                    )
                    images.append(add_label(image, label))
                frames.append(np.concatenate(images, axis=1))

            normalized_obs = normalizer.normalize(obs)
            bank_stride = max(1, int(args.state_bank_stride))
            current_bank_phase = int(runner.phase_idx[dataset_world].item())
            save_bank_state = (
                args.state_bank_output is not None
                and frame >= int(args.state_bank_start_frame)
                and (frame - int(args.state_bank_start_frame)) % bank_stride == 0
                and (
                    args.state_bank_end_phase is None
                    or current_bank_phase <= int(args.state_bank_end_phase)
                )
                and alive[dataset_world]
            )
            if save_bank_state:
                bank_qpos.append(
                    runner.qpos[dataset_world].detach().cpu().numpy().astype(
                        np.float32, copy=True
                    )
                )
                bank_qvel.append(
                    runner.qvel[dataset_world].detach().cpu().numpy().astype(
                        np.float32, copy=True
                    )
                )
                bank_act.append(
                    runner.act[dataset_world].detach().cpu().numpy().astype(
                        np.float32, copy=True
                    )
                )
                bank_ctrl.append(
                    runner.ctrl[dataset_world].detach().cpu().numpy().astype(
                        np.float32, copy=True
                    )
                )
                bank_prev_activation.append(
                    runner.prev_activation[dataset_world]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=True)
                )
                bank_qacc_warmstart.append(
                    runner.qacc_warmstart[dataset_world]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=True)
                )
                bank_site_xpos.append(
                    runner.site_xpos[dataset_world].detach().cpu().numpy().astype(
                        np.float32, copy=True
                    )
                )
                bank_phase.append(current_bank_phase)
                bank_x_align_mask.append(
                    bool(runner.x_align_mask[dataset_world].item())
                )
            assisted_proprio = proprio_frame(
                runner.qpos[dataset_world].detach().cpu().numpy(),
                runner.qvel[dataset_world].detach().cpu().numpy(),
                runner.applied_exo_ctrl[dataset_world].detach().cpu().numpy(),
                proprio_qpos,
                proprio_qvel,
                args.exo_sensor_mode,
            )
            assisted_proprio_history = append_history(
                assisted_history, assisted_proprio, int(args.history_steps)
            )
            action, _, _, _ = actor.get_action_and_value(
                normalized_obs, deterministic=True
            )
            action = torch.clamp(action, -1.0, 1.0)
            policy_ctrl = policy_action_to_ctrl(
                action,
                runner.ctrl_low,
                runner.ctrl_high,
                muscle_count=muscle_count,
                muscle_mapping=mapping,
            )
            modified_action = action.clone()
            modified_action[:, muscle_count : muscle_count + 2] = 0.0

            torque_maps: list[np.ndarray] = []
            exo_maps: list[np.ndarray] = []
            target_torques: list[np.ndarray] = []
            allocated_targets: list[np.ndarray] = []
            allocated_exos: list[np.ndarray] = []
            teacher_targets: list[np.ndarray] = []
            full_allocated_targets: list[np.ndarray] = []
            full_allocated_exos: list[np.ndarray] = []
            solver_success = [True, True, True]
            torque_errors = [0.0, 0.0, 0.0]

            for world in range(3):
                current = runner.act[world].detach().cpu().numpy().astype(
                    np.float64, copy=True
                )
                qpos = runner.qpos[world].detach().cpu().numpy().astype(
                    np.float64, copy=True
                )
                qvel = runner.qvel[world].detach().cpu().numpy().astype(
                    np.float64, copy=True
                )
                torque_map = active_torque_map(
                    model, probe_data, qpos, qvel, joint_dofs
                )
                current_exo_map = exo_torque_map(
                    model, probe_data, qpos, qvel, joint_dofs
                )
                teacher_excitation = (
                    policy_ctrl[world, :muscle_count]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
                teacher_next = evolve_activation(
                    model, current, teacher_excitation, runner.frame_skip
                )
                target_torque = torque_map @ teacher_next
                allocated_next = teacher_next.copy()
                allocated_exo = np.zeros(current_exo_map.shape[1], dtype=np.float64)
                full_allocated_next = teacher_next.copy()
                full_allocated_exo = allocated_exo.copy()

                if world > 0:
                    lower = evolve_activation(
                        model, current, excitation_low, runner.frame_skip
                    )
                    upper = evolve_activation(
                        model, current, excitation_high, runner.frame_skip
                    )
                    if world == 1:
                        if args.muscle_allocation_mode == "hip_pair":
                            allocated_next, _noexo, success, _ = (
                                allocate_hip_pair_replacement(
                                    torque_map,
                                    current_exo_map[:, :0],
                                    target_torque,
                                    teacher_next,
                                    np.empty(0, dtype=np.float64),
                                    lower,
                                    upper,
                                    np.empty(0, dtype=np.float64),
                                    np.empty(0, dtype=np.float64),
                                    hip_pair_indices,
                                )
                            )
                        else:
                            allocated_next, success, _ = allocate_activation(
                                torque_map, target_torque, teacher_next, lower, upper
                            )
                    else:
                        previous_exo = (
                            runner.applied_exo_ctrl[world]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float64, copy=True)
                        )
                        exo_low = np.full(
                            current_exo_map.shape[1], -runner.exo_policy_max_ctrl
                        )
                        exo_high = np.full(
                            current_exo_map.shape[1], runner.exo_policy_max_ctrl
                        )
                        if runner.exo_policy_max_delta > 0.0:
                            exo_low = np.maximum(
                                exo_low, previous_exo - runner.exo_policy_max_delta
                            )
                            exo_high = np.minimum(
                                exo_high, previous_exo + runner.exo_policy_max_delta
                            )
                        if args.assisted_allocation_mode == "hip_pair_impulse":
                            allocated_next, allocated_exo, success, full_error = (
                                allocate_hip_pair_impulse_replacement(
                                    model,
                                    torque_map,
                                    current_exo_map,
                                    current,
                                    teacher_excitation,
                                    excitation_low,
                                    excitation_high,
                                    exo_low,
                                    exo_high,
                                    hip_pair_indices,
                                    runner.frame_skip,
                                    prevent_exo_opposition=bool(
                                        args.prevent_exo_opposition
                                    ),
                                )
                            )
                        elif args.assisted_allocation_mode == "hip_pair":
                            allocated_next, allocated_exo, success, _ = (
                                allocate_hip_pair_replacement(
                                    torque_map,
                                    current_exo_map,
                                    target_torque,
                                    teacher_next,
                                    previous_exo,
                                    lower,
                                    upper,
                                    exo_low,
                                    exo_high,
                                    hip_pair_indices,
                                    prevent_exo_opposition=bool(
                                        args.prevent_exo_opposition
                                    ),
                                )
                            )
                        else:
                            allocated_next, allocated_exo, success, _ = (
                                allocate_activation_and_exo(
                                    torque_map,
                                    current_exo_map,
                                    target_torque,
                                    teacher_next,
                                    previous_exo,
                                    lower,
                                    upper,
                                    exo_low,
                                    exo_high,
                                    max(0.0, float(args.exo_l2_weight)),
                                    max(0.0, float(args.exo_delta_weight)),
                                    previous_exo,
                                    prevent_exo_opposition=bool(
                                        args.prevent_exo_opposition
                                    ),
                                )
                            )
                    if (
                        world == 1
                        or args.assisted_allocation_mode != "hip_pair_impulse"
                    ):
                        full_error = float(
                            np.linalg.norm(
                                torque_map @ allocated_next
                                + current_exo_map @ allocated_exo
                                - target_torque
                            )
                        )
                    if not success or full_error > 1.0e-3:
                        allocated_next = teacher_next.copy()
                        allocated_exo[:] = 0.0
                    else:
                        full_allocated_next = allocated_next.copy()
                        full_allocated_exo = allocated_exo.copy()
                        allocator_mix = float(
                            np.clip(args.allocator_execution_mix, 0.0, 1.0)
                        )
                        allocated_next = teacher_next + allocator_mix * (
                            allocated_next - teacher_next
                        )
                        allocated_exo = allocator_mix * allocated_exo
                        allocated_excitation = excitation_for_target_activation(
                            model,
                            current,
                            allocated_next,
                            runner.frame_skip,
                            excitation_low,
                            excitation_high,
                        )
                        modified_action[world, :muscle_count] = torch.tensor(
                            excitation_to_action(allocated_excitation, mapping),
                            dtype=action.dtype,
                            device=device,
                        )
                        if world == 2:
                            modified_action[
                                world, muscle_count : muscle_count + allocated_exo.size
                            ] = torch.tensor(
                                allocated_exo
                                / max(float(runner.exo_policy_max_ctrl), 1.0e-8),
                                dtype=action.dtype,
                                device=device,
                            )
                    solver_success[world] = bool(success)
                    torque_errors[world] = full_error

                torque_maps.append(torque_map)
                exo_maps.append(current_exo_map)
                target_torques.append(target_torque)
                allocated_targets.append(allocated_next)
                allocated_exos.append(allocated_exo)
                teacher_targets.append(teacher_next)
                full_allocated_targets.append(full_allocated_next)
                full_allocated_exos.append(full_allocated_exo)

            if args.dataset_output is not None:
                fractions = args.dataset_counterfactual_fractions or [1.0]
                for fraction_value in fractions:
                    fraction = float(np.clip(fraction_value, 0.0, 1.0))
                    if args.dataset_counterfactual_fractions is None:
                        labelled_action = modified_action[dataset_world].clone()
                    else:
                        effective_mix = float(
                            np.clip(args.allocator_execution_mix, 0.0, 1.0)
                        ) * fraction
                        target_activation = teacher_targets[dataset_world] + effective_mix * (
                            full_allocated_targets[dataset_world]
                            - teacher_targets[dataset_world]
                        )
                        target_excitation = excitation_for_target_activation(
                            model,
                            runner.act[dataset_world]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float64, copy=True),
                            target_activation,
                            runner.frame_skip,
                            excitation_low,
                            excitation_high,
                        )
                        labelled_action = action[dataset_world].clone()
                        labelled_action[:muscle_count] = torch.tensor(
                            excitation_to_action(target_excitation, mapping),
                            dtype=action.dtype,
                            device=device,
                        )
                        labelled_action[muscle_count : muscle_count + 2] = torch.tensor(
                            effective_mix
                            * full_allocated_exos[dataset_world]
                            / max(float(runner.exo_policy_max_ctrl), 1.0e-8),
                            dtype=action.dtype,
                            device=device,
                        )
                    dataset_obs.append(
                        normalized_obs[dataset_world]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32, copy=True)
                    )
                    dataset_proprio.append(assisted_proprio_history.copy())
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
                    dataset_solver_success.append(bool(solver_success[dataset_world]))
                    dataset_torque_error.append(
                        float(torque_errors[dataset_world])
                    )
                    dataset_exo_supervision_mask.append(
                        args.dataset_counterfactual_fractions is None
                        or abs(fraction - 1.0) < 1.0e-8
                    )
                    dataset_counterfactual_fraction.append(fraction)
                    dataset_phase.append(
                        int(runner.phase_idx[dataset_world].item())
                    )
            if human_student is not None:
                teacher_route_action = modified_action[dataset_world].clone()
                student_action, _, _, _ = human_student.get_action_and_value(
                    normalized_obs[dataset_world : dataset_world + 1], deterministic=True
                )
                if args.dataset_route == "muscle_allocator":
                    student_muscle_action = student_action[0, :muscle_count]
                    if conditioned_human is not None:
                        student_muscle_action = conditioned_human(
                            normalized_obs[dataset_world : dataset_world + 1],
                            student_action[:, :muscle_count],
                            torch.zeros(
                                (1, 2),
                                dtype=student_action.dtype,
                                device=device,
                            ),
                        )[0]
                    modified_action[
                        dataset_world, :muscle_count
                    ] = student_muscle_action
                    modified_action[dataset_world, muscle_count : muscle_count + 2] = 0.0
                elif args.student_exo_mode == "unified":
                    modified_action[dataset_world, :muscle_count] = student_action[
                        0, :muscle_count
                    ]
                    modified_action[
                        dataset_world, muscle_count : muscle_count + 2
                    ] = student_action[0, muscle_count : muscle_count + 2]
                elif args.student_exo_mode == "full_obs":
                    modified_action[dataset_world, :muscle_count] = student_action[
                        0, :muscle_count
                    ]
                    student_exo_action = exo_student(
                        normalized_obs[dataset_world : dataset_world + 1]
                    )[0]
                    modified_action[
                        dataset_world, muscle_count : muscle_count + 2
                    ] = student_exo_action
                elif args.student_exo_mode == "plan_conditioned":
                    proprio_tensor = torch.from_numpy(
                        assisted_proprio
                    ).to(device)
                    normalized_proprio = (
                        proprio_tensor - proprio_mean
                    ) / proprio_std
                    student_plan, recurrent_plan_hidden = exo_student.step(
                        normalized_proprio.unsqueeze(0),
                        recurrent_plan_hidden,
                    )
                    student_exo_action = student_plan[0, 0]
                    command_history = torch.from_numpy(
                        assisted_proprio_history.reshape(
                            int(args.history_steps), 6
                        )[:, -2:].reshape(1, int(args.history_steps) * 2)
                    ).to(device)
                    exo_context = torch.cat(
                        (command_history, student_plan.flatten(1)), dim=-1
                    )
                    student_muscle_action = conditioned_human(
                        normalized_obs[dataset_world : dataset_world + 1],
                        student_action[:, :muscle_count],
                        exo_context,
                    )[0]
                    modified_action[
                        dataset_world, :muscle_count
                    ] = student_muscle_action
                    modified_action[
                        dataset_world, muscle_count : muscle_count + 2
                    ] = student_exo_action
                else:
                    proprio_tensor = torch.from_numpy(assisted_proprio_history).to(device)
                    if target_pd_model is None:
                        normalized_proprio = (
                            proprio_tensor - proprio_mean
                        ) / proprio_std
                        student_exo_action = exo_student(
                            normalized_proprio.unsqueeze(0)
                        )[0]
                    else:
                        history = proprio_tensor.reshape(
                            int(args.history_steps), 6
                        )
                        right_features = history.reshape(-1)
                        left_features = history[:, [1, 0, 3, 2, 5, 4]].reshape(-1)
                        pd_features = torch.stack(
                            (right_features, left_features), dim=0
                        )
                        pd_features = (pd_features - target_pd_mean) / target_pd_std
                        offsets = target_pd_model(pd_features)
                        hip_velocity = history[-1, 2:4]
                        torque_nm = torch.clamp(
                            float(target_pd_payload["kp_nm_per_rad"]) * offsets
                            + float(target_pd_payload["kd_nm_s_per_rad"])
                            * hip_velocity,
                            -float(target_pd_payload.get("torque_limit_nm", 10.0)),
                            float(target_pd_payload.get("torque_limit_nm", 10.0)),
                        )
                        student_exo_action = torque_nm / float(
                            target_pd_payload.get("torque_scale_nm", 10.0)
                        )
                    if args.student_exo_mode == "coupled_proprio_history":
                        student_muscle_action = conditioned_human(
                            normalized_obs[dataset_world : dataset_world + 1],
                            student_action[:, :muscle_count],
                            student_exo_action.unsqueeze(0),
                        )[0]
                    else:
                        student_muscle_action = student_action[0, :muscle_count]
                    modified_action[
                        dataset_world, :muscle_count
                    ] = student_muscle_action
                    modified_action[
                        dataset_world, muscle_count : muscle_count + 2
                    ] = student_exo_action
                student_mix = max(0.0, min(1.0, float(args.student_execution_mix)))
                modified_action[dataset_world] = torch.lerp(
                    teacher_route_action,
                    modified_action[dataset_world],
                    student_mix,
                )

            obs, _reward, done, _terms = runner.step(modified_action)
            actual_activations = [
                runner.last_step_activation[world]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
                for world in range(3)
            ]
            actual_exo_controls = (
                runner.applied_exo_ctrl.detach().cpu().numpy().astype(np.float64)
            )
            for world in range(3):
                alive[world] = alive[world] and not bool(done[world].item())

            row: dict[str, float | int | bool] = {
                "frame": frame,
                "phase": int(runner.phase_idx[0].item()),
            }
            for world, name in enumerate(ROUTE_NAMES):
                activation = actual_activations[world]
                muscle_torque = torque_maps[world] @ activation
                exo_control = actual_exo_controls[world]
                exo_torque = exo_maps[world] @ exo_control
                muscle_contribution = torque_maps[world] * activation[None, :]
                joint_cocontraction = np.minimum(
                    np.maximum(muscle_contribution, 0.0).sum(axis=1),
                    np.maximum(-muscle_contribution, 0.0).sum(axis=1),
                )
                hip_mask = np.any(
                    np.abs(torque_maps[world][[0, 3], :]) > 1.0e-8, axis=0
                )
                opposed_nm = np.where(
                    muscle_torque[[0, 3]] * exo_torque[[0, 3]] < 0.0,
                    np.minimum(
                        np.abs(muscle_torque[[0, 3]]),
                        np.abs(exo_torque[[0, 3]]),
                    ),
                    0.0,
                )
                qvel = runner.qvel[world].detach().cpu().numpy()
                exo_power = exo_torque[[0, 3]] * qvel[joint_dofs[[0, 3]]]
                effective_phase = int(
                    runner.target_phase_idx()[world].item()
                ) % int(reference["length"])
                actual_pose = (
                    runner.qpos[world, tracked_qpos]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
                reference_pose = (
                    reference["q_ref"][effective_phase]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
                angle_error = np.arctan2(
                    np.sin(actual_pose[angular_track] - reference_pose[angular_track]),
                    np.cos(actual_pose[angular_track] - reference_pose[angular_track]),
                )
                policy_pose = (
                    runner.qpos[0, tracked_qpos]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
                policy_angle_delta = np.arctan2(
                    np.sin(actual_pose[angular_track] - policy_pose[angular_track]),
                    np.cos(actual_pose[angular_track] - policy_pose[angular_track]),
                )
                row.update(
                    {
                        f"{name}_alive": alive[world],
                        f"{name}_low_height_done": float(
                            _terms["low_height_done"][world].item()
                        ),
                        f"{name}_out_of_trajectory_done": float(
                            _terms["out_of_trajectory_done"][world].item()
                        ),
                        f"{name}_nonfinite_done": float(
                            _terms["nonfinite_done"][world].item()
                        ),
                        f"{name}_truncated_done": float(
                            _terms["truncated_done"][world].item()
                        ),
                        f"{name}_activation_l2": float(np.mean(np.square(activation))),
                        f"{name}_hip_activation_l2": float(
                            np.mean(np.square(activation[hip_mask]))
                        ),
                        f"{name}_cocontraction_nm": cocontraction_nm(
                            torque_maps[world], activation
                        ),
                        f"{name}_pelvis_tx": float(
                            runner.qpos[world, pelvis_tx_qpos].item()
                        ),
                        f"{name}_solver_success": solver_success[world],
                        f"{name}_target_torque_error_nm": torque_errors[world],
                        f"{name}_exo_ctrl_abs_mean": float(
                            np.mean(np.abs(exo_control))
                        ),
                        f"{name}_exo_hip_torque_abs_mean_nm": float(
                            np.mean(np.abs(exo_torque[[0, 3]]))
                        ),
                        f"{name}_human_exo_opposed_torque_nm": float(
                            np.mean(opposed_nm)
                        ),
                        f"{name}_exo_positive_power_w": float(
                            np.maximum(exo_power, 0.0).sum()
                        ),
                        f"{name}_exo_negative_power_w": float(
                            np.minimum(exo_power, 0.0).sum()
                        ),
                        f"{name}_exo_hip_r_torque_nm": float(exo_torque[0]),
                        f"{name}_exo_hip_l_torque_nm": float(exo_torque[3]),
                        f"{name}_muscle_hip_r_torque_nm": float(muscle_torque[0]),
                        f"{name}_muscle_hip_l_torque_nm": float(muscle_torque[3]),
                        f"{name}_pelvis_ty": float(actual_pose[0]),
                        f"{name}_ref_pelvis_ty": float(reference_pose[0]),
                        f"{name}_pose_rmse_deg": float(
                            np.rad2deg(np.sqrt(np.mean(np.square(angle_error))))
                        ),
                        f"{name}_pelvis_tilt_error_deg": float(
                            np.rad2deg(
                                np.arctan2(
                                    np.sin(
                                        actual_pose[pelvis_tilt_track]
                                        - reference_pose[pelvis_tilt_track]
                                    ),
                                    np.cos(
                                        actual_pose[pelvis_tilt_track]
                                        - reference_pose[pelvis_tilt_track]
                                    ),
                                )
                            )
                        ),
                        f"{name}_pose_delta_vs_policy_deg": float(
                            np.rad2deg(
                                np.sqrt(np.mean(np.square(policy_angle_delta)))
                            )
                        ),
                    }
                )
                for local_index in angular_track:
                    joint_name = TRACK_JOINTS[int(local_index)]
                    row[f"{name}_{joint_name}_deg"] = float(
                        np.rad2deg(actual_pose[int(local_index)])
                    )
                    row[f"{name}_ref_{joint_name}_deg"] = float(
                        np.rad2deg(reference_pose[int(local_index)])
                    )
                    row[f"{name}_{joint_name}_error_deg"] = float(
                        np.rad2deg(angle_error[int(local_index) - 1])
                    )
                    row[f"{name}_{joint_name}_delta_vs_policy_deg"] = float(
                        np.rad2deg(policy_angle_delta[int(local_index) - 1])
                    )
                for joint_index, joint_name in enumerate(JOINTS):
                    row[f"{name}_{joint_name}_cocontraction_nm"] = float(
                        joint_cocontraction[joint_index]
                    )
                for muscle_index, muscle_name in enumerate(actuator_names):
                    row[f"{name}_activation_{muscle_name}"] = float(
                        activation[muscle_index]
                    )
                if runner.sensordata is not None:
                    foot_force = (
                        runner.sensordata[world, :4]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float64)
                    )
                    row[f"{name}_right_foot_force_n"] = float(
                        foot_force[0] + foot_force[1]
                    )
                    row[f"{name}_left_foot_force_n"] = float(
                        foot_force[2] + foot_force[3]
                    )
                else:
                    row[f"{name}_right_foot_force_n"] = 0.0
                    row[f"{name}_left_foot_force_n"] = 0.0
                reference_contact = (
                    reference["foot_contact_ref"][effective_phase]
                    .detach()
                    .cpu()
                    .numpy()
                )
                row[f"{name}_ref_right_contact"] = bool(
                    reference_contact[0] or reference_contact[1]
                )
                row[f"{name}_ref_left_contact"] = bool(
                    reference_contact[2] or reference_contact[3]
                )
            rows.append(row)
            if args.stop_on_any_done and bool(done.any().item()):
                break
            if (
                args.stop_on_route is not None
                and not alive[ROUTE_NAMES.index(args.stop_on_route)]
            ):
                break
            if (
                args.stop_forward_x is not None
                and alive[dataset_world]
                and float(runner.qpos[dataset_world, pelvis_tx_qpos].item())
                >= float(args.stop_forward_x)
            ):
                break
            if (
                args.stop_dataset_route_on_done
                and args.dataset_output is not None
                and not alive[dataset_world]
            ):
                break
            if not any(alive):
                break
    finally:
        close_render_assets(render_assets)

    if not rows:
        raise RuntimeError("rollout produced no rows")
    args.outdir.mkdir(parents=True, exist_ok=True)
    with (args.outdir / "timeseries.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if frames:
        video_name = (
            "policy_vs_noexo_vs_assisted.mp4"
            if args.video_route == "all"
            else f"{args.video_route}.mp4"
        )
        imageio.mimwrite(
            args.outdir / video_name,
            frames,
            fps=int(config["control"]["control_hz"]),
            quality=8,
        )
    if args.dataset_output is not None:
        args.dataset_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.dataset_output,
            normalized_obs=np.stack(dataset_obs),
            proprio_history=np.stack(dataset_proprio),
            muscle_action=np.stack(dataset_muscle_action),
            exo_action=np.stack(dataset_exo_action),
            solver_success=np.asarray(dataset_solver_success, dtype=np.bool_),
            torque_error=np.asarray(dataset_torque_error, dtype=np.float32),
            exo_supervision_mask=np.asarray(
                dataset_exo_supervision_mask, dtype=np.bool_
            ),
            counterfactual_fraction=np.asarray(
                dataset_counterfactual_fraction, dtype=np.float32
            ),
            phase=np.asarray(dataset_phase, dtype=np.int64),
            history_steps=np.asarray([int(args.history_steps)], dtype=np.int64),
            exo_sensor_mode=np.asarray([str(args.exo_sensor_mode)]),
            dataset_route=np.asarray([str(args.dataset_route)]),
        )
    if args.state_bank_output is not None:
        if not bank_qpos:
            raise RuntimeError("state bank collection produced no states")
        args.state_bank_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.state_bank_output,
            qpos=np.stack(bank_qpos),
            qvel=np.stack(bank_qvel),
            act=np.stack(bank_act),
            ctrl=np.stack(bank_ctrl),
            prev_activation=np.stack(bank_prev_activation),
            qacc_warmstart=np.stack(bank_qacc_warmstart),
            site_xpos=np.stack(bank_site_xpos),
            phase=np.asarray(bank_phase, dtype=np.int64),
            x_align_mask=np.asarray(bank_x_align_mask, dtype=np.bool_),
            metadata=np.asarray(
                {
                    "source": str(args.checkpoint),
                    "route": str(args.dataset_route),
                    "phase_start": int(args.phase),
                    "steps": int(args.steps),
                    "state_bank_start_frame": int(args.state_bank_start_frame),
                    "state_bank_end_phase": args.state_bank_end_phase,
                    "state_bank_stride": int(args.state_bank_stride),
                    "allocator_execution_mix": float(
                        args.allocator_execution_mix
                    ),
                },
                dtype=object,
            ),
        )

    steady_start = min(max(0, int(args.steady_start)), len(rows) - 1)
    steady = rows[steady_start:]
    summary: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "global_step": int(checkpoint.get("global_step", 0)),
        "phase": int(args.phase),
        "frames": len(rows),
        "steady_start": steady_start,
        "exo_max_torque_nm": float(args.exo_max_torque_nm),
        "exo_max_delta": float(args.exo_max_delta),
        "exo_l2_weight": float(args.exo_l2_weight),
        "exo_delta_weight": float(args.exo_delta_weight),
        "allocator_execution_mix": float(args.allocator_execution_mix),
        "assisted_allocation_mode": str(args.assisted_allocation_mode),
        "muscle_allocation_mode": str(args.muscle_allocation_mode),
        "prevent_exo_opposition": bool(args.prevent_exo_opposition),
        "video_route": str(args.video_route),
    }
    for name in ROUTE_NAMES:
        route_rows = [
            row for row in steady if bool(row[f"{name}_alive"])
        ]
        if not route_rows:
            route_rows = steady
        last_alive_row = next(
            (
                row
                for row in reversed(rows)
                if bool(row[f"{name}_alive"])
            ),
            rows[-1],
        )
        summary[name] = {
            "survival_frames": next(
                (index + 1 for index, row in enumerate(rows) if not row[f"{name}_alive"]),
                len(rows),
            ),
            "forward_distance_m": float(last_alive_row[f"{name}_pelvis_tx"]) - origin_x[
                ROUTE_NAMES.index(name)
            ],
            "activation_l2": summarize(
                [float(row[f"{name}_activation_l2"]) for row in route_rows]
            ),
            "hip_activation_l2": summarize(
                [float(row[f"{name}_hip_activation_l2"]) for row in route_rows]
            ),
            "cocontraction_nm": summarize(
                [float(row[f"{name}_cocontraction_nm"]) for row in route_rows]
            ),
            "exo_hip_torque_abs_mean_nm": summarize(
                [float(row[f"{name}_exo_hip_torque_abs_mean_nm"]) for row in route_rows]
            ),
            "human_exo_opposed_torque_nm": summarize(
                [float(row[f"{name}_human_exo_opposed_torque_nm"]) for row in route_rows]
            ),
            "exo_positive_power_w": summarize(
                [float(row[f"{name}_exo_positive_power_w"]) for row in route_rows]
            ),
            "exo_negative_power_w": summarize(
                [float(row[f"{name}_exo_negative_power_w"]) for row in route_rows]
            ),
            "pose_rmse_deg": summarize(
                [float(row[f"{name}_pose_rmse_deg"]) for row in route_rows]
            ),
            "pelvis_tilt_error_deg": summarize(
                [
                    abs(float(row[f"{name}_pelvis_tilt_error_deg"]))
                    for row in route_rows
                ]
            ),
            "pose_delta_vs_policy_deg": summarize(
                [
                    float(row[f"{name}_pose_delta_vs_policy_deg"])
                    for row in route_rows
                ]
            ),
            "solver_success_rate": float(
                np.mean([bool(row[f"{name}_solver_success"]) for row in route_rows])
            ),
            "target_torque_error_nm": summarize(
                [float(row[f"{name}_target_torque_error_nm"]) for row in route_rows]
            ),
        }

    policy_l2 = float(summary["policy"]["activation_l2"]["mean"])
    noexo_l2 = float(summary["muscle_allocator"]["activation_l2"]["mean"])
    assisted_l2 = float(summary["assisted_allocator"]["activation_l2"]["mean"])
    policy_hip_l2 = float(summary["policy"]["hip_activation_l2"]["mean"])
    noexo_hip_l2 = float(summary["muscle_allocator"]["hip_activation_l2"]["mean"])
    assisted_hip_l2 = float(summary["assisted_allocator"]["hip_activation_l2"]["mean"])
    summary["comparisons"] = {
        "muscle_reallocation_total_reduction_vs_policy": 1.0
        - noexo_l2 / max(policy_l2, 1.0e-12),
        "assisted_total_reduction_vs_policy": 1.0
        - assisted_l2 / max(policy_l2, 1.0e-12),
        "exo_incremental_total_reduction_vs_reallocated_noexo": 1.0
        - assisted_l2 / max(noexo_l2, 1.0e-12),
        "muscle_reallocation_hip_reduction_vs_policy": 1.0
        - noexo_hip_l2 / max(policy_hip_l2, 1.0e-12),
        "assisted_hip_reduction_vs_policy": 1.0
        - assisted_hip_l2 / max(policy_hip_l2, 1.0e-12),
        "exo_incremental_hip_reduction_vs_reallocated_noexo": 1.0
        - assisted_hip_l2 / max(noexo_hip_l2, 1.0e-12),
    }
    (args.outdir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
