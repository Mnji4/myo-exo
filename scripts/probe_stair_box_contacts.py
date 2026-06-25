#!/usr/bin/env python3
"""Small diagnostics for MuJoCo stair-box contact behavior."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cleanrl.ppo_muscle_mjwarp import (  # noqa: E402
    DEFAULT_REFERENCE_PATH,
    FOOT_SITE_NAMES,
    apply_joint_equalities_np,
    build_muscle_model,
    course_height_np,
    cpu_policy_obs,
    joint_id,
    load_config,
    load_reference_from_config,
    muscle_action_to_activation,
    set_cpu_reference_state,
)
from cleanrl.sac_muscle_mjwarp import (  # noqa: E402
    SACActor,
    SymmetricSACActor,
    build_sagittal_mirror_spec,
)


def geom_name(model: mujoco.MjModel, geom_id: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id))
    return str(name) if name is not None else f"geom_{int(geom_id)}"


def simple_box_probe() -> None:
    print("== standalone MuJoCo box probe ==")
    top_xml = """
<mujoco>
  <option timestep="0.001" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="ground" type="plane" pos="0 0 -1" size="5 5 0.1"/>
    <geom name="stair" type="box" pos="0 0 0.1" size="0.2 0.25 0.1"/>
    <body name="probe" pos="0 0 0.55">
      <freejoint/>
      <geom name="probe_sphere" type="sphere" size="0.04" density="1000"/>
    </body>
  </worldbody>
</mujoco>
"""
    model = mujoco.MjModel.from_xml_string(top_xml)
    data = mujoco.MjData(model)
    min_dist = 1.0
    stair_contacts = 0
    for _ in range(900):
        mujoco.mj_step(model, data)
        for idx in range(int(data.ncon)):
            c = data.contact[idx]
            names = {geom_name(model, c.geom1), geom_name(model, c.geom2)}
            if "stair" in names:
                stair_contacts += 1
                min_dist = min(min_dist, float(c.dist))
    print(
        "top_drop:",
        f"final_sphere_z={data.qpos[2]:.4f}",
        f"expected_about={0.1 + 0.1 + 0.04:.4f}",
        f"stair_contacts={stair_contacts}",
        f"min_dist={min_dist:.5f}",
    )

    side_xml = """
<mujoco>
  <option timestep="0.001" gravity="0 0 0"/>
  <worldbody>
    <geom name="stair" type="box" pos="0 0 0.1" size="0.2 0.25 0.1"/>
    <body name="probe" pos="-0.45 0 0.10">
      <freejoint/>
      <geom name="probe_sphere" type="sphere" size="0.04" density="1000"/>
    </body>
  </worldbody>
</mujoco>
"""
    model = mujoco.MjModel.from_xml_string(side_xml)
    data = mujoco.MjData(model)
    data.qvel[0] = 1.0
    first_contact: tuple[int, float, np.ndarray] | None = None
    min_x = float(data.qpos[0])
    for step in range(700):
        mujoco.mj_step(model, data)
        min_x = min(min_x, float(data.qpos[0]))
        for idx in range(int(data.ncon)):
            c = data.contact[idx]
            names = {geom_name(model, c.geom1), geom_name(model, c.geom2)}
            if "stair" in names and first_contact is None:
                first_contact = (step, float(c.dist), np.array(c.frame[:3], dtype=np.float64))
    if first_contact is None:
        print("side_hit: no stair contact")
    else:
        step, dist, normal = first_contact
        print(
            "side_hit:",
            f"first_step={step}",
            f"dist={dist:.5f}",
            f"normal={np.array2string(normal, precision=3, suppress_small=True)}",
            f"final_sphere_x={data.qpos[0]:.4f}",
            f"expected_block_before={-0.2 - 0.04:.4f}",
        )


def print_active_stair_boxes(model: mujoco.MjModel) -> None:
    print("== active stair boxes ==")
    for index in range(64):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"terrain_stair_box_{index:02d}")
        if gid < 0:
            break
        pos = model.geom_pos[gid].copy()
        size = model.geom_size[gid].copy()
        if pos[2] < -10.0:
            continue
        x0 = pos[0] - size[0]
        x1 = pos[0] + size[0]
        top = pos[2] + size[2]
        print(
            f"{index:02d}",
            f"x=[{x0:.4f},{x1:.4f}]",
            f"y_half={size[1]:.3f}",
            f"z=[{pos[2]-size[2]:.4f},{top:.4f}]",
        )


def contact_summary(model: mujoco.MjModel, data: mujoco.MjData, title: str, limit: int = 12) -> None:
    stair_rows: list[str] = []
    terrain_rows: list[str] = []
    min_stair_dist = 1.0
    for idx in range(int(data.ncon)):
        c = data.contact[idx]
        g1 = geom_name(model, c.geom1)
        g2 = geom_name(model, c.geom2)
        normal = np.array(c.frame[:3], dtype=np.float64)
        row = (
            f"  {idx:02d} {g1} <-> {g2} "
            f"dist={float(c.dist):+.5f} "
            f"pos={np.array2string(np.array(c.pos), precision=3, suppress_small=True)} "
            f"n={np.array2string(normal, precision=3, suppress_small=True)}"
        )
        if "terrain_stair_box" in g1 or "terrain_stair_box" in g2:
            stair_rows.append(row)
            min_stair_dist = min(min_stair_dist, float(c.dist))
        if g1 in {"terrain", "ground-plane"} or g2 in {"terrain", "ground-plane"}:
            terrain_rows.append(row)
    print(
        f"-- {title}: ncon={int(data.ncon)}",
        f"stair_contacts={len(stair_rows)}",
        f"terrain_ground_contacts={len(terrain_rows)}",
        f"min_stair_dist={min_stair_dist if stair_rows else float('nan'):+.5f}",
    )
    for row in stair_rows[:limit]:
        print(row)
    if len(stair_rows) > limit:
        print(f"  ... {len(stair_rows) - limit} more stair contacts")


def foot_site_clearance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    config: dict[str, Any],
) -> None:
    site_ids = []
    names = []
    for name in FOOT_SITE_NAMES:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid >= 0:
            site_ids.append(int(sid))
            names.append(name)
    if not site_ids:
        return
    segments = list(config.get("terrain_course", {}).get("segments", []))
    xs = data.site_xpos[site_ids, 0].copy()
    zs = data.site_xpos[site_ids, 2].copy()
    terrain_z = course_height_np(xs, segments) if segments else np.zeros_like(xs)
    print("foot_sites:")
    for name, x, z, h in zip(names, xs, zs, terrain_z):
        print(f"  {name}: x={x:.4f} z={z:.4f} terrain={h:.4f} clearance={z-h:+.4f}")


def model_probe(config_path: Path, phases: list[int]) -> None:
    print("== full MyoAssist stair-box probe ==")
    config = load_config(config_path)
    model, data = build_muscle_model(config)
    control_hz = float(config["control"]["control_hz"])
    reference = load_reference_from_config(DEFAULT_REFERENCE_PATH, model, control_hz, torch.device("cpu"), config)
    print_active_stair_boxes(model)
    print("reference_offsets:")
    for item in reference.get("reference_offsets", []):
        print(f"  {item['start']:04d}-{item['end']:04d} {item['name']}")

    pelvis_ty_qpos = int(model.jnt_qposadr[joint_id(model, "pelvis_ty")])
    pelvis_tx_qpos = int(model.jnt_qposadr[joint_id(model, "pelvis_tx")])
    for phase in phases:
        phase = int(phase) % int(reference["length"])
        mujoco.mj_resetData(model, data)
        set_cpu_reference_state(model, data, reference, phase)
        label = "unknown"
        names = reference.get("reference_names")
        ids = reference.get("reference_id")
        if names is not None and ids is not None:
            label = str(names[int(ids[phase].item())])
        print()
        print(
            f"phase {phase} {label}:",
            f"pelvis_tx={data.qpos[pelvis_tx_qpos]:.4f}",
            f"pelvis_ty={data.qpos[pelvis_ty_qpos]:.4f}",
        )
        foot_site_clearance(model, data, config)
        contact_summary(model, data, "reference")

        base_qpos = data.qpos.copy()
        base_qvel = data.qvel.copy()
        for dz in (-0.02, -0.05, -0.10):
            data.qpos[:] = base_qpos
            data.qvel[:] = base_qvel
            data.qpos[pelvis_ty_qpos] += dz
            apply_joint_equalities_np(model, data)
            mujoco.mj_forward(model, data)
            contact_summary(model, data, f"manual pelvis_ty {dz:+.2f}m", limit=8)


def rollout_probe(config_path: Path, checkpoint_path: Path, phases: list[int], steps: int) -> None:
    print("== checkpoint rollout stair-contact probe ==")
    config = load_config(config_path)
    device = torch.device("cpu")
    model, data = build_muscle_model(config)
    reference = load_reference_from_config(DEFAULT_REFERENCE_PATH, model, float(config["control"]["control_hz"]), device, config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    obs_dim = int(checkpoint.get("run_config", {}).get("obs_dim", 0) or 0)
    if obs_dim <= 0:
        raise ValueError("checkpoint has no run_config.obs_dim")
    act_dim = int(checkpoint.get("run_config", {}).get("act_dim", model.nu) or model.nu)
    sac_cfg = config.get("sac", config.get("ppo", {}))
    base_actor = SACActor(
        obs_dim,
        act_dim,
        logstd_init=float(sac_cfg.get("actor_logstd_init", -0.5)),
        initial_action_mean=float(sac_cfg.get("initial_actor_action_mean", -0.2)),
    ).to(device)
    if bool(checkpoint.get("run_config", {}).get("symmetric_policy", sac_cfg.get("symmetric_policy", False))):
        mirror_spec = build_sagittal_mirror_spec(
            model,
            obs_dim=obs_dim,
            future_steps=int(config.get("imitation", {}).get("reference_future_steps", 0)),
            device=device,
        )
        actor = SymmetricSACActor(
            base_actor,
            obs_perm=mirror_spec["obs_perm"],
            obs_sign=mirror_spec["obs_sign"],
            act_perm=mirror_spec["act_perm"],
            act_sign=mirror_spec["act_sign"],
        ).to(device)
    else:
        actor = base_actor
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()

    from cleanrl.ppo_muscle_mjwarp import ObsNormalizer  # local import keeps this script focused.

    obs_normalizer = ObsNormalizer(
        obs_dim,
        device,
        enabled=bool(sac_cfg.get("normalize_observations", True)),
        clip=float(sac_cfg.get("obs_norm_clip", 10.0)),
    )
    if "obs_normalizer" in checkpoint:
        obs_normalizer.load_state_dict(checkpoint["obs_normalizer"])

    frame_skip = int(config["control"]["frame_skip"])
    pelvis_ty_qpos = int(model.jnt_qposadr[joint_id(model, "pelvis_ty")])
    for start_phase in phases:
        phase = int(start_phase) % int(reference["length"])
        mujoco.mj_resetData(model, data)
        set_cpu_reference_state(model, data, reference, phase)
        min_stair_dist = 1.0
        max_qvel = 0.0
        first_rows: list[str] = []
        stair_by_geom: dict[str, int] = {}
        for frame in range(int(steps)):
            obs = cpu_policy_obs(model, data, reference, config, phase, device)
            with torch.no_grad():
                action, _, _, _ = actor.get_action_and_value(obs_normalizer.normalize(obs), deterministic=True)
            data.ctrl[:] = muscle_action_to_activation(action[0]).detach().cpu().numpy().astype(np.float64)
            for substep in range(frame_skip):
                mujoco.mj_step(model, data)
                max_qvel = max(max_qvel, float(np.max(np.abs(data.qvel))))
                for idx in range(int(data.ncon)):
                    c = data.contact[idx]
                    g1 = geom_name(model, c.geom1)
                    g2 = geom_name(model, c.geom2)
                    if "terrain_stair_box" not in g1 and "terrain_stair_box" not in g2:
                        continue
                    min_stair_dist = min(min_stair_dist, float(c.dist))
                    other = g2 if "terrain_stair_box" in g1 else g1
                    stair_by_geom[other] = stair_by_geom.get(other, 0) + 1
                    if len(first_rows) < 10:
                        first_rows.append(
                            f"    frame={frame} sub={substep} {g1}<->{g2} "
                            f"dist={float(c.dist):+.5f} "
                            f"n={np.array2string(np.array(c.frame[:3]), precision=3, suppress_small=True)}"
                        )
            phase = (phase + 1) % int(reference["length"])
        label = "unknown"
        names = reference.get("reference_names")
        ids = reference.get("reference_id")
        if names is not None and ids is not None:
            label = str(names[int(ids[int(start_phase) % int(reference["length"])].item())])
        print(
            f"rollout phase {int(start_phase)} {label}:",
            f"steps={int(steps)}",
            f"pelvis_ty={data.qpos[pelvis_ty_qpos]:.4f}",
            f"max_qvel={max_qvel:.2f}",
            f"min_stair_dist={min_stair_dist if min_stair_dist < 1.0 else float('nan'):+.5f}",
            f"stair_geoms={stair_by_geom}",
        )
        for row in first_rows:
            print(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/muscle_2d_mjwarp_stageF_h192_camargo_multigait_heightmap_stair_riser_trial_sac.json"),
    )
    parser.add_argument("--phases", type=int, nargs="*", default=[253, 287, 542, 579])
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--rollout-steps", type=int, default=0)
    args = parser.parse_args()
    simple_box_probe()
    model_probe(args.config, args.phases)
    if args.checkpoint is not None and int(args.rollout_steps) > 0:
        rollout_probe(args.config, args.checkpoint, args.phases, int(args.rollout_steps))


if __name__ == "__main__":
    main()
