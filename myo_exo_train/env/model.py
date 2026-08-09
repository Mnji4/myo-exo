"""MuJoCo model loading, control mapping, and terrain construction."""
from __future__ import annotations

import math
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Sequence

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TRACK_JOINTS = [
    "pelvis_ty", "pelvis_tilt", "hip_flexion_r", "knee_angle_r", "ankle_angle_r",
    "mtp_angle_r", "hip_flexion_l", "knee_angle_l", "ankle_angle_l", "mtp_angle_l",
]
RESET_JOINTS = ["pelvis_tx", *TRACK_JOINTS]
FOOT_SITE_NAMES = ["r_heel_btm", "r_toe_btm", "l_heel_btm", "l_toe_btm"]
def muscle_action_mapping_mode(config: dict[str, Any] | None = None) -> str:
    if config is None:
        return "linear"
    control_cfg = config.get("control", {})
    mapping = control_cfg.get("muscle_action_mapping", config.get("muscle_action_mapping", "linear"))
    return str(mapping or "linear").lower()

def muscle_action_to_activation(action: torch.Tensor, mapping: str = "linear") -> torch.Tensor:
    normalized_action = torch.clamp(action, -1.0, 1.0)
    if str(mapping).lower() in {"myosuite", "myosuite_sigmoid", "myosuite_muscle_sigmoid"}:
        return torch.sigmoid(5.0 * (normalized_action - 0.5))
    return 0.5 * (normalized_action + 1.0)

def normalized_action_to_ctrl(action: torch.Tensor, ctrl_low: torch.Tensor, ctrl_high: torch.Tensor) -> torch.Tensor:
    normalized_action = torch.clamp(action, -1.0, 1.0)
    return ctrl_low.unsqueeze(0) + 0.5 * (normalized_action + 1.0) * (ctrl_high - ctrl_low).unsqueeze(0)

def policy_action_to_ctrl(
    action: torch.Tensor,
    ctrl_low: torch.Tensor,
    ctrl_high: torch.Tensor,
    *,
    muscle_count: int,
    muscle_mapping: str = "linear",
) -> torch.Tensor:
    ctrl = normalized_action_to_ctrl(action, ctrl_low, ctrl_high)
    muscle_count = max(0, min(int(muscle_count), int(ctrl.shape[1])))
    if muscle_count > 0:
        ctrl[:, :muscle_count] = muscle_action_to_activation(action[:, :muscle_count], mapping=muscle_mapping)
    return ctrl

def apply_non_muscle_ctrl_override(
    ctrl: torch.Tensor,
    config: dict[str, Any] | None,
    *,
    muscle_count: int,
    ctrl_low: torch.Tensor | None = None,
    ctrl_high: torch.Tensor | None = None,
) -> torch.Tensor:
    if config is None:
        return ctrl
    cfg = config.get("non_muscle_ctrl_override", config.get("control", {}).get("non_muscle_ctrl_override", {}))
    if not isinstance(cfg, dict) or not bool(cfg.get("enabled", False)):
        return ctrl
    muscle_count = max(0, min(int(muscle_count), int(ctrl.shape[-1])))
    non_muscle_count = int(ctrl.shape[-1]) - muscle_count
    if non_muscle_count <= 0:
        return ctrl
    raw_value = cfg.get("values", cfg.get("value", 0.0))
    if isinstance(raw_value, (list, tuple)):
        values = torch.tensor(raw_value, dtype=ctrl.dtype, device=ctrl.device).flatten()
        if int(values.numel()) == 1:
            values = values.expand(non_muscle_count)
        elif int(values.numel()) != non_muscle_count:
            raise ValueError(
                f"non_muscle_ctrl_override values has {int(values.numel())} entries, expected 1 or {non_muscle_count}"
            )
    else:
        values = torch.full((non_muscle_count,), float(raw_value), dtype=ctrl.dtype, device=ctrl.device)
    if ctrl_low is not None and ctrl_high is not None:
        lo = ctrl_low.to(dtype=ctrl.dtype, device=ctrl.device)[muscle_count:]
        hi = ctrl_high.to(dtype=ctrl.dtype, device=ctrl.device)[muscle_count:]
        values = torch.minimum(torch.maximum(values, lo), hi)
    out = ctrl.clone()
    out[..., muscle_count:] = values
    return out

def name_id(model: mujoco.MjModel, objtype: mujoco.mjtObj, name: str) -> int:
    idx = mujoco.mj_name2id(model, objtype, name)
    if idx < 0:
        raise KeyError(f"Missing MuJoCo object {objtype}: {name}")
    return int(idx)

def joint_id_or_none(model: mujoco.MjModel, name: str) -> int | None:
    idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return int(idx) if idx >= 0 else None

def site_id(model: mujoco.MjModel, name: str) -> int:
    return name_id(model, mujoco.mjtObj.mjOBJ_SITE, name)

def site_id_or_none(model: mujoco.MjModel, name: str) -> int | None:
    idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    return int(idx) if idx >= 0 else None

def sensor_adr_or_none(model: mujoco.MjModel, name: str) -> int | None:
    idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if idx < 0:
        return None
    return int(model.sensor_adr[idx])

def freejoint_root_id(model: mujoco.MjModel) -> int | None:
    for jid in range(int(model.njnt)):
        if int(model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_FREE):
            return int(jid)
    return None

def semantic_qpos_index(model: mujoco.MjModel, name: str) -> int:
    jid = joint_id_or_none(model, name)
    if jid is not None:
        return int(model.jnt_qposadr[jid])
    root_jid = freejoint_root_id(model)
    if root_jid is None:
        raise KeyError(f"Missing MuJoCo joint {name} and no freejoint root alias is available")
    root_adr = int(model.jnt_qposadr[root_jid])
    alias_offset = {
        "pelvis_tx": 1,
        "pelvis_ty": 2,
        "pelvis_tilt": 3,
    }.get(name)
    if alias_offset is None:
        raise KeyError(f"Missing MuJoCo joint {name}")
    return root_adr + alias_offset

def semantic_qvel_index(model: mujoco.MjModel, name: str) -> int:
    jid = joint_id_or_none(model, name)
    if jid is not None:
        return int(model.jnt_dofadr[jid])
    root_jid = freejoint_root_id(model)
    if root_jid is None:
        raise KeyError(f"Missing MuJoCo joint {name} and no freejoint root alias is available")
    root_adr = int(model.jnt_dofadr[root_jid])
    alias_offset = {
        "pelvis_tx": 1,
        "pelvis_ty": 2,
        "pelvis_tilt": 3,
    }.get(name)
    if alias_offset is None:
        raise KeyError(f"Missing MuJoCo joint {name}")
    return root_adr + alias_offset

def model_foot_site_names(model: mujoco.MjModel, config: dict[str, Any] | None = None) -> list[str]:
    configured = []
    if config is not None:
        configured = [str(v) for v in config.get("model", {}).get("foot_site_names", []) if str(v)]
    candidates = [
        configured,
        FOOT_SITE_NAMES,
        ["r_foot_touch", "r_toes_touch", "l_foot_touch", "l_toes_touch"],
    ]
    for names in candidates:
        if len(names) == len(FOOT_SITE_NAMES) and all(site_id_or_none(model, name) is not None for name in names):
            return list(names)
    raise KeyError("Could not find a complete 4-site foot contact set for this model")

def model_foot_sensor_names(model: mujoco.MjModel, config: dict[str, Any] | None = None) -> list[str]:
    configured = []
    if config is not None:
        configured = [str(v) for v in config.get("model", {}).get("foot_sensor_names", []) if str(v)]
    candidates = [
        configured,
        ["r_foot", "r_toes", "l_foot", "l_toes"],
        ["r_foot", "l_foot", "r_toes", "l_toes"],
    ]
    for names in candidates:
        if len(names) == len(FOOT_SITE_NAMES) and all(sensor_adr_or_none(model, name) is not None for name in names):
            return list(names)
    return []

def model_ground_force_sensor_names(
    model: mujoco.MjModel,
    config: dict[str, Any] | None = None,
) -> list[str]:
    configured = []
    if config is not None:
        configured = [
            str(value)
            for value in config.get("model", {}).get("ground_force_sensor_names", [])
            if str(value)
        ]
    if configured:
        if len(configured) % 2 != 0 or not all(
            sensor_adr_or_none(model, name) is not None for name in configured
        ):
            raise KeyError(
                "model.ground_force_sensor_names must contain complete right/left sensor halves"
            )
        return configured
    return model_foot_sensor_names(model, config)

def terrain_forward_axis(config: dict[str, Any] | None = None) -> str:
    if config is None:
        return "x"
    axis = config.get("terrain_forward_axis", config.get("model", {}).get("forward_axis", config.get("terrain_course", {}).get("forward_axis", "x")))
    axis = str(axis or "x").lower()
    if axis not in {"x", "y"}:
        raise ValueError(f"terrain_forward_axis must be 'x' or 'y', got {axis!r}")
    return axis

def terrain_forward_site_dim(config: dict[str, Any] | None = None) -> int:
    return 1 if terrain_forward_axis(config) == "y" else 0

def site_forward_coord_tensor(site_xpos: torch.Tensor, config: dict[str, Any] | None = None) -> torch.Tensor:
    return site_xpos[:, :, terrain_forward_site_dim(config)]

def site_lateral_coord_tensor(site_xpos: torch.Tensor, config: dict[str, Any] | None = None) -> torch.Tensor:
    return site_xpos[:, :, 0 if terrain_forward_axis(config) == "y" else 1]

def site_forward_coord_np(site_xpos: np.ndarray, config: dict[str, Any] | None = None) -> np.ndarray:
    return site_xpos[:, terrain_forward_site_dim(config)]

def key_id_or_none(model: mujoco.MjModel, name: str) -> int | None:
    idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name)
    return int(idx) if idx >= 0 else None

def joint_equality_specs_np(model: mujoco.MjModel) -> list[tuple[int, int, int, int, np.ndarray]]:
    specs = []
    for eq_id in range(model.neq):
        if int(model.eq_type[eq_id]) != int(mujoco.mjtEq.mjEQ_JOINT):
            continue
        joint1 = int(model.eq_obj1id[eq_id])
        joint2 = int(model.eq_obj2id[eq_id])
        specs.append(
            (
                int(model.jnt_qposadr[joint1]),
                int(model.jnt_qposadr[joint2]),
                int(model.jnt_dofadr[joint1]),
                int(model.jnt_dofadr[joint2]),
                np.asarray(model.eq_data[eq_id, :5], dtype=np.float32),
            )
        )
    return specs


def apply_joint_equalities_np(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Project polynomial joint equalities into a CPU MuJoCo state."""
    for qpos1, qpos2, qvel1, qvel2, poly in joint_equality_specs_np(model):
        q = float(data.qpos[qpos2])
        dq = float(data.qvel[qvel2])
        data.qpos[qpos1] = float(
            poly[0] + poly[1] * q + poly[2] * q**2 + poly[3] * q**3 + poly[4] * q**4
        )
        derivative = float(
            poly[1] + 2.0 * poly[2] * q + 3.0 * poly[3] * q**2 + 4.0 * poly[4] * q**3
        )
        data.qvel[qvel1] = derivative * dq

def source_xml_with_stair_box_contacts(config: dict[str, Any]) -> tuple[Path, bool]:
    source_xml = Path(config["model"]["source_xml"]).expanduser()
    course_cfg = config.get("terrain_course", {})
    if not bool(course_cfg.get("enabled", False)):
        return source_xml, False

    tree = ET.parse(source_xml)
    root = tree.getroot()

    complete_force_cfg = config.get("model", {}).get(
        "complete_ground_force_sensors", {}
    )
    if isinstance(complete_force_cfg, dict) and bool(
        complete_force_cfg.get("enabled", False)
    ):
        sensor_root = root.find("sensor")
        if sensor_root is None:
            sensor_root = ET.SubElement(root, "sensor")
        radius = float(complete_force_cfg.get("site_radius", 1.0))
        for side in ("r", "l"):
            for body_stem in ("talus", "calcn", "toes", "foot_attachment"):
                body_name = f"{body_stem}_{side}"
                body = root.find(f".//body[@name='{body_name}']")
                if body is None:
                    raise KeyError(
                        f"missing {body_name} body for complete force sensor"
                    )
                site_name = f"{side}_{body_stem}_touch_full"
                sensor_name = f"{side}_{body_stem}_full"
                if root.find(f".//site[@name='{site_name}']") is None:
                    ET.SubElement(
                        body,
                        "site",
                        {
                            "name": site_name,
                            "type": "sphere",
                            "pos": "0 0 0",
                            "size": f"{radius:g}",
                            "rgba": "0 0 0 0",
                        },
                    )
                if root.find(f".//touch[@name='{sensor_name}']") is None:
                    ET.SubElement(
                        sensor_root,
                        "touch",
                        {"name": sensor_name, "site": site_name},
                    )

    def numeric_attr(values: Any) -> str:
        if isinstance(values, str):
            return values
        return " ".join(f"{float(value):g}" for value in values)

    def parse_numeric(values: Any, default: Sequence[float]) -> list[float]:
        if values is None:
            return [float(value) for value in default]
        if isinstance(values, str):
            raw = values.split()
        else:
            raw = list(values)
        out = [float(value) for value in raw]
        while len(out) < len(default):
            out.append(float(default[len(out)]))
        return out

    def stair_box_geom_pose(x0: float, x1: float, height: float, half_width: float) -> tuple[list[float], list[float]]:
        if x1 <= x0 or height <= 1e-6:
            return [0.0, 0.0, -100.0], [0.01, 0.01, 0.01]
        center = (float(x0) + float(x1)) * 0.5
        half_length = (float(x1) - float(x0)) * 0.5
        if terrain_forward_axis(config) == "y":
            return [0.0, center, float(height) * 0.5], [float(half_width), half_length, float(height) * 0.5]
        return [center, 0.0, float(height) * 0.5], [half_length, float(half_width), float(height) * 0.5]

    def terrain_box_geom_pose(segment: dict[str, Any], half_width: float, thickness: float) -> tuple[list[float], list[float], list[float]]:
        nominal_x0 = float(segment.get("x0", 0.0))
        nominal_x1 = float(segment.get("x1", nominal_x0))
        x0 = nominal_x0 - max(
            float(segment.get("geometry_overlap_before_m", 0.0)), 0.0
        )
        x1 = nominal_x1 + max(
            float(segment.get("geometry_overlap_after_m", 0.0)), 0.0
        )
        if x1 <= x0:
            return [0.0, 0.0, -100.0], [0.01, 0.01, 0.01], [1.0, 0.0, 0.0, 0.0]
        kind = str(segment.get("type", "flat_box"))
        center = 0.5 * (x0 + x1)
        half_length = 0.5 * (x1 - x0)
        thickness = max(float(thickness), 1e-4)
        half_thickness = 0.5 * thickness
        if kind == "ramp_box":
            h0 = float(segment.get("height0", 0.0))
            if "height1" in segment:
                h1 = float(segment["height1"])
            else:
                h1 = h0 + float(segment.get("slope", 0.0)) * (x1 - x0)
            theta = math.atan2(h1 - h0, x1 - x0)
            length = math.hypot(x1 - x0, h1 - h0)
            top_mid = 0.5 * (h0 + h1)
            if terrain_forward_axis(config) == "y":
                # Rotation about world x: local y follows the ramp, local z is the top normal.
                quat = [math.cos(0.5 * theta), math.sin(0.5 * theta), 0.0, 0.0]
                normal_y = -math.sin(theta)
                normal_z = math.cos(theta)
                pos = [0.0, center - half_thickness * normal_y, top_mid - half_thickness * normal_z]
                size = [float(half_width), 0.5 * length, half_thickness]
            else:
                # Rotation about world y with local x following the ramp.
                quat = [math.cos(0.5 * theta), 0.0, -math.sin(0.5 * theta), 0.0]
                normal_x = -math.sin(theta)
                normal_z = math.cos(theta)
                pos = [center - half_thickness * normal_x, 0.0, top_mid - half_thickness * normal_z]
                size = [0.5 * length, float(half_width), half_thickness]
            return pos, size, quat
        height = float(segment.get("height", 0.0))
        if terrain_forward_axis(config) == "y":
            return [0.0, center, height - half_thickness], [float(half_width), half_length, half_thickness], [1.0, 0.0, 0.0, 0.0]
        return [center, 0.0, height - half_thickness], [half_length, float(half_width), half_thickness], [1.0, 0.0, 0.0, 0.0]

    def generated_terrain_box_segments() -> list[dict[str, Any]]:
        return [
            segment
            for segment in list(course_cfg.get("segments", []))
            if str(segment.get("type", "flat")) in {"flat_box", "ramp_box"}
        ]

    def ensure_terrain_box_geoms(include_root: ET.Element) -> int:
        box_segments = generated_terrain_box_segments()
        if not box_segments:
            return 0
        worldbody = include_root.find("worldbody")
        if worldbody is None:
            return 0
        half_width = float(course_cfg.get("terrain_box_half_width", course_cfg.get("stair_box_half_width", 5.0)))
        thickness = float(course_cfg.get("terrain_box_thickness", 0.08))
        box_count = max(int(course_cfg.get("terrain_box_pair_count", 16) or 16), len(box_segments))
        palette = list(
            course_cfg.get(
                "terrain_box_rgba_palette",
                [
                    [0.60, 0.70, 0.62, 1.0],
                    [0.76, 0.67, 0.46, 1.0],
                    [0.58, 0.67, 0.60, 1.0],
                ],
            )
        )
        made = 0
        for index in range(box_count):
            box_name = f"terrain_box_{index:02d}"
            geom = include_root.find(f".//geom[@name='{box_name}']")
            if geom is None:
                geom = ET.SubElement(worldbody, "geom", {"name": box_name, "type": "box"})
            if index < len(box_segments):
                pos, size, quat = terrain_box_geom_pose(box_segments[index], half_width, thickness)
                rgba = palette[index % len(palette)]
            else:
                pos, size, quat, rgba = [0.0, 0.0, -100.0], [0.01, 0.01, 0.01], [1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0]
            geom.set("type", "box")
            geom.set("pos", numeric_attr(pos))
            geom.set("size", numeric_attr(size))
            geom.set("quat", numeric_attr(quat))
            geom.set("rgba", numeric_attr(rgba))
            geom.set("contype", str(int(course_cfg.get("terrain_box_contype", 1))))
            geom.set("conaffinity", str(int(course_cfg.get("terrain_box_conaffinity", 1))))
            made += 1
        return made

    def generated_stair_box_segments() -> list[tuple[float, float, float]]:
        generated: list[tuple[float, float, float]] = []
        generated_walls: list[tuple[float, float, float]] = []
        for segment in list(course_cfg.get("segments", [])):
            if str(segment.get("type", "flat")) != "stairs_box":
                continue
            generated.extend(stair_box_treads(segment))
            if bool(course_cfg.get("stair_box_add_riser_walls", False)):
                generated_walls.extend(
                    stair_box_riser_walls(segment, float(course_cfg.get("stair_box_riser_wall_thickness", 0.025)))
                )
        return merge_stair_box_treads(generated) + generated_walls

    def ensure_stair_box_geoms(include_root: ET.Element) -> int:
        if not any(str(segment.get("type", "flat")) == "stairs_box" for segment in list(course_cfg.get("segments", []))):
            return 0
        if bool(course_cfg.get("stairs_box_as_hfield", False)):
            return 0
        worldbody = include_root.find("worldbody")
        if worldbody is None:
            return 0
        half_width = float(course_cfg.get("stair_box_half_width", 0.6))
        collision_boxes = generated_stair_box_segments()
        box_count = max(int(course_cfg.get("stair_box_pair_count", 16) or 16), len(collision_boxes))
        palette = list(
            course_cfg.get(
                "stair_box_rgba_palette",
                [
                    [0.95, 0.18, 0.18, 1.0],
                    [0.98, 0.62, 0.12, 1.0],
                    [0.20, 0.70, 0.34, 1.0],
                    [0.12, 0.45, 0.95, 1.0],
                    [0.58, 0.24, 0.88, 1.0],
                ],
            )
        )
        made = 0
        for index in range(box_count):
            box_name = f"terrain_stair_box_{index:02d}"
            geom = include_root.find(f".//geom[@name='{box_name}']")
            if geom is None:
                geom = ET.SubElement(worldbody, "geom", {"name": box_name, "type": "box"})
            if index < len(collision_boxes):
                pos, size = stair_box_geom_pose(*collision_boxes[index], half_width)
                rgba = palette[index % len(palette)]
            else:
                pos, size, rgba = [0.0, 0.0, -100.0], [0.01, 0.01, 0.01], [1.0, 1.0, 1.0, 0.0]
            geom.set("type", "box")
            geom.set("pos", numeric_attr(pos))
            geom.set("size", numeric_attr(size))
            geom.set("rgba", numeric_attr(rgba))
            geom.set("contype", str(int(course_cfg.get("stair_box_contype", 1))))
            geom.set("conaffinity", str(int(course_cfg.get("stair_box_conaffinity", 1))))
            made += 1
        return made

    def add_sole_contact_spheres(include_root: ET.Element) -> list[str]:
        sole_cfg = course_cfg.get("sole_contact_spheres", {})
        if not bool(sole_cfg.get("enabled", False)):
            return []
        site_names = [str(name) for name in sole_cfg.get("site_names", []) if str(name)]
        if not site_names:
            site_names = ["r_foot_touch", "r_toes_touch", "l_foot_touch", "l_toes_touch"]
        radius = float(sole_cfg.get("radius", 0.018))
        rgba = sole_cfg.get("rgba", [0.1, 0.9, 0.2, 0.35])
        added: list[str] = []
        for body in include_root.iter("body"):
            existing = {str(geom.get("name", "")) for geom in body.findall("geom")}
            for site in list(body.findall("site")):
                site_name = str(site.get("name", ""))
                if site_name not in site_names:
                    continue
                geom_name = f"{site_name}_sole_contact"
                if geom_name in existing:
                    continue
                pos = parse_numeric(site.get("pos"), [0.0, 0.0, 0.0])[:3]
                ET.SubElement(
                    body,
                    "geom",
                    {
                        "name": geom_name,
                        "type": "sphere",
                        "pos": numeric_attr(pos),
                        "size": f"{radius:g}",
                        "rgba": numeric_attr(rgba),
                        "contype": str(int(sole_cfg.get("contype", 1))),
                        "conaffinity": str(int(sole_cfg.get("conaffinity", 0))),
                        "condim": str(int(sole_cfg.get("condim", 3))),
                        "margin": f"{float(sole_cfg.get('margin', 0.001)):g}",
                    },
                )
                existing.add(geom_name)
                added.append(geom_name)
        return added

    include_patched = False
    hfield_name = str(course_cfg.get("hfield_name", "terrain"))
    terrain_geom = str(course_cfg.get("terrain_geom", "terrain"))
    ground_geom = str(course_cfg.get("ground_geom", "ground-plane"))
    disable_ground_plane_contact = bool(course_cfg.get("disable_ground_plane_contact", False))
    generated_sole_contact_geoms: list[str] = []
    for include in root.findall(".//include"):
        include_file = str(include.get("file", "") or "")
        if not include_file:
            continue
        include_path = Path(include_file).expanduser()
        if not include_path.is_absolute():
            include_path = (source_xml.parent / include_path).resolve()
        if not include_path.exists():
            continue
        try:
            include_tree = ET.parse(include_path)
        except ET.ParseError:
            continue
        include_root = include_tree.getroot()
        hfield = include_root.find(f".//hfield[@name='{hfield_name}']")
        patched_this_include = False
        if hfield is not None:
            size_values = [float(value) for value in str(hfield.get("size", "1 1 1 1")).split()]
            while len(size_values) < 4:
                size_values.append(1.0)
            if "hfield_size_x" in course_cfg:
                size_values[0] = max(float(course_cfg["hfield_size_x"]), 1e-6)
            if "hfield_size_y" in course_cfg:
                size_values[1] = max(float(course_cfg["hfield_size_y"]), 1e-6)
            if "hfield_size_z" in course_cfg:
                size_values[2] = max(float(course_cfg["hfield_size_z"]), 1e-6)
            if "hfield_base" in course_cfg:
                size_values[3] = max(float(course_cfg["hfield_base"]), 0.0)
            hfield.set("size", numeric_attr(size_values))
            terrain_geom_node = include_root.find(f".//geom[@name='{terrain_geom}']")
            if terrain_geom_node is not None and "terrain_geom_z" in course_cfg:
                pos_values = parse_numeric(terrain_geom_node.get("pos"), [0.0, 0.0, 0.0])[:3]
                pos_values[2] = float(course_cfg["terrain_geom_z"])
                terrain_geom_node.set("pos", numeric_attr(pos_values))
            ground_geom_node = include_root.find(f".//geom[@name='{ground_geom}']")
            if ground_geom_node is not None and disable_ground_plane_contact:
                ground_geom_node.set("contype", "0")
                ground_geom_node.set("conaffinity", "0")
            terrain_geom_node = include_root.find(f".//geom[@name='{terrain_geom}']")
            if terrain_geom_node is not None and bool(course_cfg.get("terrain_box_hide_hfield", False)):
                terrain_geom_node.set("rgba", "1 1 1 0")
                terrain_geom_node.set("contype", "0")
                terrain_geom_node.set("conaffinity", "0")
            ensure_terrain_box_geoms(include_root)
            ensure_stair_box_geoms(include_root)
            patched_this_include = True
        sole_geoms = add_sole_contact_spheres(include_root)
        if sole_geoms:
            generated_sole_contact_geoms.extend(sole_geoms)
            patched_this_include = True
        if not patched_this_include:
            continue
        temp_include = tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=f"_{include_path.stem}.xml",
            prefix=f"{source_xml.stem}_",
            dir=str(include_path.parent),
            delete=False,
        )
        temp_include_path = Path(temp_include.name)
        try:
            include_tree.write(temp_include, encoding="utf-8", xml_declaration=False)
        finally:
            temp_include.close()
        include.set("file", str(temp_include_path))
        include_patched = True

    stair_box_segments = generated_stair_box_segments()
    terrain_box_segments = generated_terrain_box_segments()
    add_stair_pairs = bool(
        course_cfg.get(
            "add_stair_box_contact_pairs",
            bool(course_cfg.get("stair_boxes_precompiled", False)) or bool(stair_box_segments),
        )
    )
    add_terrain_box_pairs = bool(course_cfg.get("add_terrain_box_contact_pairs", bool(terrain_box_segments)))
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")

    hide_hfield_contact = bool(course_cfg.get("terrain_box_hide_hfield", False))
    terrain_pairs = [
        pair
        for pair in list(contact.findall("pair"))
        if terrain_geom in {
            str(pair.get("geom1", "")),
            str(pair.get("geom2", "")),
        }
    ]
    if not terrain_pairs and not include_patched:
        return source_xml, False

    # Explicit MuJoCo contact pairs bypass contype/conaffinity filtering. Remove
    # the source-model pairs as well when a legacy support surface is disabled.
    if hide_hfield_contact:
        for pair in terrain_pairs:
            contact.remove(pair)
    if disable_ground_plane_contact:
        for pair in list(contact.findall("pair")):
            if ground_geom in {
                str(pair.get("geom1", "")),
                str(pair.get("geom2", "")),
            }:
                contact.remove(pair)

    existing = {(pair.get("geom1", ""), pair.get("geom2", "")) for pair in contact.findall("pair")}

    stair_box_match_terrain_contact = bool(course_cfg.get("stair_box_match_terrain_contact", True))

    def pair_friction_attr(values: Any) -> str:
        parsed = parse_numeric(values, [1.0, 0.005, 0.0001])
        if len(parsed) == 3:
            # MuJoCo pair friction has five entries. Geom-style friction is
            # [sliding, torsional, rolling], so mirror sliding/rolling across
            # the two tangent/rolling dimensions.
            parsed = [parsed[0], parsed[0], parsed[1], parsed[2], parsed[2]]
        return numeric_attr(parsed[:5])

    def apply_stair_box_contact_overrides(attrs: dict[str, str]) -> None:
        if stair_box_match_terrain_contact:
            return
        if "stair_box_contact_margin" in course_cfg:
            attrs["margin"] = f"{float(course_cfg['stair_box_contact_margin']):g}"
        if "stair_box_contact_solref" in course_cfg:
            attrs["solref"] = numeric_attr(course_cfg["stair_box_contact_solref"])
        if "stair_box_contact_solimp" in course_cfg:
            attrs["solimp"] = numeric_attr(course_cfg["stair_box_contact_solimp"])
        if "stair_box_contact_friction" in course_cfg:
            attrs["friction"] = pair_friction_attr(course_cfg["stair_box_contact_friction"])

    def add_explicit_pair(geom1: str, geom2: str, *, stair_box: bool = False, template: ET.Element | None = None) -> None:
        key = (geom1, geom2)
        if key in existing:
            return
        if stair_box and stair_box_match_terrain_contact and template is not None:
            attrs = dict(template.attrib)
            attrs["geom1"] = geom1
            attrs["geom2"] = geom2
        else:
            attrs = {
                "geom1": geom1,
                "geom2": geom2,
                "condim": str(int(course_cfg.get("contact_condim", 3))),
            }
        if stair_box:
            apply_stair_box_contact_overrides(attrs)
        else:
            if "terrain_contact_margin" in course_cfg:
                attrs["margin"] = f"{float(course_cfg['terrain_contact_margin']):g}"
            if "terrain_contact_friction" in course_cfg:
                attrs["friction"] = pair_friction_attr(course_cfg["terrain_contact_friction"])
        ET.SubElement(contact, "pair", attrs)
        existing.add(key)

    def add_pair_from_template(geom1: str, geom2: str, template: ET.Element) -> None:
        key = (geom1, geom2)
        if key in existing:
            return
        attrs = dict(template.attrib)
        attrs["geom1"] = geom1
        attrs["geom2"] = geom2
        if geom1.startswith("terrain_stair_box_"):
            apply_stair_box_contact_overrides(attrs)
        ET.SubElement(contact, "pair", attrs)
        existing.add(key)

    extra_foot_geoms = [
        str(name)
        for name in course_cfg.get("extra_foot_contact_geoms", [])
        if str(name)
    ]
    extra_foot_geoms.extend(name for name in generated_sole_contact_geoms if name not in extra_foot_geoms)
    default_foot_geoms = [
        "r_foot_col1",
        "r_foot_col3",
        "r_foot_col4",
        "r_bofoot_col1",
        "r_bofoot_col2",
        "l_foot_col1",
        "l_foot_col3",
        "l_foot_col4",
        "l_bofoot_col1",
        "l_bofoot_col2",
    ]
    configured_foot_geoms = [
        str(name)
        for name in course_cfg.get("foot_contact_geoms", default_foot_geoms)
        if str(name)
    ]
    direct_foot_geoms = []
    for name in configured_foot_geoms + extra_foot_geoms:
        if name not in direct_foot_geoms:
            direct_foot_geoms.append(name)
    template_by_side = {}
    if terrain_pairs:
        template_by_side = {
            "_r_": next((pair for pair in terrain_pairs if str(pair.get("geom2", "")) == "calcn_r_geom_1"), terrain_pairs[0]),
            "_l_": next((pair for pair in terrain_pairs if str(pair.get("geom2", "")) == "calcn_l_geom_1"), terrain_pairs[0]),
        }
    ground_pairs = [
        pair
        for pair in list(contact.findall("pair"))
        if str(pair.get("geom1", "")) == ground_geom
    ]
    ground_template_by_side = {
        "_r_": next((pair for pair in ground_pairs if str(pair.get("geom2", "")) == "calcn_r_geom_1"), None),
        "_l_": next((pair for pair in ground_pairs if str(pair.get("geom2", "")) == "calcn_l_geom_1"), None),
    }
    if add_stair_pairs and terrain_pairs:
        for geom2 in extra_foot_geoms:
            side = "_l_" if "_l_" in geom2 else "_r_"
            add_pair_from_template(terrain_geom, geom2, template_by_side.get(side, terrain_pairs[0]))
            ground_template = ground_template_by_side.get(side)
            if ground_template is not None and not disable_ground_plane_contact:
                add_pair_from_template(ground_geom, geom2, ground_template)

        terrain_pairs = [
            pair
            for pair in list(contact.findall("pair"))
            if str(pair.get("geom1", "")) == terrain_geom
        ]
        box_count = int(course_cfg.get("stair_box_pair_count", 16) or 16)
        for index in range(max(0, box_count)):
            box_name = f"terrain_stair_box_{index:02d}"
            for terrain_pair in terrain_pairs:
                geom2 = str(terrain_pair.get("geom2", ""))
                add_pair_from_template(box_name, geom2, terrain_pair)
    if (add_stair_pairs or add_terrain_box_pairs) and direct_foot_geoms:
        if bool(course_cfg.get("add_ground_contact_pairs", True)) and not disable_ground_plane_contact:
            for geom2 in direct_foot_geoms:
                add_explicit_pair(ground_geom, geom2)
        if (
            bool(course_cfg.get("add_terrain_contact_pairs", True))
            and not hide_hfield_contact
        ):
            for geom2 in direct_foot_geoms:
                add_explicit_pair(terrain_geom, geom2)
        terrain_pairs = [
            pair
            for pair in list(contact.findall("pair"))
            if str(pair.get("geom1", "")) == terrain_geom
        ]
        terrain_pair_by_geom2 = {str(pair.get("geom2", "")): pair for pair in terrain_pairs}

        def terrain_template_for_geom(geom2: str) -> ET.Element | None:
            exact = terrain_pair_by_geom2.get(geom2)
            if exact is not None:
                return exact
            side = "_l_" if "_l_" in geom2 else "_r_"
            return template_by_side.get(side, terrain_pairs[0] if terrain_pairs else None)

        if add_stair_pairs:
            box_count = max(
                int(course_cfg.get("stair_box_pair_count", 16) or 16),
                len(stair_box_segments),
            )
            for index in range(max(0, box_count)):
                box_name = f"terrain_stair_box_{index:02d}"
                for geom2 in direct_foot_geoms:
                    add_explicit_pair(box_name, geom2, stair_box=True, template=terrain_template_for_geom(geom2))

        if add_terrain_box_pairs:
            terrain_box_count = max(
                int(course_cfg.get("terrain_box_pair_count", 16) or 16),
                len(terrain_box_segments),
            )
            for index in range(max(0, terrain_box_count)):
                box_name = f"terrain_box_{index:02d}"
                for geom2 in direct_foot_geoms:
                    template = terrain_template_for_geom(geom2)
                    if template is not None:
                        add_pair_from_template(box_name, geom2, template)
                    else:
                        add_explicit_pair(box_name, geom2)

    temp = tempfile.NamedTemporaryFile(
        mode="wb",
        suffix="_terrain_course.xml",
        prefix=f"{source_xml.stem}_",
        dir=str(source_xml.parent),
        delete=False,
    )
    temp_path = Path(temp.name)
    try:
        tree.write(temp, encoding="utf-8", xml_declaration=False)
    finally:
        temp.close()
    return temp_path, True

def apply_body_inertial_overrides(
    model: mujoco.MjModel,
    config: dict[str, Any],
) -> bool:
    """Apply explicit body inertial corrections before MJWarp model creation."""
    overrides = config.get("model", {}).get("body_inertial_overrides", {})
    if not isinstance(overrides, dict):
        raise TypeError("model.body_inertial_overrides must be an object")
    changed = False
    for body_name, spec in overrides.items():
        if not isinstance(spec, dict):
            raise TypeError(f"body inertial override for {body_name!r} must be an object")
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(body_name))
        if body_id < 0:
            raise KeyError(f"body inertial override references missing body: {body_name}")
        source_name = spec.get("copy_from")
        if source_name is not None:
            source_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                str(source_name),
            )
            if source_id < 0:
                raise KeyError(
                    f"body inertial override references missing source body: {source_name}"
                )
            model.body_mass[body_id] = model.body_mass[source_id]
            model.body_inertia[body_id] = model.body_inertia[source_id]
            model.body_ipos[body_id] = model.body_ipos[source_id]
            model.body_iquat[body_id] = model.body_iquat[source_id]
            changed = True
        for key, target in (
            ("mass", model.body_mass),
            ("inertia", model.body_inertia),
            ("ipos", model.body_ipos),
            ("iquat", model.body_iquat),
        ):
            if key in spec:
                target[body_id] = spec[key]
                changed = True
    return changed


def build_muscle_model(config: dict[str, Any]) -> tuple[mujoco.MjModel, mujoco.MjData]:
    xml_path, temporary_xml = source_xml_with_stair_box_contacts(config)
    try:
        model = mujoco.MjModel.from_xml_path(str(xml_path))
    finally:
        if temporary_xml:
            try:
                xml_path.unlink()
            except OSError:
                pass
    inertials_modified = apply_body_inertial_overrides(model, config)
    configure_hfield_course(model, config)
    torque_action_cfg = config.get("torque_action", {})
    direct_torque_mode = (
        isinstance(torque_action_cfg, dict)
        and bool(torque_action_cfg.get("enabled", False))
        and str(torque_action_cfg.get("mode", "allocator")).lower() == "direct"
    )
    if direct_torque_mode and bool(torque_action_cfg.get("disable_model_actuators", True)):
        # Preserve the compiled activation-state layout for observation compatibility.
        # Active muscle force is removed; passive muscle-tendon bias can remain.
        model.actuator_gaintype[:] = mujoco.mjtGain.mjGAIN_FIXED
        model.actuator_gainprm[:] = 0.0
        passive_count = int(model.na) if bool(
            torque_action_cfg.get("preserve_passive_muscle_forces", False)
        ) else 0
        model.actuator_biastype[passive_count:] = mujoco.mjtBias.mjBIAS_NONE
        model.actuator_biasprm[passive_count:] = 0.0
    direct_exo_cfg = config.get("model", {}).get("exo_direct_hip_motor", {})
    if (
        not direct_torque_mode
        and isinstance(direct_exo_cfg, dict)
        and bool(direct_exo_cfg.get("enabled", False))
    ):
        max_torque_nm = max(0.0, float(direct_exo_cfg.get("max_torque_nm", 10.0)))
        if max_torque_nm <= 0.0:
            raise ValueError("model.exo_direct_hip_motor.max_torque_nm must be positive")
        for actuator_name, joint_name in (
            ("Exo_R", "hip_flexion_r"),
            ("Exo_L", "hip_flexion_l"),
        ):
            actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if actuator_id < 0 or joint_id < 0:
                raise KeyError(f"missing direct Exo actuator/joint pair: {actuator_name}/{joint_name}")
            model.actuator_trntype[actuator_id] = mujoco.mjtTrn.mjTRN_JOINT
            model.actuator_trnid[actuator_id] = (-1, -1)
            model.actuator_trnid[actuator_id, 0] = joint_id
            model.actuator_gear[actuator_id] = 0.0
            model.actuator_gear[actuator_id, 0] = max_torque_nm
            model.actuator_ctrlrange[actuator_id] = (-1.0, 1.0)
            model.actuator_ctrllimited[actuator_id] = 1
            model.actuator_dyntype[actuator_id] = mujoco.mjtDyn.mjDYN_NONE
            model.actuator_gaintype[actuator_id] = mujoco.mjtGain.mjGAIN_FIXED
            model.actuator_biastype[actuator_id] = mujoco.mjtBias.mjBIAS_NONE
            model.actuator_dynprm[actuator_id] = 0.0
            model.actuator_gainprm[actuator_id] = 0.0
            model.actuator_gainprm[actuator_id, 0] = 1.0
            model.actuator_biasprm[actuator_id] = 0.0
    physics_hz = float(config.get("control", {}).get("physics_hz", 0.0) or 0.0)
    if physics_hz > 0.0:
        model.opt.timestep = 1.0 / physics_hz
    if bool(config["model"].get("disable_multiccd", True)):
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
    if (
        str(config.get("reward_mode", "")).lower() == "myoassist_exact"
        and bool(config.get("myoassist_exact", {}).get("match_myoassist_model_options", True))
    ):
        model.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_WARMSTART)
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL)
    data = mujoco.MjData(model)
    if inertials_modified:
        mujoco.mj_setConst(model, data)
    key_name = str(config["reset"].get("keyframe", ""))
    kid = key_id_or_none(model, key_name) if key_name else None
    if kid is not None:
        mujoco.mj_resetDataKeyframe(model, data, kid)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    return model, data

def stair_box_treads(segment: dict[str, Any]) -> list[tuple[float, float, float]]:
    step_height = max(float(segment.get("step_height", 0.127)), 0.0)
    step_depth = max(float(segment.get("step_depth", 0.32)), 1e-6)
    direction = 1.0 if float(segment.get("direction", 1.0)) >= 0.0 else -1.0
    steps = max(1, int(segment.get("steps", 2)))
    start_x = float(segment.get("x0", 0.0))
    base_height_key = (
        "source_base_height"
        if direction < 0.0 and "source_base_height" in segment
        else "base_height"
    )
    base_height = float(
        segment.get(
            base_height_key,
            steps * step_height if direction < 0.0 else 0.0,
        )
    )
    platform_depth = max(float(segment.get("platform_depth", 0.8)), 0.0)
    treads: list[tuple[float, float, float]] = []
    if direction > 0.0:
        for index in range(steps):
            seg_x0 = start_x + index * step_depth
            treads.append((seg_x0, seg_x0 + step_depth, base_height + (index + 1) * step_height))
        if platform_depth > 0.0:
            top_x0 = start_x + steps * step_depth
            top_height = float(segment.get("top_platform_height", base_height + steps * step_height))
            treads.append((top_x0, top_x0 + platform_depth, top_height))
    else:
        if platform_depth > 0.0:
            treads.append((start_x - platform_depth, start_x, base_height))
        for index in range(steps):
            seg_x0 = start_x + index * step_depth
            treads.append(
                (
                    seg_x0,
                    seg_x0 + step_depth,
                    max(0.0, base_height - (index + 1) * step_height),
                )
            )
    return treads

def stair_box_riser_walls(segment: dict[str, Any], thickness: float) -> list[tuple[float, float, float]]:
    step_height = max(float(segment.get("step_height", 0.127)), 0.0)
    step_depth = max(float(segment.get("step_depth", 0.32)), 1e-6)
    direction = 1.0 if float(segment.get("direction", 1.0)) >= 0.0 else -1.0
    steps = max(1, int(segment.get("steps", 2)))
    start_x = float(segment.get("x0", 0.0))
    base_height_key = (
        "source_base_height"
        if direction < 0.0 and "source_base_height" in segment
        else "base_height"
    )
    base_height = float(
        segment.get(
            base_height_key,
            steps * step_height if direction < 0.0 else 0.0,
        )
    )
    thickness = max(float(thickness), 1e-4)
    walls: list[tuple[float, float, float]] = []
    if direction > 0.0:
        for index in range(steps):
            face_x = start_x + index * step_depth
            walls.append((face_x - 0.5 * thickness, face_x + 0.5 * thickness, base_height + (index + 1) * step_height))
    else:
        for index in range(steps):
            face_x = start_x + (index + 1) * step_depth
            height = max(0.0, base_height - index * step_height)
            walls.append((face_x - 0.5 * thickness, face_x + 0.5 * thickness, height))
    return walls

def course_height_np(x: np.ndarray, segments: list[dict[str, Any]]) -> np.ndarray:
    height = np.zeros_like(x, dtype=np.float64)
    for segment in segments:
        x0 = float(segment.get("x0", -np.inf))
        x1 = float(segment.get("x1", np.inf))
        mask = (x >= x0) & (x <= x1)
        kind = str(segment.get("type", "flat"))
        if kind in {"flat", "flat_box"}:
            value = float(segment.get("height", 0.0))
            height[mask] = value
        elif kind in {"slope", "ramp_box"}:
            height0 = float(segment.get("height0", 0.0))
            if "height1" in segment and np.isfinite(x0) and np.isfinite(x1) and x1 != x0:
                slope = (float(segment["height1"]) - height0) / (x1 - x0)
            else:
                slope = float(segment.get("slope", 0.0))
            height[mask] = height0 + slope * (x[mask] - x0)
        elif kind == "stairs":
            step_height = max(float(segment.get("step_height", 0.127)), 1e-6)
            step_depth = max(float(segment.get("step_depth", 0.32)), 1e-6)
            direction = 1.0 if float(segment.get("direction", 1.0)) >= 0.0 else -1.0
            steps = max(1, int(segment.get("steps", 4)))
            base_height = float(segment.get("base_height", 0.0))
            progressed = np.maximum(x[mask] - x0, 0.0)
            step_index = np.clip(np.floor(progressed / step_depth), 0, steps)
            height[mask] = base_height + direction * step_index * step_height
        elif kind == "stairs_box":
            for tread_x0, tread_x1, tread_height in stair_box_treads(segment):
                tread_mask = (x >= tread_x0) & (x <= tread_x1)
                height[tread_mask] = tread_height
    return height

def merge_stair_box_treads(
    treads: Sequence[tuple[float, float, float]],
    *,
    height_tol: float = 1e-6,
    gap_tol: float = 1e-6,
) -> list[tuple[float, float, float]]:
    ordered = sorted(
        [(float(x0), float(x1), float(height)) for x0, x1, height in treads if float(x1) > float(x0) and float(height) > 1e-6],
        key=lambda item: (item[0], item[1], item[2]),
    )
    merged: list[tuple[float, float, float]] = []
    for x0, x1, height in ordered:
        if not merged:
            merged.append((x0, x1, height))
            continue
        prev_x0, prev_x1, prev_height = merged[-1]
        if abs(height - prev_height) <= height_tol and x0 <= prev_x1 + gap_tol:
            merged[-1] = (prev_x0, max(prev_x1, x1), prev_height)
        else:
            merged.append((x0, x1, height))
    return merged

def terrain_height_np_from_params(x: np.ndarray, terrain_type_id: int, params: Sequence[float]) -> np.ndarray:
    values = np.asarray(params, dtype=np.float64)
    out = np.zeros_like(x, dtype=np.float64)
    if int(terrain_type_id) == 1:
        slope = float(values[0]) if values.size > 0 else 0.0
        anchor_x = float(values[1]) if values.size > 1 else 0.0
        anchor_height = float(values[2]) if values.size > 2 else 0.0
        return anchor_height + (x - anchor_x) * slope
    if int(terrain_type_id) == 2:
        step_height = max(float(values[0]) if values.size > 0 else 0.127, 0.0)
        step_depth = max(float(values[1]) if values.size > 1 else 0.32, 1e-6)
        direction = 1.0 if (float(values[2]) if values.size > 2 else 1.0) >= 0.0 else -1.0
        steps = max(1, int(round(float(values[3]) if values.size > 3 else 2.0)))
        start_x = float(values[4]) if values.size > 4 else 0.0
        base_height = float(values[5]) if values.size > 5 else (steps * step_height if direction < 0.0 else 0.0)
        platform_depth = max(float(values[6]) if values.size > 6 else 0.8, 0.0)
        if direction > 0.0:
            progressed = x - start_x
            active = progressed >= 0.0
            step_index = np.clip(np.floor(np.maximum(progressed, 0.0) / step_depth), 0, steps - 1)
            top = base_height + (step_index + 1.0) * step_height
            if platform_depth > 0.0:
                top = np.where(progressed >= steps * step_depth, base_height + steps * step_height, top)
            return np.where(active, top, out)
        top = np.zeros_like(x, dtype=np.float64)
        if platform_depth > 0.0:
            platform = (x >= start_x - platform_depth) & (x < start_x)
            top = np.where(platform, base_height, top)
        stair = (x >= start_x) & (x <= start_x + steps * step_depth)
        step_index = np.clip(np.floor(np.maximum(x - start_x, 0.0) / step_depth), 0, steps - 1)
        stair_height = np.maximum(0.0, base_height - step_index * step_height)
        return np.where(stair, stair_height, top)
    return out

def parse_terrain_type_and_params(metadata: dict[str, Any]) -> tuple[int, list[float]]:
    terrain_type = str(metadata.get("terrain_type", "flat") or "flat")
    raw = str(metadata.get("terrain_params", "") or "")
    values = [float(item) for item in raw.split()] if raw.strip() else []
    params = [0.0] * 7
    if terrain_type == "slope":
        params[: min(len(values), 3)] = values[:3]
        return 1, params
    if terrain_type == "stairs_box":
        params[: min(len(values), 7)] = values[:7]
        return 2, params
    return 0, params


def source_terrain_height_np(metadata: dict[str, Any], x: np.ndarray) -> np.ndarray:
    source_segments = metadata.get("source_terrain_segments")
    if isinstance(source_segments, list) and source_segments:
        return course_height_np(x, source_segments)
    course_segments = metadata.get("terrain_course_segments")
    if isinstance(course_segments, list) and course_segments:
        return course_height_np(x, course_segments)
    source_metadata = metadata.get("source_metadata")
    if isinstance(source_metadata, dict):
        source_segments = source_metadata.get("source_terrain_segments")
        if isinstance(source_segments, list) and source_segments:
            return course_height_np(x, source_segments)
        course_segments = source_metadata.get("terrain_course_segments")
        if isinstance(course_segments, list) and course_segments:
            return course_height_np(x, course_segments)
    terrain_type, terrain_params = parse_terrain_type_and_params(metadata)
    return terrain_height_np_from_params(x, terrain_type, terrain_params)

def configure_hfield_course(model: mujoco.MjModel, config: dict[str, Any]) -> None:
    """Disable the legacy hfield; physical terrain is provided by generated box geoms."""
    course_cfg = config.get("terrain_course", {})
    if not bool(course_cfg.get("enabled", False)) or int(model.nhfield) <= 0:
        return
    hfield_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, str(course_cfg.get("hfield_name", "terrain")))
    if hfield_id >= 0:
        adr = int(model.hfield_adr[hfield_id])
        count = int(model.hfield_nrow[hfield_id]) * int(model.hfield_ncol[hfield_id])
        model.hfield_data[adr : adr + count] = 0.0
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, str(course_cfg.get("terrain_geom", "terrain")))
    if geom_id >= 0:
        model.geom_rgba[geom_id, 3] = 0.0
        model.geom_contype[geom_id] = 0
        model.geom_conaffinity[geom_id] = 0
    ground_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, str(course_cfg.get("ground_geom", "ground-plane")))
    if ground_id >= 0 and bool(course_cfg.get("disable_ground_plane_contact", False)):
        model.geom_rgba[ground_id, 3] = float(course_cfg.get("ground_plane_alpha", 0.0))
        model.geom_contype[ground_id] = 0
        model.geom_conaffinity[ground_id] = 0
