#!/usr/bin/env python3
"""CleanRL-style PPO for MJWarp batched 22-muscle MyoAssist training."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Sequence

import imageio.v2 as imageio
import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import warp as wp
from torch.distributions.normal import Normal


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_REFERENCE_PATH = Path("/home/lzn/myoassist/rl_train/reference_data/short_reference_gait.npz")
TRACK_JOINTS = [
    "pelvis_ty",
    "pelvis_tilt",
    "hip_flexion_r",
    "knee_angle_r",
    "ankle_angle_r",
    "mtp_angle_r",
    "hip_flexion_l",
    "knee_angle_l",
    "ankle_angle_l",
    "mtp_angle_l",
]
RESET_JOINTS = ["pelvis_tx", *TRACK_JOINTS]
FOOT_SITE_NAMES = ["r_heel_btm", "r_toe_btm", "l_heel_btm", "l_toe_btm"]
EMG_COLUMNS = ["HAB", "HAD", "HFL", "GLU", "HAM", "RF", "VAS", "BFSH", "GAS", "SOL", "TA"]
EMG_ACTUATOR_MAP = {
    "hamstrings": "HAM",
    "bifemsh": "BFSH",
    "glutmax": "GLU",
    "iliopsoas": "HFL",
    "rectfem": "RF",
    "vasti": "VAS",
    "gastroc": "GAS",
    "soleus": "SOL",
    "tibant": "TA",
}


def muscle_action_to_activation(action: torch.Tensor) -> torch.Tensor:
    normalized_action = torch.clamp(action, -1.0, 1.0)
    return 0.5 * (normalized_action + 1.0)


def activation_to_muscle_action(activation: np.ndarray) -> np.ndarray:
    return 2.0 * np.clip(activation, 0.0, 1.0) - 1.0


class Agent(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, logstd_init: float, initial_action_mean: float):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )
        self.actor_mean = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, act_dim),
        )
        self.actor_logstd = nn.Parameter(torch.full((1, act_dim), float(logstd_init)))
        self.initialize_actor_mean(initial_action_mean)

    def initialize_actor_mean(self, initial_action_mean: float) -> None:
        final_layer = self.actor_mean[-1]
        if not isinstance(final_layer, nn.Linear):
            raise TypeError("Expected actor_mean final layer to be nn.Linear")
        nn.init.zeros_(final_layer.weight)
        nn.init.constant_(final_layer.bias, float(initial_action_mean))

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)

    def get_action_and_value(
        self,
        x: torch.Tensor,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = self.actor_mean(x)
        logstd = self.actor_logstd.expand_as(mean)
        dist = Normal(mean, torch.exp(logstd))
        if action is None:
            action = mean if deterministic else dist.sample()
        return action, dist.log_prob(action).sum(1), dist.entropy().sum(1), self.critic(x)


class ObsNormalizer:
    def __init__(self, obs_dim: int, device: torch.device, *, enabled: bool = True, clip: float = 10.0, eps: float = 1e-4):
        self.enabled = enabled
        self.clip = float(clip)
        self.eps = float(eps)
        self.mean = torch.zeros(obs_dim, dtype=torch.float32, device=device)
        self.var = torch.ones(obs_dim, dtype=torch.float32, device=device)
        self.count = torch.tensor(float(eps), dtype=torch.float32, device=device)

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        if not self.enabled:
            return
        batch = x.detach()
        batch_mean = torch.mean(batch, dim=0)
        batch_var = torch.var(batch, dim=0, unbiased=False)
        batch_count = torch.tensor(float(batch.shape[0]), dtype=torch.float32, device=batch.device)
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        correction = torch.square(delta) * self.count * batch_count / total_count
        new_var = (m_a + m_b + correction) / total_count
        self.mean.copy_(new_mean)
        self.var.copy_(torch.clamp(new_var, min=1e-6))
        self.count.copy_(total_count)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        normalized = (x - self.mean) / torch.sqrt(self.var + self.eps)
        return torch.clamp(normalized, -self.clip, self.clip)

    def state_dict(self) -> dict[str, torch.Tensor | bool | float]:
        return {
            "enabled": self.enabled,
            "clip": self.clip,
            "eps": self.eps,
            "mean": self.mean.detach().clone(),
            "var": self.var.detach().clone(),
            "count": self.count.detach().clone(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.enabled = bool(state.get("enabled", self.enabled))
        self.clip = float(state.get("clip", self.clip))
        self.eps = float(state.get("eps", self.eps))
        mean = state["mean"].to(self.mean.device)
        var = state["var"].to(self.var.device)
        n = min(int(mean.numel()), int(self.mean.numel()))
        self.mean.zero_()
        self.var.fill_(1.0)
        self.mean[:n].copy_(mean[:n])
        self.var[:n].copy_(var[:n])
        self.count.copy_(state["count"].to(self.count.device))


class BcReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int, device: torch.device):
        self.capacity = int(capacity)
        self.device = device
        self.obs = torch.empty((self.capacity, obs_dim), dtype=torch.float32, device=device)
        self.target_action = torch.empty((self.capacity, act_dim), dtype=torch.float32, device=device)
        self.target_activation = torch.empty((self.capacity, act_dim), dtype=torch.float32, device=device)
        self.emg_activation = torch.empty((self.capacity, act_dim), dtype=torch.float32, device=device)
        self.next_idx = 0
        self.size = 0

    @torch.no_grad()
    def add(
        self,
        obs: torch.Tensor,
        target_action: torch.Tensor,
        target_activation: torch.Tensor,
        emg_activation: torch.Tensor,
    ) -> None:
        n = int(obs.shape[0])
        if n <= 0:
            return
        if n >= self.capacity:
            self.obs.copy_(obs[-self.capacity :].detach())
            self.target_action.copy_(target_action[-self.capacity :].detach())
            self.target_activation.copy_(target_activation[-self.capacity :].detach())
            self.emg_activation.copy_(emg_activation[-self.capacity :].detach())
            self.next_idx = 0
            self.size = self.capacity
            return
        first = min(n, self.capacity - self.next_idx)
        dst = slice(self.next_idx, self.next_idx + first)
        self.obs[dst].copy_(obs[:first].detach())
        self.target_action[dst].copy_(target_action[:first].detach())
        self.target_activation[dst].copy_(target_activation[:first].detach())
        self.emg_activation[dst].copy_(emg_activation[:first].detach())
        remain = n - first
        if remain > 0:
            self.obs[:remain].copy_(obs[first:].detach())
            self.target_action[:remain].copy_(target_action[first:].detach())
            self.target_activation[:remain].copy_(target_activation[first:].detach())
            self.emg_activation[:remain].copy_(emg_activation[first:].detach())
        self.next_idx = (self.next_idx + n) % self.capacity
        self.size = min(self.capacity, self.size + n)

    def sample(self, n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.size <= 0:
            raise RuntimeError("Cannot sample from an empty BC replay buffer")
        idx = torch.randint(0, self.size, (int(n),), device=self.device)
        return self.obs[idx], self.target_action[idx], self.target_activation[idx], self.emg_activation[idx]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def name_id(model: mujoco.MjModel, objtype: mujoco.mjtObj, name: str) -> int:
    idx = mujoco.mj_name2id(model, objtype, name)
    if idx < 0:
        raise KeyError(f"Missing MuJoCo object {objtype}: {name}")
    return int(idx)


def joint_id(model: mujoco.MjModel, name: str) -> int:
    return name_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def site_id(model: mujoco.MjModel, name: str) -> int:
    return name_id(model, mujoco.mjtObj.mjOBJ_SITE, name)


def key_id_or_none(model: mujoco.MjModel, name: str) -> int | None:
    idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name)
    return int(idx) if idx >= 0 else None


def coordinate_names(model: mujoco.MjModel, *, kind: str) -> list[str]:
    if kind == "qpos":
        names = [f"qpos_{i}" for i in range(model.nq)]
        adr_array = model.jnt_qposadr
        size = model.nq
    elif kind == "qvel":
        names = [f"qvel_{i}" for i in range(model.nv)]
        adr_array = model.jnt_dofadr
        size = model.nv
    else:
        raise ValueError(kind)
    for jid in range(model.njnt):
        adr = int(adr_array[jid])
        if 0 <= adr < size:
            names[adr] = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or names[adr]
    return names


def top_abs_named(values: np.ndarray, names: list[str]) -> tuple[str, float]:
    if values.size == 0:
        return "", 0.0
    idx = int(np.argmax(np.abs(values)))
    return names[idx] if idx < len(names) else str(idx), float(values[idx])


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
    for qpos1, qpos2, qvel1, qvel2, poly in joint_equality_specs_np(model):
        q = float(data.qpos[qpos2])
        dq = float(data.qvel[qvel2])
        data.qpos[qpos1] = float(poly[0] + poly[1] * q + poly[2] * q**2 + poly[3] * q**3 + poly[4] * q**4)
        derivative = float(poly[1] + 2.0 * poly[2] * q + 3.0 * poly[3] * q**2 + 4.0 * poly[4] * q**3)
        data.qvel[qvel1] = derivative * dq


def source_xml_with_stair_box_contacts(config: dict[str, Any]) -> tuple[Path, bool]:
    source_xml = Path(config["model"]["source_xml"]).expanduser()
    course_cfg = config.get("terrain_course", {})
    if not bool(course_cfg.get("enabled", False)) or not bool(course_cfg.get("add_stair_box_contact_pairs", True)):
        return source_xml, False

    tree = ET.parse(source_xml)
    root = tree.getroot()
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")

    terrain_pairs = [
        pair
        for pair in list(contact.findall("pair"))
        if str(pair.get("geom1", "")) == str(course_cfg.get("terrain_geom", "terrain"))
    ]
    if not terrain_pairs:
        return source_xml, False

    existing = {(pair.get("geom1", ""), pair.get("geom2", "")) for pair in contact.findall("pair")}

    def numeric_attr(values: Any) -> str:
        if isinstance(values, str):
            return values
        return " ".join(f"{float(value):g}" for value in values)

    def pair_friction_attr(values: Any) -> str:
        parsed = [float(value) for value in (values.split() if isinstance(values, str) else list(values))]
        if len(parsed) == 3:
            parsed = [parsed[0], parsed[0], parsed[1], parsed[2], parsed[2]]
        return numeric_attr(parsed[:5])

    stair_box_match_terrain_contact = bool(course_cfg.get("stair_box_match_terrain_contact", True))
    disable_ground_plane_contact = bool(course_cfg.get("disable_ground_plane_contact", False))

    def add_pair_from_template(geom1: str, geom2: str, template: ET.Element) -> None:
        key = (geom1, geom2)
        if key in existing:
            return
        attrs = dict(template.attrib)
        attrs["geom1"] = geom1
        attrs["geom2"] = geom2
        if geom1.startswith("terrain_stair_box_") and not stair_box_match_terrain_contact:
            if "stair_box_contact_margin" in course_cfg:
                attrs["margin"] = f"{float(course_cfg['stair_box_contact_margin']):g}"
            if "stair_box_contact_solref" in course_cfg:
                attrs["solref"] = numeric_attr(course_cfg["stair_box_contact_solref"])
            if "stair_box_contact_solimp" in course_cfg:
                attrs["solimp"] = numeric_attr(course_cfg["stair_box_contact_solimp"])
            if "stair_box_contact_friction" in course_cfg:
                attrs["friction"] = pair_friction_attr(course_cfg["stair_box_contact_friction"])
        ET.SubElement(contact, "pair", attrs)
        existing.add(key)

    extra_foot_geoms = [
        str(name)
        for name in course_cfg.get("extra_foot_contact_geoms", [])
        if str(name)
    ]
    terrain_geom = str(course_cfg.get("terrain_geom", "terrain"))
    ground_geom = str(course_cfg.get("ground_geom", "ground-plane"))
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

    temp = tempfile.NamedTemporaryFile(
        mode="wb",
        suffix="_stairbox_contacts.xml",
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
    configure_hfield_course(model, config)
    if bool(config["model"].get("disable_multiccd", True)):
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
    data = mujoco.MjData(model)
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
    base_height = float(segment.get("base_height", steps * step_height if direction < 0.0 else 0.0))
    platform_depth = max(float(segment.get("platform_depth", 0.8)), 0.0)
    treads: list[tuple[float, float, float]] = []
    if direction > 0.0:
        for index in range(steps):
            seg_x0 = start_x + index * step_depth
            treads.append((seg_x0, seg_x0 + step_depth, base_height + (index + 1) * step_height))
        if platform_depth > 0.0:
            top_x0 = start_x + steps * step_depth
            treads.append((top_x0, top_x0 + platform_depth, base_height + steps * step_height))
    else:
        if platform_depth > 0.0:
            treads.append((start_x - platform_depth, start_x, base_height))
        for index in range(steps):
            seg_x0 = start_x + index * step_depth
            treads.append((seg_x0, seg_x0 + step_depth, max(0.0, base_height - index * step_height)))
    return treads


def stair_box_riser_walls(segment: dict[str, Any], thickness: float) -> list[tuple[float, float, float]]:
    step_height = max(float(segment.get("step_height", 0.127)), 0.0)
    step_depth = max(float(segment.get("step_depth", 0.32)), 1e-6)
    direction = 1.0 if float(segment.get("direction", 1.0)) >= 0.0 else -1.0
    steps = max(1, int(segment.get("steps", 2)))
    start_x = float(segment.get("x0", 0.0))
    base_height = float(segment.get("base_height", steps * step_height if direction < 0.0 else 0.0))
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
        if kind == "flat":
            value = float(segment.get("height", 0.0))
            height[mask] = value
        elif kind == "slope":
            slope = float(segment.get("slope", 0.0))
            height0 = float(segment.get("height0", 0.0))
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


def course_hfield_height_np(
    x: np.ndarray,
    segments: list[dict[str, Any]],
    *,
    include_stairs_box: bool = False,
) -> np.ndarray:
    hfield_segments: list[dict[str, Any]] = []
    for segment in segments:
        if str(segment.get("type", "flat")) == "stairs_box" and not bool(include_stairs_box):
            continue
        hfield_segments.append(segment)
    return course_height_np(x, hfield_segments)


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


def source_terrain_height_np(metadata: dict[str, Any], x: np.ndarray) -> np.ndarray:
    source_segments = metadata.get("source_terrain_segments")
    if isinstance(source_segments, list) and source_segments:
        return course_height_np(x, source_segments)
    course_segments = metadata.get("terrain_course_segments")
    if isinstance(course_segments, list) and course_segments:
        return course_height_np(x, course_segments)
    terrain_type, terrain_params = parse_terrain_type_and_params(metadata)
    return terrain_height_np_from_params(x, terrain_type, terrain_params)


def hide_stair_box_geoms(model: mujoco.MjModel) -> list[int]:
    geom_ids: list[int] = []
    for index in range(64):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"terrain_stair_box_{index:02d}")
        if geom_id < 0:
            break
        geom_ids.append(int(geom_id))
        model.geom_pos[geom_id] = np.array([0.0, 0.0, -100.0], dtype=np.float64)
        model.geom_size[geom_id] = np.array([0.01, 0.01, 0.01], dtype=np.float64)
        model.geom_rgba[geom_id] = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32)
    return geom_ids


def set_stair_box_geom(
    model: mujoco.MjModel,
    geom_id: int,
    x0: float,
    x1: float,
    height: float,
    half_width: float,
    rgba: Sequence[float],
) -> None:
    if x1 <= x0 or height <= 1e-6:
        model.geom_pos[geom_id] = np.array([0.0, 0.0, -100.0], dtype=np.float64)
        model.geom_size[geom_id] = np.array([0.01, 0.01, 0.01], dtype=np.float64)
        model.geom_rgba[geom_id] = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32)
        return
    model.geom_pos[geom_id] = np.array([(float(x0) + float(x1)) * 0.5, 0.0, float(height) * 0.5], dtype=np.float64)
    model.geom_size[geom_id] = np.array([(float(x1) - float(x0)) * 0.5, float(half_width), float(height) * 0.5], dtype=np.float64)
    model.geom_rgba[geom_id] = np.array(rgba, dtype=np.float32)
    if hasattr(model, "geom_matid"):
        model.geom_matid[geom_id] = -1


def configure_stair_box_course(model: mujoco.MjModel, course_cfg: dict[str, Any], hfield_half_width: float) -> None:
    box_geom_ids = hide_stair_box_geoms(model)
    box_segments = [
        segment
        for segment in list(course_cfg.get("segments", []))
        if str(segment.get("type", "flat")) == "stairs_box"
    ]
    if not box_segments:
        return
    if not box_geom_ids:
        raise ValueError("terrain_course has stairs_box segments but model has no terrain_stair_box_XX geoms")
    half_width = float(course_cfg.get("stair_box_half_width", min(5.0, float(hfield_half_width))))
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
    generated: list[tuple[float, float, float]] = []
    generated_walls: list[tuple[float, float, float]] = []
    for segment in box_segments:
        generated.extend(stair_box_treads(segment))
        if bool(course_cfg.get("stair_box_add_riser_walls", False)):
            generated_walls.extend(
                stair_box_riser_walls(segment, float(course_cfg.get("stair_box_riser_wall_thickness", 0.025)))
            )
    collision_treads = merge_stair_box_treads(generated)
    collision_boxes = collision_treads + generated_walls
    if len(collision_boxes) > len(box_geom_ids):
        raise ValueError(
            f"terrain_course stairs_box needs {len(collision_boxes)} box geoms but model only has {len(box_geom_ids)}"
        )
    for index, generated_segment in enumerate(collision_boxes):
        rgba = palette[index % len(palette)]
        set_stair_box_geom(model, box_geom_ids[index], *generated_segment, half_width, rgba)


def configure_hfield_course(model: mujoco.MjModel, config: dict[str, Any]) -> None:
    course_cfg = config.get("terrain_course", {})
    if not bool(course_cfg.get("enabled", False)):
        return
    if int(model.nhfield) <= 0:
        raise ValueError("terrain_course enabled but model has no hfield")
    hfield_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, str(course_cfg.get("hfield_name", "terrain")))
    if hfield_id < 0:
        raise ValueError("terrain_course enabled but hfield 'terrain' is missing")
    if "hfield_size_z" in course_cfg:
        model.hfield_size[hfield_id, 2] = max(float(course_cfg["hfield_size_z"]), 1e-6)
    nrow = int(model.hfield_nrow[hfield_id])
    ncol = int(model.hfield_ncol[hfield_id])
    adr = int(model.hfield_adr[hfield_id])
    size = model.hfield_size[hfield_id].copy()
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, str(course_cfg.get("terrain_geom", "terrain")))
    if geom_id >= 0:
        model.geom_rgba[geom_id] = np.array(course_cfg.get("terrain_rgba", [1.0, 1.0, 1.0, 1.0]), dtype=np.float32)
        model.geom_pos[geom_id, 2] = float(course_cfg.get("terrain_geom_z", 0.0))
    center_x = float(model.geom_pos[geom_id, 0]) if geom_id >= 0 else 0.0
    tile_size_col = 2.0 * float(size[0]) / max(int(ncol), 1)
    x = center_x - float(size[0]) + np.arange(ncol, dtype=np.float64) * tile_size_col
    stairs_box_as_hfield = bool(course_cfg.get("stairs_box_as_hfield", False))
    if stairs_box_as_hfield:
        hide_stair_box_geoms(model)
    else:
        configure_stair_box_course(model, course_cfg, float(size[1]))
    heights = course_hfield_height_np(
        x,
        list(course_cfg.get("segments", [])),
        include_stairs_box=stairs_box_as_hfield,
    )
    normalized = np.clip(heights / max(float(size[2]), 1e-6), 0.0, 1.0)
    model.hfield_data[adr : adr + nrow * ncol] = np.tile(normalized[None, :], (nrow, 1)).reshape(-1)
    ground_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, str(course_cfg.get("ground_geom", "ground-plane")))
    if ground_id >= 0:
        model.geom_rgba[ground_id, 3] = float(course_cfg.get("ground_plane_alpha", 0.0))
        if bool(course_cfg.get("disable_ground_plane_contact", False)):
            model.geom_contype[ground_id] = 0
            model.geom_conaffinity[ground_id] = 0
        if bool(course_cfg.get("lower_ground_plane", False)):
            model.geom_pos[ground_id, 2] = float(course_cfg.get("ground_plane_z", -10.0))


def estimate_cycle_steps(q_ref_np: np.ndarray, control_hz: float) -> int:
    if q_ref_np.shape[0] < 8:
        return max(1, int(round(control_hz)))
    x = q_ref_np.astype(np.float64)
    std = np.std(x, axis=0)
    keep = std > 1e-6
    if not np.any(keep):
        return max(1, int(round(control_hz)))
    x = x[:, keep]
    x = (x - np.mean(x, axis=0)) / np.maximum(np.std(x, axis=0), 1e-6)
    min_lag = max(2, int(round(0.5 * control_hz)))
    max_lag = min(x.shape[0] // 2, max(min_lag + 1, int(round(3.0 * control_hz))))
    best_lag = max(1, int(round(control_hz)))
    best_score = -np.inf
    for lag in range(min_lag, max_lag + 1):
        score = float(np.mean(np.sum(x[:-lag] * x[lag:], axis=1) / x.shape[1]))
        if score > best_score:
            best_lag = lag
            best_score = score
    return max(1, int(best_lag))


def resample_periodic_profile(profile: np.ndarray, target_length: int) -> np.ndarray:
    target_length = max(1, int(target_length))
    source = np.asarray(profile, dtype=np.float32)
    x_source = np.arange(source.shape[0] + 1, dtype=np.float32)
    wrapped = np.concatenate([source, source[:1]], axis=0)
    x_target = np.arange(target_length, dtype=np.float32) * (float(source.shape[0]) / float(target_length))
    out = np.zeros((target_length, source.shape[1]), dtype=np.float32)
    for col in range(source.shape[1]):
        out[:, col] = np.interp(x_target, x_source, wrapped[:, col])
    return out


def load_activation_prior(
    config: dict[str, Any],
    model: mujoco.MjModel,
    reference_length: int,
    q_ref_np: np.ndarray,
    control_hz: float,
) -> dict[str, Any]:
    prior_cfg = config.get("activation_prior", {})
    path = str(prior_cfg.get("path", "") or "")
    mask = np.zeros(model.nu, dtype=bool)
    activation_table = np.zeros((reference_length, model.nu), dtype=np.float32)
    metadata: dict[str, Any] = {
        "enabled": False,
        "path": path,
        "cycle_steps": 0,
        "supervised_actuators": [],
    }
    if not bool(prior_cfg.get("enabled", False)) or not path:
        return {
            "activation": activation_table,
            "action": activation_to_muscle_action(activation_table).astype(np.float32),
            "mask": mask,
            "metadata": metadata,
        }

    expanded_path = Path(path).expanduser()
    if not expanded_path.is_absolute():
        expanded_path = (ROOT / expanded_path).resolve()
    if not expanded_path.exists():
        print(f"[activation_prior] missing file: {expanded_path}", flush=True)
        return {
            "activation": activation_table,
            "action": activation_to_muscle_action(activation_table).astype(np.float32),
            "mask": mask,
            "metadata": metadata,
        }

    if expanded_path.suffix == ".npz":
        data = np.load(expanded_path, allow_pickle=True)
        key = str(prior_cfg.get("key", "target_activations"))
        index_key = str(prior_cfg.get("index_key", "reference_indices"))
        actions = np.asarray(data[key], dtype=np.float32)
        ref_indices = np.asarray(data[index_key], dtype=np.int64)
        if actions.ndim != 2 or actions.shape[0] != ref_indices.shape[0] or actions.shape[1] != model.nu:
            raise ValueError(
                f"Invalid activation prior npz: {key}{actions.shape}, {index_key}{ref_indices.shape}, model.nu={model.nu}"
            )
        source_scale = str(prior_cfg.get("source_scale", "activation01"))
        if source_scale in {"minus1_plus1", "action_minus1_plus1", "control_minus1_plus1"}:
            actions = 0.5 * (actions + 1.0)
        actions = np.clip(actions, 0.0, 1.0)
        sums = np.zeros_like(activation_table)
        counts = np.zeros(reference_length, dtype=np.int64)
        for activation, ref_index in zip(actions, ref_indices):
            phase = int(ref_index) % int(reference_length)
            sums[phase] += activation
            counts[phase] += 1
        observed = counts > 0
        activation_table[observed] = sums[observed] / counts[observed, None]
        if np.any(observed) and not np.all(observed):
            observed_idx = np.flatnonzero(observed)
            for phase in np.flatnonzero(~observed):
                distances = np.minimum(np.abs(observed_idx - phase), reference_length - np.abs(observed_idx - phase))
                activation_table[phase] = activation_table[int(observed_idx[int(np.argmin(distances))])]
        elif not np.any(observed):
            raise ValueError(f"No observed phases in activation prior {expanded_path}")
        mask[:] = True
        metadata = {
            "enabled": True,
            "path": str(expanded_path),
            "format": "npz",
            "key": key,
            "index_key": index_key,
            "source_scale": source_scale,
            "observed_phases": int(np.sum(observed)),
            "supervised_actuators": [
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or str(i) for i in range(model.nu)
            ],
        }
        return {
            "activation": activation_table,
            "action": np.clip(activation_to_muscle_action(activation_table), -1.0, 1.0).astype(np.float32),
            "mask": mask,
            "metadata": metadata,
        }

    raw = np.genfromtxt(expanded_path, delimiter=",").astype(np.float32)
    if raw.ndim != 2 or raw.shape[1] != len(EMG_COLUMNS):
        raise ValueError(f"Expected EMG CSV shape [N,{len(EMG_COLUMNS)}], got {raw.shape} from {expanded_path}")
    raw = np.clip(raw, 0.0, 1.0)
    cycle_steps = int(prior_cfg.get("cycle_steps", 0) or 0)
    if cycle_steps <= 0:
        cycle_steps = estimate_cycle_steps(q_ref_np, control_hz)
    profile = resample_periodic_profile(raw, cycle_steps)
    emg_col = {name: i for i, name in enumerate(EMG_COLUMNS)}
    phase_offset = int(prior_cfg.get("phase_offset", 0) or 0)
    right_phase_offset = int(prior_cfg.get("right_phase_offset", 0) or 0)
    left_phase_offset = int(prior_cfg.get("left_phase_offset", cycle_steps // 2) or 0)

    supervised: list[str] = []
    for actuator_id in range(model.nu):
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id) or ""
        if actuator_name.endswith("_r"):
            base_name = actuator_name[:-2]
            side_offset = right_phase_offset
        elif actuator_name.endswith("_l"):
            base_name = actuator_name[:-2]
            side_offset = left_phase_offset
        else:
            continue
        emg_name = EMG_ACTUATOR_MAP.get(base_name)
        if emg_name is None:
            continue
        col = emg_col[emg_name]
        for phase in range(reference_length):
            row = (phase + phase_offset + side_offset) % cycle_steps
            activation_table[phase, actuator_id] = profile[row, col]
        mask[actuator_id] = True
        supervised.append(actuator_name)

    metadata = {
        "enabled": bool(np.any(mask)),
        "path": str(expanded_path),
        "cycle_steps": int(cycle_steps),
        "phase_offset": int(phase_offset),
        "right_phase_offset": int(right_phase_offset),
        "left_phase_offset": int(left_phase_offset),
        "columns": list(EMG_COLUMNS),
        "supervised_actuators": supervised,
    }
    return {
        "activation": activation_table,
        "action": np.clip(activation_to_muscle_action(activation_table), -1.0, 1.0).astype(np.float32),
        "mask": mask,
        "metadata": metadata,
    }


def load_emg_prior(
    config: dict[str, Any],
    model: mujoco.MjModel,
    reference_length: int,
    control_hz: float,
) -> dict[str, Any]:
    prior_cfg = config.get("emg_prior", {})
    path = str(prior_cfg.get("path", "") or "")
    mask = np.zeros(model.nu, dtype=bool)
    activation_table = np.zeros((reference_length, model.nu), dtype=np.float32)
    metadata: dict[str, Any] = {
        "enabled": False,
        "path": path,
        "cycle_steps": 0,
        "supervised_actuators": [],
    }
    if not bool(prior_cfg.get("enabled", False)) or not path:
        return {"activation": activation_table, "mask": mask, "metadata": metadata}

    expanded_path = Path(path).expanduser()
    if not expanded_path.is_absolute():
        expanded_path = (ROOT / expanded_path).resolve()
    if not expanded_path.exists():
        print(f"[emg_prior] missing file: {expanded_path}", flush=True)
        return {"activation": activation_table, "mask": mask, "metadata": metadata}

    raw = np.genfromtxt(expanded_path, delimiter=",").astype(np.float32)
    if raw.ndim != 2 or raw.shape[1] != len(EMG_COLUMNS):
        raise ValueError(f"Expected EMG CSV shape [N,{len(EMG_COLUMNS)}], got {raw.shape} from {expanded_path}")
    raw = np.clip(raw, 0.0, 1.0)
    cycle_steps = int(prior_cfg.get("cycle_steps", 0) or 0)
    if cycle_steps <= 0:
        cycle_steps = max(1, int(round(control_hz)))
    profile = resample_periodic_profile(raw, cycle_steps)
    emg_col = {name: i for i, name in enumerate(EMG_COLUMNS)}
    phase_offset = int(prior_cfg.get("phase_offset", 0) or 0)
    right_phase_offset = int(prior_cfg.get("right_phase_offset", 0) or 0)
    left_phase_offset = int(prior_cfg.get("left_phase_offset", cycle_steps // 2) or 0)

    supervised: list[str] = []
    for actuator_id in range(model.nu):
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id) or ""
        if actuator_name.endswith("_r"):
            base_name = actuator_name[:-2]
            side_offset = right_phase_offset
        elif actuator_name.endswith("_l"):
            base_name = actuator_name[:-2]
            side_offset = left_phase_offset
        else:
            continue
        emg_name = EMG_ACTUATOR_MAP.get(base_name)
        if emg_name is None:
            continue
        col = emg_col[emg_name]
        for phase in range(reference_length):
            row = (phase + phase_offset + side_offset) % cycle_steps
            activation_table[phase, actuator_id] = profile[row, col]
        mask[actuator_id] = True
        supervised.append(actuator_name)

    metadata = {
        "enabled": bool(np.any(mask)),
        "path": str(expanded_path),
        "cycle_steps": int(cycle_steps),
        "phase_offset": int(phase_offset),
        "right_phase_offset": int(right_phase_offset),
        "left_phase_offset": int(left_phase_offset),
        "columns": list(EMG_COLUMNS),
        "supervised_actuators": supervised,
    }
    return {"activation": activation_table, "mask": mask, "metadata": metadata}


def activation_prior_execution_mix_for_update(config: dict[str, Any], update: int) -> float:
    prior_cfg = config.get("activation_prior", {})
    schedule = prior_cfg.get("execution_mix_schedule")
    if not isinstance(schedule, dict):
        return max(0.0, min(1.0, float(prior_cfg.get("execution_mix", 0.0))))

    start = float(schedule.get("start", prior_cfg.get("execution_mix", 0.0)))
    final = float(schedule.get("final", 0.0))
    decay_updates = int(schedule.get("decay_updates", 0) or 0)
    if decay_updates <= 0:
        return max(0.0, min(1.0, final))
    progress = min(1.0, max(0.0, float(update - 1) / float(decay_updates)))
    mix = start + progress * (final - start)
    return max(0.0, min(1.0, mix))


def scheduled_value(schedule: dict[str, Any] | None, default: float, update: int) -> float:
    if not isinstance(schedule, dict):
        return float(default)
    start = float(schedule.get("start", default))
    final = float(schedule.get("final", start))
    decay_updates = int(schedule.get("decay_updates", 0) or 0)
    if decay_updates <= 0:
        return final
    progress = min(1.0, max(0.0, float(update - 1) / float(decay_updates)))
    return start + progress * (final - start)


def reference_curriculum_for_update(config: dict[str, Any], update: int) -> dict[str, float | int]:
    cfg = config.get("reference_curriculum", {})
    phase_lead = scheduled_value(cfg.get("phase_lead_schedule"), float(cfg.get("phase_lead_steps", 0)), update)
    phase_tolerance = scheduled_value(
        cfg.get("phase_tolerance_schedule"),
        float(cfg.get("phase_tolerance_steps", 0)),
        update,
    )
    swing_exaggeration = scheduled_value(
        cfg.get("swing_exaggeration_schedule"),
        float(cfg.get("swing_exaggeration_scale", 1.0)),
        update,
    )
    return {
        "phase_lead_steps": int(round(phase_lead)),
        "phase_tolerance_steps": max(0, int(round(phase_tolerance))),
        "swing_exaggeration_scale": max(1.0, float(swing_exaggeration)),
    }


def current_reference_curriculum(config: dict[str, Any]) -> dict[str, float | int]:
    cfg = config.get("reference_curriculum", {})
    return {
        "phase_lead_steps": int(cfg.get("current_phase_lead_steps", cfg.get("phase_lead_steps", 0)) or 0),
        "phase_tolerance_steps": int(cfg.get("current_phase_tolerance_steps", cfg.get("phase_tolerance_steps", 0)) or 0),
        "swing_exaggeration_scale": max(
            1.0,
            float(cfg.get("current_swing_exaggeration_scale", cfg.get("swing_exaggeration_scale", 1.0))),
        ),
    }


def reference_q_dq_tensor(
    reference: dict[str, Any],
    phases: torch.Tensor,
    *,
    swing_exaggeration_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    ref_q = reference["q_ref"][phases].clone()
    ref_dq = reference["dq_ref"][phases].clone()
    scale = float(swing_exaggeration_scale)
    if scale <= 1.0:
        return ref_q, ref_dq
    contact = reference["foot_contact_ref"][phases]
    q_mean = reference["q_ref_mean"]
    dq_mean = reference["dq_ref_mean"]
    for mask, cols in (
        (
            ~(contact[:, 0] | contact[:, 1]),
            torch.tensor([TRACK_JOINTS.index("hip_flexion_r"), TRACK_JOINTS.index("knee_angle_r")], dtype=torch.long, device=phases.device),
        ),
        (
            ~(contact[:, 2] | contact[:, 3]),
            torch.tensor([TRACK_JOINTS.index("hip_flexion_l"), TRACK_JOINTS.index("knee_angle_l")], dtype=torch.long, device=phases.device),
        ),
    ):
        if bool(mask.any().item()):
            rows = torch.nonzero(mask, as_tuple=False).flatten()
            ref_q[rows[:, None], cols[None, :]] = q_mean[cols] + scale * (ref_q[rows[:, None], cols[None, :]] - q_mean[cols])
            ref_dq[rows[:, None], cols[None, :]] = dq_mean[cols] + scale * (ref_dq[rows[:, None], cols[None, :]] - dq_mean[cols])
    return ref_q, ref_dq


def reference_foot_tensor(
    reference: dict[str, Any],
    phases: torch.Tensor,
    *,
    swing_exaggeration_scale: float,
) -> torch.Tensor:
    ref_foot = reference["foot_site_ref"][phases].clone()
    scale = float(swing_exaggeration_scale)
    if scale <= 1.0:
        return ref_foot
    contact = reference["foot_contact_ref"][phases]
    min_z = reference["foot_site_min_z"].unsqueeze(0)
    exaggerated_z = min_z + scale * (ref_foot[:, :, 2] - min_z)
    ref_foot[:, :, 2] = torch.where(~contact, exaggerated_z, ref_foot[:, :, 2])
    return ref_foot


def terrain_preview_dim(config: dict[str, Any]) -> int:
    cfg = config.get("terrain_context", {})
    if not bool(cfg.get("include_height_preview", False)):
        return 0
    return max(0, int(cfg.get("num_preview_samples", 0) or 0))


def foot_obs_feature_dim(config: dict[str, Any]) -> int:
    obs_cfg = config.get("observation", {})
    per_foot = 2
    if bool(obs_cfg.get("include_foot_rel_z", False)):
        per_foot += 1
    if bool(obs_cfg.get("include_foot_ground_slope", False)):
        per_foot += 1
    if bool(obs_cfg.get("include_contact_obs", False)):
        per_foot += 2
    return per_foot * len(FOOT_SITE_NAMES)


def frame_stack_prev_steps(config: dict[str, Any]) -> int:
    return max(0, int(config.get("observation", {}).get("frame_stack_prev_steps", 0) or 0))


def frame_stack_feature_dim(config: dict[str, Any], *, nq: int, nv: int, na: int) -> int:
    return int(nq) + int(nv) + int(na) + foot_obs_feature_dim(config)


def post_reference_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = config.get("post_reference", {})
    return cfg if isinstance(cfg, dict) else {}


def post_reference_enabled(config: dict[str, Any]) -> bool:
    return bool(post_reference_config(config).get("enabled", False))


def post_reference_valid_steps(reference: dict[str, Any], config: dict[str, Any]) -> int:
    raw = int(post_reference_config(config).get("valid_steps", 0) or 0)
    if raw <= 0:
        raw = int(reference["length"])
    return max(1, min(raw, int(reference["length"])))


def reference_obs_extra_dim(config: dict[str, Any]) -> int:
    return 1 if bool(post_reference_config(config).get("include_reference_valid_obs", False)) else 0


def reference_index(phases: torch.Tensor, reference: dict[str, Any], config: dict[str, Any]) -> torch.Tensor:
    if post_reference_enabled(config):
        return torch.clamp(phases, min=0, max=int(reference["length"]) - 1)
    return phases % int(reference["length"])


def terrain_height_from_params(x: torch.Tensor, terrain_type_id: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    height = torch.zeros_like(x)
    slope_mask = terrain_type_id == 1
    if bool(slope_mask.any().item()):
        slope = params[:, 0:1]
        anchor_x = params[:, 1:2]
        anchor_height = params[:, 2:3]
        height = torch.where(slope_mask[:, None], anchor_height + (x - anchor_x) * slope, height)
    stair_mask = terrain_type_id == 2
    if bool(stair_mask.any().item()):
        step_height = torch.clamp(params[:, 0:1], min=1e-6)
        step_depth = torch.clamp(params[:, 1:2], min=1e-6)
        direction = torch.where(params[:, 2:3] >= 0.0, torch.ones_like(step_height), -torch.ones_like(step_height))
        step_count = torch.clamp(torch.round(params[:, 3:4]), min=1.0)
        start_x = params[:, 4:5]
        base_height = params[:, 5:6]
        platform_depth = torch.clamp(params[:, 6:7], min=0.0)
        progressed = x - start_x
        step_index = torch.clamp(torch.floor(torch.clamp(progressed, min=0.0) / step_depth), min=0.0)
        step_index = torch.minimum(step_index, step_count - 1.0)
        up_height = base_height + (step_index + 1.0) * step_height
        up_platform = (platform_depth > 0.0) & (progressed >= step_count * step_depth)
        up_height = torch.where(up_platform, base_height + step_count * step_height, up_height)
        down_platform = (platform_depth > 0.0) & (x >= start_x - platform_depth) & (x < start_x)
        down_stair = (x >= start_x) & (x <= start_x + step_count * step_depth)
        down_height = torch.where(down_platform, base_height, torch.zeros_like(height))
        down_step_height = torch.clamp(base_height - step_index * step_height, min=0.0)
        down_height = torch.where(down_stair, down_step_height, down_height)
        stair_height = torch.where(direction > 0.0, torch.where(progressed >= 0.0, up_height, torch.zeros_like(up_height)), down_height)
        height = torch.where(stair_mask[:, None], stair_height, height)
    return height


def course_height_tensor(x: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    course_cfg = config.get("terrain_course", {})
    segments = course_cfg.get("segments", [])
    if not isinstance(segments, list) or not segments:
        return torch.zeros_like(x)
    height = torch.zeros_like(x)
    for segment in segments:
        x0 = float(segment.get("x0", -1e9))
        x1 = float(segment.get("x1", 1e9))
        mask = (x >= x0) & (x <= x1)
        kind = str(segment.get("type", "flat"))
        if kind == "flat":
            value = torch.full_like(x, float(segment.get("height", 0.0)))
            height = torch.where(mask, value, height)
        elif kind == "slope":
            value = float(segment.get("height0", 0.0)) + float(segment.get("slope", 0.0)) * (x - x0)
            height = torch.where(mask, value, height)
        elif kind == "stairs":
            step_height = max(float(segment.get("step_height", 0.127)), 1e-6)
            step_depth = max(float(segment.get("step_depth", 0.32)), 1e-6)
            direction = 1.0 if float(segment.get("direction", 1.0)) >= 0.0 else -1.0
            steps = max(1, int(segment.get("steps", 4)))
            base_height = float(segment.get("base_height", 0.0))
            progressed = torch.clamp(x - x0, min=0.0)
            step_index = torch.clamp(torch.floor(progressed / step_depth), min=0.0, max=float(steps))
            value = base_height + direction * step_index * step_height
            height = torch.where(mask, value, height)
        elif kind == "stairs_box":
            for tread_x0, tread_x1, tread_height in stair_box_treads(segment):
                tread_mask = (x >= tread_x0) & (x <= tread_x1)
                height = torch.where(tread_mask, torch.full_like(x, tread_height), height)
    return height


def terrain_slope_from_params(x: torch.Tensor, terrain_type_id: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    slope = torch.zeros_like(x)
    slope_mask = terrain_type_id == 1
    if bool(slope_mask.any().item()):
        slope = torch.where(slope_mask[:, None], params[:, 0:1].expand_as(x), slope)
    return slope


def course_slope_tensor(x: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    course_cfg = config.get("terrain_course", {})
    segments = course_cfg.get("segments", [])
    if not isinstance(segments, list) or not segments:
        return torch.zeros_like(x)
    slope = torch.zeros_like(x)
    for segment in segments:
        x0 = float(segment.get("x0", -1e9))
        x1 = float(segment.get("x1", 1e9))
        mask = (x >= x0) & (x <= x1)
        if str(segment.get("type", "flat")) == "slope":
            slope = torch.where(mask, torch.full_like(x, float(segment.get("slope", 0.0))), slope)
    return slope


def terrain_height_preview_tensor(qpos: torch.Tensor, phase_idx: torch.Tensor, reference: dict[str, Any], config: dict[str, Any]) -> torch.Tensor:
    count = terrain_preview_dim(config)
    if count <= 0:
        return torch.empty((qpos.shape[0], 0), dtype=torch.float32, device=qpos.device)
    course_enabled = bool(config.get("terrain_course", {}).get("enabled", False))
    terrain_type_id = reference.get("terrain_type_id")
    terrain_params = reference.get("terrain_params_tensor")
    if not course_enabled and (terrain_type_id is None or terrain_params is None):
        return torch.zeros((qpos.shape[0], count), dtype=torch.float32, device=qpos.device)
    cfg = config.get("terrain_context", {})
    start_m = float(cfg.get("preview_start_m", 0.1))
    end_m = float(cfg.get("preview_end_m", 2.4))
    scale = max(float(cfg.get("height_scale", 0.2)), 1e-6)
    offsets = torch.linspace(start_m, end_m, count, dtype=torch.float32, device=qpos.device).unsqueeze(0)
    x0 = qpos[:, int(reference["pelvis_tx_qpos"])].unsqueeze(1)
    if course_enabled:
        h0 = course_height_tensor(x0, config)
        h = course_height_tensor(x0 + offsets, config)
    else:
        phase = phase_idx % int(reference["length"])
        type_rows = terrain_type_id[phase]
        param_rows = terrain_params[phase]
        h0 = terrain_height_from_params(x0, type_rows, param_rows)
        h = terrain_height_from_params(x0 + offsets, type_rows, param_rows)
    return torch.clamp((h - h0) / scale, -5.0, 5.0)


def current_terrain_height_tensor(
    qpos: torch.Tensor,
    phase_idx: torch.Tensor,
    reference: dict[str, Any],
    config: dict[str, Any],
) -> torch.Tensor:
    x = qpos[:, int(reference["pelvis_tx_qpos"])].unsqueeze(1)
    if bool(config.get("terrain_course", {}).get("enabled", False)):
        return course_height_tensor(x, config).squeeze(1)
    terrain_type_id = reference.get("terrain_type_id")
    terrain_params = reference.get("terrain_params_tensor")
    if terrain_type_id is None or terrain_params is None:
        return torch.zeros((qpos.shape[0],), dtype=torch.float32, device=qpos.device)
    phase = phase_idx % int(reference["length"])
    return terrain_height_from_params(x, terrain_type_id[phase], terrain_params[phase]).squeeze(1)


def current_terrain_slope_tensor(
    qpos: torch.Tensor,
    phase_idx: torch.Tensor,
    reference: dict[str, Any],
    config: dict[str, Any],
) -> torch.Tensor:
    x = qpos[:, int(reference["pelvis_tx_qpos"])].unsqueeze(1)
    if bool(config.get("terrain_course", {}).get("enabled", False)):
        return course_slope_tensor(x, config).squeeze(1)
    terrain_type_id = reference.get("terrain_type_id")
    terrain_params = reference.get("terrain_params_tensor")
    if terrain_type_id is None or terrain_params is None:
        return torch.zeros((qpos.shape[0],), dtype=torch.float32, device=qpos.device)
    phase = phase_idx % int(reference["length"])
    return terrain_slope_from_params(x, terrain_type_id[phase], terrain_params[phase]).squeeze(1)


def terrain_height_for_world_x_tensor(
    x: torch.Tensor,
    phase_idx: torch.Tensor,
    reference: dict[str, Any],
    config: dict[str, Any],
) -> torch.Tensor:
    if bool(config.get("terrain_course", {}).get("enabled", False)):
        return course_height_tensor(x, config)
    terrain_type_id = reference.get("terrain_type_id")
    terrain_params = reference.get("terrain_params_tensor")
    if terrain_type_id is None or terrain_params is None:
        return torch.zeros_like(x)
    phase = phase_idx % int(reference["length"])
    return terrain_height_from_params(x, terrain_type_id[phase], terrain_params[phase])


def terrain_slope_for_world_x_tensor(
    x: torch.Tensor,
    phase_idx: torch.Tensor,
    reference: dict[str, Any],
    config: dict[str, Any],
) -> torch.Tensor:
    if bool(config.get("terrain_course", {}).get("enabled", False)):
        return course_slope_tensor(x, config)
    terrain_type_id = reference.get("terrain_type_id")
    terrain_params = reference.get("terrain_params_tensor")
    if terrain_type_id is None or terrain_params is None:
        return torch.zeros_like(x)
    phase = phase_idx % int(reference["length"])
    return terrain_slope_from_params(x, terrain_type_id[phase], terrain_params[phase])


def build_policy_state_feature_tensor(
    *,
    qpos: torch.Tensor,
    qvel: torch.Tensor,
    act: torch.Tensor,
    site_xpos: torch.Tensor,
    phase_idx: torch.Tensor,
    pelvis_tx_qpos: int,
    foot_site_indices: torch.Tensor,
    reference: dict[str, Any],
    config: dict[str, Any],
) -> torch.Tensor:
    obs_cfg = config.get("observation", {})
    localize_obs = bool(obs_cfg.get("localize_root", False))
    qpos_obs = qpos
    if localize_obs:
        qpos_obs = qpos.clone()
        qpos_obs[:, pelvis_tx_qpos] = 0.0
        pelvis_ty_qpos = int(reference["qpos_indices"][TRACK_JOINTS.index("pelvis_ty")].detach().cpu().item())
        terrain_height = current_terrain_height_tensor(qpos, phase_idx, reference, config)
        qpos_obs[:, pelvis_ty_qpos] = qpos[:, pelvis_ty_qpos] - terrain_height
    curriculum = current_reference_curriculum(config)
    target_phase = reference_index(phase_idx + int(curriculum["phase_lead_steps"]), reference, config)
    foot = site_xpos[:, foot_site_indices, :]
    foot_rel_x = foot[:, :, 0] - qpos[:, pelvis_tx_qpos].unsqueeze(1)
    pelvis_ty_qpos = int(reference["qpos_indices"][TRACK_JOINTS.index("pelvis_ty")].detach().cpu().item())
    foot_rel_z = foot[:, :, 2] - qpos[:, pelvis_ty_qpos].unsqueeze(1)
    foot_terrain_height = terrain_height_for_world_x_tensor(foot[:, :, 0], phase_idx, reference, config)
    foot_clearance = foot[:, :, 2] - foot_terrain_height
    foot_ground_slope = terrain_slope_for_world_x_tensor(foot[:, :, 0], phase_idx, reference, config)
    ref_contact_obs = reference["foot_contact_ref"][target_phase].float()
    current_contact_obs = (
        foot_clearance < float(config.get("reference_contact", {}).get("z_threshold", 0.025))
    ).float()
    foot_z_feature = foot_clearance if localize_obs else foot[:, :, 2]
    feature_groups = [foot_rel_x]
    if bool(obs_cfg.get("include_foot_rel_z", False)):
        feature_groups.append(foot_rel_z)
    feature_groups.append(foot_z_feature)
    if bool(obs_cfg.get("include_foot_ground_slope", False)):
        feature_groups.append(foot_ground_slope)
    if bool(obs_cfg.get("include_contact_obs", False)):
        feature_groups.extend([current_contact_obs, ref_contact_obs])
    foot_features = torch.cat(feature_groups, dim=1)
    return torch.cat([qpos_obs, qvel, act, foot_features], dim=1)


def current_terrain_height_np(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    reference: dict[str, Any],
    config: dict[str, Any],
    phase: int,
) -> float:
    pelvis_tx_qpos = int(model.jnt_qposadr[joint_id(model, "pelvis_tx")])
    x = np.array([float(data.qpos[pelvis_tx_qpos])], dtype=np.float64)
    if bool(config.get("terrain_course", {}).get("enabled", False)):
        return float(course_height_np(x, list(config.get("terrain_course", {}).get("segments", [])))[0])
    terrain_type_id = reference.get("terrain_type_id")
    terrain_params = reference.get("terrain_params_tensor")
    if terrain_type_id is None or terrain_params is None:
        return 0.0
    ref_phase = int(phase) % int(reference["length"])
    terrain_type = int(terrain_type_id[ref_phase].detach().cpu().item())
    params = terrain_params[ref_phase].detach().cpu().numpy().astype(np.float64)
    return float(terrain_height_np_from_params(x, terrain_type, params)[0])


def build_policy_obs_tensor(
    *,
    qpos: torch.Tensor,
    qvel: torch.Tensor,
    act: torch.Tensor,
    site_xpos: torch.Tensor,
    phase_idx: torch.Tensor,
    episode_step: torch.Tensor | None = None,
    pelvis_tx_qpos: int,
    foot_site_indices: torch.Tensor,
    reference: dict[str, Any],
    config: dict[str, Any],
    state_history_obs: torch.Tensor | None = None,
) -> torch.Tensor:
    curriculum = current_reference_curriculum(config)
    valid_steps = post_reference_valid_steps(reference, config)
    if post_reference_enabled(config):
        if episode_step is None:
            reference_valid = (phase_idx < valid_steps).float()
        else:
            reference_valid = (episode_step < valid_steps).float()
        raw_target_phase = phase_idx + int(curriculum["phase_lead_steps"])
        target_phase = reference_index(raw_target_phase, reference, config)
        phase_for_obs = torch.where(reference_valid > 0.0, phase_idx.float(), torch.zeros_like(phase_idx.float()))
    else:
        reference_valid = torch.ones((qpos.shape[0],), dtype=torch.float32, device=qpos.device)
        raw_target_phase = phase_idx + int(curriculum["phase_lead_steps"])
        target_phase = reference_index(raw_target_phase, reference, config)
        phase_for_obs = phase_idx.float()
    reference_valid_col = reference_valid.unsqueeze(1)
    obs_cfg = config.get("observation", {})
    localize_obs = bool(obs_cfg.get("localize_root", False))
    phase_mode = str(obs_cfg.get("phase_obs", "reference") or "reference")
    phase = phase_for_obs * (2.0 * torch.pi / float(reference["length"]))
    phase_features = torch.stack([torch.sin(phase), torch.cos(phase)], dim=1) * reference_valid_col
    if phase_mode in {"none", "zero", "disabled"}:
        phase_features = torch.zeros_like(phase_features)
    qpos_obs = qpos
    if localize_obs:
        qpos_obs = qpos.clone()
        qpos_obs[:, pelvis_tx_qpos] = 0.0
        pelvis_ty_qpos = int(reference["qpos_indices"][TRACK_JOINTS.index("pelvis_ty")].detach().cpu().item())
        terrain_height = current_terrain_height_tensor(qpos, phase_idx, reference, config)
        qpos_obs[:, pelvis_ty_qpos] = qpos[:, pelvis_ty_qpos] - terrain_height
    q = qpos[:, reference["qpos_indices"]]
    dq = qvel[:, reference["qvel_indices"]]
    ref_q, ref_dq = reference_q_dq_tensor(
        reference,
        target_phase,
        swing_exaggeration_scale=float(curriculum["swing_exaggeration_scale"]),
    )
    foot = site_xpos[:, foot_site_indices, :]
    foot_rel_x = foot[:, :, 0] - qpos[:, pelvis_tx_qpos].unsqueeze(1)
    pelvis_ty_qpos = int(reference["qpos_indices"][TRACK_JOINTS.index("pelvis_ty")].detach().cpu().item())
    foot_rel_z = foot[:, :, 2] - qpos[:, pelvis_ty_qpos].unsqueeze(1)
    foot_terrain_height = terrain_height_for_world_x_tensor(foot[:, :, 0], phase_idx, reference, config)
    foot_clearance = foot[:, :, 2] - foot_terrain_height
    foot_ground_slope = terrain_slope_for_world_x_tensor(foot[:, :, 0], phase_idx, reference, config)
    ref_contact_obs = reference["foot_contact_ref"][target_phase].float()
    current_contact_obs = (
        foot_clearance < float(config.get("reference_contact", {}).get("z_threshold", 0.025))
    ).float()
    foot_z_feature = foot_clearance if localize_obs else foot[:, :, 2]
    feature_groups = [foot_rel_x]
    if bool(obs_cfg.get("include_foot_rel_z", False)):
        feature_groups.append(foot_rel_z)
    feature_groups.append(foot_z_feature)
    if bool(obs_cfg.get("include_foot_ground_slope", False)):
        feature_groups.append(foot_ground_slope)
    if bool(obs_cfg.get("include_contact_obs", False)):
        feature_groups.extend([current_contact_obs, ref_contact_obs])
    foot_features = torch.cat(feature_groups, dim=1)
    state_features = torch.cat([qpos_obs, qvel, act, foot_features], dim=1)
    obs_parts = [
        qpos_obs,
        qvel,
        act,
        (ref_q - q) * reference_valid_col,
        (ref_dq - dq) * reference_valid_col,
        phase_features,
        foot_features,
    ]
    if reference_obs_extra_dim(config) > 0:
        obs_parts.append(reference_valid_col)
    history_steps = frame_stack_prev_steps(config)
    if history_steps > 0:
        expected_history_dim = frame_stack_feature_dim(
            config,
            nq=int(qpos.shape[1]),
            nv=int(qvel.shape[1]),
            na=int(act.shape[1]),
        )
        if state_history_obs is None:
            history = state_features.unsqueeze(1).expand(-1, history_steps, -1)
        else:
            history = state_history_obs
        if int(history.shape[1]) != history_steps or int(history.shape[2]) != expected_history_dim:
            raise ValueError(
                f"state history shape mismatch: got {tuple(history.shape)}, expected (*, {history_steps}, {expected_history_dim})"
            )
        obs_parts.append(history.reshape(qpos.shape[0], history_steps * expected_history_dim))
    future_steps = max(0, int(config.get("imitation", {}).get("reference_future_steps", 0) or 0))
    future_dropout_prob = max(0.0, min(1.0, float(config.get("imitation", {}).get("current_future_obs_dropout_prob", 0.0) or 0.0)))
    future_keep_mask = None
    if future_steps > 0 and future_dropout_prob > 0.0:
        future_keep_mask = (torch.rand((qpos.shape[0], 1), dtype=torch.float32, device=qpos.device) >= future_dropout_prob).float()
    for offset in range(1, future_steps + 1):
        raw_future_phase = raw_target_phase + offset
        future_phase = reference_index(raw_future_phase, reference, config)
        if post_reference_enabled(config):
            future_valid = (
                raw_future_phase < valid_steps if episode_step is None else (episode_step + offset) < valid_steps
            ).float().unsqueeze(1)
        else:
            future_valid = torch.ones_like(reference_valid_col)
        future_q, _future_dq = reference_q_dq_tensor(
            reference,
            future_phase,
            swing_exaggeration_scale=float(curriculum["swing_exaggeration_scale"]),
        )
        future_foot = reference_foot_tensor(
            reference,
            future_phase,
            swing_exaggeration_scale=float(curriculum["swing_exaggeration_scale"]),
        )
        future_foot_z = future_foot[:, :, 2]
        future_pelvis_ty = reference["reset_q_ref"][future_phase, RESET_JOINTS.index("pelvis_ty")]
        future_foot_rel_z = future_foot[:, :, 2] - future_pelvis_ty.unsqueeze(1)
        future_ref_contact = reference["foot_contact_ref"][future_phase].float()
        future_current_contact = future_ref_contact
        future_foot_ground_slope = torch.zeros_like(future_foot[:, :, 0])
        if localize_obs:
            future_pelvis_tx = reference["reset_q_ref"][future_phase, RESET_JOINTS.index("pelvis_tx")]
            future_foot_world_x = future_pelvis_tx.unsqueeze(1) + future_foot[:, :, 0]
            future_foot_ground_slope = terrain_slope_for_world_x_tensor(
                future_foot_world_x,
                future_phase,
                reference,
                config,
            )
            future_foot_z = future_foot[:, :, 2] - terrain_height_for_world_x_tensor(
                future_foot_world_x,
                future_phase,
                reference,
                config,
            )
        future_feature_groups = [future_foot[:, :, 0]]
        if bool(obs_cfg.get("include_foot_rel_z", False)):
            future_feature_groups.append(future_foot_rel_z)
        future_feature_groups.append(future_foot_z)
        if bool(obs_cfg.get("include_foot_ground_slope", False)):
            future_feature_groups.append(future_foot_ground_slope)
        if bool(obs_cfg.get("include_contact_obs", False)):
            future_feature_groups.extend([future_current_contact, future_ref_contact])
        future_foot_features = torch.cat(future_feature_groups, dim=1)
        future_q_delta = future_q - q
        future_foot_delta = future_foot_features - foot_features
        future_q_delta = future_q_delta * future_valid
        future_foot_delta = future_foot_delta * future_valid
        if future_keep_mask is not None:
            future_q_delta = future_q_delta * future_keep_mask
            future_foot_delta = future_foot_delta * future_keep_mask
        obs_parts.append(future_q_delta)
        obs_parts.append(future_foot_delta)
    terrain_preview = terrain_height_preview_tensor(qpos, phase_idx, reference, config)
    if terrain_preview.shape[1] > 0:
        obs_parts.append(terrain_preview)
    return torch.cat(obs_parts, dim=1)


def load_reference(
    reference_path: Path,
    model: mujoco.MjModel,
    control_hz: float,
    device: torch.device,
    config: dict[str, Any],
) -> dict[str, Any]:
    raw = np.load(reference_path, allow_pickle=True)
    metadata = raw["metadata"].item()
    series = raw["series_data"].item()
    source_hz = float(metadata.get("sample_rate", 500.0))
    length = len(next(iter(series.values())))
    indices = np.round(np.arange(0.0, float(length), source_hz / float(control_hz))).astype(np.int64)
    indices = np.unique(np.clip(indices, 0, length - 1))

    qpos_indices = []
    qvel_indices = []
    q_ref = []
    dq_ref = []
    pose_scales = []
    vel_scales = []
    for joint in TRACK_JOINTS:
        jid = joint_id(model, joint)
        qpos_indices.append(int(model.jnt_qposadr[jid]))
        qvel_indices.append(int(model.jnt_dofadr[jid]))
        q_ref.append(np.asarray(series.get(f"q_{joint}", np.zeros(length)), dtype=np.float32)[indices])
        dq_ref.append(np.asarray(series.get(f"dq_{joint}", np.zeros(length)), dtype=np.float32)[indices])
        pose_scales.append(0.15 if joint.startswith("pelvis") else 0.45)
        vel_scales.append(1.0 if joint.startswith("pelvis") else 4.0)

    reset_qpos_indices = []
    reset_qvel_indices = []
    reset_q_ref = []
    reset_dq_ref = []
    for joint in RESET_JOINTS:
        jid = joint_id(model, joint)
        reset_qpos_indices.append(int(model.jnt_qposadr[jid]))
        reset_qvel_indices.append(int(model.jnt_dofadr[jid]))
        reset_q_ref.append(np.asarray(series.get(f"q_{joint}", np.zeros(length)), dtype=np.float32)[indices])
        reset_dq_ref.append(np.asarray(series.get(f"dq_{joint}", np.zeros(length)), dtype=np.float32)[indices])

    q_ref_np = np.stack(q_ref, axis=1)
    dq_ref_np = np.stack(dq_ref, axis=1)
    reset_q_np = np.stack(reset_q_ref, axis=1)
    reset_dq_np = np.stack(reset_dq_ref, axis=1)
    pelvis_tx_reset_col = RESET_JOINTS.index("pelvis_tx")
    pelvis_tx_ref_np = reset_q_np[:, pelvis_tx_reset_col].copy()
    reset_q_np[:, pelvis_tx_reset_col] = 0.0

    foot_site_indices = [site_id(model, name) for name in FOOT_SITE_NAMES]
    ref_data = mujoco.MjData(model)
    foot_site_xpos = np.zeros((len(indices), len(FOOT_SITE_NAMES), 3), dtype=np.float32)
    pelvis_tx_index = int(model.jnt_qposadr[joint_id(model, "pelvis_tx")])
    for frame in range(len(indices)):
        mujoco.mj_resetData(model, ref_data)
        ref_data.qpos[reset_qpos_indices] = reset_q_np[frame]
        ref_data.qvel[reset_qvel_indices] = reset_dq_np[frame]
        apply_joint_equalities_np(model, ref_data)
        mujoco.mj_forward(model, ref_data)
        foot_site_xpos[frame] = ref_data.site_xpos[foot_site_indices]
        foot_site_xpos[frame, :, 0] -= float(ref_data.qpos[pelvis_tx_index])

    contact_cfg = config.get("reference_contact", {})
    foot_site_world = foot_site_xpos.copy()
    foot_site_world[:, :, 0] += pelvis_tx_ref_np[:, None]
    foot_site_velocity = np.gradient(foot_site_world, 1.0 / float(control_hz), axis=0).astype(np.float32)
    foot_site_speed = np.linalg.norm(foot_site_velocity[:, :, [0, 2]], axis=2).astype(np.float32)
    terrain_height = source_terrain_height_np(metadata, foot_site_world[:, :, 0])
    foot_clearance = foot_site_world[:, :, 2] - terrain_height
    contact_z_threshold = float(contact_cfg.get("z_threshold", 0.025))
    contact_speed_threshold = float(contact_cfg.get("speed_threshold", 0.4))
    foot_contact_ref = (foot_clearance < contact_z_threshold) & (foot_site_speed < contact_speed_threshold)

    activation_prior = load_activation_prior(config, model, len(indices), q_ref_np, control_hz)
    emg_prior = load_emg_prior(config, model, len(indices), control_hz)

    return {
        "path": str(reference_path),
        "metadata": metadata,
        "control_hz": float(control_hz),
        "source_indices": indices,
        "length": int(len(indices)),
        "joint_names": list(TRACK_JOINTS),
        "qpos_indices": torch.tensor(qpos_indices, dtype=torch.long, device=device),
        "qvel_indices": torch.tensor(qvel_indices, dtype=torch.long, device=device),
        "q_ref": torch.tensor(q_ref_np, dtype=torch.float32, device=device),
        "dq_ref": torch.tensor(dq_ref_np, dtype=torch.float32, device=device),
        "q_ref_mean": torch.tensor(np.mean(q_ref_np, axis=0), dtype=torch.float32, device=device),
        "dq_ref_mean": torch.tensor(np.mean(dq_ref_np, axis=0), dtype=torch.float32, device=device),
        "reset_qpos_indices": torch.tensor(reset_qpos_indices, dtype=torch.long, device=device),
        "reset_qvel_indices": torch.tensor(reset_qvel_indices, dtype=torch.long, device=device),
        "reset_q_ref": torch.tensor(reset_q_np, dtype=torch.float32, device=device),
        "reset_dq_ref": torch.tensor(reset_dq_np, dtype=torch.float32, device=device),
        "pose_scales": torch.tensor(pose_scales, dtype=torch.float32, device=device),
        "vel_scales": torch.tensor(vel_scales, dtype=torch.float32, device=device),
        "foot_site_names": list(FOOT_SITE_NAMES),
        "foot_site_indices": torch.tensor(foot_site_indices, dtype=torch.long, device=device),
        "foot_site_ref": torch.tensor(foot_site_xpos, dtype=torch.float32, device=device),
        "foot_site_min_z": torch.tensor(np.amin(foot_site_xpos[:, :, 2], axis=0), dtype=torch.float32, device=device),
        "foot_contact_ref": torch.tensor(foot_contact_ref, dtype=torch.bool, device=device),
        "foot_speed_ref": torch.tensor(foot_site_speed, dtype=torch.float32, device=device),
        "pelvis_tx_ref": torch.tensor(pelvis_tx_ref_np, dtype=torch.float32, device=device),
        "activation_prior_metadata": activation_prior["metadata"],
        "activation_prior_ref": torch.tensor(activation_prior["activation"], dtype=torch.float32, device=device),
        "activation_prior_action_ref": torch.tensor(activation_prior["action"], dtype=torch.float32, device=device),
        "activation_prior_mask": torch.tensor(activation_prior["mask"], dtype=torch.bool, device=device),
        "emg_prior_metadata": emg_prior["metadata"],
        "emg_prior_ref": torch.tensor(emg_prior["activation"], dtype=torch.float32, device=device),
        "emg_prior_mask": torch.tensor(emg_prior["mask"], dtype=torch.bool, device=device),
        "pelvis_tx_qpos": pelvis_tx_index,
    }


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


def stairs_box_course_bounds(config: dict[str, Any]) -> dict[str, float]:
    bounds: dict[str, float] = {}
    for segment in list(config.get("terrain_course", {}).get("segments", [])):
        if str(segment.get("type", "flat")) != "stairs_box":
            continue
        direction = 1.0 if float(segment.get("direction", 1.0)) >= 0.0 else -1.0
        treads = stair_box_treads(segment)
        if not treads:
            continue
        if direction > 0.0 and "up_x0" not in bounds:
            bounds["up_x0"] = float(treads[0][0])
            bounds["up_top_x0"] = float(treads[-1][0])
            bounds["up_x1"] = float(treads[-1][1])
        elif direction < 0.0 and "down_x0" not in bounds:
            first_step = treads[1] if float(segment.get("platform_depth", 0.0)) > 0.0 and len(treads) > 1 else treads[0]
            bounds["down_x0"] = float(first_step[0])
            bounds["down_second_x0"] = float(treads[min(len(treads) - 1, 1)][0])
            bounds["down_x1"] = float(treads[-1][1])
    return bounds


def slope_course_bounds(config: dict[str, Any]) -> dict[str, float]:
    bounds: dict[str, float] = {}
    for segment in list(config.get("terrain_course", {}).get("segments", [])):
        if str(segment.get("type", "flat")) != "slope":
            continue
        x0 = float(segment.get("x0", 0.0))
        x1 = float(segment.get("x1", x0))
        slope = float(segment.get("slope", 0.0))
        if slope > 0.0 and "up_x0" not in bounds:
            bounds["up_x0"] = x0
            bounds["up_x1"] = x1
        elif slope < 0.0 and "down_x0" not in bounds:
            bounds["down_x0"] = x0
            bounds["down_x1"] = x1
    return bounds


def metadata_forward_distance(metadata: dict[str, Any]) -> float:
    transform = metadata.get("root_transform", {})
    if isinstance(transform, dict):
        try:
            return max(float(transform.get("forward_distance", 0.0)), 0.0)
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def metadata_slope_segment(metadata: dict[str, Any], sign: int) -> dict[str, Any] | None:
    segments = metadata.get("source_terrain_segments")
    if not isinstance(segments, list) or not segments:
        segments = metadata.get("terrain_course_segments")
    if not isinstance(segments, list):
        return None
    wanted = 1.0 if int(sign) >= 0 else -1.0
    for segment in segments:
        if not isinstance(segment, dict) or str(segment.get("type", "")) != "slope":
            continue
        slope = float(segment.get("slope", 0.0) or 0.0)
        if slope * wanted > 0.0:
            return segment
    return None


def default_course_offset(metadata: dict[str, Any], config: dict[str, Any] | None = None) -> float:
    terrain_id = str(metadata.get("terrain_id", "") or "")
    label = str(metadata.get("source_label", "") or "")
    slope_bounds = slope_course_bounds(config or {})
    bounds = stairs_box_course_bounds(config or {})
    if "stitched_level_to_rampascent" in label or "walk-rampascent" in label:
        segment = metadata_slope_segment(metadata, 1)
        if segment is not None:
            return float(slope_bounds.get("up_x0", 4.0)) - float(segment.get("x0", 0.0))
    if "stitched_level_to_rampdescent" in label or "walk-rampdescent" in label:
        segment = metadata_slope_segment(metadata, -1)
        if segment is not None:
            return float(slope_bounds.get("down_x0", 14.0)) - float(segment.get("x0", 0.0))
    if "stitched_rampascent_to_level" in label or "rampascent-walk" in label:
        segment = metadata_slope_segment(metadata, 1)
        if segment is not None:
            return float(slope_bounds.get("up_x1", 10.0)) - float(segment.get("x1", 0.0))
    if "stitched_rampdescent_to_level" in label or "rampdescent-walk" in label:
        segment = metadata_slope_segment(metadata, -1)
        if segment is not None:
            return float(slope_bounds.get("down_x1", 20.0)) - float(segment.get("x1", 0.0))
    if terrain_id == "levelwalking":
        return float((config or {}).get("terrain_course", {}).get("levelwalking_reference_x_shift", 0.0))
    if terrain_id == "slopeascent":
        up_x0 = float(slope_bounds.get("up_x0", 4.0))
        up_x1 = float(slope_bounds.get("up_x1", 10.0))
        course_cfg = (config or {}).get("terrain_course", {})
        if "rampascent-walk" in label:
            return up_x1 + float(course_cfg.get("slopeascent_exit_shift", 0.0))
        return up_x0 + float(course_cfg.get("slopeascent_entry_shift", 0.0))
    if terrain_id == "slopedescent":
        down_x0 = float(slope_bounds.get("down_x0", 14.0))
        down_x1 = float(slope_bounds.get("down_x1", 20.0))
        course_cfg = (config or {}).get("terrain_course", {})
        if "rampdescent-walk" in label:
            return down_x1 + float(course_cfg.get("slopedescent_exit_shift", 0.0))
        return down_x0 + float(course_cfg.get("slopedescent_entry_shift", 0.0))
    if terrain_id == "stairascent":
        if "walk-" in label or label == "stairascent":
            return float(bounds.get("up_x0", 24.0))
        return float(bounds.get("up_top_x0", 24.62215872))
    if terrain_id == "stairdescent":
        course_cfg = (config or {}).get("terrain_course", {})
        return float(bounds.get("down_second_x0", bounds.get("down_x0", 25.42215872))) + float(
            course_cfg.get("stairdescent_reference_x_shift", 0.0)
        )
    return 0.0


def smooth_reference_correction_np(values: np.ndarray, window: int, max_step: float) -> np.ndarray:
    out = np.asarray(values, dtype=np.float32).copy()
    if out.size <= 1:
        return out
    window = max(1, int(window))
    if window >= 3:
        if window % 2 == 0:
            window += 1
        radius = window // 2
        padded = np.pad(out, (radius, radius), mode="edge")
        kernel = np.full((window,), 1.0 / float(window), dtype=np.float32)
        out = np.convolve(padded, kernel, mode="valid").astype(np.float32)
    max_step = float(max_step)
    if max_step > 0.0:
        for index in range(1, out.size):
            out[index] = float(np.clip(out[index], out[index - 1] - max_step, out[index - 1] + max_step))
        for index in range(out.size - 2, -1, -1):
            out[index] = float(np.clip(out[index], out[index + 1] - max_step, out[index + 1] + max_step))
    return out


def apply_reference_course_transform(ref: dict[str, Any], config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    course_cfg = config.get("terrain_course", {})
    contact_cfg = config.get("reference_contact", {})
    offset = float(default_course_offset(ref["metadata"], config))
    local_x = ref["pelvis_tx_ref"].detach().cpu().numpy().astype(np.float64)
    local_height = source_terrain_height_np(ref["metadata"], local_x)
    if bool(course_cfg.get("enabled", False)):
        world_height = course_height_np(local_x + offset, list(course_cfg.get("segments", [])))
    else:
        world_height = local_height
    delta_np = (world_height - local_height).astype(np.float32)
    offset_tensor = torch.full((ref["length"],), offset, dtype=torch.float32, device=device)

    ref["course_offset"] = offset_tensor
    ref["reset_q_ref"] = ref["reset_q_ref"].clone()
    ref["q_ref"] = ref["q_ref"].clone()
    ref["foot_site_ref"] = ref["foot_site_ref"].clone()
    ref["reset_q_ref"][:, RESET_JOINTS.index("pelvis_tx")] = ref["pelvis_tx_ref"] + offset_tensor

    target_clearance = float(contact_cfg.get("course_clearance_target", 0.0))
    max_vertical_correction = float(contact_cfg.get("max_course_vertical_correction", 0.8))
    foot_ref = ref["foot_site_ref"].detach().cpu().numpy().astype(np.float64)
    foot_ref_shifted = foot_ref.copy()
    foot_ref_shifted[:, :, 2] += delta_np[:, None]
    foot_world_x = foot_ref[:, :, 0] + (local_x + offset)[:, None]
    foot_terrain_z = course_height_np(foot_world_x, list(course_cfg.get("segments", []))) if bool(course_cfg.get("enabled", False)) else np.zeros_like(foot_world_x)
    clearance = foot_ref_shifted[:, :, 2] - foot_terrain_z
    correction_np = np.zeros((ref["length"],), dtype=np.float32)
    for frame in range(int(ref["length"])):
        current = float(np.min(clearance[frame]))
        correction = target_clearance - current
        correction_np[frame] = float(np.clip(correction, -max_vertical_correction, max_vertical_correction))
    total_delta_np = smooth_reference_correction_np(
        (delta_np + correction_np).astype(np.float32),
        int(contact_cfg.get("course_correction_smoothing_window", 1) or 1),
        float(contact_cfg.get("max_course_vertical_correction_step", 0.0) or 0.0),
    )
    min_foot_clearance = max(target_clearance, float(contact_cfg.get("course_min_clearance", 0.0) or 0.0))
    foot_clearance_lift_np = np.zeros((ref["length"],), dtype=np.float32)
    if min_foot_clearance > -1e-6:
        post_clearance = foot_ref[:, :, 2] + total_delta_np[:, None] - foot_terrain_z
        foot_clearance_lift_np = np.maximum(0.0, min_foot_clearance - np.min(post_clearance, axis=1)).astype(np.float32)
        total_delta_np = (total_delta_np + foot_clearance_lift_np).astype(np.float32)
    min_pelvis_clearance = float(contact_cfg.get("min_pelvis_height_above_course", 0.0) or 0.0)
    pelvis_clearance_lift_np = np.zeros((ref["length"],), dtype=np.float32)
    if min_pelvis_clearance > 0.0:
        pelvis_ty_np = ref["reset_q_ref"][:, RESET_JOINTS.index("pelvis_ty")].detach().cpu().numpy().astype(np.float32)
        pelvis_clearance_np = pelvis_ty_np + total_delta_np - world_height.astype(np.float32)
        pelvis_clearance_lift_np = np.maximum(0.0, min_pelvis_clearance - pelvis_clearance_np).astype(np.float32)
        total_delta_np = (total_delta_np + pelvis_clearance_lift_np).astype(np.float32)
    total_delta = torch.tensor(total_delta_np, dtype=torch.float32, device=device)
    ref["course_height_delta"] = total_delta
    ref["course_vertical_correction"] = total_delta - torch.tensor(delta_np, dtype=torch.float32, device=device)
    ref["course_foot_clearance_lift"] = torch.tensor(foot_clearance_lift_np, dtype=torch.float32, device=device)
    ref["course_pelvis_clearance_lift"] = torch.tensor(pelvis_clearance_lift_np, dtype=torch.float32, device=device)
    ref["reset_q_ref"][:, RESET_JOINTS.index("pelvis_ty")] += total_delta
    ref["q_ref"][:, TRACK_JOINTS.index("pelvis_ty")] += total_delta
    ref["foot_site_ref"][:, :, 2] += total_delta[:, None]
    ref["foot_site_min_z"] = torch.amin(ref["foot_site_ref"][:, :, 2], dim=0)
    foot_ref_post = ref["foot_site_ref"].detach().cpu().numpy().astype(np.float64)
    pelvis_x_post = ref["reset_q_ref"][:, RESET_JOINTS.index("pelvis_tx")].detach().cpu().numpy().astype(np.float64)
    foot_world_post = foot_ref_post.copy()
    foot_world_post[:, :, 0] += pelvis_x_post[:, None]
    control_hz = float(ref.get("control_hz", config.get("control", {}).get("control_hz", 30.0)) or 30.0)
    foot_velocity_post = np.gradient(foot_world_post, 1.0 / control_hz, axis=0).astype(np.float32)
    foot_speed_post = np.linalg.norm(foot_velocity_post[:, :, [0, 2]], axis=2).astype(np.float32)
    if bool(course_cfg.get("enabled", False)):
        foot_terrain_post = course_height_np(foot_world_post[:, :, 0], list(course_cfg.get("segments", [])))
    else:
        foot_terrain_post = np.zeros_like(foot_world_post[:, :, 0])
    foot_clearance_post = foot_world_post[:, :, 2] - foot_terrain_post
    ref["foot_speed_ref"] = torch.tensor(foot_speed_post, dtype=torch.float32, device=device)
    ref["foot_contact_ref"] = torch.tensor(
        (foot_clearance_post < float(contact_cfg.get("z_threshold", 0.025)))
        & (foot_speed_post < float(contact_cfg.get("speed_threshold", 0.4))),
        dtype=torch.bool,
        device=device,
    )
    return ref


def concat_reference_pool(references: list[dict[str, Any]], device: torch.device, config: dict[str, Any]) -> dict[str, Any]:
    if not references:
        raise ValueError("reference pool is empty")
    references = [apply_reference_course_transform(ref, config, device) for ref in references]
    if len(references) == 1:
        ref = references[0]
        terrain_type, terrain_params = parse_terrain_type_and_params(ref["metadata"])
        ref["terrain_type_id"] = torch.full((ref["length"],), terrain_type, dtype=torch.long, device=device)
        ref["terrain_params_tensor"] = torch.tensor([terrain_params] * ref["length"], dtype=torch.float32, device=device)
        ref["reference_id"] = torch.zeros((ref["length"],), dtype=torch.long, device=device)
        label = str(ref["metadata"].get("terrain_id", Path(ref["path"]).stem))
        source_label = str(ref["metadata"].get("source_label", "") or "")
        if source_label and source_label not in label:
            label = f"{label}:{source_label}"
        ref["reference_names"] = [label]
        ref["reference_offsets"] = [{"name": label, "start": 0, "end": int(ref["length"])}]
        return ref

    first = references[0]
    out: dict[str, Any] = {
        "path": "<reference_pool>",
        "metadata": {
            "reference_pool": True,
            "references": [
                {
                    "path": ref["path"],
                    "length": int(ref["length"]),
                    "metadata": ref["metadata"],
                }
                for ref in references
            ],
        },
        "joint_names": first["joint_names"],
        "qpos_indices": first["qpos_indices"],
        "qvel_indices": first["qvel_indices"],
        "reset_qpos_indices": first["reset_qpos_indices"],
        "reset_qvel_indices": first["reset_qvel_indices"],
        "pose_scales": first["pose_scales"],
        "vel_scales": first["vel_scales"],
        "foot_site_names": first["foot_site_names"],
        "foot_site_indices": first["foot_site_indices"],
        "pelvis_tx_qpos": first["pelvis_tx_qpos"],
        "activation_prior_metadata": {"enabled": False, "reference_pool": True},
        "emg_prior_metadata": {"enabled": False, "reference_pool": True},
    }
    cat_keys = [
        "q_ref",
        "dq_ref",
        "reset_q_ref",
        "reset_dq_ref",
        "foot_site_ref",
        "foot_contact_ref",
        "foot_speed_ref",
        "pelvis_tx_ref",
        "activation_prior_ref",
        "activation_prior_action_ref",
        "emg_prior_ref",
    ]
    for key in cat_keys:
        out[key] = torch.cat([ref[key] for ref in references], dim=0)
    out["source_indices"] = torch.arange(int(out["q_ref"].shape[0]), dtype=torch.long, device=device)
    out["length"] = int(out["q_ref"].shape[0])
    out["q_ref_mean"] = torch.mean(out["q_ref"], dim=0)
    out["dq_ref_mean"] = torch.mean(out["dq_ref"], dim=0)
    out["foot_site_min_z"] = torch.amin(out["foot_site_ref"][:, :, 2], dim=0)
    out["activation_prior_mask"] = torch.any(torch.stack([ref["activation_prior_mask"] for ref in references], dim=0), dim=0)
    out["emg_prior_mask"] = torch.any(torch.stack([ref["emg_prior_mask"] for ref in references], dim=0), dim=0)

    terrain_type_rows = []
    terrain_param_rows = []
    reference_id_rows = []
    course_offset_rows = []
    course_height_delta_rows = []
    course_foot_clearance_lift_rows = []
    course_pelvis_clearance_lift_rows = []
    names = []
    offsets = []
    cursor = 0
    for idx, ref in enumerate(references):
        label = str(ref["metadata"].get("terrain_id", Path(ref["path"]).stem))
        source_label = str(ref["metadata"].get("source_label", "") or "")
        if source_label and source_label not in label:
            label = f"{label}:{source_label}"
        names.append(label)
        offsets.append({"name": label, "start": cursor, "end": cursor + int(ref["length"])})
        terrain_type, terrain_params = parse_terrain_type_and_params(ref["metadata"])
        terrain_type_rows.append(torch.full((ref["length"],), terrain_type, dtype=torch.long, device=device))
        terrain_param_rows.append(torch.tensor([terrain_params] * ref["length"], dtype=torch.float32, device=device))
        reference_id_rows.append(torch.full((ref["length"],), idx, dtype=torch.long, device=device))
        course_offset_rows.append(ref["course_offset"])
        course_height_delta_rows.append(ref["course_height_delta"])
        course_foot_clearance_lift_rows.append(ref.get("course_foot_clearance_lift", torch.zeros(ref["length"], dtype=torch.float32, device=device)))
        course_pelvis_clearance_lift_rows.append(ref.get("course_pelvis_clearance_lift", torch.zeros(ref["length"], dtype=torch.float32, device=device)))
        cursor += int(ref["length"])
    out["terrain_type_id"] = torch.cat(terrain_type_rows, dim=0)
    out["terrain_params_tensor"] = torch.cat(terrain_param_rows, dim=0)
    out["reference_id"] = torch.cat(reference_id_rows, dim=0)
    out["course_offset"] = torch.cat(course_offset_rows, dim=0)
    out["course_height_delta"] = torch.cat(course_height_delta_rows, dim=0)
    out["course_foot_clearance_lift"] = torch.cat(course_foot_clearance_lift_rows, dim=0)
    out["course_pelvis_clearance_lift"] = torch.cat(course_pelvis_clearance_lift_rows, dim=0)
    out["reference_names"] = names
    out["reference_offsets"] = offsets
    return out


def load_reference_from_config(
    reference_path: Path,
    model: mujoco.MjModel,
    control_hz: float,
    device: torch.device,
    config: dict[str, Any],
) -> dict[str, Any]:
    pool_cfg = config.get("reference_pool", {})
    paths = pool_cfg.get("paths", [])
    if isinstance(paths, list) and paths:
        refs = [
            load_reference(Path(path).expanduser(), model, control_hz, device, config)
            for path in paths
        ]
        return concat_reference_pool(refs, device, config)
    return concat_reference_pool([load_reference(reference_path, model, control_hz, device, config)], device, config)


def named_weights(config: dict[str, Any], section: str, names: list[str], default: float = 1.0) -> torch.Tensor:
    values = config.get("imitation", {}).get(section, {})
    return torch.tensor([float(values.get(name, default)) for name in names], dtype=torch.float32)


def reference_phase_windows_for_names(reference: dict[str, Any], names: list[str]) -> list[dict[str, int]]:
    if not names:
        return []
    wanted = set(str(name) for name in names)
    windows: list[dict[str, int]] = []
    missing = set(wanted)
    for item in reference.get("reference_offsets", []):
        name = str(item.get("name", ""))
        if name not in wanted:
            continue
        windows.append({"start": int(item["start"]), "end": int(item["end"])})
        missing.discard(name)
    if missing:
        available = [str(item.get("name", "")) for item in reference.get("reference_offsets", [])]
        raise ValueError(f"reference_pool_schedule contains unknown references {sorted(missing)}; available={available}")
    return windows


def reference_pool_schedule_for_step(
    config: dict[str, Any],
    reference: dict[str, Any],
    global_step: int,
    run_start_global_step: int,
) -> dict[str, Any] | None:
    schedule = config.get("reference_pool_schedule", [])
    if not isinstance(schedule, list) or not schedule:
        return None
    schedule_step = int(global_step)
    if str(config.get("reward_schedule_mode", "relative")) == "relative":
        schedule_step = max(0, int(global_step) - int(run_start_global_step))
    current = schedule[0]
    for item in sorted(schedule, key=lambda x: int(x.get("after_steps", 0))):
        if schedule_step >= int(item.get("after_steps", 0)):
            current = item
    names = [str(name) for name in current.get("references", [])]
    return {
        "name": str(current.get("name", "")),
        "after_steps": int(current.get("after_steps", 0)),
        "references": names,
        "phase_windows": reference_phase_windows_for_names(reference, names),
    }


class MJWarpMuscleRunner:
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
        self.episode_steps = int(config["reset"]["episode_steps"])
        self.safe_pelvis_height = float(config["reset"]["safe_pelvis_height"])
        self.max_abs_pelvis_tilt = float(config["reset"].get("max_abs_pelvis_tilt", 0.65))
        self.max_abs_pelvis_tilt_error = float(config["reset"].get("max_abs_pelvis_tilt_error", self.max_abs_pelvis_tilt))
        self.max_abs_qvel = float(config["reset"].get("max_abs_qvel", 80.0))
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
        self.phase_start = int(config["reset"].get("phase_start", 0))
        self.phase_end = int(config["reset"].get("phase_end", 0) or reference["length"])
        self.phase_choices = self.build_phase_choices(
            config["reset"].get("phase_windows", []),
            config["reset"].get("phase_indices", []),
            int(config["reset"].get("phase_index_jitter", 0) or 0),
            int(reference["length"]),
        )
        self.reward_weights = {k: float(v) for k, v in config["reward"].items()}
        self.activation_prior_execution_mix = float(config.get("activation_prior", {}).get("execution_mix", 0.0))
        self.pose_weights = named_weights(config, "joint_pose_weights", TRACK_JOINTS).to(device)
        self.vel_weights = named_weights(config, "joint_vel_weights", TRACK_JOINTS).to(device)
        self.foot_site_weights = named_weights(config, "foot_site_weights", FOOT_SITE_NAMES).to(device)
        self.foot_x_scale = float(config.get("imitation", {}).get("foot_x_scale", 0.18))
        self.foot_z_scale = float(config.get("imitation", {}).get("foot_z_scale", 0.05))
        self.pelvis_tx_vel_scale = float(config.get("imitation", {}).get("pelvis_tx_vel_scale", 0.5))
        self.pelvis_ty_vel_scale = float(config.get("imitation", {}).get("pelvis_ty_vel_scale", 0.25))
        self.pelvis_tangent_vel_scale = float(config.get("imitation", {}).get("pelvis_tangent_vel_scale", self.pelvis_tx_vel_scale))
        self.pelvis_normal_vel_scale = float(config.get("imitation", {}).get("pelvis_normal_vel_scale", self.pelvis_ty_vel_scale))
        self.reference_future_steps = max(0, int(config.get("imitation", {}).get("reference_future_steps", 0) or 0))
        self.reference_reward_future_steps = max(
            0,
            int(config.get("imitation", {}).get("reference_reward_future_steps", 0) or 0),
        )
        self.post_reference_enabled = post_reference_enabled(config)
        self.reference_valid_steps = post_reference_valid_steps(reference, config)
        self.reference_swing_foot_weight = max(1.0, float(config.get("imitation", {}).get("reference_swing_foot_weight", 1.0)))
        contact_cfg = config.get("reference_contact", {})
        self.contact_slip_scale = float(contact_cfg.get("slip_scale", 0.04))
        self.contact_z_scale = float(contact_cfg.get("z_scale", self.foot_z_scale))
        heuristic_cfg = config.get("heuristic_reward", {})
        self.swing_knee_flex_target = float(heuristic_cfg.get("swing_knee_flex_target", 0.55))
        self.swing_knee_flex_scale = float(heuristic_cfg.get("swing_knee_flex_scale", 0.12))
        self.swing_hip_flex_target = float(heuristic_cfg.get("swing_hip_flex_target", 0.18))
        self.swing_hip_flex_scale = float(heuristic_cfg.get("swing_hip_flex_scale", 0.08))
        self.swing_foot_forward_target = float(heuristic_cfg.get("swing_foot_forward_target", 0.18))
        self.swing_foot_forward_scale = float(heuristic_cfg.get("swing_foot_forward_scale", 0.08))
        bias_cfg = config.get("muscle_bias", {})
        self.muscle_bias_enabled = bool(bias_cfg.get("enabled", False))
        self.muscle_bias_add = {str(k): float(v) for k, v in bias_cfg.get("swing_add", {}).items()}
        self.muscle_bias_add_late = {str(k): float(v) for k, v in bias_cfg.get("swing_add_late", {}).items()}
        self.muscle_bias_sub = {str(k): float(v) for k, v in bias_cfg.get("swing_sub", {}).items()}
        self.muscle_bias_scale = float(bias_cfg.get("scale", 1.0))
        self.muscle_bias_late_hip_threshold = float(bias_cfg.get("late_hip_threshold", 0.0))
        self.right_bias_add_indices, self.right_bias_add_values = self.build_muscle_bias_tensors(self.muscle_bias_add, "r")
        self.left_bias_add_indices, self.left_bias_add_values = self.build_muscle_bias_tensors(self.muscle_bias_add, "l")
        self.right_bias_add_late_indices, self.right_bias_add_late_values = self.build_muscle_bias_tensors(
            self.muscle_bias_add_late, "r"
        )
        self.left_bias_add_late_indices, self.left_bias_add_late_values = self.build_muscle_bias_tensors(
            self.muscle_bias_add_late, "l"
        )
        self.right_bias_sub_indices, self.right_bias_sub_values = self.build_muscle_bias_tensors(self.muscle_bias_sub, "r")
        self.left_bias_sub_indices, self.left_bias_sub_values = self.build_muscle_bias_tensors(self.muscle_bias_sub, "l")
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
        post_cfg = post_reference_config(config)
        post_vx = post_cfg.get("target_vx", None)
        if post_vx is None:
            vx_start = max(0, int(self.reference_valid_steps) - 10)
            vx_end = max(vx_start + 1, int(self.reference_valid_steps))
            post_vx_tensor = reference["reset_dq_ref"][vx_start:vx_end, RESET_JOINTS.index("pelvis_tx")]
            self.post_reference_target_vx = float(torch.mean(post_vx_tensor).detach().cpu().item())
        else:
            self.post_reference_target_vx = float(post_vx)
        perturb_cfg = config.get("perturbation", {})
        self.push_interval_steps = int(perturb_cfg.get("push_interval_steps", 0) or 0)
        self.push_probability = float(perturb_cfg.get("push_probability", 0.0))
        self.push_pelvis_qvel_std = float(perturb_cfg.get("push_pelvis_qvel_std", 0.0))
        self.push_joint_qvel_std = float(perturb_cfg.get("push_joint_qvel_std", 0.0))
        self.pelvis_track_indices = torch.tensor(
            [TRACK_JOINTS.index("pelvis_ty"), TRACK_JOINTS.index("pelvis_tilt")],
            dtype=torch.long,
            device=device,
        )
        self.hip_track_indices = torch.tensor(
            [TRACK_JOINTS.index("hip_flexion_r"), TRACK_JOINTS.index("hip_flexion_l")],
            dtype=torch.long,
            device=device,
        )
        self.knee_track_indices = torch.tensor(
            [TRACK_JOINTS.index("knee_angle_r"), TRACK_JOINTS.index("knee_angle_l")],
            dtype=torch.long,
            device=device,
        )
        self.ankle_track_indices = torch.tensor(
            [TRACK_JOINTS.index("ankle_angle_r"), TRACK_JOINTS.index("ankle_angle_l")],
            dtype=torch.long,
            device=device,
        )
        self.mtp_track_indices = torch.tensor(
            [TRACK_JOINTS.index("mtp_angle_r"), TRACK_JOINTS.index("mtp_angle_l")],
            dtype=torch.long,
            device=device,
        )
        self.right_swing_joint_indices = torch.tensor(
            [TRACK_JOINTS.index("hip_flexion_r"), TRACK_JOINTS.index("knee_angle_r")],
            dtype=torch.long,
            device=device,
        )
        self.left_swing_joint_indices = torch.tensor(
            [TRACK_JOINTS.index("hip_flexion_l"), TRACK_JOINTS.index("knee_angle_l")],
            dtype=torch.long,
            device=device,
        )
        self.right_foot_error_cols = torch.tensor([0, 1, 2, 3], dtype=torch.long, device=device)
        self.left_foot_error_cols = torch.tensor([4, 5, 6, 7], dtype=torch.long, device=device)
        self.right_foot_z_error_cols = torch.tensor([1, 3], dtype=torch.long, device=device)
        self.left_foot_z_error_cols = torch.tensor([5, 7], dtype=torch.long, device=device)
        self.right_limb_track_indices = torch.tensor(
            [
                TRACK_JOINTS.index("hip_flexion_r"),
                TRACK_JOINTS.index("knee_angle_r"),
                TRACK_JOINTS.index("ankle_angle_r"),
            ],
            dtype=torch.long,
            device=device,
        )
        self.left_limb_track_indices = torch.tensor(
            [
                TRACK_JOINTS.index("hip_flexion_l"),
                TRACK_JOINTS.index("knee_angle_l"),
                TRACK_JOINTS.index("ankle_angle_l"),
            ],
            dtype=torch.long,
            device=device,
        )

        wp.init()
        self.warp_model = mjw.put_model(model)
        self.warp_data = mjw.put_data(model, data, nworld=nworld, nconmax=nconmax, njmax=njmax)
        self.qpos = wp.to_torch(self.warp_data.qpos)
        self.qvel = wp.to_torch(self.warp_data.qvel)
        self.act = wp.to_torch(self.warp_data.act)
        self.ctrl = wp.to_torch(self.warp_data.ctrl)
        self.site_xpos = wp.to_torch(self.warp_data.site_xpos)
        self.qacc_warmstart = wp.to_torch(self.warp_data.qacc_warmstart)
        self.time = wp.to_torch(self.warp_data.time)

        self.key_qpos = torch.tensor(data.qpos.copy(), dtype=torch.float32, device=device)
        self.key_qvel = torch.tensor(data.qvel.copy(), dtype=torch.float32, device=device)
        self.pelvis_tx_qpos = int(model.jnt_qposadr[joint_id(model, "pelvis_tx")])
        self.pelvis_tx_qvel = int(model.jnt_dofadr[joint_id(model, "pelvis_tx")])
        self.pelvis_ty_qpos = int(model.jnt_qposadr[joint_id(model, "pelvis_ty")])
        self.pelvis_ty_qvel = int(model.jnt_dofadr[joint_id(model, "pelvis_ty")])
        self.pelvis_tilt_qpos = int(model.jnt_qposadr[joint_id(model, "pelvis_tilt")])
        self.foot_site_indices = reference["foot_site_indices"]
        self.push_joint_qvel_indices = reference["qvel_indices"]
        self.right_actuator_indices, self.left_actuator_indices = self.build_actuator_pair_indices()

        equality_specs = joint_equality_specs_np(model)
        self.eq_qpos1 = torch.tensor([v[0] for v in equality_specs], dtype=torch.long, device=device)
        self.eq_qpos2 = torch.tensor([v[1] for v in equality_specs], dtype=torch.long, device=device)
        self.eq_qvel1 = torch.tensor([v[2] for v in equality_specs], dtype=torch.long, device=device)
        self.eq_qvel2 = torch.tensor([v[3] for v in equality_specs], dtype=torch.long, device=device)
        eq_poly = np.stack([v[4] for v in equality_specs], axis=0) if equality_specs else np.zeros((0, 5), dtype=np.float32)
        self.eq_poly = torch.tensor(eq_poly, dtype=torch.float32, device=device)

        self.phase_idx = torch.zeros(nworld, dtype=torch.long, device=device)
        self.episode_step = torch.zeros(nworld, dtype=torch.long, device=device)
        self.prev_activation = torch.full((nworld, model.nu), self.initial_activation, dtype=torch.float32, device=device)
        self.state_history_prev_steps = frame_stack_prev_steps(config)
        self.state_history_feature_dim = frame_stack_feature_dim(config, nq=int(model.nq), nv=int(model.nv), na=int(model.na))
        self.state_history_obs = torch.zeros(
            (nworld, self.state_history_prev_steps, self.state_history_feature_dim),
            dtype=torch.float32,
            device=device,
        )
        self.last_activation_bias_abs = torch.zeros(nworld, dtype=torch.float32, device=device)
        self.last_activation_bias_signed = torch.zeros(nworld, dtype=torch.float32, device=device)
        self.episode_return = torch.zeros(nworld, dtype=torch.float32, device=device)
        self.episode_length = torch.zeros(nworld, dtype=torch.float32, device=device)
        self.reset(torch.ones(nworld, dtype=torch.bool, device=device))

    def set_reference_curriculum(self, *, phase_lead_steps: int, phase_tolerance_steps: int, swing_exaggeration_scale: float) -> None:
        self.reference_phase_lead_steps = int(phase_lead_steps)
        self.reference_phase_tolerance_steps = max(0, int(phase_tolerance_steps))
        self.reference_swing_exaggeration_scale = max(1.0, float(swing_exaggeration_scale))

    def set_phase_choices_from_windows(self, phase_windows: list[Any]) -> None:
        self.phase_choices = self.build_phase_choices(
            phase_windows,
            [],
            int(self.config["reset"].get("phase_index_jitter", 0) or 0),
            int(self.reference["length"]),
        )

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

    def build_actuator_pair_indices(self) -> tuple[torch.Tensor, torch.Tensor]:
        right_indices: list[int] = []
        left_indices: list[int] = []
        for actuator_id in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id) or ""
            if not name.endswith("_r"):
                continue
            left_name = f"{name[:-2]}_l"
            left_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, left_name)
            if left_id < 0:
                continue
            right_indices.append(int(actuator_id))
            left_indices.append(int(left_id))
        if not right_indices:
            right_indices = list(range(self.model.nu))
            left_indices = list(range(self.model.nu))
        return (
            torch.tensor(right_indices, dtype=torch.long, device=self.device),
            torch.tensor(left_indices, dtype=torch.long, device=self.device),
        )

    def build_muscle_bias_tensors(self, base_values: dict[str, float], side: str) -> tuple[torch.Tensor, torch.Tensor]:
        indices: list[int] = []
        values: list[float] = []
        for base_name, value in base_values.items():
            actuator_name = f"{base_name}_{side}"
            idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            if idx < 0:
                print(f"[muscle_bias] missing actuator: {actuator_name}", flush=True)
                continue
            indices.append(int(idx))
            values.append(float(value))
        return (
            torch.tensor(indices, dtype=torch.long, device=self.device),
            torch.tensor(values, dtype=torch.float32, device=self.device),
        )

    def apply_muscle_bias(self, activation: torch.Tensor) -> torch.Tensor:
        if not self.muscle_bias_enabled:
            self.last_activation_bias_abs.zero_()
            self.last_activation_bias_signed.zero_()
            return activation
        stance = self.reference["foot_contact_ref"][reference_index(self.phase_idx, self.reference, self.config)]
        right_swing = ~(stance[:, 0] | stance[:, 1])
        left_swing = ~(stance[:, 2] | stance[:, 3])
        bias = torch.zeros_like(activation)
        hip_r = self.right_swing_joint_indices[0]
        hip_l = self.left_swing_joint_indices[0]
        for mask, hip_idx, add_idx, add_val, late_idx, late_val, sub_idx, sub_val in (
            (
                right_swing,
                hip_r,
                self.right_bias_add_indices,
                self.right_bias_add_values,
                self.right_bias_add_late_indices,
                self.right_bias_add_late_values,
                self.right_bias_sub_indices,
                self.right_bias_sub_values,
            ),
            (
                left_swing,
                hip_l,
                self.left_bias_add_indices,
                self.left_bias_add_values,
                self.left_bias_add_late_indices,
                self.left_bias_add_late_values,
                self.left_bias_sub_indices,
                self.left_bias_sub_values,
            ),
        ):
            if not bool(mask.any().item()):
                continue
            rows = mask.nonzero(as_tuple=False).flatten()
            if add_idx.numel() > 0:
                bias[rows[:, None], add_idx[None, :]] += self.muscle_bias_scale * add_val[None, :]
            late_rows = rows[self.qpos[rows, self.reference["qpos_indices"][hip_idx]] > self.muscle_bias_late_hip_threshold]
            if late_idx.numel() > 0 and late_rows.numel() > 0:
                bias[late_rows[:, None], late_idx[None, :]] += self.muscle_bias_scale * late_val[None, :]
            if sub_idx.numel() > 0:
                bias[rows[:, None], sub_idx[None, :]] -= self.muscle_bias_scale * sub_val[None, :]
        biased_activation = torch.clamp(activation + bias, 0.0, 1.0)
        actual_bias = biased_activation - activation
        self.last_activation_bias_abs = torch.mean(torch.abs(actual_bias), dim=1)
        self.last_activation_bias_signed = torch.mean(actual_bias, dim=1)
        return biased_activation

    @property
    def obs_dim(self) -> int:
        foot_dim = foot_obs_feature_dim(self.config)
        history_dim = frame_stack_prev_steps(self.config) * frame_stack_feature_dim(
            self.config,
            nq=int(self.model.nq),
            nv=int(self.model.nv),
            na=int(self.model.na),
        )
        future_dim = self.reference_future_steps * (len(TRACK_JOINTS) + foot_dim)
        return (
            self.model.nq
            + self.model.nv
            + self.model.na
            + 2 * len(TRACK_JOINTS)
            + 2
            + foot_dim
            + reference_obs_extra_dim(self.config)
            + history_dim
            + future_dim
            + terrain_preview_dim(self.config)
        )

    @property
    def act_dim(self) -> int:
        return self.model.nu

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

    def reset(self, mask: torch.Tensor) -> None:
        if not bool(mask.any().item()):
            return
        rows = torch.nonzero(mask, as_tuple=False).flatten()
        count = int(rows.numel())
        if self.phase_choices is not None and int(self.phase_choices.numel()) > 0:
            choice_idx = torch.randint(0, int(self.phase_choices.numel()), (count,), generator=self.rng, device=self.device)
            phase = self.phase_choices[choice_idx]
        else:
            phase_low = max(0, min(self.phase_start, int(self.reference["length"]) - 1))
            phase_high = max(phase_low + 1, min(self.phase_end, int(self.reference["length"])))
            phase = torch.randint(phase_low, phase_high, (count,), generator=self.rng, device=self.device)
        self.phase_idx[rows] = phase
        self.qpos[rows] = self.key_qpos
        self.qvel[rows] = self.key_qvel
        self.qpos[rows[:, None], self.reference["reset_qpos_indices"][None, :]] = self.reference["reset_q_ref"][phase]
        self.qvel[rows[:, None], self.reference["reset_qvel_indices"][None, :]] = self.reference["reset_dq_ref"][phase]
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
        if self.initial_activation_high > self.initial_activation_low:
            initial_activation = self.initial_activation_low + (
                torch.rand((count, self.model.nu), generator=self.rng, device=self.device)
                * (self.initial_activation_high - self.initial_activation_low)
            )
        else:
            initial_activation = torch.full((count, self.model.nu), self.initial_activation, dtype=torch.float32, device=self.device)
        self.ctrl[rows] = initial_activation
        self.act[rows] = initial_activation
        self.qacc_warmstart[rows] = 0.0
        self.time[rows] = 0.0
        self.prev_activation[rows] = initial_activation
        self.episode_step[rows] = 0
        self.episode_return[rows] = 0.0
        self.episode_length[rows] = 0.0
        self.apply_joint_equalities(rows)
        if hasattr(self, "state_history_obs") and self.state_history_prev_steps > 0:
            reset_state = self.current_state_history_features()[rows]
            self.state_history_obs[rows] = reset_state.unsqueeze(1).expand(-1, self.state_history_prev_steps, -1)

    def obs(self) -> torch.Tensor:
        return build_policy_obs_tensor(
            qpos=self.qpos,
            qvel=self.qvel,
            act=self.act,
            site_xpos=self.site_xpos,
            phase_idx=self.phase_idx,
            episode_step=self.episode_step,
            pelvis_tx_qpos=self.pelvis_tx_qpos,
            foot_site_indices=self.foot_site_indices,
            reference=self.reference,
            config=self.config,
            state_history_obs=self.state_history_obs if self.state_history_prev_steps > 0 else None,
        )

    def current_state_history_features(self) -> torch.Tensor:
        return build_policy_state_feature_tensor(
            qpos=self.qpos,
            qvel=self.qvel,
            act=self.act,
            site_xpos=self.site_xpos,
            phase_idx=self.phase_idx,
            pelvis_tx_qpos=self.pelvis_tx_qpos,
            foot_site_indices=self.foot_site_indices,
            reference=self.reference,
            config=self.config,
        )

    def target_phase_idx(self) -> torch.Tensor:
        return reference_index(self.phase_idx + int(self.reference_phase_lead_steps), self.reference, self.config)

    def reference_valid_mask(self) -> torch.Tensor:
        if not self.post_reference_enabled:
            return torch.ones((self.nworld,), dtype=torch.bool, device=self.device)
        return self.episode_step < int(self.reference_valid_steps)

    def reference_q_dq(self, phases: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return reference_q_dq_tensor(
            self.reference,
            phases,
            swing_exaggeration_scale=float(self.reference_swing_exaggeration_scale),
        )

    def reference_foot(self, phases: torch.Tensor) -> torch.Tensor:
        return reference_foot_tensor(
            self.reference,
            phases,
            swing_exaggeration_scale=float(self.reference_swing_exaggeration_scale),
        )

    def best_reference_phase(
        self,
        q: torch.Tensor,
        dq: torch.Tensor,
        foot_rel_x: torch.Tensor,
        foot_z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        target = self.target_phase_idx()
        tolerance = int(self.reference_phase_tolerance_steps)
        if tolerance <= 0:
            ref_q, ref_dq = self.reference_q_dq(target)
            return target, ref_q, ref_dq, self.reference_foot(target)

        best_phase = target
        best_ref_q, best_ref_dq = self.reference_q_dq(target)
        best_ref_foot = self.reference_foot(target)
        best_err = self.reference_match_error(q, dq, foot_rel_x, foot_z, best_ref_q, best_ref_dq, best_ref_foot)
        for offset in range(-tolerance, tolerance + 1):
            if offset == 0:
                continue
            phase = reference_index(target + offset, self.reference, self.config)
            ref_q, ref_dq = self.reference_q_dq(phase)
            ref_foot = self.reference_foot(phase)
            err = self.reference_match_error(q, dq, foot_rel_x, foot_z, ref_q, ref_dq, ref_foot)
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_phase = torch.where(better, phase, best_phase)
            best_ref_q = torch.where(better[:, None], ref_q, best_ref_q)
            best_ref_dq = torch.where(better[:, None], ref_dq, best_ref_dq)
            best_ref_foot = torch.where(better[:, None, None], ref_foot, best_ref_foot)
        return best_phase, best_ref_q, best_ref_dq, best_ref_foot

    def reference_match_error(
        self,
        q: torch.Tensor,
        dq: torch.Tensor,
        foot_rel_x: torch.Tensor,
        foot_z: torch.Tensor,
        ref_q: torch.Tensor,
        ref_dq: torch.Tensor,
        ref_foot: torch.Tensor,
    ) -> torch.Tensor:
        pose_sq = torch.square((q - ref_q) / self.reference["pose_scales"])
        vel_sq = torch.square((dq - ref_dq) / self.reference["vel_scales"])
        pose_err = torch.sum(pose_sq * self.pose_weights, dim=1) / torch.clamp(torch.sum(self.pose_weights), min=1e-6)
        vel_err = torch.sum(vel_sq * self.vel_weights, dim=1) / torch.clamp(torch.sum(self.vel_weights), min=1e-6)
        foot_z_err = torch.mean(torch.square((foot_z - ref_foot[:, :, 2]) / self.foot_z_scale), dim=1)
        foot_x_err = torch.mean(torch.square((foot_rel_x - ref_foot[:, :, 0]) / self.foot_x_scale), dim=1)
        return pose_err + 0.25 * vel_err + 0.25 * foot_z_err + 0.1 * foot_x_err

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        policy_activation = muscle_action_to_activation(action)
        mix = max(0.0, min(1.0, self.activation_prior_execution_mix))
        if mix > 0.0 and "activation_prior_ref" in self.reference:
            prior_phase = reference_index(self.phase_idx, self.reference, self.config)
            prior_activation = self.reference["activation_prior_ref"][prior_phase]
            activation = torch.clamp((1.0 - mix) * policy_activation + mix * prior_activation, 0.0, 1.0)
        else:
            activation = policy_activation
        activation = self.apply_muscle_bias(activation)
        prev_state_history = self.current_state_history_features() if self.state_history_prev_steps > 0 else None
        prev_foot = self.site_xpos[:, self.foot_site_indices, :].clone()
        prev_terrain_height = terrain_height_for_world_x_tensor(prev_foot[:, :, 0], self.phase_idx, self.reference, self.config)
        prev_foot_contact = (
            prev_foot[:, :, 2] - prev_terrain_height
        ) < float(self.config.get("reference_contact", {}).get("z_threshold", 0.025))
        self.ctrl.copy_(activation)
        for _ in range(self.frame_skip):
            mjw.step(self.warp_model, self.warp_data)
        wp.synchronize()
        if self.post_reference_enabled:
            self.phase_idx = self.phase_idx + 1
        else:
            self.phase_idx = (self.phase_idx + 1) % int(self.reference["length"])
        self.episode_step += 1
        self.maybe_apply_push()
        if self.state_history_prev_steps > 0 and prev_state_history is not None:
            if self.state_history_prev_steps > 1:
                self.state_history_obs[:, 1:] = self.state_history_obs[:, :-1].clone()
            self.state_history_obs[:, 0] = prev_state_history
        reward, terms = self.reward(action, activation, prev_foot)
        terrain_height = current_terrain_height_tensor(self.qpos, self.phase_idx, self.reference, self.config)
        pelvis_height_above_terrain = self.qpos[:, self.pelvis_ty_qpos] - terrain_height
        low_height = pelvis_height_above_terrain < self.safe_pelvis_height
        ref_tilt = self.reference_q_dq(self.target_phase_idx())[0][:, TRACK_JOINTS.index("pelvis_tilt")]
        tilt_error = torch.where(
            self.reference_valid_mask(),
            self.qpos[:, self.pelvis_tilt_qpos] - ref_tilt,
            self.qpos[:, self.pelvis_tilt_qpos],
        )
        bad_tilt = torch.abs(tilt_error) > self.max_abs_pelvis_tilt_error
        fallen = low_height | bad_tilt
        qvel_bad = torch.amax(torch.abs(self.qvel), dim=1) > self.max_abs_qvel
        truncated = self.episode_step >= self.episode_steps
        done = fallen | qvel_bad | truncated
        self.episode_return += reward
        self.episode_length += 1.0
        terms["done"] = done.float()
        terms["fall_done"] = fallen.float()
        terms["low_height_done"] = low_height.float()
        terms["tilt_done"] = bad_tilt.float()
        terms["qvel_done"] = qvel_bad.float()
        terms["done_count"] = done.float()
        terms["episode_return_done_sum"] = torch.where(done, self.episode_return, torch.zeros_like(reward))
        terms["episode_length_done_sum"] = torch.where(done, self.episode_length, torch.zeros_like(reward))
        self.prev_activation.copy_(activation)
        if bool(done.any().item()):
            self.reset(done)
        return self.obs(), reward, done, terms

    def maybe_apply_push(self) -> None:
        if self.push_interval_steps <= 0:
            return
        if self.push_probability <= 0.0:
            return
        due = (self.episode_step % self.push_interval_steps) == 0
        if not bool(due.any().item()):
            return
        rows = due & (torch.rand(self.nworld, dtype=torch.float32, device=self.device, generator=self.rng) < self.push_probability)
        if not bool(rows.any().item()):
            return
        row_idx = torch.nonzero(rows, as_tuple=False).flatten()
        if self.push_pelvis_qvel_std > 0.0:
            pelvis_cols = torch.tensor(
                [self.pelvis_tx_qvel, int(self.model.jnt_dofadr[joint_id(self.model, "pelvis_ty")]), int(self.model.jnt_dofadr[joint_id(self.model, "pelvis_tilt")])],
                dtype=torch.long,
                device=self.device,
            )
            impulse = torch.randn(
                (int(row_idx.numel()), int(pelvis_cols.numel())),
                dtype=torch.float32,
                device=self.device,
                generator=self.rng,
            ) * self.push_pelvis_qvel_std
            self.qvel[row_idx.unsqueeze(1), pelvis_cols.unsqueeze(0)] += impulse
        if self.push_joint_qvel_std > 0.0:
            impulse = torch.randn(
                (int(row_idx.numel()), int(self.push_joint_qvel_indices.numel())),
                dtype=torch.float32,
                device=self.device,
                generator=self.rng,
            ) * self.push_joint_qvel_std
            self.qvel[row_idx.unsqueeze(1), self.push_joint_qvel_indices.unsqueeze(0)] += impulse

    def reward(
        self,
        action: torch.Tensor,
        activation: torch.Tensor,
        prev_foot: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        q = self.qpos[:, self.reference["qpos_indices"]]
        dq = self.qvel[:, self.reference["qvel_indices"]]
        foot = self.site_xpos[:, self.foot_site_indices, :]
        foot_rel_x = foot[:, :, 0] - self.qpos[:, self.pelvis_tx_qpos].unsqueeze(1)
        foot_z = foot[:, :, 2]
        ref_phase, ref_q, ref_dq, ref_foot = self.best_reference_phase(q, dq, foot_rel_x, foot_z)
        reference_valid_bool = self.reference_valid_mask()
        reference_valid = reference_valid_bool.float()
        reference_valid_col = reference_valid.unsqueeze(1)
        pose_sq = torch.square((q - ref_q) / self.reference["pose_scales"])
        vel_sq = torch.square((dq - ref_dq) / self.reference["vel_scales"])
        pose_err = torch.sum(pose_sq * self.pose_weights, dim=1) / torch.clamp(torch.sum(self.pose_weights), min=1e-6)
        vel_err = torch.sum(vel_sq * self.vel_weights, dim=1) / torch.clamp(torch.sum(self.vel_weights), min=1e-6)
        ref_pose = torch.exp(-pose_err)
        ref_vel = torch.exp(-vel_err)
        pelvis_pose_ref = self.group_pose_reward(pose_sq, self.pelvis_track_indices)
        hip_pose_ref = self.group_pose_reward(pose_sq, self.hip_track_indices)
        knee_pose_ref = self.group_pose_reward(pose_sq, self.knee_track_indices)
        knee_vel_ref = self.group_vel_reward(vel_sq, self.knee_track_indices)
        ankle_pose_ref = self.group_pose_reward(pose_sq, self.ankle_track_indices)
        mtp_pose_ref = self.group_pose_reward(pose_sq, self.mtp_track_indices)

        ref_foot_rel_x = ref_foot[:, :, 0]
        ref_foot_z = ref_foot[:, :, 2]
        site_weight_sum = torch.clamp(torch.sum(self.foot_site_weights), min=1e-6)
        foot_error_flat = torch.stack(
            [
                foot_rel_x[:, 0] - ref_foot_rel_x[:, 0],
                foot_z[:, 0] - ref_foot_z[:, 0],
                foot_rel_x[:, 1] - ref_foot_rel_x[:, 1],
                foot_z[:, 1] - ref_foot_z[:, 1],
                foot_rel_x[:, 2] - ref_foot_rel_x[:, 2],
                foot_z[:, 2] - ref_foot_z[:, 2],
                foot_rel_x[:, 3] - ref_foot_rel_x[:, 3],
                foot_z[:, 3] - ref_foot_z[:, 3],
            ],
            dim=1,
        )
        ref_stance = self.reference["foot_contact_ref"][ref_phase]
        swing_foot_weights = self.foot_site_weights.unsqueeze(0) * torch.where(
            ref_stance,
            torch.ones_like(self.foot_site_weights).unsqueeze(0),
            torch.full_like(self.foot_site_weights, self.reference_swing_foot_weight).unsqueeze(0),
        )
        foot_weights_flat = torch.repeat_interleave(swing_foot_weights, 2, dim=1)
        foot_weight_sum_flat = torch.clamp(torch.sum(foot_weights_flat, dim=1), min=1e-6)
        foot_z_err = torch.sum(
            torch.square((foot_z - ref_foot_z) / self.foot_z_scale) * self.foot_site_weights,
            dim=1,
        ) / site_weight_sum
        foot_x_err = torch.sum(
            torch.square((foot_rel_x - ref_foot_rel_x) / self.foot_x_scale) * self.foot_site_weights,
            dim=1,
        ) / site_weight_sum
        foot_z_ref = torch.exp(-foot_z_err)
        foot_x_ref = torch.exp(-foot_x_err)
        ref_pelvis_tx_vel = self.reference["reset_dq_ref"][ref_phase, RESET_JOINTS.index("pelvis_tx")]
        ref_pelvis_ty_vel = self.reference["reset_dq_ref"][ref_phase, RESET_JOINTS.index("pelvis_ty")]
        ref_pelvis_tx_vel = torch.where(
            reference_valid_bool,
            ref_pelvis_tx_vel,
            torch.full_like(ref_pelvis_tx_vel, float(self.post_reference_target_vx)),
        )
        ref_pelvis_ty_vel = torch.where(
            reference_valid_bool,
            ref_pelvis_ty_vel,
            torch.zeros_like(ref_pelvis_ty_vel),
        )
        pelvis_tx_vel_err = torch.square((self.qvel[:, self.pelvis_tx_qvel] - ref_pelvis_tx_vel) / self.pelvis_tx_vel_scale)
        pelvis_tx_vel_ref = torch.exp(-pelvis_tx_vel_err)
        pelvis_ty_vel_err = torch.square((self.qvel[:, self.pelvis_ty_qvel] - ref_pelvis_ty_vel) / self.pelvis_ty_vel_scale)
        pelvis_ty_vel_ref = torch.exp(-pelvis_ty_vel_err)
        pelvis_slope = current_terrain_slope_tensor(self.qpos, self.phase_idx, self.reference, self.config)
        pelvis_slope_denom = torch.sqrt(1.0 + torch.square(pelvis_slope))
        pelvis_tangent_vel = (self.qvel[:, self.pelvis_tx_qvel] + pelvis_slope * self.qvel[:, self.pelvis_ty_qvel]) / pelvis_slope_denom
        pelvis_normal_vel = (-pelvis_slope * self.qvel[:, self.pelvis_tx_qvel] + self.qvel[:, self.pelvis_ty_qvel]) / pelvis_slope_denom
        ref_pelvis_tangent_vel = (ref_pelvis_tx_vel + pelvis_slope * ref_pelvis_ty_vel) / pelvis_slope_denom
        ref_pelvis_normal_vel = (-pelvis_slope * ref_pelvis_tx_vel + ref_pelvis_ty_vel) / pelvis_slope_denom
        pelvis_tangent_vel_err = torch.square((pelvis_tangent_vel - ref_pelvis_tangent_vel) / self.pelvis_tangent_vel_scale)
        pelvis_normal_vel_err = torch.square((pelvis_normal_vel - ref_pelvis_normal_vel) / self.pelvis_normal_vel_scale)
        pelvis_tangent_vel_ref = torch.exp(-pelvis_tangent_vel_err)
        pelvis_normal_vel_ref = torch.exp(-pelvis_normal_vel_err)

        if bool(self.config.get("terrain_course", {}).get("enabled", False)):
            foot_terrain_height = course_height_tensor(foot[:, :, 0], self.config)
        else:
            terrain_type_id = self.reference.get("terrain_type_id")
            terrain_params = self.reference.get("terrain_params_tensor")
            if terrain_type_id is not None and terrain_params is not None:
                foot_terrain_height = terrain_height_from_params(
                    foot[:, :, 0],
                    terrain_type_id[ref_phase],
                    terrain_params[ref_phase],
                )
            else:
                foot_terrain_height = torch.zeros_like(foot_z)
        current_stance = (foot_z - foot_terrain_height) < float(self.config.get("reference_contact", {}).get("z_threshold", 0.025))
        stance = torch.where(reference_valid_bool[:, None], ref_stance, current_stance)
        swing = ~stance
        current_dx = foot[:, :, 0] - prev_foot[:, :, 0]
        current_dz = foot[:, :, 2] - prev_foot[:, :, 2]
        foot_slope = terrain_slope_for_world_x_tensor(foot[:, :, 0], self.phase_idx, self.reference, self.config)
        foot_slope_denom = torch.sqrt(1.0 + torch.square(foot_slope))
        current_tangent_delta = (current_dx + foot_slope * current_dz) / foot_slope_denom
        current_normal_delta = (-foot_slope * current_dx + current_dz) / foot_slope_denom
        stance_count = torch.clamp(stance.float().sum(dim=1), min=1.0)
        swing_count = torch.clamp(swing.float().sum(dim=1), min=1.0)
        foot_slip = -torch.sum(torch.square(current_tangent_delta / self.contact_slip_scale) * stance.float(), dim=1) / stance_count
        stance_z_err = torch.sum(
            torch.square((foot_z - ref_foot_z) / self.contact_z_scale) * stance.float(),
            dim=1,
        ) / stance_count
        swing_z_err = torch.sum(
            torch.square((foot_z - ref_foot_z) / self.contact_z_scale) * swing.float(),
            dim=1,
        ) / swing_count
        has_stance = stance.any(dim=1)
        has_swing = swing.any(dim=1)
        ref_stance_foot_z = torch.where(has_stance, torch.exp(-stance_z_err), torch.zeros_like(ref_pose))
        ref_swing_foot_z = torch.where(has_swing, torch.exp(-swing_z_err), torch.zeros_like(ref_pose))
        ref_stance_fraction = torch.mean(stance.float(), dim=1)

        right_swing = swing[:, 0] & swing[:, 1]
        left_swing = swing[:, 2] & swing[:, 3]
        swing_foot_site_penalty = torch.zeros_like(ref_pose)
        for mask, cols in (
            (right_swing, self.right_foot_error_cols),
            (left_swing, self.left_foot_error_cols),
        ):
            swing_foot_site_penalty = swing_foot_site_penalty + (
                -torch.mean(torch.square(foot_error_flat[:, cols]), dim=1) * mask.float()
            )
        swing_hip_err = torch.zeros_like(ref_pose)
        swing_knee_err = torch.zeros_like(ref_pose)
        swing_knee_vel_err = torch.zeros_like(ref_pose)
        swing_hip_abs = torch.zeros_like(ref_pose)
        swing_knee_abs = torch.zeros_like(ref_pose)
        swing_knee_vel_abs = torch.zeros_like(ref_pose)
        swing_side_count = right_swing.float() + left_swing.float()
        hip_r = self.right_swing_joint_indices[0]
        knee_r = self.right_swing_joint_indices[1]
        hip_l = self.left_swing_joint_indices[0]
        knee_l = self.left_swing_joint_indices[1]
        for mask, hip_idx, knee_idx in ((right_swing, hip_r, knee_r), (left_swing, hip_l, knee_l)):
            mask_f = mask.float()
            hip_delta = q[:, hip_idx] - ref_q[:, hip_idx]
            knee_delta = q[:, knee_idx] - ref_q[:, knee_idx]
            knee_vel_delta = dq[:, knee_idx] - ref_dq[:, knee_idx]
            swing_hip_err = swing_hip_err + torch.square(hip_delta / self.reference["pose_scales"][hip_idx]) * mask_f
            swing_knee_err = swing_knee_err + torch.square(knee_delta / self.reference["pose_scales"][knee_idx]) * mask_f
            swing_knee_vel_err = swing_knee_vel_err + torch.square(knee_vel_delta / self.reference["vel_scales"][knee_idx]) * mask_f
            swing_hip_abs = swing_hip_abs + torch.abs(hip_delta) * mask_f
            swing_knee_abs = swing_knee_abs + torch.abs(knee_delta) * mask_f
            swing_knee_vel_abs = swing_knee_vel_abs + torch.abs(knee_vel_delta) * mask_f
        has_swing_side = swing_side_count > 0.0
        swing_side_count_clamped = torch.clamp(swing_side_count, min=1.0)
        ref_swing_hip_pose = torch.where(
            has_swing_side,
            torch.exp(-swing_hip_err / swing_side_count_clamped),
            torch.zeros_like(ref_pose),
        )
        ref_swing_knee_pose = torch.where(
            has_swing_side,
            torch.exp(-swing_knee_err / swing_side_count_clamped),
            torch.zeros_like(ref_pose),
        )
        ref_swing_knee_vel = torch.where(
            has_swing_side,
            torch.exp(-swing_knee_vel_err / swing_side_count_clamped),
            torch.zeros_like(ref_pose),
        )
        swing_hip_abs_err = torch.where(
            has_swing_side,
            -(swing_hip_abs / swing_side_count_clamped),
            torch.zeros_like(ref_pose),
        )
        swing_knee_abs_err = torch.where(
            has_swing_side,
            -(swing_knee_abs / swing_side_count_clamped),
            torch.zeros_like(ref_pose),
        )
        swing_knee_vel_abs_err = torch.where(
            has_swing_side,
            -(swing_knee_vel_abs / swing_side_count_clamped),
            torch.zeros_like(ref_pose),
        )
        right_foot_x = torch.mean(foot_rel_x[:, 0:2], dim=1)
        left_foot_x = torch.mean(foot_rel_x[:, 2:4], dim=1)
        swing_knee_flex_sum = torch.zeros_like(ref_pose)
        swing_hip_flex_sum = torch.zeros_like(ref_pose)
        swing_foot_forward_sum = torch.zeros_like(ref_pose)
        swing_knee_flex_margin_sum = torch.zeros_like(ref_pose)
        swing_hip_flex_margin_sum = torch.zeros_like(ref_pose)
        swing_foot_forward_margin_sum = torch.zeros_like(ref_pose)
        swing_knee_flex_angle_sum = torch.zeros_like(ref_pose)
        swing_hip_flex_angle_sum = torch.zeros_like(ref_pose)
        swing_foot_forward_delta_sum = torch.zeros_like(ref_pose)
        heuristic_specs = (
            (right_swing, hip_r, knee_r, right_foot_x - left_foot_x),
            (left_swing, hip_l, knee_l, left_foot_x - right_foot_x),
        )
        for mask, hip_idx, knee_idx, foot_forward_delta in heuristic_specs:
            mask_f = mask.float()
            knee_flex_angle = -q[:, knee_idx]
            hip_flex_angle = q[:, hip_idx]
            knee_flex_margin = (knee_flex_angle - self.swing_knee_flex_target) / max(self.swing_knee_flex_scale, 1e-6)
            hip_flex_margin = (hip_flex_angle - self.swing_hip_flex_target) / max(self.swing_hip_flex_scale, 1e-6)
            foot_forward_margin = (foot_forward_delta - self.swing_foot_forward_target) / max(self.swing_foot_forward_scale, 1e-6)
            swing_knee_flex_sum = swing_knee_flex_sum + torch.sigmoid(knee_flex_margin) * mask_f
            swing_hip_flex_sum = swing_hip_flex_sum + torch.sigmoid(hip_flex_margin) * mask_f
            swing_foot_forward_sum = swing_foot_forward_sum + torch.sigmoid(foot_forward_margin) * mask_f
            swing_knee_flex_margin_sum = swing_knee_flex_margin_sum + knee_flex_margin * mask_f
            swing_hip_flex_margin_sum = swing_hip_flex_margin_sum + hip_flex_margin * mask_f
            swing_foot_forward_margin_sum = swing_foot_forward_margin_sum + foot_forward_margin * mask_f
            swing_knee_flex_angle_sum = swing_knee_flex_angle_sum + knee_flex_angle * mask_f
            swing_hip_flex_angle_sum = swing_hip_flex_angle_sum + hip_flex_angle * mask_f
            swing_foot_forward_delta_sum = swing_foot_forward_delta_sum + foot_forward_delta * mask_f
        swing_knee_flex_heuristic = torch.where(
            has_swing_side,
            swing_knee_flex_sum / swing_side_count_clamped,
            torch.zeros_like(ref_pose),
        )
        swing_hip_flex_heuristic = torch.where(
            has_swing_side,
            swing_hip_flex_sum / swing_side_count_clamped,
            torch.zeros_like(ref_pose),
        )
        swing_foot_forward_heuristic = torch.where(
            has_swing_side,
            swing_foot_forward_sum / swing_side_count_clamped,
            torch.zeros_like(ref_pose),
        )
        swing_knee_flex_margin = torch.where(
            has_swing_side,
            swing_knee_flex_margin_sum / swing_side_count_clamped,
            torch.zeros_like(ref_pose),
        )
        swing_hip_flex_margin = torch.where(
            has_swing_side,
            swing_hip_flex_margin_sum / swing_side_count_clamped,
            torch.zeros_like(ref_pose),
        )
        swing_foot_forward_margin = torch.where(
            has_swing_side,
            swing_foot_forward_margin_sum / swing_side_count_clamped,
            torch.zeros_like(ref_pose),
        )
        swing_knee_flex_angle = torch.where(
            has_swing_side,
            swing_knee_flex_angle_sum / swing_side_count_clamped,
            torch.zeros_like(ref_pose),
        )
        swing_hip_flex_angle = torch.where(
            has_swing_side,
            swing_hip_flex_angle_sum / swing_side_count_clamped,
            torch.zeros_like(ref_pose),
        )
        swing_foot_forward_delta = torch.where(
            has_swing_side,
            swing_foot_forward_delta_sum / swing_side_count_clamped,
            torch.zeros_like(ref_pose),
        )
        right_swing_count = torch.clamp(right_swing.float().sum(), min=1.0)
        left_swing_count = torch.clamp(left_swing.float().sum(), min=1.0)
        right_swing_present = right_swing.any()
        left_swing_present = left_swing.any()
        right_hip_mean = torch.sum(q[:, hip_r] * right_swing.float()) / right_swing_count
        left_hip_mean = torch.sum(q[:, hip_l] * left_swing.float()) / left_swing_count
        right_knee_mean = torch.sum((-q[:, knee_r]) * right_swing.float()) / right_swing_count
        left_knee_mean = torch.sum((-q[:, knee_l]) * left_swing.float()) / left_swing_count
        right_forward_mean = torch.sum((right_foot_x - left_foot_x) * right_swing.float()) / right_swing_count
        left_forward_mean = torch.sum((left_foot_x - right_foot_x) * left_swing.float()) / left_swing_count
        side_balance_loss = (
            torch.square((right_hip_mean - left_hip_mean) / 0.25)
            + torch.square((right_knee_mean - left_knee_mean) / 0.25)
            + torch.square((right_forward_mean - left_forward_mean) / 0.35)
        ) / 3.0
        side_balance_penalty = torch.where(
            right_swing_present & left_swing_present,
            -side_balance_loss.expand_as(ref_pose),
            torch.zeros_like(ref_pose),
        )

        qpos_penalty = -torch.sum(torch.square(q - ref_q) * self.pose_weights, dim=1) / torch.clamp(
            torch.sum(self.pose_weights), min=1e-6
        )
        qvel_penalty = -torch.sum(torch.square(dq - ref_dq) * self.vel_weights, dim=1) / torch.clamp(
            torch.sum(self.vel_weights), min=1e-6
        )
        foot_site_penalty = -torch.sum(torch.square(foot_error_flat) * foot_weights_flat, dim=1) / foot_weight_sum_flat
        future_foot_site_penalty = torch.zeros_like(ref_pose)
        if self.reference_reward_future_steps > 0:
            future_sum = torch.zeros_like(ref_pose)
            target = self.target_phase_idx()
            future_weight_sum = torch.clamp(torch.sum(foot_weights_flat, dim=1), min=1e-6)
            for offset in range(1, self.reference_reward_future_steps + 1):
                future_phase = reference_index(target + offset, self.reference, self.config)
                future_foot = self.reference_foot(future_phase)
                future_error_flat = torch.stack(
                    [
                        foot_rel_x[:, 0] - future_foot[:, 0, 0],
                        foot_z[:, 0] - future_foot[:, 0, 2],
                        foot_rel_x[:, 1] - future_foot[:, 1, 0],
                        foot_z[:, 1] - future_foot[:, 1, 2],
                        foot_rel_x[:, 2] - future_foot[:, 2, 0],
                        foot_z[:, 2] - future_foot[:, 2, 2],
                        foot_rel_x[:, 3] - future_foot[:, 3, 0],
                        foot_z[:, 3] - future_foot[:, 3, 2],
                    ],
                    dim=1,
                )
                future_sum = future_sum + torch.sum(torch.square(future_error_flat) * foot_weights_flat, dim=1) / future_weight_sum
            future_foot_site_penalty = -(future_sum / float(self.reference_reward_future_steps))
        pelvis_ty_error = q[:, TRACK_JOINTS.index("pelvis_ty")] - ref_q[:, TRACK_JOINTS.index("pelvis_ty")]
        pelvis_tilt_error = q[:, TRACK_JOINTS.index("pelvis_tilt")] - ref_q[:, TRACK_JOINTS.index("pelvis_tilt")]
        pelvis_vx_error = self.qvel[:, self.pelvis_tx_qvel] - ref_pelvis_tx_vel
        pelvis_penalty = -(
            4.0 * torch.square(pelvis_ty_error)
            + 2.0 * torch.square(pelvis_tilt_error)
            + 0.5 * torch.square(pelvis_vx_error)
        ) / 6.5
        swing_hip_penalty = torch.where(
            has_swing_side,
            -(swing_hip_abs / swing_side_count_clamped) ** 2,
            torch.zeros_like(ref_pose),
        )
        swing_limb_sq = torch.zeros_like(ref_pose)
        for mask, cols in (
            (right_swing, self.right_limb_track_indices),
            (left_swing, self.left_limb_track_indices),
        ):
            swing_limb_sq = swing_limb_sq + torch.mean(torch.square(q[:, cols] - ref_q[:, cols]), dim=1) * mask.float()
        swing_limb_penalty = torch.where(has_swing_side, -(swing_limb_sq / swing_side_count_clamped), torch.zeros_like(ref_pose))
        terminal_swing_landing_penalty = torch.zeros_like(ref_pose)
        landing_phase = self.episode_step.float() / max(float(self.episode_steps - 1), 1.0)
        landing_mask = landing_phase >= 0.7
        if bool(landing_mask.any().item()):
            for mask, cols, zcols in (
                (right_swing | (stance[:, 0] & stance[:, 1]), self.right_foot_error_cols, self.right_foot_z_error_cols),
                (left_swing | (stance[:, 2] & stance[:, 3]), self.left_foot_error_cols, self.left_foot_z_error_cols),
            ):
                side_mask = landing_mask & mask
                height_excess = torch.relu(foot_error_flat[:, zcols] - 0.02)
                landing_loss = torch.mean(torch.square(foot_error_flat[:, cols]), dim=1) + 4.0 * torch.mean(
                    torch.square(height_excess), dim=1
                )
                terminal_swing_landing_penalty = terminal_swing_landing_penalty - landing_loss * side_mask.float()

        activation_l2 = -torch.mean(torch.square(activation), dim=1)
        activation_smooth = -torch.mean(torch.square(activation - self.prev_activation), dim=1)
        activation_symmetry_penalty = -torch.mean(
            torch.square(activation[:, self.right_actuator_indices] - activation[:, self.left_actuator_indices]),
            dim=1,
        )
        activation_mean = torch.mean(activation, dim=1)
        activation_max = torch.amax(activation, dim=1)
        normalized_action = torch.clamp(action, -1.0, 1.0)
        normalized_action_mean = torch.mean(normalized_action, dim=1)
        normalized_action_std = torch.std(normalized_action, dim=1, unbiased=False)
        normalized_action_max = torch.amax(normalized_action, dim=1)
        action_clip_fraction = torch.mean((torch.abs(action) > 1.0).float(), dim=1)
        upright = torch.exp(-4.0 * torch.square(self.qpos[:, self.pelvis_tilt_qpos]))
        terrain_height = current_terrain_height_tensor(self.qpos, self.phase_idx, self.reference, self.config)
        pelvis_height_above_terrain = self.qpos[:, self.pelvis_ty_qpos] - terrain_height
        height = torch.exp(-20.0 * torch.square(torch.relu(0.82 - pelvis_height_above_terrain)))
        alive = torch.ones_like(ref_pose)
        fall_tilt_error = torch.where(
            reference_valid_bool,
            self.qpos[:, self.pelvis_tilt_qpos] - ref_q[:, TRACK_JOINTS.index("pelvis_tilt")],
            self.qpos[:, self.pelvis_tilt_qpos],
        )
        fall_condition = (pelvis_height_above_terrain < self.safe_pelvis_height) | (
            torch.abs(fall_tilt_error)
            > self.max_abs_pelvis_tilt_error
        )
        fall = torch.where(fall_condition, -torch.ones_like(ref_pose), torch.zeros_like(ref_pose))
        terms = {
            "ref_pose": ref_pose,
            "ref_vel": ref_vel,
            "pelvis_pose_ref": pelvis_pose_ref,
            "hip_pose_ref": hip_pose_ref,
            "knee_pose_ref": knee_pose_ref,
            "knee_vel_ref": knee_vel_ref,
            "ankle_pose_ref": ankle_pose_ref,
            "mtp_pose_ref": mtp_pose_ref,
            "foot_z_ref": foot_z_ref,
            "foot_x_ref": foot_x_ref,
            "ref_stance_foot_z": ref_stance_foot_z,
            "ref_swing_foot_z": ref_swing_foot_z,
            "ref_swing_hip_pose": ref_swing_hip_pose,
            "ref_swing_knee_pose": ref_swing_knee_pose,
            "ref_swing_knee_vel": ref_swing_knee_vel,
            "swing_knee_flex_heuristic": swing_knee_flex_heuristic,
            "swing_hip_flex_heuristic": swing_hip_flex_heuristic,
            "swing_foot_forward_heuristic": swing_foot_forward_heuristic,
            "swing_knee_flex_angle": swing_knee_flex_angle,
            "swing_hip_flex_angle": swing_hip_flex_angle,
            "swing_foot_forward_delta": swing_foot_forward_delta,
            "batch_swing_side_balance_penalty": side_balance_penalty,
            "ref_stance_fraction": ref_stance_fraction,
            "reference_phase_lead_steps": torch.full_like(ref_pose, float(self.reference_phase_lead_steps)),
            "reference_phase_tolerance_steps": torch.full_like(ref_pose, float(self.reference_phase_tolerance_steps)),
            "reference_swing_exaggeration_scale": torch.full_like(ref_pose, float(self.reference_swing_exaggeration_scale)),
            "pelvis_tx_vel_ref": pelvis_tx_vel_ref,
            "pelvis_ty_vel_ref": pelvis_ty_vel_ref,
            "pelvis_tangent_vel_ref": pelvis_tangent_vel_ref,
            "pelvis_normal_vel_ref": pelvis_normal_vel_ref,
            "foot_slip": foot_slip,
            "foot_tangent_delta_abs": -torch.mean(torch.abs(current_tangent_delta), dim=1),
            "foot_normal_delta_abs": -torch.mean(torch.abs(current_normal_delta), dim=1),
            "activation_l2": activation_l2,
            "activation_smooth": activation_smooth,
            "activation_mean": activation_mean,
            "activation_max": activation_max,
            "activation_bias_abs": self.last_activation_bias_abs,
            "activation_bias_signed": self.last_activation_bias_signed,
            "normalized_action_mean": normalized_action_mean,
            "normalized_action_std": normalized_action_std,
            "normalized_action_max": normalized_action_max,
            "action_clip_fraction": action_clip_fraction,
            "upright": upright,
            "height": height,
            "pelvis_height_above_terrain": pelvis_height_above_terrain,
            "alive": alive,
            "fall": fall,
            "tracking_qpos_penalty": qpos_penalty,
            "tracking_qvel_penalty": qvel_penalty,
            "tracking_foot_site_penalty": foot_site_penalty,
            "tracking_swing_foot_site_penalty": swing_foot_site_penalty,
            "tracking_swing_hip_penalty": swing_hip_penalty,
            "tracking_swing_limb_penalty": swing_limb_penalty,
            "tracking_activation_symmetry_penalty": activation_symmetry_penalty,
            "tracking_future_foot_site_penalty": future_foot_site_penalty,
            "terminal_swing_landing_penalty": terminal_swing_landing_penalty,
            "tracking_pelvis_penalty": pelvis_penalty,
            "tracking_energy_penalty": activation_l2,
            "pelvis_ty_abs_err": -torch.abs(q[:, TRACK_JOINTS.index("pelvis_ty")] - ref_q[:, TRACK_JOINTS.index("pelvis_ty")]),
            "pelvis_tilt_abs_err": -torch.abs(
                q[:, TRACK_JOINTS.index("pelvis_tilt")] - ref_q[:, TRACK_JOINTS.index("pelvis_tilt")]
            ),
            "pelvis_tx_vel_abs_err": -torch.abs(self.qvel[:, self.pelvis_tx_qvel] - ref_pelvis_tx_vel),
            "pelvis_ty_vel_abs_err": -torch.abs(self.qvel[:, self.pelvis_ty_qvel] - ref_pelvis_ty_vel),
            "pelvis_tangent_vel_abs_err": -torch.abs(pelvis_tangent_vel - ref_pelvis_tangent_vel),
            "pelvis_normal_vel_abs_err": -torch.abs(pelvis_normal_vel - ref_pelvis_normal_vel),
            "hip_abs_err": -torch.mean(torch.abs(q[:, self.hip_track_indices] - ref_q[:, self.hip_track_indices]), dim=1),
            "knee_abs_err": -torch.mean(torch.abs(q[:, self.knee_track_indices] - ref_q[:, self.knee_track_indices]), dim=1),
            "swing_hip_abs_err": swing_hip_abs_err,
            "swing_knee_abs_err": swing_knee_abs_err,
            "swing_knee_vel_abs_err": swing_knee_vel_abs_err,
            "swing_knee_flex_margin": swing_knee_flex_margin,
            "swing_hip_flex_margin": swing_hip_flex_margin,
            "swing_foot_forward_margin": swing_foot_forward_margin,
            "ankle_abs_err": -torch.mean(torch.abs(q[:, self.ankle_track_indices] - ref_q[:, self.ankle_track_indices]), dim=1),
            "reference_valid": reference_valid,
        }
        post_reference_masked_terms = {
            "ref_pose",
            "ref_vel",
            "pelvis_pose_ref",
            "hip_pose_ref",
            "knee_pose_ref",
            "knee_vel_ref",
            "ankle_pose_ref",
            "mtp_pose_ref",
            "foot_z_ref",
            "foot_x_ref",
            "ref_stance_foot_z",
            "ref_swing_foot_z",
            "ref_swing_hip_pose",
            "ref_swing_knee_pose",
            "ref_swing_knee_vel",
            "swing_knee_flex_heuristic",
            "swing_hip_flex_heuristic",
            "swing_foot_forward_heuristic",
            "batch_swing_side_balance_penalty",
            "tracking_qpos_penalty",
            "tracking_qvel_penalty",
            "tracking_foot_site_penalty",
            "tracking_swing_foot_site_penalty",
            "tracking_swing_hip_penalty",
            "tracking_swing_limb_penalty",
            "tracking_future_foot_site_penalty",
            "terminal_swing_landing_penalty",
            "tracking_pelvis_penalty",
            "pelvis_ty_abs_err",
            "pelvis_tilt_abs_err",
            "pelvis_ty_vel_abs_err",
            "pelvis_tangent_vel_abs_err",
            "pelvis_normal_vel_abs_err",
            "hip_abs_err",
            "knee_abs_err",
            "swing_hip_abs_err",
            "swing_knee_abs_err",
            "swing_knee_vel_abs_err",
            "swing_knee_flex_margin",
            "swing_hip_flex_margin",
            "swing_foot_forward_margin",
            "ankle_abs_err",
        }
        if self.post_reference_enabled:
            for key in post_reference_masked_terms:
                if key in terms:
                    terms[key] = terms[key] * reference_valid
        reward = torch.zeros_like(ref_pose)
        for key, value in terms.items():
            reward = reward + self.reward_weights.get(key, 0.0) * value
        return reward, terms

    def group_pose_reward(self, pose_sq: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        weights = self.pose_weights[indices]
        err = torch.sum(pose_sq[:, indices] * weights, dim=1) / torch.clamp(torch.sum(weights), min=1e-6)
        return torch.exp(-err)

    def group_vel_reward(self, vel_sq: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        weights = self.vel_weights[indices]
        err = torch.sum(vel_sq[:, indices] * weights, dim=1) / torch.clamp(torch.sum(weights), min=1e-6)
        return torch.exp(-err)


@torch.no_grad()
def evaluate(
    *,
    agent: Agent,
    obs_normalizer: ObsNormalizer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    config: dict[str, Any],
    reference: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    update: int,
    global_step: int,
) -> dict[str, Any]:
    runner = MJWarpMuscleRunner(
        model=model,
        data=data,
        config=config,
        reference=reference,
        nworld=int(args.eval_worlds),
        nconmax=int(args.nconmax),
        njmax=int(args.njmax),
        seed=int(args.seed) + 10000 + update,
        device=device,
    )
    obs = runner.obs()
    reward_sum = torch.zeros(args.eval_worlds, dtype=torch.float32, device=device)
    done_count = torch.zeros((), dtype=torch.float32, device=device)
    fall_count = torch.zeros((), dtype=torch.float32, device=device)
    qvel_count = torch.zeros((), dtype=torch.float32, device=device)
    for _ in range(int(args.eval_steps)):
        action, _, _, _ = agent.get_action_and_value(obs_normalizer.normalize(obs), deterministic=True)
        obs, reward, _, terms = runner.step(action)
        reward_sum += reward
        done_count += terms["done_count"].sum()
        fall_count += terms["fall_done"].sum()
        qvel_count += terms["qvel_done"].sum()
    return {
        "global_step": global_step,
        "update": update,
        "eval_mean_reward_sum": float(reward_sum.mean().item()),
        "eval_done_rate_per_step": float((done_count / (args.eval_worlds * args.eval_steps)).item()),
        "eval_fall_rate_per_step": float((fall_count / (args.eval_worlds * args.eval_steps)).item()),
        "eval_qvel_done_rate_per_step": float((qvel_count / (args.eval_worlds * args.eval_steps)).item()),
        "eval_mean_pelvis_tx": float(runner.qpos[:, runner.pelvis_tx_qpos].mean().item()),
        "eval_mean_pelvis_height": float(runner.qpos[:, runner.pelvis_ty_qpos].mean().item()),
    }


def set_cpu_reference_state(model: mujoco.MjModel, data: mujoco.MjData, reference: dict[str, Any], phase: int) -> None:
    data.qpos[reference["reset_qpos_indices"].detach().cpu().numpy()] = reference["reset_q_ref"][phase].detach().cpu().numpy()
    data.qvel[reference["reset_qvel_indices"].detach().cpu().numpy()] = reference["reset_dq_ref"][phase].detach().cpu().numpy()
    if "course_offset" in reference:
        course_offset = float(reference["course_offset"][phase].detach().cpu().item())
        pelvis_tx_ref = float(reference["pelvis_tx_ref"][phase].detach().cpu().item())
        data.qpos[int(model.jnt_qposadr[joint_id(model, "pelvis_tx")])] = course_offset + pelvis_tx_ref
    else:
        data.qpos[int(model.jnt_qposadr[joint_id(model, "pelvis_tx")])] = 0.0
    data.ctrl[:] = 0.05
    data.act[:] = 0.05
    apply_joint_equalities_np(model, data)
    mujoco.mj_forward(model, data)


def cpu_policy_obs(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    reference: dict[str, Any],
    config: dict[str, Any],
    phase: int,
    episode_step: int,
    device: torch.device,
) -> torch.Tensor:
    return build_policy_obs_tensor(
        qpos=torch.tensor(data.qpos[None, :], dtype=torch.float32, device=device),
        qvel=torch.tensor(data.qvel[None, :], dtype=torch.float32, device=device),
        act=torch.tensor(data.act[None, :], dtype=torch.float32, device=device),
        site_xpos=torch.tensor(data.site_xpos[None, :, :], dtype=torch.float32, device=device),
        phase_idx=torch.tensor([int(phase)], dtype=torch.long, device=device),
        episode_step=torch.tensor([int(episode_step)], dtype=torch.long, device=device),
        pelvis_tx_qpos=int(model.jnt_qposadr[joint_id(model, "pelvis_tx")]),
        foot_site_indices=reference["foot_site_indices"],
        reference=reference,
        config=config,
    )


def reference_phase_label(reference: dict[str, Any], phase: int) -> str:
    names = reference.get("reference_names")
    ids = reference.get("reference_id")
    if names is None or ids is None:
        return f"phase{int(phase)}"
    ref_id = int(ids[int(phase) % int(reference["length"])].detach().cpu().item())
    name = str(names[ref_id]) if 0 <= ref_id < len(names) else f"ref{ref_id}"
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)
    return f"{safe}_phase{int(phase)}"


@torch.no_grad()
def render_policy_video(
    *,
    agent: Agent,
    obs_normalizer: ObsNormalizer,
    config: dict[str, Any],
    reference: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    update: int,
    global_step: int,
) -> dict[str, Any]:
    model, data = build_muscle_model(config)
    phase = int(args.video_phase) % int(reference["length"])
    set_cpu_reference_state(model, data, reference, phase)
    frame_skip = int(config["control"]["frame_skip"])
    pelvis_tx_qpos = int(model.jnt_qposadr[joint_id(model, "pelvis_tx")])
    pelvis_ty_qpos = int(model.jnt_qposadr[joint_id(model, "pelvis_ty")])
    pelvis_tilt_qpos = int(model.jnt_qposadr[joint_id(model, "pelvis_tilt")])
    qpos_names = coordinate_names(model, kind="qpos")
    qvel_names = coordinate_names(model, kind="qvel")
    act_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or str(i) for i in range(model.nu)]

    renderer = mujoco.Renderer(model, height=int(args.video_height), width=int(args.video_width))
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = float(args.video_camera_distance)
    camera.azimuth = 90.0
    camera.elevation = -8.0

    frames = []
    rows: list[dict[str, Any]] = []
    prev_qpos: np.ndarray | None = None
    prev_ctrl = data.ctrl.copy()
    fell = False
    try:
        for frame in range(int(args.video_steps)):
            ref_phase = int(
                reference_index(
                    torch.tensor([int(phase)], dtype=torch.long, device=device),
                    reference,
                    config,
                )[0]
                .detach()
                .cpu()
                .item()
            )
            reference_valid = (not post_reference_enabled(config)) or frame < int(post_reference_valid_steps(reference, config))
            qpos_before = data.qpos.copy()
            qvel_before = data.qvel.copy()
            render_delta = np.zeros_like(qpos_before) if prev_qpos is None else qpos_before - prev_qpos
            render_delta_name, render_delta_value = top_abs_named(render_delta, qpos_names)
            qvel_name, qvel_value = top_abs_named(qvel_before, qvel_names)
            camera.lookat[:] = [float(data.qpos[pelvis_tx_qpos]), 0.0, float(args.video_camera_height)]
            renderer.update_scene(data, camera=camera)
            frames.append(renderer.render())
            obs = cpu_policy_obs(model, data, reference, config, phase, frame, device)
            action, _, _, _ = agent.get_action_and_value(obs_normalizer.normalize(obs), deterministic=True)
            clipped_action = torch.clamp(action[0], -1.0, 1.0)
            policy_ctrl = muscle_action_to_activation(action[0]).detach().cpu().numpy().astype(np.float64)
            action_np = clipped_action.detach().cpu().numpy().astype(np.float64)
            target_activation = reference["activation_prior_ref"][ref_phase].detach().cpu().numpy().astype(np.float64)
            target_mask = reference["activation_prior_mask"].detach().cpu().numpy().astype(bool)
            video_mix = getattr(args, "video_activation_prior_execution_mix", None)
            if video_mix is None:
                execution_mix = max(0.0, min(1.0, float(config.get("activation_prior", {}).get("execution_mix", 0.0))))
            else:
                execution_mix = max(0.0, min(1.0, float(video_mix)))
            ctrl = np.clip((1.0 - execution_mix) * policy_ctrl + execution_mix * target_activation, 0.0, 1.0)
            activation_prior_mse = (
                float(np.mean(np.square(ctrl[target_mask] - target_activation[target_mask])))
                if bool(np.any(target_mask))
                else 0.0
            )
            data.ctrl[:] = ctrl
            for _ in range(frame_skip):
                mujoco.mj_step(model, data)
            logged_phase = int(ref_phase)
            if post_reference_enabled(config):
                phase = phase + 1
            else:
                phase = (phase + 1) % int(reference["length"])
            qpos_after = data.qpos.copy()
            qvel_after = data.qvel.copy()
            transition_delta = qpos_after - qpos_before
            transition_name, transition_value = top_abs_named(transition_delta, qpos_names)
            qvel_after_name, qvel_after_value = top_abs_named(qvel_after, qvel_names)
            ctrl_delta = ctrl - prev_ctrl
            ctrl_name, ctrl_value = top_abs_named(ctrl, act_names)
            ctrl_delta_name, ctrl_delta_value = top_abs_named(ctrl_delta, act_names)
            terrain_height = current_terrain_height_np(model, data, reference, config, phase)
            pelvis_height_above_terrain = float(data.qpos[pelvis_ty_qpos]) - terrain_height
            low_height = bool(pelvis_height_above_terrain < float(config["reset"]["safe_pelvis_height"]))
            ref_tilt = float(reference["q_ref"][ref_phase, TRACK_JOINTS.index("pelvis_tilt")].detach().cpu().item())
            pelvis_tilt_error = float(data.qpos[pelvis_tilt_qpos] - ref_tilt) if reference_valid else float(data.qpos[pelvis_tilt_qpos])
            bad_tilt = bool(
                abs(pelvis_tilt_error)
                > float(config["reset"].get("max_abs_pelvis_tilt_error", config["reset"].get("max_abs_pelvis_tilt", 0.65)))
            )
            fell = low_height or bad_tilt
            rows.append(
                {
                    "video_frame": frame,
                    "phase": logged_phase,
                    "reference_valid": bool(reference_valid),
                    "time": float(data.time),
                    "ncon": int(data.ncon),
                    "pelvis_tx": float(data.qpos[pelvis_tx_qpos]),
                    "pelvis_ty": float(data.qpos[pelvis_ty_qpos]),
                    "terrain_height": terrain_height,
                    "pelvis_height_above_terrain": pelvis_height_above_terrain,
                    "pelvis_tilt": float(data.qpos[pelvis_tilt_qpos]),
                    "ref_pelvis_tilt": ref_tilt,
                    "pelvis_tilt_error": pelvis_tilt_error,
                    "fell": fell,
                    "low_height": low_height,
                    "bad_tilt": bad_tilt,
                    "max_abs_render_qpos_delta": float(np.max(np.abs(render_delta))),
                    "top_render_qpos_delta_name": render_delta_name,
                    "top_render_qpos_delta": render_delta_value,
                    "max_abs_transition_qpos_delta": float(np.max(np.abs(transition_delta))),
                    "top_transition_qpos_delta_name": transition_name,
                    "top_transition_qpos_delta": transition_value,
                    "max_abs_qvel_before": float(np.max(np.abs(qvel_before))),
                    "top_qvel_before_name": qvel_name,
                    "top_qvel_before": qvel_value,
                    "max_abs_qvel_after": float(np.max(np.abs(qvel_after))),
                    "top_qvel_after_name": qvel_after_name,
                    "top_qvel_after": qvel_after_value,
                    "mean_ctrl": float(np.mean(ctrl)),
                    "max_ctrl": float(np.max(ctrl)),
                    "mean_policy_ctrl": float(np.mean(policy_ctrl)),
                    "max_policy_ctrl": float(np.max(policy_ctrl)),
                    "activation_prior_execution_mix": execution_mix,
                    "mean_activation_prior_target": float(np.mean(target_activation[target_mask])) if bool(np.any(target_mask)) else 0.0,
                    "activation_prior_mse": activation_prior_mse,
                    "top_ctrl_name": ctrl_name,
                    "top_ctrl": ctrl_value,
                    "max_abs_ctrl_delta": float(np.max(np.abs(ctrl_delta))),
                    "top_ctrl_delta_name": ctrl_delta_name,
                    "top_ctrl_delta": ctrl_delta_value,
                    "mean_normalized_action": float(np.mean(action_np)),
                    "std_normalized_action": float(np.std(action_np)),
                    "max_normalized_action": float(np.max(action_np)),
                    "action_clip_fraction": float(torch.mean((torch.abs(action[0]) > 1.0).float()).item()),
                }
            )
            prev_qpos = qpos_before
            prev_ctrl = ctrl.copy()
            if fell and not bool(getattr(args, "ignore_fall", False)):
                break
    finally:
        renderer.close()

    video_dir = args.outdir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    label = reference_phase_label(reference, int(args.video_phase) % int(reference["length"]))
    video_path = video_dir / f"policy_update_{update:05d}_step_{global_step:09d}_{label}.mp4"
    diagnostic_path = video_dir / f"policy_update_{update:05d}_step_{global_step:09d}_{label}_diagnostics.csv"
    imageio.mimsave(video_path, frames, fps=30)
    write_csv_rows(diagnostic_path, rows)
    return {
        "global_step": global_step,
        "update": update,
        "video": str(video_path),
        "video_diagnostics": str(diagnostic_path),
        "video_frames": len(frames),
        "video_fell": fell,
        "video_final_pelvis_tx": float(data.qpos[pelvis_tx_qpos]),
        "video_final_pelvis_height": float(data.qpos[pelvis_ty_qpos]),
        "video_max_qvel": max((float(row["max_abs_qvel_after"]) for row in rows), default=0.0),
        "video_max_ctrl_delta": max((float(row["max_abs_ctrl_delta"]) for row in rows), default=0.0),
    }


def save_checkpoint(path: Path, agent: Agent, optimizer: optim.Optimizer, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"agent_state_dict": agent.state_dict(), "optimizer_state_dict": optimizer.state_dict(), **payload}, path)


def pretrain_actor_on_activation_prior(
    *,
    agent: Agent,
    optimizer: optim.Optimizer,
    obs_normalizer: ObsNormalizer,
    runner: MJWarpMuscleRunner,
    reference: dict[str, Any],
    batches: int,
    learning_rate: float,
    max_grad_norm: float,
) -> dict[str, Any]:
    mask = reference["activation_prior_mask"]
    if batches <= 0 or not bool(mask.any().item()):
        return {
            "activation_prior_pretrain_batches": 0,
            "activation_prior_pretrain_loss": 0.0,
            "activation_prior_pretrain_activation_mse": 0.0,
        }
    all_rows = torch.ones(runner.nworld, dtype=torch.bool, device=runner.device)
    losses = []
    activation_losses = []
    mean_actions = []
    std_actions = []
    pretrain_optimizer = optim.Adam(agent.actor_mean.parameters(), lr=float(learning_rate), eps=1e-5)
    agent.train()
    for _ in range(int(batches)):
        with torch.no_grad():
            runner.reset(all_rows)
            obs_raw = runner.obs()
            obs_normalizer.update(obs_raw)
            obs = obs_normalizer.normalize(obs_raw)
            prior_phase = reference_index(runner.phase_idx, reference, config)
            target_action = reference["activation_prior_action_ref"][prior_phase]
            target_activation = reference["activation_prior_ref"][prior_phase]
        actor_mean = agent.actor_mean(obs)
        delta = actor_mean[:, mask] - target_action[:, mask]
        loss = torch.mean(torch.square(delta))
        pretrain_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.actor_mean.parameters(), float(max_grad_norm))
        pretrain_optimizer.step()
        with torch.no_grad():
            activation = muscle_action_to_activation(actor_mean)
            activation_loss = torch.mean(torch.square(activation[:, mask] - target_activation[:, mask]))
            losses.append(loss.detach())
            activation_losses.append(activation_loss.detach())
            mean_actions.append(torch.mean(torch.clamp(actor_mean, -1.0, 1.0)).detach())
            std_actions.append(torch.std(torch.clamp(actor_mean, -1.0, 1.0), unbiased=False).detach())

    return {
        "activation_prior_pretrain_batches": int(batches),
        "activation_prior_pretrain_loss": float(torch.stack(losses).mean().item()) if losses else 0.0,
        "activation_prior_pretrain_final_loss": float(losses[-1].item()) if losses else 0.0,
        "activation_prior_pretrain_activation_mse": float(torch.stack(activation_losses).mean().item()) if activation_losses else 0.0,
        "activation_prior_pretrain_final_activation_mse": float(activation_losses[-1].item()) if activation_losses else 0.0,
        "activation_prior_pretrain_mean_action": float(torch.stack(mean_actions).mean().item()) if mean_actions else 0.0,
        "activation_prior_pretrain_std_action": float(torch.stack(std_actions).mean().item()) if std_actions else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "muscle_2d_mjwarp_curriculum.json")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE_PATH)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--nworld", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--nconmax", type=int, default=128)
    parser.add_argument("--njmax", type=int, default=512)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--actor-logstd-init", type=float, default=None)
    parser.add_argument("--initial-actor-action-mean", type=float, default=None)
    parser.add_argument("--initial-actor-ctrl-mean", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--no-obs-norm", action="store_true")
    parser.add_argument("--episode-steps", type=int, default=None)
    parser.add_argument("--max-abs-pelvis-tilt-error", type=float, default=None)
    parser.add_argument("--phase-start", type=int, default=None)
    parser.add_argument("--phase-end", type=int, default=None)
    parser.add_argument("--qpos-noise", type=float, default=None)
    parser.add_argument("--qvel-noise", type=float, default=None)
    parser.add_argument("--activation-prior-phase-offset", type=int, default=None)
    parser.add_argument("--activation-prior-left-phase-offset", type=int, default=None)
    parser.add_argument("--activation-prior-right-phase-offset", type=int, default=None)
    parser.add_argument("--activation-prior-execution-mix", type=float, default=None)
    parser.add_argument("--rollout-bc-only", action="store_true")
    parser.add_argument("--rollout-bc-execution-mix", type=float, default=1.0)
    parser.add_argument("--dagger-bc-replay-capacity", type=int, default=None)
    parser.add_argument("--dagger-bc-replay-ratio", type=float, default=None)
    parser.add_argument("--push-interval-steps", type=int, default=None)
    parser.add_argument("--push-probability", type=float, default=None)
    parser.add_argument("--push-pelvis-qvel-std", type=float, default=None)
    parser.add_argument("--push-joint-qvel-std", type=float, default=None)
    parser.add_argument("--pretrain-only-video", action="store_true")
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-worlds", type=int, default=32)
    parser.add_argument("--eval-steps", type=int, default=96)
    parser.add_argument("--video-every", type=int, default=5)
    parser.add_argument("--video-steps", type=int, default=96)
    parser.add_argument("--video-phase", type=int, default=0)
    parser.add_argument("--video-height", type=int, default=720)
    parser.add_argument("--video-width", type=int, default=1280)
    parser.add_argument("--video-camera-distance", type=float, default=7.0)
    parser.add_argument("--video-camera-height", type=float, default=0.9)
    parser.add_argument("--video-activation-prior-execution-mix", type=float, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    args = parser.parse_args()

    if args.device != "cuda":
        raise SystemExit("MJWarp PPO is intended for --device cuda")
    device = torch.device(args.device)
    config = load_config(args.config)
    if args.episode_steps is not None:
        config["reset"]["episode_steps"] = int(args.episode_steps)
    if args.max_abs_pelvis_tilt_error is not None:
        config["reset"]["max_abs_pelvis_tilt_error"] = float(args.max_abs_pelvis_tilt_error)
    if args.phase_start is not None:
        config["reset"]["phase_start"] = int(args.phase_start)
    if args.phase_end is not None:
        config["reset"]["phase_end"] = int(args.phase_end)
    if args.qpos_noise is not None:
        config["reset"]["qpos_noise"] = float(args.qpos_noise)
    if args.qvel_noise is not None:
        config["reset"]["qvel_noise"] = float(args.qvel_noise)
    if args.activation_prior_phase_offset is not None:
        config.setdefault("activation_prior", {})["phase_offset"] = int(args.activation_prior_phase_offset)
    if args.activation_prior_left_phase_offset is not None:
        config.setdefault("activation_prior", {})["left_phase_offset"] = int(args.activation_prior_left_phase_offset)
    if args.activation_prior_right_phase_offset is not None:
        config.setdefault("activation_prior", {})["right_phase_offset"] = int(args.activation_prior_right_phase_offset)
    if args.activation_prior_execution_mix is not None:
        config.setdefault("activation_prior", {})["execution_mix"] = float(args.activation_prior_execution_mix)
        config.setdefault("activation_prior", {}).pop("execution_mix_schedule", None)
    if args.rollout_bc_only and args.video_activation_prior_execution_mix is None:
        args.video_activation_prior_execution_mix = 0.0
    if args.dagger_bc_replay_capacity is not None:
        config.setdefault("ppo", {})["dagger_bc_replay_capacity"] = int(args.dagger_bc_replay_capacity)
    if args.dagger_bc_replay_ratio is not None:
        config.setdefault("ppo", {})["dagger_bc_replay_ratio"] = float(args.dagger_bc_replay_ratio)
    if args.push_interval_steps is not None:
        config.setdefault("perturbation", {})["push_interval_steps"] = int(args.push_interval_steps)
    if args.push_probability is not None:
        config.setdefault("perturbation", {})["push_probability"] = float(args.push_probability)
    if args.push_pelvis_qvel_std is not None:
        config.setdefault("perturbation", {})["push_pelvis_qvel_std"] = float(args.push_pelvis_qvel_std)
    if args.push_joint_qvel_std is not None:
        config.setdefault("perturbation", {})["push_joint_qvel_std"] = float(args.push_joint_qvel_std)
    ppo_cfg = config["ppo"]
    seed = int(args.seed if args.seed is not None else config["seed"])
    args.seed = seed
    total_timesteps = int(args.total_timesteps if args.total_timesteps is not None else ppo_cfg["total_timesteps"])
    num_steps = int(args.num_steps if args.num_steps is not None else ppo_cfg["num_steps"])
    nworld = int(args.nworld if args.nworld is not None else ppo_cfg["num_envs"])
    batch_size = nworld * num_steps
    num_minibatches = int(ppo_cfg["num_minibatches"])
    minibatch_size = batch_size // num_minibatches
    num_updates = total_timesteps // batch_size
    if num_updates < 1:
        raise SystemExit("total_timesteps must be at least nworld * num_steps")
    checkpoint_every = int(args.checkpoint_every if args.checkpoint_every is not None else args.video_every)
    if checkpoint_every <= 0:
        checkpoint_every = int(ppo_cfg.get("checkpoint_every_updates", 5))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if args.outdir is None:
        args.outdir = ROOT / "results" / f"mjwarp_muscle_ppo_{time.strftime('%Y%m%d-%H%M%S')}"
    args.outdir.mkdir(parents=True, exist_ok=True)

    model, data = build_muscle_model(config)
    reference = load_reference_from_config(args.reference, model, float(config["control"]["control_hz"]), device, config)
    runner = MJWarpMuscleRunner(
        model=model,
        data=data,
        config=config,
        reference=reference,
        nworld=nworld,
        nconmax=int(args.nconmax),
        njmax=int(args.njmax),
        seed=seed,
        device=device,
    )
    logstd_init = float(args.actor_logstd_init if args.actor_logstd_init is not None else ppo_cfg.get("actor_logstd_init", -0.7))
    initial_action_mean = float(
        args.initial_actor_action_mean
        if args.initial_actor_action_mean is not None
        else ppo_cfg.get("initial_actor_action_mean", 0.0)
    )
    if args.initial_actor_ctrl_mean is not None:
        initial_action_mean = float(args.initial_actor_ctrl_mean)
    agent = Agent(
        runner.obs_dim,
        runner.act_dim,
        logstd_init=logstd_init,
        initial_action_mean=initial_action_mean,
    ).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=float(ppo_cfg["learning_rate"]), eps=1e-5)
    obs_normalizer = ObsNormalizer(
        runner.obs_dim,
        device,
        enabled=bool(ppo_cfg.get("normalize_observations", True)) and not bool(args.no_obs_norm),
        clip=float(ppo_cfg.get("obs_norm_clip", 10.0)),
    )
    global_step = 0
    start_update = 1
    resumed_from: str | None = None
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        agent.load_state_dict(checkpoint["agent_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "obs_normalizer" in checkpoint:
            obs_normalizer.load_state_dict(checkpoint["obs_normalizer"])
        global_step = int(checkpoint.get("global_step", 0))
        start_update = int(checkpoint.get("update", 0)) + 1
        resumed_from = str(args.resume)
        if global_step >= total_timesteps:
            raise SystemExit(f"resume checkpoint global_step={global_step} is already >= total_timesteps={total_timesteps}")
        expected_update = global_step // batch_size + 1
        if start_update != expected_update:
            raise SystemExit(
                f"resume checkpoint update/global_step mismatch: start_update={start_update}, expected={expected_update}"
            )
    initial_execution_mix = activation_prior_execution_mix_for_update(config, start_update)
    config.setdefault("activation_prior", {})["execution_mix"] = initial_execution_mix
    runner.activation_prior_execution_mix = initial_execution_mix

    pretrain_row: dict[str, Any] | None = None
    pretrain_batches = int(ppo_cfg.get("activation_prior_pretrain_batches", 0) or 0)
    if args.resume is None and pretrain_batches > 0:
        pretrain_start = time.perf_counter()
        pretrain_row = pretrain_actor_on_activation_prior(
            agent=agent,
            optimizer=optimizer,
            obs_normalizer=obs_normalizer,
            runner=runner,
            reference=reference,
            batches=pretrain_batches,
            learning_rate=float(ppo_cfg.get("activation_prior_pretrain_lr", ppo_cfg["learning_rate"])),
            max_grad_norm=float(ppo_cfg["max_grad_norm"]),
        )
        pretrain_row.update(
            {
                "seconds_pretrain": time.perf_counter() - pretrain_start,
                "supervised_actuators": ",".join(reference["activation_prior_metadata"].get("supervised_actuators", [])),
            }
        )
        append_csv(args.outdir / "pretrain_metrics.csv", pretrain_row)
        print(json.dumps({"pretrain": pretrain_row}, ensure_ascii=False), flush=True)

    run_config = {
        "config": config,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "reference": {
            "path": reference["path"],
            "length": reference["length"],
            "metadata": reference["metadata"],
            "source_index_count": int(len(reference["source_indices"])),
            "activation_prior": reference["activation_prior_metadata"],
            "emg_prior": reference["emg_prior_metadata"],
        },
        "obs_dim": runner.obs_dim,
        "act_dim": runner.act_dim,
        "action_mapping": "activation = 0.5 * (clamp(action, -1, 1) + 1)",
        "normalize_observations": obs_normalizer.enabled,
        "initial_actor_action_mean": initial_action_mean,
        "activation_prior_pretrain": pretrain_row,
        "resumed_from": resumed_from,
        "start_update": start_update,
        "start_global_step": global_step,
        "checkpoint_every_updates": checkpoint_every,
    }
    write_json(args.outdir / "run_config.json", run_config)

    if args.pretrain_only_video:
        video_row = render_policy_video(
            agent=agent,
            obs_normalizer=obs_normalizer,
            config=config,
            reference=reference,
            args=args,
            device=device,
            update=0,
            global_step=global_step,
        )
        append_csv(args.outdir / "video_metrics.csv", video_row)
        print(json.dumps({"video": video_row}, ensure_ascii=False), flush=True)
        payload = {
            "global_step": global_step,
            "update": 0,
            "run_config": run_config,
            "obs_normalizer": obs_normalizer.state_dict(),
        }
        save_checkpoint(args.outdir / "latest.pt", agent, optimizer, payload)
        return

    obs_buf = torch.zeros((num_steps, nworld, runner.obs_dim), dtype=torch.float32, device=device)
    action_buf = torch.zeros((num_steps, nworld, runner.act_dim), dtype=torch.float32, device=device)
    activation_prior_action_buf = torch.zeros((num_steps, nworld, runner.act_dim), dtype=torch.float32, device=device)
    activation_prior_activation_buf = torch.zeros((num_steps, nworld, runner.act_dim), dtype=torch.float32, device=device)
    emg_prior_activation_buf = torch.zeros((num_steps, nworld, runner.act_dim), dtype=torch.float32, device=device)
    logprob_buf = torch.zeros((num_steps, nworld), dtype=torch.float32, device=device)
    reward_buf = torch.zeros((num_steps, nworld), dtype=torch.float32, device=device)
    done_buf = torch.zeros((num_steps, nworld), dtype=torch.float32, device=device)
    value_buf = torch.zeros((num_steps, nworld), dtype=torch.float32, device=device)
    next_obs_raw = runner.obs()
    next_done = torch.zeros(nworld, dtype=torch.float32, device=device)
    dagger_bc_replay_capacity = int(ppo_cfg.get("dagger_bc_replay_capacity", 0) or 0)
    dagger_bc_replay_ratio = max(0.0, float(ppo_cfg.get("dagger_bc_replay_ratio", 0.0)))
    bc_replay = (
        BcReplayBuffer(dagger_bc_replay_capacity, runner.obs_dim, runner.act_dim, device)
        if dagger_bc_replay_capacity > 0
        else None
    )
    if start_update > num_updates:
        raise SystemExit(f"resume start_update={start_update} exceeds num_updates={num_updates}")

    for update in range(start_update, num_updates + 1):
        update_start = time.perf_counter()
        term_sums: dict[str, torch.Tensor] = {}
        if args.rollout_bc_only:
            activation_prior_execution_mix = max(0.0, min(1.0, float(args.rollout_bc_execution_mix)))
        else:
            activation_prior_execution_mix = activation_prior_execution_mix_for_update(config, update)
        config.setdefault("activation_prior", {})["execution_mix"] = activation_prior_execution_mix
        runner.activation_prior_execution_mix = activation_prior_execution_mix
        reference_curriculum = reference_curriculum_for_update(config, update)
        config.setdefault("reference_curriculum", {})["current_phase_lead_steps"] = int(reference_curriculum["phase_lead_steps"])
        config.setdefault("reference_curriculum", {})["current_phase_tolerance_steps"] = int(reference_curriculum["phase_tolerance_steps"])
        config.setdefault("reference_curriculum", {})["current_swing_exaggeration_scale"] = float(
            reference_curriculum["swing_exaggeration_scale"]
        )
        runner.set_reference_curriculum(
            phase_lead_steps=int(reference_curriculum["phase_lead_steps"]),
            phase_tolerance_steps=int(reference_curriculum["phase_tolerance_steps"]),
            swing_exaggeration_scale=float(reference_curriculum["swing_exaggeration_scale"]),
        )
        activation_prior_mask = reference["activation_prior_mask"]
        activation_prior_active = bool(activation_prior_mask.any().item())
        emg_prior_mask = reference["emg_prior_mask"]
        emg_prior_active = bool(emg_prior_mask.any().item())
        emg_prior_bc_weight = float(config.get("emg_prior", {}).get("bc_weight", 0.0)) if emg_prior_active else 0.0
        bc_weight_start = float(ppo_cfg.get("activation_prior_bc_weight", 0.0))
        bc_weight_final = float(ppo_cfg.get("activation_prior_bc_final_weight", bc_weight_start))
        bc_decay_updates = int(ppo_cfg.get("activation_prior_bc_decay_updates", 0) or 0)
        if bc_decay_updates > 0:
            progress = min(1.0, max(0.0, float(update - 1) / float(bc_decay_updates)))
            activation_prior_bc_weight = bc_weight_start + progress * (bc_weight_final - bc_weight_start)
        else:
            activation_prior_bc_weight = bc_weight_start
        done_count_sum = torch.zeros((), dtype=torch.float32, device=device)
        fall_count_sum = torch.zeros((), dtype=torch.float32, device=device)
        qvel_count_sum = torch.zeros((), dtype=torch.float32, device=device)
        done_return_sum = torch.zeros((), dtype=torch.float32, device=device)
        done_length_sum = torch.zeros((), dtype=torch.float32, device=device)

        rollout_start = time.perf_counter()
        for step in range(num_steps):
            global_step += nworld
            obs_normalizer.update(next_obs_raw)
            next_obs = obs_normalizer.normalize(next_obs_raw)
            obs_buf[step] = next_obs
            done_buf[step] = next_done
            prior_phase = reference_index(runner.phase_idx, reference, config)
            activation_prior_action_buf[step] = reference["activation_prior_action_ref"][prior_phase]
            activation_prior_activation_buf[step] = reference["activation_prior_ref"][prior_phase]
            emg_prior_activation_buf[step] = reference["emg_prior_ref"][prior_phase]
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs, deterministic=bool(args.rollout_bc_only))
            action_buf[step] = action
            logprob_buf[step] = logprob
            value_buf[step] = value.flatten()
            with torch.no_grad():
                next_obs_raw, reward, done, terms = runner.step(action)
            reward_buf[step] = reward
            next_done = done.float()
            for key, value in terms.items():
                if key in {"episode_return_done_sum", "episode_length_done_sum", "done_count"}:
                    continue
                term_sums[key] = term_sums.get(key, torch.zeros((), dtype=torch.float32, device=device)) + value.sum()
            done_count_sum += terms["done_count"].sum()
            fall_count_sum += terms["fall_done"].sum()
            qvel_count_sum += terms["qvel_done"].sum()
            done_return_sum += terms["episode_return_done_sum"].sum()
            done_length_sum += terms["episode_length_done_sum"].sum()
        rollout_seconds = time.perf_counter() - rollout_start

        b_obs = obs_buf.reshape((-1, runner.obs_dim))
        b_logprobs = logprob_buf.reshape(-1)
        b_actions = action_buf.reshape((-1, runner.act_dim))
        b_activation_prior_actions = activation_prior_action_buf.reshape((-1, runner.act_dim))
        b_activation_prior_activations = activation_prior_activation_buf.reshape((-1, runner.act_dim))
        b_emg_prior_activations = emg_prior_activation_buf.reshape((-1, runner.act_dim))
        b_values = value_buf.reshape(-1)
        if bc_replay is not None and activation_prior_active:
            bc_replay.add(
                b_obs,
                b_activation_prior_actions,
                b_activation_prior_activations,
                b_emg_prior_activations,
            )
        clipfracs = []
        bc_losses = []
        activation_bc_losses = []
        emg_bc_losses = []

        learn_start = time.perf_counter()
        if args.rollout_bc_only:
            if not activation_prior_active:
                raise SystemExit("--rollout-bc-only requires an enabled activation_prior")
            for _ in range(int(ppo_cfg["update_epochs"])):
                b_inds = torch.randperm(batch_size, device=device)
                for start in range(0, batch_size, minibatch_size):
                    mb_inds = b_inds[start : start + minibatch_size]
                    actor_mean = agent.actor_mean(b_obs[mb_inds])
                    prior_delta = actor_mean[:, activation_prior_mask] - b_activation_prior_actions[mb_inds][:, activation_prior_mask]
                    activation_prior_bc_loss = torch.mean(torch.square(prior_delta))
                    actor_activation = muscle_action_to_activation(actor_mean)
                    activation_prior_activation_loss = torch.mean(
                        torch.square(
                            actor_activation[:, activation_prior_mask]
                            - b_activation_prior_activations[mb_inds][:, activation_prior_mask]
                        )
                    )
                    loss = activation_prior_bc_loss
                    if emg_prior_active and emg_prior_bc_weight > 0.0:
                        emg_loss = torch.mean(
                            torch.square(actor_activation[:, emg_prior_mask] - b_emg_prior_activations[mb_inds][:, emg_prior_mask])
                        )
                        loss = loss + emg_prior_bc_weight * emg_loss
                        emg_bc_losses.append(emg_loss.detach())
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.actor_mean.parameters(), float(ppo_cfg["max_grad_norm"]))
                    optimizer.step()
                    bc_losses.append(activation_prior_bc_loss.detach())
                    activation_bc_losses.append(activation_prior_activation_loss.detach())
            explained_var = torch.tensor(float("nan"), device=device)
        else:
            with torch.no_grad():
                next_value = agent.get_value(obs_normalizer.normalize(next_obs_raw)).reshape(1, -1)
                advantages = torch.zeros_like(reward_buf)
                lastgaelam = torch.zeros(nworld, dtype=torch.float32, device=device)
                for t in reversed(range(num_steps)):
                    if t == num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - done_buf[t + 1]
                        nextvalues = value_buf[t + 1]
                    delta = reward_buf[t] + float(ppo_cfg["gamma"]) * nextvalues * nextnonterminal - value_buf[t]
                    lastgaelam = delta + float(ppo_cfg["gamma"]) * float(ppo_cfg["gae_lambda"]) * nextnonterminal * lastgaelam
                    advantages[t] = lastgaelam
                returns = advantages + value_buf
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            for _ in range(int(ppo_cfg["update_epochs"])):
                b_inds = torch.randperm(batch_size, device=device)
                for start in range(0, batch_size, minibatch_size):
                    mb_inds = b_inds[start : start + minibatch_size]
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    ratio = (newlogprob - b_logprobs[mb_inds]).exp()
                    with torch.no_grad():
                        clipfracs.append(((ratio - 1.0).abs() > float(ppo_cfg["clip_coef"])).float().mean())
                    mb_adv = b_advantages[mb_inds]
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                    pg_loss = torch.max(
                        -mb_adv * ratio,
                        -mb_adv * torch.clamp(ratio, 1 - float(ppo_cfg["clip_coef"]), 1 + float(ppo_cfg["clip_coef"])),
                    ).mean()
                    value_loss = 0.5 * torch.square(newvalue.view(-1) - b_returns[mb_inds]).mean()
                    entropy_loss = entropy.mean()
                    loss = pg_loss - float(ppo_cfg["ent_coef"]) * entropy_loss + float(ppo_cfg["vf_coef"]) * value_loss
                    if activation_prior_active and activation_prior_bc_weight > 0.0:
                        bc_obs = b_obs[mb_inds]
                        bc_target_action = b_activation_prior_actions[mb_inds]
                        bc_target_activation = b_activation_prior_activations[mb_inds]
                        bc_emg_activation = b_emg_prior_activations[mb_inds]
                        if bc_replay is not None and bc_replay.size > 0 and dagger_bc_replay_ratio > 0.0:
                            replay_n = max(1, int(round(float(mb_inds.numel()) * dagger_bc_replay_ratio)))
                            replay_obs, replay_action, replay_activation, replay_emg = bc_replay.sample(replay_n)
                            bc_obs = torch.cat([bc_obs, replay_obs], dim=0)
                            bc_target_action = torch.cat([bc_target_action, replay_action], dim=0)
                            bc_target_activation = torch.cat([bc_target_activation, replay_activation], dim=0)
                            bc_emg_activation = torch.cat([bc_emg_activation, replay_emg], dim=0)
                        actor_mean = agent.actor_mean(bc_obs)
                        prior_delta = actor_mean[:, activation_prior_mask] - bc_target_action[:, activation_prior_mask]
                        activation_prior_bc_loss = torch.mean(torch.square(prior_delta))
                        loss = loss + float(activation_prior_bc_weight) * activation_prior_bc_loss
                        actor_activation = muscle_action_to_activation(actor_mean)
                        activation_prior_activation_loss = torch.mean(
                            torch.square(actor_activation[:, activation_prior_mask] - bc_target_activation[:, activation_prior_mask])
                        )
                        if emg_prior_active and emg_prior_bc_weight > 0.0:
                            emg_loss = torch.mean(
                                torch.square(actor_activation[:, emg_prior_mask] - bc_emg_activation[:, emg_prior_mask])
                            )
                            loss = loss + emg_prior_bc_weight * emg_loss
                            emg_bc_losses.append(emg_loss.detach())
                        bc_losses.append(activation_prior_bc_loss.detach())
                        activation_bc_losses.append(activation_prior_activation_loss.detach())
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), float(ppo_cfg["max_grad_norm"]))
                    optimizer.step()
            y_pred = b_values.detach()
            y_true = b_returns.detach()
            var_y = torch.var(y_true)
            explained_var = torch.tensor(float("nan"), device=device) if float(var_y.item()) == 0 else 1 - torch.var(y_true - y_pred) / var_y
        learn_seconds = time.perf_counter() - learn_start

        denom = float(num_steps * nworld)
        update_seconds = time.perf_counter() - update_start
        row: dict[str, Any] = {
            "global_step": global_step,
            "update": update,
            "mean_reward": float(reward_buf.mean().item()),
            "mean_value": float(value_buf.mean().item()),
            "explained_var": float(explained_var.item()),
            "clipfrac": float(torch.stack(clipfracs).mean().item()) if clipfracs else 0.0,
            "done_count": float(done_count_sum.item()),
            "fall_count": float(fall_count_sum.item()),
            "qvel_done_count": float(qvel_count_sum.item()),
            "fall_rate_per_step": float(fall_count_sum.item() / denom),
            "qvel_done_rate_per_step": float(qvel_count_sum.item() / denom),
            "mean_episode_return_done": float((done_return_sum / torch.clamp(done_count_sum, min=1.0)).item()),
            "mean_episode_len_done": float((done_length_sum / torch.clamp(done_count_sum, min=1.0)).item()),
            "seconds_rollout": rollout_seconds,
            "seconds_learn": learn_seconds,
            "seconds_update": update_seconds,
            "samples_per_sec_rollout": denom / rollout_seconds if rollout_seconds > 0 else 0.0,
            "samples_per_sec_learn": denom / learn_seconds if learn_seconds > 0 else 0.0,
            "samples_per_sec_update": denom / update_seconds if update_seconds > 0 else 0.0,
            "activation_prior_execution_mix": activation_prior_execution_mix,
            "reference_phase_lead_steps": int(reference_curriculum["phase_lead_steps"]),
            "reference_phase_tolerance_steps": int(reference_curriculum["phase_tolerance_steps"]),
            "reference_swing_exaggeration_scale": float(reference_curriculum["swing_exaggeration_scale"]),
            "rollout_bc_only": int(bool(args.rollout_bc_only)),
            "dagger_bc_replay_size": int(bc_replay.size) if bc_replay is not None else 0,
            "dagger_bc_replay_capacity": int(dagger_bc_replay_capacity),
            "dagger_bc_replay_ratio": float(dagger_bc_replay_ratio),
            "push_interval_steps": int(runner.push_interval_steps),
            "push_probability": float(runner.push_probability),
            "push_pelvis_qvel_std": float(runner.push_pelvis_qvel_std),
            "push_joint_qvel_std": float(runner.push_joint_qvel_std),
            "activation_prior_bc_weight": activation_prior_bc_weight if activation_prior_active else 0.0,
            "activation_prior_bc_loss": float(torch.stack(bc_losses).mean().item()) if bc_losses else 0.0,
            "activation_prior_activation_mse": float(torch.stack(activation_bc_losses).mean().item()) if activation_bc_losses else 0.0,
            "emg_prior_bc_weight": emg_prior_bc_weight,
            "emg_prior_activation_mse": float(torch.stack(emg_bc_losses).mean().item()) if emg_bc_losses else 0.0,
        }
        for key, value in sorted(term_sums.items()):
            row[f"reward_mean_{key}"] = float((value / denom).item())
        append_csv(args.outdir / "train_metrics.csv", row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

        if args.eval_every > 0 and (update % int(args.eval_every) == 0 or update == num_updates):
            saved_execution_mix = float(config.get("activation_prior", {}).get("execution_mix", 0.0))
            if args.rollout_bc_only:
                config.setdefault("activation_prior", {})["execution_mix"] = 0.0
            try:
                eval_row = evaluate(
                    agent=agent,
                    obs_normalizer=obs_normalizer,
                    model=model,
                    data=data,
                    config=config,
                    reference=reference,
                    args=args,
                    device=device,
                    update=update,
                    global_step=global_step,
                )
            finally:
                config.setdefault("activation_prior", {})["execution_mix"] = saved_execution_mix
            append_csv(args.outdir / "eval_metrics.csv", eval_row)
            print(json.dumps({"eval": eval_row}, ensure_ascii=False), flush=True)

        if args.video_every > 0 and (update % int(args.video_every) == 0 or update == num_updates):
            video_row = render_policy_video(
                agent=agent,
                obs_normalizer=obs_normalizer,
                config=config,
                reference=reference,
                args=args,
                device=device,
                update=update,
                global_step=global_step,
            )
            append_csv(args.outdir / "video_metrics.csv", video_row)
            print(json.dumps({"video": video_row}, ensure_ascii=False), flush=True)

        if update % checkpoint_every == 0 or update == num_updates:
            payload = {
                "global_step": global_step,
                "update": update,
                "run_config": run_config,
                "obs_normalizer": obs_normalizer.state_dict(),
            }
            save_checkpoint(args.outdir / f"agent_step_{global_step}.pt", agent, optimizer, payload)
            save_checkpoint(args.outdir / "latest.pt", agent, optimizer, payload)


if __name__ == "__main__":
    main()
