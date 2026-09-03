from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
import numpy as np
import re

from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_error_magnitude

from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _get_body_indexes(
  command: MotionCommand, body_names: tuple[str, ...] | None
) -> list[int]:
  return [
    i
    for i, name in enumerate(command.cfg.body_names)
    if (body_names is None) or (name in body_names)
  ]


def motion_global_anchor_position_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.sum(
    torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1
  )
  return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
  return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_pos_relative_w[:, body_indexes]
      - command.robot_body_pos_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = (
    quat_error_magnitude(
      command.body_quat_relative_w[:, body_indexes],
      command.robot_body_quat_w[:, body_indexes],
    )
    ** 2
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_lin_vel_w[:, body_indexes]
      - command.robot_body_lin_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_ang_vel_w[:, body_indexes]
      - command.robot_body_ang_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions.

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  return data.found.squeeze(-1)

def _leg_prefix(name: str) -> str:
  m = re.match(r"^(FL|FR|RL|RR)", str(name))
  if m is None:
      raise ValueError(f"Couldn't extract a leg prefix (FL/FR/RL/RR) from '{name}'")
  return m.group(1)

class motion_contact_tracking:
  """Reward/penalty for matching a reference foot-contact schedule.

  Reads ``foot_contacts`` ([T, F], 1=contact) and ``foot_contact_names`` ([F])
  from the same .npz used by *command_name*'s MotionCommand, permutes columns
  to match *sensor_name*'s ContactSensor.primary_names order, then penalizes
  frame-by-frame disagreement between reference and actual (sensor ``found``)
  contact state, indexed by the command's own ``time_steps`` so it stays in
  sync with the position/orientation tracking terms above.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    command = cast(
      MotionCommand, env.command_manager.get_term(cfg.params["command_name"])
    )
    sensor: ContactSensor = env.scene[cfg.params["sensor_name"]]

    data = np.load(command.cfg.motion_file)
    if "foot_contacts" not in data or "foot_contact_names" not in data:
      raise KeyError(
        "Motion file is missing 'foot_contacts'/'foot_contact_names' -- add "
        "both when building the .npz."
      )

    sensor_primary_names = list(dict.fromkeys(
      slot.primary_name for slot in sensor._slots
    ))

    contact_names = [str(n) for n in data["foot_contact_names"]]
    contact_prefix_to_idx = {_leg_prefix(n): i for i, n in enumerate(contact_names)}
    perm = [contact_prefix_to_idx[_leg_prefix(n)] for n in sensor_primary_names]
    foot_contacts = data["foot_contacts"][:, perm]

    assert foot_contacts.shape[0] == command.motion.time_step_total, (
      "foot_contacts row count doesn't match the rest of the motion -- "
    )
    self.foot_contacts = torch.tensor(
      foot_contacts, dtype=torch.float32, device=env.device
    )

  def __call__(
    self, env: ManagerBasedRlEnv, command_name: str, sensor_name: str
  ) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    sensor: ContactSensor = env.scene[sensor_name]
    assert sensor.data.found is not None

    ref_contact = self.foot_contacts[command.time_steps]  # [B, F]
    actual_contact = (sensor.data.found > 0).float()  # [B, F]

    false_contact = actual_contact * (1.0 - ref_contact)  # foot down, shouldn't be
    missed_contact = (1.0 - actual_contact) * ref_contact  # foot up, should be down
    penalty = 2.0 * false_contact + 0.5 * missed_contact
    return -penalty.mean(dim=-1)

class swing_leg_ground_clearance:
  """Penalize the thigh/calf links dropping below a height margin during
  their reference-scheduled swing phase (not just the foot contact point) --
  assumes flat terrain at world z=0.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    command = cast(
      MotionCommand, env.command_manager.get_term(cfg.params["command_name"])
    )
    sensor: ContactSensor = env.scene[cfg.params["sensor_name"]]

    data = np.load(command.cfg.motion_file)
    if "foot_contacts" not in data or "foot_contact_names" not in data:
      raise KeyError(
        "Motion file is missing 'foot_contacts'/'foot_contact_names'."
      )

    sensor_primary_names = list(dict.fromkeys(
      slot.primary_name for slot in sensor._slots
    ))
    contact_names = [str(n) for n in data["foot_contact_names"]]
    contact_prefix_to_idx = {_leg_prefix(n): i for i, n in enumerate(contact_names)}
    perm = [contact_prefix_to_idx[_leg_prefix(n)] for n in sensor_primary_names]
    foot_contacts = data["foot_contacts"][:, perm]

    assert foot_contacts.shape[0] == command.motion.time_step_total
    self.foot_contacts = torch.tensor(
      foot_contacts, dtype=torch.float32, device=env.device
    )
    # this array's columns are in sensor_primary_names / leg order:
    self.leg_order = [_leg_prefix(n) for n in sensor_primary_names]

    # For each leg, which entries of command.cfg.body_names are its thigh/calf.
    self.leg_link_indexes: dict[str, list[int]] = {}
    for leg in ("FL", "FR", "RL", "RR"):
      self.leg_link_indexes[leg] = [
        i for i, name in enumerate(command.cfg.body_names)
        if name.startswith(leg) and ("thigh" in name or "calf" in name)
      ]

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    margin: float,
  ) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    ref_contact = self.foot_contacts[command.time_steps]
    swing = 1.0 - ref_contact

    total = torch.zeros(env.num_envs, device=env.device)
    for col, leg in enumerate(self.leg_order):
      idx = self.leg_link_indexes[leg]
      if not idx:
        continue
      z_actual = command.robot_body_pos_w[:, idx, 2]
      z_ref = command.body_pos_w[:, idx, 2]
      target = torch.clamp(z_ref - margin, min=0.0)   # reference height, backed off by a margin
      deficit = torch.clamp(target - z_actual, min=0.0)
      normalized = (deficit / max(margin, 1e-3)) ** 2
      total += swing[:, col] * torch.sum(normalized, dim=-1)
    return total

def motion_joint_position_error_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
) -> torch.Tensor:
    motion = env.command_manager.get_term(command_name)

    error = motion.joint_pos - motion.robot_joint_pos

    return torch.exp(
        -(error.square().sum(dim=-1)) / (std**2)
    )

def motion_joint_velocity_error_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
) -> torch.Tensor:
    motion = env.command_manager.get_term(command_name)

    error = motion.joint_vel - motion.robot_joint_vel

    return torch.exp(
        -(error.square().sum(dim=-1)) / (std**2)
    )

def motion_anchor_linear_velocity_error_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
) -> torch.Tensor:
    motion = env.command_manager.get_term(command_name)

    error = (
        motion.anchor_lin_vel_w
        - motion.robot_anchor_lin_vel_w
    )

    return torch.exp(
        -(error.square().sum(dim=-1)) / (std**2)
    )

def motion_anchor_angular_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.sum(
    torch.square(command.anchor_ang_vel_w - command.robot_anchor_ang_vel_w), dim=-1
  )
  return torch.exp(-error / std**2)

def _quat_apply(quat_wxyz: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
  """Rotate vec (..., 3) by quat_wxyz (..., 4), scalar-first."""
  w, x, y, z = quat_wxyz.unbind(-1)
  qvec = torch.stack([x, y, z], dim=-1)
  uv = torch.cross(qvec, vec, dim=-1)
  uuv = torch.cross(qvec, uv, dim=-1)
  return vec + 2 * (w.unsqueeze(-1) * uv + uuv)

class nose_ground_clearance:
  """Penalize/terminate the front sensor-housing bump (base3_collision, a
  fixed offset on base_link -- go2.xml has no separate head body) getting
  too close to the ground."""

  NOSE_OFFSET_LOCAL = (0.293, 0.0, -0.06)  # base3_collision pos, from go2.xml

  def __call__(self, env, asset_cfg, min_height: float) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    base_pos = asset.data.body_link_pos_w[:, 0]    # base_link, index 0
    base_quat = asset.data.body_link_quat_w[:, 0]
    offset = torch.tensor(self.NOSE_OFFSET_LOCAL, device=env.device)
    world_offset = _quat_apply(base_quat, offset.expand(base_pos.shape[0], -1))
    nose_z = (base_pos + world_offset)[:, 2]
    return min_height - nose_z    # reward version: clamp(..., min=0) below; termination: compare > 0


class handstand_horizontal_drift_penalty:
  LEAD_FRAMES = 25  # tune against how many frames before liftoff the drift visibly starts

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    command = cast(MotionCommand, env.command_manager.get_term(cfg.params["command_name"]))
    sensor: ContactSensor = env.scene[cfg.params["sensor_name"]]
    data = np.load(command.cfg.motion_file)
    sensor_primary_names = list(dict.fromkeys(slot.primary_name for slot in sensor._slots))
    contact_names = [str(n) for n in data["foot_contact_names"]]
    contact_prefix_to_idx = {_leg_prefix(n): i for i, n in enumerate(contact_names)}
    perm = [contact_prefix_to_idx[_leg_prefix(n)] for n in sensor_primary_names]
    foot_contacts = torch.tensor(data["foot_contacts"][:, perm], dtype=torch.float32, device=env.device)
    leg_order = [_leg_prefix(n) for n in sensor_primary_names]
    rear_cols = [i for i, leg in enumerate(leg_order) if leg in ("RL", "RR")]
    rear_off = foot_contacts[:, rear_cols].sum(dim=-1) == 0
    active = rear_off.clone()
    for shift in range(1, self.LEAD_FRAMES + 1):
      shifted = torch.roll(rear_off, shifts=-shift, dims=0)
      shifted[-shift:] = False
      active = active | shifted
    self.phase_active = active.float()

  def __call__(self, env: ManagerBasedRlEnv, command_name: str, sensor_name: str) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    active = self.phase_active[command.time_steps]
    lin_vel_err = command.anchor_lin_vel_w[:, :2] - command.robot_anchor_lin_vel_w[:, :2]
    return active * torch.sum(torch.square(lin_vel_err), dim=-1)