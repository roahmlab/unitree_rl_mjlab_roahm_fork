"""Unitree Go2 flat tracking environment configurations."""

from src.assets.robots.unitree_go2.go2_constants import (
  GO2_ACTION_SCALE,
  get_go2_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg

import src.tasks.tracking.mdp as mdp
from src.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg


def unitree_go2_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 flat terrain tracking configuration."""
  cfg = make_tracking_env_cfg()

  cfg.scene.entities = {"robot": get_go2_robot_cfg()}

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="geom",
      pattern=("FL_foot_collision", "FR_foot_collision", "RL_foot_collision", "RR_foot_collision"),
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="maxforce",
    num_slots=1,
  )

  cfg.scene.sensors = (self_collision_cfg, feet_ground_cfg)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = GO2_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = "base_link"
  motion_cmd.body_names = (
    "base_link",
    "FL_hip",
    "FL_thigh",
    "FL_calf",
    "FR_hip",
    "FR_thigh",
    "FR_calf",
    "RL_hip",
    "RL_thigh",
    "RL_calf",
    "RR_hip",
    "RR_thigh",
    "RR_calf",
  )

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(FL|FR|RL|RR)_foot_collision$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)

  cfg.events["joint_friction"] = EventTermCfg(
    mode="startup",
    func=dr.joint_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "operation": "abs",
      "ranges": {
        ".*hip_joint": (0.1454, 0.1469),
        ".*thigh_joint": (0.1206, 0.1228),
        ".*calf_joint": (1.4809, 1.4868),
      },
    },
  )

  cfg.events["joint_damping"] = EventTermCfg(
    mode="startup",
    func=dr.joint_damping,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "operation": "abs",
      "ranges": {
        ".*hip_joint": (0.0269, 0.0295),
        ".*thigh_joint": (0.0389, 0.0423),
        ".*calf_joint": (0.0247, 0.0323),
      },
    },
  )

  cfg.events["joint_armature"] = EventTermCfg(
    mode="startup",
    func=dr.joint_armature,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "operation": "abs",
      "ranges": {
        ".*hip_joint": (0.0001, 0.0008),      # unidentifiable (point est. 0.0003985) -- wide
        ".*thigh_joint": (0.0038277, 0.0042258),  # identifiable -- tight, from CI
        ".*calf_joint": (0.0183563, 0.0187847),   # identifiable -- tight, from CI
      },
    },
  )

  cfg.events["joint_bias_torque"] = EventTermCfg(
    mode="reset",
    func=mdp.joint_bias_torque,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "bias_ranges": {
        ".*hip_joint": (-0.3, 0.3),          # unidentifiable -- wide, centered near 0
        ".*thigh_joint": (-0.2, 0.2),        # unidentifiable -- wide, centered near 0
        ".*calf_joint": (-0.1152, -0.1098),  # identifiable -- tight, from CI
      },
    },
  )

  cfg.events["motor_strength_scale"] = EventTermCfg(
    mode="reset",
    func=mdp.motor_strength_scale,
    params={"asset_cfg": SceneEntityCfg("robot"), "scale_range": (0.8, 1.0)},
  )

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "FL_calf",
    "FR_calf",
    "RL_calf",
    "RR_calf",
  )

  cfg.viewer.body_name = "base_link"

  cfg.rewards["motion_contact_tracking"] = RewardTermCfg(
    func=mdp.motion_contact_tracking,
    weight=2.0,  # tune this
    params={"command_name": "motion", "sensor_name": "feet_ground_contact"},
  )

  cfg.rewards["swing_leg_ground_clearance"] = RewardTermCfg(
    func=mdp.swing_leg_ground_clearance,
    weight=-2.0,  # tune -- start moderate, increase if calves still skim
    params={"command_name": "motion", "sensor_name": "feet_ground_contact", "clearance": 0.06},
  )

  cfg.curriculum["anneal_body_pos_std"] = CurriculumTermCfg(
    func=mdp.param_curriculum,
    params={
      "reward_name": "motion_body_pos",
      "param_name": "std",
      "stages": [{"step": 0, "value": 0.3}, {"step": 2000, "value": 0.15}, {"step": 5000, "value": 0.1}],
    },
  )
  cfg.curriculum["anneal_global_root_pos_std"] = CurriculumTermCfg(
    func=mdp.param_curriculum,
    params={
      "reward_name": "motion_global_root_pos",
      "param_name": "std",
      "stages": [{"step": 0, "value": 0.3}, {"step": 2000, "value": 0.15}, {"step": 5000, "value": 0.1}],
    },
  )

  # Modify observations if we don't have state estimation.
  if not has_state_estimation:
    new_actor_terms = {
      k: v
      for k, v in cfg.observations["actor"].terms.items()
      if k not in ["motion_anchor_pos_b", "base_lin_vel"]
    }
    cfg.observations["actor"] = ObservationGroupCfg(
      terms=new_actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    # Disable RSI randomization.
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}

    motion_cmd.sampling_mode = "start"

  return cfg
