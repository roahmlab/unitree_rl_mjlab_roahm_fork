from mjlab.actuator.dc_actuator import DcMotorActuator
from mjlab.entity import Entity
from mjlab.utils.lab_api.math import sample_uniform
from mjlab.managers.scene_entity_config import SceneEntityCfg

def joint_bias_torque(
    env, env_ids, bias_ranges: dict[str, tuple[float, float]], asset_cfg=SceneEntityCfg("robot"),
) -> None:
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  asset: Entity = env.scene[asset_cfg.name]
  for pattern, (lo, hi) in bias_ranges.items():
    local_ids, _ = asset.find_joints(pattern, preserve_order=True)
    dof_adr = asset.indexing.joint_v_adr[local_ids]
    bias = sample_uniform(lo, hi, (len(env_ids), len(dof_adr)), env.device)
    env.sim.data.qfrc_applied[env_ids[:, None], dof_adr] = bias

def motor_strength_scale(
    env, env_ids, scale_range: tuple[float, float], asset_cfg=SceneEntityCfg("robot"),
) -> None:
  """Scale the whole torque-speed curve (peak torque) to approximate
  motor-to-motor strength variation. Requires DcMotorActuator."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  asset: Entity = env.scene[asset_cfg.name]

  if isinstance(asset_cfg.actuator_ids, list):
    actuators = [asset.actuators[i] for i in asset_cfg.actuator_ids]
  else:
    actuators = asset.actuators[asset_cfg.actuator_ids]
  if not isinstance(actuators, list):
    actuators = [actuators]

  for actuator in actuators:
    if not isinstance(actuator, DcMotorActuator):
      raise TypeError("motor_strength_scale requires DcMotorActuator-based actuators")
    assert actuator.saturation_effort is not None
    n = actuator.saturation_effort.shape[1]
    scale = sample_uniform(scale_range[0], scale_range[1], (len(env_ids), n), env.device)
    # Scale from the nominal cfg value, not the current tensor, so repeated
    # resets don't compound drift.
    actuator.saturation_effort[env_ids] = actuator.cfg.saturation_effort * scale