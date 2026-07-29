from __future__ import annotations
from typing import TYPE_CHECKING
import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _interpolate_stages(step: int, stages: list[dict]) -> float:
  stages = sorted(stages, key=lambda s: s["step"])
  value = stages[0]["value"]
  for s0, s1 in zip(stages, stages[1:]):
    if s0["step"] <= step <= s1["step"]:
      frac = (step - s0["step"]) / max(s1["step"] - s0["step"], 1)
      value = s0["value"] + frac * (s1["value"] - s0["value"])
      break
  else:
    if step > stages[-1]["step"]:
      value = stages[-1]["value"]
  return value


class param_curriculum:
  """Anneal a reward term's param (e.g. `std`)."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self._term_cfg = env.reward_manager.get_term_cfg(cfg.params["reward_name"])

  def __call__(self, env, env_ids, reward_name, param_name, stages) -> torch.Tensor:
    del env_ids, reward_name
    value = _interpolate_stages(env.common_step_counter, stages)
    self._term_cfg.params[param_name] = value
    return torch.tensor([value])


class termination_param_curriculum:
  """Anneal a termination term's param (e.g. `threshold`)."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self._term_cfg = env.termination_manager.get_term_cfg(cfg.params["termination_name"])

  def __call__(self, env, env_ids, termination_name, param_name, stages) -> torch.Tensor:
    del env_ids, termination_name
    value = _interpolate_stages(env.common_step_counter, stages)
    self._term_cfg.params[param_name] = value
    return torch.tensor([value])