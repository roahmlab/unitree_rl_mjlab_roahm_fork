from __future__ import annotations
from typing import TYPE_CHECKING
import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class param_curriculum:
  """Linearly anneal one reward term's param (e.g. `std`) across training-step
  stages. Extends this mjlab version's `reward_weight` (which only touches
  `.weight`) to work on `.params` entries instead.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self._term_cfg = env.reward_manager.get_term_cfg(cfg.params["reward_name"])

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    reward_name: str,
    param_name: str,
    stages: list[dict],  # [{"step": int, "value": float}, ...], sorted by step
  ) -> torch.Tensor:
    del env_ids, reward_name
    step = env.common_step_counter
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
    self._term_cfg.params[param_name] = value
    return torch.tensor([value])