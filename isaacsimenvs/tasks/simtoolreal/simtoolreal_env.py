"""Thin DirectRLEnv wrapper for SimToolReal.

The env owns Isaac Lab hook wiring and state buffers. Task math lives in the
utility modules called from each hook.
"""

from __future__ import annotations

import torch

from isaaclab.envs import DirectRLEnv

from .simtoolreal_env_cfg import SimToolRealEnvCfg
from .utils.action_utils import apply_action_pipeline, apply_wrench_dr
from .utils.logging_utils import log_step_metrics
from .utils.obs_utils import (
    build_observations,
    build_student_observations,
    compute_intermediate_values,
    compute_obs_dim,
)
from .utils.reset_utils import allocate_state_buffers, reset_env_state, reset_goal_trackers
from .utils.reward_utils import compute_rewards
from .utils.scene_utils import apply_physx_material_properties, setup_scene
from .utils.termination_utils import compute_terminations, update_tolerance_curriculum


__all__ = ["SimToolRealEnv", "SimToolRealEnvCfg"]


class SimToolRealEnv(DirectRLEnv):
    cfg: SimToolRealEnvCfg
    supports_timeout_bootstrap = True

    def __init__(
        self, cfg: SimToolRealEnvCfg, render_mode: str | None = None, **kwargs
    ) -> None:
        # Override obs/state space from configured field lists before
        # DirectRLEnv / rl_games observes the configclass.
        cfg.observation_space = compute_obs_dim(cfg.obs.obs_list)
        cfg.state_space = compute_obs_dim(cfg.obs.state_list)

        super().__init__(cfg, render_mode, **kwargs)  # runs _setup_scene
        apply_physx_material_properties(self)
        allocate_state_buffers(self)

    def _setup_scene(self) -> None:
        setup_scene(self)

    def _reset_idx(self, env_ids) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        super()._reset_idx(env_ids)
        reset_env_state(
            self,
            torch.as_tensor(env_ids, device=self.device, dtype=torch.long),
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Legacy Isaac Gym applies a queued goal reset at the beginning of the
        # next policy step. Thus the observation returned on the success step
        # still contains the goal that produced that success.
        goal_reset_ids = self._pending_goal_reset.nonzero(as_tuple=False).squeeze(-1)
        if goal_reset_ids.numel() > 0:
            reset_goal_trackers(self, goal_reset_ids)
            self.episode_length_buf[goal_reset_ids] = 0
            self._pending_goal_reset[goal_reset_ids] = False
        apply_action_pipeline(self, actions)
        apply_wrench_dr(self)

    def _apply_action(self) -> None:
        # Called decimation times per policy step; idempotent.
        self.robot.set_joint_position_target(self._cur_targets)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        update_tolerance_curriculum(self)
        compute_intermediate_values(self)
        return compute_terminations(self)

    def _get_rewards(self) -> torch.Tensor:
        reward = compute_rewards(self)
        # DirectRLEnv assigns reward_buf only after this hook returns, while the
        # observation contains the current reward. Make it current before
        # capturing terminal observations.
        self.reward_buf = reward
        log_step_metrics(self)

        done_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            # Terminal observations are extra bookkeeping. They must not
            # perturb the random stream used by the next regular observation.
            cpu_rng_state = torch.random.get_rng_state()
            device = torch.device(self.device)
            cuda_rng_state = (
                torch.cuda.get_rng_state(device) if device.type == "cuda" else None
            )
            try:
                final_obs = build_observations(self, done_ids)
            finally:
                torch.random.set_rng_state(cpu_rng_state)
                if cuda_rng_state is not None:
                    torch.cuda.set_rng_state(cuda_rng_state, device)
            for name, value in final_obs.items():
                self._final_obs_buf[name][done_ids] = value
            self.extras["final_obs"] = self._final_obs_buf
            self.extras["_final_obs"] = self.reset_buf.detach().clone()
        else:
            self.extras.pop("final_obs", None)
            self.extras.pop("_final_obs", None)

        return reward

    def _get_observations(self) -> dict[str, torch.Tensor]:
        return build_observations(self)

    def get_student_obs(self) -> dict[str, torch.Tensor]:
        """Return opt-in student observations for distillation code."""
        return build_student_observations(self)
