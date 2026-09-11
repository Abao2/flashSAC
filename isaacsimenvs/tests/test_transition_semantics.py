"""Focused regression checks for reward/reset transition ordering.

Run with:
    .venv_isaacsim/bin/python isaacsimenvs/tests/test_transition_semantics.py --headless
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace
from unittest.mock import patch

from isaaclab.app import AppLauncher


def main() -> None:
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    print("[test] launching Isaac Sim", flush=True)
    app = AppLauncher(args).app
    print("[test] Isaac Sim ready", flush=True)

    import torch

    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env import SimToolRealEnv
    from isaacsimenvs.tasks.simtoolreal.utils.reward_utils import compute_rewards
    from isaacsimenvs.tasks.simtoolreal.utils.termination_utils import compute_terminations

    assert SimToolRealEnv.supports_timeout_bootstrap is True

    termination_env = SimpleNamespace(
        cfg=SimpleNamespace(
            termination=SimpleNamespace(
                max_consecutive_successes=0,
                reset_when_dropped=False,
            )
        ),
        scene=SimpleNamespace(env_origins=torch.zeros(1, 3)),
        object=SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.tensor([[0.0, 0.0, 0.5]]))),
        _is_success=torch.tensor([True]),
        _successes=torch.zeros(1, dtype=torch.long),
        _pending_goal_reset=torch.tensor([False]),
        _curr_fingertip_distances=torch.zeros(1, 5),
        _object_init_z=torch.tensor([0.5]),
        _lifted_object=torch.tensor([False]),
        episode_length_buf=torch.tensor([20]),
        max_episode_length=20,
    )
    terminated, truncated = compute_terminations(termination_env)
    assert not terminated.item() and not truncated.item()
    assert termination_env._pending_goal_reset.item()
    assert termination_env.episode_length_buf.item() == 20
    print("[test] goal hit is queued and does not truncate before reward", flush=True)

    termination_env.cfg.termination.reset_when_dropped = True
    termination_env._is_success.zero_()
    termination_env._lifted_object.fill_(True)
    termination_env.object.data.root_pos_w[:, 2] = 0.49
    terminated, truncated = compute_terminations(termination_env)
    assert terminated.item() and not truncated.item()
    assert termination_env._termination_reasons["dropped"].item()

    termination_env._lifted_object.zero_()
    terminated, truncated = compute_terminations(termination_env)
    assert not terminated.item() and truncated.item()
    print("[test] legacy resetWhenDropped is a failure, not a timeout", flush=True)

    reward_cfg = SimpleNamespace(
        lifting_bonus_threshold=0.15,
        lifting_bonus=0.0,
        lifting_rew_scale=0.0,
        distance_delta_rew_scale=0.0,
        keypoint_rew_scale=10.0,
        kuka_actions_penalty_scale=0.0,
        hand_actions_penalty_scale=0.0,
        reach_goal_bonus=0.0,
    )
    env = SimpleNamespace(
        cfg=SimpleNamespace(
            reward=reward_cfg,
            termination=SimpleNamespace(success_steps=1, force_consecutive_near_goal_steps=True),
        ),
        scene=SimpleNamespace(env_origins=torch.zeros(1, 3)),
        object=SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.tensor([[0.0, 0.0, 0.5]]))),
        robot=SimpleNamespace(data=SimpleNamespace(joint_vel=torch.zeros(1, 2))),
        _object_init_z=torch.tensor([0.5]),
        _lifted_object=torch.tensor([True]),
        _curr_fingertip_distances=torch.zeros(1, 1),
        _closest_fingertip_dist=torch.zeros(1, 1),
        _keypoints_max_dist=torch.tensor([0.1]),
        _closest_keypoint_max_dist=torch.tensor([0.2]),
        _arm_joint_ids=torch.tensor([0]),
        _hand_joint_ids=torch.tensor([1]),
        _near_goal=torch.tensor([True]),
        _is_success=torch.tensor([True]),
        reset_buf=torch.tensor([False]),
        _pending_goal_reset=torch.tensor([True]),
        episode_length_buf=torch.tensor([17]),
        reward_buf=torch.zeros(1),
        extras={},
    )

    def reset_goal(fake_env, env_ids):
        fake_env._closest_keypoint_max_dist[env_ids] = -1.0

    with (
        patch(
            "isaacsimenvs.tasks.simtoolreal.simtoolreal_env.compute_rewards",
            side_effect=compute_rewards,
        ),
        patch("isaacsimenvs.tasks.simtoolreal.simtoolreal_env.log_step_metrics"),
        patch(
            "isaacsimenvs.tasks.simtoolreal.simtoolreal_env.reset_goal_trackers",
            side_effect=reset_goal,
        ),
    ):
        reward = SimToolRealEnv._get_rewards(env)

    torch.testing.assert_close(reward, torch.tensor([1.0]))
    assert env._reward_terms["keypoint_rew"].item() == 1.0
    torch.testing.assert_close(env._closest_keypoint_max_dist, torch.tensor([0.1]))
    assert env.episode_length_buf.item() == 17
    assert env._pending_goal_reset.item()

    with (
        patch("isaacsimenvs.tasks.simtoolreal.simtoolreal_env.apply_action_pipeline"),
        patch("isaacsimenvs.tasks.simtoolreal.simtoolreal_env.apply_wrench_dr"),
        patch(
            "isaacsimenvs.tasks.simtoolreal.simtoolreal_env.reset_goal_trackers",
            side_effect=reset_goal,
        ),
    ):
        SimToolRealEnv._pre_physics_step(env, torch.zeros(1, 2))

    assert env._closest_keypoint_max_dist.item() == -1.0
    assert env.episode_length_buf.item() == 0
    assert not env._pending_goal_reset.item()
    print("[test] successful-step observation keeps old goal; next pre-step resets it", flush=True)

    env = SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        reset_buf=torch.tensor([True]),
        _pending_goal_reset=torch.tensor([False]),
        reward_buf=torch.zeros(1),
        extras={},
        state=torch.tensor([[7.0]]),
        _final_obs_buf={
            "policy": torch.zeros(1, 1),
            "critic": torch.zeros(1, 1),
        },
    )

    def observations(fake_env, env_ids):
        assert env_ids.tolist() == [0]
        torch.rand(1)  # terminal obs-side noise must not advance global RNG
        return {
            "policy": fake_env.state[env_ids].clone(),
            "critic": (fake_env.state[env_ids] + 10.0).clone(),
        }

    torch.manual_seed(123)
    expected_next_random = torch.rand(1)
    torch.manual_seed(123)
    with (
        patch(
            "isaacsimenvs.tasks.simtoolreal.simtoolreal_env.compute_rewards",
            return_value=torch.tensor([3.0]),
        ),
        patch("isaacsimenvs.tasks.simtoolreal.simtoolreal_env.log_step_metrics"),
        patch(
            "isaacsimenvs.tasks.simtoolreal.simtoolreal_env.build_observations",
            side_effect=observations,
        ),
    ):
        SimToolRealEnv._get_rewards(env)

    env.state.fill_(99.0)  # stand-in for DirectRLEnv's immediate autoreset
    torch.testing.assert_close(env.extras["final_obs"]["policy"], torch.tensor([[7.0]]))
    torch.testing.assert_close(env.extras["final_obs"]["critic"], torch.tensor([[17.0]]))
    assert env.extras["_final_obs"].tolist() == [True]
    torch.testing.assert_close(torch.rand(1), expected_next_random)
    print("[test] final policy+critic observations survive autoreset unchanged", flush=True)

    import gymnasium as gym

    import isaacsimenvs  # noqa: F401  registers the task
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg
    from isaacsimenvs.tasks.simtoolreal.utils.obs_utils import OBS_FIELD_SIZES

    cfg = SimToolRealEnvCfg()
    cfg.scene.num_envs = 1
    cfg.assets.num_assets_per_type = 1
    cfg.domain_randomization.use_obs_delay = False
    cfg.domain_randomization.use_object_state_delay_noise = False
    cfg.domain_randomization.joint_velocity_obs_noise_std = 0.0
    cfg.termination.eval_success_tolerance = 0.0
    cfg.termination.reset_when_dropped = True
    live_env = gym.make("Isaacsimenvs-SimToolReal-Direct-v0", cfg=cfg)
    inner = live_env.unwrapped
    live_env.reset()
    # DirectRLEnv increments before _get_dones(); legacy terminates when the
    # resulting progress reaches max_episode_length - 1.
    inner.episode_length_buf.fill_(inner.max_episode_length - 2)
    obs, _, terminated, truncated, info = live_env.step(
        torch.zeros(1, cfg.action_space, device=inner.device)
    )
    assert not terminated.item() and truncated.item()
    progress_offset = sum(
        OBS_FIELD_SIZES[name]
        for name in cfg.obs.state_list[: cfg.obs.state_list.index("progress")]
    )
    final_progress = info["final_obs"]["critic"][0, progress_offset]
    reset_progress = obs["critic"][0, progress_offset]
    expected_progress = torch.log(
        torch.tensor((inner.max_episode_length - 1) / 10.0 + 1.0, device=inner.device)
    )
    torch.testing.assert_close(final_progress, expected_progress)
    torch.testing.assert_close(reset_progress, torch.zeros_like(reset_progress))
    assert info["final_obs"]["policy"].shape == (1, cfg.observation_space)
    assert info["final_obs"]["critic"].shape == (1, cfg.state_space)
    print("[test] live DirectRLEnv timeout returns pre-reset final_obs", flush=True)

    live_env.close()

    app.close()


if __name__ == "__main__":
    main()
