import tempfile
import unittest
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from flash_rl.agents.flashSAC.agent import _resolve_observation_layout
from flash_rl.common.logger import AverageMeterDict, TensorboardTrainerLogger
from flash_rl.envs.isaaclab import (
    IsaacLabVectorEnv,
    apply_registered_task_overrides,
    combined_observation_shape,
    observation_subspaces,
)
from flash_rl.evaluation import evaluate


class AsymmetricObservationLayoutTest(unittest.TestCase):
    def test_separate_isaaclab_critic_uses_only_critic_state(self) -> None:
        layout = _resolve_observation_layout(
            302,
            {
                "actor_observation_size": (140,),
                "critic_observation_size": (162,),
                "critic_observation_offset": 140,
            },
            True,
        )
        self.assertEqual(layout, (140, 162, 140))

    def test_legacy_privileged_prefix_convention_remains_supported(self) -> None:
        self.assertEqual(
            _resolve_observation_layout(
                100,
                {"actor_observation_size": (50,)},
                True,
            ),
            (50, 100, 0),
        )


class _Config:
    def __init__(self) -> None:
        self.values: dict = {}

    def from_dict(self, values: dict) -> None:
        self.values.update(values)


class _FakeIsaacEnv:
    def step(self, actions: torch.Tensor):
        self.last_actions = actions
        observations = {
            "policy": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "critic": torch.tensor([[5.0], [6.0]]),
        }
        rewards = torch.tensor([1.0, 2.0])
        terminated = torch.tensor([False, True])
        truncated = torch.tensor([True, False])
        infos = {
            "kept_by_adapter": torch.tensor([7.0, 8.0]),
            "episode_cumulative": {"bonus": torch.tensor([0.5, 1.0])},
            "episode_final": {
                "successes": torch.tensor([3.0, 4.0]),
                "all_goals_hit": torch.tensor([1.0, 0.0]),
            },
            "current_success_tolerance": 0.05,
        }
        return observations, rewards, terminated, truncated, infos


class _FinalObsIsaacEnv(_FakeIsaacEnv):
    def step(self, actions: torch.Tensor):
        observations, rewards, terminated, truncated, infos = super().step(actions)
        infos["final_obs"] = {
            "policy": torch.tensor([[9.0, 8.0], [7.0, 6.0]]),
            "critic": torch.tensor([[5.0], [4.0]]),
        }
        return observations, rewards, terminated, truncated, infos


class ExternalIsaacLabAdapterTest(unittest.TestCase):
    def test_gym_dict_critic_subspace_is_detected_by_key(self) -> None:
        space = gym.spaces.Dict(
            {
                "policy": gym.spaces.Box(-1.0, 1.0, shape=(140,), dtype=np.float32),
                "critic": gym.spaces.Box(-1.0, 1.0, shape=(162,), dtype=np.float32),
            }
        )
        subspaces = observation_subspaces(space)
        self.assertIn("critic", subspaces)
        self.assertEqual(subspaces["policy"].shape, (140,))
        self.assertEqual(subspaces["critic"].shape, (162,))
        self.assertEqual(combined_observation_shape(subspaces["policy"].shape, subspaces["critic"].shape), (302,))

    def test_registered_yaml_is_applied_before_explicit_overrides(self) -> None:
        task_id = "FlashSAC-External-Config-Test-v0"
        with tempfile.TemporaryDirectory() as directory:
            yaml_path = Path(directory) / "task.yaml"
            yaml_path.write_text("controller:\n  smoothing: 0.07\n", encoding="utf-8")
            gym.register(
                id=task_id,
                entry_point=lambda: None,
                kwargs={"env_cfg_yaml_entry_point": str(yaml_path)},
            )
            try:
                cfg = _Config()
                apply_registered_task_overrides(
                    cfg,
                    env_name=task_id,
                    yaml_entry_point_key="env_cfg_yaml_entry_point",
                    task_cfg_overrides={"controller": {"smoothing": 0.09}},
                )
                self.assertEqual(cfg.values["controller"]["smoothing"], 0.09)
            finally:
                del gym.registry[task_id]

    def test_step_preserves_metrics_and_masks_autoreset_timeouts(self) -> None:
        env = object.__new__(IsaacLabVectorEnv)
        env.envs = _FakeIsaacEnv()
        env.device = "cpu"
        env.action_bounds = 1.0
        env.asymmetric_obs = True
        env.to_numpy = True
        env.bootstrap_timeouts = False
        env.success_path = "episode_final.all_goals_hit"
        env.num_envs = 2
        env._episode_returns = torch.zeros(2)
        env._episode_lengths = torch.zeros(2)
        env._episode_cumulative = {}

        observations, rewards, terminated, truncated, infos = env.step(
            np.array([[2.0, -2.0], [0.25, -0.25]], dtype=np.float32)
        )

        np.testing.assert_allclose(env.envs.last_actions, [[1.0, -1.0], [0.25, -0.25]])
        self.assertEqual(observations.shape, (2, 3))
        np.testing.assert_allclose(rewards, [1.0, 2.0])
        np.testing.assert_array_equal(terminated, [True, True])
        np.testing.assert_array_equal(truncated, [False, False])
        np.testing.assert_array_equal(infos["time_outs"], [True, False])
        self.assertNotIn("final_obs", infos)
        self.assertIn("kept_by_adapter", infos)
        np.testing.assert_array_equal(infos["success"], [1.0, 0.0])
        self.assertEqual(infos["episode_info"]["episode/return"], (1.5, 2))
        self.assertEqual(infos["episode_info"]["episode/cumulative/bonus"], (0.75, 2))
        self.assertEqual(infos["episode_info"]["episode/final/successes"], (3.5, 2))

    def test_timeout_bootstrap_preserves_masks_and_combines_final_obs(self) -> None:
        env = object.__new__(IsaacLabVectorEnv)
        env.envs = _FinalObsIsaacEnv()
        env.device = "cpu"
        env.action_bounds = 1.0
        env.asymmetric_obs = True
        env.to_numpy = True
        env.bootstrap_timeouts = True
        env.success_path = None
        env.num_envs = 2
        env._episode_returns = torch.zeros(2)
        env._episode_lengths = torch.zeros(2)
        env._episode_cumulative = {}

        _, _, terminated, truncated, infos = env.step(np.zeros((2, 2), dtype=np.float32))

        np.testing.assert_array_equal(terminated, [False, True])
        np.testing.assert_array_equal(truncated, [True, False])
        np.testing.assert_allclose(infos["final_obs"], [[9.0, 8.0, 5.0], [7.0, 6.0, 4.0]])

    def test_timeout_bootstrap_rejects_missing_final_obs(self) -> None:
        env = object.__new__(IsaacLabVectorEnv)
        env.envs = _FakeIsaacEnv()
        env.device = "cpu"
        env.action_bounds = 1.0
        env.asymmetric_obs = True
        env.to_numpy = True
        env.bootstrap_timeouts = True
        env.success_path = None
        env.num_envs = 2
        env._episode_returns = torch.zeros(2)
        env._episode_lengths = torch.zeros(2)
        env._episode_cumulative = {}

        with self.assertRaisesRegex(RuntimeError, "requires final_obs"):
            env.step(np.zeros((2, 2), dtype=np.float32))

    def test_timeout_bootstrap_allows_missing_final_obs_without_timeout(self) -> None:
        env = object.__new__(IsaacLabVectorEnv)
        env.envs = _FakeIsaacEnv()
        env.envs.step = lambda actions: (
            {"policy": torch.zeros((2, 2)), "critic": torch.zeros((2, 1))},
            torch.zeros(2),
            torch.zeros(2, dtype=torch.bool),
            torch.zeros(2, dtype=torch.bool),
            {},
        )
        env.device = "cpu"
        env.action_bounds = 1.0
        env.asymmetric_obs = True
        env.to_numpy = True
        env.bootstrap_timeouts = True
        env.success_path = None
        env.num_envs = 2
        env._episode_returns = torch.zeros(2)
        env._episode_lengths = torch.zeros(2)
        env._episode_cumulative = {}

        _, _, terminated, truncated, infos = env.step(np.zeros((2, 2), dtype=np.float32))

        np.testing.assert_array_equal(terminated, [False, False])
        np.testing.assert_array_equal(truncated, [False, False])
        self.assertNotIn("final_obs", infos)


class _Agent:
    def sample_actions(self, interaction_step, prev_transition, training):
        return np.zeros((2, 1), dtype=np.float32)


class _AsyncDoneEnv:
    num_envs = 2

    def __init__(self) -> None:
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        return np.zeros((2, 1), dtype=np.float32), {}

    def step(self, actions):
        self.step_count += 1
        observations = np.zeros((2, 1), dtype=np.float32)
        rewards = np.ones(2, dtype=np.float32)
        if self.step_count == 1:
            terminated = np.array([True, False])
            success = np.array([1.0, 0.0])
        else:
            terminated = np.array([False, True])
            success = np.array([0.0, 1.0])
        truncated = np.zeros(2, dtype=bool)
        return observations, rewards, terminated, truncated, {"final_info": {"success": success}}


class EvaluationTest(unittest.TestCase):
    def test_success_end_is_not_overwritten_after_an_env_finishes(self) -> None:
        result = evaluate(_Agent(), _AsyncDoneEnv(), num_episodes=2, env_type="test")
        self.assertEqual(result["avg_success_end"], 1.0)

    def test_episode_subbatches_are_weighted_by_done_count(self) -> None:
        logger = object.__new__(TensorboardTrainerLogger)
        logger.average_meter_dict = AverageMeterDict()
        logger.media_dict = {}
        logger.update_metric(success=(0.5, 2))
        logger.update_metric(success=(1.0, 1))
        self.assertAlmostEqual(logger.average_meter_dict.averages()["success"], 2.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
