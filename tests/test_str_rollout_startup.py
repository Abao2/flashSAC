"""CPU-only startup gate checks; no Isaac application or GPU initialization."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gymnasium as gym
import numpy as np
import torch

ROOT = Path(__file__).parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from diagnose_str_rollouts import numerical_settings, parse_args, rollout_action
from flash_rl.envs.isaaclab import IsaacLabVectorEnv


class StartupTest(unittest.TestCase):
    def test_first_random_action_only_does_not_advance_policy(self):
        space = gym.spaces.Box(-1, 1, (2, 29), dtype=np.float32)
        space.seed(7)
        expected = space.sample()
        space.seed(7)
        env = SimpleNamespace(action_space=space)
        obs = np.zeros((2, 324), dtype=np.float32)
        policy = SimpleNamespace(actions=Mock(return_value=np.ones((2, 29))))
        np.testing.assert_array_equal(rollout_action(policy, env, obs, 0, True), expected)
        policy.actions.assert_not_called()
        rollout_action(policy, env, obs, 1, True)
        rollout_action(policy, env, obs, 0, False)
        self.assertEqual(policy.actions.call_count, 2)
        self.assertEqual(policy.actions.call_args_list[0].args[1], 1)
        self.assertEqual(policy.actions.call_args_list[1].args[1], 0)

    def test_existing_adapter_reset_returns_pre_randomization_observation(self):
        raw = SimpleNamespace(episode_length_buf=torch.zeros(2, dtype=torch.int64))

        def reset():
            raw.episode_length_buf.zero_()
            return {"policy": torch.zeros((2, 162)), "critic": torch.zeros((2, 162))}, {}

        fake = SimpleNamespace(envs=SimpleNamespace(reset=reset, unwrapped=raw),
            _reset_episode_statistics=lambda: None, asymmetric_obs=True, to_numpy=True,
            max_episode_steps=600, obs_size=(162,), critic_obs_size=(162,))
        with patch("torch.randint_like", return_value=torch.tensor([3, 597])) as draw:
            obs, _ = IsaacLabVectorEnv.reset(fake, random_start_init=True)
            np.testing.assert_array_equal(obs, 0)
            self.assertEqual(raw.episode_length_buf.tolist(), [3, 597])
            draw.assert_called_once()
            IsaacLabVectorEnv.reset(fake, random_start_init=False)
            self.assertEqual(raw.episode_length_buf.tolist(), [0, 0])
            draw.assert_called_once()

    def test_cli_default_preserved_and_gate_explicit(self):
        base = ["rollout", "--checkpoint", "/unused", "--output-dir", "/tmp/not-created-startup-test"]
        with patch.object(sys, "argv", base):
            self.assertFalse(parse_args().training_startup)
            self.assertFalse(parse_args().training_numerics)
        with patch.object(sys, "argv", base + ["--training-startup"]):
            self.assertTrue(parse_args().training_startup)
        with patch.object(sys, "argv", base + ["--training-numerics"]):
            self.assertTrue(parse_args().training_numerics)

    def test_numerics_bundle_only_changes_explicit_opt_in(self):
        fake = SimpleNamespace(backends=SimpleNamespace(
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
            cudnn=SimpleNamespace(allow_tf32=False, benchmark=False)),
            set_float32_matmul_precision=Mock(),
            get_float32_matmul_precision=Mock(return_value="highest"))
        before = numerical_settings(fake)
        self.assertEqual(before, {"cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False,
            "cudnn_benchmark": False, "float32_matmul_precision": "highest"})
        fake.set_float32_matmul_precision.assert_not_called()
        fake.set_float32_matmul_precision.side_effect = lambda value: setattr(
            fake.get_float32_matmul_precision, "return_value", value)
        after = numerical_settings(fake, enable=True)
        self.assertEqual(after, {"cuda_matmul_allow_tf32": True, "cudnn_allow_tf32": True,
            "cudnn_benchmark": True, "float32_matmul_precision": "high"})
        fake.set_float32_matmul_precision.assert_called_once_with("high")
        self.assertEqual(numerical_settings(fake), after)
        fake.set_float32_matmul_precision.assert_called_once_with("high")


if __name__ == "__main__":
    unittest.main()
