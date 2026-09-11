"""CPU Bellman/projection oracle, independent of upstream/adapter equivalence.

Run: python -m unittest discover -s tests -p test_str_td_target_oracle.py -v
This tests actual replay aggregation and categorical targets, not simulator
termination classification, observation construction, or CUDA/AMP kernels.
"""

import bisect
import math
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch

from flash_rl.agents.flashSAC.update import _compute_categorical_td_target
from flash_rl.buffers.torch_buffer import TorchUniformBuffer


def scalar_oracle(probabilities, support, rewards, gamma, terminated, alpha_log_pi):
    """Forward sum raw rewards; distribute each Bellman atom between neighbors.

    No production buffer/target helper, tensor scatter, or index formula is used.
    Truncation ends the supplied reward list, but only termination cancels value.
    """
    accumulated = sum(gamma**i * reward for i, reward in enumerate(rewards))
    bootstrap = 0.0 if terminated else gamma ** len(rewards)
    result = [0.0] * len(support)
    for probability, value in zip(probabilities, support):
        target = accumulated + bootstrap * (value - alpha_log_pi)
        if target <= support[0]:
            result[0] += probability
        elif target >= support[-1]:
            result[-1] += probability
        else:
            right = bisect.bisect_right(support, target)
            left = right - 1
            interval = support[right] - support[left]
            result[left] += probability * (support[right] - target) / interval
            result[right] += probability * (target - support[left]) / interval
    return torch.tensor(result, dtype=torch.float32)


class TestStrTdTargetOracle(unittest.TestCase):
    @staticmethod
    def actual(probabilities, rewards, terminated, discount, alpha_log_pi, support):
        # Exercise the real function while explicitly avoiding compilation/GPU.
        with torch.compiler.set_stance("force_eager"):
            return _compute_categorical_td_target(
                target_log_probs=torch.tensor(probabilities, dtype=torch.float32).log(),
                reward=torch.as_tensor(rewards, dtype=torch.float32),
                done=torch.as_tensor(terminated, dtype=torch.float32),
                actor_entropy=torch.as_tensor(alpha_log_pi, dtype=torch.float32),
                gamma=discount,
                num_bins=len(support), min_v=support[0], max_v=support[-1],
            )

    def assert_distribution(self, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=0, atol=3e-6)
        torch.testing.assert_close(actual.sum(-1), torch.ones(actual.shape[:-1]), rtol=0, atol=2e-6)
        self.assertTrue(bool(torch.isfinite(actual).all()))
        self.assertTrue(bool((actual >= 0).all()))

    def test_real_replay_n3_all_boundary_horizons_and_negative_control(self):
        gamma = .5
        rewards = [.25, .5, -.125]
        # (effective horizon, terminated, truncated); includes both flags at once.
        cases = [(k, terminal, timeout) for k in (1, 2, 3)
                 for terminal, timeout in ((True, False), (False, True), (True, True))]
        cases.append((3, False, False))
        count = len(cases)
        with patch("torch.cuda.is_available", return_value=False):
            buffer = TorchUniformBuffer(
                observation_space=gym.spaces.Box(-1000, 1000, shape=(1,), dtype=np.float32),
                action_space=gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32),
                n_step=3, gamma=gamma, max_length=32, min_length=1,
                sample_batch_size=count, device_type="cpu",
            )
        for step in range(3):
            buffer.add(dict(
                observation=np.full((count, 1), step, dtype=np.float32),
                next_observation=np.full((count, 1), step + 1, dtype=np.float32),
                action=np.zeros((count, 1), dtype=np.float32),
                # Large new-episode rewards must not leak across an earlier boundary.
                reward=np.array([rewards[step] if step < k else 100
                                 for k, _, _ in cases], dtype=np.float32),
                terminated=np.array([terminal and step == k - 1 for k, terminal, _ in cases]),
                truncated=np.array([timeout and step == k - 1 for k, _, timeout in cases]),
            ))
        batch = buffer.sample(np.arange(count))
        torch.testing.assert_close(batch["discount"], torch.tensor([gamma**k for k, _, _ in cases]), rtol=0, atol=0)
        torch.testing.assert_close(batch["terminated"], torch.tensor([float(t) for _, t, _ in cases]), rtol=0, atol=0)
        torch.testing.assert_close(batch["truncated"], torch.tensor([float(t) for _, _, t in cases]), rtol=0, atol=0)
        support = [-5 + i / 10 for i in range(101)]  # Actual FlashSAC support.
        probabilities = [1 / 101] * 101
        entropy = .2 * math.log(.25)  # Code's actor_entropy means alpha * log pi.
        expected = torch.stack([
            scalar_oracle(probabilities, support, rewards[:k], gamma, terminal, entropy)
            for k, terminal, _ in cases
        ])
        actual = self.actual([probabilities] * count, batch["reward"], batch["terminated"],
                             batch["discount"], [entropy] * count, support)
        self.assert_distribution(actual, expected)

        # Negative control: emulate losing per-sample discount and falling back
        # to gamma**3. Only early bootstrapping timeouts must disagree.
        wrong = self.actual([probabilities] * count, batch["reward"], batch["terminated"],
                            gamma**3, [entropy] * count, support)
        for row, (k, terminal, timeout) in enumerate(cases):
            with self.subTest(horizon=k, terminated=terminal, truncated=timeout):
                if timeout and not terminal and k < 3:
                    with self.assertRaises(AssertionError):
                        torch.testing.assert_close(wrong[row], expected[row], rtol=0, atol=3e-6)
                    self.assertGreater(float((wrong[row] - expected[row]).abs().max()), .01)
                else:
                    torch.testing.assert_close(wrong[row], expected[row], rtol=0, atol=3e-6)

    def test_projection_saturation_exact_bins_and_probability_mass(self):
        support = [-2 + i / 2 for i in range(9)]
        probabilities = [.05, .1, .05, .15, .3, .05, .1, .05, .15]
        # Identity, all lower, all upper, terminal exact interior atom, and
        # non-bin-aligned projection that distributes mass to adjacent bins.
        cases = [(0, 1, False), (-100, .9, False), (100, .9, False),
                 (.5, .9, True), (.13, .73, False)]
        expected = torch.stack([scalar_oracle(probabilities, support, [reward], gamma, terminal, 0)
                                for reward, gamma, terminal in cases])
        actual = self.actual([probabilities] * len(cases), [x[0] for x in cases],
                             [x[2] for x in cases], torch.tensor([x[1] for x in cases]),
                             [0] * len(cases), support)
        self.assert_distribution(actual, expected)
        torch.testing.assert_close(actual[0], torch.tensor(probabilities), rtol=0, atol=1e-7)
        self.assertAlmostEqual(float(actual[1, 0]), 1, places=6)
        self.assertAlmostEqual(float(actual[2, -1]), 1, places=6)
        self.assertAlmostEqual(float(actual[3, 5]), 1, places=6)

    def test_entropy_sign_and_true_terminal_suppression(self):
        support = [-2 + i / 2 for i in range(9)]
        probabilities = [0, 0, 0, 0, 1, 0, 0, 0, 0]  # Next Q = zero exactly.
        # A continuous-action density can have either sign of log pi.
        entropy = [-.3, 0, .3, -.3, .3]
        terminal = [False, False, False, True, True]
        expected = torch.stack([scalar_oracle(probabilities, support, [.25], .5, t, e)
                                for t, e in zip(terminal, entropy)])
        actual = self.actual([probabilities] * 5, [.25] * 5, terminal, .5, entropy, support)
        self.assert_distribution(actual, expected)
        means = actual @ torch.tensor(support)
        torch.testing.assert_close(means, torch.tensor([.4, .25, .1, .25, .25]), rtol=0, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
