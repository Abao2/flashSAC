import gymnasium as gym
import numpy as np
import pytest
import torch

from flash_rl.agents.flashSAC.update import _compute_categorical_td_target
from flash_rl.buffers.numpy_buffer import NpyUniformBuffer
from flash_rl.buffers.torch_buffer import TorchUniformBuffer


@pytest.mark.parametrize("buffer_type", ["numpy", "torch"])
def test_n_step_discount_uses_effective_horizon(buffer_type: str) -> None:
    kwargs = dict(
        observation_space=gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32),
        action_space=gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32),
        n_step=3,
        gamma=0.5,
        max_length=16,
        min_length=1,
        sample_batch_size=3,
    )
    buffer = NpyUniformBuffer(**kwargs) if buffer_type == "numpy" else TorchUniformBuffer(**kwargs, device_type="cpu")

    # env 0 truncates after one step, env 1 runs all three, env 2 terminates after two.
    for step in range(3):
        buffer.add(
            {
                "observation": np.full((3, 1), step, dtype=np.float32),
                "action": np.zeros((3, 1), dtype=np.float32),
                "reward": np.ones(3, dtype=np.float32),
                "terminated": np.array([False, False, step == 1]),
                "truncated": np.array([step == 0, False, False]),
                "next_observation": np.full((3, 1), step + 1, dtype=np.float32),
            }
        )

    batch = buffer.sample(np.arange(3))
    as_numpy = lambda value: value.cpu().numpy() if isinstance(value, torch.Tensor) else value
    np.testing.assert_allclose(as_numpy(batch["reward"]), [1.0, 1.75, 1.5])
    np.testing.assert_allclose(as_numpy(batch["discount"]), [0.5, 0.125, 0.25])
    np.testing.assert_array_equal(as_numpy(batch["terminated"]).astype(bool), [False, False, True])
    np.testing.assert_array_equal(as_numpy(batch["truncated"]).astype(bool), [True, False, False])


@pytest.mark.parametrize("buffer_type", ["numpy", "torch"])
def test_one_step_discount_is_gamma(buffer_type: str) -> None:
    kwargs = dict(
        observation_space=gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32),
        action_space=gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32),
        n_step=1,
        gamma=0.9,
        max_length=2,
        min_length=1,
        sample_batch_size=1,
    )
    buffer = NpyUniformBuffer(**kwargs) if buffer_type == "numpy" else TorchUniformBuffer(**kwargs, device_type="cpu")
    buffer.add(
        {
            "observation": np.zeros((1, 1), dtype=np.float32),
            "action": np.zeros((1, 1), dtype=np.float32),
            "reward": np.ones(1, dtype=np.float32),
            "terminated": np.zeros(1, dtype=bool),
            "truncated": np.zeros(1, dtype=bool),
            "next_observation": np.ones((1, 1), dtype=np.float32),
        }
    )
    discount = buffer.sample(np.array([0]))["discount"]
    if isinstance(discount, torch.Tensor):
        discount = discount.cpu().numpy()
    np.testing.assert_allclose(discount, [0.9])


def test_categorical_target_accepts_per_sample_discount() -> None:
    target = _compute_categorical_td_target(
        target_log_probs=torch.tensor([[-torch.inf, -torch.inf, 0.0], [-torch.inf, -torch.inf, 0.0]]),
        reward=torch.zeros(2),
        done=torch.zeros(2),
        actor_entropy=torch.zeros(2),
        gamma=torch.tensor([0.5, 0.25]),
        num_bins=3,
        min_v=0.0,
        max_v=2.0,
    )
    torch.testing.assert_close(target, torch.tensor([[0.0, 1.0, 0.0], [0.5, 0.5, 0.0]]))
