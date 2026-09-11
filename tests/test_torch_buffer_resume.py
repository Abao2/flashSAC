"""Tiny CPU-only replay resume checks, including the GPU load-location regression."""
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import pytest
import torch

from flash_rl.buffers.torch_buffer import TorchUniformBuffer


def make_buffer():
    with patch("torch.cuda.is_available", return_value=False):
        return TorchUniformBuffer(
            observation_space=gym.spaces.Box(-100, 100, shape=(2, 3), dtype=np.float32),
            action_space=gym.spaces.Box(-1, 1, shape=(2, 2), dtype=np.float32),
            n_step=2, gamma=.9, max_length=7, min_length=1,
            sample_batch_size=2, device_type="cpu",
        )


@pytest.mark.parametrize("steps", [4, 6])  # Partially filled and wrapped ring.
@pytest.mark.parametrize("legacy_without_discount", [False, True])
def test_resume_stages_on_cpu_and_preserves_ring(tmp_path, monkeypatch, steps, legacy_without_discount):
    source = make_buffer()
    for step in range(steps):
        obs = torch.arange(6, dtype=torch.float32).reshape(2, 3) + step * 10
        transition = dict(
            observation=obs, next_observation=obs + .5, action=obs[:, :2] / 100,
            reward=torch.tensor([step, step + .25]),
            terminated=torch.tensor([step == 1, False]),
            truncated=torch.tensor([False, step == 2]),
        )
        source.add(transition)

    path = str(tmp_path / "replay_buffer.pt")
    source.save(path)
    indices = np.arange(len(source))
    expected = source.sample(indices)
    if legacy_without_discount:
        dataset = torch.load(path, map_location="cpu")
        del dataset["discount"]
        torch.save(dataset, path)
        expected["discount"] = torch.full((len(source),), .9 ** 2)

    restored = make_buffer()
    restored.add(transition)
    assert restored._n_step_transitions  # In-flight state must be discarded on load.
    original_storage = restored._observations.data_ptr()
    # A fake configured CUDA device exercises map_location without allocating CUDA.
    restored._device = torch.device("cuda:0")
    real_load = torch.load
    load_locations = []

    def cpu_only_load(*args, **kwargs):
        assert kwargs.get("map_location") == "cpu"  # Fail before any CUDA allocation.
        load_locations.append(kwargs["map_location"])
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", cpu_only_load)
    restored.load(path)
    restored._device = torch.device("cpu")

    assert load_locations == ["cpu"]
    assert len(restored) == min((steps - 1) * 2, 7)
    assert restored._current_idx == source._current_idx == ((steps - 1) * 2) % 7
    assert restored._observations.data_ptr() == original_storage
    assert not restored._n_step_transitions
    actual = restored.sample(indices)
    assert actual.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
