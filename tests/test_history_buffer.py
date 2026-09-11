"""CPU-only sequence/ring tests; no simulator, optimizer, or GPU allocation."""
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import pytest
import torch

from flash_rl.buffers.history_buffer import TorchHistoryBuffer
from flash_rl.buffers.torch_buffer import TorchUniformBuffer


@pytest.fixture(scope="module", autouse=True)
def two_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def make_buffer(num_envs=2, history=3, capacity=11, **kwargs):
    options = dict(observation_space=gym.spaces.Box(-10000, 10000, shape=(num_envs, 4), dtype=np.float32),
                   action_space=gym.spaces.Box(-1, 1, shape=(num_envs, 2), dtype=np.float32),
                   n_step=1, gamma=.99, max_length=capacity, min_length=1,
                   sample_batch_size=32, device_type="cpu", actor_observation_dim=2, history_length=history)
    options.update(kwargs)
    # Parent buffer may pin CPU memory on CUDA-capable hosts; tests need no CUDA runtime.
    with patch("torch.cuda.is_available", return_value=False):
        return TorchHistoryBuffer(**options)


def transition(step, num_envs=2, terminated=(), truncated=()):
    obs = torch.arange(num_envs * 4, dtype=torch.float32).reshape(num_envs, 4) + step * 100
    term, trunc = torch.zeros(num_envs), torch.zeros(num_envs)
    term[list(terminated)] = 1
    trunc[list(truncated)] = 1
    return dict(observation=obs, next_observation=obs + .5, action=obs[:, :2] / 1000,
                reward=torch.arange(num_envs, dtype=torch.float32) + step, terminated=term, truncated=trunc)


def test_partial_padding_order_and_unmodified_uniform_transition_batch():
    buf = make_buffer()
    for step in range(2):
        tr = transition(step)
        buf.add({key: value.numpy() for key, value in tr.items()} if step == 0 else tr)
    indices = np.array([3, 0, 2, 3])  # Random anchor ordering and duplicates.
    batch = buf.sample(indices)
    history = batch["actor_history"]
    assert history.shape == (4, 3, 3)
    assert history[0].tolist() == [[0, 0, 0], [4, 5, 1], [104, 105, 1]]
    assert history[1].tolist() == [[0, 0, 0], [0, 0, 0], [0, 1, 1]]
    assert batch["actor_next_history"][0].tolist() == [[4, 5, 1], [104, 105, 1], [104.5, 105.5, 1]]
    original = TorchUniformBuffer.sample(buf, indices)
    assert set(batch) == set(original) | {"actor_history", "actor_next_history"}
    for key in original:
        torch.testing.assert_close(batch[key], original[key], rtol=0, atol=0)
    torch.manual_seed(4)
    uniform = TorchUniformBuffer.sample(buf)
    torch.manual_seed(4)
    history_uniform = buf.sample()
    for key in uniform:
        torch.testing.assert_close(history_uniform[key], uniform[key], rtol=0, atol=0)
    assert buf._observations.shape == (11, 4) and buf._next_observations.shape == (11, 4)
    assert buf._write_ids.shape == buf._episode_ids.shape == (11,)


def test_terminal_and_timeout_keep_final_next_frame_but_cut_next_episode():
    buf = make_buffer(capacity=9)
    buf.add(transition(0))
    ending = transition(1, terminated=(0,), truncated=(1,))
    ending["next_observation"] = torch.tensor([[700., 701., 702., 703.], [800., 801., 802., 803.]])
    buf.add(ending)
    buf.add(transition(9))  # Auto-reset observations, deliberately far from final_obs.
    terminal = buf.sample(np.array([2, 3]))
    assert terminal["actor_next_history"][:, -1].tolist() == [[700, 701, 1], [800, 801, 1]]
    assert terminal["actor_next_history"][:, :-1].tolist() == [
        [[0, 1, 1], [100, 101, 1]], [[4, 5, 1], [104, 105, 1]]]
    assert terminal["terminated"].tolist() == [1, 0]
    assert terminal["truncated"].tolist() == [0, 1]
    torch.testing.assert_close(terminal["discount"], torch.full((2,), .99))
    reset = buf.sample(np.array([4, 5]))
    assert not reset["actor_history"][:, :-1].any()
    assert reset["actor_history"][:, -1].tolist() == [[900, 901, 1], [904, 905, 1]]
    assert reset["actor_next_history"][:, -1].tolist() == [[900.5, 901.5, 1], [904.5, 905.5, 1]]


def test_nondivisible_ring_random_order_matches_independent_episode_reference():
    n, h, capacity = 3, 4, 13
    buf = make_buffer(num_envs=n, history=h, capacity=capacity)
    reference, episodes = {}, [0] * n
    for step in range(12):
        tr = transition(step, n, terminated=(0,) if step in (1, 7) else (),
                        truncated=(1,) if step in (3, 8) else ())
        for env in range(n):
            reference[step * n + env] = (episodes[env], tr["observation"][env].clone(), tr["next_observation"][env].clone())
            episodes[env] += int(tr["terminated"][env] or tr["truncated"][env])
        buf.add(tr)
        total = (step + 1) * n
        low = total - capacity + (h - 1) * n if total >= capacity else 0
        anchors = np.random.default_rng(step).permutation(np.arange(low, total))
        batch = buf.sample(anchors % capacity)
        for row, absolute in enumerate(anchors.tolist()):
            expected = torch.zeros(h, 3)
            for slot, age in enumerate(range(h - 1, -1, -1)):
                previous = absolute - age * n
                if previous >= 0 and reference[previous][0] == reference[absolute][0]:
                    expected[slot, :2], expected[slot, 2] = reference[previous][1][:2], 1
            torch.testing.assert_close(batch["actor_history"][row], expected, rtol=0, atol=0)
            torch.testing.assert_close(batch["actor_next_history"][row, :-1], expected[1:], rtol=0, atol=0)
            torch.testing.assert_close(batch["actor_next_history"][row, -1, :2], reference[absolute][2][:2])
        if total >= capacity:
            for absolute in range(total - capacity, low):
                with pytest.raises(ValueError, match="overwritten"):
                    buf.sample(np.array([absolute % capacity]))
    assert buf._current_idx == 36 % 13 and len(buf) == 13


def test_default_sampling_is_one_uniform_draw_over_only_eligible_anchors():
    buf = make_buffer(capacity=7, sample_batch_size=1000)
    for step in range(5):
        buf.add(transition(step))
    # Writes 0..9, retained 3..9, exclude oldest 4 => eligible absolute IDs 7..9.
    torch.manual_seed(91)
    expected_ids = torch.randint(7, 10, (1000,))
    torch.manual_seed(91)
    batch = buf.sample()
    torch.testing.assert_close(batch["observation"], buf._observations[expected_ids % 7], rtol=0, atol=0)
    assert batch["actor_history"][..., -1].bool().all()
    counts = torch.bincount(expected_ids - 7, minlength=3)
    assert (counts > 250).all() and (counts < 400).all()


def test_reset_clears_all_history_metadata_and_new_episode_padding():
    buf = make_buffer()
    buf.add(transition(0, terminated=(0,), truncated=(1,)))
    buf.add(transition(1))
    with patch("torch.cuda.is_available", return_value=False):
        buf.reset()
    assert len(buf) == buf._total_written == buf._current_idx == 0
    assert (buf._write_ids == -1).all() and (buf._episode_ids == -1).all()
    assert not buf._env_episode_ids.any()
    with pytest.raises(ValueError, match="empty"):
        buf.sample()
    buf.add(transition(20))
    assert not buf.sample(np.array([0]))["actor_history"][:, :-1].any()


def test_first_batch_infers_fixed_num_envs_for_unbatched_space():
    space = gym.spaces.Box(-10000, 10000, shape=(4,), dtype=np.float32)
    buf = make_buffer(observation_space=space)
    assert buf._num_envs is None
    buf.add(transition(0, 3))
    assert buf._num_envs == 3
    with pytest.raises(ValueError, match="fixed"):
        buf.add(transition(1, 2))
    assert len(buf) == 3 and buf._total_written == 3
    small = make_buffer(observation_space=space, capacity=6)
    with pytest.raises(ValueError, match="history_length"):
        small.add(transition(0, 3))
    assert len(small) == 0


@pytest.mark.parametrize("options,match", [({"n_step": 2}, "n_step"), ({"history_length": 1}, "at least 2"),
    ({"actor_observation_dim": 5}, "actor_observation_dim"), ({"actor_observation_dim": 0}, "actor_observation_dim"),
    ({"capacity": 5}, "history_length")])
def test_constructor_guards(options, match):
    with pytest.raises(ValueError, match=match):
        make_buffer(**options)


def test_invalid_transition_or_anchor_cannot_mutate_history():
    buf = make_buffer()
    wrong = transition(0)
    wrong["truncated"] = torch.zeros(3)
    with pytest.raises(ValueError, match="fields"):
        buf.add(wrong)
    assert len(buf) == buf._total_written == 0
    buf.add(transition(0))
    for ids in (np.array([-1]), np.array([2]), np.array([[0]]), np.array([.5]), np.array([True])):
        with pytest.raises(ValueError):
            buf.sample(ids)


def test_replay_persistence_is_explicitly_unsupported(tmp_path):
    buf = make_buffer()
    for operation in (buf.save, buf.load):
        with pytest.raises(NotImplementedError, match="sequence and episode IDs"):
            operation(str(tmp_path / "replay.pt"))
    assert not (tmp_path / "replay.pt").exists()
