"""Uniform replay with on-demand, episode-safe actor observation windows."""

from operator import index
from typing import cast

import gymnasium as gym
import torch

from flash_rl.buffers.base_buffer import Batch
from flash_rl.buffers.torch_buffer import TorchUniformBuffer
from flash_rl.types import NDArray


class TorchHistoryBuffer(TorchUniformBuffer):
    """Store single transitions, gather [B,H,actor_dim+1] windows at sampling.

    Vector batches must always contain the same environments in the same order.
    The final window column is a 0/1 validity mask; genuine episode beginnings
    are left-padded with zeros. ``next_observation`` MUST be the pre-auto-reset
    final observation for both terminated and truncated transitions.

    Once full, uniformly sample all but the oldest (H-1)*num_envs slots. This
    conservative exclusion also rejects some valid short episodes, but never
    disguises overwritten history as episode-start padding. Explicit indices
    use physical ring slots and obey the same exclusion. Capacity need not be
    divisible by num_envs. Replay save/load is deliberately unsupported.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space[NDArray],
        action_space: gym.spaces.Space[NDArray],
        n_step: int,
        gamma: float,
        max_length: int,
        min_length: int,
        sample_batch_size: int,
        device_type: str,
        actor_observation_dim: int,
        history_length: int,
    ):
        self._actor_observation_dim = index(actor_observation_dim)
        self._history_length = index(history_length)
        shape = observation_space.shape
        if n_step != 1:
            raise ValueError("TorchHistoryBuffer only supports n_step=1")
        if self._history_length < 2:
            raise ValueError("history_length must be at least 2")
        if shape is None or len(shape) not in (1, 2) or not 0 < self._actor_observation_dim <= shape[-1]:
            raise ValueError("Expected flat/vector observation space and a valid actor_observation_dim")
        self._configured_num_envs = shape[0] if len(shape) == 2 else None
        if max_length < self._history_length * (self._configured_num_envs or 1):
            raise ValueError("max_length must be at least history_length * num_envs")
        super().__init__(
            observation_space, action_space, n_step, gamma, max_length, min_length, sample_batch_size, device_type
        )

    def reset(self) -> None:
        super().reset()
        self._total_written = 0
        self._num_envs = self._configured_num_envs
        self._write_ids = torch.full((self._max_length,), -1, dtype=torch.long, device=self._device)
        self._episode_ids = torch.full_like(self._write_ids, -1)
        self._env_episode_ids = torch.zeros(self._num_envs or 0, dtype=torch.long, device=self._device)
        self._history_lags = torch.arange(self._history_length - 1, -1, -1, device=self._device)

    def add(self, transition: Batch) -> None:
        shape = transition["observation"].shape
        if len(shape) != 2 or shape[1] != self._observations.shape[1] or shape[0] < 1:
            raise ValueError("observation must have shape [num_envs, observation_dim]")
        num_envs = shape[0]
        if self._num_envs is not None and num_envs != self._num_envs:
            raise ValueError("Vector batch size/environment ordering must stay fixed")
        if self._max_length < self._history_length * num_envs:
            raise ValueError("max_length must be at least history_length * num_envs")
        expected = {
            "next_observation": shape,
            "action": (num_envs, self._actions.shape[1]),
            "reward": (num_envs,),
            "terminated": (num_envs,),
            "truncated": (num_envs,),
        }
        if any(tuple(transition[key].shape) != tuple(wanted) for key, wanted in expected.items()):
            raise ValueError("Transition fields must use the same fixed vector batch")
        if self._num_envs is None:
            self._num_envs = num_envs
            self._env_episode_ids = torch.zeros(num_envs, dtype=torch.long, device=self._device)
        write_ids = torch.arange(num_envs, device=self._device) + self._total_written
        slots = write_ids.remainder(self._max_length)
        super().add(transition)
        self._write_ids[slots] = write_ids
        self._episode_ids[slots] = self._env_episode_ids
        stored = self._n_step_transitions[-1]  # Already copied onto the buffer device by the parent.
        self._env_episode_ids += (stored["terminated"].bool() | stored["truncated"].bool()).long()
        self._total_written += num_envs

    def sample(self, sample_idxs: NDArray | torch.Tensor | None = None) -> Batch:
        if not self._num_in_buffer:
            raise ValueError("Cannot sample an empty history buffer")
        assert self._num_envs is not None  # Established by the first successful add().
        margin = (self._history_length - 1) * self._num_envs
        oldest = self._total_written - self._max_length + margin if self._num_in_buffer == self._max_length else 0
        if sample_idxs is None:
            # One uniform draw over eligible anchors, without data-dependent rejection.
            absolute = torch.randint(oldest, self._total_written, (self._sample_batch_size,), device=self._device)
            slots = absolute.remainder(self._max_length)
        else:
            slots = torch.as_tensor(sample_idxs, device=self._device)
            if slots.ndim != 1 or slots.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                raise ValueError("sample_idxs must be a one-dimensional integer array of physical slots")
            slots = slots.long()
            if bool(((slots < 0) | (slots >= self._num_in_buffer)).any()):
                raise ValueError("sample_idxs contain unwritten/out-of-range slots")
            absolute = self._write_ids[slots]
            if bool((absolute < oldest).any()):
                raise ValueError("Anchor excluded: its oldest history may have been overwritten")
        batch = super().sample(slots)
        previous_ids = absolute[:, None] - self._history_lags[None, :] * self._num_envs
        previous_slots = previous_ids.remainder(self._max_length)
        valid = (
            (previous_ids >= 0)
            & (self._write_ids[previous_slots] == previous_ids)
            & (self._episode_ids[previous_slots] == self._episode_ids[slots, None])
        )
        history = torch.zeros(
            (len(slots), self._history_length, self._actor_observation_dim + 1),
            dtype=self._observations.dtype,
            device=self._device,
        )
        history[..., :-1] = torch.where(
            valid[..., None], self._observations[previous_slots, : self._actor_observation_dim], 0
        )
        history[..., -1] = valid
        final = torch.ones((len(slots), 1, self._actor_observation_dim + 1), dtype=history.dtype, device=self._device)
        final[:, 0, :-1] = cast(torch.Tensor, batch["next_observation"])[:, : self._actor_observation_dim]
        batch["actor_history"] = history
        batch["actor_next_history"] = torch.cat((history[:, 1:], final), dim=1)
        return batch

    def save(self, path: str) -> None:
        raise NotImplementedError("History replay save/load must also preserve sequence and episode IDs")

    def load(self, path: str) -> None:
        raise NotImplementedError("History replay save/load must also preserve sequence and episode IDs")
