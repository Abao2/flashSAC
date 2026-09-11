"""Finite-window recurrent actor; this is not an episode-long recurrent policy."""

import torch
import torch.nn as nn

from flash_rl.agents.flashSAC.network import FlashSACActor


class FlashSACHistoryActor(nn.Module):
    """Encode raw history, then retain the existing FlashSAC actor backbone.

    Inputs are [batch, history_length, input_dim + 1], oldest to newest.
    The last feature is a binary valid mask: a left-padded suffix of valid
    frames, including the final/current frame. Padding values are ignored.

    Every call recomputes the window from zero hidden/cell state, with gradients
    through all valid frames. No hidden state is cached in the actor or replay.
    A failed finite-window experiment cannot rule out longer-term memory.
    """

    def __init__(
        self,
        num_blocks: int,
        input_dim: int,
        hidden_dim: int,
        action_dim: int,
        history_length: int = 32,
        lstm_hidden_dim: int = 128,
    ):
        super().__init__()
        if min(input_dim, hidden_dim, action_dim, history_length, lstm_hidden_dim) < 1:
            raise ValueError("Actor dimensions and history_length must be positive")
        self.input_dim = input_dim
        self.history_length = history_length
        self.lstm_hidden_dim = lstm_hidden_dim
        self.frame_norm = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, num_layers=1, batch_first=True)
        self.head = FlashSACActor(num_blocks, input_dim + lstm_hidden_dim, hidden_dim, action_dim)

    def encode_history(self, observations: torch.Tensor) -> torch.Tensor:
        """Return [batch, lstm_hidden_dim] without consuming padded timesteps."""
        if observations.ndim != 3 or observations.shape[1:] != (self.history_length, self.input_dim + 1):
            raise ValueError("Expected [batch, history_length, input_dim + 1] observations")
        mask = observations[..., -1]
        valid = mask == 1
        # Async assertions avoid a host-device synchronization on each CUDA call.
        torch._assert_async(((mask == 0) | valid).all(), "History mask must contain only 0 or 1")
        torch._assert_async(valid[:, -1].all(), "The current history frame must be valid")
        torch._assert_async((~valid[:, :-1] | valid[:, 1:]).all(), "Valid history must be a contiguous suffix")
        lengths = valid.sum(dim=1)
        positions = torch.arange(self.history_length, device=observations.device)
        # Move the valid suffix to the left; no pack/padded hidden-state advance.
        indices = (positions[None, :] + self.history_length - lengths[:, None]).clamp_max(self.history_length - 1)
        frames = observations[..., :self.input_dim].gather(1, indices[..., None].expand(-1, -1, self.input_dim))
        frames = torch.where((positions[None, :] < lengths[:, None])[..., None], frames, 0.)
        sequence, _ = self.lstm(self.frame_norm(frames))  # Omitted state means zeros on every call.
        memory = sequence.gather(1, (lengths - 1)[:, None, None].expand(-1, 1, self.lstm_hidden_dim))
        return memory[:, 0, :]

    def _actor_input(self, observations: torch.Tensor) -> torch.Tensor:
        memory = self.encode_history(observations)
        # Only B current embeddings enter the original head's BatchNorm, never B*H padding.
        return torch.cat((observations[:, -1, :self.input_dim], memory), dim=-1)

    def get_mean_and_std(
        self, observations: torch.Tensor, training: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.head.get_mean_and_std(self._actor_input(observations), training)

    def forward(
        self, observations: torch.Tensor, training: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.head(self._actor_input(observations), training)
