import math
import os
from dataclasses import dataclass, replace
from typing import Any, MutableMapping, Optional, cast

import gymnasium as gym
import torch
import torch.optim as optim
from torch.amp.grad_scaler import GradScaler

from flash_rl.agents.base_agent import BaseAgent
from flash_rl.agents.flashSAC.history_network import FlashSACHistoryActor
from flash_rl.agents.flashSAC.network import (
    FlashSACActor,
    FlashSACDoubleCritic,
    FlashSACTemperature,
)
from flash_rl.agents.flashSAC.update import (
    update_actor,
    update_critic,
    update_target_network,
    update_temperature,
)
from flash_rl.agents.utils.network import Network
from flash_rl.agents.utils.reward_normalization import RewardNormalizer
from flash_rl.agents.utils.scheduler import warmup_cosine_decay_scheduler
from flash_rl.buffers.torch_buffer import TorchUniformBuffer
from flash_rl.types import NDArray, Tensor


@dataclass
class FlashSACConfig:
    seed: int
    normalize_reward: bool
    normalized_G_max: float

    asymmetric_observation: bool
    device_type: str

    buffer_max_length: int
    buffer_min_length: int
    buffer_device_type: str
    sample_batch_size: int

    learning_rate_init: float
    learning_rate_peak: float
    learning_rate_end: float
    learning_rate_warmup_rate: float
    learning_rate_warmup_step: int
    learning_rate_decay_rate: float
    learning_rate_decay_step: int

    actor_num_blocks: int
    actor_hidden_dim: int
    actor_bc_alpha: float
    actor_noise_zeta_mu: float
    actor_noise_zeta_max: int
    actor_update_period: int

    critic_num_blocks: int
    critic_hidden_dim: int
    critic_num_bins: int
    critic_min_v: float
    critic_max_v: float
    critic_target_update_tau: float

    temp_initial_value: float
    temp_target_sigma: float
    temp_target_entropy: float

    gamma: float
    n_step: int

    use_compile: bool
    compile_mode: str
    use_amp: bool

    load_optimizer: bool
    load_reward_normalizer: bool

    # Opt-in finite-history diagnostic. Default preserves the native FF path.
    actor_history_length: int = 1
    actor_lstm_hidden_dim: int = 128


def _init_flashsac_networks(
    actor_observation_dim: int,
    critic_observation_dim: int,
    action_dim: int,
    cfg: FlashSACConfig,
    device: torch.device,
) -> tuple[Network, Network, Network, Network]:
    # Create learning rate schedule
    warmup_cosine_decay_lr = warmup_cosine_decay_scheduler(
        init_value=cfg.learning_rate_init,
        peak_value=cfg.learning_rate_peak,
        end_value=cfg.learning_rate_end,
        warmup_steps=cfg.learning_rate_warmup_step,
        decay_steps=cfg.learning_rate_decay_step,
    )

    # Initialize actor
    actor_class: type[FlashSACActor] | type[FlashSACHistoryActor] = FlashSACActor
    actor_kwargs: dict[str, Any] = {}
    if cfg.actor_history_length > 1:
        actor_class = FlashSACHistoryActor
        actor_kwargs = dict(history_length=cfg.actor_history_length, lstm_hidden_dim=cfg.actor_lstm_hidden_dim)
    actor_net = actor_class(
        num_blocks=cfg.actor_num_blocks,
        input_dim=actor_observation_dim,
        hidden_dim=cfg.actor_hidden_dim,
        action_dim=action_dim,
        **actor_kwargs,
    ).to(device)

    use_fused = device.type == "cuda" and torch.cuda.is_available()
    actor_optimizer = optim.Adam(actor_net.parameters(), lr=cfg.learning_rate_peak, fused=use_fused)
    actor_scheduler = torch.optim.lr_scheduler.LambdaLR(
        actor_optimizer,
        lr_lambda=lambda step: warmup_cosine_decay_lr(step) / cfg.learning_rate_peak,
    )
    actor = Network(
        network=actor_net,
        optimizer=actor_optimizer,
        scheduler=actor_scheduler,
        # cuDNN LSTM runs eagerly; leave the unchanged categorical critics compiled.
        compile_network=cfg.use_compile and cfg.actor_history_length == 1,
        compile_mode=cfg.compile_mode,
        use_weight_normalization=True,
    )
    # Manually compile `get_mean_and_std` function
    if cfg.use_compile and cfg.actor_history_length == 1:
        actor.network.get_mean_and_std = torch.compile(actor.network.get_mean_and_std, mode=cfg.compile_mode)  # type: ignore

    # Initialize critic
    critic_net = FlashSACDoubleCritic(
        num_blocks=cfg.critic_num_blocks,
        input_dim=critic_observation_dim + action_dim,
        hidden_dim=cfg.critic_hidden_dim,
        num_bins=cfg.critic_num_bins,
        min_v=cfg.critic_min_v,
        max_v=cfg.critic_max_v,
    ).to(device)

    critic_optimizer = optim.Adam(
        critic_net.parameters(),
        lr=cfg.learning_rate_peak,
        fused=use_fused,
    )
    critic_scheduler = torch.optim.lr_scheduler.LambdaLR(
        critic_optimizer,
        lr_lambda=lambda step: warmup_cosine_decay_lr(step) / cfg.learning_rate_peak,
    )
    critic = Network(
        network=critic_net,
        optimizer=critic_optimizer,
        scheduler=critic_scheduler,
        compile_network=cfg.use_compile,
        compile_mode=cfg.compile_mode,
        use_weight_normalization=True,
    )

    # Initialize target critic (same as critic but no optimizer)
    target_critic_net = FlashSACDoubleCritic(
        num_blocks=cfg.critic_num_blocks,
        input_dim=critic_observation_dim + action_dim,
        hidden_dim=cfg.critic_hidden_dim,
        num_bins=cfg.critic_num_bins,
        min_v=cfg.critic_min_v,
        max_v=cfg.critic_max_v,
    ).to(device)
    target_critic_net.load_state_dict(critic_net.state_dict())
    target_critic = Network(
        network=target_critic_net,
        optimizer=None,
        scheduler=None,
        compile_network=cfg.use_compile,
        compile_mode=cfg.compile_mode,
        use_weight_normalization=True,
        ema_source=critic,  # wire EMA update source
        ema_tau=cfg.critic_target_update_tau,
    )

    # Initialize temperature
    temp_net = FlashSACTemperature(cfg.temp_initial_value).to(device)
    temp_optimizer = optim.Adam(
        temp_net.parameters(),
        lr=cfg.learning_rate_peak,
        fused=use_fused,
    )
    temp_scheduler = torch.optim.lr_scheduler.LambdaLR(
        temp_optimizer,
        lr_lambda=lambda step: warmup_cosine_decay_lr(step) / cfg.learning_rate_peak,
    )
    temperature = Network(
        network=temp_net,
        optimizer=temp_optimizer,
        scheduler=temp_scheduler,
        compile_network=cfg.use_compile,
        compile_mode=cfg.compile_mode,
        use_weight_normalization=False,
    )

    # normalize network parameters after initialization
    actor.normalize_parameters()
    critic.normalize_parameters()
    target_critic.normalize_parameters()

    return actor, critic, target_critic, temperature


@torch.compile
def _build_truncated_zeta_cdf(mu: float, max_n: int) -> torch.Tensor:
    """
    Build the truncated zeta CDF for the given mu and max_n.
    """
    ns = torch.arange(1, max_n + 1, dtype=torch.float32)
    pmf = ns ** (-mu)
    pmf = pmf / torch.sum(pmf)
    cdf = torch.cumsum(pmf, dim=0)
    return cdf


@torch.compile
def _sample_integer_from_cdf(cdf: torch.Tensor) -> torch.Tensor:
    """
    Sample an integer from the given CDF.
    Returns a 0-d int32 tensor.
    """
    u = torch.rand((), device=cdf.device)
    idx = torch.argmax((u < cdf).to(torch.int32))
    return (idx + 1).to(torch.int32)


def _sample_flashsac_actions(
    actor: Network,
    noise: torch.Tensor,
    observations: torch.Tensor,
    temperature: float,
    cur_count: torch.Tensor,
    cur_n: torch.Tensor,
    zeta_cdf: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample actions with noise repeat logic fully in torch."""
    # forward actor → distribution mean and std
    mean, std = actor.apply(
        "get_mean_and_std",
        observations=observations,
        training=False,
    )
    # return deterministic actions without changing noise sampling params
    if temperature == 0.0:
        actions = torch.tanh(mean)
        return noise, actions, cur_count, cur_n

    # reinit noise after a certain number of steps (only during training)
    reinit = (cur_count == 0) | (cur_count >= cur_n)

    new_noise = torch.randn_like(mean)
    new_n = _sample_integer_from_cdf(zeta_cdf)

    noise = torch.where(reinit, new_noise, noise)
    cur_n = torch.where(reinit, new_n, cur_n)
    cur_count = torch.where(reinit, torch.zeros_like(cur_count), cur_count)

    # sample action
    actions = torch.tanh(mean + std * noise * temperature)

    return noise, actions, cur_count + 1, cur_n


def _update_networks(
    batch: dict[str, torch.Tensor],
    actor: Network,
    critic: Network,
    target_critic: Network,
    temperature: Network,
    cfg: FlashSACConfig,
    do_actor_update: bool,
    device: torch.device,
    grad_scaler: Optional[GradScaler],
) -> dict[str, torch.Tensor]:
    if do_actor_update:
        # Update actor
        actor_info = update_actor(
            actor=actor,
            critic=critic,
            temperature=temperature,
            batch=batch,  # type: ignore
            bc_alpha=cfg.actor_bc_alpha,
            device=device,
            use_amp=cfg.use_amp,
            grad_scaler=grad_scaler,
        )

        # Update temperature
        temperature_info = update_temperature(
            temperature=temperature,
            entropy=actor_info["actor/entropy"],
            target_entropy=cfg.temp_target_entropy,
        )
    else:
        actor_info = {}
        temperature_info = {}

    # Update critic
    critic_info = update_critic(
        actor=actor,  # updated
        critic=critic,
        target_critic=target_critic,
        temperature=temperature,  # updated
        batch=batch,  # type: ignore
        min_v=cfg.critic_min_v,
        max_v=cfg.critic_max_v,
        num_bins=cfg.critic_num_bins,
        gamma=cfg.gamma,
        n_step=cfg.n_step,
        device=device,
        use_amp=cfg.use_amp,
        grad_scaler=grad_scaler,
    )

    target_critic_info = update_target_network(
        target_network=target_critic,
    )

    # Merge all info dicts
    update_info = {
        **actor_info,
        **critic_info,
        **target_critic_info,
        **temperature_info,
    }

    return update_info


def _resolve_compile_mode(mode: str) -> str:
    """Resolve 'auto' compile mode based on the installed torch version."""
    if mode != "auto":
        return mode
    major, minor = (int(x) for x in torch.__version__.split(".")[:2])
    if (major, minor) >= (2, 9):
        return "max-autotune"
    return "reduce-overhead"


def _resolve_observation_layout(
    observation_dim: int,
    env_info: dict[str, Any],
    asymmetric: bool,
) -> tuple[int, int, int]:
    """Return actor width, critic width, and critic offset in replay observations."""
    if not asymmetric:
        return observation_dim, observation_dim, 0

    actor_dim = int(env_info["actor_observation_size"][-1])
    critic_size = env_info.get("critic_observation_size")
    if critic_size is None:
        # Backward-compatible convention used by Genesis/MJX: the privileged
        # tensor is the complete critic input and begins with the actor input.
        critic_dim = observation_dim
        critic_offset = 0
    else:
        critic_dim = int(critic_size[-1] if hasattr(critic_size, "__len__") else critic_size)
        critic_offset = int(env_info.get("critic_observation_offset", 0))

    if actor_dim > observation_dim:
        raise ValueError(f"actor observation width {actor_dim} exceeds replay width {observation_dim}")
    if critic_offset < 0 or critic_offset + critic_dim > observation_dim:
        raise ValueError(
            f"critic observation slice [{critic_offset}:{critic_offset + critic_dim}] "
            f"exceeds replay width {observation_dim}"
        )
    return actor_dim, critic_dim, critic_offset


class FlashSACAgent(BaseAgent[FlashSACConfig]):
    def __init__(
        self,
        observation_space: gym.spaces.Space[NDArray],
        action_space: gym.spaces.Space[NDArray],
        env_info: dict[str, Any],
        cfg: FlashSACConfig,
    ):
        """
        FlashSAC agent implementation in PyTorch.
        """

        observation_dim: int = observation_space.shape[-1]  # type: ignore
        self._action_dim: int = action_space.shape[-1]  # type: ignore
        (
            self._actor_observation_dim,
            self._critic_observation_dim,
            self._critic_observation_offset,
        ) = _resolve_observation_layout(
            observation_dim,
            env_info,
            cfg.asymmetric_observation,
        )

        temp_target_entropy = 0.5 * self._action_dim * math.log(2 * math.pi * math.e * cfg.temp_target_sigma**2)
        compile_mode = _resolve_compile_mode(cfg.compile_mode)
        cfg = replace(cfg, temp_target_entropy=temp_target_entropy, compile_mode=compile_mode)

        super().__init__(
            observation_space,
            action_space,
            env_info,
            cfg,
        )
        self._cfg = cfg

        if cfg.actor_history_length < 1 or cfg.actor_lstm_hidden_dim < 1:
            raise ValueError("History length and LSTM hidden size must be positive")
        if cfg.actor_history_length > 1 and cfg.n_step != 1:
            raise ValueError("Finite-history FlashSAC currently requires n_step=1")
        self._online_actor_history: torch.Tensor | None = None
        self._critic_network_observation_dim = self._critic_observation_dim
        if cfg.actor_history_length > 1:
            # Q must condition on the policy's history, not just current state.
            self._critic_network_observation_dim += cfg.actor_history_length * (self._actor_observation_dim + 1)

        device_type = cfg.device_type
        device_type = (
            device_type
            if device_type.startswith("cuda") and ":" in device_type
            else ("cuda:0" if device_type.startswith("cuda") else "cpu")
        )
        self._device = torch.device(device_type)

        # Initialize networks
        (
            self._actor,
            self._critic,
            self._target_critic,
            self._temperature,
        ) = _init_flashsac_networks(
            actor_observation_dim=self._actor_observation_dim,
            critic_observation_dim=self._critic_network_observation_dim,
            action_dim=self._action_dim,
            cfg=self._cfg,
            device=self._device,
        )
        self._update_step = 0

        # Grad scaler for FP16 AMP
        self._grad_scaler = GradScaler(device=self._device.type, enabled=self._cfg.use_amp)

        # Noise repetition (zeta distribution)
        self._zeta_cdf = _build_truncated_zeta_cdf(
            mu=self._cfg.actor_noise_zeta_mu, max_n=self._cfg.actor_noise_zeta_max
        ).to(self._device)
        self._cur_noise_repeat_n = torch.tensor(1, dtype=torch.int32, device=self._device)
        self._cur_noise_repeat_count = torch.tensor(0, dtype=torch.int32, device=self._device)
        action_shape = tuple(action_space.shape) if action_space.shape is not None else ()
        self._cached_noise = torch.randn(action_shape, device=self._device)

        # Reward normalizer
        self.reward_normalizer = None
        if self._cfg.normalize_reward:
            self.reward_normalizer = RewardNormalizer(
                gamma=self._cfg.gamma,
                G_max=self._cfg.normalized_G_max,
                load_rms=self._cfg.load_reward_normalizer,
                device=self._device,
            )

        # Replay buffer
        buffer_class: type[TorchUniformBuffer] = TorchUniformBuffer
        buffer_kwargs: dict[str, Any] = {}
        if cfg.actor_history_length > 1:
            from flash_rl.buffers.history_buffer import TorchHistoryBuffer

            buffer_class = TorchHistoryBuffer
            buffer_kwargs = dict(
                actor_observation_dim=self._actor_observation_dim, history_length=cfg.actor_history_length
            )
        self._replay_buffer = buffer_class(
            observation_space=observation_space,
            action_space=action_space,
            n_step=self._cfg.n_step,
            gamma=self._cfg.gamma,
            max_length=self._cfg.buffer_max_length,
            min_length=self._cfg.buffer_min_length,
            sample_batch_size=self._cfg.sample_batch_size,
            device_type=self._cfg.buffer_device_type,
            **buffer_kwargs,
        )

    def _actor_history_context(self, observations: torch.Tensor) -> torch.Tensor:
        """History before the current action; also populated during random warmup."""
        frame = torch.cat((observations, torch.ones_like(observations[:, :1])), dim=-1).unsqueeze(1)
        if self._online_actor_history is None:
            self._online_actor_history = observations.new_zeros(
                observations.shape[0], self._cfg.actor_history_length - 1, observations.shape[-1] + 1
            )
        if self._online_actor_history.shape[0] != observations.shape[0]:
            raise ValueError("Online history requires a fixed vector-environment batch")
        return torch.cat((self._online_actor_history, frame), dim=1)

    def sample_actions(
        self,
        interaction_step: int,
        prev_transition: MutableMapping[str, Tensor],
        training: bool,
    ) -> Tensor:
        if training:
            temperature = 1.0
        else:
            temperature = 0.0

        observations = prev_transition["next_observation"]
        if self._cfg.asymmetric_observation:
            observations = observations[:, : self._actor_observation_dim]

        observations = torch.as_tensor(observations, dtype=torch.float32).to(self._device)
        if self._cfg.actor_history_length > 1:
            # Generic evaluate/record does not provide per-env episode boundaries.
            # Fail explicitly rather than silently leaking history across resets.
            if not training:
                raise RuntimeError(
                    "History-policy playback requires an explicit sequential rollout; generic eval is unsupported"
                )
            observations = self._actor_history_context(observations)

        with torch.no_grad():
            (
                self._cached_noise,
                actions,
                self._cur_noise_repeat_count,
                self._cur_noise_repeat_n,
            ) = _sample_flashsac_actions(
                actor=self._actor,
                noise=self._cached_noise,
                observations=observations,
                temperature=temperature,
                cur_count=self._cur_noise_repeat_count,
                cur_n=self._cur_noise_repeat_n,
                zeta_cdf=self._zeta_cdf,
            )

        return actions.cpu().numpy()

    def process_transition(self, transition: MutableMapping[str, Tensor]) -> None:
        # add to replay buffer
        self._replay_buffer.add(transition)

        if self._cfg.actor_history_length > 1:
            current = torch.as_tensor(transition["observation"], dtype=torch.float32, device=self._device)[
                :, : self._actor_observation_dim
            ]
            context = self._actor_history_context(current)
            self._online_actor_history = context[:, 1:].clone()
            done = (
                torch.as_tensor(transition["terminated"], device=self._device).bool()
                | torch.as_tensor(transition["truncated"], device=self._device).bool()
            )
            self._online_actor_history[done] = 0

        # update reward normalizer
        if self._cfg.normalize_reward:
            assert "reward" in transition and self.reward_normalizer is not None
            self.reward_normalizer.update_reward_stats(
                reward=torch.as_tensor(transition["reward"], device=self._device),
                terminated=torch.as_tensor(transition["terminated"], device=self._device),
                truncated=torch.as_tensor(transition["truncated"], device=self._device),
            )

    def can_start_training(self) -> bool:
        return self._replay_buffer.can_sample()

    def update(self) -> dict[str, Any]:
        batch = cast(dict[str, torch.Tensor], self._replay_buffer.sample())

        for k, v in batch.items():
            batch[k] = v.to(self._device, non_blocking=True)

        if self._cfg.asymmetric_observation:
            batch["actor_observation"] = batch["observation"][:, : self._actor_observation_dim]
            batch["actor_next_observation"] = batch["next_observation"][:, : self._actor_observation_dim]
            critic_start = self._critic_observation_offset
            critic_end = critic_start + self._critic_observation_dim
            batch["observation"] = batch["observation"][:, critic_start:critic_end]
            batch["next_observation"] = batch["next_observation"][:, critic_start:critic_end]
        else:
            batch["actor_observation"] = batch["observation"]
            batch["actor_next_observation"] = batch["next_observation"]

        if self._cfg.actor_history_length > 1:
            history = batch.pop("actor_history")
            next_history = batch.pop("actor_next_history")
            batch["actor_observation"], batch["actor_next_observation"] = history, next_history
            batch["observation"] = torch.cat((batch["observation"], history.flatten(1)), dim=-1)
            batch["next_observation"] = torch.cat((batch["next_observation"], next_history.flatten(1)), dim=-1)

        if self._cfg.normalize_reward:
            assert self.reward_normalizer is not None
            # batch["unnormalized_reward"] = batch["reward"].clone()
            batch["reward"] = self.reward_normalizer.normalize_rewards(batch["reward"])

        # Update step
        _update_info = _update_networks(
            batch=batch,
            actor=self._actor,
            critic=self._critic,
            target_critic=self._target_critic,
            temperature=self._temperature,
            cfg=self._cfg,
            do_actor_update=(self._update_step % self._cfg.actor_update_period == 0),
            device=self._device,
            grad_scaler=self._grad_scaler,
        )
        self._update_step += 1

        # Convert tensors to floats
        update_info: dict[str, float] = {}
        for key, value in _update_info.items():
            if isinstance(value, torch.Tensor):
                update_info[key] = value.item()
            elif not isinstance(value, dict):
                update_info[key] = float(value)

        return update_info

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self._actor.save(os.path.join(path, "actor.pt"))
        self._critic.save(os.path.join(path, "critic.pt"))
        self._target_critic.save(os.path.join(path, "target_critic.pt"))
        self._temperature.save(os.path.join(path, "temperature.pt"))
        if self.reward_normalizer is not None:
            self.reward_normalizer.save(os.path.join(path, "reward_normalizer.pt"))

        agent_state: dict[str, Any] = {
            "update_step": self._update_step,
            "grad_scaler_state_dict": self._grad_scaler.state_dict(),
        }
        if self._cfg.actor_history_length > 1:
            agent_state["history_policy"] = {
                "history_length": self._cfg.actor_history_length,
                "lstm_hidden_dim": self._cfg.actor_lstm_hidden_dim,
                "actor_observation_dim": self._actor_observation_dim,
                "critic_current_observation_dim": self._critic_observation_dim,
                "critic_network_observation_dim": self._critic_network_observation_dim,
            }
        torch.save(agent_state, os.path.join(path, "agent_state.pt"))
        print(f"\033[32m[FlashSAC]\033[0m Successfully saved checkpoint {self._update_step} at {path}.")

    def save_replay_buffer(self, path: str) -> None:
        self._replay_buffer.save(os.path.join(path, "replay_buffer.pt"))
        print(f"\033[32m[FlashSAC]\033[0m Successfully saved replay buffer at {path}.")

    def load(self, path: str) -> None:
        self._online_actor_history = None  # Environment state is not restored by a model checkpoint.
        load_optimizer = self._cfg.load_optimizer
        self._actor.load(os.path.join(path, "actor.pt"), load_optimizer=load_optimizer)
        self._critic.load(os.path.join(path, "critic.pt"), load_optimizer=load_optimizer)
        self._target_critic.load(os.path.join(path, "target_critic.pt"), load_optimizer=False)
        self._temperature.load(os.path.join(path, "temperature.pt"), load_optimizer=load_optimizer)

        # Load agent-level optimizer state
        if load_optimizer:
            agent_state_path = os.path.join(path, "agent_state.pt")
            assert os.path.exists(agent_state_path)
            agent_state = torch.load(agent_state_path, map_location=self._device)
            self._update_step = agent_state["update_step"]
            self._grad_scaler.load_state_dict(agent_state["grad_scaler_state_dict"])

        if self._cfg.load_reward_normalizer:
            assert self.reward_normalizer is not None
            self.reward_normalizer.load(os.path.join(path, "reward_normalizer.pt"))

        print(f"\033[32m[FlashSAC]\033[0m Successfully loaded checkpoint from {path}.")

    def load_replay_buffer(self, path: str) -> None:
        self._replay_buffer.load(os.path.join(path, "replay_buffer.pt"))
        print(f"\033[32m[FlashSAC]\033[0m Successfully loaded replay buffer from {path}.")

    def get_metrics(self) -> dict[str, Any]:
        return {}
