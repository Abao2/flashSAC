import importlib
from collections.abc import Iterator, Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any, Union, cast

import gymnasium as gym
import numpy as np
import torch
import yaml
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space
from omegaconf import DictConfig, OmegaConf

from ..types import F32NDArray, NDArray

# NOTE: There is no way to get the action bounds from the env, so we hardcode them here following FastTD3
ACTION_BOUNDS = {
    "Isaac-Repose-Cube-Shadow-Direct-v0": 1.0,
    "Isaac-Repose-Cube-Allegro-Direct-v0": 1.0,
    "Isaac-Velocity-Flat-G1-v0": 1.0,
    "Isaac-Velocity-Rough-G1-v0": 1.0,
    "Isaac-Velocity-Flat-H1-v0": 1.0,
    "Isaac-Velocity-Rough-H1-v0": 1.0,
    "Isaac-Lift-Cube-Franka-v0": 3.0,
    "Isaac-Open-Drawer-Franka-v0": 3.0,
    "Isaac-Velocity-Flat-Anymal-C-v0": 1.0,
    "Isaac-Velocity-Rough-Anymal-C-v0": 1.0,
    "Isaac-Velocity-Flat-Anymal-D-v0": 1.0,
    "Isaac-Velocity-Rough-Anymal-D-v0": 1.0,
}


def import_registration_modules(modules: Sequence[str] | None) -> None:
    """Import packages whose module import registers external Gym tasks."""
    for module in modules or ():
        importlib.import_module(str(module))


def _to_plain_mapping(value: Mapping[str, Any] | DictConfig | None) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, DictConfig):
        result = OmegaConf.to_container(value, resolve=True)
        if not isinstance(result, dict):
            raise TypeError("task_cfg_overrides must resolve to a mapping")
        return result
    return dict(value)


def _load_yaml_entry_point(entry_point: str) -> dict[str, Any]:
    path = Path(entry_point)
    if not path.exists():
        try:
            module_name, relative_path = entry_point.split(":", maxsplit=1)
        except ValueError as exc:
            raise FileNotFoundError(f"Task YAML entry point does not exist: {entry_point}") from exc
        module = importlib.import_module(module_name)
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            raise FileNotFoundError(f"Cannot resolve task YAML entry point: {entry_point}")
        path = Path(module_file).resolve().parent / relative_path

    with path.open(encoding="utf-8") as stream:
        overlay = yaml.safe_load(stream) or {}
    if not isinstance(overlay, dict):
        raise TypeError(f"Task YAML must contain a mapping: {path}")
    print(f"[INFO]: Applying task YAML overlay from: {path}")
    return overlay


def apply_registered_task_overrides(
    env_cfg: Any,
    env_name: str,
    yaml_entry_point_key: str | None,
    task_cfg_overrides: Mapping[str, Any] | DictConfig | None,
) -> Any:
    """Apply a registered task YAML and then FlashSAC-side config overrides."""
    if yaml_entry_point_key:
        entry_point = gym.spec(env_name.split(":")[-1]).kwargs.get(yaml_entry_point_key)
        if entry_point:
            if not isinstance(entry_point, str):
                raise TypeError(f"{yaml_entry_point_key} must be a YAML path or module:path string")
            env_cfg.from_dict(_load_yaml_entry_point(entry_point))

    overrides = _to_plain_mapping(task_cfg_overrides)
    if overrides:
        env_cfg.from_dict(overrides)
    return env_cfg


def _numeric_leaves(value: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_prefix = f"{prefix}/{key}" if prefix else str(key)
            yield from _numeric_leaves(child, child_prefix)
    elif isinstance(value, (torch.Tensor, np.ndarray, Real)):
        yield prefix, value


def _nested_value(mapping: Mapping[str, Any], dotted_path: str | None) -> Any:
    value: Any = mapping
    for key in (dotted_path or "").split("."):
        if not key:
            return None
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def observation_subspaces(space: Any) -> Mapping[str, gym.Space[Any]]:
    """Return named subspaces without invoking Gymnasium Space.__contains__."""
    spaces = space.spaces if isinstance(space, gym.spaces.Dict) else space
    if not isinstance(spaces, Mapping):
        raise TypeError("Isaac Lab observations must be a mapping with a 'policy' space")
    return spaces


def combined_observation_shape(*shapes: tuple[int, ...]) -> tuple[int]:
    """Flatten named actor/critic vectors into the adapter's concatenated shape."""
    return (sum(int(np.prod(shape)) for shape in shapes),)


def recursive_to_numpy(
    data: Union[torch.Tensor, dict[str, Any], list[Any], tuple[Any, ...], NDArray],
) -> Union[NDArray, dict[str, Any], list[Any], tuple[Any, ...]]:
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    elif isinstance(data, dict):
        return {k: recursive_to_numpy(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(recursive_to_numpy(v) for v in data)
    else:
        return data


class IsaacLabVectorEnv(
    VectorEnv[Union[torch.Tensor, F32NDArray], Union[torch.Tensor, F32NDArray], Union[torch.Tensor, F32NDArray]]
):
    """
    Gymnasium "SyncVectorEnv" implementation for IsaacLab environments.

    As all jax-based env does, IsaacLab does not internally store the 'state' of the env.

    Args:
        env_name (str): The environment name registered in IsaacLab.
        num_envs (int): The number of parallel environments. This is only used if the env argument is a string
        device (str):
        seed (int):
        action_bounds (float):
        to_numpy (bool): If True, will convert all outputs from jnp.ndarray to np.array.
    """

    def __init__(
        self,
        env_name: str,
        num_envs: int,
        seed: int,
        device: str,
        action_bounds: float,
        to_numpy: bool = True,
        headless: bool = True,
        registration_modules: Sequence[str] | None = None,
        env_cfg_yaml_entry_point: str | None = "env_cfg_yaml_entry_point",
        task_cfg_overrides: Mapping[str, Any] | DictConfig | None = None,
        bootstrap_timeouts: bool = False,
        success_path: str | None = None,
    ):
        if bootstrap_timeouts:
            raise ValueError(
                "bootstrap_timeouts=true is unsafe for Isaac Lab DirectRLEnv: autoreset does not expose the real "
                "terminal observation. Keep it false until the task provides one explicitly."
            )

        from isaaclab.app import AppLauncher

        app_launcher = AppLauncher(headless=headless, device=device, enable_cameras=not headless)
        self.simulation_app = app_launcher.app

        # Isaac/Omniverse packages must be imported only after SimulationApp starts.
        import_registration_modules(registration_modules)

        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

        env_cfg = parse_env_cfg(
            env_name,
            device=device,
            num_envs=num_envs,
        )
        env_cfg = apply_registered_task_overrides(
            env_cfg,
            env_name=env_name,
            yaml_entry_point_key=env_cfg_yaml_entry_point,
            task_cfg_overrides=task_cfg_overrides,
        )
        # Trainer topology and selected device are authoritative over task presets.
        env_cfg.sim.device = device
        env_cfg.scene.num_envs = num_envs
        env_cfg.seed = seed
        self.seed = seed
        self.device = device
        self.bootstrap_timeouts = bootstrap_timeouts
        self.success_path = success_path
        self.envs = gym.make(env_name, cfg=env_cfg, render_mode=None)

        self.num_envs = cast(Any, self.envs.unwrapped).num_envs
        self.max_episode_steps = cast(Any, self.envs.unwrapped).max_episode_length
        self.to_numpy = to_numpy

        # Get observation/action spaces
        # NOTE: Action range: [-1, 1] * action_bounds (https://github.com/google-deepmind/mujoco_playground/issues/19)
        env_observation_spaces = observation_subspaces(cast(Any, self.envs.unwrapped).single_observation_space)
        self.obs_size = env_observation_spaces["policy"].shape
        self.asymmetric_obs = "critic" in env_observation_spaces
        if self.asymmetric_obs:
            # NOTE: Env will treat concatenate actor & critic states as the observation,
            # but will give 'actual' observation size in the info.
            self.critic_obs_size = env_observation_spaces["critic"].shape
            # NOTE: setting to [0, 0] since we only need the shape and dtype
            combined_obs_size = combined_observation_shape(self.obs_size, self.critic_obs_size)
            self.single_observation_space = gym.spaces.Box(
                low=0.0, high=0.0, shape=combined_obs_size, dtype=np.float32
            )
            self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        else:
            self.critic_obs_size = 0
            self.single_observation_space = gym.spaces.Box(low=0.0, high=0.0, shape=self.obs_size, dtype=np.float32)
            self.observation_space = batch_space(self.single_observation_space, self.num_envs)

        self.action_bounds = action_bounds
        self.action_size = cast(Any, self.envs.unwrapped).single_action_space.shape
        self.single_action_space = gym.spaces.Box(
            low=-1.0 * self.action_bounds, high=1.0 * self.action_bounds, shape=self.action_size, dtype=np.float32
        )
        self.action_space = batch_space(self.single_action_space, self.num_envs)
        print(
            f"[FlashSAC][IsaacLab] task={env_name} envs={self.num_envs} "
            f"actor_obs={self.obs_size[-1]} critic_input={self.single_observation_space.shape[-1]} "
            f"actions={self.action_size[-1]} bootstrap_timeouts={self.bootstrap_timeouts}"
        )
        self._episode_returns = torch.zeros(self.num_envs, device=self.device)
        self._episode_lengths = torch.zeros(self.num_envs, device=self.device)
        self._episode_cumulative: dict[str, torch.Tensor] = {}
        self._closed = False

    def _as_per_env_tensor(self, value: Any) -> torch.Tensor | None:
        if not isinstance(value, (torch.Tensor, np.ndarray, Real)):
            return None
        tensor = torch.as_tensor(value, device=self.device, dtype=torch.float32)
        if tensor.ndim == 0:
            return tensor.expand(self.num_envs)
        if tensor.shape[0] != self.num_envs:
            return None
        if tensor.ndim > 1:
            tensor = tensor.reshape(self.num_envs, -1).mean(dim=-1)
        return tensor

    def _reset_episode_statistics(self, mask: torch.Tensor | None = None) -> None:
        if mask is None:
            self._episode_returns.zero_()
            self._episode_lengths.zero_()
            self._episode_cumulative.clear()
            return
        self._episode_returns[mask] = 0
        self._episode_lengths[mask] = 0
        for values in self._episode_cumulative.values():
            values[mask] = 0

    def _collect_episode_info(
        self,
        infos: Mapping[str, Any],
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict[str, float | tuple[float, int]]:
        self._episode_returns += rewards
        self._episode_lengths += 1

        for name, value in _numeric_leaves(infos.get("episode_cumulative", {})):
            values = self._as_per_env_tensor(value)
            if values is None:
                continue
            if name not in self._episode_cumulative:
                self._episode_cumulative[name] = torch.zeros(self.num_envs, device=self.device)
            self._episode_cumulative[name] += values

        metrics: dict[str, float | tuple[float, int]] = {}
        if torch.any(dones):
            done_count = int(dones.sum().item())
            metrics["episode/return"] = (float(self._episode_returns[dones].mean().item()), done_count)
            metrics["episode/length"] = (float(self._episode_lengths[dones].mean().item()), done_count)
            for name, values in self._episode_cumulative.items():
                metrics[f"episode/cumulative/{name}"] = (float(values[dones].mean().item()), done_count)
            for name, value in _numeric_leaves(infos.get("episode_final", {})):
                values = self._as_per_env_tensor(value)
                if values is not None:
                    metrics[f"episode/final/{name}"] = (float(values[dones].mean().item()), done_count)
            self._reset_episode_statistics(dones)

        tolerance = infos.get("current_success_tolerance")
        if isinstance(tolerance, Real):
            metrics["task/current_success_tolerance"] = float(tolerance)
        return metrics

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
        random_start_init: bool = True,
    ) -> tuple[Union[torch.Tensor, F32NDArray], dict[str, Any]]:
        obs_dict, infos = self.envs.reset()
        self._reset_episode_statistics()
        obs = obs_dict["policy"]
        if self.asymmetric_obs:
            critic_obs = obs_dict["critic"]
            obs = torch.cat((obs, critic_obs), dim=-1)
        else:
            critic_obs = None
        # NOTE: decorrelate episode horizons like RSL‑RL
        # In IsaacLab, `dones` is computed as follows:
        # `time_out = self.episode_length_buf >= self.max_episode_length - 1`
        # While training, this code spreads out the resets to avoid spikes
        # when many environments reset at a similar time.
        if random_start_init:
            # step in current episode (per env)
            cast(Any, self.envs.unwrapped).episode_length_buf = torch.randint_like(
                cast(Any, self.envs.unwrapped).episode_length_buf, high=int(self.max_episode_steps)
            )
        if self.to_numpy:
            obs = obs.cpu().numpy()
            infos = recursive_to_numpy(infos)  # type: ignore
        infos.update({"actor_observation_size": self.obs_size, "asymmetric_obs": self.asymmetric_obs})
        return obs, infos

    def step(self, actions: Union[torch.Tensor, F32NDArray]) -> tuple[
        Union[torch.Tensor, F32NDArray],
        Union[torch.Tensor, F32NDArray],
        Union[torch.Tensor, F32NDArray],
        Union[torch.Tensor, F32NDArray],
        dict[str, Any],
    ]:
        if isinstance(actions, torch.Tensor):
            torch_actions = actions.to(self.device)
        else:
            torch_actions = torch.from_numpy(actions).to(self.device)

        if self.action_bounds is not None:
            torch_actions = torch.clamp(torch_actions, -1.0, 1.0) * self.action_bounds
        obs_dict, rew, terminations, truncations, infos = cast(Any, self.envs.step(torch_actions))
        obs = obs_dict["policy"]
        if self.asymmetric_obs:
            critic_obs = obs_dict["critic"]
            obs = torch.cat((obs, critic_obs), dim=-1)
        else:
            critic_obs = None
        infos = dict(infos)
        dones = torch.logical_or(terminations, truncations)
        episode_info = self._collect_episode_info(infos, rew, dones)
        if episode_info:
            infos["episode_info"] = episode_info

        success = _nested_value(infos, self.success_path)
        if success is not None:
            infos["success"] = success
            final_info = dict(infos.get("final_info", {}))
            final_info["success"] = success
            infos["final_info"] = final_info

        infos["time_outs"] = truncations
        infos["observations"] = {"critic": critic_obs}

        # DirectRLEnv returns reset observations for done environments. Masking
        # timeouts as terminal prevents bootstrapping from a different episode.
        if not self.bootstrap_timeouts:
            terminations = dones

        if self.to_numpy:
            obs = obs.cpu().numpy()
            rew = rew.cpu().numpy()
            terminations = terminations.cpu().numpy()
            truncations = truncations.cpu().numpy()
            infos = recursive_to_numpy(infos)
        return obs, rew, terminations, truncations, infos

    def close(self, **kwargs: Any) -> None:
        if getattr(self, "_closed", False):
            return
        self._closed = True
        if hasattr(self, "envs") and hasattr(self.envs, "close"):
            self.envs.close(**kwargs)
        if hasattr(self, "simulation_app"):
            self.simulation_app.close()

    def render(self) -> None:
        raise NotImplementedError("We don't support rendering for IsaacLab environments")


def make_isaaclab_env(
    env_name: str,
    num_envs: int,
    seed: int,
    headless: bool = True,
    device: str | None = None,
    action_bounds: float | None = None,
    registration_modules: Sequence[str] | None = None,
    env_cfg_yaml_entry_point: str | None = "env_cfg_yaml_entry_point",
    task_cfg_overrides: Mapping[str, Any] | DictConfig | None = None,
    bootstrap_timeouts: bool = False,
    success_path: str | None = None,
) -> IsaacLabVectorEnv:
    if action_bounds is None and env_name not in ACTION_BOUNDS:
        print(f"Action bounds not defined for {env_name}; using default value 1.0.")
    action_bounds = action_bounds if action_bounds is not None else ACTION_BOUNDS.get(env_name, 1.0)
    env = IsaacLabVectorEnv(
        env_name=env_name,
        num_envs=num_envs,
        seed=seed,
        device=device or ("cuda:0" if torch.cuda.is_available() else "cpu"),
        action_bounds=action_bounds,
        to_numpy=True,
        headless=headless,
        registration_modules=registration_modules,
        env_cfg_yaml_entry_point=env_cfg_yaml_entry_point,
        task_cfg_overrides=task_cfg_overrides,
        bootstrap_timeouts=bootstrap_timeouts,
        success_path=success_path,
    )
    return env
