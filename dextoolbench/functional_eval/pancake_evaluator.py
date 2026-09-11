"""Vectorized, policy-independent functional evaluation for pancake tasks.

All positions passed to this module must already be expressed in the same
environment-local frame.  Quaternions use Isaac Lab's ``wxyz`` convention.
The evaluator only reads simulator ground truth and never changes policy
observations, rewards, actions, or termination state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Literal, Mapping, Optional, Sequence, Tuple, Union

import torch

Tensor = torch.Tensor
TensorState = Mapping[str, Tensor]
TaskName = Literal["scoop_and_lift", "serve_to_plate", "flip_and_settle"]


class FailureCode(IntEnum):
    """Stable integer values suitable for batched tensors and JSON export."""

    NONE = 0
    NEVER_CONTACTED_OR_LIFTED = 1
    NOT_FLIPPED = 2
    FLIPPED_BUT_OUTSIDE = 3
    FLIPPED_BUT_UNSTABLE = 4
    PANCAKE_DROPPED = 5
    TOOL_TRAJECTORY_FAILED = 6
    TIMEOUT = 7


@dataclass(frozen=True)
class PancakeEvaluatorCfg:
    """Thresholds shared by the three functional pancake tasks."""

    task_name: TaskName = "scoop_and_lift"
    pancake_local_normal: Tuple[float, float, float] = (0.0, 0.0, 1.0)

    # flat_spatula has one rigid body.  Its OBJ blade spans roughly
    # x=[0.10, 0.232] m, so this offset targets the blade center.
    spatula_blade_link_name: Optional[str] = None
    spatula_blade_local_offset: Tuple[float, float, float] = (0.166, 0.0, 0.0)
    max_blade_pancake_distance: float = 0.08

    lift_height_threshold: float = 0.03
    serve_lift_height_threshold: float = 0.02
    lift_hold_steps: int = 10

    stable_hold_steps: int = 30
    max_linear_speed: float = 0.05
    max_angular_speed: float = 0.5
    max_upright_tilt_deg: float = 30.0
    min_flip_angle_deg: float = 150.0

    target_center_xy: Tuple[float, float] = (0.0, 0.0)
    target_valid_radius: float = 0.08
    containment_margin: float = 0.0
    support_height: float = 0.536
    support_height_tolerance: float = 0.02

    drop_height: float = 0.48
    post_trajectory_settle_steps: int = 60
    max_episode_steps: int = 3600

    def __post_init__(self) -> None:
        valid_tasks = {"scoop_and_lift", "serve_to_plate", "flip_and_settle"}
        if self.task_name not in valid_tasks:
            raise ValueError(f"unsupported task_name: {self.task_name!r}")
        positive = {
            "max_blade_pancake_distance": self.max_blade_pancake_distance,
            "lift_height_threshold": self.lift_height_threshold,
            "serve_lift_height_threshold": self.serve_lift_height_threshold,
            "lift_hold_steps": self.lift_hold_steps,
            "stable_hold_steps": self.stable_hold_steps,
            "max_linear_speed": self.max_linear_speed,
            "max_angular_speed": self.max_angular_speed,
            "target_valid_radius": self.target_valid_radius,
            "support_height_tolerance": self.support_height_tolerance,
            "post_trajectory_settle_steps": self.post_trajectory_settle_steps,
            "max_episode_steps": self.max_episode_steps,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"configuration values must be positive: {invalid}")
        if not 0.0 <= self.max_upright_tilt_deg <= 180.0:
            raise ValueError("max_upright_tilt_deg must be in [0, 180]")
        if not 0.0 <= self.min_flip_angle_deg <= 180.0:
            raise ValueError("min_flip_angle_deg must be in [0, 180]")
        normal_norm = math.sqrt(
            sum(value * value for value in self.pancake_local_normal)
        )
        if normal_norm == 0.0:
            raise ValueError("pancake_local_normal must be non-zero")


def quat_rotate_wxyz(quat: Tensor, vector: Tensor) -> Tensor:
    """Rotate batched 3D vectors by batched ``wxyz`` quaternions."""
    if quat.shape[-1] != 4 or vector.shape[-1] != 3:
        raise ValueError("expected quaternion (..., 4) and vector (..., 3)")
    quat = quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp_min(1e-12)
    quat_xyz = quat[..., 1:]
    twice_cross = 2.0 * torch.linalg.cross(quat_xyz, vector, dim=-1)
    return (
        vector
        + quat[..., :1] * twice_cross
        + torch.linalg.cross(quat_xyz, twice_cross, dim=-1)
    )


def normal_angle_deg(reference_normal: Tensor, current_normal: Tensor) -> Tensor:
    """Return the unsigned angle between batched normals in degrees."""
    reference_normal = torch.nn.functional.normalize(reference_normal, dim=-1)
    current_normal = torch.nn.functional.normalize(current_normal, dim=-1)
    cosine = (reference_normal * current_normal).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine))


def target_contains(
    position_xy: Tensor, center_xy: Tensor, valid_radius: float
) -> Tensor:
    """Test circular target containment for batched environment-local points."""
    return torch.linalg.vector_norm(position_xy - center_xy, dim=-1) <= valid_radius


def is_stable(
    linear_velocity: Tensor,
    angular_velocity: Tensor,
    max_linear_speed: float,
    max_angular_speed: float,
) -> Tensor:
    """Return whether both translational and angular speeds are below limits."""
    return (torch.linalg.vector_norm(linear_velocity, dim=-1) <= max_linear_speed) & (
        torch.linalg.vector_norm(angular_velocity, dim=-1) <= max_angular_speed
    )


def update_hold_counter(predicate: Tensor, counter: Tensor) -> Tensor:
    """Increment consecutive true frames and clear the counter on false frames."""
    return torch.where(predicate, counter + 1, torch.zeros_like(counter))


class PancakeFunctionalEvaluator:
    """Maintain per-environment functional state without touching the policy."""

    _REQUIRED_STATE = (
        "pancake_pos",
        "pancake_quat",
        "pancake_lin_vel",
        "pancake_ang_vel",
        "spatula_pos",
        "spatula_quat",
    )

    def __init__(
        self,
        cfg: PancakeEvaluatorCfg,
        num_envs: int,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.dtype = dtype

        self.initial_pancake_pos = torch.zeros(
            (num_envs, 3), device=self.device, dtype=dtype
        )
        self.initial_pancake_quat = torch.zeros(
            (num_envs, 4), device=self.device, dtype=dtype
        )
        self.initial_pancake_normal = torch.zeros(
            (num_envs, 3), device=self.device, dtype=dtype
        )
        self.initial_target_distance = torch.zeros(
            num_envs, device=self.device, dtype=dtype
        )
        self.max_lift_height = torch.zeros(num_envs, device=self.device, dtype=dtype)
        self.has_lifted = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
        self.success_counter = torch.zeros(
            num_envs, device=self.device, dtype=torch.long
        )
        self.success_ever = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
        self.functional_progress = torch.zeros(
            num_envs, device=self.device, dtype=dtype
        )
        self.failure_code = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.min_blade_distance = torch.full(
            (num_envs,), torch.inf, device=self.device, dtype=dtype
        )
        self.near_blade_steps = torch.zeros(
            num_envs, device=self.device, dtype=torch.long
        )
        self._spatula_ever_approached = torch.zeros(
            num_envs, device=self.device, dtype=torch.bool
        )
        self._initialized = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
        self._all_initialized = False

    def reset(
        self,
        env_ids: Union[Tensor, Sequence[int]],
        initial_state: TensorState,
    ) -> None:
        """Reset selected environments from ground-truth local-frame state."""
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()
        if ids.numel() == 0:
            return
        if torch.any(ids < 0) or torch.any(ids >= self.num_envs):
            raise IndexError("env_ids contains an out-of-range environment index")
        self._validate_state(initial_state, allow_subset=ids.numel())

        pancake_pos = self._select_initial(initial_state["pancake_pos"], ids)
        pancake_quat = self._select_initial(initial_state["pancake_quat"], ids)
        local_normal = self._constant(
            self.cfg.pancake_local_normal, pancake_pos.shape[0], width=3
        )
        initial_normal = quat_rotate_wxyz(pancake_quat, local_normal)
        target_center = self._constant(
            self.cfg.target_center_xy, pancake_pos.shape[0], width=2
        )

        self.initial_pancake_pos[ids] = pancake_pos
        self.initial_pancake_quat[ids] = pancake_quat
        self.initial_pancake_normal[ids] = initial_normal
        self.initial_target_distance[ids] = torch.linalg.vector_norm(
            pancake_pos[:, :2] - target_center, dim=-1
        )
        self.max_lift_height[ids] = 0.0
        self.has_lifted[ids] = False
        self.success_counter[ids] = 0
        self.success_ever[ids] = False
        self.functional_progress[ids] = 0.0
        self.failure_code[ids] = int(FailureCode.NONE)
        self.min_blade_distance[ids] = torch.inf
        self.near_blade_steps[ids] = 0
        self._spatula_ever_approached[ids] = False
        self._initialized[ids] = True
        self._all_initialized = bool(torch.all(self._initialized).item())

    def update(self, current_state: TensorState) -> Dict[str, Tensor]:
        """Update all environments and return the common metric dictionary."""
        self._validate_state(current_state)
        if not self._all_initialized:
            missing = (~self._initialized).nonzero(as_tuple=False).flatten().tolist()
            raise RuntimeError(
                f"reset must be called before update; missing envs: {missing}"
            )
        if self.cfg.task_name != "scoop_and_lift":
            raise NotImplementedError(
                f"{self.cfg.task_name!r} is scheduled for a later implementation phase"
            )

        pancake_pos = current_state["pancake_pos"]
        pancake_quat = current_state["pancake_quat"]
        local_normal = self._constant(
            self.cfg.pancake_local_normal, self.num_envs, width=3
        )
        current_normal = quat_rotate_wxyz(pancake_quat, local_normal)
        flip_angle = normal_angle_deg(self.initial_pancake_normal, current_normal)

        target_center = self._constant(
            self.cfg.target_center_xy, self.num_envs, width=2
        )
        target_distance = torch.linalg.vector_norm(
            pancake_pos[:, :2] - target_center, dim=-1
        )

        lift_height = pancake_pos[:, 2] - self.initial_pancake_pos[:, 2]
        self.max_lift_height = torch.maximum(self.max_lift_height, lift_height)
        self.has_lifted |= lift_height >= self.cfg.lift_height_threshold

        blade_center = self._blade_center(current_state)
        blade_distance = torch.linalg.vector_norm(pancake_pos - blade_center, dim=-1)
        self.min_blade_distance = torch.minimum(self.min_blade_distance, blade_distance)
        near_blade = blade_distance <= self.cfg.max_blade_pancake_distance
        self.near_blade_steps += near_blade.long()
        self._spatula_ever_approached |= near_blade

        not_dropped = pancake_pos[:, 2] >= self.cfg.drop_height
        instant_success = (
            (lift_height >= self.cfg.lift_height_threshold) & near_blade & not_dropped
        )
        self.success_counter = update_hold_counter(
            instant_success, self.success_counter
        )
        functional_success = self.success_counter >= self.cfg.lift_hold_steps
        self.success_ever |= functional_success
        self.functional_progress = torch.clamp(
            lift_height / self.cfg.lift_height_threshold, 0.0, 1.0
        )

        unresolved = torch.full_like(self.failure_code, int(FailureCode.NONE))
        no_attempt = ~self.has_lifted & ~self._spatula_ever_approached
        unresolved[no_attempt] = int(FailureCode.NEVER_CONTACTED_OR_LIFTED)
        unresolved[~not_dropped] = int(FailureCode.PANCAKE_DROPPED)
        unresolved[self.success_ever] = int(FailureCode.NONE)
        self.failure_code = unresolved

        return {
            "functional_progress": self.functional_progress,
            "instant_success": instant_success,
            "functional_success": functional_success,
            "success_ever": self.success_ever,
            "max_lift_height": self.max_lift_height,
            "flip_angle_deg": flip_angle,
            "target_distance": target_distance,
            "failure_code": self.failure_code,
            "min_blade_distance": self.min_blade_distance,
            "blade_distance": blade_distance,
            "lift_height": lift_height,
            "not_dropped": not_dropped,
            "near_blade_steps": self.near_blade_steps,
            "success_counter": self.success_counter,
            "spatula_never_approached": ~self._spatula_ever_approached,
            "has_lifted": self.has_lifted,
        }

    def _blade_center(self, state: TensorState) -> Tensor:
        if "blade_pos" in state:
            blade_pos = state["blade_pos"]
            self._validate_tensor("blade_pos", blade_pos, 3, self.num_envs)
            return blade_pos
        offset = self._constant(
            self.cfg.spatula_blade_local_offset, self.num_envs, width=3
        )
        return state["spatula_pos"] + quat_rotate_wxyz(state["spatula_quat"], offset)

    def _constant(self, values: Sequence[float], rows: int, width: int) -> Tensor:
        tensor = torch.as_tensor(values, device=self.device, dtype=self.dtype)
        if tensor.shape != (width,):
            raise ValueError(f"expected {width} configured values, got {tensor.shape}")
        return tensor.expand(rows, -1)

    def _select_initial(self, tensor: Tensor, env_ids: Tensor) -> Tensor:
        if tensor.shape[0] == self.num_envs:
            return tensor[env_ids]
        if tensor.shape[0] == env_ids.numel():
            return tensor
        raise ValueError(
            "initial state batch must have num_envs rows or one row per env_id"
        )

    def _validate_state(
        self, state: TensorState, allow_subset: Optional[int] = None
    ) -> None:
        missing = [name for name in self._REQUIRED_STATE if name not in state]
        if missing:
            raise KeyError(f"state is missing required tensors: {missing}")
        rows = self.num_envs if allow_subset is None else (self.num_envs, allow_subset)
        widths = {
            "pancake_pos": 3,
            "pancake_quat": 4,
            "pancake_lin_vel": 3,
            "pancake_ang_vel": 3,
            "spatula_pos": 3,
            "spatula_quat": 4,
        }
        for name, width in widths.items():
            self._validate_tensor(name, state[name], width, rows)

    def _validate_tensor(
        self,
        name: str,
        tensor: Tensor,
        width: int,
        rows: Union[int, Tuple[int, int]],
    ) -> None:
        valid_rows = (rows,) if isinstance(rows, int) else rows
        if (
            tensor.ndim != 2
            or tensor.shape[0] not in valid_rows
            or tensor.shape[1] != width
        ):
            raise ValueError(
                f"{name} must have shape ({valid_rows}, {width}); "
                f"got {tuple(tensor.shape)}"
            )
        if tensor.device != self.device:
            raise ValueError(
                f"{name} is on {tensor.device}, evaluator is on {self.device}"
            )
        if not tensor.is_floating_point():
            raise TypeError(f"{name} must be a floating-point tensor")
        if tensor.dtype != self.dtype:
            raise TypeError(
                f"{name} has dtype {tensor.dtype}, evaluator expects {self.dtype}"
            )


__all__ = [
    "FailureCode",
    "PancakeEvaluatorCfg",
    "PancakeFunctionalEvaluator",
    "is_stable",
    "normal_angle_deg",
    "quat_rotate_wxyz",
    "target_contains",
    "update_hold_counter",
]
