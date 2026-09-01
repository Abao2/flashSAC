"""Pure-Torch task equations from the official Dex4D AP2AP teacher task.

Dex4D source: https://github.com/Dex4D/Dex4D-Simulation
The simulator-facing port uses wxyz quaternions internally while preserving
the legacy Isaac Gym xyzw observation layout.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch


LEGACY_JOINT_NAMES = (
    "joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
    "1", "0", "2", "3", "12", "13", "14", "15",
    "5", "4", "6", "7", "9", "8", "10", "11",
)

OBS_SLICES = {
    "joint_pos": slice(0, 22),
    "joint_vel": slice(22, 44),
    "joint_force": slice(44, 66),
    "fingertip_states": slice(66, 118),
    "fingertip_wrenches": slice(118, 142),
    "palm_state": slice(142, 155),
    "actions": slice(155, 177),
    "object_pos": slice(177, 180),
    "object_quat": slice(180, 184),
    "object_linvel": slice(184, 187),
    "object_angvel": slice(187, 190),
    "goal_delta_pos": slice(190, 193),
    "goal_delta_quat": slice(193, 197),
    "object_keypoints": slice(197, 581),
    "goal_keypoints": slice(581, 965),
    "visual_features": slice(965, 1029),
    "fingertip_object_vectors": slice(1029, 1041),
}

NUM_ARM_DOFS = 6


def observation_dim(num_dofs: int, num_fingertips: int, num_keypoints: int) -> int:
    """Return the Dex4D observation width for the configured robot/object."""
    return 4 * num_dofs + 22 * num_fingertips + 6 * num_keypoints + 97


OBS_DIM = observation_dim(len(LEGACY_JOINT_NAMES), 4, 128)


def quat_wxyz_to_xyzw(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert any batched quaternion tensor from wxyz to xyzw."""
    if quaternion.shape[-1] != 4:
        raise ValueError(f"Expected quaternion[..., 4], got {tuple(quaternion.shape)}")
    return torch.cat((quaternion[..., 1:], quaternion[..., :1]), dim=-1)


def _quat_mul_wxyz(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = lhs.unbind(-1)
    rw, rx, ry, rz = rhs.unbind(-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def transform_keypoints(
    local_keypoints: torch.Tensor,
    position: torch.Tensor,
    quaternion_wxyz: torch.Tensor,
) -> torch.Tensor:
    """Transform ``(..., K, 3)`` local keypoints into the world frame."""
    q_vec = quaternion_wxyz[..., 1:].unsqueeze(-2).expand_as(local_keypoints)
    q_w = quaternion_wxyz[..., :1].unsqueeze(-2)
    rotated = (
        local_keypoints * (2.0 * q_w.square() - 1.0)
        + 2.0 * q_w * torch.cross(q_vec, local_keypoints, dim=-1)
        + 2.0 * q_vec * torch.sum(q_vec * local_keypoints, dim=-1, keepdim=True)
    )
    return rotated + position.unsqueeze(-2)


def unscale_joints(
    joint_pos: torch.Tensor,
    joint_lower: torch.Tensor,
    joint_upper: torch.Tensor,
) -> torch.Tensor:
    """Map joint positions from their limits to the legacy ``[-1, 1]`` range."""
    return 2.0 * (joint_pos - joint_lower) / (joint_upper - joint_lower) - 1.0


def _state_wxyz_to_xyzw(state: torch.Tensor) -> torch.Tensor:
    return torch.cat((state[..., :3], quat_wxyz_to_xyzw(state[..., 3:7]), state[..., 7:]), dim=-1)


def _closest_keypoint_vectors(fingertip_pos: torch.Tensor, keypoints: torch.Tensor) -> torch.Tensor:
    vectors = keypoints.unsqueeze(-3) - fingertip_pos.unsqueeze(-2)
    nearest = vectors.square().sum(-1).argmin(-1)
    return vectors.gather(-2, nearest[..., None, None].expand(*nearest.shape, 1, 3)).squeeze(-2)


def build_observation(
    *,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    joint_force: torch.Tensor,
    joint_lower: torch.Tensor,
    joint_upper: torch.Tensor,
    fingertip_states_wxyz: torch.Tensor,
    fingertip_wrenches: torch.Tensor,
    fingertip_positions: torch.Tensor,
    palm_state_wxyz: torch.Tensor,
    actions: torch.Tensor,
    object_pos: torch.Tensor,
    object_quat_wxyz: torch.Tensor,
    object_linvel: torch.Tensor,
    object_angvel: torch.Tensor,
    goal_pos: torch.Tensor,
    goal_quat_wxyz: torch.Tensor,
    local_keypoints: torch.Tensor,
    visual_features: torch.Tensor,
    clip: float | None = None,
) -> torch.Tensor:
    """Build a Dex4D teacher observation while preserving the legacy layout.

    ``fingertip_positions`` are the task-corrected positions used by Dex4D
    for nearest-keypoint vectors, not the raw body origins.
    """
    num_dofs = joint_pos.shape[-1]
    num_fingertips = fingertip_positions.shape[-2]
    num_keypoints = local_keypoints.shape[-2]
    dof_dims = {
        "joint_vel": joint_vel.shape[-1],
        "joint_force": joint_force.shape[-1],
        "joint_lower": joint_lower.shape[-1],
        "joint_upper": joint_upper.shape[-1],
        "actions": actions.shape[-1],
    }
    fingertip_counts = {
        "fingertip_states_wxyz": fingertip_states_wxyz.shape[-2],
        "fingertip_wrenches": fingertip_wrenches.shape[-2],
    }
    if any(dim != num_dofs for dim in dof_dims.values()):
        raise ValueError(f"Inconsistent DOF dimensions: joint_pos={num_dofs}, {dof_dims}")
    if any(count != num_fingertips for count in fingertip_counts.values()):
        raise ValueError(
            f"Inconsistent fingertip counts: fingertip_positions={num_fingertips}, {fingertip_counts}"
        )

    object_keypoints = transform_keypoints(local_keypoints, object_pos, object_quat_wxyz)
    goal_keypoints = transform_keypoints(local_keypoints, goal_pos, goal_quat_wxyz)
    goal_delta_quat = _quat_mul_wxyz(
        goal_quat_wxyz,
        torch.cat((object_quat_wxyz[..., :1], -object_quat_wxyz[..., 1:]), dim=-1),
    )

    palm_state = _state_wxyz_to_xyzw(palm_state_wxyz).clone()
    palm_state[..., 10:13] *= 0.2
    observation = torch.cat(
        (
            unscale_joints(joint_pos, joint_lower, joint_upper),
            0.2 * joint_vel,
            10.0 * joint_force,
            _state_wxyz_to_xyzw(fingertip_states_wxyz).flatten(-2),
            (10.0 * fingertip_wrenches).flatten(-2),
            palm_state,
            actions,
            object_pos,
            quat_wxyz_to_xyzw(object_quat_wxyz),
            object_linvel,
            0.2 * object_angvel,
            goal_pos - object_pos,
            quat_wxyz_to_xyzw(goal_delta_quat),
            object_keypoints.flatten(-2),
            goal_keypoints.flatten(-2),
            0.1 * visual_features,
            _closest_keypoint_vectors(fingertip_positions, object_keypoints).flatten(-2),
        ),
        dim=-1,
    )
    expected_dim = observation_dim(num_dofs, num_fingertips, num_keypoints)
    if observation.shape[-1] != expected_dim:
        raise ValueError(
            f"Expected {expected_dim}-D Dex4D observation for D={num_dofs}, "
            f"F={num_fingertips}, K={num_keypoints}; got {observation.shape[-1]}"
        )
    return observation.clamp(-clip, clip) if clip is not None else observation


def compute_metrics(
    *,
    object_keypoints: torch.Tensor,
    goal_keypoints: torch.Tensor,
    object_pos: torch.Tensor,
    palm_pos: torch.Tensor,
    fingertip_positions: torch.Tensor,
    hand_joint_pos: torch.Tensor,
    finger_distance_scale: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Compute the shared distance and regularization metrics used by Dex4D."""
    goal_obj_dist = torch.linalg.vector_norm(goal_keypoints - object_keypoints, dim=-1).mean(-1)
    obj_hand_dist = torch.linalg.vector_norm(object_pos - palm_pos, dim=-1).clamp(max=0.5)
    obj_finger_dist = (
        finger_distance_scale
        * torch.linalg.vector_norm(object_pos.unsqueeze(-2) - fingertip_positions, dim=-1).sum(-1)
    ).clamp(max=3.0)
    min_body_height = torch.cat((fingertip_positions[..., 2], palm_pos[..., 2:3]), dim=-1).min(-1).values
    return {
        "goal_obj_dist": goal_obj_dist,
        "obj_hand_dist": obj_hand_dist,
        "obj_finger_dist": obj_finger_dist,
        "grasped": (obj_finger_dist <= 0.48) & (obj_hand_dist <= 0.12),
        "finger_curl_dist": torch.linalg.vector_norm(hand_joint_pos, dim=-1),
        "min_body_height": min_body_height,
    }


def _weight(weights: Mapping[str, float] | object, name: str) -> float:
    return float(weights.get(name, 0.0) if isinstance(weights, Mapping) else getattr(weights, name, 0.0))


def compute_reward(
    metrics: Mapping[str, torch.Tensor],
    actions: torch.Tensor,
    achieved_steps: torch.Tensor,
    weights: Mapping[str, float] | object,
    *,
    success_threshold: float = 0.05,
    achieved_steps_required: int = 30,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute scaled Dex4D reward terms and their sum."""
    goal_dist = metrics["goal_obj_dist"]
    grasped = metrics["grasped"]
    zero = torch.zeros_like(goal_dist)
    terms = {
        "obj_finger": -0.5 * metrics["obj_finger_dist"],
        "obj_hand": -0.5 * metrics["obj_hand_dist"],
        "goal_obj": torch.where(grasped, 1.4 - 3.0 * goal_dist, zero),
        "success_bonus": torch.where(
            grasped & (goal_dist <= success_threshold), 5.0 / (1.0 + 10.0 * goal_dist), zero
        ),
        "terminal_bonus": torch.where(achieved_steps >= achieved_steps_required, 10.0, zero),
        "finger_curl_reg": -0.001 * metrics["finger_curl_dist"].square(),
        "table_collision_penalty": torch.where(
            metrics["min_body_height"] < 0.62, 10.0 * (metrics["min_body_height"] - 0.62), zero
        ),
        "action_penalty": (-0.01 * actions.square().sum(-1)).clamp(min=-0.5, max=0.0),
    }
    scaled_terms = {name: value * _weight(weights, name) for name, value in terms.items()}
    return sum(scaled_terms.values(), start=zero), scaled_terms


def compute_action_targets(
    actions: torch.Tensor,
    previous_targets: torch.Tensor,
    joint_lower: torch.Tensor,
    joint_upper: torch.Tensor,
    *,
    speed_scale: float,
    legacy_dt: float = 1.0 / 60.0,
    moving_average: float = 1.0,
    num_arm_dofs: int = NUM_ARM_DOFS,
    hand_moving_average: float | None = None,
) -> torch.Tensor:
    """Map the noised legacy action to arm-increment/hand-absolute targets."""
    arm = previous_targets[..., :num_arm_dofs] + speed_scale * legacy_dt * actions[..., :num_arm_dofs]
    hand = joint_lower[..., num_arm_dofs:] + 0.5 * (actions[..., num_arm_dofs:] + 1.0) * (
        joint_upper[..., num_arm_dofs:] - joint_lower[..., num_arm_dofs:]
    )
    arm = moving_average * arm + (1.0 - moving_average) * previous_targets[..., :num_arm_dofs]
    hand_average = moving_average if hand_moving_average is None else hand_moving_average
    hand = hand_average * hand + (1.0 - hand_average) * previous_targets[..., num_arm_dofs:]
    targets = torch.cat((arm, hand), dim=-1)
    return torch.maximum(torch.minimum(targets, joint_upper), joint_lower)


__all__ = [
    "LEGACY_JOINT_NAMES",
    "OBS_DIM",
    "OBS_SLICES",
    "build_observation",
    "compute_action_targets",
    "compute_metrics",
    "compute_reward",
    "observation_dim",
    "quat_wxyz_to_xyzw",
    "transform_keypoints",
    "unscale_joints",
]
