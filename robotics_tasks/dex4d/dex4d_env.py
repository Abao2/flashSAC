"""Isaac Lab port of Dex4D's official XArm6 + LEAP teacher task.

This module rewrites the simulator-facing lifecycle for ``DirectRLEnv`` while
keeping the legacy 22-D action, 1041-D observation, reward, goal switching,
reset, curriculum, push, and domain-randomization contracts.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.utils import math as math_utils

from .data import (
    assign_object_specs,
    load_object_observation_data,
    load_object_specs,
    prepare_sanitized_robot_urdf,
    resolve_dex4d_root,
    validate_object_specs,
)
from .dex4d_env_cfg import (
    Dex4DEnvCfg,
    Dex4DM6WujiEnvCfg,
    Dex4DM6WujiStage12EnvCfg,
    Dex4DStage12EnvCfg,
)
from .scene import get_robot_profile, resolve_robot_contract, setup_scene
from .task_math import (
    build_observation,
    compute_action_targets,
    compute_metrics,
    compute_reward,
    observation_dim,
    transform_keypoints,
)


class Dex4DEnv(DirectRLEnv):
    """Dex4D teacher-state task on Isaac Lab."""

    cfg: Dex4DEnvCfg

    def __init__(self, cfg: Dex4DEnvCfg, render_mode: str | None = None, **kwargs: Any) -> None:
        self._robot_profile = get_robot_profile(cfg.assets.robot_profile)
        expected_obs = observation_dim(
            self._robot_profile.num_joints,
            len(self._robot_profile.fingertip_body_names),
            cfg.num_keypoints,
        )
        if cfg.action_space != self._robot_profile.num_joints or cfg.observation_space != expected_obs:
            raise ValueError(
                f"Dex4D {self._robot_profile.name} space mismatch: "
                f"actions={cfg.action_space}/{self._robot_profile.num_joints}, "
                f"observations={cfg.observation_space}/{expected_obs}"
            )
        self._dex4d_root = resolve_dex4d_root(cfg.assets.dex4d_root)
        self._object_specs = load_object_specs(
            self._dex4d_root,
            cfg.assets.manifest,
            include_classes=cfg.assets.object_include_classes,
            exclude_classes=cfg.assets.object_exclude_classes,
            object_count=cfg.assets.object_count or None,
        )
        if not self._object_specs:
            raise ValueError("Dex4D object filter produced an empty asset set")
        self._assignment_indices, self._assigned_object_specs = assign_object_specs(
            cfg.scene.num_envs, self._object_specs
        )
        validate_object_specs(self._assigned_object_specs)
        self._sanitized_robot_urdf = prepare_sanitized_robot_urdf(
            self._dex4d_root,
            cfg.assets.robot_urdf,
            cfg.assets.cache_root,
            rename_numeric_leap_joints=self._robot_profile.rename_numeric_leap_joints,
        )

        super().__init__(cfg, render_mode, **kwargs)

        self._setup_goal_markers()

        contract = resolve_robot_contract(self)
        self._joint_ids = contract.joint_ids
        self._palm_body_id = contract.palm_body_id
        self._fingertip_body_ids = contract.fingertip_body_ids

        features, keypoints = load_object_observation_data(
            self._object_specs,
            self._assignment_indices,
            num_keypoints=cfg.num_keypoints,
            # Official keypoint seeds depend on object/scale IDs, not the RL seed.
            seed=0,
            cache_root=cfg.assets.cache_root,
        )
        self._visual_features = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        self._object_keypoints_local = torch.as_tensor(keypoints, dtype=torch.float32, device=self.device)
        if self._visual_features.shape != (self.num_envs, 64):
            raise RuntimeError(f"Dex4D feature shape mismatch: {tuple(self._visual_features.shape)}")
        expected_kp = (self.num_envs, cfg.num_keypoints, 3)
        if self._object_keypoints_local.shape != expected_kp:
            raise RuntimeError(
                f"Dex4D keypoint shape mismatch: {tuple(self._object_keypoints_local.shape)} != {expected_kp}"
            )

        self._allocate_buffers()
        self._capture_domain_randomization_defaults()
        print(
            "[Dex4D][IsaacLab] "
            f"profile={self._robot_profile.name} stage={cfg.stage} envs={self.num_envs} "
            f"objects={len(self._object_specs)} obs={cfg.observation_space} "
            f"actions={cfg.action_space} decimation={cfg.decimation} physics_dt={cfg.sim.dt:.8f}"
        )

    # ---------------------------------------------------------------------
    # Scene and buffers
    # ---------------------------------------------------------------------

    def _setup_scene(self) -> None:
        setup_scene(self)

    def _allocate_buffers(self) -> None:
        n = self.num_envs
        device = self.device
        self.actions = torch.zeros((n, self.cfg.action_space), dtype=torch.float32, device=device)
        self._prev_targets = torch.zeros_like(self.actions)
        self._cur_targets = torch.zeros_like(self.actions)
        self._last_joint_vel = torch.zeros_like(self.actions)
        self._goal_pos_l = torch.zeros((n, 3), dtype=torch.float32, device=device)
        self._goal_quat_w = torch.zeros((n, 4), dtype=torch.float32, device=device)
        self._goal_quat_w[:, 0] = 1.0
        self._achieved_steps = torch.zeros(n, dtype=torch.long, device=device)
        self._successes = torch.zeros(n, dtype=torch.float32, device=device)
        self._current_successes = torch.zeros(n, dtype=torch.float32, device=device)
        self._consecutive_successes = torch.zeros(n, dtype=torch.float32, device=device)
        self._pending_goal_reset = torch.zeros(n, dtype=torch.bool, device=device)
        self._steps_since_property_randomization = torch.full(
            (n,), self.cfg.domain_randomization.frequency_legacy_frames, dtype=torch.long, device=device
        )
        self._action_correlated_noise = torch.zeros_like(self.actions)
        self._observation_correlated_noise = torch.zeros(
            (n, self.cfg.observation_space), dtype=torch.float32, device=device
        )
        self._last_global_randomization_frame = -self.cfg.domain_randomization.frequency_legacy_frames
        self._noise_schedule_scale = 0.0
        self._cached_pre_reset_obs: torch.Tensor | None = None
        self._default_q = torch.tensor(self.cfg.robot_default_dof_pos, dtype=torch.float32, device=device)
        if self._default_q.numel() != self.cfg.action_space:
            raise ValueError(
                f"robot_default_dof_pos must have {self.cfg.action_space} entries, "
                f"got {self._default_q.numel()}"
            )

        rpy = torch.tensor(self._robot_profile.palm_body_offset_rpy, dtype=torch.float32, device=device)
        self._body_to_palm_quat = math_utils.quat_from_euler_xyz(rpy[0], rpy[1], rpy[2])
        local_offset = torch.tensor(
            self._robot_profile.palm_body_offset_xyz, dtype=torch.float32, device=device
        )
        self._body_to_palm_pos = math_utils.quat_apply(self._body_to_palm_quat, local_offset)
        self._palm_task_offset = torch.tensor(
            self._robot_profile.palm_task_offset, dtype=torch.float32, device=device
        )
        self._tip_offsets = torch.tensor(
            self._robot_profile.fingertip_offsets, dtype=torch.float32, device=device
        )

        self._update_curriculum(force=True)

    def _setup_goal_markers(self) -> None:
        # Markers are a viewport-only play aid. Creating a PointInstancer in a
        # headless Fabric stage is both useless and produces a shutdown warning.
        if not self.cfg.show_goal_marker or not self.sim.has_gui():
            return
        from isaaclab.markers import VisualizationMarkers
        from isaaclab.markers.config import FRAME_MARKER_CFG

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.prim_path = "/Visuals/Dex4DGoal"
        marker_cfg.markers = {"frame": marker_cfg.markers["frame"]}
        marker_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)
        self.goal_markers = VisualizationMarkers(marker_cfg)

    def _capture_domain_randomization_defaults(self) -> None:
        self._default_joint_limits = self.robot.data.joint_pos_limits.clone()
        self._default_joint_stiffness = self.robot.data.joint_stiffness.clone()
        self._default_joint_damping = self.robot.data.joint_damping.clone()
        self._default_robot_mass = self.robot.data.default_mass.clone()
        self._default_object_mass = self.object.data.default_mass.clone()
        self._default_robot_inertia = self.robot.data.default_inertia.clone()
        self._default_object_inertia = self.object.data.default_inertia.clone()
        self._default_robot_material = self.robot.root_physx_view.get_material_properties().clone()
        self._default_object_material = self.object.root_physx_view.get_material_properties().clone()

    # ---------------------------------------------------------------------
    # Reset and goal lifecycle
    # ---------------------------------------------------------------------

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        if env_ids is None:
            ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        else:
            ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if ids.numel() == 0:
            return

        super()._reset_idx(ids)
        self._maybe_refresh_global_randomization()
        self._randomize_physical_properties(ids)

        q = self._default_q.unsqueeze(0).expand(ids.numel(), -1).clone()
        qd = torch.zeros_like(q)
        if self.cfg.reset_dof_pos_noise > 0.0:
            limits = self._joint_limits(ids)
            delta_min = limits[..., 0] - q
            delta_max = limits[..., 1] - q
            u = torch.rand_like(q)
            q += self.cfg.reset_dof_pos_noise * (delta_min + u * (delta_max - delta_min))
        if self.cfg.reset_dof_vel_noise > 0.0:
            qd.uniform_(-self.cfg.reset_dof_vel_noise, self.cfg.reset_dof_vel_noise)
        self.robot.write_joint_state_to_sim(q, qd, joint_ids=self._joint_ids, env_ids=ids)
        self.robot.set_joint_position_target(q, joint_ids=self._joint_ids, env_ids=ids)
        self._prev_targets[ids] = q
        self._cur_targets[ids] = q
        self._last_joint_vel[ids] = qd

        self._reset_object(ids)
        self._reset_goal(ids, initial=True)
        self.actions[ids] = 0.0
        self._achieved_steps[ids] = 0
        self._successes[ids] = 0.0
        self._consecutive_successes[ids] = 0.0
        self._pending_goal_reset[ids] = False

    def _reset_object(self, env_ids: torch.Tensor) -> None:
        m = env_ids.numel()
        roll = torch.empty(m, device=self.device).uniform_(-math.pi, math.pi)
        pitch = torch.empty(m, device=self.device).uniform_(-math.pi, math.pi)
        yaw = torch.empty(m, device=self.device).uniform_(-math.pi, math.pi)
        stable = torch.rand(m, device=self.device) < self.cfg.object_init_stable_ratio
        roll = torch.where(stable, torch.full_like(roll, math.pi / 2.0), roll)
        pitch = torch.where(stable, torch.zeros_like(pitch), pitch)
        quat = math_utils.quat_from_euler_xyz(roll, pitch, yaw)

        pos_l = torch.zeros((m, 3), dtype=torch.float32, device=self.device)
        pos_l[:, 0].uniform_(*self.cfg.delta_x_range)
        pos_l[:, 1].uniform_(*self.cfg.delta_y_range)
        pos_l[:, 2] = 0.7
        state = self.object.data.default_root_state[env_ids].clone()
        state[:, :3] = pos_l + self.scene.env_origins[env_ids]
        state[:, 3:7] = quat
        state[:, 7:] = 0.0
        self.object.write_root_state_to_sim(state, env_ids=env_ids)

    def _reset_goal(self, env_ids: torch.Tensor, *, initial: bool) -> None:
        if env_ids.numel() == 0:
            return
        if initial:
            object_pos_l = self.object.data.root_pos_w[env_ids] - self.scene.env_origins[env_ids]
            self._goal_pos_l[env_ids] = object_pos_l
            self._goal_pos_l[env_ids, 2] += 0.2
            self._goal_quat_w[env_ids] = self.object.data.root_quat_w[env_ids]
        else:
            object_pos_l = self.object.data.root_pos_w[env_ids] - self.scene.env_origins[env_ids]
            pos = object_pos_l + torch.empty_like(object_pos_l).uniform_(-0.1, 0.1)
            lower = torch.tensor((-0.3, -0.5, 0.65), device=self.device)
            upper = torch.tensor((1.0, 0.5, 1.1), device=self.device)
            pos = torch.clamp(pos, lower, upper)

            axis = torch.randn((env_ids.numel(), 3), device=self.device)
            axis = axis / torch.clamp(torch.linalg.vector_norm(axis, dim=-1, keepdim=True), min=1.0e-8)
            angle = torch.rand(env_ids.numel(), device=self.device) * 0.5
            delta_q = math_utils.quat_from_angle_axis(angle, axis)
            quat = math_utils.quat_mul(delta_q, self.object.data.root_quat_w[env_ids])
            quat = quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True)

            stable = torch.rand(env_ids.numel(), device=self.device) < self._goal_stable_ratio
            stable_yaw = torch.empty(env_ids.numel(), device=self.device).uniform_(-math.pi, math.pi)
            stable_q = math_utils.quat_from_euler_xyz(
                torch.full_like(stable_yaw, math.pi / 2.0), torch.zeros_like(stable_yaw), stable_yaw
            )
            pos[stable, 2] = 0.7
            quat[stable] = stable_q[stable]
            self._goal_pos_l[env_ids] = pos
            self._goal_quat_w[env_ids] = quat

        if hasattr(self, "goal_markers"):
            self.goal_markers.visualize(
                translations=self._goal_pos_l + self.scene.env_origins,
                orientations=self._goal_quat_w,
            )

        self._achieved_steps[env_ids] = 0
        self._pending_goal_reset[env_ids] = False

    # ---------------------------------------------------------------------
    # Action and simulation hooks
    # ---------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        goal_ids = self._pending_goal_reset.nonzero(as_tuple=False).squeeze(-1)
        if goal_ids.numel() > 0:
            self._reset_goal(goal_ids, initial=False)

        action = actions.clone()
        if self.cfg.domain_randomization.enabled:
            dr = self.cfg.domain_randomization
            action += self._action_correlated_noise
            action += torch.randn_like(action) * (dr.action_noise_std * self._noise_schedule_scale)
        self.actions = action

        limits = self._joint_limits()
        self._cur_targets = compute_action_targets(
            action,
            self._prev_targets,
            limits[..., 0],
            limits[..., 1],
            speed_scale=self._arm_speed_scale,
            legacy_dt=self.cfg.legacy_sim_dt,
            moving_average=self.cfg.actions_moving_average,
            num_arm_dofs=self._robot_profile.num_arm_dofs,
            hand_moving_average=self.cfg.hand_actions_moving_average,
        )

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self._cur_targets, joint_ids=self._joint_ids)

    def _push_objects_if_due(self) -> None:
        if not self.cfg.push_objects:
            return
        interval = max(1, int(math.ceil(self.cfg.push_interval_s / self.step_dt)))
        if self.common_step_counter % interval != 0:
            return
        velocity = self.object.data.root_com_vel_w.clone()
        velocity[:, :2].uniform_(-self.cfg.max_push_linvel_xy, self.cfg.max_push_linvel_xy)
        velocity[:, 3:].uniform_(-self.cfg.max_push_angvel, self.cfg.max_push_angvel)
        self.object.write_root_com_velocity_to_sim(velocity)

    # ---------------------------------------------------------------------
    # State, observation, reward, and done
    # ---------------------------------------------------------------------

    def _joint_limits(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        limits = self.robot.data.joint_pos_limits
        if env_ids is not None:
            limits = limits[env_ids]
        return limits[:, self._joint_ids]

    def _compute_state(self) -> None:
        self._joint_pos = self.robot.data.joint_pos[:, self._joint_ids]
        self._joint_vel = self.robot.data.joint_vel[:, self._joint_ids]
        self._joint_force = self.robot.data.applied_torque[:, self._joint_ids]

        body_pos = self.robot.data.body_pos_w
        body_quat = self.robot.data.body_quat_w
        body_linvel = self.robot.data.body_lin_vel_w
        body_angvel = self.robot.data.body_ang_vel_w
        tips = self._fingertip_body_ids
        self._tip_raw_pos_l = body_pos[:, tips] - self.scene.env_origins[:, None, :]
        self._tip_raw_quat_w = body_quat[:, tips]
        self._tip_raw_linvel = body_linvel[:, tips]
        self._tip_raw_angvel = body_angvel[:, tips]
        self._tip_wrench = self.robot.data.body_incoming_joint_wrench_b[:, tips]

        palm_body_pos = body_pos[:, self._palm_body_id] - self.scene.env_origins
        palm_body_quat = body_quat[:, self._palm_body_id]
        palm_offset = self._body_to_palm_pos.expand(self.num_envs, -1)
        world_offset = math_utils.quat_apply(palm_body_quat, palm_offset)
        self._palm_raw_pos_l = palm_body_pos + world_offset
        palm_rel_q = self._body_to_palm_quat.expand(self.num_envs, -1)
        self._palm_raw_quat_w = math_utils.quat_mul(palm_body_quat, palm_rel_q)
        palm_body_lin = body_linvel[:, self._palm_body_id]
        palm_body_ang = body_angvel[:, self._palm_body_id]
        self._palm_raw_linvel = palm_body_lin + torch.linalg.cross(
            palm_body_ang, world_offset, dim=-1
        )
        self._palm_raw_angvel = palm_body_ang

        palm_task_offset = self._palm_task_offset.expand(self.num_envs, -1)
        self._palm_pos_l = self._palm_raw_pos_l + math_utils.quat_apply(
            self._palm_raw_quat_w, palm_task_offset
        )
        num_tips = len(self._fingertip_body_ids)
        self._tip_pos_l = self._tip_raw_pos_l + math_utils.quat_apply(
            self._tip_raw_quat_w.reshape(-1, 4),
            self._tip_offsets[None].expand(self.num_envs, -1, -1).reshape(-1, 3),
        ).reshape(self.num_envs, num_tips, 3)

        self._object_pos_l = self.object.data.root_pos_w - self.scene.env_origins
        self._object_quat_w = self.object.data.root_quat_w
        self._object_linvel = self.object.data.root_lin_vel_w
        self._object_angvel = self.object.data.root_ang_vel_w
        self._object_keypoints = transform_keypoints(
            self._object_keypoints_local, self._object_pos_l, self._object_quat_w
        )
        self._goal_keypoints = transform_keypoints(
            self._object_keypoints_local, self._goal_pos_l, self._goal_quat_w
        )
        self._metrics = compute_metrics(
            object_keypoints=self._object_keypoints,
            goal_keypoints=self._goal_keypoints,
            object_pos=self._object_pos_l,
            palm_pos=self._palm_pos_l,
            fingertip_positions=self._tip_pos_l,
            hand_joint_pos=self._joint_pos[:, self._robot_profile.num_arm_dofs :],
            finger_distance_scale=4.0 / len(self._fingertip_body_ids),
        )
        self._goal_distance = self._metrics["goal_obj_dist"]
        self._object_hand_distance = self._metrics["obj_hand_dist"]
        self._object_finger_distance = self._metrics["obj_finger_dist"]

    def _build_observation(self, *, add_noise: bool = True) -> torch.Tensor:
        self._compute_state()
        limits = self._joint_limits()
        fingertip_states = torch.cat(
            (
                self._tip_raw_pos_l,
                self._tip_raw_quat_w,
                self._tip_raw_linvel,
                self._tip_raw_angvel,
            ),
            dim=-1,
        )
        palm_state = torch.cat(
            (
                self._palm_raw_pos_l,
                self._palm_raw_quat_w,
                self._palm_raw_linvel,
                self._palm_raw_angvel,
            ),
            dim=-1,
        )
        obs = build_observation(
            joint_pos=self._joint_pos,
            joint_vel=self._joint_vel,
            joint_force=self._joint_force,
            joint_lower=limits[..., 0],
            joint_upper=limits[..., 1],
            fingertip_states_wxyz=fingertip_states,
            fingertip_wrenches=self._tip_wrench,
            fingertip_positions=self._tip_pos_l,
            palm_state_wxyz=palm_state,
            actions=self.actions,
            object_pos=self._object_pos_l,
            object_quat_wxyz=self._object_quat_w,
            object_linvel=self._object_linvel,
            object_angvel=self._object_angvel,
            goal_pos=self._goal_pos_l,
            goal_quat_wxyz=self._goal_quat_w,
            local_keypoints=self._object_keypoints_local,
            visual_features=self._visual_features,
        )

        if add_noise and self.cfg.domain_randomization.enabled:
            dr = self.cfg.domain_randomization
            obs += self._observation_correlated_noise
            obs += torch.randn_like(obs) * (dr.observation_noise_std * self._noise_schedule_scale)
        return torch.clamp(obs, -self.cfg.clip_observations, self.cfg.clip_observations)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        fresh = self._build_observation(add_noise=True)
        if self._cached_pre_reset_obs is not None:
            # Reuse the exact pre-reset noisy observation for continuing envs;
            # only reset envs receive the freshly constructed initial state.
            obs = torch.where(self.reset_buf[:, None], fresh, self._cached_pre_reset_obs)
            self._cached_pre_reset_obs = None
        else:
            obs = fresh
        return {"policy": obs}

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._update_curriculum()
        self._push_objects_if_due()
        self._compute_state()
        timeout = self.episode_length_buf >= self.cfg.episode_steps
        too_far = self._goal_distance >= self._too_far_threshold
        self._time_out = timeout
        self._too_far = too_far
        # Legacy Dex4D exposes one done mask for both causes. Returning both as
        # terminations prevents off-policy timeout bootstrapping semantic drift.
        return timeout | too_far, torch.zeros_like(timeout)

    def _get_rewards(self) -> torch.Tensor:
        # _get_dones() has refreshed all state tensors.
        self._achieved_steps = torch.where(
            self._goal_distance <= self.cfg.success_threshold,
            self._achieved_steps + 1,
            torch.zeros_like(self._achieved_steps),
        )
        reward, scaled_terms = compute_reward(
            self._metrics,
            self.actions,
            self._achieved_steps,
            self.cfg.rewards,
            success_threshold=self.cfg.success_threshold,
            achieved_steps_required=self.cfg.achieved_steps,
        )
        close_flag = self._metrics["grasped"]
        fall_penalty = torch.where(self._goal_distance >= self._too_far_threshold, -5.0, 0.0)
        hand_vel_penalty = torch.where(
            close_flag & (self._goal_distance <= self.cfg.success_threshold),
            -0.1 * torch.square(self._palm_raw_linvel).sum(dim=-1),
            0.0,
        )
        dof_vel = torch.clamp(-1.0e-3 * torch.square(self._joint_vel).sum(dim=-1), -0.5, 0.0)
        dof_acc_value = (self._last_joint_vel - self._joint_vel) / self.cfg.legacy_sim_dt
        dof_acc = torch.clamp(-1.0e-8 * torch.square(dof_acc_value).sum(dim=-1), -0.5, 0.0)

        optional_terms = {
            "fall_penalty": fall_penalty,
            "hand_vel_penalty": hand_vel_penalty,
            "dof_vel": dof_vel,
            "dof_acc": dof_acc,
        }
        for name, value in optional_terms.items():
            scale = float(getattr(self.cfg.rewards, name))
            if scale != 0.0:
                scaled_terms[name] = scale * value
                reward += scaled_terms[name]

        reached = self._goal_distance <= self.cfg.success_threshold
        completed_goal = self._achieved_steps >= self.cfg.achieved_steps
        self._successes = torch.where(reached, torch.ones_like(self._successes), self._successes)
        self._consecutive_successes += completed_goal.to(torch.float32)
        self._pending_goal_reset |= completed_goal
        self._achieved_steps = torch.where(completed_goal, torch.zeros_like(self._achieved_steps), self._achieved_steps)
        self._current_successes = torch.where(self.reset_buf, self._successes, self._current_successes)

        self.extras["episode_cumulative"] = {
            "reward": {name: value for name, value in scaled_terms.items()},
            "goal_distance": self._goal_distance,
        }
        self.extras["episode_final"] = {
            "success": self._successes.clone(),
            "completed_goals": self._consecutive_successes.clone(),
            "goal_distance": self._goal_distance.clone(),
            "too_far": self._too_far.to(torch.float32),
            "time_out": self._time_out.to(torch.float32),
        }
        self.extras["successes"] = self._successes
        self.extras["current_successes"] = self._current_successes
        self.extras["consecutive_successes"] = self._consecutive_successes
        self.extras["curriculum_mature"] = float(self._curriculum_mature)

        # Capture a genuine terminal observation before DirectRLEnv auto-reset.
        terminal_obs = self._build_observation(add_noise=True)
        self._cached_pre_reset_obs = terminal_obs
        self.extras["final_obs"] = terminal_obs

        self._prev_targets.copy_(self._cur_targets)
        self._last_joint_vel.copy_(self._joint_vel)
        self._steps_since_property_randomization += 1
        return reward

    # ---------------------------------------------------------------------
    # Curriculum and domain randomization
    # ---------------------------------------------------------------------

    def _update_curriculum(self, *, force: bool = False) -> None:
        mature = bool(self.cfg.curriculum_mature_at_start)
        if self.cfg.curriculum and self.common_step_counter >= self.cfg.curriculum_policy_step:
            mature = True
        if not force and getattr(self, "_curriculum_mature", None) == mature:
            return
        self._curriculum_mature = mature
        self._too_far_threshold = 1.0e6 if mature else self.cfg.too_far_reset_threshold
        self._goal_stable_ratio = 0.2 if mature else self.cfg.goal_reset_stable_ratio
        self._arm_speed_scale = 1.5 if mature else self.cfg.dof_speed_scale
        print(
            "[Dex4D][curriculum] "
            f"policy_step={getattr(self, 'common_step_counter', 0)} mature={mature} "
            f"far={self._too_far_threshold:g} goal_stable={self._goal_stable_ratio:g} "
            f"arm_speed={self._arm_speed_scale:g}"
        )

    def _legacy_frame_count(self) -> int:
        return int(self.common_step_counter * self.cfg.legacy_control_decimation)

    def _maybe_refresh_global_randomization(self) -> None:
        dr = self.cfg.domain_randomization
        if not dr.enabled:
            self._noise_schedule_scale = 0.0
            return
        frame = self._legacy_frame_count()
        if frame - self._last_global_randomization_frame < dr.frequency_legacy_frames:
            return
        self._last_global_randomization_frame = frame
        self._noise_schedule_scale = min(frame / max(1, dr.noise_schedule_legacy_frames), 1.0)
        self._action_correlated_noise = torch.randn_like(self.actions) * (
            dr.action_correlated_std * self._noise_schedule_scale
        )
        self._observation_correlated_noise = torch.randn_like(self._observation_correlated_noise) * (
            dr.observation_correlated_std * self._noise_schedule_scale
        )
        gravity_noise = float(torch.randn((), device=self.device).item()) * (
            dr.gravity_noise_std * self._noise_schedule_scale
        )
        try:
            import carb

            self.sim.physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, -9.81 + gravity_noise))
        except (AttributeError, RuntimeError):
            # Gravity randomization is diagnostic, not a reason to leave a GPU
            # simulation half-initialized on older Isaac Lab builds.
            pass

    def _randomize_physical_properties(self, env_ids: torch.Tensor) -> None:
        dr = self.cfg.domain_randomization
        if not dr.enabled:
            return
        eligible = self._steps_since_property_randomization[env_ids] >= dr.frequency_legacy_frames
        ids = env_ids[eligible]
        if ids.numel() == 0:
            return
        scale = min(self._legacy_frame_count() / max(1, dr.property_schedule_legacy_frames), 1.0)
        ids_cpu = ids.cpu()

        def scheduled_uniform(bounds: tuple[float, float], shape: tuple[int, ...], *, cpu: bool = False):
            lo = 1.0 + (bounds[0] - 1.0) * scale
            hi = 1.0 + (bounds[1] - 1.0) * scale
            device = "cpu" if cpu else self.device
            return torch.empty(shape, device=device).uniform_(lo, hi)

        # Joint gains and limits. Isaac Lab's PhysX setters consume CPU tensors.
        stiffness = self._default_joint_stiffness[ids].clone()
        damping = self._default_joint_damping[ids].clone()
        stiffness *= scheduled_uniform(dr.robot_gain_scale_range, stiffness.shape)
        damping *= scheduled_uniform(dr.robot_gain_scale_range, damping.shape)
        self.robot.write_joint_stiffness_to_sim(stiffness, env_ids=ids)
        self.robot.write_joint_damping_to_sim(damping, env_ids=ids)
        limits = self._default_joint_limits[ids].clone()
        if scale > 0.0:
            limits += torch.randn_like(limits) * (dr.joint_limit_noise_std * scale)
            invalid = limits[..., 0] >= limits[..., 1]
            limits[..., 0] = torch.where(invalid, self._default_joint_limits[ids][..., 0], limits[..., 0])
            limits[..., 1] = torch.where(invalid, self._default_joint_limits[ids][..., 1], limits[..., 1])
        self.robot.write_joint_position_limit_to_sim(limits, env_ids=ids, warn_limit_violation=False)

        robot_mass = self.robot.root_physx_view.get_masses()
        robot_factor = scheduled_uniform(
            dr.robot_mass_scale_range, (ids.numel(), self.robot.num_bodies), cpu=True
        )
        robot_mass[ids_cpu] = self._default_robot_mass.cpu()[ids_cpu] * robot_factor
        self.robot.root_physx_view.set_masses(robot_mass, ids_cpu)
        robot_inertia = self.robot.root_physx_view.get_inertias()
        robot_inertia[ids_cpu] = self._default_robot_inertia.cpu()[ids_cpu] * robot_factor[..., None]
        self.robot.root_physx_view.set_inertias(robot_inertia, ids_cpu)

        object_mass = self.object.root_physx_view.get_masses()
        object_factor = scheduled_uniform(dr.object_mass_scale_range, (ids.numel(), 1), cpu=True)
        object_mass[ids_cpu] = self._default_object_mass.cpu()[ids_cpu] * object_factor
        self.object.root_physx_view.set_masses(object_mass, ids_cpu)
        object_inertia = self.object.root_physx_view.get_inertias()
        object_inertia[ids_cpu] = self._default_object_inertia.cpu()[ids_cpu] * object_factor
        self.object.root_physx_view.set_inertias(object_inertia, ids_cpu)

        self._randomize_friction(
            self.robot,
            self._default_robot_material,
            ids,
            dr.robot_friction_scale_range,
            scale,
            dr.friction_buckets,
        )
        self._randomize_friction(
            self.object,
            self._default_object_material,
            ids,
            dr.object_friction_scale_range,
            scale,
            dr.friction_buckets,
        )
        self._steps_since_property_randomization[ids] = 0

    @staticmethod
    def _randomize_friction(asset, defaults, env_ids, bounds, schedule_scale: float, num_buckets: int) -> None:
        ids_cpu = env_ids.cpu()
        materials = asset.root_physx_view.get_material_properties()
        lo = 1.0 + (bounds[0] - 1.0) * schedule_scale
        hi = 1.0 + (bounds[1] - 1.0) * schedule_scale
        buckets = torch.empty(max(1, num_buckets), device="cpu").uniform_(lo, hi)
        bucket_ids = torch.randint(
            len(buckets), (env_ids.numel(), materials.shape[1]), device="cpu"
        )
        factor = buckets[bucket_ids]
        materials[ids_cpu] = defaults.cpu()[ids_cpu]
        materials[ids_cpu, :, 0] *= factor
        materials[ids_cpu, :, 1] *= factor
        asset.root_physx_view.set_material_properties(materials, ids_cpu)


__all__ = ["Dex4DEnv", "Dex4DEnvCfg", "Dex4DStage12EnvCfg"]
