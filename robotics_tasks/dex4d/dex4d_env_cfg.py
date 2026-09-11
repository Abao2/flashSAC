"""Configuration for the Dex4D XArm6 + LEAP Isaac Lab task.

The values mirror Dex4D's official ``xarm6_leap_hand_ap2ap_stage_{1_2,3}.yaml``.
Isaac Gym's ``dt=1/60, substeps=2`` is represented as 120 Hz Isaac Lab physics;
therefore legacy control decimation 2/12 becomes Isaac Lab decimation 4/24.
"""

from __future__ import annotations

from pathlib import Path

from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass


def _sim_cfg(render_interval: int = 24) -> SimulationCfg:
    return SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=render_interval,
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            solver_type=1,
            min_position_iteration_count=8,
            max_position_iteration_count=8,
            min_velocity_iteration_count=0,
            max_velocity_iteration_count=0,
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=8_388_608,
            gpu_max_rigid_patch_count=2_097_152,
        ),
    )


@configclass
class Dex4DAssetsCfg:
    """External official assets; large datasets stay outside this repository."""

    dex4d_root: str = ""
    robot_profile: str = "xarm6_leap"
    manifest: str = "dex4d_policy/dex4d/cfg/train_set.yaml"
    robot_urdf: str = "dex4d_policy/assets/urdf/xarm6_leap_description/xarm6_leap_right_2023.urdf"
    object_include_classes: tuple[str, ...] | None = None
    object_exclude_classes: tuple[str, ...] | None = None
    # Isaac Lab's configclass validates overrides against the concrete default
    # type.  Use zero as the "all objects" sentinel so Hydra can set an int.
    object_count: int = 0
    cache_root: str = ""


@configclass
class Dex4DRewardCfg:
    obj_finger: float = 0.0
    obj_hand: float = 0.0
    goal_obj: float = 1.0
    success_bonus: float = 1.0
    terminal_bonus: float = 1.0
    finger_curl_reg: float = 10.0
    table_collision_penalty: float = 1.0
    fall_penalty: float = 0.0
    hand_vel_penalty: float = 0.0
    dof_vel: float = 0.0
    dof_acc: float = 0.0
    action_penalty: float = 5.0


@configclass
class Dex4DDomainRandomizationCfg:
    enabled: bool = True
    frequency_legacy_frames: int = 720
    observation_noise_std: float = 0.002
    observation_correlated_std: float = 0.001
    action_noise_std: float = 0.05
    action_correlated_std: float = 0.015
    noise_schedule_legacy_frames: int = 40_000
    property_schedule_legacy_frames: int = 30_000
    robot_gain_scale_range: tuple[float, float] = (0.9, 1.1)
    joint_limit_noise_std: float = 0.01
    robot_mass_scale_range: tuple[float, float] = (0.5, 1.5)
    object_mass_scale_range: tuple[float, float] = (0.5, 1.5)
    robot_friction_scale_range: tuple[float, float] = (0.7, 1.3)
    object_friction_scale_range: tuple[float, float] = (0.7, 1.3)
    friction_buckets: int = 250
    gravity_noise_std: float = 0.4


@configclass
class Dex4DEnvCfg(DirectRLEnvCfg):
    """Official stage-3/all-category teacher-state task."""

    # 12 legacy 60 Hz frames, each represented by two 120 Hz Isaac Lab steps.
    decimation: int = 24
    episode_length_s: float = 80.0  # 400 policy steps at 5 Hz
    action_space: int = 22
    observation_space: int = 1041
    state_space: None = None
    is_finite_horizon: bool = True

    sim: SimulationCfg = _sim_cfg(render_interval=24)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=1.5,
        replicate_physics=False,
        clone_in_fabric=False,
    )
    viewer: ViewerCfg = ViewerCfg(
        eye=(1.0, -1.2, 1.3),
        lookat=(0.0, 0.0, 0.75),
        resolution=(960, 720),
    )

    assets: Dex4DAssetsCfg = Dex4DAssetsCfg()
    rewards: Dex4DRewardCfg = Dex4DRewardCfg()
    domain_randomization: Dex4DDomainRandomizationCfg = Dex4DDomainRandomizationCfg()

    stage: str = "stage3"
    legacy_sim_dt: float = 1.0 / 60.0
    legacy_control_decimation: int = 12
    episode_steps: int = 400
    num_keypoints: int = 128
    clip_observations: float = 5.0
    robot_default_dof_pos: tuple[float, ...] = (
        0.0, -0.52, -0.70, 0.0, 1.22, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    )
    dof_speed_scale: float = 10.0
    actions_moving_average: float = 1.0
    hand_actions_moving_average: float = 1.0
    reset_dof_pos_noise: float = 0.0
    reset_dof_vel_noise: float = 0.0
    object_init_stable_ratio: float = 0.2
    goal_reset_stable_ratio: float = 0.1
    delta_x_range: tuple[float, float] = (-0.1, 0.1)
    delta_y_range: tuple[float, float] = (-0.1, 0.1)
    success_threshold: float = 0.05
    achieved_steps: int = 30
    too_far_reset_threshold: float = 0.3
    push_objects: bool = True
    push_interval_s: float = 4.0
    max_push_linvel_xy: float = 0.2
    max_push_angvel: float = 0.2
    curriculum: bool = True
    # PPO updates the task once per 8-step rollout; iteration 15,000 means
    # 120,000 vector environment steps, independent of num_envs.
    curriculum_policy_step: int = 120_000
    curriculum_mature_at_start: bool = True
    # Visual-only coordinate frame. The goal remains tensor-only and cannot
    # collide with the hand or object.
    show_goal_marker: bool = False


@configclass
class Dex4DStage12EnvCfg(Dex4DEnvCfg):
    """Official stage-1/2 bottle-only teacher-state task."""

    decimation: int = 4
    episode_length_s: float = 400.0 * 4.0 / 120.0
    sim: SimulationCfg = _sim_cfg(render_interval=4)
    stage: str = "stage1_2"
    legacy_control_decimation: int = 2
    robot_default_dof_pos: tuple[float, ...] = (
        0.0, 0.38, -1.37, 0.0, 0.99, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    )
    assets: Dex4DAssetsCfg = Dex4DAssetsCfg(object_include_classes=("bottle",))
    rewards: Dex4DRewardCfg = Dex4DRewardCfg(
        obj_finger=1.0,
        obj_hand=1.0,
        finger_curl_reg=1.0,
        action_penalty=1.0,
    )
    curriculum_mature_at_start: bool = False


_M6_WUJI_URDF = str(
    Path(__file__).resolve().parents[2]
    / "robotics_assets/urdf/wuji_m6_description/generated/wuji_m6_left.urdf"
)
_M6_WUJI_DEFAULT_Q = (
    1.7, -1.1, -1.1, -2.0, -0.37, 0.13, 0.55,
    0.06, 0.0, 0.0, 0.0,
    0.06, 0.0, 0.0, 0.0,
    0.06, 0.0, 0.0, 0.0,
    0.06, 0.0, 0.0, 0.0,
    0.06, 0.0, 0.0, 0.0,
)


@configclass
class Dex4DM6WujiAssetsCfg(Dex4DAssetsCfg):
    """Validated M6-696-V4 + Wuji generation-one left-hand asset."""

    robot_profile: str = "m6_wuji_left"
    robot_urdf: str = _M6_WUJI_URDF


@configclass
class Dex4DM6WujiEnvCfg(Dex4DEnvCfg):
    """Dex4D stage-3 task with the 7+20 DoF M6/Wuji robot."""

    action_space: int = 27
    observation_space: int = 1083
    assets: Dex4DM6WujiAssetsCfg = Dex4DM6WujiAssetsCfg()
    robot_default_dof_pos: tuple[float, ...] = _M6_WUJI_DEFAULT_Q
    # Stage 3 runs at 5 Hz; a full-scale action changes the target by 0.1 rad.
    dof_speed_scale: float = 6.0
    # Wuji actions are absolute targets; smooth them to respect joint velocity limits.
    hand_actions_moving_average: float = 0.07


@configclass
class Dex4DM6WujiStage12EnvCfg(Dex4DStage12EnvCfg):
    """Dex4D stage-1/2 bottle task with the 7+20 DoF M6/Wuji robot."""

    action_space: int = 27
    observation_space: int = 1083
    assets: Dex4DM6WujiAssetsCfg = Dex4DM6WujiAssetsCfg(
        object_include_classes=("bottle",)
    )
    robot_default_dof_pos: tuple[float, ...] = _M6_WUJI_DEFAULT_Q
    # Stage 1/2 runs at 30 Hz; a full-scale action changes the target by 0.1 rad.
    dof_speed_scale: float = 6.0
    hand_actions_moving_average: float = 0.07


__all__ = [
    "Dex4DEnvCfg",
    "Dex4DStage12EnvCfg",
    "Dex4DM6WujiEnvCfg",
    "Dex4DM6WujiStage12EnvCfg",
    "Dex4DAssetsCfg",
    "Dex4DM6WujiAssetsCfg",
    "Dex4DRewardCfg",
    "Dex4DDomainRandomizationCfg",
]
