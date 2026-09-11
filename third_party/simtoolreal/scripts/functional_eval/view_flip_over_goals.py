"""View the official flat-spatula flip_over goals without loading a policy."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
TRAJECTORY_PATH = (
    REPO_ROOT
    / "dextoolbench/trajectories/spatula/flat_spatula/flip_over.json"
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--waypoint_seconds",
    type=float,
    default=1.0,
    help="Wall-clock seconds to display each goal waypoint.",
)
parser.add_argument(
    "--steps",
    type=int,
    default=0,
    help="Render iterations; 0 keeps the GUI open until closed.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.waypoint_seconds <= 0.0:
    parser.error("--waypoint_seconds must be positive")

simulation_app = AppLauncher(args).app


import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
import isaacsimenvs  # noqa: F401  (registers the environment)
from dextoolbench.objects import NAME_TO_OBJECT
from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg
from isaaclab.sim.utils import get_current_stage
from pxr import UsdGeom


def main() -> None:
    with open(TRAJECTORY_PATH) as f:
        trajectory = json.load(f)
    goals_xyzw = trajectory["goals"]

    # SimToolReal's fixed-trajectory loader expects batched wxyz quaternions.
    converted = {
        "pos": [[goal[:3] for goal in goals_xyzw]],
        "quat_wxyz": [
            [[goal[6], goal[3], goal[4], goal[5]] for goal in goals_xyzw]
        ],
    }
    tmp_dir = tempfile.TemporaryDirectory(prefix="flip_over_viewer_")
    converted_path = Path(tmp_dir.name) / "trajectory_isaac_format.json"
    converted_path.write_text(json.dumps(converted))

    obj = NAME_TO_OBJECT["flat_spatula"]
    cfg = SimToolRealEnvCfg()
    cfg.scene.num_envs = 1
    cfg.assets.enable_functional_object = True
    cfg.assets.object_urdf = str(obj.decomposed_urdf_path)
    cfg.assets.object_scale = tuple(obj.scale)
    cfg.assets.table_urdf = str(REPO_ROOT / "assets/urdf/table_narrow.urdf")

    reset = cfg.reset
    reset.reset_position_noise_x = 0.0
    reset.reset_position_noise_y = 0.0
    reset.reset_position_noise_z = 0.0
    reset.reset_dof_pos_random_interval_arm = 0.0
    reset.reset_dof_pos_random_interval_fingers = 0.0
    reset.reset_dof_vel_random_interval = 0.0
    reset.table_reset_z = 0.38
    reset.table_reset_z_range = 0.0
    reset.start_arm_higher = True
    start = trajectory["start_pose"].copy()
    start[2] += 0.03  # Same table-clearance offset as official evaluation.
    reset.fixed_start_pose = (
        start[0], start[1], start[2], start[6], start[3], start[4], start[5]
    )
    reset.fixed_trajectory_file = str(converted_path)

    domain_rand = cfg.domain_randomization
    domain_rand.use_obs_delay = False
    domain_rand.use_action_delay = False
    domain_rand.use_object_state_delay_noise = False
    domain_rand.object_scale_noise_multiplier_range = (1.0, 1.0)
    domain_rand.joint_velocity_obs_noise_std = 0.0
    domain_rand.force_scale = 0.0
    domain_rand.torque_scale = 0.0
    domain_rand.force_prob_range = (0.0001, 0.0001)
    domain_rand.torque_prob_range = (0.0001, 0.0001)
    cfg.termination.max_consecutive_successes = len(goals_xyzw)

    env = gym.make("Isaacsimenvs-SimToolReal-Direct-v0", cfg=cfg)
    inner = env.unwrapped
    env.reset()

    # MultiUsdFileCfg builds through a temporary template, so bind the viewer-only
    # material after cloning, directly on the final GoalViz prim.
    goal_viz_path = "/World/envs/env_0/GoalViz"
    goal_material_path = "/World/Looks/FlipOverGoalGreen"
    goal_material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.0, 1.0, 0.0), opacity=1.0
    )
    goal_material.func(goal_material_path, goal_material)
    sim_utils.bind_visual_material(goal_viz_path, goal_material_path)
    goal_viz_prim = get_current_stage().GetPrimAtPath(goal_viz_path)
    UsdGeom.Imageable(goal_viz_prim).MakeVisible()
    print(
        f"[flip-over-viewer] GoalViz prim valid={goal_viz_prim.IsValid()} "
        f"visibility={UsdGeom.Imageable(goal_viz_prim).ComputeVisibility()}"
    )

    goal_pos = inner._fixed_traj_pos[0]
    goal_quat = inner._fixed_traj_quat[0]
    assert goal_pos.shape == (45, 3)
    env_origin = inner.scene.env_origins[0]
    goal_pos_w = goal_pos + env_origin

    waypoint_marker_cfg = sim_utils.SphereCfg(
        radius=0.006,
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(1.0, 0.8, 0.0)
        ),
    )
    for waypoint_idx, position in enumerate(goal_pos_w.cpu().tolist()):
        waypoint_marker_cfg.func(
            f"/Visuals/FlipOverPath/Waypoint_{waypoint_idx:02d}",
            waypoint_marker_cfg,
            translation=tuple(position),
        )
    inner.sim.set_camera_view(
        eye=(0.55, -0.65, 0.95), target=(0.03, 0.05, 0.68)
    )

    print(f"[flip-over-viewer] source={TRAJECTORY_PATH}")
    print(f"[flip-over-viewer] loaded {len(goals_xyzw)} waypoints")
    print(
        "[flip-over-viewer] yellow dots=all tool-root positions; "
        "opaque green GoalViz=current tool goal; orange disk=pancake"
    )
    print(
        f"[flip-over-viewer] each waypoint lasts {args.waypoint_seconds:.2f} "
        "wall-clock seconds; physics is paused"
    )

    start_wall = time.perf_counter()
    previous_idx = -1
    step = 0
    while simulation_app.is_running() and (args.steps <= 0 or step < args.steps):
        wall_elapsed = time.perf_counter() - start_wall
        waypoint_idx = int(wall_elapsed / args.waypoint_seconds) % len(goals_xyzw)
        pose_w = torch.cat(
            (goal_pos_w[waypoint_idx], goal_quat[waypoint_idx])
        ).unsqueeze(0)
        inner.goal_viz.write_root_pose_to_sim(pose_w)
        inner.sim.forward()
        inner.sim.render()

        if waypoint_idx != previous_idx:
            local_pose = torch.cat(
                (goal_pos[waypoint_idx], goal_quat[waypoint_idx])
            )
            print(
                f"[flip-over-viewer] wall_t={wall_elapsed:6.2f}s "
                f"waypoint={waypoint_idx + 1:02d}/{len(goals_xyzw)} "
                f"pose_xyzw_source={goals_xyzw[waypoint_idx]} "
                f"pose_wxyz_loaded={local_pose.cpu().tolist()}"
            )
            previous_idx = waypoint_idx
        step += 1
        if not args.headless:
            time.sleep(1.0 / 60.0)

    print(f"[flip-over-viewer] stopped after {step} render iterations")
    env.close()
    simulation_app.close()
    tmp_dir.cleanup()


if __name__ == "__main__":
    main()
