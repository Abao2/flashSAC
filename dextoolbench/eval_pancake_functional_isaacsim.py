"""Roll out the official checkpoint with an external pancake evaluator.

Phase 1 supports ``scoop_and_lift`` only.  The policy still receives the
unaltered SimToolReal observation; pancake state is simulator ground truth
read after each physics step and is used only for evaluation.

Example (headless)::

    .venv_isaacsim/bin/python dextoolbench/eval_pancake_functional_isaacsim.py \
      --task scoop_and_lift --num_episodes 1

Add ``--play`` to open the native Isaac Lab GUI.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
CONTROL_HZ = 60.0
SCOOP_GOAL_COUNT = 18


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=("scoop_and_lift", "serve_to_plate", "flip_and_settle"),
        default="scoop_and_lift",
    )
    parser.add_argument(
        "--config_path",
        "--config-path",
        dest="config_path",
        default=str(REPO_ROOT / "pretrained_policy/config.yaml"),
    )
    parser.add_argument(
        "--checkpoint_path",
        "--checkpoint-path",
        dest="checkpoint_path",
        default=str(REPO_ROOT / "pretrained_policy/model.pth"),
    )
    parser.add_argument("--num_episodes", "--num-episodes", type=int, default=1)
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        dest="output_dir",
        default=str(REPO_ROOT / "evals_pancake/scoop_and_lift"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--goal_z_offset",
        "--goal-z-offset",
        type=float,
        default=0.0,
        help="Add this many metres to every goal Z (negative lowers the path).",
    )
    parser.add_argument(
        "--pan_height",
        "--pan-height",
        type=float,
        default=0.0,
        help="Raise the pancake on a fixed circular support by this many metres.",
    )
    parser.add_argument(
        "--support_radius",
        "--support-radius",
        type=float,
        default=0.11,
        help="Radius in metres of the circular support below the pancake.",
    )
    parser.add_argument(
        "--support_half_spacing",
        "--support-half-spacing",
        type=float,
        default=None,
        help=(
            "Place two supports at center Y +/- this distance. "
            "Zero uses one centered support; the pedestal trajectory default "
            "comes from its JSON metadata."
        ),
    )
    parser.add_argument(
        "--pancake_x",
        "--pancake-x",
        type=float,
        default=None,
        help="Override the trajectory's recommended pancake center X.",
    )
    parser.add_argument(
        "--pancake_y",
        "--pancake-y",
        type=float,
        default=None,
        help="Override the trajectory's recommended pancake center Y.",
    )
    parser.add_argument(
        "--goal_sequence",
        choices=(
            "pedestal_scoop",
            "pitched_scoop",
            "aligned_scoop",
            "official_prefix",
        ),
        default="pitched_scoop",
        help=(
            "Use the pitched insertion path, the level contact diagnostic, "
            "or the original non-contacting diagnostic."
        ),
    )
    parser.add_argument("--rl_device", default="cuda")
    parser.add_argument("--settle_physics_steps", type=int, default=60)
    parser.add_argument("--max_episode_steps", type=int, default=3600)
    parser.add_argument("--post_trajectory_settle_steps", type=int, default=60)
    parser.add_argument("--drop_height", type=float, default=0.48)
    parser.add_argument("--max_blade_pancake_distance", type=float, default=0.08)
    parser.add_argument(
        "--blade_local_offset",
        type=float,
        nargs=3,
        default=(0.166, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument("--log_interval", type=int, default=30)
    parser.add_argument(
        "--play",
        action="store_true",
        help="Open the native GUI and pace rollout at the policy's 60 Hz.",
    )
    parser.add_argument(
        "--keep-open",
        action="store_true",
        help=(
            "In GUI mode, finish every waypoint and keep the final frame open "
            "until the window is closed."
        ),
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _launch_app():
    parser = _build_parser()
    args = parser.parse_args()
    if args.task != "scoop_and_lift":
        parser.error("Phase 1 currently implements --task scoop_and_lift only")
    if args.num_episodes <= 0:
        parser.error("--num_episodes must be positive")
    if args.max_episode_steps <= 0:
        parser.error("--max_episode_steps must be positive")
    if args.pan_height < 0.0:
        parser.error("--pan_height must be non-negative")
    if args.support_radius <= 0.0:
        parser.error("--support_radius must be positive")
    if args.support_half_spacing is not None and args.support_half_spacing < 0.0:
        parser.error("--support_half_spacing must be non-negative")
    args.headless = not args.play
    return AppLauncher(args).app, args


_app, _args = _launch_app()


def _write_trajectory(goals_xyzw, directory: Path) -> str:
    payload = {
        "pos": [[goal[:3] for goal in goals_xyzw]],
        "quat_wxyz": [[[goal[6], goal[3], goal[4], goal[5]] for goal in goals_xyzw]],
    }
    path = directory / "trajectory_isaac_format.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _disable_domain_randomization(cfg) -> None:
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


def _functional_state(inner):
    origins = inner.scene.env_origins
    pancake = inner.functional_object.data
    spatula = inner.object.data
    return {
        "pancake_pos": pancake.root_pos_w - origins,
        "pancake_quat": pancake.root_quat_w,
        "pancake_lin_vel": pancake.root_lin_vel_w,
        "pancake_ang_vel": pancake.root_ang_vel_w,
        "spatula_pos": spatula.root_pos_w - origins,
        "spatula_quat": spatula.root_quat_w,
    }


def _settle_without_policy(inner, physics_steps: int, render: bool) -> None:
    """Hold reset joint targets while only physics advances."""
    inner.robot.set_joint_position_target(inner._cur_targets)
    for _ in range(physics_steps):
        inner.scene.write_data_to_sim()
        inner.sim.step(render=render)
        inner.scene.update(dt=inner.physics_dt)


def _setup_gui(inner, blade_centers, pancake_center_xy):
    import isaaclab.sim as sim_utils
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
    from isaaclab.sim.utils import get_current_stage
    from pxr import UsdGeom

    goal_path = "/World/envs/env_0/GoalViz"
    material_path = "/World/Looks/FunctionalGoalGreen"
    material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), opacity=1.0)
    material.func(material_path, material)
    sim_utils.bind_visual_material(goal_path, material_path)
    UsdGeom.Imageable(get_current_stage().GetPrimAtPath(goal_path)).MakeVisible()
    inner.sim.set_camera_view(
        eye=(0.55, -0.65, 0.95),
        target=(pancake_center_xy[0] + 0.03, pancake_center_xy[1], 0.68),
    )

    marker = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/World/Visuals/FunctionalBladeCenter",
            markers={
                "blade_center": sim_utils.SphereCfg(
                    radius=0.012,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.8, 0.0)
                    ),
                )
            },
        )
    )
    path_marker = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/World/Visuals/FunctionalBladePath",
            markers={
                "blade_path": sim_utils.SphereCfg(
                    radius=0.006,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 0.8, 1.0)
                    ),
                )
            },
        )
    )
    if blade_centers:
        import torch

        path_marker.visualize(
            translations=torch.tensor(
                blade_centers, device=inner.device, dtype=torch.float32
            )
            + inner.scene.env_origins[0]
        )
    print(
        "[play] black=real spatula, green=current goal, orange=pancake, "
        "yellow dot=computed blade center, cyan dots=designed blade path",
        flush=True,
    )
    return marker


def _move_blade_marker(marker, state, origins, evaluator_cfg) -> None:
    if marker is None:
        return
    import torch

    from dextoolbench.functional_eval import quat_rotate_wxyz

    offset = torch.tensor(
        evaluator_cfg.spatula_blade_local_offset,
        device=state["spatula_pos"].device,
        dtype=state["spatula_pos"].dtype,
    ).expand(state["spatula_pos"].shape[0], -1)
    blade_center_local = state["spatula_pos"] + quat_rotate_wxyz(
        state["spatula_quat"], offset
    )
    marker.visualize(translations=blade_center_local + origins)


def _failure_reason(metrics, stopped_by: str) -> str:
    from dextoolbench.functional_eval import FailureCode

    if metrics is not None and bool(metrics["success_ever"][0].item()):
        return "none"
    if metrics is not None:
        code = FailureCode(int(metrics["failure_code"][0].item()))
        if code != FailureCode.NONE:
            return code.name.lower()
    if stopped_by == "environment_timeout":
        return "timeout"
    if stopped_by == "environment_termination":
        return "tool_trajectory_failed"
    return "timeout"


def main() -> None:
    args = _args

    import gymnasium as gym
    import numpy as np
    import torch

    import isaacsimenvs  # noqa: F401
    from deployment.rl_player import RlPlayer
    from dextoolbench.functional_eval import (
        PancakeEvaluatorCfg,
        PancakeFunctionalEvaluator,
        quat_rotate_wxyz,
    )
    from dextoolbench.objects import NAME_TO_OBJECT
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg
    from isaacsimenvs.tasks.simtoolreal.utils.obs_utils import (
        compute_intermediate_values,
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.goal_sequence == "pedestal_scoop":
        trajectory_path = (
            REPO_ROOT
            / "dextoolbench/trajectories/spatula/flat_spatula/"
            "scoop_from_pedestal.json"
        )
    elif args.goal_sequence == "pitched_scoop":
        trajectory_path = (
            REPO_ROOT
            / "dextoolbench/trajectories/spatula/flat_spatula/"
            "scoop_and_lift_pitched.json"
        )
    elif args.goal_sequence == "aligned_scoop":
        trajectory_path = (
            REPO_ROOT
            / "dextoolbench/trajectories/spatula/flat_spatula/scoop_and_lift.json"
        )
    else:
        trajectory_path = (
            REPO_ROOT
            / "dextoolbench/trajectories/spatula/flat_spatula/serve_plate.json"
        )
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    recommended_pancake_xy = trajectory.get("pancake_center_xy", [0.0, 0.05])
    pancake_center_xy = (
        recommended_pancake_xy[0] if args.pancake_x is None else args.pancake_x,
        recommended_pancake_xy[1] if args.pancake_y is None else args.pancake_y,
    )
    goal_limit = None if args.goal_sequence == "pedestal_scoop" else SCOOP_GOAL_COUNT
    goals = [goal.copy() for goal in trajectory["goals"][:goal_limit]]
    scoop_start_goal_index = int(trajectory.get("scoop_start_goal_index", 0))
    retract_support_after_goal_count = trajectory.get(
        "retract_support_after_goal_count"
    )
    support_half_spacing = args.support_half_spacing
    if support_half_spacing is None:
        support_half_spacing = float(
            trajectory.get("recommended_support_half_spacing", 0.0)
        )
    for goal in goals:
        goal[2] += args.goal_z_offset
    blade_centers = [center.copy() for center in trajectory.get("blade_centers", [])]
    for center in blade_centers:
        center[2] += args.goal_z_offset
    n_goals = len(goals)
    temporary_dir = tempfile.TemporaryDirectory(prefix="pancake_functional_eval_")
    trajectory_file = _write_trajectory(goals, Path(temporary_dir.name))

    cfg = SimToolRealEnvCfg()
    cfg.seed = args.seed
    cfg.scene.num_envs = 1
    cfg.assets.enable_functional_object = True
    cfg.assets.functional_object_position_xy = pancake_center_xy
    cfg.assets.functional_support_height = args.pan_height
    cfg.assets.functional_support_radius = args.support_radius
    cfg.assets.functional_support_half_spacing = support_half_spacing
    flat_spatula = NAME_TO_OBJECT["flat_spatula"]
    cfg.assets.object_urdf = str(flat_spatula.decomposed_urdf_path)
    cfg.assets.object_scale = tuple(flat_spatula.scale)
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
    start_pose = trajectory["start_pose"].copy()
    start_pose[2] += 0.03
    reset.fixed_start_pose = (
        start_pose[0],
        start_pose[1],
        start_pose[2],
        start_pose[6],
        start_pose[3],
        start_pose[4],
        start_pose[5],
    )
    reset.fixed_trajectory_file = trajectory_file
    _disable_domain_randomization(cfg)

    cfg.termination.eval_success_tolerance = 0.01
    cfg.termination.success_steps = 1
    # Goal hits still advance fixed_trajectory; zero only disables automatic
    # termination after the final waypoint so the external evaluator owns done.
    cfg.termination.max_consecutive_successes = 0

    env = gym.make("Isaacsimenvs-SimToolReal-Direct-v0", cfg=cfg)
    inner = env.unwrapped
    inner._replay_target_lab_order = None
    player = RlPlayer(
        num_observations=inner.cfg.observation_space,
        num_actions=cfg.action_space,
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        device=args.rl_device,
        num_envs=1,
    )

    evaluator_cfg = PancakeEvaluatorCfg(
        task_name="scoop_and_lift",
        spatula_blade_local_offset=tuple(args.blade_local_offset),
        max_blade_pancake_distance=args.max_blade_pancake_distance,
        target_center_xy=pancake_center_xy,
        drop_height=args.drop_height,
        post_trajectory_settle_steps=args.post_trajectory_settle_steps,
        max_episode_steps=args.max_episode_steps,
    )
    evaluator = PancakeFunctionalEvaluator(
        evaluator_cfg, num_envs=1, device=inner.device
    )
    blade_marker = (
        _setup_gui(inner, blade_centers, pancake_center_xy) if args.play else None
    )

    print(f"[functional-eval] checkpoint={args.checkpoint_path}", flush=True)
    print(
        f"[functional-eval] goal_sequence={args.goal_sequence} "
        f"trajectory={trajectory_path} goals={n_goals} "
        f"goal_z_offset={args.goal_z_offset:+.3f}m "
        f"pan_height={args.pan_height:.3f}m "
        f"support_radius={args.support_radius:.3f}m "
        f"support_half_spacing={support_half_spacing:.3f}m "
        f"pancake_xy={pancake_center_xy}",
        flush=True,
    )
    print(
        "[functional-eval] success=lift>=0.03m & blade_distance<=0.08m "
        f"& z>={args.drop_height:.3f}m for 10 consecutive policy steps",
        flush=True,
    )

    episode_records = []
    for episode_index in range(args.num_episodes):
        player.player.init_rnn()
        episode_seed = args.seed + episode_index
        env.reset(seed=episode_seed)
        _settle_without_policy(inner, args.settle_physics_steps, render=args.play)
        compute_intermediate_values(inner)
        obs = inner._get_observations()
        state = _functional_state(inner)
        evaluator.reset(torch.arange(1, device=inner.device), state)
        initial_pancake_pos = state["pancake_pos"][0].clone()
        max_pancake_displacement_xy = 0.0
        max_pre_scoop_pancake_displacement_xy = 0.0
        blade_offset = torch.tensor(
            evaluator_cfg.spatula_blade_local_offset,
            device=inner.device,
            dtype=state["spatula_pos"].dtype,
        ).unsqueeze(0)
        blade_center = state["spatula_pos"] + quat_rotate_wxyz(
            state["spatula_quat"], blade_offset
        )
        alignment_delta = blade_center - state["pancake_pos"]
        print(
            f"[episode {episode_index + 1}] initial "
            f"pancake_xyz={state['pancake_pos'][0].tolist()} "
            f"blade_xyz={blade_center[0].tolist()} "
            f"blade_minus_pancake={alignment_delta[0].tolist()}",
            flush=True,
        )
        _move_blade_marker(blade_marker, state, inner.scene.env_origins, evaluator_cfg)

        metrics = None
        goals_reached = 0
        trajectory_complete_step = None
        stopped_by = "max_episode_steps"
        support_retracted = False
        start_time = time.perf_counter()

        for step in range(1, args.max_episode_steps + 1):
            if args.play and not _app.is_running():
                stopped_by = "gui_closed"
                break
            step_started = time.perf_counter()
            goals_before_step = goals_reached
            policy_obs = obs["policy"].to(args.rl_device)
            action = player.get_normalized_action(
                policy_obs, deterministic_actions=True
            )
            obs, _, terminated, truncated, _ = env.step(action.to(inner.device))

            is_terminated = bool(terminated[0].item())
            is_truncated = bool(truncated[0].item())
            if is_terminated or is_truncated:
                stopped_by = (
                    "environment_termination"
                    if is_terminated
                    else "environment_timeout"
                )
                goals_reached = max(
                    goals_reached,
                    int(inner._prev_episode_successes[0].item()),
                )
                break

            goals_reached = max(
                goals_reached,
                min(int(inner._successes[0].item()), n_goals),
            )
            if (
                not support_retracted
                and retract_support_after_goal_count is not None
                and goals_reached >= int(retract_support_after_goal_count)
            ):
                support_pos = inner.functional_support.data.root_pos_w.clone()
                support_quat = inner.functional_support.data.root_quat_w.clone()
                old_support_z = support_pos[0, 2].item()
                support_pos[:, 2] -= args.pan_height + 0.05
                inner.functional_support.write_root_pose_to_sim(
                    torch.cat([support_pos, support_quat], dim=-1)
                )
                inner.functional_support.write_root_velocity_to_sim(
                    torch.zeros((1, 6), device=inner.device)
                )
                support_retracted = True
                print(
                    f"[episode {episode_index + 1}] support retracted "
                    f"after_goal={goals_reached:02d}/{n_goals} "
                    f"z={old_support_z:.3f}->{support_pos[0, 2].item():.3f}m",
                    flush=True,
                )
            if goals_reached >= n_goals and trajectory_complete_step is None:
                trajectory_complete_step = step

            state = _functional_state(inner)
            metrics = evaluator.update(state)
            current_blade_center = state["spatula_pos"] + quat_rotate_wxyz(
                state["spatula_quat"], blade_offset
            )
            pancake_displacement_xy = torch.linalg.vector_norm(
                state["pancake_pos"][0, :2] - initial_pancake_pos[:2]
            ).item()
            max_pancake_displacement_xy = max(
                max_pancake_displacement_xy, pancake_displacement_xy
            )
            if goals_before_step < scoop_start_goal_index:
                max_pre_scoop_pancake_displacement_xy = max(
                    max_pre_scoop_pancake_displacement_xy,
                    pancake_displacement_xy,
                )
            if (
                scoop_start_goal_index
                and goals_before_step < scoop_start_goal_index <= goals_reached
            ):
                print(
                    f"[episode {episode_index + 1}] pre-scoop check "
                    f"max_pancake_xy_delta="
                    f"{max_pre_scoop_pancake_displacement_xy:.4f}m",
                    flush=True,
                )
            if goals_reached > goals_before_step:
                completed_goal_index = goals_reached - 1
                target_blade_center = torch.tensor(
                    blade_centers[completed_goal_index],
                    device=inner.device,
                    dtype=current_blade_center.dtype,
                )
                blade_goal_error = torch.linalg.vector_norm(
                    current_blade_center[0] - target_blade_center
                ).item()
                print(
                    f"[episode {episode_index + 1}] reached_goal="
                    f"{goals_reached:02d}/{n_goals} "
                    f"blade_xyz={current_blade_center[0].tolist()} "
                    f"blade_goal_error={blade_goal_error:.4f}m "
                    f"pancake_xyz={state['pancake_pos'][0].tolist()}",
                    flush=True,
                )
            _move_blade_marker(
                blade_marker, state, inner.scene.env_origins, evaluator_cfg
            )

            if step == 1 or step % args.log_interval == 0:
                print(
                    f"[episode {episode_index + 1}] step={step:04d} "
                    f"goals={goals_reached:02d}/{n_goals} "
                    f"lift={metrics['lift_height'][0].item():+.3f}m "
                    f"pancake_xy_delta={pancake_displacement_xy:.3f}m "
                    f"blade_dist={metrics['blade_distance'][0].item():.3f}m "
                    f"hold={int(metrics['success_counter'][0].item()):02d}/10 "
                    f"success={bool(metrics['success_ever'][0].item())}",
                    flush=True,
                )

            finish_visible_trajectory = args.play and args.keep_open
            if (
                bool(metrics["success_ever"][0].item())
                and not finish_visible_trajectory
            ):
                stopped_by = "functional_success"
                break
            if (
                not bool(metrics["not_dropped"][0].item())
                and not finish_visible_trajectory
            ):
                stopped_by = "pancake_dropped"
                break
            if (
                trajectory_complete_step is not None
                and step - trajectory_complete_step >= args.post_trajectory_settle_steps
            ):
                stopped_by = "post_trajectory_settle_timeout"
                break

            if args.play:
                time.sleep(
                    max(
                        0.0,
                        1.0 / CONTROL_HZ - (time.perf_counter() - step_started),
                    )
                )

        if stopped_by == "gui_closed":
            print("[functional-eval] GUI closed; partial episode discarded")
            break

        if metrics is None:
            state = _functional_state(inner)
            metrics = evaluator.update(state)

        success = bool(metrics["success_ever"][0].item())
        elapsed_steps = step
        record = {
            "task": args.task,
            "goal_sequence": args.goal_sequence,
            "goal_z_offset": args.goal_z_offset,
            "pan_height": args.pan_height,
            "support_radius": args.support_radius,
            "support_half_spacing": support_half_spacing,
            "pancake_center_xy": pancake_center_xy,
            "seed": episode_seed,
            "functional_success": success,
            "functional_progress": float(metrics["functional_progress"][0].item()),
            "trajectory_progress": min(goals_reached, n_goals) / n_goals,
            "max_lift_height": float(metrics["max_lift_height"][0].item()),
            "initial_pancake_xyz": initial_pancake_pos.tolist(),
            "final_pancake_xyz": state["pancake_pos"][0].tolist(),
            "max_pancake_displacement_xy": max_pancake_displacement_xy,
            "max_pre_scoop_pancake_displacement_xy": (
                max_pre_scoop_pancake_displacement_xy
            ),
            "final_flip_angle_deg": float(metrics["flip_angle_deg"][0].item()),
            "final_target_distance": float(metrics["target_distance"][0].item()),
            "min_blade_distance": float(metrics["min_blade_distance"][0].item()),
            "near_blade_steps": int(metrics["near_blade_steps"][0].item()),
            "time_to_success_sec": elapsed_steps / CONTROL_HZ if success else None,
            "episode_length": elapsed_steps,
            "wall_time_sec": time.perf_counter() - start_time,
            "failure_reason": _failure_reason(metrics, stopped_by),
            "stopped_by": stopped_by,
        }
        episode_records.append(record)
        print(f"[functional-eval] episode result: {json.dumps(record)}", flush=True)

    if not episode_records:
        env.close()
        _app.close()
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    failures = Counter(
        record["failure_reason"]
        for record in episode_records
        if not record["functional_success"]
    )
    summary = {
        "num_episodes": len(episode_records),
        "functional_success_rate": float(
            np.mean([record["functional_success"] for record in episode_records])
        ),
        "mean_functional_progress": float(
            np.mean([record["functional_progress"] for record in episode_records])
        ),
        "mean_trajectory_progress": float(
            np.mean([record["trajectory_progress"] for record in episode_records])
        ),
        "failure_counts": dict(failures),
    }
    (output_dir / "episodes.json").write_text(
        json.dumps(episode_records, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"[functional-eval] summary: {json.dumps(summary)}", flush=True)
    print(f"[functional-eval] wrote {output_dir}", flush=True)

    if args.play and args.keep_open:
        print("[play] final frame held; close Isaac Sim to exit", flush=True)
        try:
            while _app.is_running():
                _app.update()
        except KeyboardInterrupt:
            pass

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
