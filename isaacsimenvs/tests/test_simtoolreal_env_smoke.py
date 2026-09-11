"""Lightweight env-creation smoke test for SimToolReal.

Skips Hydra + rl_games + tensorboard. Just spins up AppLauncher, instantiates
the env via `gym.make`, runs a few steps with random actions, and asserts
the per-env Object prim count matches num_envs (the original cloner-drop
guard rail). Useful when iterating on scene_utils.

    .venv_isaacsim/bin/python isaacsimenvs/tests/test_simtoolreal_env_smoke.py \\
      --num_envs 8 --num_assets_per_type 2 --steps 10
"""

from __future__ import annotations

import argparse
import time

from isaaclab.app import AppLauncher


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--num_assets_per_type", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--gui", action="store_true", help="Open the Isaac Sim viewer")
    parser.add_argument(
        "--reset_demo",
        action="store_true",
        help="Visually move the pancake at sim_t=3 s and reset it at sim_t=6 s",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = not args.gui

    app_launcher = AppLauncher(args)
    app = app_launcher.app

    import gymnasium as gym
    import torch

    import isaacsimenvs  # noqa: F401  (registers gym envs)
    from isaaclab.sim.utils import find_matching_prim_paths
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg

    cfg = SimToolRealEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.assets.num_assets_per_type = args.num_assets_per_type
    cfg.assets.enable_functional_object = True

    env = gym.make("Isaacsimenvs-SimToolReal-Direct-v0", cfg=cfg)
    inner = env.unwrapped

    object_prims = find_matching_prim_paths("/World/envs/env_.*/Object")
    goal_prims = find_matching_prim_paths("/World/envs/env_.*/GoalViz")
    functional_prims = find_matching_prim_paths("/World/envs/env_.*/FunctionalObject")
    assert len(object_prims) == args.num_envs, (
        f"Object prims: {len(object_prims)}, expected {args.num_envs}"
    )
    assert len(goal_prims) == args.num_envs, (
        f"GoalViz prims: {len(goal_prims)}, expected {args.num_envs}"
    )
    assert len(functional_prims) == args.num_envs, (
        f"FunctionalObject prims: {len(functional_prims)}, expected {args.num_envs}"
    )
    print(
        f"[smoke] OK — {len(object_prims)} Object + {len(goal_prims)} GoalViz + "
        f"{len(functional_prims)} FunctionalObject prims"
    )

    # Diagnostic: all PhysX joint params that could dampen response —
    # stiffness, damping, armature (virtual inertia), friction, effort limit.
    from isaacsimenvs.tasks.simtoolreal.utils.scene_utils import JOINT_NAMES_CANONICAL
    stiffness = inner.robot.data.joint_stiffness[0].cpu().numpy()
    damping = inner.robot.data.joint_damping[0].cpu().numpy()
    armature = inner.robot.data.joint_armature[0].cpu().numpy()
    friction = inner.robot.data.joint_friction_coeff[0].cpu().numpy()
    effort_lim = inner.robot.data.joint_effort_limits[0].cpu().numpy()
    perm = inner._perm_lab_to_canon.cpu().numpy()
    # Dump per-link masses / inertias Isaac Lab sees from the URDF.
    # URDF declares thumb chain masses around 1e-3 kg per link — if
    # PhysX 5 shows e.g. 1 kg per link we've found the inertia inflation
    # that explains the 100x slow PD response.
    masses = inner.robot.data.default_mass[0].cpu().numpy()  # (num_bodies,)
    inertias = inner.robot.data.default_inertia[0].cpu().numpy()  # (num_bodies, 9) or similar
    body_names = inner.robot.data.body_names
    print(f"[smoke] Body masses / inertias (all {len(body_names)} bodies):")
    print(f"  {'idx':>3}  {'name':<25s}  {'mass (kg)':>10s}  {'Ixx':>10s}  {'Iyy':>10s}  {'Izz':>10s}")
    for i, name in enumerate(body_names):
        ixx = inertias[i, 0] if inertias.ndim == 2 else inertias[i]
        iyy = inertias[i, 4] if inertias.ndim == 2 and inertias.shape[1] >= 5 else 0
        izz = inertias[i, 8] if inertias.ndim == 2 and inertias.shape[1] >= 9 else 0
        print(f"  [{i:2d}] {name:<25s}  {masses[i]:>10.5f}  {ixx:>10.2e}  {iyy:>10.2e}  {izz:>10.2e}")
    print()
    print(f"[smoke] PhysX joint parameters (canonical order):")
    print(f"  {'idx':>3}  {'name':<22s}  {'K':>8s}  {'D':>8s}  {'armature':>8s}  {'friction':>8s}  {'effort':>8s}")
    for i in range(29):
        lab_j = perm[i]
        print(
            f"  [{i:2d}] {JOINT_NAMES_CANONICAL[i]:<22s}  "
            f"{stiffness[lab_j]:>8.4f}  {damping[lab_j]:>8.4f}  "
            f"{armature[lab_j]:>8.5f}  {friction[lab_j]:>8.5f}  {effort_lim[lab_j]:>8.4f}"
        )

    env.reset()
    displaced_pose = inner.functional_object.data.root_pose_w.clone()
    displaced_pose[:, :3] += torch.tensor(
        [0.3, 0.0, 0.2], device=inner.device
    )
    displaced_pose[:, 3:] = torch.tensor(
        [0.0, 1.0, 0.0, 0.0], device=inner.device
    )
    inner.functional_object.write_root_pose_to_sim(displaced_pose)
    inner.functional_object.write_root_velocity_to_sim(
        torch.ones(args.num_envs, 6, device=inner.device)
    )

    obs, _ = env.reset()
    functional_pos_local = (
        inner.functional_object.data.root_pos_w - inner.scene.env_origins
    )
    expected_pos_local = torch.zeros_like(functional_pos_local)
    expected_pos_local[:, 1] = 0.05
    expected_pos_local[:, 2] = inner._table_z_per_env + 0.15 + 0.006
    torch.testing.assert_close(functional_pos_local, expected_pos_local)
    torch.testing.assert_close(
        inner.functional_object.data.root_quat_w,
        torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=inner.device
        ).expand(args.num_envs, -1),
    )
    torch.testing.assert_close(
        inner.functional_object.data.root_vel_w,
        torch.zeros(args.num_envs, 6, device=inner.device),
    )
    print(
        "[smoke] OK — FunctionalObject reset face-up at local pose "
        f"{functional_pos_local[0].cpu().tolist()} with zero velocity"
    )
    print(
        f"[smoke] reset → policy obs {obs['policy'].shape}, "
        f"critic obs {obs['critic'].shape}"
    )

    action_dim = env.unwrapped.action_space.shape[-1] if hasattr(env.unwrapped.action_space, "shape") else cfg.action_space
    if args.reset_demo:
        args.steps = 540  # 9 seconds at the 60 Hz environment rate.
        print("[reset-demo] sim_t=0s: pancake is at its normal reset pose")
    steps_run = 0
    for step in range(args.steps):
        if not app.is_running():
            break
        if args.reset_demo and 180 <= step < 360:
            if step == 180:
                print(
                    "[reset-demo] sim_t=3s: MOVED pancake 0.3 m sideways, "
                    "0.2 m upward, and face-down"
                )
            inner.functional_object.write_root_pose_to_sim(displaced_pose)
            inner.functional_object.write_root_velocity_to_sim(
                torch.zeros(args.num_envs, 6, device=inner.device)
            )
        elif args.reset_demo and step == 360:
            obs, _ = env.reset()
            reset_pos_local = (
                inner.functional_object.data.root_pos_w - inner.scene.env_origins
            )
            print(
                "[reset-demo] sim_t=6s: RESET called; pancake returned to "
                f"{reset_pos_local[0].cpu().tolist()}"
            )
        action = torch.zeros(
            (args.num_envs, action_dim), device=inner.device, dtype=torch.float32
        )
        obs, reward, terminated, truncated, info = env.step(action)
        any_nan = (
            torch.isnan(obs["policy"]).any().item()
            or torch.isnan(obs["critic"]).any().item()
            or torch.isnan(reward).any().item()
        )
        if any_nan:
            raise RuntimeError(f"step {step}: NaN detected in obs/reward")
        steps_run += 1
        if args.gui and args.reset_demo:
            time.sleep(1.0 / 60.0)
    if args.reset_demo and steps_run == args.steps:
        print("[reset-demo] sim_t=9s: DONE")
    print(f"[smoke] OK — {steps_run} steps, no NaN")

    env.close()
    app.close()


if __name__ == "__main__":
    main()
