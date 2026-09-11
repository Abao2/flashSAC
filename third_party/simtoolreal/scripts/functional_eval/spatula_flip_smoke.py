"""Visual smoke test for the pancake flip evaluator.

This is deliberately independent of SimToolReal: no robot, policy, goal
trajectory, or checkpoint is loaded.  The pancake is teleported face-down
after two seconds so the evaluator can be checked in isolation.
"""

import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument(
    "--steps",
    type=int,
    default=0,
    help="Physics steps to run; 0 keeps the GUI open until it is closed.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils.math import quat_apply


PANCAKE_RADIUS = 0.08
PANCAKE_HEIGHT = 0.012
PHYSICS_HZ = 120
HOLD_STEPS = PHYSICS_HZ  # one second


def create_scene() -> RigidObject:
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/Ground", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    light_cfg.func("/World/Light", light_cfg)

    target_cfg = sim_utils.CylinderCfg(
        radius=0.15,
        height=0.001,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.55, 0.1)),
    )
    target_cfg.func("/World/TargetArea", target_cfg, translation=(0.0, 0.0, 0.0005))

    pancake_cfg = RigidObjectCfg(
        prim_path="/World/Pancake",
        spawn=sim_utils.CylinderCfg(
            radius=PANCAKE_RADIUS,
            height=PANCAKE_HEIGHT,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.08),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.6,
                restitution=0.05,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.85, 0.55, 0.20)
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.02),
            rot=(1.0, 0.0, 0.0, 0.0),  # Isaac Lab uses wxyz.
        ),
    )
    pancake = RigidObject(cfg=pancake_cfg)

    # A cylinder looks identical after a 180-degree flip.  These visual-only
    # dots make the two faces distinguishable without changing the physics.
    top_dot_cfg = sim_utils.SphereCfg(
        radius=0.018,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.3, 1.0)),
    )
    top_dot_cfg.func(
        "/World/Pancake/BlueTop",
        top_dot_cfg,
        translation=(0.04, 0.0, PANCAKE_HEIGHT / 2 + 0.003),
    )
    bottom_dot_cfg = sim_utils.SphereCfg(
        radius=0.018,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.1, 0.1)),
    )
    bottom_dot_cfg.func(
        "/World/Pancake/RedBottom",
        bottom_dot_cfg,
        translation=(0.04, 0.0, -PANCAKE_HEIGHT / 2 - 0.003),
    )
    return pancake


def compute_flip_metrics(pancake: RigidObject, stable_counter: torch.Tensor):
    quat_w = pancake.data.root_quat_w
    local_normal = torch.tensor(
        [[0.0, 0.0, 1.0]], device=quat_w.device
    ).expand(quat_w.shape[0], -1)
    normal_z = quat_apply(quat_w, local_normal)[:, 2]

    flipped = normal_z < -0.866  # face-down within 30 degrees
    inside_target = torch.linalg.vector_norm(
        pancake.data.root_pos_w[:, :2], dim=-1
    ) < 0.15
    linear_speed = torch.linalg.vector_norm(
        pancake.data.root_lin_vel_w, dim=-1
    )
    angular_speed = torch.linalg.vector_norm(
        pancake.data.root_ang_vel_w, dim=-1
    )
    stable = (linear_speed < 0.05) & (angular_speed < 0.5)
    valid = flipped & inside_target & stable

    stable_counter[:] = torch.where(
        valid, stable_counter + 1, torch.zeros_like(stable_counter)
    )
    return {
        "normal_z": normal_z,
        "progress": torch.acos(torch.clamp(normal_z, -1.0, 1.0)) / torch.pi,
        "flipped": flipped,
        "inside_target": inside_target,
        "stable": stable,
        "angular_speed": angular_speed,
        "success": stable_counter >= HOLD_STEPS,
    }


def set_pancake_state(
    pancake: RigidObject,
    *,
    x: float = 0.0,
    face_down: bool = True,
    angular_speed_z: float = 0.0,
) -> None:
    quat = [0.0, 1.0, 0.0, 0.0] if face_down else [1.0, 0.0, 0.0, 0.0]
    pose = torch.tensor(
        [[x, 0.0, 0.05, *quat]],
        device=pancake.device,
        dtype=torch.float32,
    )
    velocity = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 0.0, angular_speed_z]],
        device=pancake.device,
        dtype=torch.float32,
    )
    pancake.write_root_pose_to_sim(pose)
    pancake.write_root_velocity_to_sim(velocity)


def main() -> None:
    sim = SimulationContext(
        SimulationCfg(dt=1.0 / PHYSICS_HZ, device=args_cli.device)
    )
    sim.set_camera_view(eye=(0.45, -0.45, 0.28), target=(0.0, 0.0, 0.0))
    pancake = create_scene()
    sim.reset()

    stable_counter = torch.zeros(
        pancake.num_instances, dtype=torch.long, device=pancake.device
    )
    step = 0
    checks = {"outside": False, "spinning": False, "success": False, "reset": False}

    while simulation_app.is_running() and (args_cli.steps <= 0 or step < args_cli.steps):
        if step == 2 * PHYSICS_HZ:
            print("[test 1] face-down but outside target", flush=True)
            set_pancake_state(pancake, x=0.30)
            stable_counter.zero_()
        elif step == 4 * PHYSICS_HZ:
            print("[test 2] face-down but spinning", flush=True)
            set_pancake_state(pancake, angular_speed_z=5.0)
            stable_counter.zero_()
        elif 4 * PHYSICS_HZ < step < 6 * PHYSICS_HZ:
            pancake.write_root_velocity_to_sim(
                torch.tensor(
                    [[0.0, 0.0, 0.0, 0.0, 0.0, 5.0]],
                    device=pancake.device,
                )
            )
        elif step == 6 * PHYSICS_HZ:
            print("[test 3] face-down, inside target, and stationary", flush=True)
            set_pancake_state(pancake)
            stable_counter.zero_()
        elif step == 8 * PHYSICS_HZ:
            print("[test 4] reset face-up and clear counters", flush=True)
            set_pancake_state(pancake, face_down=False)
            stable_counter.zero_()

        sim.step(render=not args_cli.headless)
        pancake.update(sim.get_physics_dt())
        metrics = compute_flip_metrics(pancake, stable_counter)

        if 2 * PHYSICS_HZ <= step < 4 * PHYSICS_HZ:
            passed_now = (
                metrics["flipped"].item()
                and not metrics["inside_target"].item()
                and metrics["stable"].item()
                and stable_counter.item() == 0
            )
            if passed_now and not checks["outside"]:
                print("[test 1] PASS — outside target keeps hold at zero", flush=True)
            checks["outside"] |= passed_now
        elif 4 * PHYSICS_HZ <= step < 6 * PHYSICS_HZ:
            passed_now = (
                metrics["flipped"].item()
                and metrics["inside_target"].item()
                and not metrics["stable"].item()
                and stable_counter.item() == 0
            )
            if passed_now and not checks["spinning"]:
                print("[test 2] PASS — angular motion keeps hold at zero", flush=True)
            checks["spinning"] |= passed_now
        elif 6 * PHYSICS_HZ <= step < 8 * PHYSICS_HZ:
            passed_now = metrics["success"].item()
            if passed_now and not checks["success"]:
                print("[test 3] PASS — valid state held for one second", flush=True)
            checks["success"] |= passed_now
        elif step >= 8 * PHYSICS_HZ:
            passed_now = (
                not metrics["flipped"].item()
                and stable_counter.item() == 0
                and not metrics["success"].item()
            )
            if passed_now and not checks["reset"]:
                print("[test 4] PASS — reset cleared progress state", flush=True)
            checks["reset"] |= passed_now

        if step % 30 == 0:
            print(
                f"step={step:04d} "
                f"normal_z={metrics['normal_z'].item():+.3f} "
                f"progress={metrics['progress'].item():.3f} "
                f"flipped={metrics['flipped'].item()} "
                f"inside={metrics['inside_target'].item()} "
                f"stable={metrics['stable'].item()} "
                f"ang_speed={metrics['angular_speed'].item():.2f} "
                f"hold={stable_counter.item()} "
                f"success={metrics['success'].item()}",
                flush=True,
            )

        step += 1
        if not args_cli.headless:
            time.sleep(1.0 / PHYSICS_HZ)

    if args_cli.steps > 0:
        missing = [name for name, passed in checks.items() if not passed]
        if missing:
            raise RuntimeError(f"failed checks: {', '.join(missing)}")
        print(f"[functional-smoke] ALL CHECKS PASS — {', '.join(checks)}", flush=True)
    # Isaac Sim 5.1 can hang in app.close(); match the repository smoke tests.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
