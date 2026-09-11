#!/usr/bin/env python3
"""One simulator, two identical tasks; env0 official, env1 frozen FlashSAC."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
STR = Path(os.environ.get('SIMTOOLREAL_ROOT', str(ROOT / 'third_party/simtoolreal')))
sys.path[:0] = [str(ROOT), str(STR)]
os.environ.setdefault('OMNI_KIT_ACCEPT_EULA', 'YES')
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=0, help='0: keep playing until GUI closes')
    parser.add_argument('--flash-checkpoint', type=Path, default=ROOT / 'diagnostics/credit_nstep3_20260910/candidate/models/seed0/step48830')
    parser.add_argument('--report', type=Path, default=None)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app

    import gymnasium as gym
    import numpy as np
    import torch
    import isaacsimenvs
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg
    from deployment.rl_player import RlPlayer
    from dextoolbench.objects import NAME_TO_OBJECT
    from scripts.diagnose_str_critic import checkpoint_models

    os.chdir(STR)
    torch.set_num_threads(2)
    torch.manual_seed(0)
    data = json.loads((STR / 'dextoolbench/trajectories/hammer/claw_hammer/swing_down.json').read_text())
    goals = data['goals']
    # Both envs receive the exact same full task, not different random goals.
    with tempfile.TemporaryDirectory(prefix='str_pair_') as temp:
        trajectory = Path(temp) / 'goals.json'
        trajectory.write_text(json.dumps({'pos': [[g[:3] for g in goals]],
            'quat_wxyz': [[[g[6], g[3], g[4], g[5]] for g in goals]]}))
        cfg = SimToolRealEnvCfg()
        cfg.seed = 0
        cfg.scene.num_envs = 2
        cfg.scene.env_spacing = 1.5
        obj = NAME_TO_OBJECT['claw_hammer']
        cfg.assets.object_urdf = str(obj.decomposed_urdf_path)
        cfg.assets.object_scale = tuple(obj.scale)
        cfg.assets.table_urdf = 'assets/urdf/table_narrow.urdf'
        rs = cfg.reset
        for name in ('reset_position_noise_x', 'reset_position_noise_y', 'reset_position_noise_z',
                     'reset_dof_pos_random_interval_arm', 'reset_dof_pos_random_interval_fingers',
                     'reset_dof_vel_random_interval', 'table_reset_z_range'):
            setattr(rs, name, 0.)
        rs.table_reset_z = .38
        rs.start_arm_higher = True
        sp = data['start_pose']
        rs.fixed_start_pose = (sp[0], sp[1], sp[2] + .03, sp[6], sp[3], sp[4], sp[5])
        rs.fixed_trajectory_file = str(trajectory)
        dr = cfg.domain_randomization
        dr.use_obs_delay = dr.use_action_delay = dr.use_object_state_delay_noise = False
        dr.object_scale_noise_multiplier_range = (1., 1.)
        dr.object_friction_scale_range = dr.fingertip_friction_scale_range = (1., 1.)
        dr.joint_velocity_obs_noise_std = dr.force_scale = dr.torque_scale = 0.
        # A single common controller, matching the selected Flash arm1 run.
        cfg.action.arm_moving_average = 1.
        cfg.action.hand_moving_average = .1
        cfg.termination.success_tolerance = cfg.termination.target_success_tolerance = .01
        cfg.termination.eval_success_tolerance = .01
        cfg.termination.success_steps = 1
        cfg.termination.max_consecutive_successes = len(goals)
        env = gym.make('Isaacsimenvs-SimToolReal-Direct-v0', cfg=cfg)
        raw = env.unwrapped
        raw._replay_target_lab_order = None
        official = RlPlayer(num_observations=140, num_actions=29,
            config_path=str(STR / 'pretrained_policy/config.yaml'),
            checkpoint_path=str(STR / 'pretrained_policy/model.pth'), device=str(raw.device), num_envs=1)
        actor, critic, target, _, _ = checkpoint_models(args.flash_checkpoint, str(raw.device))
        del critic, target
        actor.requires_grad_(False)
        official.player.init_rnn()
        obs, _ = env.reset()
        origins = raw.scene.env_origins
        initial = {'policy': obs['policy'], 'critic': obs['critic'],
            'joint_pos': raw.robot.data.joint_pos, 'joint_vel': raw.robot.data.joint_vel,
            'object_pos': raw.object.data.root_pos_w - origins,
            'object_quat': raw.object.data.root_quat_w,
            'goal_pos': raw.goal_viz.data.root_pos_w - origins,
            'goal_quat': raw.goal_viz.data.root_quat_w, 'targets': raw._prev_targets}
        errors = {k: float((v[0] - v[1]).abs().max()) for k, v in initial.items()}
        assert max(errors.values()) < 1e-5, f'Paired reset mismatch: {errors}'
        # Camera right vector follows env0 -> env1, so official is on the left.
        centers = origins.detach().cpu().numpy()
        right = centers[1] - centers[0]
        right /= np.linalg.norm(right)
        forward = np.cross(np.array([0., 0., 1.]), right)
        middle = centers.mean(0) + np.array([0., 0., .65])
        raw.sim.set_camera_view(eye=tuple(middle - 2.3 * forward + np.array([0., 0., .9])),
                                target=tuple(middle))
        import isaaclab.sim as sim_utils
        from isaaclab.sim.utils import get_current_stage
        from pxr import UsdGeom
        material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0., 1., 0.), opacity=.35)
        material.func('/World/Looks/CompareGoals', material)
        for i in range(2):
            path = f'/World/envs/env_{i}/GoalViz'
            sim_utils.bind_visual_material(path, '/World/Looks/CompareGoals')
            UsdGeom.Imageable(get_current_stage().GetPrimAtPath(path)).MakeVisible()
        print('PAIRED_RESET_PASS', errors, flush=True)
        print('LEFT=OFFICIAL | RIGHT=FLASH_100M (provisional best). Same hammer/task/controller; deterministic actions.', flush=True)
        print('Common settings: arm EMA=1, hand EMA=.1, tolerance=.01, DR off. No training.', flush=True)
        report = {'initial_max_abs_errors': errors, 'flash_checkpoint': str(args.flash_checkpoint),
            'task': 'claw_hammer/swing_down', 'goals': len(goals), 'arm_ema': 1., 'hand_ema': .1,
            'tolerance': .01, 'episodes': [], 'steps': 0, 'status': 'running'}
        step = 0
        while app.is_running() and (not args.steps or step < args.steps):
            begin = time.perf_counter()
            with torch.no_grad():
                oa = official.get_normalized_action(obs['policy'][:1], deterministic_actions=True)
                mean, _ = actor.get_mean_and_std(obs['policy'][1:2], training=False)
                actions = torch.cat([oa, mean.tanh()], dim=0).clamp(-1., 1.)
            assert actions.shape == (2, 29) and torch.isfinite(actions).all()
            obs, reward, terminated, truncated, info = env.step(actions)
            done = terminated | truncated
            if done[0]:
                official.player.init_rnn()
            for i in torch.nonzero(done).flatten().tolist():
                record = {'env': i, 'policy': 'official' if i == 0 else 'flash100M',
                    'goals': int(raw._prev_episode_successes[i]), 'step': step,
                    'terminated': bool(terminated[i]), 'truncated': bool(truncated[i])}
                report['episodes'].append(record)
                print(record, flush=True)
            step += 1
            if step == 1 or step % 120 == 0:
                print(f'PAIR_STEP {step}: goals official/flash={raw._successes.tolist()}', flush=True)
            if not args.headless:
                time.sleep(max(0., raw.step_dt - (time.perf_counter() - begin)))
        report.update(steps=step, status='complete')
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2))
        print('PAIR_PLAY_COMPLETE', step, flush=True)
        env.close()
    app.close()


if __name__ == '__main__':
    main()

