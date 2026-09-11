"""Frozen, paired first-episode STR play. Not a paper success benchmark.

Native stochastic sampling uses training=True ONLY in sample_actions (sampling
temperature). No learner/process_transition call and no replay population.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time


def aim_camera(camera, eye, target):
    """Set an explicit native USD look-at transform for a diagnostic camera."""
    from pxr import Gf, UsdGeom
    xform = UsdGeom.Xformable(camera.stage.GetPrimAtPath(camera.cfg.prim_path))
    matrix = Gf.Matrix4d(1.0).SetLookAt(
        Gf.Vec3d(*[float(x) for x in eye]),
        Gf.Vec3d(*[float(x) for x in target]), Gf.Vec3d(0.0, 0.0, 1.0))
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(matrix.GetInverse())


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--config-name', default='simtoolreal_full_arm1')
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--num-envs', type=int, default=64)
    parser.add_argument('--steps', type=int, default=1200)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--tolerance', type=float, default=.075,
                        help='Frozen evaluation criterion; default preserves prior play protocol.')
    args = parser.parse_args(argv)
    if not math.isfinite(args.tolerance) or args.tolerance <= 0:
        parser.error('--tolerance must be finite and positive')
    return args


def main():
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    str_root = repo.parent / 'simtoolreal'
    sys.path[:0] = [str(repo), str(str_root)]
    os.chdir(str_root)
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ.setdefault(name, '2')
    os.environ.setdefault('OMNI_KIT_ACCEPT_EULA', 'YES')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Output directory must be empty')
    # Isaac Sim 5.1's SyntheticData fails with this training venv's NumPy 2.2.
    # Opt-in process-local import from an EXISTING NumPy 1.26 install; no pip,
    # global path change or dependency mutation. Other imports use training env.
    numpy_site = os.environ.get('STR_PLAY_NUMPY_SITE')
    if numpy_site:
        sys.path.insert(0, numpy_site)
        import numpy
        sys.path.pop(0)
    import hydra
    import imageio.v2 as imageio
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from omegaconf import OmegaConf
    from flash_rl.agents import create_agent
    from flash_rl.envs.isaaclab import make_isaaclab_env
    from scripts.eval_str_full_nodr import freeze_tolerance
    from scripts.diagnose_str_rollouts import model_digest, numerical_settings

    OmegaConf.register_new_resolver('eval', lambda s: eval(s), replace=True)
    with hydra.initialize_config_dir(version_base=None, config_dir=str(repo / 'configs')):
        cfg = hydra.compose(config_name=args.config_name, overrides=[
            f'seed={args.seed}', f'num_train_envs={args.num_envs}'])
    freeze_tolerance(cfg, str_root / 'isaacsimenvs/cfg/task/SimToolReal.yaml', args.tolerance)
    cfg.agent.buffer_max_length = cfg.agent.buffer_min_length = cfg.agent.sample_batch_size = 1
    cfg.agent.use_compile = cfg.agent.load_optimizer = cfg.agent.load_reward_normalizer = False
    OmegaConf.resolve(cfg)
    OmegaConf.clear_resolver('eval')
    OmegaConf.save(cfg, output / 'resolved_config.yaml')
    assert cfg.agent.get('actor_history_length', 1) == 1
    numerical_settings(torch, enable=True)
    actor_file = args.checkpoint.resolve() / 'actor.pt'
    metadata = dict(status='running', checkpoint=str(actor_file), config_name=args.config_name,
                    actor_sha256=hashlib.sha256(actor_file.read_bytes()).hexdigest(),
                    learning=False, seed=args.seed, num_envs=args.num_envs,
                    numpy_version=np.__version__, numpy_file=np.__file__,
                    max_steps_per_mode=args.steps, frozen_tolerance=args.tolerance,
                    protocol='Same first reset in each mode; first episode per env only; censor at step cap. NOT paper evaluation.',
                    pairing_intervention='Reseed all RNGs; zero previous reward before each full reset; freeze tolerance.',
                    contact_note='No contact-force measurement. Height and fingertip distances are geometric proxies, NOT grasp success.')
    def save_json(name, value):
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + '\n')
    save_json('metadata.json', metadata)
    env = writer = None
    start = time.monotonic()
    try:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        env = make_isaaclab_env(
            env_name=cfg.env.env_name, num_envs=args.num_envs, seed=args.seed,
            headless=True, device=cfg.env.device, action_bounds=cfg.env.action_bounds,
            registration_modules=cfg.env.registration_modules,
            env_cfg_yaml_entry_point=cfg.env.env_cfg_yaml_entry_point,
            task_cfg_overrides=cfg.env.task_cfg_overrides,
            bootstrap_timeouts=cfg.env.bootstrap_timeouts, success_path=cfg.env.success_path,
            render_mode='rgb_array')
        raw = env.envs.unwrapped
        assert not raw.cfg.reset.fixed_start_pose and not raw.cfg.reset.fixed_goal_pose
        assert not raw.cfg.reset.fixed_trajectory_file
        assert raw.cfg.termination.max_consecutive_successes == 50
        save_json('actual_task_config.json', raw.cfg.to_dict())
        types = [Path(p).name.split('_', 1)[1].split('_handle_', 1)[0]
                 for p in raw._object_urdf_paths]
        assets = [dict(env_id=i, asset_idx=j, asset_type=types[j],
                       asset_path=str(raw._object_urdf_paths[j]))
                  for i, j in enumerate(raw._object_asset_index_per_env.cpu().tolist())]
        save_json('env_assets.json', assets)
        # Pick four distinct families without screening policy performance.
        selected = []
        for family in ('hammer', 'spatula', 'eraser', 'brush'):
            candidates = [a['env_id'] for a in assets if a['asset_type'] == family]
            if candidates:
                selected.append(candidates[0])
        selected += [i for i in range(args.num_envs) if i not in selected][:4-len(selected)]
        from isaaclab.sensors import Camera, CameraCfg
        import isaaclab.sim as sim_utils
        cameras = [Camera(CameraCfg(
            prim_path=f'/World/DiagnosticCamera{k}', update_period=0,
            height=360, width=480, data_types=['rgb'],
            spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(.1, 100.0)),
        )) for k in range(4)]
        raw.sim.reset()
        for camera, i in zip(cameras, selected):
            origin = raw.scene.env_origins[i]
            eye = origin + torch.tensor((.6, -1.15, 1.05), device=raw.device)
            target = origin + torch.tensor((0, .15, .65), device=raw.device)
            aim_camera(camera, eye.cpu().tolist(), target.cpu().tolist())
        metadata['video_env_ids'] = selected
        writer = imageio.get_writer(str(output / 'paired_play.mp4'), fps=30,
                                    codec='libx264', quality=7, macro_block_size=1)
        snapshots = {}
        original_rewards = raw._get_rewards
        def capture_rewards():
            reward = original_rewards()
            pos = raw.object.data.root_pos_w
            goal = raw.goal_viz.data.root_pos_w
            quat = raw.object.data.root_quat_w
            gquat = raw.goal_viz.data.root_quat_w
            dot = (quat * gquat).sum(-1).abs().clamp(0, 1)
            metrics = {
                'height_delta': pos[:, 2] - raw.scene.env_origins[:, 2] - raw._object_init_z,
                'position_error': torch.linalg.vector_norm(pos - goal, dim=-1),
                'rotation_error_deg': 2 * torch.acos(dot) * (180 / torch.pi),
                'keypoint_error': raw._keypoints_max_dist,
                'lifted': raw._lifted_object,
                'goals': raw._successes,
                'fingertip_min_center_distance': raw._curr_fingertip_distances.min(-1).values,
                'wrist_link_origin_distance': torch.linalg.vector_norm(
                    raw.robot.data.body_pos_w[:, raw._palm_body_id] - pos, dim=-1),
            }
            snapshots.clear()
            snapshots.update({k: v.detach().cpu().numpy().copy() for k, v in metrics.items()})
            return reward
        raw._get_rewards = capture_rewards
        initial_reference = None
        summaries = {}
        agent = None
        for mode in ('deterministic', 'stochastic'):
            raw.seed(args.seed)
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            raw.reward_buf.zero_()
            obs, info = env.reset(random_start_init=False)
            initial = dict(obs=obs.copy(), joint_pos=raw.robot.data.joint_pos.cpu().numpy().copy(),
                           joint_vel=raw.robot.data.joint_vel.cpu().numpy().copy(),
                           object=raw.object.data.root_state_w.cpu().numpy().copy(),
                           goal=raw.goal_viz.data.root_state_w.cpu().numpy().copy(),
                           prev_targets=raw._prev_targets.cpu().numpy().copy(),
                           cur_targets=raw._cur_targets.cpu().numpy().copy())
            np.savez_compressed(output / f'{mode}_initial.npz', **initial)
            if initial_reference is None:
                initial_reference = initial
            else:
                errors = {k: float(np.max(np.abs(v-initial_reference[k]))) for k, v in initial.items()}
                metadata['paired_initial_max_abs_errors'] = errors
                if any(v > 1e-5 for v in errors.values()):
                    raise RuntimeError(f'Initial states not paired: {errors}')
            if agent is None:
                agent = create_agent(env.observation_space, env.action_space, info, cfg.agent)
                saved = torch.load(actor_file, map_location='cpu', weights_only=True)
                weights = {k.removeprefix('_orig_mod.'): v for k, v in saved['network_state_dict'].items()}
                agent._actor.network.load_state_dict(weights, strict=True)
                agent._actor.network.eval().requires_grad_(False)
                model_hash = model_digest(agent._actor.network)
                metadata['model_before'] = model_hash
            agent._cached_noise.zero_()
            agent._cur_noise_repeat_count.zero_()
            agent._cur_noise_repeat_n.fill_(1)
            # Agent construction consumes RNG only on first mode; reseed after it.
            torch.manual_seed(args.seed + 1000)
            torch.cuda.manual_seed_all(args.seed + 1000)
            live = np.ones(args.num_envs, dtype=bool)
            lengths = np.zeros(args.num_envs, dtype=int)
            returns = np.zeros(args.num_envs)
            above_run = np.zeros(args.num_envs, dtype=int)
            above_longest = np.zeros(args.num_envs, dtype=int)
            max_height = np.full(args.num_envs, -np.inf)
            rows, traces = {}, []
            print(f'PLAY_MODE {mode} coverage={sorted(set(a["asset_type"] for a in assets))}', flush=True)
            with torch.no_grad():
                for step in range(args.steps):
                    action = agent.sample_actions(step, {'next_observation': obs}, training=mode == 'stochastic')
                    obs, reward, terminated, truncated, info = env.step(action)
                    if not np.isfinite(obs).all() or not np.isfinite(action).all():
                        raise RuntimeError('Nonfinite inference')
                    assert abs(float(info['current_success_tolerance'])-args.tolerance) < 1e-8
                    done = terminated | truncated
                    lengths += live
                    returns += reward * live
                    max_height[live] = np.maximum(max_height[live], snapshots['height_delta'][live])
                    above_run[live] = np.where(snapshots['height_delta'][live] > .10, above_run[live]+1, 0)
                    above_longest[live] = np.maximum(above_longest[live], above_run[live])
                    traces.append({k: v.copy() for k, v in snapshots.items()} | {'active': live.copy()})
                    for i in np.flatnonzero(live & (done | (step == args.steps-1))):
                        rows[int(i)] = dict(
                            **assets[i], complete=bool(done[i]), steps=int(lengths[i]),
                            return_=float(returns[i]), max_height_delta_m=float(max_height[i]),
                            longest_above_10cm_s=float(above_longest[i]*raw.step_dt),
                            **{k: float(v[i]) for k, v in snapshots.items()},
                            termination_reasons={k: bool(v[i]) for k, v in info.get('episode_final', {}).items()
                                                 if k.startswith('done_')})
                    # Render resets normally; subsequent episodes appear in video
                    # but are excluded from the paired first-episode statistics.
                    if step % 2 == 1:
                        panels = []
                        for camera, i in zip(cameras, selected):
                            camera.update(2*raw.step_dt)
                            rgb = camera.data.output['rgb'][0, :, :, :3].cpu().numpy()
                            panel = Image.fromarray(rgb)
                            draw = ImageDraw.Draw(panel)
                            draw.rectangle((0, 0, 480, 44), fill='black')
                            draw.text((7, 3), f'{mode} | {assets[i]["asset_type"]} env{i} | {(step+1)*raw.step_dt:.1f}s', fill='white')
                            draw.text((7, 20), f'first episode={bool(live[i])}  height={snapshots["height_delta"][i]:.2f}m  goals={int(snapshots["goals"][i])}', fill='white')
                            panels.append(np.asarray(panel))
                        writer.append_data(np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:]))))
                    live &= ~done
                    if (step+1) % 200 == 0:
                        print(f'PLAY_PROGRESS {mode} {step+1}/{args.steps} completed={len(rows)} elapsed={time.monotonic()-start:.0f}s', flush=True)
                records = [rows[i] for i in sorted(rows)]
                save_json(f'{mode}_episodes.json', records)
                np.savez_compressed(output / f'{mode}_traces.npz',
                                    **{k: np.stack([t[k] for t in traces]) for k in traces[0]})
                summaries[mode] = dict(
                    episodes=len(records), completed=sum(r['complete'] for r in records),
                    lifted=sum(r['lifted'] > .5 for r in records),
                    above_10cm_at_least_1s=sum(r['longest_above_10cm_s'] >= 1 for r in records),
                    at_least_one_goal=sum(r['goals'] >= 1 for r in records),
                    total_goals=sum(int(r['goals']) for r in records),
                    mean_goals=float(np.mean([r['goals'] for r in records])),
                    ended_dropped=sum(r['termination_reasons'].get('done_dropped', False) for r in records if r['complete']),
                    ended_timeout=sum(r['termination_reasons'].get('done_timeout', False) for r in records if r['complete']))
                save_json('summary.json', summaries)
                print('PLAY_RESULT ' + json.dumps({mode: summaries[mode]}), flush=True)
        metadata['model_after'] = model_digest(agent._actor.network)
        assert metadata['model_after'] == metadata['model_before']
        metadata['status'] = 'complete'
    except Exception as exc:
        import traceback
        traceback.print_exc()
        metadata.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        metadata['elapsed_seconds'] = time.monotonic()-start
        save_json('metadata.json', metadata)
        if writer is not None:
            writer.close()
        if metadata['status'] == 'complete':
            save_json('play_complete.json', metadata)
        if env is not None:
            env.envs.close()
            env.simulation_app.close(wait_for_replicator=False)


if __name__ == '__main__':
    main()
