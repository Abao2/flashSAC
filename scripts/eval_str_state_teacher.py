"""Headless deterministic evaluation of the fixed STR state-teacher diagnostic."""

import argparse
import json
import os
from pathlib import Path
import random
import sys
import warnings


def pose_errors(state, fields, goal, offsets, clip):
    """Recover pose from centered observation keypoints; return meters/radians."""
    import numpy as np

    rel = state[fields["keypoints_rel_goal"]].reshape(-1, 3)
    quat = state[fields["object_rot"]][[3, 0, 1, 2]]  # Observation xyzw -> wxyz.
    if not np.isfinite(rel).all() or np.any(np.abs(rel) >= clip - 1e-6):
        raise ValueError("terminal keypoints missing/nonfinite/clipped; pose cannot be recovered")
    if not np.isfinite(quat).all() or np.linalg.norm(quat) < 1e-8:
        raise ValueError("terminal object quaternion invalid")
    quat = quat / np.linalg.norm(quat)
    goal_quat = np.asarray(goal[3:]) / np.linalg.norm(goal[3:])
    delta = rel.mean(axis=0)  # KEYPOINT_CORNERS has exactly zero centroid.

    def rotate(q):
        return offsets + 2 * np.cross(q[1:], np.cross(q[1:], offsets) + q[0] * offsets)

    angle = 2 * np.arccos(np.clip(abs(np.dot(quat, goal_quat)), 0, 1))
    keypoint_error = np.linalg.norm(delta + rotate(quat) - rotate(goal_quat), axis=-1).max()
    return float(np.linalg.norm(delta)), float(angle), float(keypoint_error), float(goal[2] + delta[2])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Checkpoint directory containing actor.pt")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=16)
    args = parser.parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    if args.episodes < 1 or args.num_envs < 1:
        parser.error("--episodes and --num-envs must be positive")
    if not (checkpoint / "actor.pt").is_file():
        parser.error(f"actor.pt not found in {checkpoint}")
    repo = Path(__file__).resolve().parents[1]
    str_root = Path(os.environ.get("SIMTOOLREAL_ROOT", repo.parent / "simtoolreal")).resolve()
    sys.path[:0] = [str(repo), str(str_root)]
    os.chdir(str_root)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(name, "2")
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

    import hydra
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from flash_rl.agents import create_agent
    from flash_rl.envs.isaaclab import make_isaaclab_env

    OmegaConf.register_new_resolver("eval", lambda expression: eval(expression), replace=True)
    with hydra.initialize_config_dir(version_base=None, config_dir=str(repo / "configs")):
        cfg = hydra.compose(config_name="simtoolreal_state_teacher", overrides=[f"num_train_envs={args.num_envs}"])
    # Evaluation only: no replay is populated, no optimizer or reward statistics are loaded.
    cfg.agent.buffer_max_length = cfg.agent.buffer_min_length = cfg.agent.sample_batch_size = 1
    cfg.agent.use_compile = cfg.agent.load_optimizer = cfg.agent.load_reward_normalizer = False
    OmegaConf.resolve(cfg)
    OmegaConf.clear_resolver("eval")  # STR registers this name after AppLauncher starts.
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    env = make_isaaclab_env(
        env_name=cfg.env.env_name, num_envs=args.num_envs, seed=cfg.seed, headless=True,
        device=cfg.env.device, action_bounds=cfg.env.action_bounds,
        registration_modules=cfg.env.registration_modules,
        env_cfg_yaml_entry_point=cfg.env.env_cfg_yaml_entry_point,
        task_cfg_overrides=cfg.env.task_cfg_overrides,
        bootstrap_timeouts=cfg.env.bootstrap_timeouts, success_path=cfg.env.success_path,
    )
    try:
        from isaacsimenvs.tasks.simtoolreal.utils.obs_utils import OBS_FIELD_SIZES, KEYPOINT_CORNERS

        obs, info = env.reset(random_start_init=False)
        raw = env.envs.unwrapped
        assert tuple(raw.cfg.obs.obs_list) == tuple(raw.cfg.obs.state_list)
        assert tuple(info["actor_observation_size"]) == (162,), info
        agent = create_agent(env.observation_space, env.action_space, info, cfg.agent)
        # Generic loader cannot read compiled prefixes into a noncompiled module.
        saved = torch.load(checkpoint / "actor.pt", map_location="cpu", weights_only=True)
        weights = {k.removeprefix("_orig_mod."): v for k, v in saved["network_state_dict"].items()}
        agent._actor.network.load_state_dict(weights, strict=True)
        fields, start = {}, int(info["critic_observation_offset"])
        for name in raw.cfg.obs.state_list:
            fields[name] = slice(start, start + OBS_FIELD_SIZES[name])
            start += OBS_FIELD_SIZES[name]
        goal = np.asarray(raw.cfg.reset.fixed_goal_pose, dtype=float)
        assert goal.shape == (7,) and raw.cfg.termination.max_consecutive_successes == 1
        assert np.allclose(np.asarray(KEYPOINT_CORNERS).mean(axis=0), 0)
        assert raw.cfg.reward.fixed_size_keypoint_reward, "Expected fixed-size success keypoints"
        offsets = np.asarray(KEYPOINT_CORNERS) * np.asarray(raw.cfg.reward.fixed_size) * raw.cfg.reward.keypoint_scale / 2
        initial_z = float(raw.cfg.reset.fixed_start_pose[2])
        returns, lengths = np.zeros(args.num_envs), np.zeros(args.num_envs, dtype=int)
        dropped = np.zeros(args.num_envs, dtype=bool)
        drop_known = np.ones(args.num_envs, dtype=bool)
        rows, missing = [], set()
        steps = 0
        while len(rows) < args.episodes:
            actions = agent.sample_actions(0, {"next_observation": obs}, training=False)
            obs, reward, terminated, truncated, info = env.step(actions)
            if not np.isfinite(obs).all() or not np.isfinite(reward).all():
                raise RuntimeError("Nonfinite evaluation observations/rewards")
            returns += reward
            lengths += 1
            done = terminated | truncated
            final = info.get("final_obs")
            final_info = info.get("episode_final", {})
            for i in range(args.num_envs):
                state = final[i] if done[i] and final is not None else (None if done[i] else obs[i])
                metrics = None
                lift_value = np.nan if state is None else state[fields["lifted_object"]][0]
                lifted = bool(lift_value > .5) if np.isfinite(lift_value) else None
                if lifted is None:
                    drop_known[i] = False
                if state is not None:
                    try:
                        metrics = pose_errors(state, fields, goal, offsets, raw.cfg.obs.clamp_abs_observations)
                        dropped[i] |= bool(lifted) and metrics[3] < initial_z
                    except ValueError as error:
                        missing.add(str(error))
                        drop_known[i] = False
                if done[i] and len(rows) < args.episodes:
                    success = final_info.get("all_goals_hit")
                    if state is None or metrics is None or success is None:
                        missing.add("incomplete pre-reset terminal metrics; unavailable values reported as null")
                    rows.append({"episode": len(rows) + 1, "env_id": i, "return": float(returns[i]),
                                 "length": int(lengths[i]), "success": None if success is None else bool(success[i]),
                                 "lifted": lifted, "dropped": bool(dropped[i]) if drop_known[i] else None,
                                 "position_error_m": None if metrics is None else metrics[0],
                                 "angular_error_rad": None if metrics is None else metrics[1],
                                 "keypoint_error_m": None if metrics is None else metrics[2]})
                    print("EVAL_EPISODE " + json.dumps(rows[-1], allow_nan=False), flush=True)
            returns[done], lengths[done], dropped[done] = 0, 0, False
            drop_known[done] = True
            steps += 1
            if steps > args.episodes * int(raw.max_episode_length):
                raise RuntimeError("Evaluation exceeded expected episode horizon; check termination contract")
        for message in sorted(missing):
            warnings.warn(message)
        summary = {"checkpoint": str(checkpoint), "episodes": len(rows), "deterministic": True,
                   "note": "Fixed reset repeats one task; not a generalization estimate. Angles in radians.",
                   "drop_definition": "ever lifted then object center below its initial reset height",
                   "missing_metrics": sorted(missing)}
        for name in ("success", "lifted", "dropped", "return", "length", "position_error_m", "angular_error_rad", "keypoint_error_m"):
            values = [row[name] for row in rows if row[name] is not None]
            summary[name] = {"mean": float(np.mean(values)) if values else None, "count": len(values)}
        print("EVAL_RESULT " + json.dumps(summary, allow_nan=False), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
