"""Headless state-teacher contract check; creates no agent and does not train."""

import argparse
import json
import os
from pathlib import Path
import random
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=700)
    args = parser.parse_args()
    if args.num_envs < 1 or args.steps < 1:
        parser.error("--num-envs and --steps must be positive")
    repo = Path(__file__).resolve().parents[1]
    str_root = Path(os.environ.get("SIMTOOLREAL_ROOT", repo.parent / "simtoolreal")).resolve()
    sys.path[:0] = [str(repo), str(str_root)]
    os.chdir(str_root)  # Official asset paths are relative to the STR repository.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(name, "2")
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

    import hydra
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from flash_rl.envs import create_envs

    OmegaConf.register_new_resolver("eval", lambda expression: eval(expression), replace=True)
    with hydra.initialize_config_dir(version_base=None, config_dir=str(repo / "configs")):
        cfg = hydra.compose(config_name="simtoolreal_state_teacher", overrides=[f"num_train_envs={args.num_envs}"])
    OmegaConf.resolve(cfg)
    OmegaConf.clear_resolver("eval")  # STR registers this name when imported after AppLauncher.
    cfg.env.headless = True
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    env, _, _ = create_envs(**cfg.env)
    try:
        from isaacsimenvs.tasks.simtoolreal.utils.obs_utils import OBS_FIELD_SIZES

        raw = env.envs.unwrapped
        obs, info = env.reset(random_start_init=False)
        assert tuple(raw.cfg.obs.obs_list) == tuple(raw.cfg.obs.state_list)
        assert tuple(info["actor_observation_size"]) == (162,), info
        assert tuple(info["critic_observation_size"]) == (162,), info
        assert info["critic_observation_offset"] == 162, info
        assert env.single_action_space.shape == (29,), env.single_action_space
        assert obs.shape == (args.num_envs, 324), obs.shape
        np.testing.assert_allclose(obs[:, :162], obs[:, 162:], atol=1e-6, rtol=0)
        assert np.isfinite(obs).all(), "Non-finite reset observations"
        fields, offset = {}, 0
        for name in raw.cfg.obs.state_list:
            fields[name] = slice(offset, offset + OBS_FIELD_SIZES[name])
            offset += OBS_FIELD_SIZES[name]
        dr = raw.cfg.domain_randomization
        disabled = {name: getattr(dr, name) for name in (
            "use_obs_delay", "use_action_delay", "use_object_state_delay_noise",
            "joint_velocity_obs_noise_std", "force_scale", "torque_scale")}
        assert not any(disabled.values()), disabled
        for name in ("object_scale_noise_multiplier_range", "object_friction_scale_range", "fingertip_friction_scale_range"):
            assert tuple(getattr(dr, name)) == (1.0, 1.0), name
        assert raw.cfg.action.arm_moving_average == raw.cfg.action.hand_moving_average == 1.0
        assert raw.cfg.termination.success_tolerance == raw.cfg.termination.target_success_tolerance
        assert raw.cfg.termination.max_consecutive_successes == 1
        assert not raw.cfg.reset.fixed_trajectory_file
        goal = np.asarray(raw.cfg.reset.fixed_goal_pose, dtype=np.float32)
        assert goal.shape == (7,), "This diagnostic requires one fixed goal pose"
        origins = raw.scene.env_origins
        initial = (raw.object.data.root_pos_w - origins).cpu().numpy().copy()
        palm = obs[:, fields["palm_pos"]].copy()
        initial_gap = np.linalg.norm(initial - goal[:3], axis=1)
        tolerance_m = raw._current_success_tolerance * raw.cfg.reward.keypoint_scale
        assert (initial_gap > tolerance_m).all(), "Initial state already inside goal position tolerance"
        assert (goal[2] > initial[:, 2]).all(), "Goal must require lifting"
        report = {"actor_dim": 162, "critic_dim": 162, "action_dim": 29,
                  "initial_object_xyz": initial[0].tolist(), "initial_palm_center_xyz": palm[0].tolist(),
                  "initial_object_palm_distance_m": float(np.linalg.norm(initial[0] - palm[0])),
                  "goal_pose_xyz_wxyz": goal.tolist(), "initial_goal_distance_m": float(initial_gap.min()),
                  "keypoint_tolerance_m": tolerance_m, "disabled_dr": disabled,
                  "horizon": int(raw.max_episode_length), "steps": args.steps}
        print("PREFLIGHT_INITIAL " + json.dumps(report), flush=True)
        terminated_count = truncated_count = 0
        actions = np.zeros((args.num_envs, 29), dtype=np.float32)
        for _ in range(args.steps):
            obs, reward, terminated, truncated, info = env.step(actions)
            assert np.isfinite(obs).all() and np.isfinite(reward).all(), "Non-finite step output"
            np.testing.assert_allclose(obs[:, :162], obs[:, 162:], atol=1e-6, rtol=0)
            np.testing.assert_allclose((raw.goal_viz.data.root_pos_w - origins).cpu().numpy(),
                                       np.broadcast_to(goal[:3], (args.num_envs, 3)), atol=2e-5, rtol=0)
            np.testing.assert_allclose(raw.goal_viz.data.root_quat_w.cpu().numpy(),
                                       np.broadcast_to(goal[3:], (args.num_envs, 4)), atol=1e-6, rtol=0)
            assert not np.any(terminated & truncated), "Failure and timeout masks overlap"
            np.testing.assert_array_equal(info["time_outs"], truncated)
            terminated_count += int(terminated.sum())
            truncated_count += int(truncated.sum())
            if np.any(truncated):
                final = info["final_obs"]
                assert final.shape == obs.shape and np.isfinite(final[truncated]).all()
                np.testing.assert_allclose(final[truncated, :162], final[truncated, 162:], atol=1e-6, rtol=0)
                # The captured horizon counter proves final_obs is pre-reset, not new-episode data.
                expected = np.log((raw.max_episode_length - 1) / 10.0 + 1.0)
                np.testing.assert_allclose(final[truncated, fields["progress"]], expected, atol=1e-5, rtol=0)
                np.testing.assert_allclose(obs[truncated, fields["progress"]], 0, atol=1e-6, rtol=0)
        assert truncated_count > 0, "No natural timeout observed; timeout contract remains unverified"
        report.update(status="PASS", terminated=terminated_count, truncated=truncated_count,
                      final_obs_pre_reset_verified=True, clean_actor_equals_critic=True)
        print("PREFLIGHT_RESULT " + json.dumps(report), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
