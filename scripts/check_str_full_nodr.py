"""CPU config audit and optional GPU random-rollout contract smoke; never trains."""

import argparse
import ast
import json
import os
from pathlib import Path
import random
import sys

from omegaconf import OmegaConf


REPO = Path(__file__).resolve().parents[1]
STR_ROOT = Path(os.environ.get("SIMTOOLREAL_ROOT", REPO.parent / "simtoolreal")).resolve()
TASK_ROOT = STR_ROOT / "isaacsimenvs/tasks/simtoolreal"
TYPES = ["hammer", "screwdriver", "marker", "spatula", "eraser", "brush"]


def source_defaults():
    """Read literal config defaults without importing Isaac Lab or opening a GPU."""
    tree = ast.parse((TASK_ROOT / "simtoolreal_env_cfg.py").read_text())
    sections = {"AssetsCfg": "assets", "ObsCfg": "obs", "ActionCfg": "action", "RewardCfg": "reward",
                "ResetCfg": "reset", "TerminationCfg": "termination", "DomainRandomizationCfg": "domain_randomization"}
    return {sections[node.name]: {
        field.target.id: ast.literal_eval(field.value)
        for field in node.body if isinstance(field, ast.AnnAssign) and field.value is not None
    } for node in tree.body if isinstance(node, ast.ClassDef) and node.name in sections}


def observation_sizes():
    # Only evaluate these repository-owned numeric constants, never its imports.
    names = {"NUM_JOINTS", "NUM_FINGERTIPS", "NUM_KEYPOINTS", "OBS_FIELD_SIZES"}
    values = {}
    for node in ast.parse((TASK_ROOT / "utils/obs_utils.py").read_text()).body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id in names:
            values[node.target.id] = eval(compile(ast.Expression(node.value), "obs constants", "eval"),
                                          {"__builtins__": {}}, values)
    return values["OBS_FIELD_SIZES"]


def asset_pool_counts():
    tree = ast.parse((TASK_ROOT / "utils/object_size_distributions.py").read_text())
    pool = next(node.value for node in tree.body if isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name) and node.target.id == "OBJECT_SIZE_DISTRIBUTIONS")
    names = [ast.literal_eval(keyword.value) for item in pool.elts for keyword in item.keywords if keyword.arg == "type"]
    # num_assets_per_type is actually per geometric subdistribution (12 here),
    # not per unique tool name (6): preserve the original generator semantics.
    return {name: names.count(name) * 100 for name in TYPES}


def load_config(config_name="simtoolreal_full_nodr", num_envs=None):
    import hydra

    OmegaConf.register_new_resolver("eval", lambda expression: eval(expression), replace=True)
    try:
        with hydra.initialize_config_dir(version_base=None, config_dir=str(REPO / "configs")):
            cfg = hydra.compose(config_name=config_name,
                                overrides=[] if num_envs is None else [f"num_train_envs={num_envs}"])
        OmegaConf.resolve(cfg)
    finally:
        OmegaConf.clear_resolver("eval")  # STR registers this after AppLauncher starts.
    original = OmegaConf.merge(source_defaults(), OmegaConf.load(STR_ROOT / "isaacsimenvs/cfg/task/SimToolReal.yaml"))
    task = OmegaConf.merge(original, cfg.env.task_cfg_overrides)
    return cfg, OmegaConf.to_container(original, resolve=True), OmegaConf.to_container(task, resolve=True)


def validate_task(task, original, sizes):
    # The Lab registration defaults are not the official training recipe.
    # Allow exactly the launcher's consecutive-success override, never all
    # termination changes. Goal-count/timeout/tolerance contracts remain strict.
    assert task["termination"]["success_steps"] == 10
    assert task["termination"]["force_consecutive_near_goal_steps"] is True
    for section in ("assets", "obs", "action", "reward", "reset", "termination"):
        for key, expected in original[section].items():
            if section == "termination" and key == "force_consecutive_near_goal_steps":
                expected = True
            actual = task[section][key]
            assert actual == expected, f"Original task changed: {section}.{key}: {actual!r} != {expected!r}"
    for key in ("action_space", "decimation", "episode_length_s"):
        assert task[key] == original[key], f"Original task changed: {key}"
    actor = sum(sizes[name] for name in task["obs"]["obs_list"])
    critic = sum(sizes[name] for name in task["obs"]["state_list"])
    assert (actor, critic, task["action_space"]) == (140, 162, 29), (actor, critic, task["action_space"])
    assert task["assets"]["handle_head_types"] == TYPES
    assert task["assets"]["num_assets_per_type"] == 100
    reset, termination = task["reset"], task["termination"]
    assert reset["fixed_start_pose"] is None and reset["fixed_goal_pose"] is None
    assert not reset["fixed_trajectory_file"] and reset["fixed_trajectory_count"] == 0
    assert reset["goal_sampling_type"] == "delta" and reset["delta_rotation_degrees"] == 90
    assert all(reset[key] > 0 for key in ("reset_position_noise_x", "reset_position_noise_y",
                                         "reset_position_noise_z", "reset_dof_pos_random_interval_arm",
                                         "reset_dof_pos_random_interval_fingers", "reset_dof_vel_random_interval"))
    assert (termination["success_tolerance"], termination["target_success_tolerance"],
            termination["max_consecutive_successes"]) == (.075, .01, 50)
    assert task["action"]["arm_moving_average"] == task["action"]["hand_moving_average"] == .1
    dr = task["domain_randomization"]
    disabled = ("use_obs_delay", "use_action_delay", "use_object_state_delay_noise",
                "object_state_xyz_noise_std", "object_state_rotation_noise_degrees",
                "joint_velocity_obs_noise_std", "force_scale", "torque_scale")
    assert all(not dr[key] for key in disabled), {key: dr[key] for key in disabled}
    for key in ("object_scale_noise_multiplier_range", "object_friction_scale_range", "fingertip_friction_scale_range"):
        assert tuple(dr[key]) == (1., 1.), f"DR enabled: {key}"
    assert not task["scene"]["replicate_physics"] and not task["scene"]["clone_in_fabric"]
    return {"actor_dim": actor, "critic_dim": critic, "combined_dim": actor + critic,
            "action_dim": 29, "asset_pool_size": sum(asset_pool_counts().values()),
            "asset_pool_type_counts": asset_pool_counts(),
            "reward_coefficients_controller_reset_preserved": True,
            "official_consecutive_success_enabled": True,
            "success_steps": 10,
            "dr_disabled": True, "resolved_task_config": task}


def selected_assets(raw):
    indices = raw._object_asset_index_per_env.detach().cpu().tolist()
    paths = raw._object_urdf_paths
    expected = asset_pool_counts()
    assert len(paths) == sum(expected.values()), len(paths)
    pool_types = [Path(path).name.split("_")[1] for path in paths]
    assert {name: pool_types.count(name) for name in TYPES} == expected, "Generated pool type counts differ"
    assert len(indices) == raw.num_envs and min(indices) >= 0 and max(indices) < len(paths)
    selected = [Path(paths[index]).name.split("_")[1] for index in indices]
    counts = {name: selected.count(name) for name in TYPES}
    assert sum(counts.values()) == raw.num_envs, "Unknown selected asset type"
    # Small smoke batches may miss a family; full-size checks require all six.
    # The original generator shuffles the 1200-asset pool before round-robin spawn.
    if raw.num_envs >= 600:
        assert all(counts.values()), counts
    if raw.num_envs >= len(paths):
        assert len(set(indices)) == len(paths), "Not all generated assets were selected"
    return {"pool_size": len(paths), "pool_type_counts": expected,
            "type_counts": counts, "unique_assets_selected": len(set(indices)),
            "six_types_selected": "PASS" if all(counts.values()) else "not_checked_insufficient_envs",
            "all_pool_assets_selected": "PASS" if len(set(indices)) == len(paths) else "not_checked_insufficient_envs"}


def gpu_smoke(cfg, original, sizes, steps, report, save_report):
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(name, "2")
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    sys.path[:0] = [str(REPO), str(STR_ROOT)]
    os.chdir(STR_ROOT)  # Official asset paths are repository-relative.
    import numpy as np
    import torch
    from flash_rl.envs import create_envs

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    cfg.env.headless = True
    env, _, _ = create_envs(**cfg.env)
    try:
        raw = env.envs.unwrapped
        runtime = {section: getattr(raw.cfg, section).to_dict() for section in (
            "assets", "obs", "action", "reward", "reset", "termination", "domain_randomization", "scene")}
        runtime.update({key: getattr(raw.cfg, key) for key in ("action_space", "decimation", "episode_length_s")})
        # Normalize configclass tuples exactly as the CPU OmegaConf path does.
        runtime = json.loads(json.dumps(runtime))
        runtime_report = validate_task(runtime, original, sizes)
        obs, info = env.reset(random_start_init=False)
        assert tuple(info["actor_observation_size"]) == (140,), info
        assert tuple(info["critic_observation_size"]) == (162,), info
        assert info["critic_observation_offset"] == 140, info
        assert env.single_action_space.shape == (29,), env.single_action_space
        count = int(cfg.num_train_envs)
        assert obs.shape == (count, 302) and np.isfinite(obs).all()
        asset_report = selected_assets(raw)
        initial = (raw.object.data.root_pos_w - raw.scene.env_origins).cpu().numpy().copy()
        rotation = raw.object.data.root_quat_w.cpu().numpy().copy()
        goal = (raw.goal_viz.data.root_pos_w - raw.scene.env_origins).cpu().numpy().copy()
        goal_rotation = raw.goal_viz.data.root_quat_w.cpu().numpy().copy()
        env.reset(random_start_init=False)
        second = (raw.object.data.root_pos_w - raw.scene.env_origins).cpu().numpy()
        assert np.max(np.abs(second - initial)) > 1e-5, "Object starts did not change across resets"
        diversity = {}
        for name, values in (("start_position", initial), ("start_orientation", rotation),
                             ("goal_position", goal), ("goal_orientation", goal_rotation)):
            unique = len(np.unique(np.round(values, 5), axis=0))
            if count > 1:
                assert unique > 1, f"No sampled {name} diversity"
            diversity[name + "_unique"] = unique
        assert not np.allclose(goal_rotation, rotation), "Sampled goal rotations equal initial orientations"
        progress = 140 + sum(sizes[name] for name in runtime["obs"]["state_list"][:runtime["obs"]["state_list"].index("progress")])
        terminated_count = truncated_count = 0
        rng = np.random.default_rng(cfg.seed)
        for _ in range(steps):
            actions = rng.uniform(-1, 1, (count, 29)).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(actions)
            assert obs.shape == (count, 302) and np.isfinite(obs).all() and np.isfinite(reward).all()
            assert not np.any(terminated & truncated), "Failure and timeout masks overlap"
            np.testing.assert_array_equal(info["time_outs"], truncated)
            terminated_count += int(terminated.sum())
            truncated_count += int(truncated.sum())
            if np.any(truncated):
                final = info["final_obs"]
                assert final.shape == obs.shape and np.isfinite(final[truncated]).all()
                expected = np.log((raw.max_episode_length - 1) / 10. + 1.)
                np.testing.assert_allclose(final[truncated, progress], expected, atol=1e-5, rtol=0)
                np.testing.assert_allclose(obs[truncated, progress], 0, atol=1e-6, rtol=0)
        result = {"status": "PASS", "num_envs": count, "steps": steps, "terminated": terminated_count,
                  "truncated": truncated_count, "timeout_final_obs": "PASS" if truncated_count else "not_seen",
                  "random_reset_sampling": diversity, "asset_selection": asset_report,
                  "resolved_runtime_task": runtime_report}
        report.update(status="PASS", gpu=result)
        return result
    except BaseException as exc:
        report.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        # SimulationApp.close can terminate the process instead of returning.
        # Persist a complete success/failure audit BEFORE asking it to close.
        save_report()
        env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="simtoolreal_full_nodr")
    parser.add_argument("--output", type=Path, required=True, help="Fresh JSON path; refuses overwrite")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=64)
    args = parser.parse_args(argv)
    if args.num_envs < 1 or args.steps < 1:
        parser.error("--num-envs and --steps must be positive")
    report = {"status": "FAIL", "mode": "gpu_smoke" if args.gpu else "cpu_config", "config": args.config}
    # Reserve the exact requested path before any simulator startup; never overwrite.
    with args.output.expanduser().resolve().open("x", encoding="utf-8") as stream:
        def save_report():
            stream.seek(0)
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.truncate()
            stream.flush()
            os.fsync(stream.fileno())

        try:
            cfg, original, task = load_config(args.config, args.num_envs if args.gpu else None)
            assert set(cfg.env.task_cfg_overrides) == {"domain_randomization", "termination"}, \
                "Only DR and explicit official success semantics overrides are allowed"
            assert OmegaConf.to_container(cfg.env.task_cfg_overrides.termination, resolve=True) == {
                "success_steps": 10, "force_consecutive_near_goal_steps": True,
            }, "Unexpected success/termination override"
            assert cfg.env.env_cfg_yaml_entry_point == "env_cfg_yaml_entry_point"
            assert cfg.env.bootstrap_timeouts and cfg.agent.asymmetric_observation
            assert cfg.agent_load_path is None and cfg.buffer_load_path is None, "Fresh 140D actor required"
            sizes = observation_sizes()
            report["cpu"] = validate_task(task, original, sizes)
            report["resolved_training_config"] = OmegaConf.to_container(cfg, resolve=True)
            if args.gpu:
                gpu_smoke(cfg, original, sizes, args.steps, report, save_report)
            report["status"] = "PASS"
        except BaseException as exc:
            if not (isinstance(exc, SystemExit) and exc.code in (None, 0) and report["status"] == "PASS"):
                report.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            save_report()
            print("PREFLIGHT_RESULT " + json.dumps({"status": report["status"], "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
