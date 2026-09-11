"""Independent native-Lab versus FlashSAC-wrapper action-tape diagnostic.

This tests the SAME current STR Lab task through two entry paths, not legacy Gym.
Workers run separately; never trains, changes a checkpoint, or creates replay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path
import random
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
STR = REPO.parent / "simtoolreal"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    """Stable config representation: callable identity, never process addresses."""
    if callable(value):
        return value.__module__ + "." + value.__qualname__
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [canonical(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unstable configuration value: {type(value).__module__}.{type(value).__qualname__}")


def rng_hashes(torch):
    return {
        "python": hashlib.sha256(pickle.dumps(random.getstate())).hexdigest(),
        "numpy": hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest(),
        "torch_cpu": hashlib.sha256(torch.random.get_rng_state().numpy().tobytes()).hexdigest(),
        "torch_cuda": [hashlib.sha256(state.cpu().numpy().tobytes()).hexdigest() for state in torch.cuda.get_rng_state_all()],
    }


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, default=str, allow_nan=False) + "\n")


def control_oracle(action, previous, permutation, arm_lo, arm_hi, hand_lo, hand_hi, dt, speed, arm_ema, hand_ema):
    """Independent NumPy calculation in Lab order, using the task's declared map."""
    lab = np.clip(action, -1, 1)[:, permutation]
    result = previous.copy()
    arm = np.clip(previous[:, :7] + speed * dt * lab[:, :7], arm_lo, arm_hi)
    result[:, :7] = np.clip(arm_ema * arm + (1 - arm_ema) * previous[:, :7], arm_lo, arm_hi)
    hand = hand_lo + (lab[:, 7:] + 1) * (hand_hi - hand_lo) / 2
    result[:, 7:] = np.clip(hand_ema * hand + (1 - hand_ema) * previous[:, 7:], hand_lo, hand_hi)
    return result


def compare_arrays(left, right, atol):
    reports, first = {}, None
    for key in sorted(set(left) | set(right)):
        if key not in left or key not in right:
            reports[key] = {"pass": False, "reason": "missing field"}
            first = first or {"field": key, "reason": "missing field"}
            continue
        a, b = left[key], right[key]
        if a.shape != b.shape:
            reports[key] = {"pass": False, "shape_a": list(a.shape), "shape_b": list(b.shape)}
            first = first or {"field": key, "reason": "shape mismatch"}
            continue
        delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
        bad = ~np.isfinite(delta) | (delta > (0 if a.dtype.kind in "biu" else atol))
        locations = np.argwhere(bad)
        passed = not len(locations)
        reports[key] = {"pass": passed, "max_abs_difference": float(np.max(delta, initial=0)),
                        "different_elements": int(bad.sum())}
        if not passed:
            where = locations[0].tolist()
            candidate = {"field": key, "index": where, "step": 0 if key.startswith("initial/") else where[0] + 1,
                         "native": float(a[tuple(where)]), "other": float(b[tuple(where)])}
            if first is None or candidate["step"] < first.get("step", 10**9):
                first = candidate
    return {"pass": all(v["pass"] for v in reports.values()), "first_difference": first, "fields": reports}


def compare(args):
    left_meta = json.loads((args.native / "metadata.json").read_text())
    right_meta = json.loads((args.other / "metadata.json").read_text())
    checks = {key: left_meta[key] == right_meta[key] for key in
              ("seed", "num_envs", "steps", "task_signature", "assets", "control_parameters", "actor_sha256", "action_tape_sha256", "initial_rng_sha256")}
    with np.load(args.native / "trace.npz") as left, np.load(args.other / "trace.npz") as right:
        result = compare_arrays(left, right, args.atol)
    result.update(metadata_checks=checks, atol=args.atol, rtol=0,
                  native=str(args.native.resolve()), other=str(args.other.resolve()),
                  coverage_native=left_meta["coverage"], coverage_other=right_meta["coverage"],
                  scope="Current independent native STR Lab entry vs adapter; no legacy Gym or real hardware equivalence claim.",
                  caveat="A cross-process mismatch alone is not an adapter bug: run native again with --tape for a native-self baseline.")
    result["pass"] &= all(checks.values()) and left_meta["control_oracle_pass"] and right_meta["control_oracle_pass"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps({k: result[k] for k in ("pass", "metadata_checks", "first_difference", "coverage_native")}, indent=2))
    return 0 if result["pass"] else 1


def worker(args):
    import torch
    import hydra
    from omegaconf import OmegaConf

    sys.path[:0] = [str(REPO), str(STR)]
    os.chdir(STR)
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    torch.set_num_threads(2)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Worker output directory must be empty")
    actor_path = args.checkpoint.resolve() / "actor.pt"
    actor_hash = digest(actor_path)
    write_json(output / "status.json", {"status": "starting", "entry": args.entry, "pid": os.getpid(), "num_envs": args.num_envs, "steps": args.steps})
    print(f"INTERFACE_START entry={args.entry} envs={args.num_envs} steps={args.steps} headless=True cameras=False", flush=True)
    OmegaConf.register_new_resolver("eval", lambda expression: eval(expression), replace=True)
    with hydra.initialize_config_dir(version_base=None, config_dir=str(REPO / "configs")):
        cfg = hydra.compose(config_name="simtoolreal_full_arm1_nstep3", overrides=[f"seed={args.seed}", f"num_train_envs={args.num_envs}"])
    from scripts.eval_str_full_nodr import freeze_tolerance
    freeze_tolerance(cfg, STR / "isaacsimenvs/cfg/task/SimToolReal.yaml", args.tolerance)
    OmegaConf.resolve(cfg)
    OmegaConf.clear_resolver("eval")
    OmegaConf.save(cfg, output / "flash_config.yaml")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env = app = None
    try:
        if args.entry == "native":
            # Deliberately independent: no wrapper factory, parse_env_cfg, or
            # adapter apply_registered_task_overrides calls on this path.
            from isaaclab.app import AppLauncher
            app = AppLauncher(headless=True, device="cuda:0", enable_cameras=False).app
            import gymnasium as gym
            import yaml
            import isaacsimenvs
            from isaacsimenvs.tasks.simtoolreal import SimToolRealEnvCfg
            env_cfg = SimToolRealEnvCfg()
            yaml_path = Path(gym.spec(cfg.env.env_name).kwargs["env_cfg_yaml_entry_point"])
            env_cfg.from_dict(yaml.safe_load(yaml_path.read_text()))
            env_cfg.from_dict({"domain_randomization": {
                "use_obs_delay": False, "use_action_delay": False, "use_object_state_delay_noise": False,
                "object_state_xyz_noise_std": 0.0, "object_state_rotation_noise_degrees": 0.0,
                "object_scale_noise_multiplier_range": [1.0, 1.0], "joint_velocity_obs_noise_std": 0.0,
                "force_scale": 0.0, "torque_scale": 0.0,
                "object_friction_scale_range": [1.0, 1.0], "fingertip_friction_scale_range": [1.0, 1.0]},
                "action": {"arm_moving_average": 1.0},
                "termination": {"success_tolerance": args.tolerance, "target_success_tolerance": args.tolerance,
                                "eval_success_tolerance": None, "tolerance_curriculum_interval": 2**60}})
            env_cfg.sim.device, env_cfg.scene.num_envs, env_cfg.seed = "cuda:0", args.num_envs, args.seed
            env = gym.make(cfg.env.env_name, cfg=env_cfg, render_mode=None)
            raw = env.unwrapped
        else:
            from flash_rl.envs.isaaclab import make_isaaclab_env
            env = make_isaaclab_env(env_name=cfg.env.env_name, num_envs=args.num_envs, seed=args.seed,
                headless=True, device="cuda:0", action_bounds=cfg.env.action_bounds,
                registration_modules=cfg.env.registration_modules, env_cfg_yaml_entry_point=cfg.env.env_cfg_yaml_entry_point,
                task_cfg_overrides=cfg.env.task_cfg_overrides, bootstrap_timeouts=True, success_path=cfg.env.success_path)
            raw = env.envs.unwrapped
        print(f"INTERFACE_SCENE_READY entry={args.entry} step_dt={raw.step_dt} decimation={raw.cfg.decimation}", flush=True)
        write_json(output / "status.json", {"status": "scene_ready", "entry": args.entry, "pid": os.getpid()})
        from flash_rl.agents.flashSAC.network import FlashSACActor
        actor = FlashSACActor(cfg.agent.actor_num_blocks, 140, cfg.agent.actor_hidden_dim, 29).to(raw.device)
        saved = torch.load(actor_path, map_location="cpu", weights_only=True)
        actor.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in saved["network_state_dict"].items()}, strict=True)
        actor.eval().requires_grad_(False)
        from scripts.diagnose_str_rollouts import model_digest
        network_before = model_digest(actor)
        tape = np.load(args.tape, allow_pickle=False) if args.tape else None
        if args.entry == "wrapper" and tape is None:
            raise ValueError("wrapper requires the native --tape actions.npy")
        if tape is not None and tape.shape != (args.steps, args.num_envs, 29):
            raise ValueError(f"Wrong action tape shape: {tape.shape}")
        def cpu(value):
            return value.detach().cpu().numpy().copy()
        def pack(value):
            return cpu(torch.cat((value["policy"], value["critic"]), -1)) if isinstance(value, dict) else np.asarray(value).copy()
        def state():
            return {"joint_pos": cpu(raw.robot.data.joint_pos), "joint_vel": cpu(raw.robot.data.joint_vel),
                    "object": cpu(raw.object.data.root_state_w), "goal": cpu(raw.goal_viz.data.root_state_w),
                    "prev_targets": cpu(raw._prev_targets), "cur_targets": cpu(raw._cur_targets),
                    "successes": cpu(raw._successes), "lifted": cpu(raw._lifted_object),
                    "episode_length": cpu(raw.episode_length_buf), "pending_goal_reset": cpu(raw._pending_goal_reset),
                    "object_init_z": cpu(raw._object_init_z)}
        raw.seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        raw.reward_buf.zero_()
        observation, _ = env.reset() if args.entry == "native" else env.reset(random_start_init=False)
        observation = pack(observation)
        initial_rng = rng_hashes(torch)
        print(f"INTERFACE_RESET_READY observation_shape={observation.shape}", flush=True)
        initial = {"initial/" + k: v for k, v in state().items()} | {"initial/observation": observation.copy()}
        parameters = {"permutation": cpu(raw._perm_canon_to_lab).tolist(),
                      "arm_lo": cpu(raw._arm_lower).tolist(), "arm_hi": cpu(raw._arm_upper).tolist(),
                      "hand_lo": cpu(raw._hand_lower).tolist(), "hand_hi": cpu(raw._hand_upper).tolist(),
                      "dt": float(raw.step_dt), "sim_dt": float(raw.cfg.sim.dt), "decimation": int(raw.cfg.decimation),
                      "speed": raw.cfg.action.dof_speed_scale,
                      "arm_ema": raw.cfg.action.arm_moving_average, "hand_ema": raw.cfg.action.hand_moving_average}
        assert list(raw._arm_joint_ids) == list(range(7)), "Oracle assumes current STR Lab arm-first joint order"
        assert list(raw._hand_joint_ids) == list(range(7, 29)), "Oracle assumes current STR Lab hand-last joint order"
        assert raw.cfg.action.hand_moving_average == .1
        assert cfg.env.action_bounds == 1.0
        actual_cfg = canonical(raw.cfg.to_dict())
        write_json(output / "actual_task_config.json", actual_cfg)
        signature = {k: actual_cfg[k] for k in ("assets", "obs", "action", "reward", "reset", "termination", "domain_randomization", "sim", "decimation")}
        signature["scene_topology"] = {k: actual_cfg["scene"][k] for k in ("num_envs", "env_spacing", "replicate_physics", "clone_in_fabric")}
        assets = [Path(raw._object_urdf_paths[i]).name for i in cpu(raw._object_asset_index_per_env).tolist()]
        captured = {}
        original_pre, original_apply, original_reward = raw._pre_physics_step, raw._apply_action, raw._get_rewards
        def pre(actions):
            captured["received_action"] = cpu(actions)
            previous = cpu(raw._prev_targets)
            original_pre(actions)
            captured["applied_targets"] = cpu(raw._cur_targets)
            oracle_args = {k: v for k, v in parameters.items() if k not in ("sim_dt", "decimation")}
            for k in ("permutation", "arm_lo", "arm_hi", "hand_lo", "hand_hi"):
                oracle_args[k] = np.asarray(oracle_args[k], dtype=np.int64 if k == "permutation" else np.float32)
            captured["control_oracle"] = control_oracle(cpu(actions), previous, **oracle_args)
        def apply():
            original_apply()
            captured["apply_calls"] = captured.get("apply_calls", 0) + 1
            captured["physx_joint_targets"] = cpu(raw.robot.data.joint_pos_target)
        def reward_hook():
            result = original_reward()
            captured.update({"pre_reset/" + k: v for k, v in state().items()})
            captured.update({"reward/" + k: cpu(v) for k, v in raw._reward_terms.items()})
            captured.update({"reason/" + k: cpu(v) for k, v in raw._termination_reasons.items()})
            return result
        raw._pre_physics_step, raw._apply_action, raw._get_rewards = pre, apply, reward_hook
        records, actions_out = {}, []
        coverage = {"terminated": 0, "truncated": 0, "goal_hits": 0, "final_obs_rows": 0}
        max_oracle_error = 0.0
        with torch.inference_mode():
            for step in range(args.steps):
                if tape is None:
                    mean, _ = actor.get_mean_and_std(torch.as_tensor(observation[:, :140], device=raw.device), training=False)
                    action = cpu(torch.tanh(mean))
                else:
                    action = tape[step].copy()
                actions_out.append(action)
                captured.clear()
                if args.entry == "native":
                    result = env.step(torch.as_tensor(np.clip(action, -1, 1), device=raw.device))
                    obs_dict, rew, terminated, truncated, info = result
                    observation, rew, terminated, truncated = pack(obs_dict), cpu(rew), cpu(terminated), cpu(truncated)
                else:
                    observation, rew, terminated, truncated, info = env.step(action)
                done = terminated | truncated
                valid = np.zeros(args.num_envs, dtype=bool)
                final = np.zeros_like(observation)
                if done.any():
                    if "final_obs" not in info or "_final_obs" not in info:
                        raise RuntimeError("Missing pre-reset final observation contract")
                    mask = info["_final_obs"]
                    valid = cpu(mask) if torch.is_tensor(mask) else np.asarray(mask)
                    assert np.array_equal(valid, done)
                    final[valid] = pack(info["final_obs"])[valid]
                values = {"action": action, "observation": observation.copy(), "reward": rew.copy(),
                          "terminated": terminated.copy(), "truncated": truncated.copy(),
                          "final_obs_mask": valid.copy(), "final_observation": final,
                          "apply_calls": np.asarray(captured.pop("apply_calls"), dtype=np.int64), **captured,
                          **{"post_reset/" + k: v for k, v in state().items()}}
                assert int(values["apply_calls"]) == raw.cfg.decimation
                assert np.isfinite(observation).all()
                max_oracle_error = max(max_oracle_error, float(np.max(np.abs(values["applied_targets"] - values["control_oracle"]))))
                for key, value in values.items():
                    records.setdefault(key, []).append(value)
                coverage["terminated"] += int(terminated.sum())
                coverage["truncated"] += int(truncated.sum())
                coverage["goal_hits"] += int(values["pre_reset/pending_goal_reset"].sum())
                coverage["final_obs_rows"] += int(valid.sum())
                if (step + 1) % 100 == 0:
                    print(f"{args.entry}: {step + 1}/{args.steps} {coverage}", flush=True)
                    write_json(output / "status.json", {"status": "running", "entry": args.entry, "step": step + 1, "steps": args.steps, "coverage": coverage})
        np.save(output / "actions.npy", np.asarray(actions_out))
        np.savez(output / "trace.npz", **initial, **{k: np.stack(v) for k, v in records.items()})
        assert network_before == model_digest(actor) and actor_hash == digest(actor_path)
        metadata = {"entry": args.entry, "seed": args.seed, "num_envs": args.num_envs, "steps": args.steps,
                    "checkpoint": str(actor_path), "actor_sha256": actor_hash,
                    "action_tape_sha256": digest(output / "actions.npy"), "tape_source": str(args.tape),
                    "initial_rng_sha256": initial_rng, "final_rng_sha256": rng_hashes(torch),
                    "task_signature": signature, "control_parameters": parameters, "assets": assets,
                    "coverage": coverage, "control_oracle_max_abs_error": max_oracle_error,
                    "control_oracle_atol": 1e-6, "control_oracle_pass": max_oracle_error <= 1e-6,
                    "learning": False, "reset_random_start_init": False,
                    "scope": "Independent native Lab config+gym.make versus FlashSAC adapter; shared STR task/physics, not legacy Gym.",
                    "limitations": ["Fixed action tape checks environment interface, not policy quality or long-term learning.",
                                    "Reset/start protocol intentionally disables training-only randomized initial episode counters.",
                                    "Coverage must contain relevant events before claiming those branches were tested.",
                                    "Final-observation invalid rows are zeroed rather than comparing stale buffer contents."],
                    "source_sha256": {str(p): digest(p) for p in (Path(__file__), STR / "isaacsimenvs/cfg/task/SimToolReal.yaml",
                         STR / "isaacsimenvs/tasks/simtoolreal/simtoolreal_env.py", REPO / "flash_rl/envs/isaaclab.py")}}
        write_json(output / "metadata.json", metadata)
        write_json(output / "status.json", {"status": "complete", "entry": args.entry, "step": args.steps, "coverage": coverage})
        print(json.dumps({"output": str(output), "coverage": coverage, "control_oracle_max_abs_error": max_oracle_error}))
    finally:
        if env is not None:
            env.close()
        if app is not None:
            app.close()


def self_test():
    old = np.zeros((2, 29), dtype=np.float32)
    action = np.zeros_like(old)
    action[0, :7], action[1, 7:] = 2, -2
    result = control_oracle(action, old, np.arange(29), np.full(7, -1), np.ones(7),
                            np.full(22, -1), np.ones(22), .02, 1.5, 1, .1)
    np.testing.assert_allclose(result[0, :7], .03)
    np.testing.assert_allclose(result[1, 7:], -.1)
    left = {"initial/observation": np.zeros((2, 3)), "observation": np.zeros((3, 2, 3)), "terminated": np.zeros((3, 2), bool)}
    right = {k: v.copy() for k, v in left.items()}
    assert compare_arrays(left, right, 1e-6)["pass"]
    right["observation"][1, 0, 0] = 1e-4
    report = compare_arrays(left, right, 1e-6)
    assert not report["pass"] and report["first_difference"]["step"] == 2
    right["terminated"][0, 0] = True
    assert compare_arrays(left, right, 1e-6)["first_difference"]["step"] == 1
    assert canonical({"func": control_oracle, "tuple": (1, 2)}) == {"func": __name__ + ".control_oracle", "tuple": [1, 2]}
    print("CPU self-test passed: action clipping/scaling/EMA oracle, exact comparison, first-difference negative controls.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    run = subs.add_parser("worker")
    run.add_argument("--entry", choices=("native", "wrapper"), required=True)
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--tape", type=Path)
    run.add_argument("--num-envs", type=int, default=32)
    run.add_argument("--steps", type=int, default=600)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--tolerance", type=float, default=.029056536675)
    diff = subs.add_parser("compare")
    diff.add_argument("--native", type=Path, required=True)
    diff.add_argument("--other", type=Path, required=True)
    diff.add_argument("--output", type=Path, required=True)
    diff.add_argument("--atol", type=float, default=1e-6)
    subs.add_parser("self-test")
    args = parser.parse_args()
    if args.command == "self-test":
        self_test()
    elif args.command == "compare":
        raise SystemExit(compare(args))
    else:
        worker(args)


if __name__ == "__main__":
    main()

