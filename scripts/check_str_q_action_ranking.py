#!/usr/bin/env python3
"""Frozen full-STR Q/action probe. No optimizer, replay writes or training.

Re-create states by seeded reset + identical action-prefix replay, then gate on
physical/controller/goal/tracker alignment. A one-step intervention is followed
by common Gaussian policy draws. Finite soft return is NOT exact Q calibration.
"""
from __future__ import annotations

import argparse
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import pickle
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.diagnose_str_critic import checkpoint_models, reward_denominator
from scripts.diagnose_str_first_action import return_increments
from scripts.diagnose_str_rollouts import create_env, model_digest, step_env
from flash_rl.agents.utils.distribution import safe_tanh_log_det_jacobian

BRANCHES = ("mean", "mean_repeat", "zero", "arm_q_plus", "arm_q_minus", "hand_q_plus", "hand_q_minus")
STATE_BUFFERS = ("_prev_targets", "_cur_targets", "_action_queue", "_obs_queue", "_object_state_queue",
    "_lifted_object", "_closest_keypoint_max_dist", "_closest_fingertip_dist", "_successes", "_near_goal_steps",
    "_pending_goal_reset", "_object_init_z", "_table_z_per_env", "_object_forces", "_object_torques",
    "_object_scale_multiplier", "_prev_episode_successes", "episode_length_buf", "reward_buf")


def cpu(tensor):
    return tensor.detach().cpu().numpy().copy()


def snapshot(raw, obs, extended=False):
    state = {name: cpu(getattr(raw, name)) for name in STATE_BUFFERS}
    state.update(obs=np.array(obs, copy=True), robot=cpu(raw.robot.data.root_state_w),
        joint_pos=cpu(raw.robot.data.joint_pos), joint_vel=cpu(raw.robot.data.joint_vel),
        object=cpu(raw.object.data.root_state_w), goal=cpu(raw.goal_viz.data.root_state_w),
        table=cpu(raw.table.data.root_state_w))
    if extended:
        state["body_state_w"] = cpu(raw.robot.data.body_state_w)
        for name in ("joint_pos_target", "joint_vel_target", "joint_effort_target", "computed_torque", "applied_torque"):
            state[f"robot/{name}"] = cpu(getattr(raw.robot.data, name))
    return state


def checked_hard_reset(raw, reference, joint_names, body_names, apply_materials):
    # Lazy import avoids a top-level cycle with the standalone baseline check.
    from scripts.check_str_hardreset_replay import properties, property_errors
    raw.sim.reset(soft=False)
    apply_materials(raw)
    actual = properties(raw)
    errors = property_errors(reference, actual)
    names_equal = joint_names == list(raw.robot.data.joint_names) and body_names == list(raw.robot.data.body_names)
    if not names_equal or max(errors.values()) > 1e-6:
        raise RuntimeError(f"Hard reset changed physics/order: names_equal={names_equal}, errors={errors}")
    return actual, errors


def continuation_seeds(initial_seed, prefix, requested):
    """Without an explicit list retain the original per-prefix noise stream."""
    return requested if requested is not None else [initial_seed + 30000 + prefix]


def comparison_key(prefix, seed, branch):
    # Legacy rows retain their original group keys when summarized again.
    return f"{prefix}/{branch}" if seed is None else f"{prefix}/seed{seed}/{branch}"


def rng_hashes():
    states = {"python": pickle.dumps(random.getstate()), "numpy": pickle.dumps(np.random.get_state()),
              "torch_cpu": torch.random.get_rng_state().numpy().tobytes()}
    if torch.cuda.is_initialized():
        states.update({f"torch_cuda_{i}": state.cpu().numpy().tobytes()
                       for i, state in enumerate(torch.cuda.get_rng_state_all())})
    return {key: hashlib.sha256(value).hexdigest() for key, value in states.items()}


def alignment(reference, current, atol):
    n = len(reference["obs"])
    valid, errors = np.ones(n, bool), {}
    for key, expected in reference.items():
        actual = current[key]
        assert actual.shape == expected.shape
        error = np.abs(actual.astype(float) - expected.astype(float)).reshape(n, -1).max(1)
        valid &= np.isfinite(error) & (error <= atol)
        errors[key] = float(error.max())
    return valid, errors


def candidates(actor, critic, actor_obs, critic_obs, epsilon):
    with torch.no_grad():
        baseline = actor.get_mean_and_std(actor_obs, training=False)[0].tanh()
    action = baseline.detach().requires_grad_(True)
    q, _ = critic(critic_obs, action, training=False)
    gradient = torch.autograd.grad(q.min(0).values.sum(), action)[0].detach()
    result = {"mean": baseline, "mean_repeat": baseline.clone(), "zero": torch.zeros_like(baseline)}
    for part, indices in (("arm", slice(0, 7)), ("hand", slice(7, 29))):
        direction = torch.zeros_like(gradient)
        group = gradient[:, indices]
        direction[:, indices] = group / group.norm(dim=-1, keepdim=True).clamp_min(1e-20)
        for label, sign in (("plus", 1), ("minus", -1)):
            result[f"{part}_q_{label}"] = (baseline + sign * epsilon * direction).clamp(-1, 1)
    with torch.no_grad():
        scores = {name: cpu(critic(critic_obs, result[name], training=False)[0].min(0).values) for name in BRANCHES}
    return result, scores, cpu(gradient.norm(dim=-1))


def sampled_action(actor, obs, generator):
    with torch.no_grad():
        mean, std = actor.get_mean_and_std(obs, training=False)
        raw = mean + std * torch.randn(mean.shape, generator=generator, device=obs.device)
        logp = (torch.distributions.Normal(mean, std).log_prob(raw) - safe_tanh_log_det_jacobian(raw)).sum(-1)
    return cpu(raw.tanh()), cpu(logp)


def probe_return_increments(reward, logp, active, step, gamma, denominator, alpha, n_step):
    """Dense soft metric plus the unrolled n-step operator's entropy grid."""
    if n_step < 1:
        raise ValueError("n_step must be positive")
    raw, normalized, dense = return_increments(reward, logp, active, step, gamma, denominator, alpha)
    nstep_soft = dense if step > 0 and step % n_step == 0 else normalized
    return raw, normalized, dense, nstep_soft


def comparisons(rows):
    """Paired deltas only; baseline repeat exposes remaining simulator noise."""
    indexed = {(r["prefix_steps"], r.get("continuation_seed"), r["branch"], r["env_id"]): r for r in rows}
    if len(indexed) != len(rows):
        raise ValueError("Duplicate prefix/continuation-seed/branch/env result")
    result = {}
    groups = sorted({(r["prefix_steps"], r.get("continuation_seed")) for r in rows},
                    key=lambda item: (item[0], -1 if item[1] is None else item[1]))
    for prefix, seed in groups:
        selected = [r for r in rows if r["prefix_steps"] == prefix and r.get("continuation_seed") == seed]
        for branch in dict.fromkeys(r["branch"] for r in selected):
            pairs = [(r, indexed[prefix, seed, "mean", r["env_id"]]) for r in selected
                     if r["branch"] == branch and r["paired"]
                     and indexed[prefix, seed, "mean", r["env_id"]]["paired"]]
            report = {"paired_count": len(pairs)}
            if "n_step" in selected[0]:
                report["n_step"] = int(selected[0]["n_step"])
            if pairs:
                valid = [a for a, _ in pairs]
                report.update(horizon_censored_count=sum(r["horizon_censored"] for r in valid),
                    truncated_count=sum(r["truncated"] for r in valid),
                    terminated_count=sum(r["terminated"] for r in valid))
                tails = [r["unknown_tail_discount"] for r in valid if not r["terminated"]]
                report["unknown_tail_count"] = len(tails)
                report["unknown_tail_discount"] = {"min": min(tails), "max": max(tails), "mean": float(np.mean(tails))} if tails else None
                fields = ["initial_q", "finite_soft_return", "discounted_raw_return", "goals_gained",
                          "max_height_delta", "above10cm_longest_seconds"]
                if all("finite_nstep_soft_return" in r for r in selected):
                    fields.append("finite_nstep_soft_return")
                for field in fields:
                    delta = np.array([a[field] - b[field] for a, b in pairs])
                    report[field] = {"mean_difference": float(delta.mean()), "max_abs_difference": float(abs(delta).max())}
                dq = np.array([a["initial_q"] - b["initial_q"] for a, b in pairs])
                dr = np.array([a["finite_soft_return"] - b["finite_soft_return"] for a, b in pairs])
                informative = (abs(dq) > 1e-6) & (abs(dr) > 1e-4)
                report["non_tied_count"] = int(informative.sum())
                report["q_return_direction_agreement"] = float((dq[informative] * dr[informative] > 0).mean()) if informative.any() else None
                if "finite_nstep_soft_return" in fields:
                    dn = np.array([a["finite_nstep_soft_return"] - b["finite_nstep_soft_return"] for a, b in pairs])
                    informative_n = (abs(dq) > 1e-6) & (abs(dn) > 1e-4)
                    report["nstep_non_tied_count"] = int(informative_n.sum())
                    report["q_nstep_return_direction_agreement"] = float((dq[informative_n] * dn[informative_n] > 0).mean()) if informative_n.any() else None
            result[comparison_key(prefix, seed, branch)] = report
    return result


def self_test():
    from flash_rl.agents.flashSAC.network import FlashSACActor, FlashSACDoubleCritic
    torch.set_num_threads(2)
    torch.manual_seed(0)
    actor = FlashSACActor(1, 140, 8, 29).requires_grad_(False)
    critic = FlashSACDoubleCritic(1, 191, 8, 11, -5., 5.).requires_grad_(False)
    before = model_digest(actor), model_digest(critic)
    obs, state = torch.randn(4, 140), torch.randn(4, 162)
    actions, q, gradient = candidates(actor, critic, obs, state, .2)
    assert set(actions) == set(BRANCHES) and gradient.shape == (4,)
    for key in BRANCHES:
        assert actions[key].shape == (4, 29) and torch.all(actions[key].abs() <= 1) and np.isfinite(q[key]).all()
    assert torch.equal(actions["mean"], actions["mean_repeat"])
    assert torch.equal(actions["arm_q_plus"][:, 7:], actions["mean"][:, 7:])
    assert torch.equal(actions["hand_q_minus"][:, :7], actions["mean"][:, :7])
    for a, b in zip(sampled_action(actor, obs, torch.Generator().manual_seed(7)),
                    sampled_action(actor, obs, torch.Generator().manual_seed(7))):
        np.testing.assert_array_equal(a, b)
    ref = {"obs": np.zeros((4, 2)), "tracker": np.zeros(4, bool)}
    alt = {k: v.copy() for k, v in ref.items()}
    alt["tracker"][1] = True
    mask, _ = alignment(ref, alt, 1e-5)
    np.testing.assert_array_equal(mask, [True, False, True, True])
    rows = []
    for branch in BRANCHES:
        value = float(branch == "arm_q_plus")
        rows.append(dict(prefix_steps=0, branch=branch, env_id=0, paired=True,
            initial_q=value, finite_soft_return=2 * value, discounted_raw_return=value,
            goals_gained=0, max_height_delta=0., above10cm_longest_seconds=0.,
            horizon_censored=True, truncated=False, terminated=False, unknown_tail_discount=.99**300))
    report = comparisons(rows)
    assert report["0/arm_q_plus"]["q_return_direction_agreement"] == 1.
    assert report["0/mean_repeat"]["non_tied_count"] == 0
    assert report["0/mean"]["horizon_censored_count"] == 1
    assert report["0/mean"]["unknown_tail_discount"]["max"] == .99**300
    rows[3]["paired"] = False
    assert comparisons(rows)["0/arm_q_plus"]["paired_count"] == 0
    assert continuation_seeds(7, 60, None) == [30067]
    assert continuation_seeds(7, 60, [0, 19]) == [0, 19]
    multi = []
    for noise_seed in (7, 19):
        for row in rows:
            item = dict(row, continuation_seed=noise_seed, paired=True)
            item["finite_soft_return"] += 1000 * noise_seed
            if item["branch"] == "arm_q_plus":
                item["finite_soft_return"] += float(noise_seed)
            item["n_step"] = 3
            item["finite_nstep_soft_return"] = item["finite_soft_return"] + 100
            multi.append(item)
    report = comparisons(multi)
    assert report["0/seed7/arm_q_plus"]["finite_soft_return"]["mean_difference"] == 9.
    assert report["0/seed19/arm_q_plus"]["finite_soft_return"]["mean_difference"] == 21.
    assert report["0/seed19/mean_repeat"]["finite_soft_return"]["max_abs_difference"] == 0.
    assert report["0/seed19/arm_q_plus"]["finite_nstep_soft_return"]["mean_difference"] == 21.
    assert report["0/seed19/arm_q_plus"]["n_step"] == 3
    assert report["0/seed19/arm_q_plus"]["q_nstep_return_direction_agreement"] == 1.
    sums = np.zeros((4, 2))
    for step in range(7):
        inputs = (np.array([2., 99.]), np.array([-3., -9.]), np.array([True, False]), step, .5, 2., .1)
        values = probe_return_increments(*inputs, 3)
        sums += np.stack(values)
        n1 = probe_return_increments(*inputs, 1)
        np.testing.assert_array_equal(n1[2], n1[3])
    expected_rewards = sum(.5**step for step in range(7))
    expected_grid_entropy = .3 * (.5**3 + .5**6)
    expected_dense_entropy = .3 * sum(.5**step for step in range(1, 7))
    np.testing.assert_allclose(sums[3], [expected_rewards + expected_grid_entropy, 0.])
    np.testing.assert_allclose(sums[2], [expected_rewards + expected_dense_entropy, 0.])

    try:
        comparisons(multi + multi[:1])
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate seed/branch pair accepted")
    from types import SimpleNamespace
    from unittest.mock import patch
    calls = []
    dummy = SimpleNamespace(sim=SimpleNamespace(reset=lambda **kw: calls.append(("reset", kw))),
        robot=SimpleNamespace(data=SimpleNamespace(joint_names=["joint"], body_names=["body"])))
    physics = {"mass": np.array([1., 2.])}
    with patch("scripts.check_str_hardreset_replay.properties", return_value=physics):
        _, error = checked_hard_reset(dummy, physics, ["joint"], ["body"], lambda raw: calls.append("materials"))
        assert error == {"mass": 0.} and calls == [("reset", {"soft": False}), "materials"]
        for expected, names in (({"mass": np.array([3., 2.])}, ["joint"]), (physics, ["wrong"])):
            try:
                checked_hard_reset(dummy, expected, names, ["body"], lambda raw: None)
            except RuntimeError:
                pass
            else:
                raise AssertionError("Changed hard-reset physics/order accepted")
    rng_before = rng_hashes()
    assert rng_before == rng_hashes()
    torch.rand(1)
    assert rng_before["torch_cpu"] != rng_hashes()["torch_cpu"]
    assert before == (model_digest(actor), model_digest(critic))
    print("Q_ACTION_RANKING_SELF_TEST PASS: asymmetric slicing, legal actions, common noise, seed pairing, hard-reset gate, n-step entropy grid, frozen models")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--config-name", default="simtoolreal_full_arm1_nstep3")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--prefix-steps", type=int, nargs="+", default=[0, 60, 120])
    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument("--tolerance", type=float, default=.029056)
    parser.add_argument("--epsilon", type=float, default=.2)
    parser.add_argument("--alignment-atol", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hard-reset", action="store_true",
                        help="Rebuild PhysX and verify unchanged physical parameters before every reference/branch reset")
    parser.add_argument("--branches", nargs="+", choices=BRANCHES, default=list(BRANCHES),
                        help="Must include mean and mean_repeat")
    parser.add_argument("--continuation-seeds", type=int, nargs="+", default=None,
                        help="Exact Gaussian-noise seeds paired across branches; omitted retains seed+30000+prefix")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.checkpoint is None or args.output_dir is None:
        parser.error("--checkpoint and --output-dir are required")
    if not (1 <= args.horizon <= 1200 and 1 <= args.num_envs <= 128 and 0 < args.epsilon <= 1
            and 0 < args.alignment_atol <= 1e-5 and all(0 <= n <= 1200 for n in args.prefix_steps)):
        parser.error("Invalid diagnostic bounds; state tolerance must not exceed 1e-5")
    if not {"mean", "mean_repeat"}.issubset(args.branches) or len(set(args.branches)) != len(args.branches):
        parser.error("Branches must be unique and include mean and mean_repeat")
    if len(set(args.prefix_steps)) != len(args.prefix_steps):
        parser.error("Prefix steps must be unique")
    if args.continuation_seeds is not None and (len(set(args.continuation_seeds)) != len(args.continuation_seeds)
            or any(not 0 <= seed < 2**63 for seed in args.continuation_seeds)):
        parser.error("Continuation seeds must be unique integers in [0,2**63)")
    args.checkpoint, args.output_dir = args.checkpoint.resolve(), args.output_dir.resolve()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Refusing to overwrite nonempty output")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    repo, started = Path(__file__).resolve().parents[1], time.monotonic()
    str_root = Path(os.environ.get("SIMTOOLREAL_ROOT", repo.parent / "simtoolreal")).resolve()
    sys.path.insert(0, str(str_root))
    os.chdir(str_root)
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    torch.set_num_threads(4)
    args.training_numerics = True
    args.override = [f"++env.task_cfg_overrides.termination.{k}={args.tolerance}" for k in ("success_tolerance", "target_success_tolerance")]
    args.override += ["++env.task_cfg_overrides.termination.eval_success_tolerance=null",
                     "++env.task_cfg_overrides.termination.tolerance_curriculum_interval=1152921504606846976"]
    save = lambda name, value: (args.output_dir / name).write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")
    source_files = [Path(__file__), repo / "scripts/check_str_hardreset_replay.py",
                    repo / "scripts/diagnose_str_rollouts.py", repo / "scripts/diagnose_str_critic.py",
                    repo / "scripts/diagnose_str_first_action.py"]
    provenance = {"schema_version": 2, "checkpoint": str(args.checkpoint),
        "source_sha256": {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files},
        "checkpoint_file_sha256": {name: hashlib.sha256((args.checkpoint / name).read_bytes()).hexdigest()
            for name in ("actor.pt", "critic.pt", "temperature.pt", "reward_normalizer.pt")}}
    save("provenance.json", provenance)
    save("status.json", {"status": "starting", "arguments": vars(args), "provenance": provenance})
    env, cfg, resolved = create_env(args, repo, str_root)
    try:
        raw = env.envs.unwrapped
        assert all(hasattr(raw, name) for name in STATE_BUFFERS), "Missing full-state comparison buffer"
        assert (raw.cfg.observation_space, raw.cfg.state_space, raw.cfg.action_space) == (140, 162, 29)
        assert raw.cfg.termination.max_consecutive_successes == 50
        assert not raw.cfg.reset.fixed_start_pose and not raw.cfg.reset.fixed_goal_pose
        dr = raw.cfg.domain_randomization
        assert not any(getattr(dr, k) for k in ("use_action_delay", "use_obs_delay", "use_object_state_delay_noise",
            "object_state_xyz_noise_std", "object_state_rotation_noise_degrees", "joint_velocity_obs_noise_std", "force_scale", "torque_scale"))
        actor, critic, target, alpha, normalizer = checkpoint_models(args.checkpoint, env.device)
        del target
        actor.eval().requires_grad_(False)
        critic.eval().requires_grad_(False)
        denominator, gamma = reward_denominator(normalizer, float(cfg.agent.normalized_G_max)), float(cfg.agent.gamma)
        n_step = int(cfg.agent.n_step)
        assert n_step >= 1
        before = {"actor": model_digest(actor), "critic": model_digest(critic)}
        tensor = lambda x: torch.as_tensor(x, device=env.device)
        assets = [{"env_id": i, "asset": Path(raw._object_urdf_paths[j]).name}
                  for i, j in enumerate(cpu(raw._object_asset_index_per_env).tolist())]
        save("actual_task_config.json", raw.cfg.to_dict())
        save("resolved_config.json", resolved)
        save("assets.json", assets)
        physical_checks = {}
        if args.hard_reset:
            from scripts.check_str_hardreset_replay import properties
            from isaacsimenvs.tasks.simtoolreal.utils.scene_utils import apply_physx_material_properties
            original_physics = properties(raw)
            joint_names, body_names = list(raw.robot.data.joint_names), list(raw.robot.data.body_names)
            np.savez_compressed(args.output_dir / "original_physical_parameters.npz", **original_physics)
        take_snapshot = lambda observation: snapshot(raw, observation, extended=args.hard_reset)
        metrics, control = {}, {}
        original_reward, original_pre = raw._get_rewards, raw._pre_physics_step

        def capture():
            reward = original_reward()
            metrics.update(height=cpu(raw.object.data.root_pos_w[:, 2] - raw.scene.env_origins[:, 2] - raw._object_init_z),
                           goals=cpu(raw._successes))
            return reward

        def pre_step(actions):
            previous = raw._prev_targets.clone() if control.get("capture") else None
            original_pre(actions)
            if previous is not None:
                control.update(previous_targets=cpu(previous), applied_targets=cpu(raw._cur_targets),
                               joint_pos_before_physics=cpu(raw.robot.data.joint_pos), capture=False)
        raw._get_rewards, raw._pre_physics_step = capture, pre_step

        def reset(label):
            if args.hard_reset:
                try:
                    actual, errors = checked_hard_reset(raw, original_physics, joint_names, body_names,
                                                       apply_physx_material_properties)
                except Exception as exc:
                    physical_checks[label] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                    save("physical_parameter_checks.json", physical_checks)
                    raise
                physical_checks[label] = {"status": "passed", "max_abs_errors": errors, "joint_body_order_equal": True}
                np.savez_compressed(args.output_dir / f"physical_parameters_{label}.npz", **actual)
                save("physical_parameter_checks.json", physical_checks)
            raw.seed(args.seed)
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            raw.reward_buf.zero_()
            raw._successes.zero_()
            return env.reset(random_start_init=False)[0]

        obs, tape, references, alive = reset("reference"), [], {}, np.ones(args.num_envs, bool)
        for t in range(max(args.prefix_steps) + 1):
            if t in args.prefix_steps:
                references[t] = (take_snapshot(obs), alive.copy(), rng_hashes())
            if t == max(args.prefix_steps):
                break
            with torch.no_grad():
                action = cpu(actor.get_mean_and_std(tensor(obs[:, :140]), training=False)[0].tanh())
            tape.append(action)
            obs, _, terminated, truncated, _ = step_env(env, action, "wrapper")
            alive &= ~(terminated | truncated)
        all_rows, all_alignment = [], {}
        for prefix in args.prefix_steps:
            reference, first_alive, reference_rng = references[prefix]
            np.savez_compressed(args.output_dir / f"reference_{prefix}.npz", **reference, first_episode_alive=first_alive)
            initial = tensor(reference["obs"])
            first, scores, gradient = candidates(actor, critic, initial[:, :140], initial[:, 140:], args.epsilon)
            for noise_seed, branch in product(continuation_seeds(args.seed, prefix, args.continuation_seeds), args.branches):
                key = comparison_key(prefix, noise_seed, branch)
                label = f"p{prefix}_seed{noise_seed}_{branch}"
                print(f"Q_PROBE_START prefix={prefix} seed={noise_seed} branch={branch}", flush=True)
                save("status.json", {"status": "running", "prefix_steps": prefix, "continuation_seed": noise_seed,
                                    "branch": branch, "completed_rows": len(all_rows)})
                obs = reset(label)
                for action in tape[:prefix]:
                    obs = step_env(env, action, "wrapper")[0]
                paired, errors = alignment(reference, take_snapshot(obs), args.alignment_atol)
                branch_rng = rng_hashes()
                rng_equal = reference_rng == branch_rng
                paired &= first_alive & rng_equal
                all_alignment[key] = {"paired_first_episode_count": int(paired.sum()),
                    "max_abs_by_field": errors, "rng_same": rng_equal,
                    "reference_rng_hashes": reference_rng, "branch_rng_hashes": branch_rng}
                active, sums, length = paired.copy(), np.zeros((4, args.num_envs)), np.zeros(args.num_envs, int)
                ended, timed_out = np.zeros(args.num_envs, bool), np.zeros(args.num_envs, bool)
                height_max = reference["object"][:, 2] - cpu(raw.scene.env_origins[:, 2]) - reference["_object_init_z"]
                goals, above_run, longest = reference["_successes"].copy(), np.zeros(args.num_envs, int), np.zeros(args.num_envs, int)
                generator = torch.Generator(device=env.device).manual_seed(noise_seed)
                control.clear()
                for t in range(args.horizon):
                    if not active.any():
                        break
                    if t == 0:
                        action, logp = cpu(first[branch]), np.zeros(args.num_envs)
                        control.update(capture=True, action=action.copy())
                    else:
                        action, logp = sampled_action(actor, tensor(obs[:, :140]), generator)
                    obs, reward, terminated, truncated, _ = step_env(env, action, "wrapper")
                    if t == 0:
                        control.update(joint_pos_after_first_step=cpu(raw.robot.data.joint_pos),
                                       first_step_done=terminated | truncated)
                    sums += np.stack(probe_return_increments(reward, logp, active, t, gamma, denominator, alpha, n_step))
                    length[active] += 1
                    height_max[active] = np.maximum(height_max[active], metrics["height"][active])
                    goals[active] = metrics["goals"][active]
                    above_run[active] = np.where(metrics["height"][active] > .10, above_run[active] + 1, 0)
                    longest[active] = np.maximum(longest[active], above_run[active])
                    ended |= active & terminated
                    timed_out |= active & truncated
                    active &= ~(terminated | truncated)
                np.savez_compressed(args.output_dir / f"control_{label}.npz", **control)
                for i in range(args.num_envs):
                    all_rows.append({"prefix_steps": prefix, "continuation_seed": noise_seed, "branch": branch, "env_id": i, "paired": bool(paired[i]),
                        "initial_q": float(scores[branch][i]), "q_action_gradient_norm": float(gradient[i]),
                        "discounted_raw_return": float(sums[0, i]), "discounted_normalized_return": float(sums[1, i]),
                        "finite_soft_return": float(sums[2, i]), "finite_nstep_soft_return": float(sums[3, i]), "n_step": n_step,
                        "length": int(length[i]), "terminated": bool(ended[i]),
                        "failure_terminated": bool(ended[i] and goals[i] < 50), "truncated": bool(timed_out[i]),
                        "horizon_censored": bool(active[i]), "goals_gained": int(goals[i] - reference["_successes"][i]),
                        "unknown_tail_discount": float(gamma ** length[i]) if paired[i] and not ended[i] else 0.,
                        "max_height_delta": float(height_max[i]), "above10cm_longest_seconds": float(longest[i] * raw.step_dt),
                        "asset": assets[i]["asset"]})
                print(f"Q_PROBE prefix={prefix} seed={noise_seed} branch={branch} paired={paired.sum()}/{args.num_envs}", flush=True)
                save("progress.json", {"rows": all_rows, "alignment": all_alignment, "current_key": key})
        assert before == {"actor": model_digest(actor), "critic": model_digest(critic)}
        save("summary.json", {"status": "complete", "arguments": vars(args), "alpha": alpha, "reward_denominator": denominator,
            "gamma": gamma, "policy_dt": float(raw.step_dt), "model_digests": before, "model_unchanged": True,
            "gamma_to_horizon": gamma ** args.horizon, "n_step": n_step, "provenance": provenance,
            "return_definitions": {
                "finite_soft_return": "Dense entropy diagnostic: reward/RMS every t, -alpha*logp every t>0; not the n-step training target.",
                "finite_nstep_soft_return": "Reward/RMS every t, -alpha*logp only t>0 and t%n_step==0; truncated unrolling with no unknown tail. Not exact historical replay Q.",
                "q_return_direction_agreement": "Uses dense finite_soft_return; NOT Q accuracy.",
                "q_nstep_return_direction_agreement": "Uses finite_nstep_soft_return; NOT Q accuracy."},
            "hard_reset": args.hard_reset, "physical_parameter_checks": physical_checks,
            "branches": args.branches,
            "continuation_seeds_by_prefix": {str(p): continuation_seeds(args.seed, p, args.continuation_seeds) for p in args.prefix_steps},
            "rows": all_rows, "alignment": all_alignment, "paired_comparisons": comparisons(all_rows),
            "wallclock_seconds": time.monotonic() - started, "limitations": [
                "Finite soft return is NOT exact Q calibration: unknown horizon/timeout tail omitted.",
                "Q/finite-return direction agreement is NOT Q accuracy: omitted tail can exceed the one-action Q difference.",
                "Independent Gaussian continuation is SAC's one-step actor distribution, NOT temporal-Zeta behavior exploration.",
                "No first-action entropy; both soft metrics use frozen checkpoint alpha and reward RMS.",
                "The dense soft metric differs from n-step training, whose intermediate rewards omit entropy; the n-step-grid metric records this distinction.",
                "Even n-step-grid returns under Gaussian continuation are not exact historical replay Q: behavior uses temporal-Zeta noise and projection/off-policy effects remain.",
                "Single-action intervention, not action holding; prefix deterministic policy, first episodes only.",
                "Seeded reset/action replay checks accessible state, NOT a hidden PhysX solver snapshot; mean_repeat checks residual noise.",
                "Only state-gated rows are interpretable. Same standardized policy draws do not guarantee equal future goal RNG after differing resets.",
                "Tolerance is frozen; lifetime curriculum counters are irrelevant and not restored. Post-step caches overwritten before use are omitted.",
                "Height is not a grasp/contact success test. Small environment sample does not cover every training asset.",
                "Multi-seed comparisons pair only matching prefix/continuation-seed/env; seeds are not independent initial states."]})
        save("status.json", {"status": "complete", "rows": len(all_rows), "wallclock_seconds": time.monotonic() - started})
    except Exception as exc:
        save("status.json", {"status": "error", "error": f"{type(exc).__name__}: {exc}",
                            "wallclock_seconds": time.monotonic() - started})
        raise
    finally:
        env.close()


if __name__ == "__main__":
    main()


