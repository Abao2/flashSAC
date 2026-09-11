#!/usr/bin/env python3
"""One bounded hard-reset baseline/replay check; no Q claims or training."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.check_str_q_action_ranking import alignment, cpu, rng_hashes, snapshot
from scripts.diagnose_str_critic import checkpoint_models
from scripts.diagnose_str_rollouts import create_env, model_digest, step_env


def properties(raw):
    result = {}
    for name in ("robot", "object", "table", "goal_viz"):
        view = getattr(raw, name).root_physx_view
        for field in ("material_properties", "masses", "inertias"):
            result[f"{name}/{field}"] = cpu(getattr(view, f"get_{field}")())
    for field in ("dof_limits", "dof_stiffnesses", "dof_dampings", "dof_max_forces", "dof_max_velocities"):
        result[f"robot/{field}"] = cpu(getattr(raw.robot.root_physx_view, f"get_{field}")())
    return result


def property_errors(reference, current):
    errors = {}
    for key, expected in reference.items():
        actual = current[key]
        if actual.shape != expected.shape:
            raise RuntimeError(f"Property shape changed: {key}: {expected.shape} -> {actual.shape}")
        if not np.isfinite(actual).all():
            raise RuntimeError(f"Nonfinite physical property: {key}")
        errors[key] = float(np.max(np.abs(actual.astype(float) - expected.astype(float))))
    return errors


def state(raw, obs):
    result = snapshot(raw, obs)
    result["body_state_w"] = cpu(raw.robot.data.body_state_w)
    for name in ("joint_pos_target", "joint_vel_target", "joint_effort_target", "computed_torque", "applied_torque"):
        result[f"robot/{name}"] = cpu(getattr(raw.robot.data, name))
    return result


def self_test():
    initial = {"mass": np.array([1., 2.]), "limits": np.array([[0., 1.]])}
    current = {k: v.copy() for k, v in initial.items()}
    assert max(property_errors(initial, current).values()) == 0.
    current["mass"][0] += 1.
    assert property_errors(initial, current)["mass"] == 1.
    current["mass"][0] = np.nan
    try:
        property_errors(initial, current)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Nonfinite physical parameter accepted")
    print("HARDRESET_REPLAY_SELF_TEST PASS: unchanged/changed/nonfinite property checks; no simulator")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-summary", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.source_summary is None or args.output_dir is None or not 1 <= args.steps <= 120:
        parser.error("source-summary/output-dir required; at most 120 steps")
    source_path, output = args.source_summary.resolve(), args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Refusing to overwrite existing output")
    source = json.loads(source_path.read_text())
    protocol = source["arguments"]
    setup = argparse.Namespace(config_name=protocol["config_name"], num_envs=protocol["num_envs"],
        seed=protocol["seed"], training_numerics=True, override=protocol["override"])
    output.mkdir(parents=True, exist_ok=True)
    save = lambda name, data: (output / name).write_text(json.dumps(data, indent=2, allow_nan=False, default=str) + "\n")
    started = time.monotonic()
    report = {"status": "starting", "source_summary": str(source_path), "steps": args.steps,
              "num_envs": setup.num_envs, "state_atol": 1e-5, "learning": False,
              "protocol": "Two hard resets; after each, reapply original STR materials, verify physical parameters, seed and task-reset. Reference deterministic actions then exact tape replay."}
    save("summary.json", report)
    repo = Path(__file__).resolve().parents[1]
    str_root = Path(os.environ.get("SIMTOOLREAL_ROOT", repo.parent / "simtoolreal")).resolve()
    sys.path.insert(0, str(str_root))
    os.chdir(str_root)
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    torch.set_num_threads(4)
    env = None
    try:
        env, cfg, resolved = create_env(setup, repo, str_root)
        raw = env.envs.unwrapped
        assert raw.cfg.observation_space == 140 and raw.cfg.state_space == 162
        actor, critic, target, _, _ = checkpoint_models(Path(protocol["checkpoint"]), env.device)
        del critic, target
        actor.eval().requires_grad_(False)
        before = model_digest(actor)
        expected = properties(raw)
        joint_names = list(raw.robot.data.joint_names)
        body_names = list(raw.robot.data.body_names)
        np.savez_compressed(output / "original_physical_parameters.npz", **expected)
        save("actual_task_config.json", raw.cfg.to_dict())
        from isaacsimenvs.tasks.simtoolreal.utils.scene_utils import apply_physx_material_properties

        def reset(label):
            raw.sim.reset(soft=False)
            apply_physx_material_properties(raw)
            actual = properties(raw)
            errors = property_errors(expected, actual)
            report.setdefault("physical_parameter_max_abs_errors", {})[label] = errors
            names_equal = joint_names == list(raw.robot.data.joint_names) and body_names == list(raw.robot.data.body_names)
            report.setdefault("joint_body_order_equal", {})[label] = names_equal
            np.savez_compressed(output / f"{label}_physical_parameters.npz", **actual)
            save("summary.json", report)
            if not names_equal or max(errors.values()) > 1e-6:
                raise RuntimeError(f"Hard reset changed physics/order beyond tolerance: {errors}")
            raw.seed(setup.seed)
            random.seed(setup.seed)
            np.random.seed(setup.seed)
            torch.manual_seed(setup.seed)
            torch.cuda.manual_seed_all(setup.seed)
            raw.reward_buf.zero_()
            raw._successes.zero_()
            return env.reset(random_start_init=False)[0]

        obs = reset("reference")
        tape, references, hashes, rewards, dones = [], [], [], [], []
        for t in range(args.steps + 1):
            references.append(state(raw, obs))
            hashes.append(rng_hashes())
            if t == args.steps:
                break
            with torch.no_grad():
                action = cpu(actor.get_mean_and_std(torch.as_tensor(obs[:, :140], device=env.device), training=False)[0].tanh())
            tape.append(action)
            obs, reward, term, trunc, _ = step_env(env, action, "wrapper")
            rewards.append(reward.copy())
            dones.append(np.stack([term, trunc]))
        np.savez_compressed(output / "reference_trace.npz", **{key: np.stack([s[key] for s in references]) for key in references[0]},
                            actions=np.stack(tape), rewards=np.stack(rewards), dones=np.stack(dones))
        obs = reset("repeat")
        rows, repeat_states, repeat_rewards, repeat_dones = [], [], [], []
        for t in range(args.steps + 1):
            current = state(raw, obs)
            repeat_states.append(current)
            valid, errors = alignment(references[t], current, 1e-5)
            current_rng = rng_hashes()
            row = {"step": t, "state_paired_count": int(valid.sum()), "rng_same": hashes[t] == current_rng,
                   "state_max_abs_by_field": errors, "reference_rng_hashes": hashes[t], "repeat_rng_hashes": current_rng}
            rows.append(row)
            if t in (0, 1, 30, 60, args.steps):
                print(f"HARDRESET_REPLAY step={t} state_pairs={valid.sum()}/{setup.num_envs} rng={row['rng_same']}", flush=True)
            if t == args.steps:
                break
            obs, reward, term, trunc, _ = step_env(env, tape[t], "wrapper")
            repeat_rewards.append(reward.copy())
            repeat_dones.append(np.stack([term, trunc]))
        np.savez_compressed(output / "repeat_trace.npz", **{key: np.stack([s[key] for s in repeat_states]) for key in repeat_states[0]},
                            rewards=np.stack(repeat_rewards), dones=np.stack(repeat_dones))
        bad = [r for r in rows if r["state_paired_count"] != setup.num_envs or not r["rng_same"]]
        identical_events = np.array_equal(np.stack(dones), np.stack(repeat_dones))
        report.update(status="complete", all_steps_full_state_and_rng_paired=not bad,
            first_failed_gate_step=bad[0]["step"] if bad else None, per_step=rows,
            reward_max_abs_error=float(np.max(np.abs(np.stack(rewards) - np.stack(repeat_rewards)))),
            identical_terminal_and_timeout_events=identical_events, model_unchanged=model_digest(actor) == before,
            actor_digest=before, interpretation="Bounded replay reproducibility only; not Q accuracy or training success. Hard reset uses the standard simulator API and unchanged checked physical parameters.",
            limitations=["No hidden PhysX state snapshot or guarantee of GPU determinism.",
                "Passing this deterministic prefix would not guarantee long stochastic rollout repeatability.",
                "A failure here is a diagnostic pairing limit, not proof of a training environment bug."])
        assert report["model_unchanged"]
    except Exception as exc:
        report.update(status="error", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["wallclock_seconds"] = time.monotonic() - started
        save("summary.json", report)
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
