#!/usr/bin/env python3
"""Paired fixed-reset first-action interventions, then frozen FlashSAC continuation.

Finite-horizon soft returns are NOT exact Q calibration: unknown timeout tails
are never bootstrapped. --self-test requires CPU only; normal execution uses Isaac.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.diagnose_str_critic import checkpoint_models, reward_denominator
from scripts.diagnose_str_rollouts import (
    create_env, field_slices, model_digest, step_env, terminal_next_observation,
)
from flash_rl.agents.utils.distribution import safe_tanh_log_det_jacobian

BRANCHES = ("mean", "zero", "arm_q_plus", "arm_q_minus", "hand_q_plus", "hand_q_minus", "uniform")


def first_actions(actor, critic, obs, replicates, epsilon, seed):
    with torch.no_grad():
        mean, _ = actor.get_mean_and_std(obs, training=False)
        baseline = mean.tanh()
    a = baseline.detach().requires_grad_(True)
    q, _ = critic(obs, a, training=False)
    gradient = torch.autograd.grad(q.min(0).values.sum(), a)[0].detach()
    directions = torch.zeros_like(gradient).reshape(replicates, len(BRANCHES), 29)
    grouped = gradient.reshape_as(directions)
    for index, sl, sign in ((2, slice(0, 7), 1), (3, slice(0, 7), -1),
                            (4, slice(7, 29), 1), (5, slice(7, 29), -1)):
        g = grouped[:, index, sl]
        directions[:, index, sl] = sign * g / g.norm(dim=-1, keepdim=True).clamp_min(1e-20)
    proposed = baseline.reshape_as(directions) + epsilon * directions
    proposed[:, 1] = 0
    generator = torch.Generator(device=obs.device).manual_seed(seed + 10001)
    proposed[:, 6] = torch.rand(29, device=obs.device, generator=generator) * 2 - 1
    actions = proposed.clamp(-1, 1)
    with torch.no_grad():
        q, _ = critic(obs, actions.reshape(-1, 29), training=False)
    return actions.reshape(-1, 29), {
        "q": q.min(0).values.detach().cpu().numpy().reshape(replicates, -1),
        "gradient_l2": gradient.norm(dim=-1).cpu().numpy().reshape(replicates, -1),
        "actual_perturbation_l2": (actions - baseline.reshape_as(actions)).norm(dim=-1).cpu().numpy(),
        "clipped_components": (actions != proposed).sum(dim=-1).cpu().numpy(),
    }


def continuation_actions(actor, obs, replicates, generator):
    """Independent across t/replicates; identical standardized noise within pairs."""
    with torch.no_grad():
        mean, std = actor.get_mean_and_std(obs, training=False)
        noise = torch.randn((replicates, 1, 29), device=obs.device, generator=generator)
        noise = noise.expand(-1, len(BRANCHES), -1).reshape(-1, 29)
        raw = mean + std * noise
        logp = (torch.distributions.Normal(mean, std).log_prob(raw)
                - safe_tanh_log_det_jacobian(raw)).sum(-1)
    return raw.tanh(), logp


def return_increments(reward, logp, active, step, gamma, denominator, alpha):
    raw = np.where(active, reward, 0.) * gamma ** step
    entropy = np.where(active, -alpha * logp, 0.) * gamma ** step if step else np.zeros_like(raw)
    return raw, raw / denominator, raw / denominator + entropy


def paired_stats(values):
    """Unadjusted normal-approximate paired CI, not an across-initial-state CI."""
    difference = values - values[:, :1]
    rows = {}
    for index, name in enumerate(BRANCHES):
        d = difference[:, index]
        sd = float(d.std(ddof=1)) if len(d) > 1 else 0.
        se, mean = sd / np.sqrt(len(d)), float(d.mean())
        rows[name] = {"mean": float(values[:, index].mean()), "paired_difference_mean": mean,
                      "paired_difference_std": sd, "paired_difference_se": float(se),
                      "paired_difference_normal95ci": [float(mean - 1.96 * se), float(mean + 1.96 * se)]}
    return rows


def initial_alignment(raw, obs, replicates, atol):
    origins = raw.scene.env_origins
    states = {"actor_observation": obs[:, :162],
              "object_local_root": torch.cat([raw.object.data.root_pos_w - origins,
                  raw.object.data.root_quat_w, raw.object.data.root_vel_w], -1).cpu().numpy(),
              "robot_local_root": torch.cat([raw.robot.data.root_pos_w - origins,
                  raw.robot.data.root_quat_w, raw.robot.data.root_vel_w], -1).cpu().numpy(),
              "joint_pos": raw.robot.data.joint_pos.cpu().numpy(),
              "joint_vel": raw.robot.data.joint_vel.cpu().numpy(),
              "previous_targets": raw._prev_targets.cpu().numpy()}
    report = {}
    for key, values in states.items():
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite initial state: {key}")
        grouped = values.reshape(replicates, len(BRANCHES), -1)
        delta = float(np.max(np.abs(grouped - grouped[:, :1])))
        report[key] = {"within_pair_max_abs": delta,
                       "across_all_envs_max_abs": float(np.max(np.abs(values - values[:1])))}
        if max(report[key].values()) > atol:
            raise ValueError(f"Fixed-reset alignment failed for {key}: {report[key]}, atol={atol}")
    return report, states


def self_test():
    torch.set_num_threads(2)
    from flash_rl.agents.flashSAC.network import FlashSACActor, FlashSACDoubleCritic
    torch.manual_seed(0)
    actor = FlashSACActor(1, 162, 8, 29)
    critic = FlashSACDoubleCritic(1, 191, 8, 11, -5., 5.).requires_grad_(False)
    obs = torch.randn(2, 1, 162).expand(-1, len(BRANCHES), -1).reshape(-1, 162)
    digest = model_digest(actor), model_digest(critic)
    a, report = first_actions(actor, critic, obs, 2, .2, 7)
    assert a.shape == (14, 29) and (a.abs() <= 1).all()
    assert (a.reshape(2, 7, 29)[:, 1] == 0).all()
    assert (report["actual_perturbation_l2"][:, 2:6] <= .200001).all()
    a1, lp1 = continuation_actions(actor, obs, 2, torch.Generator().manual_seed(7))
    a2, lp2 = continuation_actions(actor, obs, 2, torch.Generator().manual_seed(7))
    torch.testing.assert_close(a1, a2); torch.testing.assert_close(lp1, lp2)
    torch.testing.assert_close(a1.reshape(2, 7, 29), a1.reshape(2, 7, 29)[:, :1].expand(-1, 7, -1))
    raw, norm, soft = return_increments(np.array([2., 99.]), np.array([-3., -9.]),
                                        np.array([True, False]), 0, .5, 2., .1)
    np.testing.assert_allclose(soft, [1., 0.])  # No entropy for forced first action.
    raw, norm, soft = return_increments(np.array([4., 99.]), np.array([-2., -9.]),
                                        np.array([True, False]), 1, .5, 2., .1)
    np.testing.assert_allclose(raw, [2., 0.]); np.testing.assert_allclose(soft, [1.1, 0.])
    values = np.arange(7)[None, :] + np.arange(3)[:, None]
    assert paired_stats(values)["zero"]["paired_difference_normal95ci"] == [1., 1.]
    from types import SimpleNamespace as NS
    origins = torch.arange(14)[:, None].expand(-1, 3).float()
    body = NS(root_pos_w=origins.clone(), root_quat_w=torch.tensor([[1., 0., 0., 0.]]).expand(14, -1),
              root_vel_w=torch.zeros(14, 6), joint_pos=torch.zeros(14, 29), joint_vel=torch.zeros(14, 29))
    raw = NS(scene=NS(env_origins=origins), object=NS(data=body), robot=NS(data=body),
             _prev_targets=torch.zeros(14, 29))
    initial_alignment(raw, np.zeros((14, 324)), 2, 5e-5)
    body.joint_pos[1, 0] = .001
    try:
        initial_alignment(raw, np.zeros((14, 324)), 2, 5e-5)
    except ValueError:
        pass
    else:
        raise AssertionError("Mismatched physical initial state was accepted")
    assert digest == (model_digest(actor), model_digest(critic))
    print("FIRST_ACTION_SELF_TEST PASS (CPU only: intervention, paired noise, entropy timing, inactive mask, CI, frozen models)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--replicates", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=600)
    parser.add_argument("--epsilon", type=float, default=.2, help="L2 action perturbation per arm/hand group")
    parser.add_argument("--alignment-atol", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config-name", default="simtoolreal_state_teacher")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    if args.checkpoint is None or args.output_dir is None:
        parser.error("--checkpoint and --output-dir required")
    if not 2 <= args.replicates <= 36 or not 1 <= args.horizon <= 600:
        parser.error("Use 2..36 paired replicates (<=252 envs) and 1..600 steps")
    if not 0 < args.epsilon <= 1 or not 0 < args.alignment_atol <= .001:
        parser.error("epsilon must be (0,1], alignment-atol (0,.001]")
    args.checkpoint, args.output_dir = args.checkpoint.resolve(), args.output_dir.resolve()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory must be absent/empty; refusing overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    str_root = Path(os.environ.get("SIMTOOLREAL_ROOT", repo.parent / "simtoolreal")).resolve()
    sys.path[:0] = [str(repo), str(str_root)]
    os.chdir(str_root)
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    torch.set_num_threads(4)
    args.num_envs = args.replicates * len(BRANCHES)
    started = time.monotonic()
    env, cfg, resolved = create_env(args, repo, str_root)
    try:
        from isaacsimenvs.tasks.simtoolreal.utils.obs_utils import OBS_FIELD_SIZES
        obs, info = env.reset(random_start_init=False)
        raw = env.envs.unwrapped
        fields, size = field_slices(raw.cfg.obs.state_list, OBS_FIELD_SIZES)
        assert size == 162 and obs.shape == (args.num_envs, 324)
        assert tuple(raw.cfg.obs.obs_list) == tuple(raw.cfg.obs.state_list)
        assert cfg.env.bootstrap_timeouts and raw.cfg.termination.max_consecutive_successes == 1
        dr = raw.cfg.domain_randomization
        assert not any(getattr(dr, name) for name in ("use_obs_delay", "use_action_delay",
            "use_object_state_delay_noise", "object_state_xyz_noise_std", "object_state_rotation_noise_degrees",
            "joint_velocity_obs_noise_std", "force_scale", "torque_scale")), "Paired probe requires DR/noise off"
        assert all(tuple(getattr(dr, name)) == (1., 1.) for name in
                   ("object_scale_noise_multiplier_range", "object_friction_scale_range", "fingertip_friction_scale_range"))
        alignment, initial_physical = initial_alignment(raw, obs, args.replicates, args.alignment_atol)
        actor, critic, target, alpha, normalizer = checkpoint_models(args.checkpoint, env.device)
        del target
        denominator = reward_denominator(normalizer, float(cfg.agent.normalized_G_max))
        before = {"actor": model_digest(actor), "critic": model_digest(critic)}
        initial_obs = obs[:, :162].copy()
        first, first_report = first_actions(actor, critic, torch.as_tensor(initial_obs, device=env.device),
                                            args.replicates, args.epsilon, args.seed)
        generator = torch.Generator(device=env.device).manual_seed(args.seed + 20001)
        n, shape = args.num_envs, (args.horizon, args.num_envs)
        rewards, logps, valid = np.zeros(shape), np.zeros(shape), np.zeros(shape, dtype=bool)
        active = np.ones(n, dtype=bool)
        lengths, success, terminal, timeout = np.zeros(n, int), np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool)
        sums = np.zeros((3, n))
        final_obs, first_targets = initial_obs.copy(), np.zeros((n, 29))
        action_edge_count = np.zeros(n)
        for step in range(args.horizon):
            if not active.any():
                break
            if step == 0:
                actions, lp = first, torch.zeros(n, device=env.device)
            else:
                actions, lp = continuation_actions(actor, torch.as_tensor(obs[:, :162], device=env.device),
                                                   args.replicates, generator)
            actions_np, logp = actions.detach().cpu().numpy().copy(), lp.cpu().numpy()
            actions_np[~active] = 0
            result, reward, terminated, truncated, step_info = step_env(env, actions_np, "wrapper")
            done = terminated | truncated
            nxt = terminal_next_observation(result, done, step_info)[:, :162]
            if not np.isfinite(reward).all() or not np.isfinite(logp).all():
                raise RuntimeError("Nonfinite reward/log probability")
            valid[step], rewards[step], logps[step] = active, np.where(active, reward, 0), np.where(active, logp, 0)
            sums += np.stack(return_increments(reward, logp, active, step, float(cfg.agent.gamma), denominator, alpha))
            lengths[active] += 1
            action_edge_count[active] += (np.abs(actions_np[active]) >= .999).mean(-1)
            final_obs[active] = nxt[active]
            if step == 0:
                first_targets[:] = nxt[:, fields["prev_action_targets"]]
            ending = done & active
            if ending.any():
                hit = step_info.get("episode_final", {}).get("all_goals_hit")
                if hit is None:
                    raise RuntimeError("Missing pre-reset success metric")
                success[ending], terminal[ending], timeout[ending] = np.asarray(hit)[ending], terminated[ending], truncated[ending]
            active[ending] = False
            obs = result
        assert before == {"actor": model_digest(actor), "critic": model_digest(critic)}
        matrix = lambda a: a.reshape(args.replicates, len(BRANCHES))
        rows = []
        for i in range(n):
            rows.append(dict(replicate=i // 7, branch=BRANCHES[i % 7], length=int(lengths[i]),
                             success=bool(success[i]), terminated=bool(terminal[i]), truncated=bool(timeout[i]),
                             diagnostic_horizon_stop=bool(active[i]), gamma_to_length=float(cfg.agent.gamma ** lengths[i]),
                             discounted_raw_return=float(sums[0, i]), discounted_normalized_return=float(sums[1, i]),
                             finite_soft_return=float(sums[2, i]), initial_q=float(first_report["q"].reshape(-1)[i]),
                             action_fraction_abs_ge_0999=float(action_edge_count[i] / lengths[i])))
        limitations = ["Fixed reset with paired stochastic continuation draws, NOT generalization across initial states.",
            "Finite soft returns have NO unknown tail bootstrap; timeouts/horizon survivors do NOT constitute exact Q calibration.",
            "Continuation is current actor with independent Gaussian draws each step, not cached behavior exploration noise.",
            "Forced first action has NO entropy term; entropy starts at t=1, matching soft Q(s0,a0).",
            "Normal-approximate paired 95% CIs are unadjusted for multiple branch comparisons.",
            "One action lasts one control interval; arm is incremental, hand absolute. Multi-step interventions would test a different quantity.",
            "Initial alignment covers accessible physical/controller state, not hidden PhysX solver/contact caches; floating-point simulator differences may remain."]
        report = dict(status="complete", arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                      branches=BRANCHES, resolved_config=resolved, alignment=alignment, alpha=alpha,
                      reward_denominator=denominator, gamma=float(cfg.agent.gamma), dt_seconds=float(raw.step_dt),
                      first_action={k: v.tolist() for k, v in first_report.items()},
                      paired={name: paired_stats(matrix(value)) for name, value in
                              (("initial_q", first_report["q"]), ("discounted_raw", sums[0]),
                               ("discounted_normalized", sums[1]), ("finite_soft", sums[2]))},
                      outcome_by_branch={name: {"success_fraction": float(matrix(success)[:, j].mean()),
                          "terminated_fraction": float(matrix(terminal)[:, j].mean()),
                          "truncated_fraction": float(matrix(timeout)[:, j].mean()),
                          "horizon_survivor_fraction": float(matrix(active)[:, j].mean()),
                          "unknown_tail_fraction": float(matrix(~terminal)[:, j].mean())}
                          for j, name in enumerate(BRANCHES)},
                      model_digests=before, model_unchanged=True, limitations=limitations, episodes=rows,
                      wallclock_seconds=time.monotonic() - started)
        (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        np.savez_compressed(args.output_dir / "paired_returns.npz", rewards=rewards, log_probs=logps, valid=valid,
                            initial_obs=initial_obs, initial_actions=first.cpu().numpy(), final_obs=final_obs,
                            first_controller_targets=first_targets, **initial_physical)
        print("FIRST_ACTION_RESULT " + json.dumps({"output": str(args.output_dir), "episodes": n,
              "successes": int(success.sum()), "unknown_tails": int((~terminal).sum()),
              "wallclock_seconds": report["wallclock_seconds"]}), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
