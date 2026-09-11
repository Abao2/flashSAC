#!/usr/bin/env python3
"""Frozen-checkpoint FlashSAC diagnostics; no Isaac, optimizer, or live model writes.

NPZ transitions must contain obs/next_obs [N,162], actions [N,29], raw rewards,
terminated, truncated, episode_id, step_in_episode, and optionally phase strings.
next_obs MUST be the pre-reset final observation for either kind of boundary.
These are rollout-data probes, not historical replay measurements or MC calibration.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flash_rl.agents.flashSAC.network import FlashSACActor, FlashSACDoubleCritic


def checkpoint_models(path: Path, device: str):
    def weights(name):
        payload = torch.load(path / f"{name}.pt", map_location="cpu", weights_only=True)
        return {k.removeprefix("_orig_mod."): v for k, v in payload["network_state_dict"].items()}

    def blocks(state):
        return len({k.split(".")[1] for k in state if k.startswith("encoder.")})

    actor_state = weights("actor")
    width, obs_dim = actor_state["embedder.w.w.weight"].shape
    action_dim = actor_state["predictor.mean_bias"].numel()
    actor = FlashSACActor(blocks(actor_state), obs_dim, width, action_dim).to(device)
    actor.load_state_dict(actor_state, strict=True)
    critics = []
    for name in ("critic", "target_critic"):
        state = weights(name)
        ensemble, width, input_dim = state["embedder.w.weight"].shape
        bins = state["predictor.bin_values"].flatten()
        model = FlashSACDoubleCritic(blocks(state), input_dim, width, bins.numel(),
                                    float(bins[0]), float(bins[-1]), ensemble).to(device)
        model.load_state_dict(state, strict=True)
        model.requires_grad_(False)
        critics.append(model)
    alpha = float(weights("temperature")["log_temp"].exp().item())
    normalizer = torch.load(path / "reward_normalizer.pt", map_location="cpu", weights_only=True)
    return actor, *critics, alpha, normalizer


def reward_denominator(normalizer: dict, normalized_g_max: float) -> float:
    return max(float((normalizer["G_rms_var"] + 1e-8).sqrt().item()),
               float(normalizer["G_r_max"].item()) / normalized_g_max)


def target_metrics(prob, bins, reward, terminated, alpha_log_prob, gamma):
    """Inspect BEFORE projection; truncated intentionally does not mask bootstrap."""
    unprojected = reward[:, None] + gamma * (1.0 - terminated.float()[:, None]) * (
        bins[None, :] - alpha_log_prob[:, None])
    below, above = unprojected < bins[0], unprojected > bins[-1]
    clipped = unprojected.clamp(float(bins[0]), float(bins[-1]))
    return {
        "target_unprojected_mean": (prob * unprojected).sum(-1),
        "target_clipped_mean": (prob * clipped).sum(-1),
        "target_mass_below_support": (prob * below).sum(-1),
        "target_mass_above_support": (prob * above).sum(-1),
        "target_unprojected_min": unprojected.min(-1).values,
        "target_unprojected_max": unprojected.max(-1).values,
        "target_expected_clipping_shift": (prob * (clipped - unprojected)).sum(-1),
    }


def summary(values):
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(array):
        return None
    if not np.isfinite(array).all():
        raise ValueError("Nonfinite diagnostic result")
    return dict(mean=float(array.mean()), std=float(array.std()), min=float(array.min()),
                p50=float(np.quantile(array, .5)), p90=float(np.quantile(array, .9)),
                max=float(array.max()))


def select_transition_ids(phases, rewards, rng, max_per_phase, reward_threshold=200.,
                          max_reward_events=4096):
    """Phase-balanced base sample plus ALL rare high-raw-reward transitions.

    Never silently subsample important reward events. A hard cap prevents an
    unexpectedly dense-event dataset from turning a diagnostic into a huge run.
    """
    phase_counts = {str(p): int((phases == p).sum()) for p in np.unique(phases)}
    balanced = np.concatenate([rng.choice(np.flatnonzero(phases == p),
                              min(count, max_per_phase), replace=False)
                              for p, count in phase_counts.items()])
    events = np.flatnonzero(rewards > reward_threshold)
    if len(events) > max_reward_events:
        raise ValueError(f"{len(events)} reward>{reward_threshold:g} events exceed safety cap "
                         f"{max_reward_events}; split dataset or explicitly raise --max-reward-events. "
                         "No reward events were silently discarded.")
    ids = np.union1d(balanced, events)
    rng.shuffle(ids)  # Avoid single-phase target cross-BN batches.
    selection = {
        "dataset_transitions": int(len(phases)), "selected_transitions": int(len(ids)),
        "phase_balanced_base_transitions": int(len(balanced)),
        "reward_event_predicate": f"raw reward > {reward_threshold:g}",
        "reward_event_threshold": float(reward_threshold),
        "reward_events_available": int(len(events)),
        "reward_events_selected": int(np.count_nonzero(rewards[ids] > reward_threshold)),
        "reward_events_added_beyond_phase_sample": int(len(ids) - len(balanced)),
        "max_reward_events_safety_cap": int(max_reward_events),
        "reward_event_note": "High raw reward is an event selector, not proof of stable grasp or task success.",
    }
    return ids, phase_counts, selection


def inference_metrics(actor, critic, obs, actions, random_actions):
    """Actual actor-Q path: both networks' BN running statistics, not batch stats."""
    result = {}
    with torch.no_grad():
        mean, std = actor.get_mean_and_std(obs, training=False)
        deterministic = mean.tanh()
        sampled, _ = actor(obs, training=False)
        for name, a in (("recorded", actions), ("zero", torch.zeros_like(actions)),
                        ("deterministic", deterministic), ("sampled", sampled)):
            q, info = critic(obs, a, training=False)
            result[f"q_{name}"] = q.min(0).values
            if name == "recorded":
                selected = q.argmin(0)
                prob = info["log_prob"].exp()[selected, torch.arange(len(obs), device=obs.device)]
                edge = min(5, prob.shape[-1] // 2)
                result["minq_top_atom_mass"] = prob[:, -1]
                result["minq_top5_atoms_mass"] = prob[:, -edge:].sum(-1)
                result["minq_bottom5_atoms_mass"] = prob[:, :edge].sum(-1)
                result["twin_q_gap"] = (q[0] - q[1]).abs()
        # Keep states separate: Q spread across actions, not across different states.
        repeated_obs = obs.repeat_interleave(random_actions, 0)
        uniform = torch.rand(len(repeated_obs), actions.shape[-1], device=obs.device) * 2 - 1
        uq, _ = critic(repeated_obs, uniform, training=False)
        uq = uq.min(0).values.reshape(len(obs), random_actions)
        result["uniform_action_q_range"] = uq.max(-1).values - uq.min(-1).values
        result["uniform_action_q_std"] = uq.std(-1, correction=0)
        result["actor_pretanh_std_mean"] = std.mean(-1)
        result["deterministic_action_mean_abs"] = deterministic.abs().mean(-1)
        result["sampled_action_mean_abs"] = sampled.abs().mean(-1)
    a = deterministic.detach().requires_grad_(True)
    q, _ = critic(obs, a, training=False)
    gradient = torch.autograd.grad(q.min(0).values.sum(), a)[0]
    result["deterministic_dq_da_l2"] = gradient.norm(dim=-1)
    result["deterministic_dq_da_arm_l2"] = gradient[:, :7].norm(dim=-1)
    result["deterministic_dq_da_hand_l2"] = gradient[:, 7:].norm(dim=-1)
    return {key: value.detach().cpu().numpy() for key, value in result.items()}


def actor_gradient_metrics(actor, critic, obs, next_obs, phases, alpha, repeats):
    """Match actor update BN modes in clones, with losses masked by phase.

    The cross-BN batch is common to all phases. These raw parameter gradients do
    NOT model Adam preconditioning, AMP scaling, or post-step unit normalization.
    """
    reports = {phase: [] for phase in np.unique(phases)}
    for _ in range(repeats):
        probe = copy.deepcopy(actor)
        a_all, info = probe(torch.cat([obs, next_obs]), training=True)
        actions, logp = a_all[:len(obs)], info["log_prob"][:len(obs)]
        q, _ = critic(obs, actions, training=False)
        q = q.min(0).values
        params = tuple(probe.parameters())
        for phase in reports:
            mask = torch.as_tensor(phases == phase, device=obs.device)
            q_loss = -q[mask].mean()
            entropy_loss = alpha * logp[mask].mean()
            q_parts = torch.autograd.grad(q_loss, params, retain_graph=True, allow_unused=True)
            h_parts = torch.autograd.grad(entropy_loss, params, retain_graph=True, allow_unused=True)
            qg = torch.cat([(torch.zeros_like(p) if g is None else g).flatten()
                            for p, g in zip(params, q_parts)])
            hg = torch.cat([(torch.zeros_like(p) if g is None else g).flatten()
                            for p, g in zip(params, h_parts)])
            qn, hn = qg.norm(), hg.norm()
            reports[phase].append({
                "q_loss_gradient_l2": float(qn), "entropy_loss_gradient_l2": float(hn),
                "total_loss_gradient_l2": float((qg + hg).norm()),
                "q_entropy_gradient_cosine": float((qg @ hg) / (qn * hn).clamp_min(1e-30)),
                "q_to_entropy_gradient_norm_ratio": float(qn / hn.clamp_min(1e-30)),
                "q_loss": float(q_loss.detach()), "entropy_loss": float(entropy_loss.detach()),
            })
        del probe
    return {phase: {key: summary([row[key] for row in rows]) for key in rows[0]}
            for phase, rows in reports.items()}


def diagnose(actor, critic, target, alpha, denominator, data, *, random_actions=16,
             batch_size=128, gradient_repeats=3, gamma=.99):
    device = next(actor.parameters()).device
    tensor = lambda name: torch.as_tensor(data[name], dtype=torch.float32, device=device)
    obs, next_obs, actions = tensor("obs"), tensor("next_obs"), tensor("actions")
    if obs.shape[-1] != 162 or actions.shape[-1] != 29:
        raise ValueError("This STR state-teacher probe requires 162 observations and 29 actions")
    values = {}
    for start in range(0, len(obs), batch_size):
        stop = min(start + batch_size, len(obs))
        o, no, a = obs[start:stop], next_obs[start:stop], actions[start:stop]
        chunk = inference_metrics(actor, critic, o, a, random_actions)
        with torch.no_grad():
            na, info = actor(no, training=False)
            # Match update_critic: training=True on concatenated current+next batch.
            # Work on a clone so BN probes never alter checkpoint inference state.
            target_probe = copy.deepcopy(target)
            tq, ti = target_probe(torch.cat([o, no]), torch.cat([a, na]), training=True)
            tq, lp = tq[:, len(o):], ti["log_prob"][:, len(o):]
            prob = lp.exp()[tq.argmin(0), torch.arange(len(o), device=device)]
            bins = target.predictor.bin_values.flatten()
            tm = target_metrics(prob, bins, tensor("rewards")[start:stop] / denominator,
                                tensor("terminated")[start:stop], alpha * info["log_prob"], gamma)
            tm["next_entropy_bonus"] = -alpha * info["log_prob"]
            chunk.update({k: v.cpu().numpy() for k, v in tm.items()})
            del target_probe
        for key, value in chunk.items():
            values.setdefault(key, []).append(value)
    values = {key: np.concatenate(chunks) for key, chunks in values.items()}
    phases = data["phase"]
    # Bound autograd memory while keeping each phase represented in a common batch.
    grad_ids = np.concatenate([np.flatnonzero(phases == p)[:64] for p in np.unique(phases)])
    gradients = actor_gradient_metrics(actor, critic, obs[grad_ids], next_obs[grad_ids],
                                       phases[grad_ids], alpha, gradient_repeats)
    report = {}
    for phase in np.unique(phases):
        mask = phases == phase
        report[str(phase)] = {
            "sampled_transitions": int(mask.sum()),
            "sampled_episodes": int(len(np.unique(data["episode_id"][mask]))),
            "terminated": int(data["terminated"][mask].sum()),
            "truncated": int(data["truncated"][mask].sum()),
            "raw_reward": summary(data["rewards"][mask]),
            "metrics": {k: summary(v[mask]) for k, v in values.items()},
            "actor_gradient": gradients[phase],
            "actor_gradient_phase_samples": int((phases[grad_ids] == phase).sum()),
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--next-obs-is-final", action="store_true",
                        help="Confirm next_obs at reset boundaries was replaced by final_obs")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-samples-per-phase", type=int, default=256)
    parser.add_argument("--reward-event-threshold", type=float, default=200.)
    parser.add_argument("--max-reward-events", type=int, default=4096,
                        help="Refuse oversized event sets rather than silently sampling them")
    parser.add_argument("--random-actions", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gradient-repeats", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=.99)
    parser.add_argument("--normalized-g-max", type=float, default=5.0)
    args = parser.parse_args()
    if not args.next_obs_is_final:
        parser.error("Confirm collector final_obs handling using --next-obs-is-final")
    if min(args.threads, args.max_samples_per_phase, args.max_reward_events, args.random_actions,
           args.batch_size, args.gradient_repeats) < 1 or args.normalized_g_max <= 0:
        parser.error("Counts and normalized-g-max must be positive")
    if not np.isfinite(args.reward_event_threshold):
        parser.error("reward-event-threshold must be finite")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    with np.load(args.data, allow_pickle=False) as saved:
        required = ("obs", "next_obs", "actions", "rewards", "terminated", "truncated",
                    "episode_id", "step_in_episode")
        data = {key: saved[key] for key in required}
        leading = data["obs"].shape[:-1]
        for key in required:
            trailing = (data[key].shape[-1],) if key in ("obs", "next_obs", "actions") else ()
            if data[key].shape != leading + trailing:
                raise ValueError(f"Shape mismatch for {key}: {data[key].shape}")
            data[key] = data[key].reshape((-1,) + trailing)
            if not np.isfinite(data[key]).all():
                raise ValueError(f"Nonfinite input: {key}")
        data["phase"] = (saved["phase"].reshape(-1).astype(str) if "phase" in saved else
                         np.where(data["step_in_episode"] == 0, "reset", "unlabelled"))
    if len(data["phase"]) != len(data["obs"]) or not len(data["obs"]):
        raise ValueError("Empty dataset or phase length mismatch")
    if data["next_obs"].shape != data["obs"].shape:
        raise ValueError("next_obs shape differs from obs")
    if (np.abs(data["actions"]) > 1.00001).any():
        raise ValueError("Actions must be normalized policy actions, not controller targets")
    for key in ("terminated", "truncated"):
        if not np.isin(data[key], [0, 1]).all():
            raise ValueError(f"{key} must be a boolean mask")
    ids, phase_counts, selection = select_transition_ids(
        data["phase"], data["rewards"], rng, args.max_samples_per_phase,
        args.reward_event_threshold, args.max_reward_events)
    data = {k: v[ids] for k, v in data.items()}
    actor, critic, target, alpha, normalizer = checkpoint_models(args.checkpoint, args.device)
    denominator = reward_denominator(normalizer, args.normalized_g_max)
    phases = diagnose(actor, critic, target, alpha, denominator, data,
                      random_actions=args.random_actions, batch_size=args.batch_size,
                      gradient_repeats=args.gradient_repeats, gamma=args.gamma)
    for phase, values in phases.items():
        values["reward_events_selected"] = int(np.count_nonzero(
            (data["phase"] == phase) & (data["rewards"] > args.reward_event_threshold)))
    report = {
        "checkpoint": str(args.checkpoint.resolve()), "dataset": str(args.data.resolve()),
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "dataset_phase_counts": phase_counts, "reward_denominator": denominator,
        "selection": selection,
        "alpha": alpha, "support": critic.predictor.bin_values.flatten().tolist(),
        "limitations": [
            "Frozen rollout dataset, NOT original training replay; checkpoint reward stats are frozen.",
            "No actual-return ranking or Monte Carlo calibration: Q includes entropy and reward scaling.",
            "Q/action probes use inference BN. TD targets use training target BN on mixed current+next rows.",
            "Actor gradients use training actor BN on common phase-balanced current+next rows; critic inference BN.",
            "FP32, uncompiled, no optimizer/AMP/weight-projection step; not an exact historical training update.",
            "Batch composition differs from training replay and can materially change cross-BN results.",
            "Phase-balanced sampling is augmented by all high-raw-reward events; combined sample frequencies are not replay frequencies.",
            "Target clipping mass is recomputed on these samples, not historical training clipping frequency.",
            "Phase labels are collector-provided; a lift threshold alone does not prove stable grasp.",
            "next_obs final/pre-reset semantics are asserted by the caller, not provable from this NPZ alone.",
        ],
        "phases": phases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "alpha": alpha, "reward_denominator": denominator,
                      "reward_events_selected": selection["reward_events_selected"],
                      "sampled_phases": {p: r["sampled_transitions"] for p, r in phases.items()}}, indent=2))


if __name__ == "__main__":
    main()
