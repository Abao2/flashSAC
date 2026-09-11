#!/usr/bin/env python3
"""Read-only CPU audit of full-STR n-step replay; does not reconstruct episodes.

Counts are replay ROW occupancy, not independent episodes or success rates.
Frozen-Q action spread is a diagnostic, never a test of Q ranking correctness.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.check_str_full_nodr import load_config, observation_sizes, TASK_ROOT
from scripts.diagnose_str_critic import checkpoint_models, reward_denominator

CHECKPOINTS = {
    "100M": ROOT / "diagnostics/credit_nstep3_20260910/candidate/models/seed0/step48830",
    "500M": ROOT / "diagnostics/credit_nstep3_continue_20260910/candidate/models/seed0/step195320",
}


def slices(fields, sizes, start=0):
    result = {}
    for name in fields:
        result[name] = slice(start, start + sizes[name])
        start += sizes[name]
    return result


def stats(values):
    x = np.asarray(values, dtype=float).reshape(-1)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite diagnostic values")
    return dict(mean=float(x.mean()), p10=float(np.quantile(x, .1)),
                median=float(np.median(x)), p90=float(np.quantile(x, .9)),
                min=float(x.min()), max=float(x.max()))


def task_differences(a, b, prefix=""):
    out = {}
    for key in sorted(set(a) | set(b)):
        name = prefix + key
        av, bv = a.get(key), b.get(key)
        if isinstance(av, dict) and isinstance(bv, dict):
            out.update(task_differences(av, bv, name + "."))
        elif av != bv:
            out[name] = {"original": av, "experiment": bv}
    return out


def scan(label, checkpoint, layout, actor_layout, chunk_size):
    replay = torch.load(checkpoint / "replay_buffer.pt", weights_only=True,
                        mmap=True, map_location="cpu")
    n = int(replay["num_in_buffer"])
    assert replay["observation"].shape[1] == 302
    assert replay["action"].shape[1] == 29
    count, goals_hist = Counter(), Counter()
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        obs = replay["observation"][start:stop]
        nxt = replay["next_observation"][start:stop]
        goals = torch.expm1(obs[:, layout["successes"]]).flatten().round().long()
        next_goals = torch.expm1(nxt[:, layout["successes"]]).flatten().round().long()
        lifted = obs[:, layout["lifted_object"]].flatten() > .5
        term = replay["terminated"][start:stop].bool()
        trunc = replay["truncated"][start:stop].bool()
        for threshold in (1, 2, 5, 10, 20):
            count[f"goals_so_far_ge_{threshold}"] += int((goals >= threshold).sum())
        count["ever_lifted_flag"] += int(lifted.sum())
        count["next_goal_count_increased_nonterminal"] += int(((next_goals > goals) & ~term & ~trunc).sum())
        count["terminated_nstep_rows"] += int(term.sum())
        count["truncated_nstep_rows"] += int(trunc.sum())
        count["overlapping_term_trunc"] += int((term & trunc).sum())
        count["action_components_outside_unit_interval"] += int((replay["action"][start:stop].abs() > 1.000001).sum())
        values, amounts = torch.unique(goals, return_counts=True)
        goals_hist.update({str(int(v)): int(c) for v, c in zip(values, amounts)})
        if start == 0 or stop == n or stop % (chunk_size * 40) == 0:
            print(f"{label}: scanned {stop:,}/{n:,} replay rows", flush=True)
    rng = torch.Generator().manual_seed(11092026)
    ids = torch.randint(n, (2048,), generator=rng)
    sample = replay["observation"][ids].clone()
    action_sample = replay["action"][ids].clone()
    # Four symmetric keypoint offsets sum to zero: their mean reconstructs
    # object center relative to palm. This is CURRENT z, not lift-from-reset.
    center = sample[:, layout["palm_pos"]] + sample[:, layout["keypoints_rel_palm"]].reshape(-1, 4, 3).mean(1)
    scale = sample[:, layout["object_scales"]]
    kp_error = sample[:, layout["keypoints_rel_goal"]].reshape(-1, 4, 3).norm(dim=-1).max(-1).values
    obs_errors = {name: float((sample[:, sl] - sample[:, layout[name]]).abs().max())
                  for name, sl in actor_layout.items()}
    report = dict(checkpoint=str(checkpoint), replay_rows=n, ring_current_idx=int(replay["current_idx"]),
                  counts=dict(count), row_fractions={k: v / n for k, v in count.items()
                     if k != "action_components_outside_unit_interval"},
                  goals_so_far_histogram=dict(sorted(goals_hist.items(), key=lambda kv: int(kv[0]))),
                  sample_rows=len(ids), sample_index_sha256=hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
                  sampled_current_object_z_m=stats(center[:, 2]),
                  sampled_max_keypoint_error_m=stats(kp_error),
                  sampled_object_scale_xyz={axis: stats(scale[:, i]) for i, axis in enumerate("xyz")},
                  sampled_unique_scale_triplets=int(torch.unique(scale, dim=0).shape[0]),
                  sampled_actor_vs_clean_corresponding_fields_max_abs=obs_errors,
                  sampled_action_abs=stats(action_sample.abs()),
                  sampled_near_saturated_action_fraction=float((action_sample.abs() > .99).float().mean()))
    assert not count["overlapping_term_trunc"]
    assert not count["action_components_outside_unit_interval"]
    del replay
    return report, sample[:256]


def q_sensitivity(checkpoint, common_obs):
    actor, critic, target, alpha, normalizer = checkpoint_models(checkpoint, "cpu")
    actor.requires_grad_(False)
    results = {"alpha": alpha, "reward_denominator": reward_denominator(normalizer, 5.),
               "scope": "Same common replay states for both checkpoints; no physical action rollout; not Q accuracy."}
    all_values = {}
    generator = torch.Generator().manual_seed(11092026)
    with torch.compiler.set_stance("force_eager"):
        for chunk in common_obs.split(64):
            with torch.no_grad():
                mean, std = actor.get_mean_and_std(chunk[:, :140], training=False)
                action = mean.tanh()
            probe = action.detach().requires_grad_(True)
            q, info = critic(chunk[:, 140:], probe, training=False)
            q_min = q.min(0).values
            grad = torch.autograd.grad(q_min.sum(), probe)[0]
            with torch.no_grad():
                uniform = torch.rand((len(chunk), 8, 29), generator=generator) * 2 - 1
                state = chunk[:, 140:].repeat_interleave(8, 0)
                uq, _ = critic(state, uniform.reshape(-1, 29), training=False)
                uq = uq.min(0).values.reshape(len(chunk), 8)
                chosen_prob = info["log_prob"].exp()[q.argmin(0), torch.arange(len(chunk))]
                values = dict(mean_action_q=q_min, twin_q_gap=(q[0] - q[1]).abs(),
                              uniform_action_q_range=uq.max(-1).values - uq.min(-1).values,
                              mean_action_dq_da_arm_l2=grad[:, :7].norm(dim=-1),
                              mean_action_dq_da_hand_l2=grad[:, 7:].norm(dim=-1),
                              actor_pretanh_std=std.mean(-1),
                              actor_mean_action_abs=action.abs().mean(-1),
                              minq_top5_atoms_mass=chosen_prob[:, -5:].sum(-1))
            for key, value in values.items():
                all_values.setdefault(key, []).append(value.detach().numpy())
    results.update({k: stats(np.concatenate(v)) for k, v in all_values.items()})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--chunk-size", type=int, default=65536)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "result.json"
    if output.exists():
        parser.error("result.json already exists; refusing overwrite")
    torch.set_num_threads(4)
    cfg, original, task = load_config("simtoolreal_full_arm1_nstep3")
    sizes = observation_sizes()
    layout = slices(task["obs"]["state_list"], sizes, 140)
    actor_layout = slices(task["obs"]["obs_list"], sizes)
    report = dict(status="running", scope="full STR n3 100M and 500M read-only CPU audit",
                  limitations=["Replay retains only the latest 10M rows, not the entire training history.",
                    "n-step rows overlap; counts are neither episode counts nor success rates.",
                    "No episode IDs, per-frame object initial z, contact forces, or asset identity in replay.",
                    "The lifted flag is latched. It does not prove current/stable holding.",
                    "Object size is measurable; size cannot uniquely identify tool family.",
                    "A Q-action spread is not evidence that its ranking matches real future returns."],
                  task_differences_from_original=task_differences(original, task),
                  critic_field_slices={k: [s.start, s.stop] for k, s in layout.items()},
                  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
                     TASK_ROOT / "utils/obs_utils.py", TASK_ROOT / "utils/action_utils.py")},
                  replays={}, frozen_q_common_state_sensitivity={})
    samples = []
    for label, path in CHECKPOINTS.items():
        row, sample = scan(label, path, layout, actor_layout, args.chunk_size)
        report["replays"][label] = row
        samples.append(sample)
        (args.output_dir / (label + "_coverage.json")).write_text(json.dumps(row, indent=2) + "\n")
    common = torch.cat(samples)
    for label, path in CHECKPOINTS.items():
        print(f"{label}: frozen CPU Q sensitivity on {len(common)} common replay states", flush=True)
        report["frozen_q_common_state_sensitivity"][label] = q_sensitivity(path, common)
    report["status"] = "CPU_AUDIT_COMPLETE_GPU_ROLLOUT_NOT_RUN"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(output, flush=True)


if __name__ == "__main__":
    main()
