"""Initial previous-reward counterfactual; no training or environment-core edits.

Delegate all rollout arguments to diagnose_str_rollouts. Inject only before its
first explicit reset; simulator auto-resets and subsequent rewards stay native.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import functools
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import diagnose_str_rollouts as rollout


def write_audit(output_dir, audit):
    # main() has already passed the harness's empty-output-directory guard.
    path = Path(output_dir) / "reset_reward_intervention.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def install_initial_reward_reset(env, field_sizes, previous_reward, audit, persist):
    import numpy as np
    import torch

    raw = env.envs.unwrapped
    policy, policy_dim = rollout.field_slices(raw.cfg.obs.obs_list, field_sizes)
    critic, critic_dim = rollout.field_slices(raw.cfg.obs.state_list, field_sizes)
    if policy["reward"].stop - policy["reward"].start != 1 or critic["reward"].stop - critic["reward"].start != 1:
        raise ValueError("Expected scalar reward field in both observations")
    audit.update(policy_reward_column=policy["reward"].start,
                 critic_reward_column=policy_dim + critic["reward"].start,
                 expected_encoded_reward=previous_reward * .01,
                 initial_cohort_episode_ids=list(range(raw.num_envs)),
                 explicit_reset_calls=0)
    original_reset = env.reset

    @functools.wraps(original_reset)
    def reset(*args, **kwargs):
        audit["explicit_reset_calls"] += 1
        if audit["explicit_reset_calls"] != 1:
            # Do not force another episode's reward, including later explicit resets.
            return original_reset(*args, **kwargs)
        with torch.no_grad():
            audit["raw_reward_before_injection"] = raw.reward_buf.detach().cpu().tolist()
            raw.reward_buf.fill_(previous_reward)
        audit.update(status="injected_before_first_explicit_reset",
                     first_reset_kwargs=repr(kwargs),
                     injected_utc=datetime.now(timezone.utc).isoformat())
        persist()
        try:
            observations, info = original_reset(*args, **kwargs)
            observed = np.asarray(observations)
            if observed.shape != (raw.num_envs, policy_dim + critic_dim):
                raise ValueError(f"Unexpected combined observation shape: {observed.shape}")
            for label in ("policy", "critic"):
                values = observed[:, audit[f"{label}_reward_column"]]
                audit[f"returned_{label}_reward"] = values.tolist()
                if not np.allclose(values, previous_reward * .01, rtol=0, atol=1e-6):
                    raise AssertionError(f"{label} reward observation did not preserve injected previous reward")
            audit["status"] = "initial_reset_verified"
            persist()  # Must precede Isaac's eventual process-exiting close().
            return observations, info
        except BaseException as error:
            audit.update(status="failed", error=f"{type(error).__name__}: {error}")
            persist()
            raise

    env.reset = reset
    return env


def obs_field_sizes():
    from isaacsimenvs.tasks.simtoolreal.utils.obs_utils import OBS_FIELD_SIZES
    return OBS_FIELD_SIZES


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--initial-previous-reward", type=float, required=True)
    own, forwarded = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    if not math.isfinite(own.initial_previous_reward):
        parser.error("Initial previous reward must be finite")
    original_factory, original_argv = rollout.create_env, sys.argv

    def create_env(args, repo, str_root):
        if args.policy != "flash" or args.entry != "wrapper" or args.training_startup:
            raise ValueError("Reward probe requires frozen Flash wrapper without training-startup")
        audit = {
            "status": "creating_environment", "raw_previous_reward": own.initial_previous_reward,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "intervention": "Set raw.reward_buf before first explicit env.reset only; no extra reset/physics/action calls.",
            "limits": [
                "Counterfactual initial policy state, not a reward/reset implementation fix or deployment result.",
                "Only the first vector batch is directly seeded; later auto-resets preserve native reward history.",
                "No changes to physical state, actions, controller, computed rewards, policy or optimizer.",
                "initial_reset_verified confirms this intervention only; rollout completion is in summary.json.",
            ],
        }
        persist = lambda: write_audit(args.output_dir, audit)
        persist()
        try:
            env, cfg, resolved = original_factory(args, repo, str_root)
            return install_initial_reward_reset(env, obs_field_sizes(), own.initial_previous_reward,
                                                audit, persist), cfg, resolved
        except BaseException as error:
            audit.update(status="failed", error=f"{type(error).__name__}: {error}")
            persist()
            raise

    try:
        rollout.create_env = create_env
        sys.argv = [original_argv[0], *forwarded]
        rollout.main()
    finally:
        rollout.create_env, sys.argv = original_factory, original_argv


if __name__ == "__main__":
    main()
