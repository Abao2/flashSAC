#!/usr/bin/env python3
"""Native STR training plus passive height proxies; these do NOT prove grasp/contact."""
import argparse
import functools
import json
import math
import os
from pathlib import Path
import sys

CHECKPOINT_NAMES = ("actor", "critic", "target_critic", "temperature", "reward_normalizer", "agent_state", "replay_buffer")


def resume_overrides(args):
    if args.resume_checkpoint is None:
        if args.resume_tolerance is not None or args.env_step_offset != 0:
            raise ValueError("Resume tolerance/offset require --resume-checkpoint")
        return []
    checkpoint = args.resume_checkpoint
    if not checkpoint.is_absolute() or args.env_step_offset <= 0:
        raise ValueError("Resume requires an absolute checkpoint and positive env-step-offset")
    if args.resume_tolerance is None or not math.isfinite(args.resume_tolerance) or args.resume_tolerance <= 0:
        raise ValueError("Resume requires a finite positive resume-tolerance")
    missing = [name for name in CHECKPOINT_NAMES if not (checkpoint / f"{name}.pt").is_file()
               or (checkpoint / f"{name}.pt").stat().st_size == 0]
    if missing:
        raise ValueError(f"Incomplete resume checkpoint: {missing}")
    path = json.dumps(str(checkpoint.resolve()))
    return [f"agent_load_path={path}", f"buffer_load_path={path}", "require_agent_load=true",
            "agent.load_optimizer=true", "agent.load_reward_normalizer=true"]


def install_resume_hooks(train, args):
    if args.resume_checkpoint is None:
        return lambda: None
    original_envs, original_agent, original_logger = train.create_envs, train.create_agent, train.create_logger

    def create_envs(*a, **kw):
        environments = original_envs(*a, **kw)
        raw = environments[0].envs.unwrapped
        term = raw.cfg.termination
        if not term.target_success_tolerance <= args.resume_tolerance <= term.success_tolerance:
            raise ValueError("Resume tolerance is outside original curriculum bounds")
        if term.eval_success_tolerance is not None:
            raise ValueError("Resume training must not pin eval_success_tolerance")
        raw._current_success_tolerance = args.resume_tolerance
        raw._frame_counter = raw._last_curriculum_update = 0
        raw._prev_episode_successes.zero_()
        return environments

    def create_logger(*a, **kw):
        logger = original_logger(*a, **kw)
        original_log = logger.log_metric
        logger.log_metric = lambda step: original_log(step=step + args.env_step_offset)
        return logger

    def create_agent(*a, **kw):
        agent = original_agent(*a, **kw)
        original_load_buffer = agent.load_replay_buffer

        def load_replay_buffer(path):
            if Path(path).resolve() != args.resume_checkpoint.resolve():
                raise ValueError("Replay checkpoint differs from requested resume checkpoint")
            original_load_buffer(path)  # Native train already loaded model/optimizer/RMS before this call.
            normalizer = agent.reward_normalizer
            normalizer.G_r.zero_()  # New environments: discard only their old in-flight discounted returns.
            proof = dict(checkpoint=str(args.resume_checkpoint.resolve()), env_step_offset=args.env_step_offset,
                         resume_tolerance=args.resume_tolerance, update_step=agent._update_step,
                         replay_length=len(agent._replay_buffer), replay_current_idx=agent._replay_buffer._current_idx,
                         cleared_G_r_elements=normalizer.G_r.numel(), reward_rms_count=float(normalizer.G_rms.count.item()),
                         exact_environment_resume=False,
                         boundary_note="Fresh environment/RNG/noise, empty n-step queue, native first random action; curriculum cooldown restarted.")
            (args.output_root / "resume_loaded.json").write_text(json.dumps(proof, indent=2) + "\n")
            print("RESUME_LOADED " + json.dumps(proof), flush=True)

        agent.load_replay_buffer = load_replay_buffer
        return agent

    def restore():
        train.create_envs, train.create_agent, train.create_logger = original_envs, original_agent, original_logger

    train.create_envs, train.create_agent, train.create_logger = create_envs, create_agent, create_logger
    return restore


def install_height_metrics(raw):
    import torch

    run = torch.zeros(raw.num_envs, dtype=torch.long, device=raw.device)
    longest, peak = torch.zeros_like(run), torch.zeros(raw.num_envs, device=raw.device)
    original_rewards, original_reset = raw._get_rewards, raw._reset_idx

    @functools.wraps(original_rewards)
    @torch.no_grad()
    def rewards():
        result = original_rewards()
        height = raw.object.data.root_pos_w[:, 2] - raw.scene.env_origins[:, 2] - raw._object_init_z
        run.copy_(torch.where(height > (.15 - .05), run + 1, 0))
        longest.copy_(torch.maximum(longest, run))
        peak.copy_(torch.maximum(peak, height))
        seconds = longest.float() * raw.step_dt  # Sampled occupancy, not contact/force evidence.
        raw.extras.setdefault("episode_final", {}).update(
            height_proxy_peak_delta_m=peak.clone(),
            height_proxy_longest_above_10cm_s=seconds.clone(),
            height_proxy_above_10cm_1s=(seconds >= 1.0).float().clone(),
            height_proxy_ever_above_10cm=(longest > 0).float().clone(),
        )
        return result

    @functools.wraps(original_reset)
    def reset(env_ids):
        result = original_reset(env_ids)
        # DirectRLEnv calls this AFTER reward/episode_final capture, never on goal-only changes.
        run[env_ids], longest[env_ids], peak[env_ids] = 0, 0, 0
        return result

    raw._get_rewards, raw._reset_idx = rewards, reset


def wrap_completion(env, output_root, expected_checkpoint):
    original_close = env.close

    @functools.wraps(original_close)
    def close(*args, **kwargs):
        if sys.exc_info()[0] is None:
            missing = [name for name in CHECKPOINT_NAMES if not (expected_checkpoint / f"{name}.pt").is_file()
                       or (expected_checkpoint / f"{name}.pt").stat().st_size == 0]
            state = dict(status="failed" if missing else "complete", missing=missing,
                         checkpoint=str(expected_checkpoint), replay_saved=not missing,
                         evidence="Native loop reached train_env.close after final checkpoint/replay saves.",
                         metric_note="Height above per-reset initial height; geometric proxy, not grasp success.")
            temporary = output_root / "training_complete.tmp"
            temporary.write_text(json.dumps(state, indent=2) + "\n")
            temporary.replace(output_root / "training_complete.json")
            if missing:
                raise RuntimeError(f"Final checkpoint/replay incomplete: {missing}")
        return original_close(*args, **kwargs)

    env.close = close
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-final-checkpoint", type=Path, required=True)
    parser.add_argument("--config-name", default="simtoolreal_full_arm1_hand1")
    parser.add_argument("--overrides", action="append", default=[])
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--resume-tolerance", type=float)
    parser.add_argument("--env-step-offset", type=int, default=0)
    args = parser.parse_args()
    try:
        load_overrides = resume_overrides(args)
    except ValueError as error:
        parser.error(str(error))
    args.output_root = args.output_root.expanduser().resolve()
    args.expected_final_checkpoint = args.expected_final_checkpoint.expanduser().resolve()
    args.expected_final_checkpoint.relative_to(args.output_root)
    if args.output_root.exists() and any(args.output_root.iterdir()):
        parser.error("Output root must be absent/empty; refusing overwrite")
    args.output_root.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    str_root = Path(os.environ.get("SIMTOOLREAL_ROOT", repo.parent / "simtoolreal")).resolve()
    sys.path[:0] = [str(repo), str(str_root)]
    os.chdir(str_root)
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    import train

    original_factory, original_argv0 = train.create_envs, sys.argv[0]
    def create_envs(*a, **kw):
        training, evaluation, recording = original_factory(*a, **kw)
        install_height_metrics(training.envs.unwrapped)
        return wrap_completion(training, args.output_root, args.expected_final_checkpoint), evaluation, recording
    train.create_envs, sys.argv[0] = create_envs, str(repo / "train.py")
    restore_resume = install_resume_hooks(train, args)
    try:
        train.run(argparse.Namespace(config_path=str(repo / "configs"), config_name=args.config_name,
                  overrides=[*args.overrides, *load_overrides, f"output_root={args.output_root}"]))
    finally:
        restore_resume()
        train.create_envs, sys.argv[0] = original_factory, original_argv0


if __name__ == "__main__":
    main()
