"""Frozen FlashSAC STR random-reset, multi-object, 50-goal-chain evaluation.

This is not the paper's 24-task evaluation. --tolerance .075 evaluates the
training-start criterion; --tolerance .01 evaluates the curriculum floor.
Both are explicitly frozen criteria, not restored training curriculum state.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys


def freeze_tolerance(cfg, task_config_path, requested):
    """Read the registered YAML beneath Hydra's sparse task overrides."""
    from omegaconf import OmegaConf

    task = OmegaConf.merge(OmegaConf.load(task_config_path), cfg.env.task_cfg_overrides)
    initial = float(task.termination.success_tolerance)
    tolerance = requested if requested is not None else initial
    for key in ("success_tolerance", "target_success_tolerance"):
        OmegaConf.update(cfg, f"env.task_cfg_overrides.termination.{key}", tolerance, force_add=True)
    # Isaac Lab rejects float overrides of this None default; equal start/floor
    # already pin the criterion. Clear any inherited eval override as well.
    OmegaConf.update(cfg, "env.task_cfg_overrides.termination.eval_success_tolerance", None, force_add=True)
    OmegaConf.update(cfg, "env.task_cfg_overrides.termination.tolerance_curriculum_interval",
                     2**60, force_add=True)
    return initial, tolerance


class Episodes:
    """Assign a finite episode budget and reset accumulators only at real dones."""

    def __init__(self, requested, num_envs, chain_length, env_assets=None, asset_types=None):
        import numpy as np

        self.requested, self.chain_length = requested, chain_length
        self.ids = np.arange(num_envs)
        self.next_id = num_envs
        self.returns = np.zeros(num_envs)
        self.lengths = np.zeros(num_envs, dtype=int)
        self.resets = np.zeros(num_envs, dtype=int)
        self.rows, self.final_observations = [], []
        self.env_assets, self.asset_types = env_assets, asset_types

    def step(self, reward, terminated, truncated, final_info, state, lifted_index):
        import numpy as np

        done = terminated | truncated
        self.returns += reward
        self.lengths += 1
        completed = []
        for i in np.flatnonzero(done & (self.ids >= 0)):
            for key in ("successes", "all_goals_hit", "done_timeout"):
                if key not in final_info or not np.isfinite(final_info[key][i]):
                    raise RuntimeError(f"Missing/nonfinite pre-reset episode_final.{key}")
            goals = float(final_info["successes"][i])
            success = bool(final_info["all_goals_hit"][i])
            if goals != int(goals) or not 0 <= goals <= self.chain_length:
                raise RuntimeError(f"Invalid pre-reset goal count: {goals}")
            if success != (goals == self.chain_length):
                raise RuntimeError("all_goals_hit does not match whole-chain completion")
            if bool(final_info["done_timeout"][i]) != bool(truncated[i]):
                raise RuntimeError("Pre-reset timeout metric disagrees with truncation")
            row = {"episode_id": int(self.ids[i]), "env_id": int(i), "complete": True,
                   "return": float(self.returns[i]), "length": int(self.lengths[i]),
                   "goals_reached": int(goals), "all_goals_hit": success,
                   "lifted": bool(state[i, lifted_index] > .5),
                   "timeout": bool(truncated[i]), "truncated": bool(truncated[i]),
                   "terminated": bool(terminated[i]),
                   "termination_reasons": {key.removeprefix("done_"): bool(value[i])
                                           for key, value in final_info.items()
                                           if key.startswith("done_")}}
            if self.env_assets is not None:
                row.update(self.env_assets[i])
            self.rows.append(row)
            self.final_observations.append(state[i].copy())
            completed.append(row)
            self.ids[i] = self.next_id if self.next_id < self.requested else -1
            self.next_id += int(self.next_id < self.requested)
        self.resets += done
        self.returns[done] = 0
        self.lengths[done] = 0
        return completed

    def summary(self):
        import numpy as np

        complete = len(self.rows) == self.requested
        result = {"status": "complete" if complete else "incomplete",
                  "requested_episodes": self.requested, "completed_episodes": len(self.rows),
                  "complete_budget": complete,
                  "observed_auto_resets_per_env": self.resets.tolist(),
                  "success_definition": f"all {self.chain_length} goals reached in one episode",
                  "counts": {key: sum(row[key] for row in self.rows)
                             for key in ("all_goals_hit", "lifted", "timeout", "terminated")}}
        # Empty/incomplete runs are explicitly labeled; absent results are never a 0% score.
        for key in ("return", "length", "goals_reached", "all_goals_hit", "lifted"):
            values = [row[key] for row in self.rows]
            result[key] = {"mean": float(np.mean(values)) if values else None,
                           "count": len(values)}
        if self.env_assets is not None:
            types = sorted(set(self.asset_types))
            completed = Counter(row["asset_type"] for row in self.rows)
            assigned = Counter(asset["asset_type"] for asset in self.env_assets)
            unique_completed = len({row["asset_idx"] for row in self.rows})
            result["coverage"] = {
                "asset_pool_size": len(self.asset_types),
                "asset_pool_type_counts": dict(Counter(self.asset_types)),
                "assigned_env_type_counts": {name: assigned[name] for name in types},
                "completed_episode_type_counts": {name: completed[name] for name in types},
                "completed_episodes_per_env": [sum(row["env_id"] == i for row in self.rows)
                                               for i in range(len(self.ids))],
                "unique_completed_assets": unique_completed,
                "full_asset_pool_covered": unique_completed == len(self.asset_types),
                "equal_completed_type_counts": len({completed[name] for name in types}) == 1,
                "note": "Assets remain assigned to environments; small num-envs may cover only one family."}
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="Checkpoint directory containing actor.pt, or actor.pt itself")
    parser.add_argument("--config-name", default="simtoolreal_full_nodr")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tolerance", type=float,
                        help="Frozen criterion; default is config's initial success_tolerance")
    args = parser.parse_args()
    if args.episodes < 1 or args.num_envs < 1:
        parser.error("--episodes and --num-envs must be positive")
    if args.tolerance is not None and (not math.isfinite(args.tolerance) or args.tolerance <= 0):
        parser.error("--tolerance must be finite and positive")
    actor_path = args.checkpoint.expanduser().resolve()
    if actor_path.is_dir():
        actor_path /= "actor.pt"
    if not actor_path.is_file():
        parser.error(f"Actor checkpoint not found: {actor_path}")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error("--output-dir must be empty; evaluation never overwrites previous results")
    digest = hashlib.sha256()
    with actor_path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)

    repo = Path(__file__).resolve().parents[1]
    str_root = Path(os.environ.get("SIMTOOLREAL_ROOT", repo.parent / "simtoolreal")).resolve()
    sys.path[:0] = [str(repo), str(str_root)]
    os.chdir(str_root)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(name, "2")
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

    import hydra
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from flash_rl.agents import create_agent
    from flash_rl.envs.isaaclab import make_isaaclab_env
    from scripts.diagnose_str_rollouts import terminal_next_observation

    num_envs = min(args.num_envs, args.episodes)
    OmegaConf.register_new_resolver("eval", lambda expression: eval(expression), replace=True)
    with hydra.initialize_config_dir(version_base=None, config_dir=str(repo / "configs")):
        cfg = hydra.compose(config_name=args.config_name,
                            overrides=[f"seed={args.seed}", f"num_train_envs={num_envs}"])
    task_config_path = str_root / "isaacsimenvs/cfg/task/SimToolReal.yaml"
    initial_tolerance, tolerance = freeze_tolerance(cfg, task_config_path, args.tolerance)
    # No replay is populated and only actor weights are loaded; no learner calls.
    cfg.agent.buffer_max_length = cfg.agent.buffer_min_length = cfg.agent.sample_batch_size = 1
    cfg.agent.use_compile = cfg.agent.load_optimizer = cfg.agent.load_reward_normalizer = False
    OmegaConf.resolve(cfg)
    OmegaConf.clear_resolver("eval")  # STR registers it after AppLauncher starts.
    OmegaConf.save(cfg, output / "resolved_config.yaml")
    metadata = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                "checkpoint": str(actor_path), "actor_sha256": digest.hexdigest(),
                "config_name": args.config_name, "seed": args.seed,
                "registered_task_config": str(task_config_path),
                "requested_episodes": args.episodes, "requested_num_envs": args.num_envs,
                "actual_num_envs": num_envs, "deterministic": True, "learning": False,
                "requested_tolerance": args.tolerance, "frozen_tolerance": tolerance,
                "original_config_initial_tolerance": initial_tolerance,
                "tolerance_note": "Explicit frozen criterion; training curriculum state is not restored.",
                "protocol": "STR random reset, multi-object, 50-goal chains; NOT paper 24-task evaluation"}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env, episodes = None, None
    try:
        env = make_isaaclab_env(
            env_name=cfg.env.env_name, num_envs=num_envs, seed=args.seed, headless=True,
            device=cfg.env.device, action_bounds=cfg.env.action_bounds,
            registration_modules=cfg.env.registration_modules,
            env_cfg_yaml_entry_point=cfg.env.env_cfg_yaml_entry_point,
            task_cfg_overrides=cfg.env.task_cfg_overrides,
            bootstrap_timeouts=cfg.env.bootstrap_timeouts, success_path=cfg.env.success_path,
        )
        from isaacsimenvs.tasks.simtoolreal.utils.obs_utils import OBS_FIELD_SIZES

        obs, info = env.reset(random_start_init=False)
        raw = env.envs.unwrapped
        chain_length = int(raw.cfg.termination.max_consecutive_successes)
        if (chain_length != 50 or raw.cfg.reset.fixed_start_pose or raw.cfg.reset.fixed_goal_pose
                or raw.cfg.reset.fixed_trajectory_file):
            raise RuntimeError("Expected random start/goals and a 50-goal chain, not a fixed teacher task")
        fields, offset = {}, int(info["critic_observation_offset"])
        for name in raw.cfg.obs.state_list:
            fields[name] = offset
            offset += OBS_FIELD_SIZES[name]
        if offset != obs.shape[-1] or not np.isfinite(obs).all():
            raise RuntimeError("Invalid inferred critic observation layout or initial observations")
        metadata.update(actor_observation_size=list(info["actor_observation_size"]),
                        critic_observation_size=list(info["critic_observation_size"]),
                        critic_observation_offset=int(info["critic_observation_offset"]),
                        chain_length=chain_length, explicit_initial_reset_calls=1)
        (output / "actual_task_config.json").write_text(
            json.dumps(raw.cfg.to_dict(), indent=2, default=str) + "\n")
        asset_paths = raw._object_urdf_paths
        asset_types = [Path(path).name.split("_", 1)[1].split("_handle_", 1)[0]
                       for path in asset_paths]
        if not set(asset_types) <= set(raw.cfg.assets.handle_head_types):
            raise RuntimeError("Cannot identify procedural object types from actual asset filenames")
        asset_indices = raw._object_asset_index_per_env.detach().cpu().tolist()
        env_assets = [{"asset_idx": index, "asset_type": asset_types[index],
                       "asset_file": Path(asset_paths[index]).name} for index in asset_indices]
        (output / "env_assets.json").write_text(json.dumps(env_assets, indent=2) + "\n")
        agent = create_agent(env.observation_space, env.action_space, info, cfg.agent)
        saved = torch.load(actor_path, map_location="cpu", weights_only=True)
        weights = {key.removeprefix("_orig_mod."): value
                   for key, value in saved["network_state_dict"].items()}
        agent._actor.network.load_state_dict(weights, strict=True)
        agent._actor.network.eval()
        episodes = Episodes(args.episodes, num_envs, chain_length, env_assets, asset_types)
        # Each active slot finishes within 50 per-goal horizons; the finite
        # budget needs at most this many full waves, including slow episodes.
        max_steps = math.ceil(args.episodes / num_envs) * chain_length * (int(raw.max_episode_length) + 1)
        metadata.update(max_vector_steps=max_steps, vector_steps=0)
        with (output / "episodes.jsonl").open("w", buffering=1) as log, torch.inference_mode():
            for vector_step in range(max_steps):
                actions = agent.sample_actions(0, {"next_observation": obs}, training=False)
                obs, reward, terminated, truncated, info = env.step(actions)
                metadata["vector_steps"] = vector_step + 1
                if not np.isfinite(obs).all() or not np.isfinite(reward).all():
                    raise RuntimeError("Nonfinite evaluation observation/reward")
                if not math.isclose(float(info["current_success_tolerance"]), tolerance, abs_tol=1e-8):
                    raise RuntimeError("Evaluation success tolerance changed")
                state = terminal_next_observation(obs, terminated | truncated, info)
                rows = episodes.step(reward, terminated, truncated, info.get("episode_final", {}),
                                     state, fields["lifted_object"])
                for row in rows:
                    encoded = json.dumps(row, allow_nan=False)
                    log.write(encoded + "\n")
                    print("EVAL_EPISODE " + encoded, flush=True)
                if len(episodes.rows) == args.episodes:
                    break
            else:
                raise RuntimeError("Evaluation exceeded the 50-goal-chain episode horizon")
        metadata.update(vector_steps=vector_step + 1, status="complete")
    except Exception as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
        metadata["completed_episodes"] = len(episodes.rows) if episodes is not None else 0
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        if episodes is not None:
            summary = {**episodes.summary(), **metadata}
            (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
            if episodes.final_observations:
                np.savez_compressed(output / "terminal_observations.npz",
                                    episode_ids=np.array([row["episode_id"] for row in episodes.rows]),
                                    observations=np.stack(episodes.final_observations))
            if metadata["status"] == "complete":
                print("EVAL_RESULT " + json.dumps(summary, allow_nan=False), flush=True)
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
