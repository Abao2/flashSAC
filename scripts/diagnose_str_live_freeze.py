#!/usr/bin/env python3
"""UNRUN, UNTESTED DRAFT: preparation paused after successful final checkpoint.

Do not launch without a new review. Native train.run -> live freeze -> reset.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_str_warmstart import network_audit


class ProbeComplete(Exception):
    pass


def independent_geometry(torch, position, quaternion, offsets):
    """Quaternion wxyz rotation, independent of STR/Isaac geometry helpers."""
    vector = quaternion[:, None, 1:].expand_as(offsets)
    uv = torch.cross(vector, offsets, dim=-1)
    return position[:, None, :] + offsets + 2 * (
        quaternion[:, None, :1] * uv + torch.cross(vector, uv, dim=-1))


def advance_goal_counts(torch, distance, tolerance, previous, consecutive, required):
    near = distance <= tolerance
    count = previous + near.long()
    if consecutive:
        count = count * near.long()
    return near, count, count >= required


class LiveFreezeProbe:
    def __init__(self, output, train_steps=4883, phase_steps=1200, trigger_goals=16, window=50):
        self.output = Path(output)
        self.train_steps, self.phase_steps = train_steps, phase_steps
        self.trigger_goals, self.window = trigger_goals, window
        self.phase, self.step, self.phase_step = "learning", 0, 0
        self.agent = self.env = None
        self.window_goals = 0
        self.counts = Counter()
        self.state = {"status": "initialized", "windows": [], "checks": 0,
                      "notes": ["Cold-replay native continuation, NOT uninterrupted resume.",
                                "Freeze blocks all agent.update and process_transition calls; native sampling/zeta continues unchanged.",
                                "Clean reset is native reset(random_start_init=False); previous reward is NOT zeroed, matching existing reset semantics.",
                                "Crossing episodes are not counted as fully frozen; artificial phase-boundary resets are censored, not failures."]}

    def write(self):
        self.state.update(phase=self.phase, vector_steps=self.step, phase_steps=self.phase_step,
                          episode_counts=dict(self.counts))
        self.output.mkdir(parents=True, exist_ok=True)
        temporary = self.output / "summary.tmp"
        temporary.write_text(json.dumps(self.state, indent=2, allow_nan=False) + "\n")
        temporary.replace(self.output / "summary.json")

    def audit(self):
        return {"native_update_step": int(self.agent._update_step),
                **{name: network_audit(getattr(self.agent, "_" + name))
                   for name in ("actor", "critic", "target_critic", "temperature")}}

    def bind_agent(self, agent):
        self.agent = agent
        update, process = agent.update, agent.process_transition
        agent.update = lambda: update() if self.phase == "learning" else {}
        agent.process_transition = lambda transition: process(transition) if self.phase == "learning" else None
        return agent

    def bind_env(self, env):
        import numpy as np
        import torch

        self.env, self.raw = env, env.envs.unwrapped
        raw = self.raw
        if raw.cfg.termination.max_consecutive_successes != 1:
            raise ValueError("Probe intentionally supports the fixed single-goal task only")
        if self.phase_steps < 2 * raw.max_episode_length:
            raise ValueError("Each frozen phase must cover at least two maximum episode horizons")
        n = raw.num_envs
        self.near_count = torch.zeros(n, dtype=torch.long, device=raw.device)
        self.success_count = torch.zeros_like(self.near_count)
        self.episode_serial = np.full(n, -1, dtype=np.int64)
        self.episode_steps = np.zeros(n, dtype=np.int64)
        self.start_phase = np.full(n, "learning", dtype=object)
        self.state.update(num_envs=n, max_episode_length=raw.max_episode_length,
                          success_steps=raw.cfg.termination.success_steps,
                          consecutive=raw.cfg.termination.force_consecutive_near_goal_steps)
        reset_idx, get_dones, step = raw._reset_idx, raw._get_dones, env.step

        def reset(env_ids):
            result = reset_idx(env_ids)
            ids = np.arange(n) if env_ids is None else torch.as_tensor(env_ids).cpu().numpy()
            self.near_count[ids] = 0
            self.success_count[ids] = 0
            self.episode_serial[ids] += 1
            self.episode_steps[ids] = 0
            self.start_phase[ids] = self.phase
            return result

        def dones():
            terminated, truncated = get_dones()
            with torch.no_grad():
                origins = raw.scene.env_origins
                obj_pos, obj_rot = raw.object.data.root_pos_w - origins, raw.object.data.root_quat_w
                goal_pos, goal_rot = raw.goal_viz.data.root_pos_w - origins, raw.goal_viz.data.root_quat_w
                offsets = raw._keypoint_offsets_fixed if raw.cfg.reward.fixed_size_keypoint_reward else raw._keypoint_offsets
                obj_kp = independent_geometry(torch, obj_pos, obj_rot, offsets)
                goal_kp = independent_geometry(torch, goal_pos, goal_rot, offsets)
                distance = (obj_kp - goal_kp).norm(dim=-1).amax(dim=-1)
                tolerance = (raw._keypoint_success_tolerance_m() if hasattr(raw, "_keypoint_success_tolerance_m")
                             else raw._current_success_tolerance * raw.cfg.reward.keypoint_scale)
                near, self.near_count, success = advance_goal_counts(
                    torch, distance, tolerance, self.near_count,
                    raw.cfg.termination.force_consecutive_near_goal_steps, raw.cfg.termination.success_steps)
                self.success_count += success.long()
                hand_pos = raw.robot.data.body_state_w[:, raw._fingertip_body_ids, :3] - origins[:, None, :]
                hand_far = (hand_pos - obj_pos[:, None, :]).norm(dim=-1).amax(dim=-1) > 1.5
                dropped = ((obj_pos[:, 2] < raw._object_init_z) & raw._lifted_object
                           if raw.cfg.termination.reset_when_dropped else torch.zeros_like(success))
                expected_term = (obj_pos[:, 2] < .1) | (self.success_count >= 1) | hand_far | dropped
                expected_trunc = (raw.episode_length_buf >= raw.max_episode_length - 1) & ~success & ~expected_term
                checks = [(distance - raw._keypoints_max_dist).abs().max() <= 1e-5,
                          (near == raw._near_goal).all(), (self.near_count == raw._near_goal_steps).all(),
                          (success == raw._is_success).all(), (self.success_count == raw._successes).all(),
                          (terminated == expected_term).all(), (truncated == expected_trunc).all()]
                if not bool(torch.stack(checks).all()):
                    self.state["failed_checks"] = [bool(c.item()) for c in checks]
                    raise AssertionError("Independent pre-auto-reset geometry/success/termination mismatch")
                self.state["checks"] += n
                self.episode_steps += 1
                goals = int((success & terminated).sum().item())
                self.window_goals += goals
                ids = (terminated | truncated).nonzero(as_tuple=False).flatten()
                terminal = torch.cat((obj_pos[ids], obj_rot[ids], goal_pos[ids], goal_rot[ids],
                                      distance[ids, None], raw._keypoints_max_dist[ids, None],
                                      raw.episode_length_buf[ids, None].float()), dim=1).cpu().numpy()
                term, trunc, hits = terminated[ids].cpu().tolist(), truncated[ids].cpu().tolist(), success[ids].cpu().tolist()
                with (self.output / "episodes.jsonl").open("a") as stream:
                    for row, (index, values) in enumerate(zip(ids.cpu().tolist(), terminal)):
                        started = str(self.start_phase[index])
                        key = f"{self.phase}/started_{started}"
                        self.counts[key + "/episodes"] += 1
                        self.counts[key + "/goals"] += int(hits[row])
                        stream.write(json.dumps({"env_id": index, "episode_id": int(self.episode_serial[index] * n + index),
                            "end_vector_step": self.step + 1, "started_phase": started, "ended_phase": self.phase,
                            "fully_frozen": started == self.phase and self.phase != "learning",
                            "observed_steps": int(self.episode_steps[index]), "terminated": term[row], "truncated": trunc[row],
                            "independent_goal": hits[row], "environment_goal": hits[row],
                            "object_pose_wxyz": values[:7].tolist(), "goal_pose_wxyz": values[7:14].tolist(),
                            "independent_keypoint_error": float(values[14]), "environment_keypoint_error": float(values[15]),
                            "native_episode_progress": int(values[16]), "tolerance_m": float(tolerance)}, allow_nan=False) + "\n")
            return terminated, truncated

        def wrapped_step(actions):
            result = step(actions)
            self.step += 1
            self.phase_step += 1
            boundary = self.phase_step % self.window == 0
            if boundary:
                self.state["windows"].append({"phase": self.phase, "end_step": self.step,
                    "phase_step": self.phase_step, "true_goal_episodes": self.window_goals})
                self.write()
            if self.phase == "learning" and ((boundary and self.window_goals >= self.trigger_goals) or self.step >= self.train_steps):
                self.state["trigger"] = {"reason": "goal_window" if boundary and self.window_goals >= self.trigger_goals else "budget_fallback",
                    "vector_step": self.step, "transitions": self.step * n, "window_goal_episodes": self.window_goals}
                self.state["frozen_audit"] = self.audit()
                self.agent.save(str(self.output / "frozen_checkpoint"))
                self.phase, self.phase_step, self.window_goals = "live_frozen", 0, 0
                # An auto-reset on the trigger step precedes freeze, but no action has been taken in that new episode.
                self.start_phase[self.episode_steps == 0] = "live_frozen"
                self.write()
            elif self.phase == "live_frozen" and self.phase_step >= self.phase_steps:
                self.assert_frozen()
                self.state["live_end_audit"] = self.audit()
                self.state["live_end_censored_episodes"] = int((self.episode_steps > 0).sum())
                self.state["clean_reset_previous_reward"] = {"min": float(raw.reward_buf.min()), "mean": float(raw.reward_buf.mean()), "max": float(raw.reward_buf.max())}
                self.phase, self.phase_step, self.window_goals = "clean_frozen", 0, 0
                observations, info = env.reset(random_start_init=False)
                # No synthetic first random action or zeta reset: native loop consumes this exact reset observation.
                result = (observations, result[1], result[2], result[3], {**result[4], **info})
                self.write()
            elif self.phase == "clean_frozen" and self.phase_step >= self.phase_steps:
                self.assert_frozen()
                self.state["clean_end_censored_episodes"] = int((self.episode_steps > 0).sum())
                raise ProbeComplete()
            elif boundary:
                self.window_goals = 0
            return result

        raw._reset_idx, raw._get_dones, env.step = reset, dones, wrapped_step
        self.output.mkdir(parents=True, exist_ok=True)
        return env

    def assert_frozen(self):
        if self.audit() != self.state["frozen_audit"]:
            raise AssertionError("Actor/BN, critic/target, temperature, optimizer/scheduler or native counter changed after freeze")

    def finish(self, status, error=None):
        self.state["status"] = status
        if error:
            self.state["error"] = error
        if self.agent is not None:
            self.state["final_audit"] = self.audit()
        self.write()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-steps", type=int, default=4883)
    parser.add_argument("--phase-steps", type=int, default=1200)
    parser.add_argument("--trigger-goals", type=int, default=16)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--overrides", action="append", default=[])
    args = parser.parse_args()
    args.output_root = args.output_root.resolve()
    args.initial_checkpoint = args.initial_checkpoint.resolve()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        parser.error("Output root must be absent/empty; refusing overwrite")
    if min(args.train_steps, args.phase_steps, args.trigger_goals, args.num_envs) < 1:
        parser.error("All counts must be positive")
    for name in ("actor", "critic", "target_critic", "temperature", "reward_normalizer", "agent_state"):
        if not (args.initial_checkpoint / f"{name}.pt").is_file():
            parser.error(f"Missing native checkpoint: {name}.pt")
    repo = Path(__file__).resolve().parents[1]
    str_root = Path(os.environ.get("SIMTOOLREAL_ROOT", repo.parent / "simtoolreal")).resolve()
    sys.path[:0] = [str(repo), str(str_root)]
    os.chdir(str_root)
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    import train
    probe = LiveFreezeProbe(args.output_root, args.train_steps, args.phase_steps, args.trigger_goals)
    original_factory, original_envs = train.create_agent, train.create_envs
    train.create_agent = lambda *a, **kw: probe.bind_agent(original_factory(*a, **kw))

    def create_envs(*a, **kw):
        training, evaluation, recording = original_envs(*a, **kw)
        return probe.bind_env(training), evaluation, recording

    train.create_envs = create_envs
    original_argv0, sys.argv[0] = sys.argv[0], str(repo / "train.py")
    overrides = [*args.overrides, f"output_root={args.output_root}", f"agent_load_path={args.initial_checkpoint}",
        "require_agent_load=true", "agent.load_optimizer=true", "agent.load_reward_normalizer=true", "buffer_load_path=null",
        f"num_train_envs={args.num_envs}", f"num_env_steps={(args.train_steps + 2 * args.phase_steps) * args.num_envs}",
        "save_checkpoint_per_interaction_step=null", "num_eval_episodes=0", "num_record_episodes=0"]
    probe.state.update(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, effective_overrides=overrides)
    probe.write()
    try:
        train.run(argparse.Namespace(config_path=str(repo / "configs"), config_name="simtoolreal_state_teacher", overrides=overrides))
        raise RuntimeError("Native training loop ended before both frozen phases completed")
    except ProbeComplete:
        probe.finish("complete")
    except BaseException as error:
        probe.finish("failed", f"{type(error).__name__}: {error}")
        raise
    finally:
        train.create_agent, train.create_envs, sys.argv[0] = original_factory, original_envs, original_argv0
    if probe.env is not None:
        probe.env.close()


if __name__ == "__main__":
    main()
