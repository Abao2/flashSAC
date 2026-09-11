#!/usr/bin/env python3
"""Native STR trainer with an isolated critic-only gate and update-count snapshots.

Only the first N _update_networks calls force do_actor_update=False. Afterwards
native actor scheduling resumes. No reward, buffer, controller or core code edits.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import functools
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def network_audit(network):
    def digest(items):
        result = hashlib.sha256()
        for name, value in sorted(items):
            result.update(name.encode())
            result.update(value.detach().cpu().contiguous().numpy().tobytes())
        return result.hexdigest()

    optimizer = network.optimizer
    steps = [] if optimizer is None else [int(v["step"].item()) if hasattr(v["step"], "item")
             else int(v["step"]) for v in optimizer.state.values() if "step" in v]
    return {
        "parameter_digest": digest(network.network.named_parameters()),
        "buffer_digest": digest(network.network.named_buffers()),
        "optimizer_step_min": min(steps, default=0), "optimizer_step_max": max(steps, default=0),
        "learning_rates": [] if optimizer is None else [float(g["lr"]) for g in optimizer.param_groups],
        "scheduler_last_epoch": None if network.scheduler is None else network.scheduler.last_epoch,
    }


class CriticWarmupGate:
    """Small in-process wrapper; checkpoint snapshots use agent.save unmodified."""
    def __init__(self, original, warmup_updates, output_root, snapshot_updates):
        self.original, self.warmup_updates = original, warmup_updates
        self.output_root = Path(output_root)
        self.snapshot_updates = set(snapshot_updates)
        self.agent = None
        self.completed = 0
        self.attempted = {"critic": 0, "actor": 0, "temperature": 0}
        self.state = {
            "status": "initialized", "critic_only_network_calls": warmup_updates,
            "requested_update_snapshots": sorted(self.snapshot_updates), "snapshots": [],
            "notes": [
                "Online warmstart protection test, NOT offline fixed-dataset Q calibration.",
                "Native trainer still uses a random first vector action and random initial episode progress.",
                "Native actor scheduling resumes after the gate; actor/temp have fewer optimizer steps and a different scheduler position at equal environment budget.",
                "Actor inference BN and temperature remain unchanged during critic-only warmup; target/critic BN and EMA update natively.",
                "Snapshot update numbers are agent._update_step AFTER successful agent.update returns, not environment transitions.",
            ],
        }

    def audit(self):
        return {name: network_audit(getattr(self.agent, "_" + name))
                for name in ("actor", "critic", "temperature")}

    def write(self):
        self.state.update(completed_network_calls=self.completed, attempted_updates=dict(self.attempted),
                          updated_utc=datetime.now(timezone.utc).isoformat())
        self.output_root.mkdir(parents=True, exist_ok=True)
        temporary = self.output_root / "warmup_audit.tmp"
        temporary.write_text(json.dumps(self.state, indent=2, allow_nan=False) + "\n")
        temporary.replace(self.output_root / "warmup_audit.json")

    def bind(self, agent):
        if self.agent is not None:
            raise RuntimeError("Warmstart wrapper expects exactly one native training agent")
        self.agent = agent
        original_update = agent.update

        @functools.wraps(original_update)
        def update():
            if "initial" not in self.state:
                if int(agent._update_step) != 0:
                    raise ValueError("Expected fresh BC+random-Q initialization with agent update counter 0")
                self.state["initial"] = self.audit()
                self.state["status"] = "running"
                if self.warmup_updates == 0:
                    self.state["warmup_end"] = self.state["initial"]
                    self.state["warmup_end_network_call"] = 0
                    self.state["actor_and_temperature_frozen_during_warmup"] = True
                self.snapshot(0)
                self.write()
            result = original_update()
            native_step = int(agent._update_step)
            if native_step != self.completed:
                raise RuntimeError(f"Native update counter {native_step} != gated calls {self.completed}")
            self.snapshot(native_step)
            if native_step == self.warmup_updates or native_step % 1000 == 0:
                self.state["latest_native_agent_update_step"] = native_step
                self.state["latest_network_audit"] = self.audit()
                self.write()
            return result

        agent.update = update
        return agent

    def snapshot(self, step):
        if step not in self.snapshot_updates:
            return
        path = self.output_root / "update_snapshots" / f"update{step}"
        if path.exists():
            raise FileExistsError(f"Refusing snapshot overwrite: {path}")
        self.agent.save(str(path))
        self.state["snapshots"].append({"agent_update_step": step, "path": str(path),
            "attempted_updates": dict(self.attempted), "network_audit": self.audit()})
        self.write()

    def __call__(self, **kwargs):
        warmup = self.completed < self.warmup_updates
        kwargs["do_actor_update"] = bool(kwargs["do_actor_update"]) and not warmup
        self.attempted["critic"] += 1
        if kwargs["do_actor_update"]:
            self.attempted["actor"] += 1
            self.attempted["temperature"] += 1
        result = self.original(**kwargs)
        self.completed += 1
        if self.warmup_updates and self.completed == self.warmup_updates:
            end = self.audit()
            for name in ("actor", "temperature"):
                if end[name] != self.state["initial"][name]:
                    raise RuntimeError(f"{name} parameters, BN, optimizer or scheduler changed during critic-only warmup")
            self.state.update(warmup_end=end, warmup_end_network_call=self.completed,
                              actor_and_temperature_frozen_during_warmup=True)
            self.write()
        result.update({"warmstart/critic_only": float(warmup),
                       "warmstart/actor_updates_attempted": float(self.attempted["actor"]),
                       "warmstart/critic_updates_attempted": float(self.attempted["critic"])})
        return result

    def finish(self, status, error=None):
        self.state["status"] = status
        if error is not None:
            self.state["error"] = error
        if self.agent is not None:
            self.state["final_native_agent_update_step"] = int(self.agent._update_step)
            self.state["final_network_audit"] = self.audit()
        self.write()


def wrap_training_env_close(env, gate):
    """The native trainer closes train_env only after its loop/final save.

    SimulationApp.close may exit the process, so persist completion before it.
    Cleanup during exception unwinding must not overwrite a failed/running audit.
    """
    original_close = env.close

    @functools.wraps(original_close)
    def close(*args, **kwargs):
        if sys.exc_info()[0] is None:
            gate.state["completion_marker"] = "native_training_env_close_entered_after_final_save"
            gate.finish("complete")
        return original_close(*args, **kwargs)

    env.close = close
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--critic-warmup-updates", type=int, default=2000)
    parser.add_argument("--snapshot-updates", default="1,2,10,100,1000,2000,4000")
    parser.add_argument("--config-name", "--config_name", default="simtoolreal_state_teacher")
    parser.add_argument("--overrides", action="append", default=[])
    args = parser.parse_args()
    try:
        snapshots = [int(value) for value in args.snapshot_updates.split(",") if value]
    except ValueError:
        parser.error("--snapshot-updates must be comma-separated integer update counts")
    if args.critic_warmup_updates < 0 or any(value < 0 for value in snapshots):
        parser.error("Warmup/snapshot update counts must be nonnegative")
    args.initial_checkpoint = args.initial_checkpoint.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    for name in ("actor", "critic", "target_critic", "temperature", "reward_normalizer", "agent_state"):
        if not (args.initial_checkpoint / f"{name}.pt").is_file():
            parser.error(f"Missing native checkpoint file: {name}.pt")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        parser.error("Output root must be absent/empty; refusing overwrite")
    repo = Path(__file__).resolve().parents[1]
    str_root = Path(os.environ.get("SIMTOOLREAL_ROOT", repo.parent / "simtoolreal")).resolve()
    sys.path[:0] = [str(repo), str(str_root)]
    os.chdir(str_root)
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    import train
    import flash_rl.agents.flashSAC.agent as module
    gate = CriticWarmupGate(module._update_networks, args.critic_warmup_updates, args.output_root, snapshots)
    gate.state["arguments"] = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    original_factory = train.create_agent
    train.create_agent = lambda *a, **kw: gate.bind(original_factory(*a, **kw))
    original_env_factory = train.create_envs

    def create_envs(*a, **kw):
        training, evaluation, recording = original_env_factory(*a, **kw)
        return wrap_training_env_close(training, gate), evaluation, recording

    train.create_envs = create_envs
    original_network_update = module._update_networks
    module._update_networks = gate
    # Native trainer derives relative artifact paths from argv[0]. Use its own root.
    original_argv0, sys.argv[0] = sys.argv[0], str(repo / "train.py")
    overrides = [*args.overrides, f"output_root={args.output_root}",
                 f"agent_load_path={args.initial_checkpoint}", "require_agent_load=true",
                 "save_checkpoint_per_interaction_step=${num_interaction_steps}"]
    gate.state["effective_overrides"] = overrides
    gate.write()
    try:
        train.run(argparse.Namespace(config_path=str(repo / "configs"), config_name=args.config_name,
                                     overrides=overrides))
        gate.finish("complete")
    except BaseException as error:
        gate.finish("failed", f"{type(error).__name__}: {error}")
        raise
    finally:
        module._update_networks, train.create_agent, sys.argv[0] = original_network_update, original_factory, original_argv0
        train.create_envs = original_env_factory


if __name__ == "__main__":
    main()
