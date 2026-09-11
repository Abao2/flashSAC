"""Prepare and CPU-validate the two continuous 50M jobs; never launches a job."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import hydra
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent
OVERNIGHT = ROOT.parent
REPO = OVERNIGHT.parents[1]
PYTHON = "/home/abao/Documents/Codex/flashsac-official-play/.venv/bin/python"
PHASE2 = OVERNIGHT / "control_ab"
DEPENDENCY = OVERNIGHT / "warmstart/retention_status.json"
STEPS = [9766, 19532, 29298, 39064, 48830]
BUDGET = 50_001_920


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def write_json(name, value):
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def flatten(value, prefix=""):
    if isinstance(value, dict):
        return {key: item for name, nested in value.items()
                for key, item in flatten(nested, f"{prefix}.{name}".strip(".")).items()}
    return {prefix: value}


def main():
    if (ROOT / "queue_status.json").exists():
        raise RuntimeError("Queue has a status file: do not rewrite a queued/running experiment manifest")
    assert REPO == Path("/home/abao/flashsac-robotics")
    assert BUDGET == 48830 * 1024 == 5 * 10_000_384
    source_manifest = json.loads((PHASE2 / "manifest.json").read_text())
    source_env = source_manifest["jobs"][0]["env"]
    env = {**source_env, "FLASH_SAC_OUTPUT_ROOT": str(ROOT)}
    permitted = {"num_env_steps", "env.num_env_steps", "num_interaction_steps", "num_update_steps",
                 "output_root", "group_name", "save_path", "recording_per_interaction_step"}
    OmegaConf.register_new_resolver("eval", lambda expression: eval(expression), replace=True)
    jobs, assertions, configs = [], {}, {}

    def job(job_id, kind, argv, completion, dependency=None):
        value = {"id": job_id, "kind": kind, "argv": argv, "cwd": str(REPO), "env": env,
                 "log": str(ROOT / "logs" / f"{job_id}.log"),
                 "timeout_seconds": 7200 if kind == "training" else 1200,
                 "expected_completion_file": str(completion)}
        if kind != "cpu":
            value["gpu_index"] = 0
        if dependency:
            value["depends_on"] = dependency
        jobs.append(value)
        return value

    for condition, hand in [("arm1_hand1", 1.0), ("arm1_hand01", 0.1)]:
        baseline_path = PHASE2 / f"configs/{condition}_seed0.json"
        baseline = json.loads(baseline_path.read_text())
        assert baseline["agent"]["learning_rate_warmup_step"] == 0
        assert baseline["agent"]["learning_rate_decay_step"] == 19532
        overrides = ["seed=0", f"num_env_steps={BUDGET}", "num_train_envs=1024",
                     "group_name=budget50m", f"exp_name={condition}", f"output_root={ROOT}",
                     f"save_path=models/{condition}/seed0", "save_checkpoint_per_interaction_step=9766",
                     "agent_load_path=null", "buffer_load_path=null",
                     "agent.learning_rate_warmup_step=0", "agent.learning_rate_decay_step=19532",
                     "recording_per_interaction_step=0",
                     "env.task_cfg_overrides.action.arm_moving_average=1.0",
                     f"env.task_cfg_overrides.action.hand_moving_average={hand}"]
        with hydra.initialize_config_dir(version_base=None, config_dir=str(REPO / "configs")):
            cfg = hydra.compose(config_name="simtoolreal_state_teacher", overrides=overrides)
        OmegaConf.resolve(cfg)
        resolved = OmegaConf.to_container(cfg, resolve=True)
        configs[condition] = resolved
        before, after = flatten(baseline), flatten(resolved)
        different = [key for key in sorted(set(before) | set(after)) if before.get(key) != after.get(key)]
        assert set(different) <= permitted, different
        for key in ["learning_rate_warmup_step", "learning_rate_decay_step", "buffer_max_length",
                    "buffer_min_length", "sample_batch_size", "actor_update_period"]:
            assert baseline["agent"][key] == resolved["agent"][key]
        assert resolved["num_interaction_steps"] == 48830
        assert resolved["num_update_steps"] == 97660
        assert resolved["agent"]["buffer_max_length"] == 10_000_000
        assert resolved["num_record_episodes"] == 0 and resolved["num_eval_episodes"] == 0
        assert resolved["num_record_envs"] is None and resolved["num_eval_envs"] is None
        assert resolved["agent_load_path"] is None and resolved["buffer_load_path"] is None
        assert resolved["metrics_per_interaction_step"] == baseline["metrics_per_interaction_step"] == 250
        assert resolved["logging_per_interaction_step"] == baseline["logging_per_interaction_step"] == 50
        config_path = f"configs/{condition}_seed0.json"
        write_json(config_path, resolved)
        assertions[condition] = {"baseline_config": str(baseline_path), "baseline_sha256": digest(baseline_path),
                                 "resolved_config": str(ROOT / config_path), "resolved_sha256": digest(ROOT / config_path),
                                 "differences_from_same_phase2_condition": different,
                                 "overrides": overrides, "first_10m_recipe_preserved": True}
        train_id = condition + "_seed0_train50m"
        command = [str(REPO / "scripts/run_str_state_teacher.sh")]
        for override in overrides:
            command += ["--overrides", override]
        train = job(train_id, "training", command,
                    ROOT / f"models/{condition}/seed0/step48830/agent_state.pt")
        train["resolved_config_file"] = str(ROOT / config_path)
        train["completion_marker_note"] = "agent_state.pt is saved last; inspect all six checkpoint files if interrupted"
        for step in STEPS:
            checkpoint = ROOT / f"models/{condition}/seed0/step{step}"
            for sampling, episodes in [("deterministic", 64), ("stochastic", 128)]:
                identifier = f"{condition}_seed0_step{step}_{sampling}"
                destination = ROOT / f"evaluations/{condition}_seed0/step{step}/{sampling}"
                argv = [PYTHON, str(REPO / "scripts/diagnose_str_rollouts.py"), "--policy", "flash",
                        "--checkpoint", str(checkpoint), "--entry", "wrapper", "--sampling", sampling,
                        "--noise-multiplier", "1.0", "--noise-repeat", "0", "--seed", "0",
                        "--episodes", str(episodes), "--num-envs", "32", "--max-steps", "2500",
                        "--override", "env.task_cfg_overrides.action.arm_moving_average=1.0",
                        "--override", f"env.task_cfg_overrides.action.hand_moving_average={hand}",
                        "--output-dir", str(destination)]
                job(identifier, "diagnostic", argv, destination / "summary.json", train_id)
            # Final diagnostic for both; early/middle support probes for hand=0.1.
            if step == 48830 or (hand == 0.1 and step in (9766, 29298)):
                identifier = f"{condition}_seed0_step{step}_critic"
                destination = ROOT / f"critics/{condition}_seed0_step{step}.json"
                data = ROOT / f"evaluations/{condition}_seed0/step{step}/stochastic/transitions.npz"
                argv = [PYTHON, str(REPO / "scripts/diagnose_str_critic.py"), "--checkpoint", str(checkpoint),
                        "--data", str(data), "--output", str(destination), "--next-obs-is-final",
                        "--device", "cpu", "--seed", "0", "--threads", "4",
                        "--max-samples-per-phase", "256", "--random-actions", "16",
                        "--batch-size", "128", "--gradient-repeats", "3", "--gamma", "0.99",
                        "--normalized-g-max", "5.0"]
                job(identifier, "cpu", argv, destination, f"{condition}_seed0_step{step}_stochastic")
    OmegaConf.clear_resolver("eval")
    a, b = flatten(configs["arm1_hand1"]), flatten(configs["arm1_hand01"])
    pair_differences = [key for key in sorted(a) if a[key] != b[key]]
    assert set(pair_differences) == {"exp_name", "save_path", "env.task_cfg_overrides.action.hand_moving_average"}
    assert len(jobs) == 26 and len({entry["id"] for entry in jobs}) == 26
    assert sum(entry["kind"] == "training" for entry in jobs) == 2
    assert sum(entry["kind"] == "diagnostic" for entry in jobs) == 20
    assert sum(entry["kind"] == "cpu" for entry in jobs) == 4
    for entry in jobs:
        assert entry["cwd"] == str(REPO)
        assert all(isinstance(argument, str) for argument in entry["argv"])
        assert Path(entry["argv"][0]).exists()
        assert (Path(entry["argv"][1]) if entry["argv"][0] == PYTHON else Path(entry["argv"][0])).exists()
        assert Path(entry["log"]).is_relative_to(ROOT)
        assert Path(entry["expected_completion_file"]).is_relative_to(ROOT)
    write_json("config_assertions.json", {"prepared_only": True, "launched": False,
               "all_checks_passed": True, "phase2_relative_allowed_differences": sorted(permitted),
               "pair_differences": pair_differences, "conditions": assertions,
               "fixed_learning_rate_warmup_step": 0, "fixed_learning_rate_decay_step": 19532,
               "training_transitions_each": BUDGET, "total_training_transitions": 2 * BUDGET,
               "checkpoint_steps": STEPS, "checkpoint_transitions": [step * 1024 for step in STEPS],
               "continuous_fresh_training_no_replay_reload": True,
               "nontraining_change": "recording cadence disabled; both old/new num_record_episodes=0 and recording env=None",
               "jobs": {"total": 26, "training": 2, "evaluations": 20, "cpu_critic": 4}})
    write_json("manifest.json", {"phase": "budget50m", "prepared_only": True,
               "required_prior_status": str(DEPENDENCY),
               "launch_guard": "MAIN must explicitly pass --after-status warmstart/retention_status.json; manifest field alone is not enforced by existing runner",
               "hard_deadline_utc": "2026-09-09T01:00:00Z",
               "scientific_contrast": "Same two seed0 hand-filter recipes as Phase2, continuous from-scratch 50M budget with original 10M learning-rate schedule frozen",
               "execution_order": "A train50M continuously; A saved-checkpoint evaluations; B train50M continuously; B evaluations. No mid-training simulator evaluations or replay reload.",
               "dependency_note": "depends_on is audit metadata only. Existing runner enforces serial list order and CLI --after-status queue barrier, not per-job success prerequisites.",
               "jobs": jobs})
    source_paths = [Path(item["resolved_path"]) for item in json.loads((PHASE2 / "source_hashes.json").read_text())["files"]
                    if not Path(item["resolved_path"]).is_relative_to(PHASE2)]
    source_paths += [REPO / "flash_rl/agents/utils/scheduler.py", Path(__file__), ROOT / "manifest.json", ROOT / "config_assertions.json"]
    source_paths += sorted((ROOT / "configs").glob("*.json"))
    write_json("source_hashes.json", {"captured_utc": datetime.now(timezone.utc).isoformat(),
               "captured_before_launch": True, "files": [{"path": str(path), "resolved_path": str(path.resolve()),
               "size_bytes": path.stat().st_size, "sha256": digest(path)} for path in sorted(set(source_paths))]})
    print(json.dumps({"status": "prepared_not_launched", "manifest": str(ROOT / "manifest.json"),
                      "jobs": len(jobs), "dependency_required": str(DEPENDENCY), "config_checks": "passed"}))


if __name__ == "__main__":
    main()
