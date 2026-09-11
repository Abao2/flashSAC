"""Read-only CPU manifest/config/hash review; never launches or rewrites jobs."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import hydra
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def values(argv, flag):
    return [argv[i + 1] for i, item in enumerate(argv[:-1]) if item == flag]


def main():
    OmegaConf.register_new_resolver("eval", lambda x: eval(x), replace=True)
    report = {"checked_utc": datetime.now(timezone.utc).isoformat(), "reviewer_did_not_change_manifests": True,
              "gpu_or_queue_launched": False, "manifests": {}, "source_captures": {}, "warnings": []}
    for path in (ROOT / "retention_manifest.json", ROOT.parent / "budget50m/manifest.json"):
        manifest = json.loads(path.read_text())
        jobs = manifest["jobs"]
        assert len({job["id"] for job in jobs}) == len(jobs)
        assert len({job["expected_completion_file"] for job in jobs}) == len(jobs)
        seen, configurations, cap_warnings = set(), {}, []
        for job in jobs:
            argv = job["argv"]
            assert Path(argv[0]).is_file() and Path(job["cwd"]).is_dir()
            assert job.get("depends_on") is None or job["depends_on"] in seen
            seen.add(job["id"])
            if job["kind"] == "training":
                overrides = values(argv, "--overrides")
                if "--initial-checkpoint" in argv:
                    overrides += [f"output_root={values(argv, '--output-root')[0]}",
                                  f"agent_load_path={values(argv, '--initial-checkpoint')[0]}",
                                  "require_agent_load=true", "save_checkpoint_per_interaction_step=${num_interaction_steps}"]
                with hydra.initialize_config_dir(version_base=None, config_dir=str(REPO / "configs")):
                    config = OmegaConf.to_container(hydra.compose(config_name="simtoolreal_state_teacher", overrides=overrides), resolve=True)
                assert config == json.loads(Path(job["resolved_config_file"]).read_text()), job["id"]
                assert config["agent"]["learning_rate_decay_step"] == 19532
                assert config["agent"]["learning_rate_warmup_step"] == 0
                assert config["num_train_envs"] == 1024 and config["agent"]["buffer_max_length"] == 10000000
                final = Path(config["output_root"]) / config["save_path"] / f"step{int(config['num_interaction_steps'])}" / "agent_state.pt"
                assert final == Path(job["expected_completion_file"])
                configurations[job["id"]] = {"recomposed_config_exactly_matches": True,
                    "config_sha256": digest(job["resolved_config_file"]), "transitions": config["num_env_steps"],
                    "controls": config["env"]["task_cfg_overrides"]["action"],
                    "fresh_training": config["agent_load_path"] is None}
            elif job["kind"] == "diagnostic":
                assert values(argv, "--policy") == ["flash"] and values(argv, "--entry") == ["wrapper"]
                assert values(argv, "--num-envs") == ["32"] and values(argv, "--seed") == ["0"]
                assert values(argv, "--noise-multiplier") == ["1.0"] and values(argv, "--noise-repeat") == ["0"]
                sample = values(argv, "--sampling")[0]
                assert int(values(argv, "--episodes")[0]) == {"deterministic": 64, "stochastic": 128}[sample]
                assert Path(values(argv, "--output-dir")[0]) / "summary.json" == Path(job["expected_completion_file"])
                assert "--training-startup" not in argv
                overrides = dict(item.split("=", 1) for item in values(argv, "--override"))
                for part in ("arm", "hand"):
                    assert float(overrides[f"env.task_cfg_overrides.action.{part}_moving_average"]) == config["env"]["task_cfg_overrides"]["action"][f"{part}_moving_average"]
                required = 600 * ((int(values(argv, "--episodes")[0]) + 31) // 32)
                if int(values(argv, "--max-steps")[0]) < required:
                    cap_warnings.append({"job": job["id"], "max_steps": int(values(argv, "--max-steps")[0]), "conservative_600step_budget": required})
            else:
                assert values(argv, "--device") == ["cpu"]
        report["manifests"][str(path)] = {"sha256": digest(path), "jobs": len(jobs), "configs": configurations,
                                         "unique_outputs_and_dependency_order_valid": True, "partial_episode_cap_risks": cap_warnings}
    for path in (ROOT.parent / "control_ab/source_hashes.json", ROOT.parent / "budget50m/source_hashes.json"):
        previous = json.loads(path.read_text())
        rows = [{"path": row["path"], "old_sha256": row["sha256"], "current_sha256": digest(row["path"])} for row in previous["files"]]
        changed = [row for row in rows if row["old_sha256"] != row["current_sha256"]]
        report["source_captures"][str(path)] = {"matching": len(rows) - len(changed), "captured": len(rows), "changed": changed,
            "current_fingerprint": rows, "delta_note": "Parent confirms intentional default-off --training-startup diagnostic flag in diagnose_str_rollouts.py; reproduces randomized progress/first random action. Existing manifests do not enable it. No training core change. Original capture preserved."}
        assert all(Path(row["path"]).name == "diagnose_str_rollouts.py" for row in changed)
    report["new_diagnostic_wrapper_read_time_sha256"] = {str(REPO / "scripts" / name): digest(REPO / "scripts" / name)
        for name in ("train_str_warmstart.py", "run_str_diagnostic_queue.py", "diagnose_str_rollouts.py")}
    report["warnings"] = ["Existing runner regards summary-file existence as completion; a partial rollout still needs its actual denominator checked.",
        "A1300-step cap cannot guarantee128 episodes with32env and600-step timeouts. Current disk manifest has2500 stochastic and1300 deterministic, both sufficient; earlier read showed1300 stochastic. Reviewer did not modify it. Check per-job recorded argv for actual execution.",
        "Job depends_on is documentation; serial runner may continue after a failed job. Missing inputs must remain failures, not zero metrics.",
        "Queue-level --after-status accepts finished_with_failures; budget is intentionally independent of retention success, not gated by scientific outcome.",
        "Update snapshots are network calls, not actor optimizer steps or environment transitions; warmup comparisons have unequal actor update counts.",
        "Initial BC std was expert-calibrated, not SAC trained. Per-step alphaH is not discounted total return or an optimum proof."]
    (ROOT / "independent_queue_review.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"configs_verified": sum(len(m["configs"]) for m in report["manifests"].values()),
                      "cap_risks": sum(len(m["partial_episode_cap_risks"]) for m in report["manifests"].values()),
                      "sources": {path: {k: v[k] for k in ("matching", "captured")} for path, v in report["source_captures"].items()}}))


if __name__ == "__main__":
    main()
