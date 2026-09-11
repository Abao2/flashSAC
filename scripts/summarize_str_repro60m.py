"""Rerunnable CPU-only report for the three fresh continuous60M STR seeds."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_str_control_ab import checkpoint_audit, sha256
from summarize_str_retention import bracketing_tb, evaluation, tb_curves


GOAL = "episode/final/all_goals_hit"
TAGS = {GOAL, "episode/return", "episode/cumulative/lift_bonus_rew", "actor/entropy", "temperature/value"}


def checkpoint(path, transitions):
    if not (path / "agent_state.pt").is_file():
        return {"status": "pending", "path": str(path)}
    try:
        audit = checkpoint_audit(path)
        norm = torch.load(path / "reward_normalizer.pt", map_location="cpu", weights_only=True)
        temp = torch.load(path / "temperature.pt", map_location="cpu", weights_only=True)
        fields = {"sha256", "all_finite", "optimizer_applied_steps", "scheduler_last_epoch", "stored_update_step", "grad_scaler"}
        count = float(norm["G_rms_count"])
        return {"status": "saved_complete", "path": str(path), "all_finite": audit["all_finite"],
                "files": {name: {k: v for k, v in info.items() if k in fields} for name, info in audit["files"].items()},
                "alpha": float(next(v.exp() for k, v in temp["network_state_dict"].items() if k.endswith("log_temp"))),
                "reward_normalizer_count": count, "count_matches_budget": count == transitions,
                "reward_divisor": max(math.sqrt(float(norm["G_rms_var"]) + 1e-8), float(norm["G_r_max"]) / 5),
                "expected_native_calls": (transitions // 1024 - 97) * 2,
                "native_counter_matches_budget": audit["files"]["agent_state.pt"]["stored_update_step"] == (transitions // 1024 - 97) * 2}
    except (OSError, RuntimeError, EOFError) as error:
        return {"status": "read_error_or_write_in_progress", "path": str(path), "error": str(error)}


def read_evaluation(path, expected_checkpoint):
    try:
        result = evaluation(path)
        if result["status"] != "pending":
            metadata = json.loads((path.parent / "metadata.json").read_text())
            args = metadata["arguments"]
            assert args["seed"] == 0 and Path(args["checkpoint"]) == expected_checkpoint
            result["rollout_seed"] = args["seed"]
            result["checkpoint_matches"] = True
            result["metadata_sha256"] = sha256(path.parent / "metadata.json")
        return result
    except (OSError, RuntimeError, EOFError, ValueError, AssertionError) as error:
        return {"status": "read_error_or_invalid", "path": str(path), "error": str(error)}


def build_report(root):
    manifest = json.loads((root / "manifest.json").read_text())
    validation = json.loads((root / "validation.json").read_text())
    queue_path = root / "queue_status.json"
    queue = json.loads(queue_path.read_text()) if queue_path.exists() else {}
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "queue_status": queue.get("status", "pending"),
              "protocol": "Fresh random native feed-forward SAC, continuous60M; training seeds0/1/2, common rollout seed0; no BC or reload.",
              "manifest_sha256": sha256(root / "manifest.json"), "validation_sha256": sha256(root / "validation.json"),
              "source_hashes": validation["sha256"], "seeds": {}}
    for seed in manifest["training_seeds"]:
        runs = tb_curves(root / "runs" / "repro60m" / f"arm1_hand01_seed{seed}")
        checkpoints = {}
        for step in manifest["evaluated_vector_steps"]:
            path = root / "models" / f"seed{seed}" / f"step{step}"
            checkpoints[str(step)] = {"transitions": step * 1024, "checkpoint": checkpoint(path, step * 1024),
                "online_brackets": bracketing_tb(runs, step * 1024),
                "evaluations": {sampling: read_evaluation(root / "evaluations" / f"seed{seed}" / f"step{step}" / sampling / "summary.json", path)
                                for sampling in ("deterministic", "stochastic")}}
        compact = []
        for run in runs:
            values = {}
            for name, stats in run["curves"].items():
                if name in TAGS:
                    samples = stats.get("samples", [])
                    values[name] = {**{k: v for k, v in stats.items() if k != "samples"}, "last10_samples": samples[-10:]}
            compact.append({"directory": run["directory"], "curves": values})
        report["seeds"][str(seed)] = {"training_status": queue.get("jobs", {}).get(f"seed{seed}_train60m", {}).get("status", "pending"),
            "config_sha256": sha256(root / "configs" / f"seed{seed}.json"), "online_curves": compact, "checkpoints": checkpoints}
    report["limitations"] = ["Missing data is PENDING, not zero. Partial evaluations are labeled with completed counts.",
        "TB values are episode logging-window means, not an episode-weighted overall success rate; multiple run directories are not merged.",
        "Three independent training seeds, each evaluated with the same rollout seed0; correlated vector episodes do not justify an independent-IID confidence interval.",
        "Fixed eraser/start/goal, DR off, privileged simulator-state162; not paper24-task replication or real-robot validation.",
        "Extra continuous budget tests whether reload is unnecessary, but does not isolate the previous cold-replay/full-reload cause.",
        "Checkpoint counters count attempted native calls; AMP can skip applied optimizer steps. Full file audits and normalizer budget checks are separate."]
    return report


def score(result):
    if "counts" not in result:
        return result["status"].upper()
    prefix = "" if result["complete_budget"] else "PARTIAL "
    return f"{prefix}{result['counts']['success']}/{result['episodes']}"


def render(report):
    lines = ["# Fresh continuous60M — current CPU snapshot", "", f"Updated: {report['created_utc']}; queue: {report['queue_status']}.", "",
        report["protocol"], "", "|Training seed|Checkpoint|Training job|Online goal before/after checkpoint|Frozen det goals|Frozen native-stochastic goals|", "|---|---|---|---|---|---|"]
    for seed, run in report["seeds"].items():
        for step, item in run["checkpoints"].items():
            bracket = item["online_brackets"]
            if item.get("checkpoint", {}).get("status", "pending") != "saved_complete":
                online = "PENDING (checkpoint not saved)"
            elif len(bracket) == 1:
                points = bracket[0]["brackets"][GOAL]
                before = points["at_or_before"]
                after = points["at_or_after"]
                online = " / ".join(f"{p['value']:.2%} @{p['step']:,}" if p else "PENDING" for p in (before, after))
            else:
                online = "PENDING" if not bracket else "MULTIPLE RUNS — see JSON"
            ev = item["evaluations"]
            lines.append(f"|{seed}|{item['transitions']:,}|{run['training_status']}|{online}|{score(ev['deterministic'])}|{score(ev['stochastic'])}|")
    lines += ["", "Checkpoint details (exact attempted/applied counters, alpha, finite checks, normalizer sample counts and hashes) are in `summary.json`. Online windows and frozen evaluations are different metrics.", "", "## Limits", ""]
    lines += ["- " + value for value in report["limitations"]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = args.root.resolve()
    report = build_report(root)
    for name, content in (("summary.json", json.dumps(report, indent=2, allow_nan=False) + "\n"), ("analysis.md", render(report))):
        temporary = root / (name + ".tmp")
        temporary.write_text(content)
        temporary.replace(root / name)
    print(f"Saved CPU summary: {root}; queue={report['queue_status']}")


if __name__ == "__main__":
    main()
