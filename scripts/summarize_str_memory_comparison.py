"""CPU-only training-window report; no simulator, checkpoint evaluation, or queue writes."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import mean

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


BASELINE = Path(__file__).resolve().parents[1] / "diagnostics/full_nodr_20260909_1004"
BUDGET = 100_003_840
WINDOW = 10_000_000
MILESTONES = (20_000_000, 40_000_000, 60_000_000, 80_000_000, 100_000_000)
TAGS = {
    "return": "episode/return",
    "lift_event_fraction": "episode/cumulative/lift_bonus_rew",
    "goals_per_episode": "episode/final/successes",
    "keypoint_reward": "episode/cumulative/keypoint_rew",
    "tolerance": "task/current_success_tolerance",
    "alpha": "temperature/value",
    "entropy": "actor/entropy",
}
NOTES = [
    "CPU-only TensorBoard training snapshot; no evaluation or deployment evidence.",
    "Each statistic is an unweighted mean of logged windows in the preceding 10M transitions, not an episode-weighted overall rate.",
    "Lift fraction = lift bonus / 300: an episode crossed the task height threshold at least once; not stable grasping success.",
    "Goals/episode counts reached goals, not the fraction completing the 50-goal chain.",
    "Missing data are PENDING/null, not zero. Seeds and distinct run directories remain separate.",
    "A reuses existing full STR obs140 feed-forward runs. B uses state162 feed-forward actor.",
    "C uses a finite 32-frame LSTM actor with obs140 and history-Q (current state162 + 32 * 141 = 4674 inputs before actions).",
    "C changes actor/critic capacity and critic information; it is not an actor-only causal ablation or official full-episode STR LSTM.",
    "Matched references use identical window endpoints no later than either run's last observation; do not compare baseline100M with candidate10M as a win/loss.",
]


def read_runs(parent):
    runs = []
    # Read a directory once: EventAccumulator joins its event files correctly.
    for directory in sorted({p.parent for p in parent.rglob("events.out.tfevents*")}):
        run = {"directory": str(directory), "curves": {}, "status": "PENDING"}
        try:
            accumulator = EventAccumulator(str(directory), size_guidance={"scalars": 0})
            accumulator.Reload()
            available = accumulator.Tags()["scalars"]
            for key, tag in TAGS.items():
                # Resolve repeated steps by newest wall time, not duplicate weighting.
                events = sorted(accumulator.Scalars(tag), key=lambda e: e.wall_time) if tag in available else []
                unique = {event.step: event.value for event in events}
                run["curves"][key] = sorted(unique.items())
            returns = run["curves"]["return"]
            run["last_observed_transition"] = returns[-1][0] if returns else None
            run["status"] = "OBSERVED" if returns else "PENDING"
        except (OSError, ValueError, RuntimeError) as error:
            run.update(status="READ_ERROR", error=str(error), last_observed_transition=None)
        runs.append(run)
    return runs


def window_stats(curves, end):
    result = {"start_exclusive": max(0, end - WINDOW), "end_inclusive": end, "metrics": {}}
    for key in TAGS:
        points = [(step, value) for step, value in curves.get(key, []) if max(0, end - WINDOW) < step <= end]
        if not points:
            result["metrics"][key] = None
            continue
        divisor = 300.0 if key == "lift_event_fraction" else 1.0
        values = [value / divisor for _, value in points]
        finite = all(math.isfinite(value) for value in values)
        result["metrics"][key] = {
            "status": "OBSERVED" if finite else "NONFINITE",
            "windows": len(points), "first_step": points[0][0], "last_step": points[-1][0],
            "mean": mean(values) if finite else None,
            "last": values[-1] if math.isfinite(values[-1]) else None,
        }
    return result


def summarize_run(run):
    result = {key: value for key, value in run.items() if key != "curves"}
    end = run.get("last_observed_transition")
    result["budget_observed"] = end is not None and end >= BUDGET
    result["last_10m"] = window_stats(run["curves"], end) if end is not None else None
    result["milestones"] = {
        str(step): window_stats(run["curves"], step) if end is not None and end >= step else {"status": "PENDING"}
        for step in MILESTONES
    }
    return result


def queue_snapshot(root):
    path = root / "queue_status.json"
    if not path.exists():
        return {"status": "PENDING", "path": str(path)}
    try:
        queue = json.loads(path.read_text())
        return {"status": queue.get("status", "UNSPECIFIED"), "path": str(path),
                "jobs": {key: job.get("status", "UNSPECIFIED") for key, job in queue.get("jobs", {}).items()}}
    except (OSError, ValueError) as error:
        return {"status": "READ_ERROR", "path": str(path), "error": str(error)}


def build_report(root, baseline=BASELINE):
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "notes": NOTES,
              "queue": queue_snapshot(root), "baseline_queue": queue_snapshot(baseline), "groups": {}, "matched": []}
    raw = {}
    for group, prefix in (("A", "full_distribution"), ("B", "state162_ff"), ("C", "obs140_lstm32_historyq")):
        report["groups"][group] = {}
        for seed in range(3):
            parent = (baseline / "runs/str_full_nodr" if group == "A" else root / "runs/str_memory_comparison") / f"{prefix}_seed{seed}"
            runs = read_runs(parent)
            raw[group, seed] = runs
            report["groups"][group][str(seed)] = {"path": str(parent), "status": "PENDING" if not runs else "SEE_RUNS",
                                                   "runs": [summarize_run(run) for run in runs]}
    for seed in range(3):
        for group in ("B", "C"):
            matches = []
            for a in raw["A", seed]:
                for candidate in raw[group, seed]:
                    if a.get("last_observed_transition") is None or candidate.get("last_observed_transition") is None:
                        continue
                    end = min(a["last_observed_transition"], candidate["last_observed_transition"])
                    matches.append({"baseline_directory": a["directory"], "candidate_directory": candidate["directory"],
                                    "end_transition": end, "A": window_stats(a["curves"], end),
                                    group: window_stats(candidate["curves"], end)})
            report["matched"].append({"candidate": group, "seed": seed, "status": "OBSERVED" if matches else "PENDING", "runs": matches})
    return report


def metric_cells(window):
    cells = []
    for key in TAGS:
        item = window["metrics"].get(key)
        value = item.get("mean") if item else None
        cells.append("PENDING" if item is None else ("NONFINITE" if value is None else f"{value:.6g}"))
    return " | ".join(cells)


def markdown(report):
    lines = ["# STR memory comparison — training windows", "", f"Updated: {report['created_utc']}", "",
             f"Queue: {report['queue']['status']}", "", "## Interpretation", ""]
    lines += [f"- {note}" for note in report["notes"]]
    header = "| Group / seed / run | Window end (M) | Return | Lift event fraction | Goals/episode | Keypoint reward | Tolerance | Alpha | Entropy |"
    divider = "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    lines += ["", "## Latest preceding 10M windows (different progress; not a ranking)", "", header, divider]
    for group, seeds in report["groups"].items():
        for seed, entry in seeds.items():
            if not entry["runs"]:
                lines.append(f"| {group} / {seed} | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |")
            for number, run in enumerate(entry["runs"], 1):
                label = f"{group} / {seed} / {number}"
                window = run["last_10m"]
                lines.append(f"| {label} | {window['end_inclusive']/1e6:.6f} | {metric_cells(window)} |" if window else
                             f"| {label} | {run['status']} | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |")
    lines += ["", "## Same-progress baseline references", "", header, divider]
    for match in report["matched"]:
        if not match["runs"]:
            lines.append(f"| A vs {match['candidate']} / {match['seed']} | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |")
        for number, run in enumerate(match["runs"], 1):
            for group in ("A", match["candidate"]):
                lines.append(f"| {group} / {match['seed']} / match{number} | {run['end_transition']/1e6:.6f} | {metric_cells(run[group])} |")
    lines += ["", "## Fixed milestones (preceding 10M windows)", "", header, divider]
    for group, seeds in report["groups"].items():
        for seed, entry in seeds.items():
            for number, run in enumerate(entry["runs"], 1):
                for step, window in run["milestones"].items():
                    values = " | ".join(["PENDING"] * len(TAGS)) if window.get("status") == "PENDING" else metric_cells(window)
                    lines.append(f"| {group} / {seed} / {number} | {int(step)/1e6:.0f} | {values} |")
    lines += ["", "Missing groups have no milestones yet. Exact run paths, sample counts, actual last logged steps, queue job states, and pending values are in `summary.json`.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--completion-marker", type=Path, help="Optional queue marker, not training success.")
    args = parser.parse_args()
    report = build_report(args.root)
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (args.root / "REPORT.md").write_text(markdown(report))
    if args.completion_marker is not None:
        args.completion_marker.write_text(json.dumps({"status": "summary_generated", "created_utc": report["created_utc"]}) + "\n")
    print(f"Wrote {args.root / 'summary.json'} and {args.root / 'REPORT.md'}")


if __name__ == "__main__":
    main()
