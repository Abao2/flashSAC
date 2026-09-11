"""Prepare (never launch) distinct full2500 reruns for completed partial evaluations."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if (root / "retention_completion_status.json").exists():
        parser.error("Completion queue has a status file; refusing to rewrite its manifest")
    source = json.loads((root / "retention_manifest.json").read_text())
    jobs, checked = [], []
    for original in source["jobs"]:
        path = Path(original["expected_completion_file"])
        if original["kind"] != "diagnostic" or not path.is_file():
            continue
        summary = json.loads(path.read_text())
        completed, requested = summary["episodes_completed"], summary["episodes_requested"]
        checked.append({"job": original["id"], "completed": completed, "requested": requested})
        if completed >= requested:
            continue
        job = {**original, "argv": list(original["argv"])}
        destination = path.parent.with_name(path.parent.name + "_full2500")
        job["id"] += "_full2500"
        job["argv"][job["argv"].index("--max-steps") + 1] = "2500"
        job["argv"][job["argv"].index("--output-dir") + 1] = str(destination)
        job["expected_completion_file"] = str(destination / "summary.json")
        job["log"] = str(root / "retention_completion_logs" / (job["id"] + ".log"))
        job.pop("depends_on", None)
        job["source_partial_summary"] = str(path)
        jobs.append(job)
    manifest = {"prepared_utc": datetime.now(timezone.utc).isoformat(), "prepared_only": True,
                "required_prior_status": str(root.parent / "budget50m/queue_status.json"),
                "launch_note": "Main only: existing runner requires explicit --after-status. This file never launches. Regenerate after original evaluations complete, before queue launch.",
                "hard_deadline_utc": "2026-09-09T01:00:00Z", "checked_summaries": checked, "jobs": jobs}
    output = root / "retention_completion_manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": str(output), "checked_summaries": len(checked), "partial_reruns": len(jobs), "launched": False}))


if __name__ == "__main__":
    main()
