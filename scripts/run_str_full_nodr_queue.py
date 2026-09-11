"""Run the newly authorized full-STR trial after its owned GPU preflight exits.

Reuse the serial queue implementation, not the expired overnight CLI deadline.
The new manifest supplies this experiment's explicit deadline.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import time

from run_str_diagnostic_queue import run_queue


def require_preflight(report):
    assert report["status"] == "PASS", report.get("error", "Preflight did not pass")
    assert report["gpu"]["timeout_final_obs"] == "PASS", "Natural timeout contract unverified"
    assets = report["gpu"]["asset_selection"]
    assert assets["six_types_selected"] == "PASS", assets
    assert assets["all_pool_assets_selected"] == "PASS", assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--preflight-unit", required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    deadline = datetime.fromisoformat(manifest["deadline_utc"]).timestamp()
    # Do not overlap a training process with the still-closing preflight app.
    while time.time() < deadline:
        result = subprocess.run(
            ["systemctl", "--user", "show", args.preflight_unit, "-p", "ActiveState", "--value"],
            capture_output=True, text=True, check=True, timeout=10)
        if result.stdout.strip() not in ("active", "activating", "deactivating"):
            break
        time.sleep(5)
    else:
        raise RuntimeError("Experiment deadline reached while awaiting preflight")
    require_preflight(json.loads(args.preflight.read_text()))
    print("PREFLIGHT_PASSED: launching serial full-STR queue", flush=True)
    result = run_queue(manifest, args.manifest.parent / "queue_status.json", deadline)
    print(json.dumps({"status": result["status"]}), flush=True)
    raise SystemExit(0 if result["status"] == "completed" else 1)


if __name__ == "__main__":
    main()
