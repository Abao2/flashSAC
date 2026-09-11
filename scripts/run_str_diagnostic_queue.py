"""Serial, deadline-bounded local experiment queue. No shell or foreign-process kills.

Manifest: {"jobs": [{"id": "name", "argv": ["python", "script.py"],
"cwd": "/path", "env": {}, "log": "/path/job.log", "kind": "diagnostic",
"timeout_seconds": 600, "expected_completion_file": "/path/summary.json"}]}.
Kinds: diagnostic (>=12 GiB free), training (>=36 GiB free), cpu (no GPU).
"""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

HARD_DEADLINE = datetime.fromisoformat("2026-09-09T01:00:00+00:00").timestamp()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def save_status(path, state):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2))
    temporary.replace(path)


def dependency_state(path):
    try:
        return json.loads(Path(path).read_text()).get("status", "unavailable")
    except (OSError, ValueError):
        return "unavailable"


def gpu_snapshot(index=0):
    try:
        command = ["nvidia-smi", "--format=csv,noheader,nounits"]
        memory = subprocess.run(command + ["--query-gpu=index,memory.free"],
                                capture_output=True, text=True, check=True, timeout=10).stdout
        processes = subprocess.run(command + ["--query-compute-apps=gpu_uuid,pid,process_name,used_memory"],
                                   capture_output=True, text=True, check=True, timeout=10).stdout
        free = {int(row.split(",")[0]): int(row.split(",")[1]) for row in memory.splitlines()}
        return {"checked_utc": utc_now(), "gpu_index": index, "free_mb": free[index],
                "compute_processes": processes.splitlines()}
    except (OSError, subprocess.SubprocessError, ValueError, KeyError) as error:
        return {"checked_utc": utc_now(), "free_mb": None, "error": str(error)}


def stop_owned_process(process, grace_seconds=10):
    """Only accepts the Popen child started with start_new_session=True below."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
    except ProcessLookupError:
        process.wait()


def run_queue(manifest, status_path, deadline, poll_seconds=1, after_status=None):
    jobs = manifest["jobs"]
    ids = [job["id"] for job in jobs]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate job ids")
    for job in jobs:
        if not isinstance(job["argv"], list) or not job["argv"] or not all(isinstance(x, str) for x in job["argv"]):
            raise ValueError("Each argv must be a nonempty string list; no shell strings")
        if job.get("kind", "diagnostic") not in ("diagnostic", "training", "cpu") or job["timeout_seconds"] <= 0:
            raise ValueError("Invalid job kind or timeout")
    status_path.parent.mkdir(parents=True, exist_ok=True)
    with status_path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads(status_path.read_text()) if status_path.exists() else {"jobs": {}}
        state.update(started_utc=utc_now(), deadline_utc=datetime.fromtimestamp(deadline, timezone.utc).isoformat(), status="running")
        stop_requested = False

        def request_stop(signum, frame):
            nonlocal stop_requested
            stop_requested = True

        handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
        save_status(status_path, state)
        try:
            while after_status is not None and not stop_requested and time.time() < deadline:
                dependency = dependency_state(after_status)
                state["dependency"] = {"status_file": str(after_status), "status": dependency}
                if dependency in ("completed", "finished_with_failures"):
                    state["status"] = "running"
                    save_status(status_path, state)
                    break
                if dependency in ("deadline", "interrupted", "dependency_failed"):
                    state["status"] = "dependency_failed"
                    return state
                state["status"] = "waiting_dependency"
                save_status(status_path, state)
                time.sleep(min(5, max(0, deadline - time.time())))
            for job in jobs:
                record = state["jobs"].setdefault(job["id"], {})
                record.setdefault("status", "pending")
                marker = Path(job["expected_completion_file"])
                if marker.is_file():
                    record.update(status="completed", skipped_existing_completion=True, checked_utc=utc_now())
                    save_status(status_path, state)
                    continue
                if stop_requested or time.time() >= deadline:
                    break
                kind = job.get("kind", "diagnostic")
                needed = {"cpu": 0, "diagnostic": 12 * 1024, "training": 36 * 1024}[kind]
                while needed and not stop_requested and time.time() < deadline:
                    snapshot = gpu_snapshot(job.get("gpu_index", 0))
                    record.update(status="waiting_for_gpu", gpu=snapshot, required_free_mb=needed)
                    save_status(status_path, state)
                    if snapshot["free_mb"] is not None and snapshot["free_mb"] >= needed:
                        break
                    for _ in range(30):
                        if stop_requested or time.time() >= deadline:
                            break
                        time.sleep(min(poll_seconds, max(0, deadline - time.time())))
                if stop_requested or time.time() >= deadline:
                    break
                base = Path(job["log"])
                base.parent.mkdir(parents=True, exist_ok=True)
                log, attempt = base, 1
                while log.exists():
                    attempt += 1
                    log = base.with_name(f"{base.stem}.attempt{attempt}{base.suffix}")
                process = None
                record.update(status="running", started_utc=utc_now(), log=str(log), attempt=attempt,
                              argv=job["argv"], cwd=job["cwd"], expected_completion_file=str(marker),
                              skipped_existing_completion=False)
                save_status(status_path, state)
                started = time.monotonic()
                try:
                    with log.open("x") as stream:
                        process = subprocess.Popen(job["argv"], cwd=job["cwd"], env={**os.environ, **job.get("env", {})},
                                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                        record["pid"] = process.pid
                        save_status(status_path, state)
                        while process.poll() is None:
                            reason = ("interrupted" if stop_requested else "deadline" if time.time() >= deadline else
                                      "timeout" if time.monotonic() - started >= job["timeout_seconds"] else None)
                            if reason:
                                stop_owned_process(process, grace_seconds=min(10, max(0, deadline - time.time())))
                                record["status"] = reason
                                break
                            time.sleep(min(poll_seconds, max(0, deadline - time.time())))
                        record["returncode"] = process.wait()
                        if record["status"] == "running":
                            record["status"] = "completed" if record["returncode"] == 0 and marker.is_file() else "failed"
                except Exception as error:
                    record.update(status="failed", error=f"{type(error).__name__}: {error}")
                finally:
                    if process is not None:
                        stop_owned_process(process, grace_seconds=min(10, max(0, deadline - time.time())))
                    record.update(finished_utc=utc_now(), elapsed_seconds=time.monotonic() - started)
                    save_status(status_path, state)
            complete = all(state["jobs"].get(job["id"], {}).get("status") == "completed" for job in jobs)
            state["status"] = "completed" if complete else "interrupted" if stop_requested else "deadline" if time.time() >= deadline else "finished_with_failures"
        finally:
            state["finished_utc"] = utc_now()
            save_status(status_path, state)
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--status", type=Path)
    parser.add_argument("--after-status", type=Path,
                        help="Wait for another queue's completed/finished_with_failures state before any jobs")
    parser.add_argument("--deadline", default="2026-09-09T01:00:00Z", help="May shorten, never extend, the hard UTC deadline")
    args = parser.parse_args()
    parsed = datetime.fromisoformat(args.deadline.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parser.error("Deadline requires a timezone")
    result = run_queue(json.loads(args.manifest.read_text()), args.status or args.manifest.parent / "queue_status.json",
                       min(parsed.timestamp(), HARD_DEADLINE), after_status=args.after_status)
    print(json.dumps({"status": result["status"], "jobs": {k: v.get("status", "pending") for k, v in result["jobs"].items()}}))
    raise SystemExit(0 if result["status"] in ("completed", "finished_with_failures") else 1)
