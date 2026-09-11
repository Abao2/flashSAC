"""CPU-only queue lifecycle checks; no Isaac, CUDA or nvidia-smi."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("str_queue", Path(__file__).parents[1] / "scripts/run_str_diagnostic_queue.py")
QUEUE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUEUE)


class QueueTest(unittest.TestCase):
    def test_dependency_readiness_and_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dependency = root / "phase1.json"
            self.assertEqual(QUEUE.dependency_state(dependency), "unavailable")
            for status in ("running", "waiting_for_gpu", "completed", "finished_with_failures", "interrupted"):
                dependency.write_text(json.dumps({"status": status}))
                self.assertEqual(QUEUE.dependency_state(dependency), status)
            result = QUEUE.run_queue({"jobs": []}, root / "phase2.json", time.time() + 5, after_status=dependency)
            self.assertEqual(result["status"], "dependency_failed")
            dependency.write_text(json.dumps({"status": "completed"}))
            result = QUEUE.run_queue({"jobs": []}, root / "phase2.json", time.time() + 5, after_status=dependency)
            self.assertEqual(result["status"], "completed")

    def test_gpu_guard_and_expired_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "done"
            job = {"id": "gpu", "argv": [sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).touch()"],
                   "cwd": str(root), "log": str(root / "gpu.log"), "kind": "diagnostic",
                   "timeout_seconds": 5, "expected_completion_file": str(marker)}
            with patch.object(QUEUE, "gpu_snapshot", side_effect=[{"free_mb": 100}, {"free_mb": 15000}]) as snapshot:
                result = QUEUE.run_queue({"jobs": [job]}, root / "state.json", time.time() + 5, poll_seconds=.001)
            self.assertEqual(snapshot.call_count, 2)
            self.assertEqual(result["jobs"]["gpu"]["status"], "completed")
            train_marker = root / "train_done"
            training = {**job, "id": "train", "kind": "training", "log": str(root / "train.log"),
                        "argv": [sys.executable, "-c", f"from pathlib import Path; Path({str(train_marker)!r}).touch()"],
                        "expected_completion_file": str(train_marker)}
            with patch.object(QUEUE, "gpu_snapshot", side_effect=[{"free_mb": 35000}, {"free_mb": 40000}]) as snapshot:
                result = QUEUE.run_queue({"jobs": [training]}, root / "train_state.json", time.time() + 5, poll_seconds=.001)
            self.assertEqual(snapshot.call_count, 2)
            self.assertEqual(result["jobs"]["train"]["required_free_mb"], 36 * 1024)
            self.assertEqual(result["jobs"]["train"]["status"], "completed")
            job["expected_completion_file"] = str(root / "other")
            result = QUEUE.run_queue({"jobs": [job]}, root / "past.json", time.time() - 1)
            self.assertEqual(result["status"], "deadline")
            self.assertFalse((root / "other").exists())

    def test_completion_skip_retry_and_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            done = root / "done.json"
            def job(name, code, marker, timeout=5):
                return {"id": name, "argv": [sys.executable, "-c", code], "cwd": str(root), "env": {},
                        "kind": "cpu", "log": str(root / f"{name}.log"),
                        "timeout_seconds": timeout, "expected_completion_file": str(marker)}
            success = job("success", f"from pathlib import Path; Path({str(done)!r}).write_text('ok')", done)
            failure = job("failure", "raise SystemExit(3)", root / "missing")
            timeout = job("timeout", "import time; time.sleep(30)", root / "missing2", .1)
            manifest = {"jobs": [success, failure, timeout]}
            state = QUEUE.run_queue(manifest, root / "queue_status.json", time.time() + 15, poll_seconds=.02)
            self.assertEqual([state["jobs"][k]["status"] for k in ("success", "failure", "timeout")],
                             ["completed", "failed", "timeout"])
            again = QUEUE.run_queue({"jobs": [success, failure]}, root / "queue_status.json", time.time() + 15, poll_seconds=.02)
            self.assertTrue(again["jobs"]["success"]["skipped_existing_completion"])
            self.assertEqual(again["jobs"]["failure"]["attempt"], 2)
            self.assertTrue((root / "failure.attempt2.log").is_file())


if __name__ == "__main__":
    unittest.main()
