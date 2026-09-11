"""Small CPU checks for pending/partial retention artifacts; no Isaac/GPU."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from summarize_str_retention import evaluation


class RetentionSummaryTest(unittest.TestCase):
    def test_pending_is_not_zero_success(self):
        with tempfile.TemporaryDirectory() as folder:
            result = evaluation(Path(folder) / "summary.json")
            self.assertEqual(result["status"], "pending")
            self.assertNotIn("counts", result)

    def test_partial_denominator_and_distinct_rerun(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root / "stochastic"
            output.mkdir()
            rows = []
            flags = ["success", "ever_task_lift", "ever_sustained_lift_near_palm_proxy",
                     "dropped_below_reset_after_task_lift", "terminated", "truncated"]
            for index in range(2):
                rows.append({"episode_id": index, "complete": True,
                             **{key: bool(index) for key in flags}, "reward_components": {"total_reward": 10.0}})
            (output / "episodes.jsonl").write_text("\n".join(map(json.dumps, rows)))
            summary = {"status": "complete", "episodes_completed": 2, "episodes_requested": 4,
                       **{key: {"mean": .5} for key in flags[:4]}, "return": {"mean": 10.0},
                       "length": {"mean": 600.0}, "model_unchanged": True}
            path = output / "summary.json"
            path.write_text(json.dumps(summary))
            result = evaluation(path)
            self.assertEqual(result["counts"]["success"], 1)
            self.assertEqual(result["episodes"], 2)
            self.assertFalse(result["complete_budget"])
            job = {"id": "partial", "kind": "diagnostic", "expected_completion_file": str(path),
                   "argv": ["python", "unused.py", "--max-steps", "1300", "--output-dir", str(output)]}
            manifest = root / "retention_manifest.json"
            manifest.write_text(json.dumps({"jobs": [job]}))
            original = manifest.read_bytes()
            subprocess.run([sys.executable, str(REPO / "scripts/prepare_str_retention_completion.py"),
                            "--root", str(root)], check=True, capture_output=True)
            rerun = json.loads((root / "retention_completion_manifest.json").read_text())["jobs"][0]
            self.assertEqual(rerun["id"], "partial_full2500")
            self.assertIn("2500", rerun["argv"])
            self.assertEqual(Path(rerun["expected_completion_file"]).parent.name, "stochastic_full2500")
            self.assertEqual(manifest.read_bytes(), original)
            self.assertEqual(path.read_text(), json.dumps(summary))


if __name__ == "__main__":
    unittest.main()
