import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from dextoolbench.summarize_evals import (
    expected_combinations,
    main,
    render_summary,
    summarize_run,
)


def _write_eval(
    run_dir: Path,
    combination,
    *,
    goal_pcts=(25.0, 75.0),
    avg_time_sec=2.5,
) -> Path:
    path = (
        run_dir
        / combination.category
        / combination.object_name
        / combination.task_name
        / combination.policy_name
        / "eval.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "avg_goal_pct": sum(goal_pcts) / len(goal_pcts),
                "avg_time_sec": avg_time_sec,
                "episode_goal_pcts": list(goal_pcts),
                "episode_lengths": [120] * len(goal_pcts),
            }
        ),
        encoding="utf-8",
    )
    return path


class SummarizeEvalsTest(unittest.TestCase):
    def test_expected_combinations_accepts_one_shot_policy_iterables(self):
        combinations = expected_combinations(iter(("test_policy",)))
        self.assertEqual(len(combinations), 24)

    def test_complete_run_reports_details_and_means(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            combinations = expected_combinations(("test_policy",))
            for index, combination in enumerate(combinations):
                goal = float(index)
                _write_eval(
                    run_dir,
                    combination,
                    goal_pcts=(goal, goal),
                    avg_time_sec=float(index + 1),
                )

            summary = summarize_run(run_dir)

            self.assertEqual(summary.policies, ("test_policy",))
            self.assertEqual(len(summary.results), 24)
            self.assertEqual(len(summary.successful), 24)
            self.assertTrue(summary.is_complete)
            hammer_mean = summary.category_means()[("hammer", "test_policy")]
            self.assertAlmostEqual(hammer_mean.avg_goal_pct, 1.5)
            overall = summary.overall_means()["test_policy"]
            self.assertEqual(overall.completed, 24)
            self.assertEqual(overall.expected, 24)
            self.assertAlmostEqual(overall.avg_goal_pct, 11.5)
            self.assertAlmostEqual(overall.avg_time_sec, 12.5)

            output = render_summary(summary)
            self.assertIn("Combination details", output)
            self.assertIn("Category means (successful combinations only)", output)
            self.assertIn("Overall means (successful combinations only)", output)
            self.assertIn(
                "Expected: 24 | OK: 24 | Missing: 0 | Failed: 0", output
            )

    def test_missing_and_invalid_results_are_reported_separately(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            combinations = expected_combinations(("pretrained_policy",))
            _write_eval(run_dir, combinations[0])
            invalid_path = _write_eval(run_dir, combinations[1])
            invalid_path.write_text("{not valid json", encoding="utf-8")

            summary = summarize_run(run_dir, policies=("pretrained_policy",))

            self.assertEqual(len(summary.successful), 1)
            self.assertEqual(len(summary.missing), 22)
            self.assertEqual(len(summary.failed), 1)
            self.assertFalse(summary.is_complete)
            output = render_summary(summary)
            self.assertIn("Missing items (22)", output)
            self.assertIn("Failed items (1)", output)
            self.assertIn(combinations[1].label, output)
            self.assertIn(
                "Expecting property name enclosed in double quotes", output
            )

    def test_schema_errors_are_failed_and_cli_exit_code_can_be_overridden(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            combination = expected_combinations(("pretrained_policy",))[0]
            eval_path = _write_eval(run_dir, combination)
            payload = json.loads(eval_path.read_text(encoding="utf-8"))
            payload["avg_goal_pct"] = 99.0
            eval_path.write_text(json.dumps(payload), encoding="utf-8")

            summary = summarize_run(run_dir, policies=("pretrained_policy",))
            self.assertEqual(
                summary.failed[0].error,
                "avg_goal_pct does not match the mean of episode_goal_pcts",
            )

            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main([str(run_dir), "--policy", "pretrained_policy"]), 1
                )
                self.assertEqual(
                    main(
                        [
                            str(run_dir),
                            "--policy",
                            "pretrained_policy",
                            "--allow-incomplete",
                        ]
                    ),
                    0,
                )


if __name__ == "__main__":
    unittest.main()
