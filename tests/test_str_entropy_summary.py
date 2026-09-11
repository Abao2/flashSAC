"""A wrong budget/branch checkpoint must not be silently graphed as this run."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from summarize_str_entropy_ab import checkpoint_label, validate_branch


class EntropyMappingTest(unittest.TestCase):
    def test_budget_and_model_mapping(self):
        branch = {"id": "a", "output_root": "/tmp/a", "final_checkpoint": "/tmp/a/models/final/step1954",
                  "initial_checkpoint": "/tmp/initial", "alpha": .0001}
        config = {"num_env_steps": 2000896, "num_train_envs": 1024, "num_interaction_steps": 1954,
                  "output_root": "/tmp/a", "save_path": "models/final", "agent_load_path": "/tmp/initial",
                  "agent": {"temp_initial_value": .0001}, "env": {"task_cfg_overrides": {"action": {"arm_moving_average": .1, "hand_moving_average": .1}}}}
        validate_branch(branch, config)
        self.assertEqual(checkpoint_label(branch, branch["final_checkpoint"]), "final")
        self.assertEqual(checkpoint_label(branch, "/tmp/a/update_snapshots/update2010"), "update2010")
        with self.assertRaises(ValueError):
            checkpoint_label(branch, "/tmp/other/update_snapshots/update2010")
        with self.assertRaises(AssertionError):
            validate_branch(branch, {**config, "num_env_steps": 10000384})
        with self.assertRaises(AssertionError):
            validate_branch({**branch, "final_checkpoint": "/tmp/a/models/final/step9766"}, config)


if __name__ == "__main__":
    unittest.main()
