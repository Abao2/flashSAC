"""CPU check that std fitting cannot alter the demonstrated mean policy."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from calibrate_str_bc_std import calibrate
from diagnose_str_memory import create_model, normalize_flash


class CalibrationTest(unittest.TestCase):
    def test_frozen_mean_bn_and_only_std_parameters_change(self):
        torch.set_num_threads(2)
        config = {"task": "bc", "mode": "current", "window": 8, "width": 128,
                  "num_blocks": 2, "input_dim": 162, "obs_dim": 162, "action_dim": 29}
        model = create_model(config)
        normalize_flash(model)
        original = {k: v.clone() for k, v in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.save({"format": "str_memory_probe_v1", "config": config, "model_state_dict": original}, root / "source.pt")
            np.savez(root / "data.npz", obs=np.random.default_rng(0).normal(size=(128, 162)).astype(np.float32),
                     episode_id=np.repeat(np.arange(8), 16))
            report = calibrate(root / "source.pt", root / "data.npz", root / "calibrated.pt", .05,
                               updates=12, batch_size=32, log_every=12)
            self.assertTrue(report["validation"]["deterministic_means_exact"])
            self.assertTrue(report["validation"]["all_non_std_parameters_and_BN_exact"])
            self.assertEqual(set(report["changed_keys"]), {"predictor.std_w.w.weight", "predictor.std_bias"})
            self.assertFalse(set(report["train_episode_ids"]) & set(report["validation_episode_ids"]))
            after_source = torch.load(root / "source.pt", map_location="cpu", weights_only=True)
            for key, value in original.items():
                self.assertTrue(torch.equal(value, after_source["model_state_dict"][key]))
            eigen = calibrate(root / "source.pt", root / "data.npz", root / "eigen.pt", .05,
                              updates=1, batch_size=32, method="min-variance")
            self.assertTrue(eigen["validation"]["deterministic_means_exact"])
            self.assertTrue(eigen["eigen_fit"]["fit_uses_train_episodes_only"])
            self.assertEqual(eigen["eigen_fit"]["covariance_dtype"], "float64")
            self.assertEqual(eigen["updates"], 0)


if __name__ == "__main__":
    unittest.main()
