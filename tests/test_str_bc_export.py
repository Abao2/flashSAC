"""Native-format BC export, on CPU with one replay slot only."""

from pathlib import Path
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from diagnose_str_memory import create_model, normalize_flash
from flash_rl.agents.flashSAC.network import FlashSACActor
from export_str_bc_checkpoint import export_checkpoint


class ExportTest(unittest.TestCase):
    def test_cpu_native_files_exact_actor_and_fresh_state(self):
        torch.set_num_threads(2)
        config = {"task": "bc", "mode": "current", "window": 8, "width": 128,
                  "num_blocks": 2, "input_dim": 162, "obs_dim": 162, "action_dim": 29}
        model = create_model(config)
        normalize_flash(model)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "bc.pt"
            torch.save({"format": "str_memory_probe_v1", "config": config,
                        "model_state_dict": model.state_dict()}, source)
            result = export_checkpoint(source, root / "export")
            self.assertTrue(result["validation"]["deterministic_actions_exact"])
            saved = torch.load(root / "export/actor.pt", map_location="cpu", weights_only=True)
            restored = torch.compile(FlashSACActor(2, 162, 128, 29), backend="eager")
            restored.load_state_dict(saved["network_state_dict"], strict=True)
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, restored.state_dict()["_orig_mod." + name]))
            self.assertEqual(len(result["files"]), 6)
            self.assertEqual(result["training_replay_capacity_unchanged"], 10000000)
            self.assertEqual(result["validation"]["fresh_normalizer_count"], 0)
            with self.assertRaises(FileExistsError):
                export_checkpoint(source, root / "export")


if __name__ == "__main__":
    unittest.main()
