"""CPU check that alpha cloning cannot mutate another native component."""

import math
from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from clone_str_initial_alpha import FILES, clone_checkpoint, sha256


class InitialAlphaTest(unittest.TestCase):
    def test_only_loaded_temperature_changes_and_original_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            for name in FILES:
                torch.save({"update_step": 0, "marker": torch.tensor([4, 5])}, source / name)
            torch.save({"network_state_dict": {"_orig_mod.log_temp": torch.tensor([math.log(.01)])},
                "optimizer_state_dict": {"state": {}, "param_groups": [{"lr": .0003}]},
                "scheduler_state_dict": {"last_epoch": 0}, "update_step": 0}, source / "temperature.pt")
            hashes = {name: sha256(source / name) for name in FILES}
            for alpha in (.01, .0001, .000001):
                output = root / str(alpha)
                result = clone_checkpoint(source, output, alpha)
                self.assertTrue(result["non_temperature_files_bitwise_identical"])
                self.assertTrue(result["temperature_other_fields_exact"])
                self.assertEqual(hashes, {name: sha256(source / name) for name in FILES})
                if alpha == .01:
                    self.assertEqual(hashes, result["output_sha256"])
                with self.assertRaises(FileExistsError):
                    clone_checkpoint(source, output, alpha)
            with self.assertRaises(ValueError):
                clone_checkpoint(source, root / "invalid", float("nan"))


if __name__ == "__main__":
    unittest.main()
