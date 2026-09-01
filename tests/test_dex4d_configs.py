import unittest
from pathlib import Path

import hydra
from omegaconf import OmegaConf


class Dex4DConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not OmegaConf.has_resolver("eval"):
            OmegaConf.register_new_resolver("eval", lambda value: eval(value))

    def _compose(self, name: str):
        config_dir = str(Path(__file__).resolve().parents[1] / "configs")
        with hydra.initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = hydra.compose(config_name=name)
        OmegaConf.resolve(cfg)
        return cfg

    def test_stage12_scales_curriculum_to_flashsac_budget(self) -> None:
        cfg = self._compose("dex4d_stage12_flashsac")
        self.assertEqual(cfg.num_interaction_steps, 48_829)
        self.assertEqual(cfg.env.task_cfg_overrides.curriculum_policy_step, 29_297)
        self.assertFalse(cfg.require_agent_load)

    def test_stage3_requires_stage12_checkpoint(self) -> None:
        cfg = self._compose("dex4d_flashsac")
        self.assertTrue(cfg.require_agent_load)
        self.assertEqual(cfg.agent.buffer_max_length, 1_000_000)

    def test_m6_wuji_stage12_uses_independent_task(self) -> None:
        cfg = self._compose("dex4d_m6_wuji_stage12_flashsac")
        self.assertEqual(cfg.env.env_name, "FlashSAC-Dex4D-M6Wuji-Stage12-Direct-v0")
        self.assertEqual(cfg.group_name, "dex4d_m6_wuji")
        self.assertEqual(cfg.num_interaction_steps, 48_829)
        self.assertEqual(cfg.env.task_cfg_overrides.curriculum_policy_step, 29_297)


if __name__ == "__main__":
    unittest.main()
