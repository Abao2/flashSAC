"""CPU-only checks for explicitly matched play criteria; no Isaac startup."""
from pathlib import Path
import sys

import pytest
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.play_str_full_diagnostic import parse_args
from scripts.eval_str_full_nodr import freeze_tolerance


BASE = ['--checkpoint', '/unused', '--output-dir', '/unused']


def test_default_preserves_previous_play():
    assert parse_args(BASE).tolerance == .075


def test_explicit_tolerance_is_used_without_curriculum_drift():
    requested = .02905653603374958
    args = parse_args(BASE + ['--tolerance', str(requested)])
    cfg = OmegaConf.create({'env': {'task_cfg_overrides': {}}})
    initial, frozen = freeze_tolerance(
        cfg, ROOT.parent / 'simtoolreal/isaacsimenvs/cfg/task/SimToolReal.yaml', args.tolerance)
    term = cfg.env.task_cfg_overrides.termination
    assert initial == .075
    assert frozen == term.success_tolerance == term.target_success_tolerance == requested
    assert term.eval_success_tolerance is None
    assert term.tolerance_curriculum_interval == 2**60


@pytest.mark.parametrize('value', ['0', '-1', 'nan', 'inf', '-inf'])
def test_invalid_tolerance_rejected_before_gpu_start(value):
    with pytest.raises(SystemExit) as error:
        parse_args(BASE + [f'--tolerance={value}'])
    assert error.value.code == 2
