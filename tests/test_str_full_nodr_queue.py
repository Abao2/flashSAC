import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_str_full_nodr_queue import require_preflight


def test_preflight_gate_requires_real_timeout_and_complete_pool():
    passed = {"status": "PASS", "gpu": {"timeout_final_obs": "PASS", "asset_selection": {
        "six_types_selected": "PASS", "all_pool_assets_selected": "PASS"}}}
    require_preflight(passed)
    failed = copy.deepcopy(passed)
    failed["status"] = "FAIL"
    with pytest.raises(AssertionError):
        require_preflight(failed)
    for field in ("six_types_selected", "all_pool_assets_selected"):
        failed = copy.deepcopy(passed)
        failed["gpu"]["asset_selection"][field] = "not_checked_insufficient_envs"
        with pytest.raises(AssertionError):
            require_preflight(failed)
    failed = copy.deepcopy(passed)
    failed["gpu"]["timeout_final_obs"] = "not_seen"
    with pytest.raises(AssertionError):
        require_preflight(failed)
