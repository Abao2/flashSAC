"""Official success semantics for new STR runs; historical data stay cumulative."""
import importlib.util

import pytest
import torch

from scripts.check_str_full_nodr import TASK_ROOT, load_config


@pytest.mark.parametrize("config_name", [
    "simtoolreal_flashsac", "simtoolreal_full_nodr", "simtoolreal_full_arm1",
    "simtoolreal_full_arm1_hand1", "simtoolreal_full_arm1_nstep3",
    "simtoolreal_full_state162", "simtoolreal_full_lstm32",
    "simtoolreal_fixed_debug", "simtoolreal_state_teacher",
])
def test_new_str_configs_explicitly_require_ten_consecutive_steps(config_name):
    cfg, _, task = load_config(config_name)
    assert cfg.env.task_cfg_overrides.termination.success_steps == 10
    assert cfg.env.task_cfg_overrides.termination.force_consecutive_near_goal_steps is True
    assert task["termination"]["success_steps"] == 10
    assert task["termination"]["force_consecutive_near_goal_steps"] is True


def test_real_helpers_reject_interrupted_success_and_deliver_bonus_at_tenth_step():
    # Pure Torch module: do not import Isaac or start a simulator/GPU.
    path = TASK_ROOT / "utils/reward_utils.py"
    spec = importlib.util.spec_from_file_location("str_success_contract_helpers", path)
    assert spec is not None and spec.loader is not None
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    _, _, task = load_config("simtoolreal_full_arm1_nstep3")
    steps = task["termination"]["success_steps"]
    bonus = task["reward"]["reach_goal_bonus"]
    for near_flags, expected_final_count, expected_rewards in (
        ([True] * 5 + [False] + [True] * 5, 5, [0.] * 11),
        ([True] * 10, 10, [0.] * 9 + [bonus]),
    ):
        count = torch.zeros(1, dtype=torch.long)
        rewards = []
        for near in near_flags:
            near_tensor = torch.tensor([near])
            count = helpers.update_near_goal_steps(near_tensor, count, True)
            success = count >= steps
            rewards.append(helpers.reach_goal_bonus(near_tensor, success, bonus, steps, True).item())
        assert count.item() == expected_final_count
        assert rewards == expected_rewards
