"""Arithmetic contract only: no PhysX, PD tracking, or contact validation."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch


@pytest.mark.parametrize("arm_ema", [.1, 1.])
@pytest.mark.parametrize("permuted", [False, True])
def test_real_action_function_against_independent_target_formula(arm_ema, permuted):
    path = Path('/home/abao/simtoolreal/isaacsimenvs/tasks/simtoolreal/utils/action_utils.py')
    spec = importlib.util.spec_from_file_location('str_action_cpu_contract', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    permutation = np.r_[np.arange(7)[::-1], np.arange(7, 29)[::-1]] if permuted else np.arange(29)
    act_cfg = NS(arm_moving_average=arm_ema, hand_moving_average=.1, dof_speed_scale=1.5)
    env = NS(device='cpu', step_dt=1/60, cfg=NS(action=act_cfg,
             domain_randomization=NS(use_action_delay=False)),
             _perm_canon_to_lab=torch.tensor(permutation),
             _arm_lower=torch.full((7,), -.7), _arm_upper=torch.full((7,), .7),
             _hand_lower=torch.full((22,), -.2), _hand_upper=torch.full((22,), 1.5),
             _arm_joint_ids=torch.arange(7), _hand_joint_ids=torch.arange(7, 29),
             _prev_targets=torch.zeros(8, 29), _cur_targets=torch.zeros(8, 29))
    # Exercise target saturation, persistent integration, and a nontrivial mapping.
    env._prev_targets[0, :7] = .7
    env._prev_targets[1, :7] = -.7
    expected = env._prev_targets.numpy().copy()
    rng = np.random.default_rng(11)
    for step in range(100):
        action = rng.uniform(-1, 1, (8, 29)).astype(np.float32)
        if step == 0:
            action[0, :7] = 1
            action[1, :7] = -1
        mapped = action[:, permutation]
        arm = np.clip(expected[:, :7] + .025 * mapped[:, :7], -.7, .7)
        expected[:, :7] = np.clip(arm_ema * arm + (1-arm_ema) * expected[:, :7], -.7, .7)
        hand = -.2 + .5 * (mapped[:, 7:] + 1) * 1.7
        expected[:, 7:] = np.clip(.1 * hand + .9 * expected[:, 7:], -.2, 1.5)
        module.apply_action_pipeline(env, torch.from_numpy(action))
        np.testing.assert_allclose(env._cur_targets.numpy(), expected, rtol=0, atol=5e-7)
        torch.testing.assert_close(env._cur_targets, env._prev_targets)
