"""CPU-only single-variable config and sliding n-step boundary contracts."""
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch
from omegaconf import OmegaConf

from flash_rl.buffers.torch_buffer import TorchUniformBuffer
from scripts.check_str_full_nodr import load_config, observation_sizes


def test_nstep3_preserves_full_task_agent_and_budget():
    baseline_cfg, _, baseline = load_config("simtoolreal_full_arm1")
    candidate_cfg, _, candidate = load_config("simtoolreal_full_arm1_nstep3")
    assert candidate == baseline
    assert candidate["action"]["arm_moving_average"] == 1.
    assert candidate["action"]["hand_moving_average"] == .1
    assert candidate_cfg.group_name == "str_credit_ab"
    assert candidate_cfg.exp_name == "full_arm1_nstep3_seed0"
    assert candidate_cfg.seed == 0
    assert candidate_cfg.n_step == candidate_cfg.agent.n_step == 3
    assert baseline_cfg.n_step == baseline_cfg.agent.n_step == 1
    assert candidate_cfg.gamma == candidate_cfg.agent.gamma == .99
    assert candidate_cfg.num_env_steps == 100003840
    assert candidate_cfg.num_train_envs == 2048
    assert candidate_cfg.updates_per_interaction_step == 4
    assert candidate_cfg.save_buffer_per_interaction_step == 48830
    assert candidate_cfg.agent_load_path is None and candidate_cfg.buffer_load_path is None
    sizes = observation_sizes()
    assert sum(sizes[name] for name in candidate["obs"]["obs_list"]) == 140
    assert sum(sizes[name] for name in candidate["obs"]["state_list"]) == 162

    a, b = (OmegaConf.to_container(cfg, resolve=True) for cfg in (baseline_cfg, candidate_cfg))
    b["n_step"] = b["agent"]["n_step"] = 1
    for cfg in (a, b):
        for key in ("group_name", "exp_name", "save_path", "save_buffer_per_interaction_step"):
            cfg.pop(key)
    assert a == b


def test_nstep3_sliding_windows_keep_final_obs_and_exclude_reset_rewards():
    with patch("torch.cuda.is_available", return_value=False):
        buffer = TorchUniformBuffer(
            observation_space=gym.spaces.Box(-1000, 1000, shape=(2, 1), dtype=np.float32),
            action_space=gym.spaces.Box(-1, 1, shape=(2, 1), dtype=np.float32),
            n_step=3, gamma=.5, max_length=16, min_length=1,
            sample_batch_size=6, device_type="cpu",
        )
    # Env0 truncates at t1, env1 terminates at t2. Their next episodes pay 100.
    observations = [[0, 10], [1, 11], [100, 12], [101, 110], [102, 111]]
    next_observations = [[1, 11], [901, 12], [101, 912], [102, 111], [103, 112]]
    rewards = [[1, 10], [2, 20], [100, 30], [4, 100], [5, 50]]
    for step in range(5):
        buffer.add(dict(
            observation=np.asarray(observations[step], dtype=np.float32)[:, None],
            next_observation=np.asarray(next_observations[step], dtype=np.float32)[:, None],
            action=np.zeros((2, 1), dtype=np.float32),
            reward=np.asarray(rewards[step], dtype=np.float32),
            terminated=np.array([False, step == 2]),
            truncated=np.array([step == 1, False]),
        ))

    assert len(buffer) == buffer._current_idx == 6
    batch = buffer.sample(np.arange(6))  # Interleaved env0/env1, anchors t0/t1/t2.
    expected = {
        "observation": [[0], [10], [1], [11], [100], [12]],
        "next_observation": [[901], [912], [901], [912], [103], [912]],
        "reward": [2, 27.5, 2, 35, 103.25, 30],
        "discount": [.25, .125, .5, .25, .125, .5],
        "terminated": [0, 1, 0, 1, 0, 1],
        "truncated": [1, 0, 1, 0, 0, 0],
    }
    for key, values in expected.items():
        torch.testing.assert_close(batch[key], torch.tensor(values, dtype=torch.float32), rtol=0, atol=0)
