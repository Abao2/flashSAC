from pathlib import Path
from types import SimpleNamespace
import json
import sys

import numpy as np
import pytest
import torch

from scripts import diagnose_str_reset_reward as probe


class MockEnv:
    def __init__(self):
        self.raw = SimpleNamespace(num_envs=2, reward_buf=torch.zeros(2), physical=torch.zeros(2),
            cfg=SimpleNamespace(obs=SimpleNamespace(obs_list=["physical", "reward"],
                                                     state_list=["physical", "reward"])))
        self.envs = SimpleNamespace(unwrapped=self.raw)
        self.calls = []

    def reset(self, *, seed=None, random_start_init=False):
        self.calls.append((seed, random_start_init, self.raw.reward_buf.clone()))
        self.raw.physical.fill_(7)  # Native reset's independent physical operation.
        part = torch.stack([self.raw.physical, self.raw.reward_buf * .01], dim=-1).numpy()
        return np.concatenate([part, part], axis=1), {"seed": seed}


def test_injection_precedes_first_reset_only_and_preserves_other_fields():
    env, audit, snapshots = MockEnv(), {}, []
    probe.install_initial_reward_reset(env, {"physical": 1, "reward": 1}, 100.14024353027344,
                                     audit, lambda: snapshots.append(dict(audit)))
    assert env.calls == []  # No reset or extra physics is inserted by installing.
    obs, info = env.reset(seed=17, random_start_init=False)
    assert info == {"seed": 17} and len(env.calls) == 1
    assert env.calls[0][:2] == (17, False)
    assert torch.allclose(env.calls[0][2], torch.full((2,), 100.14024353027344))
    assert np.array_equal(obs[:, [0, 2]], np.full((2, 2), 7))
    assert np.allclose(obs[:, [1, 3]], 1.0014024353)
    assert audit["policy_reward_column"] == 1 and audit["critic_reward_column"] == 3
    assert snapshots[0]["status"] == "injected_before_first_explicit_reset"
    assert snapshots[-1]["status"] == "initial_reset_verified"
    env.raw.reward_buf.fill_(3)
    again, _ = env.reset(seed=18)
    assert np.allclose(again[:, [1, 3]], .03)  # No second intervention.


def test_wrapper_guard_order_and_restoration(tmp_path, monkeypatch):
    env = MockEnv()
    factory = lambda *a: (env, {}, {})
    original_argv = sys.argv
    monkeypatch.setattr(probe.rollout, "create_env", factory)
    monkeypatch.setattr(probe, "obs_field_sizes", lambda: {"physical": 1, "reward": 1})

    def fake_main():
        assert "--initial-previous-reward" not in sys.argv
        assert sys.argv[1:] == ["--output-dir", str(tmp_path)]
        assert not any(tmp_path.iterdir())  # Harness empty-directory guard occurs first.
        args = SimpleNamespace(output_dir=tmp_path, policy="flash", entry="wrapper", training_startup=False)
        wrapped, _, _ = probe.rollout.create_env(args, None, None)
        assert (tmp_path / "reset_reward_intervention.json").is_file()
        wrapped.reset()
        assert json.loads((tmp_path / "reset_reward_intervention.json").read_text())["status"] == "initial_reset_verified"
        raise SystemExit(0)  # Isaac-style exit cannot leave audit unwritten.

    monkeypatch.setattr(probe.rollout, "main", fake_main)
    with pytest.raises(SystemExit):
        probe.main(["--initial-previous-reward", "0", "--output-dir", str(tmp_path)])
    assert probe.rollout.create_env is factory and sys.argv is original_argv


def test_reset_clearing_reward_is_detected():
    env, audit = MockEnv(), {}
    native = env.reset
    def clears_reward(**kwargs):
        env.raw.reward_buf.zero_()
        return native(**kwargs)
    env.reset = clears_reward
    probe.install_initial_reward_reset(env, {"physical": 1, "reward": 1}, 100, audit, lambda: None)
    with pytest.raises(AssertionError, match="did not preserve"):
        env.reset()
    assert audit["status"] == "failed"
