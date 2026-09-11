"""CPU integration check of the native update function through the warmup gate."""
from pathlib import Path
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from flash_rl.agents.flashSAC import agent as module
from flash_rl.agents.flashSAC import update as updates
from flash_rl.agents.flashSAC.network import FlashSACActor, FlashSACDoubleCritic, FlashSACTemperature
from flash_rl.agents.utils.network import Network
from scripts.train_str_warmstart import CriticWarmupGate, network_audit, wrap_training_env_close


@pytest.mark.parametrize("warmup,baseline,expected_actor_steps", [(0, 0, 2), (2, 0, 1),
                         (0, 97466, 2), (2, 97466, 1), (100000, 97466, 0), (1, 97467, 2)])
def test_gate_keeps_actor_bn_and_temperature_frozen_then_resumes(tmp_path, warmup, baseline, expected_actor_steps):
    torch.set_num_threads(2)
    torch.manual_seed(0)
    actor_model = FlashSACActor(1, 6, 8, 3)
    critic_model = FlashSACDoubleCritic(1, 9, 8, 11, -5., 5.)
    target_model = FlashSACDoubleCritic(1, 9, 8, 11, -5., 5.)
    temperature_model = FlashSACTemperature(.01)
    actor = Network(actor_model, torch.optim.Adam(actor_model.parameters(), lr=3e-4), use_weight_normalization=True)
    critic = Network(critic_model, torch.optim.Adam(critic_model.parameters(), lr=3e-4), use_weight_normalization=True)
    actor.normalize_parameters(); critic.normalize_parameters()
    target_model.load_state_dict(critic_model.state_dict())
    target = Network(target_model, use_weight_normalization=True, ema_source=critic, ema_tau=.01)
    temperature = Network(temperature_model, torch.optim.Adam(temperature_model.parameters(), lr=3e-4))
    cfg = SimpleNamespace(actor_bc_alpha=0., use_amp=False, temp_target_entropy=-2.,
                          critic_min_v=-5., critic_max_v=5., critic_num_bins=11, gamma=.99, n_step=1)
    obs, nxt = torch.randn(16, 6), torch.randn(16, 6)
    batch = dict(observation=obs, next_observation=nxt, actor_observation=obs, actor_next_observation=nxt,
                 action=torch.rand(16, 3) * 2 - 1, reward=torch.ones(16), terminated=torch.zeros(16))

    class FakeAgent:
        def __init__(self):
            self._actor, self._critic, self._target_critic, self._temperature = actor, critic, target, temperature
            self._update_step, self.saved = baseline, []

        def update(self):
            result = module._update_networks(batch=batch, actor=actor, critic=critic, target_critic=target,
                temperature=temperature, cfg=cfg, do_actor_update=self._update_step % 2 == 0,
                device=torch.device("cpu"), grad_scaler=None)
            self._update_step += 1
            return result

        def save(self, path):
            Path(path).mkdir(parents=True)
            self.saved.append(self._update_step)

    gate = CriticWarmupGate(module._update_networks, warmup, tmp_path, [0, 1, 2, 4])
    agent = gate.bind(FakeAgent())
    # Exercise real native math without triggering CPU torch.compile compilation.
    orig_select = getattr(updates._select_min_q_log_probs, "_torchdynamo_orig_callable", updates._select_min_q_log_probs)
    orig_target = getattr(updates._compute_categorical_td_target, "_torchdynamo_orig_callable", updates._compute_categorical_td_target)
    with patch.object(module, "_update_networks", gate), patch.object(updates, "_select_min_q_log_probs", orig_select), \
            patch.object(updates, "_compute_categorical_td_target", orig_target):
        for _ in range(4):
            result = agent.update()
            assert all(torch.isfinite(value).all() for value in result.values() if isinstance(value, torch.Tensor))
    gate.finish("complete")
    assert gate.completed == 4 and agent.saved == [baseline + n for n in (0, 1, 2, 4)]
    assert agent._update_step == baseline + 4
    assert gate.state["baseline_native_agent_update_step"] == baseline
    assert [item["relative_network_calls"] for item in gate.state["snapshots"]] == [0, 1, 2, 4]
    assert [item["agent_update_step"] for item in gate.state["snapshots"]] == agent.saved
    assert gate.attempted == {"critic": 4, "actor": expected_actor_steps, "temperature": expected_actor_steps}
    assert network_audit(actor)["optimizer_step_max"] == expected_actor_steps
    assert network_audit(temperature)["optimizer_step_max"] == expected_actor_steps
    assert network_audit(critic)["optimizer_step_max"] == 4
    if warmup <= 4:
        assert gate.state["warmup_end"]["actor"] == gate.state["initial"]["actor"]
        assert gate.state["warmup_end"]["temperature"] == gate.state["initial"]["temperature"]
    else:
        assert gate.state["actor_and_temperature_frozen_through_final"] is True
        assert "warmup_end" not in gate.state
    assert (gate.state["final_network_audit"]["actor"]["buffer_digest"] != gate.state["initial"]["actor"]["buffer_digest"]) == bool(expected_actor_steps)
    assert (tmp_path / "warmup_audit.json").is_file()


def test_completion_is_written_before_native_close_exits(tmp_path):
    gate = CriticWarmupGate(None, 20, tmp_path, [])
    gate.state["status"] = "running"

    def hard_exit(**kwargs):
        saved = json.loads((tmp_path / "warmup_audit.json").read_text())
        assert saved["status"] == "complete"
        assert saved["completion_marker"] == "native_training_env_close_entered_after_final_save"
        assert kwargs == {"wait_for_replicator": False}
        raise SystemExit(0)  # Stand-in for an uncatchable simulator process exit.

    env = wrap_training_env_close(SimpleNamespace(close=hard_exit), gate)
    with pytest.raises(SystemExit) as result:
        env.close(wait_for_replicator=False)
    assert result.value.code == 0


def test_exception_cleanup_does_not_mark_training_complete(tmp_path):
    gate = CriticWarmupGate(None, 20, tmp_path, [])
    gate.state["status"] = "running"
    gate.write()
    closed = []
    env = wrap_training_env_close(SimpleNamespace(close=lambda: closed.append(True)), gate)
    try:
        raise RuntimeError("training failed")
    except RuntimeError as error:
        env.close()
        assert json.loads((tmp_path / "warmup_audit.json").read_text())["status"] == "running"
        gate.finish("failed", str(error))
    assert closed == [True]
    assert json.loads((tmp_path / "warmup_audit.json").read_text())["status"] == "failed"
