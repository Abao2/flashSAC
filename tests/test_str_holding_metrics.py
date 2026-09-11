"""CPU checks: passive height accounting survives auto-reset without changing reward."""
import json
from types import SimpleNamespace

import pytest
import torch

from scripts.train_str_holding import install_height_metrics, wrap_completion


class FakeEnv:
    def __init__(self):
        torch.set_num_threads(2)
        self.num_envs, self.device, self.step_dt = 2, "cpu", .25
        self.scene = SimpleNamespace(env_origins=torch.tensor([[0., 0., 2.], [0., 0., 4.]]))
        self._object_init_z = torch.tensor([.5, .6])
        self.object = SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.zeros(2, 3)))
        self.extras, self.reward, self.goal = {}, torch.tensor([2., -3.]), 0
        self._reset_idx(None)

    def _get_rewards(self):
        self.extras["episode_final"] = {"successes": torch.tensor([float(self.goal), 0.])}
        return self.reward

    def _reset_idx(self, ids):
        ids = slice(None) if ids is None else ids
        self.object.data.root_pos_w[ids, 2] = (self.scene.env_origins[:, 2] + self._object_init_z)[ids]
        return "reset"

    def height(self, a, b):
        self.object.data.root_pos_w[:, 2] = self.scene.env_origins[:, 2] + self._object_init_z + torch.tensor([a, b])


def test_passive_metrics_survive_terminal_reset_and_goal_changes():
    raw = FakeEnv()
    install_height_metrics(raw)
    initial_rng = torch.random.get_rng_state().clone()
    for step in range(4):
        raw.height(.2, .05)
        if step == 2:
            raw.goal += 1  # Goal-only changes do not call episode reset.
        assert raw._get_rewards() is raw.reward
    final = raw.extras["episode_final"]
    assert final["successes"].tolist() == [1., 0.]
    assert final["height_proxy_longest_above_10cm_s"].tolist() == [1., 0.]
    assert final["height_proxy_above_10cm_1s"].tolist() == [1., 0.]
    assert final["height_proxy_ever_above_10cm"].tolist() == [1., 0.]
    assert torch.allclose(final["height_proxy_peak_delta_m"], torch.tensor([.2, .05]))
    assert raw._reset_idx(torch.tensor([0])) == "reset"
    assert final["height_proxy_longest_above_10cm_s"].tolist() == [1., 0.]  # Cloned pre-reset.
    raw.height(0., .2)
    raw._get_rewards()
    assert raw.extras["episode_final"]["height_proxy_longest_above_10cm_s"].tolist() == [0., .25]
    raw._reset_idx(None)
    raw._get_rewards()
    assert raw.extras["episode_final"]["height_proxy_ever_above_10cm"].tolist() == [0., 0.]
    assert torch.equal(torch.random.get_rng_state(), initial_rng)


def test_nonconsecutive_lifts_do_not_accumulate_holding_time():
    raw = FakeEnv()
    install_height_metrics(raw)
    for height in (.2, .2, 0., .2, .2):
        raw.height(height, 0.)
        raw._get_rewards()
    assert raw.extras["episode_final"]["height_proxy_longest_above_10cm_s"].tolist() == [.5, 0.]
    assert raw.extras["episode_final"]["height_proxy_above_10cm_1s"].tolist() == [0., 0.]


def test_completion_requires_all_files_and_precedes_app_exit(tmp_path):
    checkpoint = tmp_path / "final"
    checkpoint.mkdir()
    for name in ("actor", "critic", "target_critic", "temperature", "reward_normalizer", "agent_state", "replay_buffer"):
        (checkpoint / f"{name}.pt").write_bytes(b"test")
    def exits():
        assert json.loads((tmp_path / "training_complete.json").read_text())["status"] == "complete"
        raise SystemExit(0)
    env = wrap_completion(SimpleNamespace(close=exits), tmp_path, checkpoint)
    with pytest.raises(SystemExit):
        env.close()
    (checkpoint / "replay_buffer.pt").unlink()
    with pytest.raises(RuntimeError, match="replay_buffer"):
        env.close()
    assert json.loads((tmp_path / "training_complete.json").read_text())["status"] == "failed"


def test_exception_cleanup_never_marks_complete(tmp_path):
    env = wrap_completion(SimpleNamespace(close=lambda: None), tmp_path, tmp_path)
    try:
        raise ValueError("training failed")
    except ValueError:
        env.close()
    assert not (tmp_path / "training_complete.json").exists()
